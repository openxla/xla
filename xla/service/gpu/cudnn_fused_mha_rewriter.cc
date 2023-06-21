/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/cudnn_fused_mha_rewriter.h"

#include <numeric>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {
namespace m = match;

template <typename Pattern>
auto OptionalReshape(Pattern pattern) {
  auto shared = m::SharedSubpattern(pattern);
  return m::AnyOf<HloInstruction>(m::Reshape(shared), shared);
}

template <typename Pattern>
auto OptionalConvert(Pattern pattern) {
  auto shared = m::SharedSubpattern(pattern);
  return m::AnyOf<HloInstruction>(m::Convert(shared), shared);
}

template <typename Pattern>
auto OptionalBitcast(Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Bitcast(pattern), std::move(pattern));
}

template <typename Pattern>
auto OptionalBroadcast(Pattern pattern) {
  auto shared = m::SharedSubpattern(pattern);
  return m::AnyOf<HloInstruction>(m::Broadcast(shared), shared);
}

bool IsBatchedMatmul(const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kDot) return false;
  const DotDimensionNumbers& dot_dims = instr->dot_dimension_numbers();
  bool is_batch_dot = !dot_dims.lhs_batch_dimensions().empty() ||
                      !dot_dims.rhs_batch_dimensions().empty();
  return is_batch_dot;
}

// We need to check if current gemm is sharing a parent node with a forward
// fMHA call. We check this by doing a BFS of all operands to see if there's
// any user that is a forward fMHA custom call.
// In general, a matching case would have this type of structure:
//                         mha_input_tensor(q, k, v)
//                            /               \
//            shape_ops(bitcast, etc.)      shape_ops(bitcast, etc.)
//                          /                   \
//                   forward_mha_call           backward_gemm
// We start at the backward_gemm and add each operand to visit list,
// if the operand is a shape op, we add it to the visit list.
// For each instruction in the visit list, we go through its users to
// see if any of them is a forward_mha_call.
bool IsSharingOperandWithFwdMha(HloInstruction* gemm) {
  for (int64_t i = 0; i < gemm->operands().size(); i++) {
    std::queue<HloInstruction*> visit_list;
    visit_list.push(gemm->mutable_operand(i));
    while (!visit_list.empty()) {
      HloInstruction* current_instr = visit_list.front();
      for (int64_t user_index = 0; user_index < current_instr->user_count();
           user_index++) {
        HloInstruction* user = current_instr->users()[user_index];
        switch (user->opcode()) {
          case HloOpcode::kBitcast:
          case HloOpcode::kReshape:
          case HloOpcode::kTranspose: {
            visit_list.push(user);
            break;
          }
          case HloOpcode::kCustomCall: {
            if (IsFwdCustomCallTofMHA(*user)) {
              return true;
            }
          }
          default:
            break;
        }
      }
      visit_list.pop();
    }
  }
  return false;
}

bool IsFirstFwdMatmul(HloInstruction* gemm) {
  return (IsBatchedMatmul(gemm) && !IsFwdCustomCallTofMHA(*gemm->operand(0)) &&
          !IsFwdCustomCallTofMHA(*gemm->operand(1)) &&
          !IsSharingOperandWithFwdMha(gemm));
}

bool IsScalar(const HloInstruction* instr) {
  return ShapeUtil::IsEffectiveScalar(instr->shape());
}

bool IsReduceMax(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kReduce &&
         instr->to_apply()->root_instruction()->opcode() == HloOpcode::kMaximum;
}

bool IsReduceSum(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kReduce &&
         instr->to_apply()->root_instruction()->opcode() == HloOpcode::kAdd;
}

// Set up subpatterns for re-use.
// Matches softmax sub-pattern ->
// divide(exp(Subtract(producer, reduce_max(producer))),
// broadcast(reduce_add(exp(Subtract(...))))). There might be reshape and
// convert nodes between reduce and Subtract.
// TODO TJ: Make this more general to any patterns that has this structure when
// cudnn runner supports generic cudnnOpGraphs. producer
// |   \
// |  reduce
// |     |
// |  broadcast
// |   /
// root
auto GetUnfusedReduceMaxSumSoftmaxPattern(
    HloInstruction** softmax_input = nullptr,
    HloInstruction** softmax_reduce_sum = nullptr,
    HloInstruction** softmax_reduce_sum_bcast = nullptr) {
  // The reduce-max part of the softmax
  auto unfused_softmax_max_subpattern = m::SharedSubpattern(m::Subtract(
      m::Op(),
      m::Broadcast(OptionalConvert(OptionalConvert(
          m::Op()
              .WithPredicate(IsReduceMax)
              .WithOperand(0, OptionalConvert(m::Op(softmax_input))))))));

  // The reduce-add part of the softmax
  auto unfused_softmax_sum_subpattern = m::SharedSubpattern(m::Divide(
      m::Exp(unfused_softmax_max_subpattern),
      m::Broadcast(OptionalConvert(OptionalConvert(
                       m::Op()
                           .WithOperand(0, OptionalConvert(m::Exp(
                                               unfused_softmax_max_subpattern)))
                           .WithPredicate(IsReduceSum)
                           .WithOneUse())))
          .WithOneUse()));
  return unfused_softmax_sum_subpattern;
}

std::optional<double> GetConstantValue(const HloInstruction* inst) {
  if (!IsScalar(inst)) {
    return std::nullopt;
  }
  switch (inst->shape().element_type()) {
    case F16:
      return static_cast<float>(inst->literal().GetFirstElement<half>());
    case BF16:
      return static_cast<float>(inst->literal().GetFirstElement<bfloat16>());
    case F32:
      return inst->literal().GetFirstElement<float>();
    case F64:
      return inst->literal().GetFirstElement<double>();
    default:
      return std::nullopt;
  }
}

double GetDropoutRateFromHlo(HloInstruction* dropout) {
  std::optional<double> dropout_rate_inv;
  dropout_rate_inv = GetConstantValue(dropout);
  if (!dropout_rate_inv.has_value()) {
    return 0.0;
  }
  // In dropout, inputs are divided by (1 - rate), we need to divide 1 by
  // the constant in dropout node and substract
  // from 1 here to get the actual dropout rate.
  return (1.0 - (1.0 / *dropout_rate_inv));
}

bool IsComputeCapabilityAndCudnnSupported(
    stream_executor::CudaComputeCapability cc,
    stream_executor::dnn::VersionInfo cudnn_version,
    stream_executor::StreamExecutor* stream_exec,
    stream_executor::dnn::VersionInfo supported_cudnn_version) {
  // return true;
  se::dnn::VersionInfo real_cudnn_version;
  if (stream_exec) {
    stream_executor::dnn::DnnSupport* dnn = stream_exec->AsDnn();
    StatusOr<se::dnn::VersionInfo> se_cudnn_version = dnn->GetVersion();
    if (se_cudnn_version.ok()) {
      real_cudnn_version = (*se_cudnn_version);
    }
  } else {
    real_cudnn_version = cudnn_version;
  }

  if (!((cc.IsAtLeast(se::CudaComputeCapability::AMPERE) && cc.minor == 0) &&
        (real_cudnn_version >= supported_cudnn_version))) {
    VLOG(2) << absl::StrFormat(
        "CudnnFusedMHARewriter did not run. Unsupported compute "
        "capability(==8.0) or cudnn version(>=%d.%d.%d)",
        supported_cudnn_version.major_version(),
        supported_cudnn_version.minor_version(),
        supported_cudnn_version.patch());
    return false;
  }
  return true;
}

bool IsSupportedPrimitiveType(const HloInstruction* bmm) {
  PrimitiveType dtype = bmm->shape().element_type();
  return dtype == BF16 || dtype == F16;
}

bool IsContractingDimSupported(absl::Span<const int64_t> contracting_dims) {
  return absl::c_all_of(contracting_dims,
                        [](int64_t dim) { return dim == 64; });
}

bool IsNonContractingDimSupported(
    const std::vector<int64_t>& non_contracting_dims) {
  return absl::c_all_of(non_contracting_dims,
                        [](int64_t dim) { return dim <= 512; });
}

bool IsRankSupported(const HloInstruction* bmm) {
  return bmm->operand(0)->shape().dimensions().size() == 4 &&
         bmm->operand(1)->shape().dimensions().size() == 4;
}

bool IsBatchDimSizeSupported(const DotDimensionNumbers& dot_dims) {
  return dot_dims.lhs_batch_dimensions().size() == 2 &&
         dot_dims.rhs_batch_dimensions().size() == 2;
}

std::vector<int64_t> GetDimensionVector(absl::Span<const int64_t> dimensions,
                                        absl::Span<const int64_t> dim_nums) {
  std::vector<int64_t> vec(dim_nums.size());
  for (int i = 0; i < dim_nums.size(); i++) {
    vec[i] = dimensions.at(dim_nums.at(i));
  }
  return vec;
}

StatusOr<bool> IsSupportedBMM1(const HloInstruction* bmm_1) {
  if (!IsRankSupported(bmm_1)) return false;
  const DotDimensionNumbers& dot_dims_bmm1 = bmm_1->dot_dimension_numbers();
  if (!IsBatchDimSizeSupported(dot_dims_bmm1)) return false;
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> lhs_non_contracting_dim_nums_bmm1,
      GetNonContractingDims(bmm_1->operand(0)->shape(),
                            dot_dims_bmm1.lhs_batch_dimensions(),
                            dot_dims_bmm1.lhs_contracting_dimensions()));
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> rhs_non_contracting_dim_nums_bmm1,
      GetNonContractingDims(bmm_1->operand(1)->shape(),
                            dot_dims_bmm1.rhs_batch_dimensions(),
                            dot_dims_bmm1.rhs_contracting_dimensions()));
  std::vector<int64_t> lhs_non_contracting_dims_bmm1 =
      GetDimensionVector(bmm_1->operand(0)->shape().dimensions(),
                         lhs_non_contracting_dim_nums_bmm1);
  std::vector<int64_t> rhs_non_contracting_dims_bmm1 =
      GetDimensionVector(bmm_1->operand(1)->shape().dimensions(),
                         rhs_non_contracting_dim_nums_bmm1);
  // The non contracting dimensions for BMM1 need to be less than or equal to
  // 512.
  if (!IsNonContractingDimSupported(lhs_non_contracting_dims_bmm1) ||
      !IsNonContractingDimSupported(rhs_non_contracting_dims_bmm1)) {
    if (VLOG_IS_ON(2)) {
      VLOG(2) << "BMM1 lhs_non_contracting_dims: "
              << absl::StrJoin(lhs_non_contracting_dims_bmm1, ",")
              << " BMM1 rhs_non_contracting_dims: "
              << absl::StrJoin(rhs_non_contracting_dims_bmm1, ",")
              << " are not supported. The non-contracting dims should be less "
                 "than 512. This is a criteria for current cuDNN 8.8 support.";
    }
    return false;
  }

  std::vector<int64_t> lhs_contracting_dims_bmm1 =
      GetDimensionVector(bmm_1->operand(0)->shape().dimensions(),
                         dot_dims_bmm1.lhs_contracting_dimensions());
  std::vector<int64_t> rhs_contracting_dims_bmm1 =
      GetDimensionVector(bmm_1->operand(1)->shape().dimensions(),
                         dot_dims_bmm1.rhs_contracting_dimensions());

  // The contracting dimensions for BMM1 need to be 64.
  if (!IsContractingDimSupported(lhs_contracting_dims_bmm1) ||
      !IsContractingDimSupported(rhs_contracting_dims_bmm1)) {
    if (VLOG_IS_ON(2)) {
      VLOG(2) << "BMM1 lhs_contracting_dims: "
              << absl::StrJoin(lhs_contracting_dims_bmm1, ",")
              << " BMM1 rhs_contracting_dims: "
              << absl::StrJoin(rhs_contracting_dims_bmm1, ",")
              << " are not supported.";
    }
    return false;
  }
  return true;
}

StatusOr<bool> IsSupportedBMM2(const HloInstruction* bmm_2,
                               bool need_canonicalization) {
  if (!IsRankSupported(bmm_2)) return false;
  const DotDimensionNumbers& dot_dims_bmm2 = bmm_2->dot_dimension_numbers();
  if (!IsBatchDimSizeSupported(dot_dims_bmm2)) return false;
  // need swap lhs and rhs for bmm2 if canonicalization is needed
  int operand_index = need_canonicalization ? 0 : 1;
  auto batch_dim = need_canonicalization ? dot_dims_bmm2.lhs_batch_dimensions()
                                         : dot_dims_bmm2.rhs_batch_dimensions();
  auto contracting_dim = need_canonicalization
                             ? dot_dims_bmm2.lhs_contracting_dimensions()
                             : dot_dims_bmm2.rhs_contracting_dimensions();

  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> non_contracting_dim_nums_bmm2,
      GetNonContractingDims(bmm_2->operand(operand_index)->shape(), batch_dim,
                            contracting_dim));

  std::vector<int64_t> non_contracting_dims_bmm2 =
      GetDimensionVector(bmm_2->operand(operand_index)->shape().dimensions(),
                         non_contracting_dim_nums_bmm2);
  // The non contracting dimension for BMM2 needs to be 64 for the input matrix.
  // The input matrix is the second argument to BMM2 i.e, rhs.
  if (!absl::c_all_of(non_contracting_dims_bmm2,
                      [](int64_t dim) { return dim == 64; })) {
    if (VLOG_IS_ON(2)) {
      VLOG(2) << " BMM2 rhs_non_contracting_dims: "
              << absl::StrJoin(non_contracting_dims_bmm2, ",")
              << " are not supported.";
    }
    return false;
  }
  return true;
}

bool MatchDefaultFwdBmmBmm(int64_t bmm2_operand_position, HloInstruction* instr,
                           HloInstruction** bmm_1, HloInstruction** bmm_2,
                           std::string& custom_call_name, bool& is_training) {
  // Try matching default bmm1-bmm2 pattern
  auto default_bmm_bmm_pattern =
      m::Op(bmm_2)
          .WithPredicate(IsBatchedMatmul)
          .WithOperand(bmm2_operand_position,
                       m::Op(bmm_1).WithPredicate(IsBatchedMatmul));

  // If any of bmm1's operands is coming from a forward fMHA call, then return
  // false
  if (Match(instr, default_bmm_bmm_pattern) && IsFirstFwdMatmul((*bmm_1))) {
    is_training = (*bmm_1)->user_count() == 2;
    custom_call_name = kCudnnfMHABmmBmmCallTarget;
    return true;
  }
  return false;
}

bool MatchSoftmaxDropoutBmm(int64_t bmm2_operand_position,
                            HloInstruction* instr, HloInstruction** bmm_2,
                            HloInstruction** softmax_input,
                            HloInstruction** dropout, bool& is_training) {
  // Matches the dropout-softmax subpattern.
  // Softmax_output is a divide
  // Dropout can take multiple forms, we capture 2 forms here based on
  // heurustics Form 1 -> softmax - mul - select(dropout) - BMM2
  HloInstruction* softmax_reduce_sum;
  HloInstruction* softmax_reduce_sum_bcast;

  auto dropout_softmax_pattern_form_1 = m::Select(
      m::Op(),
      OptionalConvert(m::MultiplyAnyOrder(
          OptionalBitcast(OptionalReshape(
              OptionalConvert(GetUnfusedReduceMaxSumSoftmaxPattern(
                  softmax_input, &softmax_reduce_sum,
                  &softmax_reduce_sum_bcast)))),
          m::Broadcast(
              OptionalConvert(m::Constant(dropout).WithPredicate(IsScalar))))),
      m::Op());

  // Form 2 -> softmax - mul - BMM2
  //                     /
  //                    /
  //                 select(dropout)
  auto dropout_softmax_pattern_form_2 =
      OptionalBitcast(OptionalBitcast(OptionalConvert(m::MultiplyAnyOrder(
          OptionalReshape(OptionalConvert(GetUnfusedReduceMaxSumSoftmaxPattern(
              softmax_input, &softmax_reduce_sum, &softmax_reduce_sum_bcast))),
          m::Broadcast(
              OptionalConvert(OptionalBitcast(OptionalReshape(m::Select(
                  m::Op(),
                  m::Broadcast(m::Constant(dropout).WithPredicate(IsScalar)),
                  m::Op())))))))));

  // Try matching BMM1 - (Scale) - (Bias) - (Mask) - Softmax - (Dropout) -
  // BMM2 Dropout with non-zero drop rate has select(divide(softmax_output,
  // broadcast(1-dropout_rate)))
  auto softmax_dropout_bmm2_pattern =
      m::Op(bmm_2)
          .WithPredicate(IsBatchedMatmul)
          .WithOperand(bmm2_operand_position,
                       m::AnyOf<HloInstruction>(
                           OptionalBitcast(OptionalConvert(
                               GetUnfusedReduceMaxSumSoftmaxPattern(
                                   softmax_input, &softmax_reduce_sum,
                                   &softmax_reduce_sum_bcast))),
                           dropout_softmax_pattern_form_1,
                           dropout_softmax_pattern_form_2));

  if (!Match(instr, softmax_dropout_bmm2_pattern) ||
      !IsSupportedPrimitiveType((*bmm_2))) {
    return false;
  }
  if (softmax_reduce_sum->users()[0]->opcode() == HloOpcode::kConvert) {
    softmax_reduce_sum = softmax_reduce_sum->users()[0];
  }
  is_training = softmax_reduce_sum->user_count() == 2 &&
                softmax_reduce_sum_bcast->user_count() == 2;
  return true;
}

bool MatchBmm1UnfusedBiasSoftmaxBmm2(HloInstruction* softmax_input,
                                     HloInstruction** bmm_1,
                                     HloInstruction** bias,
                                     HloInstruction** scale, bool has_dropout,
                                     HloInstruction* dropout,
                                     double& dropout_rate,
                                     std::string& custom_call_name) {
  auto first_bmm_pattern = m::SharedSubpattern(
      m::Op(bmm_1).WithPredicate(IsBatchedMatmul).WithOneUse());
  auto unfused_scaled_bmm_subpattern = m::MultiplyAnyOrder(
      OptionalConvert(first_bmm_pattern),
      OptionalConvert(
          m::Broadcast(m::Constant(scale).WithPredicate(IsScalar))));
  auto pattern =
      m::AddAnyOrder(OptionalConvert(m::AnyOf<HloInstruction>(
                         unfused_scaled_bmm_subpattern, first_bmm_pattern)),
                     m::Op(bias));

  if (Match(softmax_input, pattern)) {
    custom_call_name = has_dropout ? kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget
                                   : kCudnnfMHAScaleBiasSoftmaxCallTarget;
    if (has_dropout) {
      dropout_rate = GetDropoutRateFromHlo(dropout);
    }
    return true;
  }
  return false;
}

bool MatchBmm1ScaleBiasMaskSoftmaxDropoutBmm2(
    HloInstruction* softmax_input, HloInstruction** bmm_1,
    HloInstruction** bias, HloInstruction** scale, HloInstruction** mask,
    bool has_dropout, HloInstruction* dropout, double& dropout_rate,
    std::string& custom_call_name) {
  // This is the subpattern for unfused scaled gemm since cublas
  // doesn't always fuse the scale into alpha.
  auto unfused_scaled_bmm_subpattern = m::SharedSubpattern(m::MultiplyAnyOrder(
      OptionalConvert(m::Op(bmm_1).WithPredicate(IsBatchedMatmul).WithOneUse()),
      m::Broadcast(m::Constant(scale).WithPredicate(IsScalar))));
  auto pattern = OptionalConvert(m::Select(
      m::Op(mask).WithPredicate([](const HloInstruction* instr) {
        return instr->shape().element_type() == PRED;
      }),
      // Match bmm1-scale-bias-mask
      m::AnyOf<HloInstruction>(
          // Scale and bias might or might not be fused with gemm
          m::Op(bmm_1).WithPredicate(IsBatchedMatmul).WithOneUse(),
          OptionalConvert(m::AnyOf<HloInstruction>(
              // Try to match unfused bias
              m::AddAnyOrder(
                  m::Op(bias),
                  m::AnyOf<HloInstruction>(
                      OptionalConvert(m::Op(bmm_1)
                                          .WithPredicate(IsBatchedMatmul)
                                          .WithOneUse()),
                      unfused_scaled_bmm_subpattern)),
              unfused_scaled_bmm_subpattern))),
      m::Op()));

  if (Match(softmax_input, pattern)) {
    if (!IsSupportedPrimitiveType((*bmm_1))) {
      return false;
    }

    if (has_dropout) {
      // Found BMM1 - Scale - (bias) - Mask - Softmax - dropout - BMM2
      custom_call_name = (*bias) == nullptr
                             ? kCudnnfMHAScaleMaskSoftmaxDropoutCallTarget
                             : kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget;
      dropout_rate = GetDropoutRateFromHlo(dropout);
    } else {
      // Found BMM1 - Scale - Mask - Softmax - BMM2
      custom_call_name = (*bias) == nullptr
                             ? kCudnnfMHAScaleMaskSoftmaxCallTarget
                             : kCudnnfMHAScaleBiasMaskSoftmaxCallTarget;
    }
    return true;
  }
  return false;
}

// We will try to match all forward patterns below:
// BMM1 - Scale - Bias - Mask - Softmax - Dropout - BMM2
// BMM1 - Scale - Mask - Softmax - Dropout - BMM2
// BMM1 - Scale - Bias - Mask - Softmax - BMM2
// BMM1 - Scale - Mask - Softmax - BMM2
// BMM1 - Scale - bias - Softmax - BMM2
// BMM1 - Softmax - Dropout - BMM2
// BMM1 - Softmax - BMM2
// BMM1 - BMM2
bool MatchFwdMHAPatternsForCanonicalization(
    HloInstruction* instr, HloInstruction** bmm_1, HloInstruction** bmm_2,
    HloInstruction** bias, HloInstruction** mask, HloInstruction** scale,
    double& dropout_rate, std::string& custom_call_name,
    bool& need_canonicalization, bool& is_training) {
  // We need to match 2 general cases:
  // 1. bmm1 --> (intermediate nodes) --> bmm2 <-- V matrix
  // 2. V matrix --> bmm2 <-- (intermediate nodes) <-- bmm1
  // to determine if we need to canonicalize bmm2.
  // So we go through both of bmm2's operands and see which one matches our
  // desired patterns, if operand 1 consumes them, then we need to canonicalize.
  for (int bmm2_operand_pos : {0, 1}) {
    if (bmm2_operand_pos == 1) {
      need_canonicalization = true;
    }
    if (MatchDefaultFwdBmmBmm(bmm2_operand_pos, instr, bmm_1, bmm_2,
                              custom_call_name, is_training)) {
      return true;
    }

    HloInstruction* softmax_input = nullptr;

    HloInstruction* dropout = nullptr;

    bool has_dropout = false;
    // We first check if bmm2 is connect to a softmax or dropout.
    // If so, we set softmax input and dropout nodes to their corresponding ops.
    if (!MatchSoftmaxDropoutBmm(bmm2_operand_pos, instr, bmm_2, &softmax_input,
                                &dropout, is_training)) {
      continue;
    }
    has_dropout = dropout != nullptr;
    if (MatchBmm1UnfusedBiasSoftmaxBmm2(softmax_input, bmm_1, bias, scale,
                                        has_dropout, dropout, dropout_rate,
                                        custom_call_name)) {
      return true;
    }
    if (MatchBmm1ScaleBiasMaskSoftmaxDropoutBmm2(
            softmax_input, bmm_1, bias, scale, mask, has_dropout, dropout,
            dropout_rate, custom_call_name)) {
      return true;
    }
  }
  // Didn't find any match
  need_canonicalization = false;
  return false;
}

bool IsBmm2GradGemm2(HloInstruction* instr) {
  // Check to see if input bmm is bmm2 gradient gemm2, it needs to be either:
  // 1. having 1 user in cases of dropout
  // 2. having 2 users in other cases.
  return (instr->user_count() == 1) || (instr->user_count() == 2);
}

bool MatchBmm1GradGemm1(
    HloInstruction* fwd_fmha_call, HloInstruction* bmm_1,
    HloInstruction** bmm_1_grad_1,
    std::vector<HloInstruction**>& bmms_need_canonicalization) {
  const HloInstruction* q_tensor = fwd_fmha_call->operand(0);
  for (int64_t i = 0; i < q_tensor->user_count(); i++) {
    HloInstruction* q_tensor_user_i = q_tensor->users()[i];
    if (IsBatchedMatmul(q_tensor_user_i) && q_tensor_user_i != bmm_1) {
      *bmm_1_grad_1 = q_tensor_user_i;
      // Check for canonicalization.
      if ((*bmm_1_grad_1)->operand_index(q_tensor) != 1) {
        bmms_need_canonicalization.push_back(bmm_1_grad_1);
      }
      return true;
    }
  }
  return false;
}

bool MatchBmm1GradGemm2(
    HloInstruction* fwd_fmha_call, HloInstruction** bmm_1_grad_2,
    HloInstruction** bmm_1_grad_1,
    std::vector<HloInstruction**>& bmms_need_canonicalization) {
  // bmm1 gradient gemm2 shares the same input as bmm1 gradient gemm1.
  // Check to see if bmm1 grad gemm1 needs canonicalization or not, if not,
  // then the shared input is the first operand.
  int64_t parent_nodex_index =
      std::find_if(bmms_need_canonicalization.begin(),
                   bmms_need_canonicalization.end(),
                   [&](HloInstruction** instr) {
                     return (*instr) == (*bmm_1_grad_1);
                   }) == bmms_need_canonicalization.end()
          ? 0
          : 1;
  HloInstruction* d_s_user_0 = (*bmm_1_grad_1);

  HloInstruction* parent_node = d_s_user_0->mutable_operand(parent_nodex_index);
  if (parent_node->opcode() == HloOpcode::kBitcast &&
      parent_node->user_count() == 1) {
    d_s_user_0 = parent_node;
    parent_node = parent_node->mutable_operand(0);
  }

  auto bmm_1_grad_2_it =
      std::find_if(parent_node->users().begin(), parent_node->users().end(),
                   [&](HloInstruction* instr) {
                     return instr != (*bmm_1_grad_1) &&
                            instr->opcode() != HloOpcode::kReduce;
                   });
  if (bmm_1_grad_2_it != parent_node->users().end()) {
    *bmm_1_grad_2 = *bmm_1_grad_2_it;
  } else {
    return false;
  }
  if ((*bmm_1_grad_2)->opcode() == HloOpcode::kBitcast &&
      (*bmm_1_grad_2)->user_count() == 1) {
    parent_node = (*bmm_1_grad_2);
    (*bmm_1_grad_2) = (*bmm_1_grad_2)->users()[0];
  }

  if ((*bmm_1_grad_2)->operand_index(parent_node) != 0) {
    bmms_need_canonicalization.push_back(bmm_1_grad_2);
  }
  return true;
}

bool MatchBmm2GradGemm1(
    HloInstruction* fwd_fmha_call, HloInstruction** bmm_2_grad_1,
    std::vector<HloInstruction**>& bmms_need_canonicalization) {
  // The second GTE of the forward MHA call is the input of the bmm2's gradient
  // gemm 1, we check to see if the current gemm satisfies above condition.
  int64_t activation_out_gte_index = 1;
  if (fwd_fmha_call->user_count() < 2 ||
      fwd_fmha_call->users()[activation_out_gte_index]->opcode() !=
          HloOpcode::kGetTupleElement ||
      fwd_fmha_call->users()[activation_out_gte_index]->user_count() > 1 ||
      !IsBatchedMatmul(
          fwd_fmha_call->users()[activation_out_gte_index]->users()[0])) {
    return false;
  }
  // Found fmha->GTE->gemm, assign it to bmm_2_grad_1 and check to see if it
  // needs canonicalization.
  *bmm_2_grad_1 = fwd_fmha_call->users()[activation_out_gte_index]->users()[0];
  if ((*bmm_2_grad_1)
          ->operand_index(fwd_fmha_call->users()[activation_out_gte_index]) !=
      0) {
    bmms_need_canonicalization.push_back(bmm_2_grad_1);
  }
  return true;
}

bool MatchBmm2GradGemm2(
    HloInstruction* fwd_fmha_call, HloInstruction** bmm_2_grad_2,
    bool v_transposed,
    std::vector<HloInstruction**>& bmms_need_canonicalization) {
  // If v tensor is transposed by forward fmha call, then we need to take fmha v
  // input's producer's producer.
  const HloInstruction* v_tensor = v_transposed
                                       ? fwd_fmha_call->operand(2)->operand(0)
                                       : fwd_fmha_call->operand(2);
  for (int64_t i = 0; i < v_tensor->user_count(); i++) {
    HloInstruction* v_tensor_user_i = v_tensor->users()[i];
    if (IsBatchedMatmul(v_tensor_user_i) && IsBmm2GradGemm2(v_tensor_user_i)) {
      *bmm_2_grad_2 = v_tensor_user_i;
      // Check for canonicalization.
      if ((*bmm_2_grad_2)->operand_index(v_tensor) != 1) {
        bmms_need_canonicalization.push_back(bmm_2_grad_2);
      }
      return true;
    }
  }

  return false;
}

bool MatchBwdBmmSoftmaxDropoutBmm(HloInstruction* fwd_fmha_call,
                                  HloInstruction* bmm_1_grad_1,
                                  const HloInstruction* bmm_2_grad_2,
                                  HloInstruction** d_intermediate,
                                  HloInstruction** mask,
                                  std::string& bwd_custom_call_name,
                                  bool is_bmm1_grad1_canonicalized) {
  bool has_dropout = false;
  bool has_mask = false;
  // Backward dropout pattern
  // select(mask, bmm2_grad2, broadcast())
  auto bwd_dropout_pattern_form_1 =
      OptionalBitcast(OptionalReshape(OptionalConvert(m::Select(
          m::Op(), m::Op().WithPredicate([&](const HloInstruction* instr) {
            return instr == bmm_2_grad_2;
          }),
          m::Broadcast(
              OptionalConvert(m::Constant().WithPredicate(IsScalar)))))));

  // multiply(bmm2_grad2, broadcast(select(mask, broadcast(), op())))
  auto bwd_dropout_pattern_form_2 = OptionalBitcast(m::MultiplyAnyOrder(
      OptionalConvert(m::Op().WithPredicate(
          [&](const HloInstruction* instr) { return instr == bmm_2_grad_2; })),
      m::Broadcast(OptionalConvert(OptionalBitcast(OptionalReshape(m::Select(
          m::Op(),
          m::Broadcast(OptionalConvert(m::Constant().WithPredicate(IsScalar))),
          m::Op())))))));
  auto bwd_dropout_pattern = m::AnyOf<HloInstruction>(
      bwd_dropout_pattern_form_1, bwd_dropout_pattern_form_2);
  // Backward softmax pattern
  HloInstruction* bwd_softmax_input = nullptr;
  HloInstruction* exp_1;
  HloInstruction* exp_2;
  HloInstruction* d_softmax;

  auto bwd_softmax_pattern =
      OptionalBitcast(OptionalConvert(m::MultiplyAnyOrder(
          &d_softmax,
          m::AddAnyOrder(
              m::Divide(),
              m::Broadcast(OptionalBitcast(
                  OptionalConvert(OptionalConvert(m::Negate(OptionalBitcast(
                      m::Op()
                          .WithPredicate(IsReduceSum)
                          .WithOperand(0, OptionalBitcast(m::MultiplyAnyOrder(
                                              m::MultiplyAnyOrder(
                                                  m::Op(&bwd_softmax_input),
                                                  m::Broadcast()),
                                              m::Exp(&exp_2, m::Op()))))))))))),
          m::Exp(&exp_1, m::Op()))));

  // Backward mask input pattern
  // we already matched this in the fwd. Just make sure the mask is used in the
  // bwd
  HloInstruction* bwd_mask_input = nullptr;
  HloInstruction* bwd_mask = nullptr;
  auto bwd_mask_pattern = OptionalConvert(
      m::Select(m::Op(&bwd_mask).WithPredicate([](const HloInstruction* instr) {
        return instr->shape().element_type() == PRED;
      }),
                m::Op(&bwd_mask_input), m::Op()));

  // Backward scale input pattern
  HloInstruction* bwd_scale_input = nullptr;

  auto bwd_scale_pattern =
      m::MultiplyAnyOrder(m::Op(&bwd_scale_input),
                          m::Broadcast(m::Constant().WithPredicate(IsScalar)));
  int intermediate_input_pos = is_bmm1_grad1_canonicalized ? 1 : 0;
  HloInstruction* intermediate_input =
      bmm_1_grad_1->mutable_operand(intermediate_input_pos);
  if (Match(intermediate_input, bwd_scale_pattern)) {
    intermediate_input = bwd_scale_input;
  }

  has_mask = Match(intermediate_input, bwd_mask_pattern) && *mask == bwd_mask;
  if (has_mask) {
    intermediate_input = bwd_mask_input;
  }
  if (!Match(intermediate_input, bwd_softmax_pattern) || exp_1 != exp_2) {
    return false;
  }
  has_dropout = Match(bwd_softmax_input, bwd_dropout_pattern);
  // If no dropout but softmax input is not coming from bmm2 gradient gemm 2,
  // then it's not the pattern that we care about.
  if (!has_dropout &&
      !Match(bwd_softmax_input,
             OptionalConvert((OptionalBitcast(
                 m::Op().WithPredicate([&](const HloInstruction* instr) {
                   return instr == bmm_2_grad_2;
                 })))))) {
    return false;
  }

  if (has_mask && has_dropout) {
    // has bias
    if (fwd_fmha_call->custom_call_target() ==
        kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget)
      bwd_custom_call_name =
          kCudnnfMHAScaleBiasMaskSoftmaxDropoutBackwardCallTarget;
    // no bias
    if (fwd_fmha_call->custom_call_target() ==
        kCudnnfMHAScaleMaskSoftmaxDropoutCallTarget)
      bwd_custom_call_name =
          kCudnnfMHAScaleMaskSoftmaxDropoutBackwardCallTarget;
  } else if (!has_mask && has_dropout) {
    // has bias
    if (fwd_fmha_call->custom_call_target() ==
        kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget)
      bwd_custom_call_name =
          kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget;
    // no bias
    if (fwd_fmha_call->custom_call_target() ==
        kCudnnfMHASoftmaxDropoutCallTarget)
      bwd_custom_call_name = kCudnnfMHASoftmaxDropoutBackwardCallTarget;
  } else if (has_mask && !has_dropout) {
    // has bias
    if (fwd_fmha_call->custom_call_target() ==
        kCudnnfMHAScaleBiasMaskSoftmaxCallTarget)
      bwd_custom_call_name = kCudnnfMHAScaleBiasMaskSoftmaxBackwardCallTarget;
    // no bias
    if (fwd_fmha_call->custom_call_target() ==
        kCudnnfMHAScaleMaskSoftmaxCallTarget)
      bwd_custom_call_name = kCudnnfMHAScaleMaskSoftmaxBackwardCallTarget;
  } else {
    // has bias
    if (fwd_fmha_call->custom_call_target() ==
        kCudnnfMHAScaleBiasSoftmaxCallTarget)
      bwd_custom_call_name = kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget;
    // no bias
    if (fwd_fmha_call->custom_call_target() == kCudnnfMHASoftmaxCallTarget)
      bwd_custom_call_name = kCudnnfMHASoftmaxBackwardCallTarget;
  }

  // If d_softmax tensor has 3 consumers, then we need to output the
  // intermediate tensor.
  bool need_d_intermediate = d_softmax->user_count() == 3;
  if ((bwd_custom_call_name ==
           kCudnnfMHAScaleBiasSoftmaxDropoutBackwardCallTarget ||
       bwd_custom_call_name == kCudnnfMHAScaleBiasSoftmaxBackwardCallTarget ||
       bwd_custom_call_name ==
           kCudnnfMHAScaleBiasMaskSoftmaxDropoutBackwardCallTarget ||
       bwd_custom_call_name ==
           kCudnnfMHAScaleBiasMaskSoftmaxBackwardCallTarget) &&
      need_d_intermediate) {
    (*d_intermediate) = d_softmax;
  }
  return true;
}
// First, we look for the bmm2 gradient gemm 1 which takes the activation
// output from a forward fmha call.
// Secondly, look for bmm2 gradient gemm 2 that takes the v tensor as an
// input. We take the v tensor from the third operand of the forward fmha
// call. If forward is canonicalized, then we skip the additional transpose in
// between.
// Then we look for bmm1 gradient gemm1 by searching for gemms that share q
// tensor with current fmha call.
bool MatchBackwardBmms(
    HloInstruction* fwd_fmha_call, HloInstruction* bmm_1,
    HloInstruction** bmm_1_grad_1, HloInstruction** bmm_1_grad_2,
    HloInstruction** bmm_2_grad_1, HloInstruction** bmm_2_grad_2,
    bool v_transposed,
    std::vector<HloInstruction**>& bmms_need_canonicalization) {
  return MatchBmm2GradGemm1(fwd_fmha_call, bmm_2_grad_1,
                            bmms_need_canonicalization) &&
         MatchBmm2GradGemm2(fwd_fmha_call, bmm_2_grad_2, v_transposed,
                            bmms_need_canonicalization) &&
         MatchBmm1GradGemm1(fwd_fmha_call, bmm_1, bmm_1_grad_1,
                            bmms_need_canonicalization) &&
         MatchBmm1GradGemm2(fwd_fmha_call, bmm_1_grad_2, bmm_1_grad_1,
                            bmms_need_canonicalization);
}
// We will match the backward graphs for all forward patterns defined in
// MatchFwdMHAPatternsForCanonicalization
bool MatchBwdMHAPatternsForCanonicalization(
    HloInstruction* fwd_fmha_call, HloInstruction* bmm_1,
    HloInstruction** bmm_1_grad_1, HloInstruction** bmm_1_grad_2,
    HloInstruction** bmm_2_grad_1, HloInstruction** bmm_2_grad_2,
    HloInstruction** d_intermediate, HloInstruction** mask,
    std::string& bwd_custom_call_name, bool v_transposed,
    std::vector<HloInstruction**>& bmms_need_canonicalization) {
  if (!MatchBackwardBmms(fwd_fmha_call, bmm_1, bmm_1_grad_1, bmm_1_grad_2,
                         bmm_2_grad_1, bmm_2_grad_2, v_transposed,
                         bmms_need_canonicalization)) {
    return false;
  }

  // Found default bmm-bmm backward graph.
  if ((*bmm_2_grad_2)->users().size() == 2 &&
      ((*bmm_1_grad_1)->IsUserOf((*bmm_2_grad_2))) &&
      ((*bmm_1_grad_2)->IsUserOf((*bmm_2_grad_2)))) {
    bwd_custom_call_name = kCudnnfMHABmmBmmBackwardCallTarget;
    return true;
  }
  // TODO match all other patterns
  bool is_bmm1_grad1_canonicalized = false;
  for (auto bmm : bmms_need_canonicalization) {
    is_bmm1_grad1_canonicalized |= (bmm == bmm_1_grad_1);
  }
  if (MatchBwdBmmSoftmaxDropoutBmm(
          fwd_fmha_call, (*bmm_1_grad_1), (*bmm_2_grad_2), d_intermediate, mask,
          bwd_custom_call_name, is_bmm1_grad1_canonicalized)) {
    return true;
  }
  return false;
}

StatusOr<bool> IsMHABlockSupported(HloInstruction* bmm_1, HloInstruction* bmm_2,
                                   bool need_canonicalization, bool is_training,
                                   std::string& custom_call_name,
                                   const DebugOptions& debug_options) {
  if (MHACallHasDropout(custom_call_name) &&
      !debug_options.xla_gpu_fused_attention_use_cudnn_rng()) {
    VLOG(3) << "Using CUDNN RNG for fused attention dropout is not enabled.\n";
    return false;
  }

  if (is_training &&
      (custom_call_name != kCudnnfMHAScaleBiasSoftmaxDropoutCallTarget &&
       custom_call_name != kCudnnfMHAScaleBiasSoftmaxCallTarget &&
       custom_call_name != kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget &&
       custom_call_name != kCudnnfMHAScaleBiasMaskSoftmaxCallTarget)) {
    VLOG(3) << "Unsupported fused MHA training pattern.\n";
    return false;
  }

  // cuDNN 8.8 currently only supports BF16 and F16 data types.
  if (!IsSupportedPrimitiveType(bmm_1) || !IsSupportedPrimitiveType(bmm_2)) {
    if (VLOG_IS_ON(2)) {
      VLOG(2) << "Unsupported primitive type for cuDNN MHA fusion:\n"
              << bmm_1->ToString() << "\nOR\n"
              << bmm_2->ToString() << "\n"
              << "BF16 and F16 are the supported Dtypes.";
    }
    return false;
  }

  TF_ASSIGN_OR_RETURN(bool is_bmm1_supported, IsSupportedBMM1(bmm_1));
  if (!is_bmm1_supported) return false;
  TF_ASSIGN_OR_RETURN(bool is_bmm2_supported,
                      IsSupportedBMM2(bmm_2, need_canonicalization));
  if (!is_bmm2_supported) return false;
  return true;
}

StatusOr<HloInstruction*> CanonicalizeBatchedGemmForcuDNNFMHA(
    HloInstruction* bmm, HloComputation* comp) {
  if (VLOG_IS_ON(3)) {
    VLOG(3) << "Before FMHA Dot Cannonicalization: \n"
            << comp->parent()->ToString();
  }
  HloInstruction* lhs_bmm = bmm->mutable_operand(0);
  HloInstruction* rhs_bmm = bmm->mutable_operand(1);
  const DotDimensionNumbers& dnums = bmm->dot_dimension_numbers();

  int64_t rank = bmm->shape().dimensions_size();
  std::vector<int64_t> perm(rank);
  std::iota(perm.begin(), perm.end(), 0);
  // Swap the non-contracting dims of BMM shape. By contract, the
  // non-contracting dims in the output are the last two dimensions.
  std::swap(perm[rank - 1], perm[rank - 2]);

  DotDimensionNumbers new_dnums = dnums;
  std::swap(*new_dnums.mutable_lhs_contracting_dimensions(),
            *new_dnums.mutable_rhs_contracting_dimensions());
  std::swap(*new_dnums.mutable_lhs_batch_dimensions(),
            *new_dnums.mutable_rhs_batch_dimensions());
  auto original_bmm_shape = bmm->shape();
  HloInstruction* new_dot = comp->AddInstruction(HloInstruction::CreateDot(
      ShapeUtil::MakeShape(original_bmm_shape.element_type(),
                           Permute(original_bmm_shape.dimensions(), perm)),
      /* lhs */ rhs_bmm, /* rhs */ lhs_bmm, new_dnums,
      bmm->precision_config()));

  TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
      bmm, HloInstruction::CreateTranspose(original_bmm_shape, new_dot, perm)));
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "After FMHA Dot Cannonicalization: \n"
            << comp->parent()->ToString();
  }
  return new_dot;
}

StatusOr<HloInstruction*> ChangeCheckedDimToFastest(
    HloComputation* comp, HloInstruction* bmm, bool is_lhs,
    bool should_contracting_be_fastest) {
  const DotDimensionNumbers& dot_dims_bmm = bmm->dot_dimension_numbers();
  DotDimensionNumbers new_dot_dims_bmm = dot_dims_bmm;
  int64_t bmm_operand = is_lhs ? 0 : 1;
  absl::Span<const int64_t> contracting_dims =
      is_lhs ? dot_dims_bmm.lhs_contracting_dimensions()
             : dot_dims_bmm.rhs_contracting_dimensions();
  absl::Span<const int64_t> batch_dims =
      is_lhs ? dot_dims_bmm.lhs_batch_dimensions()
             : dot_dims_bmm.rhs_batch_dimensions();
  absl::Span<const int64_t> lhs_minor_to_major_bmm =
      bmm->operand(0)->shape().layout().minor_to_major();
  absl::Span<const int64_t> rhs_minor_to_major_bmm =
      bmm->operand(1)->shape().layout().minor_to_major();

  absl::Span<const int64_t>& minor_to_major_to_check =
      is_lhs ? lhs_minor_to_major_bmm : rhs_minor_to_major_bmm;

  CHECK_EQ(contracting_dims.size(), 1);
  TF_ASSIGN_OR_RETURN(std::vector<int64_t> non_contracting_dim_nums_bmm,
                      GetNonContractingDims(bmm->operand(bmm_operand)->shape(),
                                            batch_dims, contracting_dims));
  CHECK_EQ(non_contracting_dim_nums_bmm.size(), 1);
  HloInstruction* operand_bmm = bmm->mutable_operand(bmm_operand);
  std::vector<int64_t> contracting_dims_to_check{contracting_dims[0]};
  std::vector<int64_t> dims_to_set = should_contracting_be_fastest
                                         ? contracting_dims_to_check
                                         : non_contracting_dim_nums_bmm;
  // If the dimension being checked(contracting or non-contracting) of the
  // target operand is not the fastest moving dimension, make it so.
  if (minor_to_major_to_check[0] != dims_to_set[0]) {
    std::vector<int64_t> perm(bmm->shape().dimensions_size());
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[dims_to_set[0]], perm[minor_to_major_to_check[0]]);

    if (is_lhs) {
      new_dot_dims_bmm.set_lhs_contracting_dimensions(
          0, non_contracting_dim_nums_bmm[0]);
    } else {
      new_dot_dims_bmm.set_rhs_contracting_dimensions(
          0, non_contracting_dim_nums_bmm[0]);
    }

    operand_bmm = comp->AddInstruction(
        HloInstruction::CreateTranspose(
            ShapeUtil::MakeShapeWithDenseLayout(
                bmm->shape().element_type(),
                Permute(operand_bmm->shape().dimensions(), perm),
                rhs_minor_to_major_bmm),
            operand_bmm, perm),
        &operand_bmm->metadata());
    *((DynCast<HloDotInstruction>(bmm))->mutable_dot_dimension_numbers()) =
        new_dot_dims_bmm;
  }
  return operand_bmm;
}

StatusOr<HloInstruction*> FuseFwdMultiHeadedAttentionBlock(
    HloComputation* comp, HloInstruction* bmm_1, HloInstruction* bmm_2,
    HloInstruction* bias, HloInstruction* mask, HloInstruction* scale,
    double dropout_rate, std::string& custom_call_name,
    stream_executor::CudaComputeCapability cc, bool is_training, bool& changed,
    bool& v_transposed) {
  double scale_value = 1.0;
  HloInstruction* lhs_bmm1;
  HloInstruction* rhs_bmm1;
  HloInstruction* rhs_bmm2;
  TF_ASSIGN_OR_RETURN(rhs_bmm1, ChangeCheckedDimToFastest(
                                    comp, bmm_1, false /*is_lhs*/,
                                    true /*should_contracting_be_fastest*/));
  TF_ASSIGN_OR_RETURN(lhs_bmm1, ChangeCheckedDimToFastest(
                                    comp, bmm_1, true /*is_lhs*/,
                                    true /*should_contracting_be_fastest*/));

  TF_ASSIGN_OR_RETURN(rhs_bmm2, ChangeCheckedDimToFastest(
                                    comp, bmm_2, false /*is_lhs*/,
                                    false /*should_contracting_be_fastest*/));

  if (rhs_bmm2 != bmm_2->mutable_operand(1)) {
    v_transposed = true;
  }

  CudnnfMHABackendConfig fmha_config;
  *fmha_config.mutable_bmm1_dot_dimension_numbers() =
      bmm_1->dot_dimension_numbers();
  *fmha_config.mutable_bmm2_dot_dimension_numbers() =
      bmm_2->dot_dimension_numbers();

  TF_RET_CHECK((dropout_rate >= 0.0 && dropout_rate <= 1.0));

  // If scale node is assigned, extract value from it.
  if (scale != nullptr) {
    std::optional<double> value;
    value = GetConstantValue(scale);
    TF_RET_CHECK(value.has_value());
    scale_value = (double)*value;
  }

  fmha_config.set_fmha_scale(scale_value);
  fmha_config.set_dropout_rate(dropout_rate);
  // Set to an arbitrary seed for now, seed is not exposed to XLA in HLO
  // graph.
  // TODO Find a way to compute original seed from dropout keys.
  fmha_config.set_seed(42);

  *fmha_config.mutable_intermediate_tensor_shape() = bmm_1->shape().ToProto();
  {
    auto* algorithm = fmha_config.mutable_algorithm();
    algorithm->set_algo_id(0);  // engine id
    algorithm->set_math_type(se::dnn::AlgorithmProto::TENSOR_OP_MATH);
    std::vector<int64_t> knob_ids = /* {0, 1} */ {17, 24};
    std::vector<int64_t> knob_vals = {1, 0};
    for (int i = 0; i < knob_ids.size(); ++i) {
      (*algorithm->mutable_tuning_knobs())[knob_ids[i]] = knob_vals[i];
    }
    algorithm->set_is_cudnn_frontend(true);
    algorithm->mutable_workspace_size()->set_value(0);
  }
  const Shape& output_shape = bmm_2->shape();

  Shape call_shape;
  // Activation output is used by backward gemm.
  HloInstruction* activation_output = nullptr;

  std::vector<Shape> output_shapes = {output_shape,
                                      ShapeUtil::MakeShape(U8, {0})};
  if (is_training) {
    // TODO Flush attention will have a different shape in training.
    activation_output = bmm_2->mutable_operand(0);
    // Sometimes activation output is bitcast, the actual activation is the
    // second user of the producer of bmm_2's first operand.
    if (activation_output->user_count() < 2 &&
        activation_output->opcode() == HloOpcode::kBitcast) {
      HloInstruction* producer = activation_output->mutable_operand(0);
      TF_RET_CHECK(producer->user_count() == 2);
      activation_output = producer->UserId(activation_output) == 0
                              ? producer->users()[1]
                              : producer->users()[0];
    }
    output_shapes.push_back(activation_output->shape());
  }
  call_shape = ShapeUtil::MakeTupleShape(output_shapes);

  std::vector<HloInstruction*> operands = {lhs_bmm1, rhs_bmm1, rhs_bmm2};
  if (mask != nullptr) {
    HloInstruction* converted_mask = comp->AddInstruction(
        HloInstruction::CreateConvert(bmm_1->shape(), mask));
    operands.push_back(converted_mask);
  }
  if (bias != nullptr) {
    HloInstruction* original_bias;
    HloInstruction* original_broadcast;
    // There will be cases where the bias is up-casted to wider float type,
    // we need to take the original bias node and broadcast it without
    // converting.
    if (Match(bias, m::Broadcast(
                        &original_broadcast,
                        m::Convert(
                            m::Op(&original_bias)
                                .WithPredicate([](const HloInstruction* instr) {
                                  return instr->shape().element_type() == F16 ||
                                         instr->shape().element_type() == BF16;
                                }))
                            .WithPredicate([](const HloInstruction* instr) {
                              return instr->shape().element_type() == F32 ||
                                     instr->shape().element_type() == F64;
                            })))) {
      absl::Span<const int64_t> original_bcast_dims =
          (DynCast<HloBroadcastInstruction>(original_broadcast))->dimensions();
      // This is to deal with cases like paxml where an extra dimension of 1 is
      // added to the left of the tensor.
      // TODO Make this logic more generic
      absl::Span<const int64_t> original_broadcast_shape_dims =
          original_broadcast->shape().dimensions();
      int64_t starting_index = original_broadcast_shape_dims.size() == 5 &&
                                       original_broadcast_shape_dims[0] == 1
                                   ? 1
                                   : 0;
      std::vector<int64_t> bcast_dimensions;
      for (auto& dim : original_bcast_dims) {
        bcast_dimensions.push_back(dim - starting_index);
      }

      Shape bcast_shape = bmm_1->shape();
      bias = comp->AddInstruction(HloInstruction::CreateBroadcast(
          bcast_shape, original_bias, bcast_dimensions));
    }
    operands.push_back(bias);
  }

  HloInstruction* fmha_call =
      comp->AddInstruction(HloInstruction::CreateCustomCall(
          call_shape, operands, absl::string_view(custom_call_name)));
  TF_RETURN_IF_ERROR(fmha_call->set_backend_config(fmha_config));
  TF_RETURN_IF_ERROR(SetFMHAInstructionName(bmm_1->GetModule(), fmha_call));
  fmha_call->set_metadata(bmm_1->metadata());

  TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
      bmm_2,
      HloInstruction::CreateGetTupleElement(bmm_2->shape(), fmha_call, 0)));

  if (activation_output) {
    TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
        activation_output, HloInstruction::CreateGetTupleElement(
                               activation_output->shape(), fmha_call, 2)));
  }

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "After CudnnFusedMHARewriter: \n" << comp->parent()->ToString();
  }
  changed = true;
  return fmha_call;
}

bool IsDbiasOnlyUserBesidesGradGemm(HloInstruction* d_intermediate,
                                    HloInstruction* bmm_1_grad_1,
                                    HloInstruction* bmm_1_grad_2,
                                    HloInstruction** dbias) {
  auto user_count = d_intermediate->user_count();
  HloInstruction* dbias_user = nullptr;
  for (auto user : d_intermediate->users()) {
    if (user == bmm_1_grad_1) {
      user_count -= 1;
    } else if (user == bmm_1_grad_2) {
      user_count -= 1;
    } else {
      dbias_user = user;
    }
  }
  HloInstruction* reduce;
  auto ConsumeExtraConvert = [](HloInstruction** instr) {
    Match((*instr)->users()[0], m::Convert(instr, m::Op()).WithOneUse());
    return true;
  };
  // user_count == 1 && (reduce-> {convert} ->bitcast)
  return user_count == 1 &&
         Match(dbias_user, m::Reduce(&reduce, m::Op(), m::Op()).WithOneUse()) &&
         ConsumeExtraConvert(&reduce) &&
         Match(reduce->users()[0],
               m::AnyOf<HloInstruction>(m::Reshape(dbias, m::Op()),
                                        m::Bitcast(dbias, m::Op()))
                   .WithOneUse());
};

StatusOr<bool> FuseBwdMultiHeadedAttentionBlock(
    HloComputation* comp, HloInstruction* bmm_1_grad_1,
    HloInstruction* bmm_1_grad_2, HloInstruction* bmm_2_grad_1,
    HloInstruction* bmm_2_grad_2, HloInstruction* fwd_fmha_call,
    HloInstruction* d_intermediate, HloInstruction* mask,
    std::string& bwd_custom_call_name, bool fwd_bmm_2_canonicalized,
    bool is_bmm2_grad1_canonicalized) {
  HloInstruction* rhs_bmm1_grad_gemm1;
  HloInstruction* lhs_bmm1_grad_gemm2;
  HloInstruction* lhs_bmm2_grad_gemm1;
  HloInstruction* rhs_bmm2_grad_gemm2;
  HloInstruction* d_output_grad;

  // Q tensor
  TF_ASSIGN_OR_RETURN(
      rhs_bmm1_grad_gemm1,
      ChangeCheckedDimToFastest(comp, bmm_1_grad_1, false /*is_lhs*/,
                                false /*should_contracting_be_fastest*/));
  // K tensor
  TF_ASSIGN_OR_RETURN(
      lhs_bmm1_grad_gemm2,
      ChangeCheckedDimToFastest(comp, bmm_1_grad_2, false /*is_lhs*/,
                                false /*should_contracting_be_fastest*/));
  // Forward activation
  TF_ASSIGN_OR_RETURN(
      lhs_bmm2_grad_gemm1,
      ChangeCheckedDimToFastest(comp, bmm_2_grad_1, true /*is_lhs*/,
                                false /*should_contracting_be_fastest*/));
  // V tensor
  TF_ASSIGN_OR_RETURN(
      rhs_bmm2_grad_gemm2,
      ChangeCheckedDimToFastest(comp, bmm_2_grad_2, false /*is_lhs*/,
                                true /*should_contracting_be_fastest*/));
  // d output
  // Since d_o is the input of 2 bmms, we set the dim number using the
  // constraint
  // -> the contracting dimension of the lhs of bmm_2_grad_2 needs to be the
  // fastest moving dimension.
  TF_ASSIGN_OR_RETURN(d_output_grad, ChangeCheckedDimToFastest(
                                         comp, bmm_2_grad_2, true /*is_lhs*/,
                                         true /*check_contracting_dim*/));
  // Operand order {Q, K, V, Fwd act, d_o, mask*}
  std::vector<HloInstruction*> operands = {
      rhs_bmm1_grad_gemm1, lhs_bmm1_grad_gemm2, rhs_bmm2_grad_gemm2,
      lhs_bmm2_grad_gemm1, d_output_grad};
  if (mask) {
    HloInstruction* converted_mask = comp->AddInstruction(
        HloInstruction::CreateConvert(bmm_2_grad_2->shape(), mask));
    operands.push_back(converted_mask);
  }
  TF_ASSIGN_OR_RETURN(CudnnfMHABackendConfig fwd_config,
                      fwd_fmha_call->backend_config<CudnnfMHABackendConfig>());
  CudnnfMHABackendConfig bwd_fmha_config;

  // If forward bmm_2 is canonicalized, the contracting dimension of lhs
  // of bmm_2_grad_1 needs to be changed to the non-contracting dimension.

  if (fwd_bmm_2_canonicalized) {
    TF_ASSIGN_OR_RETURN(
        std::vector<int64_t> bmm_2_grad_1_lhs_non_contracting_dims,
        GetNonContractingDims(
            bmm_2_grad_1->shape(),
            bmm_2_grad_1->dot_dimension_numbers().lhs_batch_dimensions(),
            bmm_2_grad_1->dot_dimension_numbers()
                .lhs_contracting_dimensions()));
    CHECK(bmm_2_grad_1_lhs_non_contracting_dims.size() == 1);
    (DynCast<HloDotInstruction>(bmm_2_grad_1))
        ->mutable_dot_dimension_numbers()
        ->set_lhs_contracting_dimensions(
            0, bmm_2_grad_1_lhs_non_contracting_dims[0]);
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> bmm_2_grad_1_new_contracting_dims,
      GetNonContractingDims(
          bmm_2_grad_1->shape(),
          bmm_2_grad_1->dot_dimension_numbers().rhs_batch_dimensions(),
          bmm_2_grad_1->dot_dimension_numbers().rhs_contracting_dimensions()));

  if (is_bmm2_grad1_canonicalized) {
    (DynCast<HloDotInstruction>(bmm_2_grad_1))
        ->mutable_dot_dimension_numbers()
        ->set_rhs_contracting_dimensions(0,
                                         bmm_2_grad_1_new_contracting_dims[0]);
  }

  *bwd_fmha_config.mutable_bmm1_grad_gemm1_dot_dimension_numbers() =
      bmm_1_grad_1->dot_dimension_numbers();
  *bwd_fmha_config.mutable_bmm1_grad_gemm2_dot_dimension_numbers() =
      bmm_1_grad_2->dot_dimension_numbers();
  *bwd_fmha_config.mutable_bmm2_grad_gemm1_dot_dimension_numbers() =
      bmm_2_grad_1->dot_dimension_numbers();
  *bwd_fmha_config.mutable_bmm2_grad_gemm2_dot_dimension_numbers() =
      bmm_2_grad_2->dot_dimension_numbers();

  bwd_fmha_config.set_fmha_scale(fwd_config.fmha_scale());
  bwd_fmha_config.set_dropout_rate(fwd_config.dropout_rate());
  // Set to an arbitrary seed for now, seed is not exposed to XLA in HLO
  // graph.
  // TODO Find a way to compute original seed from dropout keys.
  bwd_fmha_config.set_seed(fwd_config.seed());

  *bwd_fmha_config.mutable_intermediate_tensor_shape() =
      fwd_config.intermediate_tensor_shape();
  {
    auto* algorithm = bwd_fmha_config.mutable_algorithm();
    algorithm->set_algo_id(0);  // engine id
    algorithm->set_math_type(se::dnn::AlgorithmProto::TENSOR_OP_MATH);
    std::vector<int64_t> knob_ids = /* {0, 1} */ {17, 24};
    std::vector<int64_t> knob_vals = {1, 0};
    for (int i = 0; i < knob_ids.size(); ++i) {
      (*algorithm->mutable_tuning_knobs())[knob_ids[i]] = knob_vals[i];
    }
    algorithm->set_is_cudnn_frontend(true);
    algorithm->mutable_workspace_size()->set_value(0);
  }

  // Output order:
  // dQ(bmm_1_grad_2), dK(bmm_1_grad_1), dV(bmm_2_grad_1),
  // d_intermediate_tensor, d_bias_tensor
  std::vector<Shape> output_shapes = {
      bmm_1_grad_2->shape(), bmm_1_grad_1->shape(), bmm_2_grad_1->shape()};
  if (d_intermediate) {
    output_shapes.push_back(lhs_bmm2_grad_gemm1->shape());
  } else {
    output_shapes.push_back(
        ShapeUtil::MakeShape(bmm_1_grad_1->shape().element_type(), {0}));
  }
  // Reserved placeholder for workspace
  output_shapes.push_back(ShapeUtil::MakeShape(U8, {0}));

  HloInstruction* dbias = nullptr;
  if (d_intermediate &&
      IsDbiasOnlyUserBesidesGradGemm(d_intermediate, bmm_1_grad_1, bmm_1_grad_2,
                                     &dbias)) {
    output_shapes.push_back(dbias->shape());
  } else {
    output_shapes.push_back(
        ShapeUtil::MakeShape(bmm_1_grad_1->shape().element_type(), {0}));
  }

  Shape call_shape = ShapeUtil::MakeTupleShape(output_shapes);
  HloInstruction* fmha_bwd_call =
      comp->AddInstruction(HloInstruction::CreateCustomCall(
          call_shape, operands, absl::string_view(bwd_custom_call_name)));
  TF_RETURN_IF_ERROR(fmha_bwd_call->set_backend_config(bwd_fmha_config));
  TF_RETURN_IF_ERROR(
      SetFMHAInstructionName(bmm_1_grad_1->GetModule(), fmha_bwd_call));

  // Q gradient
  TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
      bmm_1_grad_2, HloInstruction::CreateGetTupleElement(bmm_1_grad_2->shape(),
                                                          fmha_bwd_call, 0)));
  // K gradient
  TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
      bmm_1_grad_1, HloInstruction::CreateGetTupleElement(bmm_1_grad_1->shape(),
                                                          fmha_bwd_call, 1)));
  // V gradient
  TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
      bmm_2_grad_1, HloInstruction::CreateGetTupleElement(bmm_2_grad_1->shape(),
                                                          fmha_bwd_call, 2)));
  // d_intermediate tensor
  if (dbias) {
    // does not really need d_intermediate
    TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
        dbias, HloInstruction::CreateGetTupleElement(dbias->shape(),
                                                     fmha_bwd_call, 5)));
  }
  return true;
}
}  // namespace

StatusOr<bool> CudnnFusedMHARewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool any_changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    const DebugOptions& debug_options =
        comp->parent()->config().debug_options();
    if (!debug_options.xla_gpu_enable_cudnn_fmha() ||
        !IsComputeCapabilityAndCudnnSupported(
            compute_capability_, cudnn_version_, stream_executor_,
            stream_executor::dnn::VersionInfo(8, 8, 0))) {
      return false;
    }
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      HloInstruction* bmm_1;
      HloInstruction* bmm_2;
      // All of the below instructions are optional
      HloInstruction* bias = nullptr;
      HloInstruction* mask = nullptr;
      HloInstruction* scale = nullptr;
      double dropout_rate = 0.0f;
      std::string custom_call_name;
      bool need_canonicalization = false;
      bool v_transposed = false;
      bool is_training = false;
      if (!MatchFwdMHAPatternsForCanonicalization(
              instr, &bmm_1, &bmm_2, &bias, &mask, &scale, dropout_rate,
              custom_call_name, need_canonicalization, is_training)) {
        continue;
      }
      // We check the validity of bmms here before canonicalization so we don't
      // modify the graph if mha fusion is not possible
      TF_ASSIGN_OR_RETURN(
          bool is_mha_module_supported,
          IsMHABlockSupported(bmm_1, bmm_2, need_canonicalization, is_training,
                              custom_call_name, debug_options));
      if (!is_mha_module_supported) continue;
      // If we need to canonicalize the bmm, we will assign the newly
      // canonicalized bmm to bmm_2.
      if (need_canonicalization) {
        TF_ASSIGN_OR_RETURN(bmm_2,
                            CanonicalizeBatchedGemmForcuDNNFMHA(bmm_2, comp));
      }
      bool changed = false;
      // if fwd uses mask input, then bwd needs cudnn 8.9.1 to take in a mask
      // input if cudnn version < 8.9.1 we won't lower the bwd pass
      if (is_training && mask != nullptr &&
          !IsComputeCapabilityAndCudnnSupported(
              compute_capability_, cudnn_version_, stream_executor_,
              stream_executor::dnn::VersionInfo(8, 9, 1))) {
        continue;
      }

      // Fuse the bmms and intermediate nodes into fMHA call, the fused call
      // will replace bmm_2.
      TF_ASSIGN_OR_RETURN(
          HloInstruction * fwd_fmha_call,
          FuseFwdMultiHeadedAttentionBlock(
              comp, bmm_1, bmm_2, bias, mask, scale, dropout_rate,
              custom_call_name, compute_capability_, is_training, changed,
              v_transposed));
      any_changed |= changed;

      if (is_training) {
        // Continue to match for backward patterns
        HloInstruction* bmm_1_grad_1 = nullptr;
        HloInstruction* bmm_1_grad_2 = nullptr;

        HloInstruction* bmm_2_grad_1 = nullptr;
        HloInstruction* bmm_2_grad_2 = nullptr;
        HloInstruction* d_intermediate = nullptr;

        // We use this to keep track of all gradient bmms that need
        // canonicalization.
        std::vector<HloInstruction**> bmms_need_canonicalization;
        std::string bwd_custom_call_name;
        if (!MatchBwdMHAPatternsForCanonicalization(
                fwd_fmha_call, bmm_1, &bmm_1_grad_1, &bmm_1_grad_2,
                &bmm_2_grad_1, &bmm_2_grad_2, &d_intermediate, &mask,
                bwd_custom_call_name, v_transposed,
                bmms_need_canonicalization)) {
          continue;
        }
        bool is_bmm2_grad1_canonicalized = false;
        // check if dbias is the only user of d_intermediate besides
        // bmm_1_grad_1 and bmm_1_grad_2 and the cudnn version is > 8.9.1. We
        // won't lower bwd if this condition is not met as we won't deal with
        // unswizzling now
        HloInstruction* dbias = nullptr;
        if (d_intermediate &&
            !IsDbiasOnlyUserBesidesGradGemm(d_intermediate, bmm_1_grad_1,
                                            bmm_1_grad_2, &dbias) &&
            !IsComputeCapabilityAndCudnnSupported(
                compute_capability_, cudnn_version_, stream_executor_,
                stream_executor::dnn::VersionInfo(8, 9, 1))) {
          continue;
        }
        for (auto bmm : bmms_need_canonicalization) {
          is_bmm2_grad1_canonicalized |= (bmm == &bmm_2_grad_1);
          if ((*bmm)) {
            TF_ASSIGN_OR_RETURN(
                (*bmm), CanonicalizeBatchedGemmForcuDNNFMHA((*bmm), comp));
          }
        }
        // Fuse the corresponding gradient graph to an fMHA fused call.s
        TF_ASSIGN_OR_RETURN(
            changed,
            FuseBwdMultiHeadedAttentionBlock(
                comp, bmm_1_grad_1, bmm_1_grad_2, bmm_2_grad_1, bmm_2_grad_2,
                fwd_fmha_call, d_intermediate, mask, bwd_custom_call_name,
                need_canonicalization, is_bmm2_grad1_canonicalized));
        any_changed |= changed;
      }
    }
  }

  return any_changed;
}
}  // namespace gpu
}  // namespace xla
