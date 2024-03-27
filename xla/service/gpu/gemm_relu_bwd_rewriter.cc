/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/gpu/gemm_relu_bwd_rewriter.h"

#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/pattern_matcher.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace m = match;

class GemmReLUBwdVisitor : public DfsHloRewriteVisitor {
 public:
  auto CublasLtMatmulF8(HloInstruction **instr) {
    return m::CustomCall(instr, {kCublasLtMatmulF8CallTarget});
  }
  absl::Status ReluConvertDF8(HloInstruction *instr, HloInstruction *fwd_gemm,
                              HloInstruction *d_scale,
                              HloInstruction *clamp_lower,
                              HloInstruction *clamp_upper,
                              HloInstruction *maximum) {
    // Verify the data types and the operands of clamp.
    if (instr->shape().element_type() == F8E4M3FN) {
      if (!clamp_lower->literal().IsAllFloat(static_cast<float>(
              std::numeric_limits<tsl::float8_e4m3fn>::lowest())) ||
          !clamp_upper->literal().IsAllFloat(static_cast<float>(
              std::numeric_limits<tsl::float8_e4m3fn>::max()))) {
        return absl::OkStatus();
      }
    } else if (instr->shape().element_type() == F8E5M2) {
      if (!clamp_lower->literal().IsAllFloat(static_cast<float>(
              std::numeric_limits<tsl::float8_e5m2>::lowest())) ||
          !clamp_upper->literal().IsAllFloat(static_cast<float>(
              std::numeric_limits<tsl::float8_e5m2>::max()))) {
        return absl::OkStatus();
      }
    } else {
      return absl::OkStatus();
    }

    if (!ShapeUtil::IsScalar(d_scale->shape())) {
      return absl::OkStatus();
    }

    // The possible second user of the GEMM must be the calculation of the
    // maximum of the absolute value of the result of the GEMM. Since it is
    // unknown in what form this operation will be used, it is identified in a
    // top-down approach by inspecting the users of the GEMM.
    const std::vector<HloInstruction *> gemm_users = fwd_gemm->users();
    HloInstruction *reduce_damax = nullptr;
    HloInstruction *compare = nullptr;
    HloInstruction *select = nullptr;
    if (gemm_users.size() == 2) {
      // Assume the user of fwd gemm are maximum and compare, due to what
      // happens in gemm_rewriter. In the presence of a ReLU activation, the abs
      // instruction is elided since abs(ReLU(x)) = ReLU(x).
      TF_ASSIGN_OR_RETURN(auto gpu_config,
                          fwd_gemm->backend_config<GpuBackendConfig>());
      const GemmBackendConfig &config = gpu_config.gemm_backend_config();
      for (int i = 0; i < gemm_users.size(); ++i) {
        HloInstruction *maybe_reduce = nullptr;
        HloInstruction *maybe_compare = nullptr;
        HloInstruction *maybe_select = nullptr;
        if (gemm_users[i]->opcode() == HloOpcode::kMaximum) {
          // Assume the maximum has 2 users, reduce and divide
          if (gemm_users[i]->users().size() != 2) return absl::OkStatus();
          for (int j = 0; i < gemm_users[i]->users().size(); ++j) {
            if (gemm_users[i]->users()[j]->opcode() == HloOpcode::kReduce) {
              maybe_reduce = gemm_users[i]->users()[j];
              reduce_damax = maybe_reduce;
              break;
            }
          }
        } else if (gemm_users[i]->opcode() == HloOpcode::kCompare) {
          maybe_compare = gemm_users[i];
          maybe_select = gemm_users[i]->users()[0];
          select = maybe_select;
          compare = maybe_compare;
        }
      }

      if (!reduce_damax || !select || !compare) {
        return absl::OkStatus();
      }
    } else {
      return absl::OkStatus();
    }
    HloInstruction *bwd_gemm = nullptr;
    if (!Match(
            select,
            m::Select(m::Compare(m::CustomCall({kCublasLtMatmulF8CallTarget}),
                                 m::Broadcast(m::ConstantScalar(0)))
                          .WithOneUser(),
                      m::CustomCall(&bwd_gemm, {kCublasLtMatmulF8CallTarget})
                          .WithOneUser(),
                      m::Broadcast(m::ConstantScalar(0))))) {
      return absl::OkStatus();
    }

    // Step (a), replace maximum with RELU_AUX
    // Step (b), deal with select
    // Step (c), add dmax to output
    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        fwd_gemm->backend_config<GpuBackendConfig>());
    GemmBackendConfig &config = *gpu_config.mutable_gemm_backend_config();
    // Step (a), replace maximum with RELU_AUX
    if (config.epilogue() == GemmBackendConfig::DEFAULT) {
      config.set_epilogue(GemmBackendConfig::RELU_AUX);
    } else if (config.epilogue() == GemmBackendConfig::BIAS) {
      config.set_epilogue(GemmBackendConfig::BIAS_RELU_AUX);
    } else {
      return absl::OkStatus();
    }
    auto total_elements = [](const HloInstruction *gemm) {
      int64_t num_e = 1;
      for (int i = 0; i < gemm->shape().rank(); ++i) {
        num_e *= gemm->shape().dimensions(i);
      }
      return num_e;
    };
    Shape mask_shape =
        ShapeUtil::MakeShape(PrimitiveType::U8, {total_elements(fwd_gemm)});
    mask_shape.mutable_layout()->set_element_size_in_bits(1);
    std::unique_ptr<HloInstruction> output = fwd_gemm->CloneWithNewShape(
        ShapeUtil::MakeTupleShape({fwd_gemm->shape(), mask_shape}));
    TF_RETURN_IF_ERROR(output->set_backend_config(gpu_config));
    HloInstruction *tuple_output =
        fwd_gemm->parent()->AddInstruction(std::move(output));
    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
        maximum, HloInstruction::CreateGetTupleElement(tuple_output, 0)));

    HloInstruction *get_tuple1 = fwd_gemm->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(tuple_output, 1));
    // Step (b), deal with select to have DRELu in bwd_gemm
    std::vector<HloInstruction *> operands(bwd_gemm->operands().begin(),
                                           bwd_gemm->operands().end());
    operands.insert(operands.end(), get_tuple1);

    HloInstruction *new_bwd_custom_call =
        bwd_gemm->parent()->AddInstruction(HloInstruction::CreateCustomCall(
            ShapeUtil::MakeShapeWithDenseLayout(
                bwd_gemm->shape().element_type(),
                bwd_gemm->shape().dimensions(),
                bwd_gemm->shape().layout().minor_to_major()),
            operands, kCublasLtMatmulF8CallTarget));

    TF_ASSIGN_OR_RETURN(auto bwd_gpu_backend_config,
                        bwd_gemm->backend_config<GpuBackendConfig>());
    GemmBackendConfig &bwd_config =
        *bwd_gpu_backend_config.mutable_gemm_backend_config();
    bwd_config.set_epilogue(GemmBackendConfig::D_RELU);
    TF_RETURN_IF_ERROR(
        new_bwd_custom_call->set_backend_config(bwd_gpu_backend_config));

    TF_RETURN_IF_ERROR(ReplaceInstruction(select, new_bwd_custom_call));
    // Step (c) add dmax to output
    return RewriteFwdBwdGemm(tuple_output, new_bwd_custom_call, reduce_damax,
                             instr);  // instr is convert
  }

  absl::Status HandleReduce(HloInstruction *instr) override {
    HloInstruction *gemm = nullptr;
    if (Match(instr,
              m::Reduce(m::CustomCall(&gemm, {kCublasLtMatmulF8CallTarget}),
                        m::ConstantScalar(0)))) {
      TF_ASSIGN_OR_RETURN(auto gpu_config,
                          gemm->backend_config<GpuBackendConfig>());
      GemmBackendConfig &config = *gpu_config.mutable_gemm_backend_config();
      if (config.epilogue() != GemmBackendConfig::D_RELU) {
        return absl::OkStatus();
      }
      config.set_epilogue(GemmBackendConfig::D_RELU_BGRAD);
      std::unique_ptr<HloInstruction> output = gemm->CloneWithNewShape(
          ShapeUtil::MakeTupleShape({gemm->shape(), instr->shape()}));

      TF_RETURN_IF_ERROR(output->set_backend_config(gpu_config));

      HloInstruction *tuple_output =
          gemm->parent()->AddInstruction(std::move(output));
      TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
          gemm, HloInstruction::CreateGetTupleElement(tuple_output, 0)));
      output = HloInstruction::CreateGetTupleElement(tuple_output, 1);
      return ReplaceWithNewInstruction(instr, std::move(output));
    }
    return absl::OkStatus();
  }

  // Adds a scalar DAmax return value to an FP8 GEMM.
  absl::Status RewriteFwdBwdGemm(HloInstruction *fwd_gemm,
                                 HloInstruction *bwd_gemm,
                                 HloInstruction *fwd_reduce_damax,
                                 HloInstruction *next_fwd_convert) {
    // Assume reduce, bwd_gemm, and divide-clamp-convert for next fwd gemm are 3
    // users of fwd_gemm(via get-tuple-element)

    // Change the output shape of the fwd_gemm Custom Call to tuple(D, bitmask,
    // DAmax) from (D, bitmask).
    Shape damax_shape = ShapeUtil::MakeScalarShape(F32);

    Shape tuple_shape = ShapeUtil::MakeTupleShape(
        {next_fwd_convert->shape(), fwd_gemm->shape().tuple_shapes(1),
         damax_shape});
    HloInstruction *gemm_and_damax =
        fwd_gemm->AddInstruction(fwd_gemm->CloneWithNewShape(tuple_shape));
    // Obtain D and DAmax separately from the output tuple.
    HloInstruction *d =
        fwd_gemm->AddInstruction(HloInstruction::CreateGetTupleElement(
            next_fwd_convert->shape(), gemm_and_damax, 0));
    HloInstruction *bitmask =
        fwd_gemm->AddInstruction(HloInstruction::CreateGetTupleElement(
            fwd_gemm->shape().tuple_shapes(1), gemm_and_damax, 1));
    HloInstruction *damax = fwd_gemm->AddInstruction(
        HloInstruction::CreateGetTupleElement(damax_shape, gemm_and_damax, 2));
    // In case non-f32, there is bitcast and convert later.
    if (!ShapeUtil::SameElementType(fwd_reduce_damax->shape(),
                                    damax->shape())) {
      if (fwd_reduce_damax->users()[0]->opcode() == HloOpcode::kBitcast &&
          fwd_reduce_damax->users()[0]->users()[0]->opcode() ==
              HloOpcode::kConvert) {
        auto convert_to_f32 = fwd_reduce_damax->users()[0]->users()[0];
        auto bitcast = fwd_reduce_damax->AddInstruction(
            HloInstruction::CreateBitcast(convert_to_f32->shape(), damax));
        TF_RETURN_IF_ERROR(ReplaceInstruction(convert_to_f32, bitcast));
      } else {
        return absl::OkStatus();
      }
    } else {
      TF_RETURN_IF_ERROR(ReplaceInstruction(fwd_reduce_damax, damax));
    }
    TF_RETURN_IF_ERROR(ReplaceInstruction(next_fwd_convert, d));
    // Replace bwd_gemm's last operand
    std::vector<HloInstruction *> bwd_gemm_operands(
        bwd_gemm->operands().begin(), bwd_gemm->operands().end());
    bwd_gemm_operands.back() = bitmask;

    HloInstruction *new_bwd_gemm = bwd_gemm->AddInstruction(
        bwd_gemm->CloneWithNewOperands(bwd_gemm->shape(), bwd_gemm_operands));

    TF_RETURN_IF_ERROR(ReplaceInstruction(bwd_gemm, new_bwd_gemm));
    return absl::OkStatus();
  }

  absl::Status HandleConvert(HloInstruction *instr) override {
    HloInstruction *clamp_lower, *clamp_upper, *d_scale, *existing_gemm,
        *binary, *maximum;
    if (Match(
            instr,
            m::Convert(m::Clamp(
                m::Broadcast(m::ConstantScalar(&clamp_lower)),
                m::AnyOf<HloInstruction>(
                    m::Divide(
                        &binary,
                        m::Maximum(&maximum,
                                   m::CustomCall(&existing_gemm,
                                                 {kCublasLtMatmulF8CallTarget}),
                                   m::Broadcast(m::ConstantScalar(0))),
                        m::Broadcast(m::Op(&d_scale))),
                    m::MultiplyAnyOrder(
                        &binary,
                        m::Maximum(&maximum,
                                   m::CustomCall(&existing_gemm,
                                                 {kCublasLtMatmulF8CallTarget}),
                                   m::Broadcast(m::ConstantScalar(0))),
                        m::Broadcast(m::Op(&d_scale)))),
                m::Broadcast(m::ConstantScalar(&clamp_upper)))))) {
      return ReluConvertDF8(instr, existing_gemm, d_scale, clamp_lower,
                            clamp_upper, maximum);
    }
    return absl::OkStatus();
  }
};

static absl::StatusOr<bool> RunOnComputation(HloComputation *computation) {
  GemmReLUBwdVisitor visitor;
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

absl::StatusOr<bool> GemmReLUBwdRewriter::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  bool changed = false;
  for (HloComputation *computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
