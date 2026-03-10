/* Copyright 2023 The OpenXLA Authors.

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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#define EIGEN_USE_THREADS

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "Eigen/Core"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_common.hpp"
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#include "xla/backends/cpu/runtime/onednn/onednn_threadpool.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/literal.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_config.pb.h"
#include "xla/service/cpu/onednn_contraction_rewriter.h"
#include "xla/service/cpu/onednn_convolution.h"
#include "xla/service/cpu/onednn_matmul.h"
#include "xla/service/cpu/onednn_memory_util.h"
#include "xla/service/cpu/onednn_pattern_utils.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace xla {
namespace cpu {

namespace {

namespace m = match;
namespace pu = ::xla::cpu::onednn_pattern_utils_internal;

inline absl::Status ValidateDotDimensionNumbers(
    const DotDimensionNumbers& dim_numbers) {
  // Checks some invariants that do not hold in general, but DotDecomposer
  // should have established for us.
  TF_RET_CHECK(dim_numbers.lhs_contracting_dimensions().size() == 1);
  std::vector<int64_t> batch_dim_numbers(
      dim_numbers.lhs_batch_dimensions().size());
  absl::c_iota(batch_dim_numbers, 0);
  TF_RET_CHECK(
      absl::c_equal(batch_dim_numbers, dim_numbers.lhs_batch_dimensions()));
  TF_RET_CHECK(
      absl::c_equal(batch_dim_numbers, dim_numbers.rhs_batch_dimensions()));
  return absl::OkStatus();
}

// Whether the element type of instr is compatible with oneDNN kernels.
// TODO(intel-tf): Restrict compatible types based on instruction kind.
inline bool CompatibleElementType(const HloInstruction* instr) {
  PrimitiveType element_type = instr->shape().element_type();
  return element_type == BF16 || element_type == F32 || element_type == F16;
}

inline bool IsRowMajor(const Shape& shape) {
  return LayoutUtil::IsMonotonicWithDim0Major(shape.layout());
}

template <typename Pattern>
inline auto BitcastWithReshapeSemantics(HloInstruction** bitcast,
                                        Pattern pattern) {
  // TODO(intel-tf): Add stronger condition that Bitcast does not have transpose
  // semantics. Some of the HLO passes replaces Transpose with Bitcast. Here
  // the layouts are checked to be rowmajor since the current pass runs after
  // the layout assignment and oneDNN matmul is enabled for rowmajor layouts.
  auto is_reshape = [](const HloInstruction* instr) -> bool {
    if (!instr) {
      return false;
    }
    auto input_shape = instr->operand(0)->shape();
    auto output_shape = instr->shape();
    bool is_same_type = ShapeUtil::SameElementType(input_shape, output_shape);
    bool has_equal_num_elems = ShapeUtil::ElementsIn(input_shape) ==
                               ShapeUtil::ElementsIn(output_shape);
    bool has_rowmajor_layout =
        IsRowMajor(input_shape) && IsRowMajor(output_shape);
    return is_same_type && has_equal_num_elems && has_rowmajor_layout;
  };
  return m::Bitcast(bitcast, pattern).WithPredicate(is_reshape);
}

template <typename Pattern>
auto ElementwiseSafeIntermediates(HloInstruction** instr,
                                  HloInstruction** optional_bitcast,
                                  Pattern pattern) {
  return m::AnyOf<HloInstruction>(
      m::Broadcast(instr, pattern.WithOneUser()),
      m::Slice(instr, pattern.WithOneUser()),
      m::Bitcast(instr, pattern.WithOneUser()),
      m::Reshape(instr, pattern.WithOneUser()),
      pu::SupportedConvert(instr, pattern.WithOneUser()),
      pu::SupportedConvert(instr, BitcastWithReshapeSemantics(
                                      optional_bitcast, pattern.WithOneUser())),
      pattern);
}

inline auto OneDnnMatmulInstr(HloInstruction** instr) {
  return m::CustomCall(instr, {"__onednn$matmul"});
}

inline auto OneDnnConvolutionInstr(HloInstruction** instr) {
  return m::CustomCall(instr, {"__onednn$convolution"});
}

inline auto OneDnnFusibleInstr(HloInstruction** instr) {
  return m::AnyOf<HloInstruction>(
      m::CustomCall(instr, {"__onednn$matmul"}),
      m::CustomCall(instr, {"__onednn$convolution"}));
}

inline bool IsOneDnnMatmulInstr(const HloInstruction* instr) {
  return Match(instr, m::CustomCall({"__onednn$matmul"}));
}

inline bool IsOneDnnConvolutionInstr(const HloInstruction* instr) {
  return Match(instr, m::CustomCall({"__onednn$convolution"}));
}

inline auto ConvertBF16ToF32(HloInstruction** instr) {
  return m::Convert(m::Op(instr).WithElementType(PrimitiveType::BF16))
      .WithElementType(PrimitiveType::F32);
}

inline auto BcastConstScalar(HloInstruction** instr, double value) {
  return m::Broadcast(instr, m::ConstantScalar(value));
}

inline auto BcastConstScalar(double value) {
  return BcastConstScalar(nullptr, value);
}

inline auto BcastConvertConstScalar(double value) {
  return m::Broadcast(pu::OptionalConvert(m::ConstantScalar(value)));
}

inline bool IsBatchDot(const HloInstruction& instr) {
  if (auto* dot_instr = DynCast<HloDotInstruction>(&instr)) {
    return !dot_instr->dot_dimension_numbers().lhs_batch_dimensions().empty();
  }
  return false;
}

auto ConstScalarNear(double value) {
  return m::ConstantScalar().WithPredicate(
      [expected = value](const HloInstruction* instr) {
        // Not a very robust floating-point comparison, but good enough for our
        // purposes.
        std::optional<double> actual =
            static_cast<const HloConstantInstruction*>(instr)
                ->literal()
                .GetAsDouble({});
        if (!actual.has_value()) {
          return false;
        }
        double epsilon;
        switch (instr->shape().element_type()) {
          case F16:
            epsilon = 128 * std::numeric_limits<Eigen::half>::epsilon();
            break;
          case BF16:
            epsilon = 128 * std::numeric_limits<bfloat16>::epsilon();
            break;
          case F32:
            epsilon = 128 * std::numeric_limits<float>::epsilon();
            break;
          case F64:
            epsilon = 128 * std::numeric_limits<double>::epsilon();
            break;
          default:
            return false;
        }
        return abs(*actual - expected) < (abs(*actual + expected) * epsilon);
      });
}

bool IsScalar(const HloInstruction* instr) {
  return ShapeUtil::IsEffectiveScalar(instr->shape());
}

bool IsVector(const HloInstruction* instr) {
  return instr->shape().dimensions_size() == 1;
}

std::optional<float> GetScalarConstantValue(const HloInstruction* inst) {
  if (!IsScalar(inst)) {
    return std::nullopt;
  }
  switch (inst->shape().element_type()) {
    case F16:
      return inst->literal().GetFirstElement<half>();
    case BF16:
      return inst->literal().GetFirstElement<bfloat16>();
    case F32:
      return inst->literal().GetFirstElement<float>();
    case S32:
      return inst->literal().GetFirstElement<int32_t>();
    default:
      return std::nullopt;
  }
}

auto GetOneDnnContractionVariant(
    absl::StatusOr<BackendConfig>* backend_config) {
  return ((*backend_config)->backend_config_oneof_case() == kOnednnConvConfig)
             ? OneDnnContractionVariant(PrimitiveTrait<kOnednnConvConfig>{})
             : OneDnnContractionVariant(PrimitiveTrait<kOnednnMatmulConfig>{});
}

// Return the correct mutable config instance for the given contraction variant
// based on the template parameter
template <typename TransformationType>
TransformationType GetTransformationConfig(
    absl::StatusOr<BackendConfig>* backend_config) {
  return std::visit(
      [&](auto&& config) -> TransformationType {
        using T = std::decay_t<decltype(config)>;
        return PrimitiveTrait<T::kConfigVal, TransformationType>::
            GetTransformationConfig(
                GetKernelConfig<T::kConfigVal>(backend_config));
      },
      GetOneDnnContractionVariant(backend_config));
}

auto GetFusionsConfig(absl::StatusOr<BackendConfig>* backend_config) {
  return GetTransformationConfig<OneDnnFusionConfig*>(backend_config);
}

auto GetOptimizationsConfig(absl::StatusOr<BackendConfig>* backend_config) {
  return GetTransformationConfig<OneDnnOptimizationConfig*>(backend_config);
}

inline auto BcastConstScalarNear(double value) {
  return m::Broadcast(ConstScalarNear(value));
}

// Associativity and commutativity properties of multiply results in various
// patterns for an equivalent computation. This function tries to capture most
// of the variations for a computation a * b * c. For example, patterns could be
// any of (a * b) * c or a * (b * c), along with the variations resulting from
// commutative patterns.
template <typename PatternA, typename PatternB, typename PatternC>
inline auto MultiplyMultiplyAnyOrder(PatternA a, PatternB b, PatternC c) {
  return m::AnyOf<HloInstruction>(
      m::MultiplyAnyOrder(a, m::MultiplyAnyOrder(b, c)),
      m::MultiplyAnyOrder(b, m::MultiplyAnyOrder(a, c)),
      m::MultiplyAnyOrder(c, m::MultiplyAnyOrder(a, b)));
}

auto GELUActivation(HloInstruction* instr, HloInstruction** src) {
  // Attempt to match GELU_TANH activation or GELU_ERF activation
  // (https://arxiv.org/abs/1606.08415), where:
  // gelu_tanh(x) = x * cdf(x)
  // cdf(x) = 0.5 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x**3))
  //                     -------------errf_approximate------------
  //
  // gelu_erf(x) = x * cdf(x)
  // cdf(x) = 0.5 * (1 + erf(x / sqrt(2)))
  //                     --errf_exact--

  HloInstruction* errf;

  // The expression 0.5 * x * (1 + errf) as common pattern for GELU exact and
  // approximate activations.
  auto common_pattern = MultiplyMultiplyAnyOrder(
      BcastConstScalar(0.5), m::Op(src),
      m::AddAnyOrder(BcastConstScalar(1.0), m::Op(&errf).WithOneUser()));

  bool matched = Match(instr, common_pattern);
  if (matched) {
    // The subexpression 0.044715 * x**3 appears in GELU approximate activation.
    // However, it is often optimized by other HLO passes into an expression of
    // 0.044715 * x * (x * x). Since there are three consecutive multiplies,
    // there could be a large number of patterns. We try to capture some of
    // those:
    //
    //      1. (0.044715 * x) * x * x
    //      2. 0.044715 * (x * x) * x
    //
    // Note each of the above could in turn have various patterns due to
    // associativity and commutativity properties of multiply.
    auto subexpr_pattern = m::AnyOf<HloInstruction>(
        MultiplyMultiplyAnyOrder(
            m::MultiplyAnyOrder(BcastConstScalarNear(0.044715),
                                m::Op().Is(*src)),
            m::Op().Is(*src), m::Op().Is(*src)),
        MultiplyMultiplyAnyOrder(
            BcastConstScalarNear(0.044715),
            m::Multiply(m::Op().Is(*src), m::Op().Is(*src)), m::Op().Is(*src)));

    auto errf_apprx_pattern =
        m::Tanh(m::MultiplyAnyOrder(
                    BcastConstScalarNear(sqrt(M_2_PI)),
                    m::AddAnyOrder(m::Op().Is(*src), subexpr_pattern)
                        .WithOneUser()))
            .WithOneUser();

    HloInstruction* erf;
    auto errf_exact_pattern =
        m::Op(&erf)
            .WithOpcode(HloOpcode::kErf)
            .WithOperand(
                0, m::MultiplyAnyOrder(m::Op(src),
                                       m::AnyOf<HloInstruction>(
                                           BcastConstScalarNear(0.707106769),
                                           BcastConstScalarNear(0.70703125),
                                           BcastConstScalarNear(0.707182348)))
                       .WithOneUser())
            .WithOneUser();

    if (Match(errf, errf_apprx_pattern)) {  // Gelu-approximate pattern.
      return OneDnnFusionConfig::GELU_TANH;
    }
    if (Match(errf, errf_exact_pattern)) {  // Gelu-exact pattern.
      return OneDnnFusionConfig::GELU_ERF;
    }
  }
  return OneDnnFusionConfig::UNDEFINED;
}

// OneDNN matmul and convolution can fuse add operation with automatic
// broadcasting along the addend's dimensions that are 1s. When compatible,
// Broadcast can be replaced by Bitcast, which is much cheaper. Compute new
// shape for the Bitcast.
absl::StatusOr<Shape> AdjustAddendShape(const HloInstruction* contraction,
                                        const HloInstruction* addend,
                                        const HloInstruction* broadcast_instr) {
  if (!broadcast_instr) {
    // TODO(intel-tf): Modify this condition when Contraction + Bias +
    // Add is enabled.
    if (IsOneDnnConvolutionInstr(contraction) &&
        ShapeUtil::TrueNumDimensions(addend->shape()) == 1 &&
        addend->shape().dimensions().size() != 1) {
      return ShapeUtil::FilterDimensions(
          [&addend](int64_t dim) {
            return ShapeUtil::GetDimension(addend->shape(), dim) != 1;
          },
          addend->shape());
    }
    return addend->shape();
  }
  if (broadcast_instr->opcode() != HloOpcode::kBroadcast) {
    return absl::InvalidArgumentError(
        "Hlo instruction is not a Broadcast insruction.");
  }
  auto bcast = Cast<HloBroadcastInstruction>(broadcast_instr);
  Shape new_shape = bcast->shape();
  // Broadcast instruction has "dimensions" parameter along which its input's
  // dimensions should not change. For example,
  //      dot = f32[3,4,5,6] dot(...)
  //      arg = f32[3,6]{1,0} parameter(0)
  //      broad = f32[3,4,5,6]{3,2,1,0} broadcast(arg), dimensions={0,3}
  //      add = f32[3,4,5,6]{3,2,1,0} add(dot, arg)
  // can be replaced with the following
  //      arg = f32[3,6]{1,0} parameter(0)
  //      bitcast = f32[3,1,1,6]{3,2,1,0} bitcast(arg)
  //      fused = f32[3,4,5,6]{3,2,1,0} custom-call((..., bitcast)
  auto kept_dimensions = bcast->dimensions();
  for (int i = 0; i < new_shape.dimensions().size(); i++) {
    if (!absl::c_linear_search(kept_dimensions, i)) {
      new_shape.set_dimensions(i, 1);
    }
  }

  // If rank(new_shape) > rank(instr), extra dimensions with value = 1 can be
  // deleted from the new_shape.
  auto instr_shape = contraction->shape();
  int64_t rank_difference = 0;
  // Since the absl APIs return unsigned dimension sizes, compute the rank
  // difference only when LHS > RHS. Performing an unconditional subtraction
  // could cause an unsigned underflow.
  if (new_shape.dimensions().size() > instr_shape.dimensions().size()) {
    rank_difference =
        new_shape.dimensions().size() - instr_shape.dimensions().size();
  }
  auto new_dims = new_shape.dimensions();
  std::vector<int64_t> dims_to_delete;
  for (int i = 0; i < rank_difference; ++i) {
    if (new_dims[i] == 1) {
      dims_to_delete.push_back(i);
    }
  }
  new_shape = ShapeUtil::DeleteDimensions(dims_to_delete, new_shape);

  // New shape for bias should satisfy the condition:
  //   rank(new_shape) <= rank(instr).
  if (new_shape.dimensions().size() > instr_shape.dimensions().size()) {
    return absl::CancelledError(
        "Bias shape could not be adjusted for a fusion.");
  }

  return new_shape;
};

// Compute new shape for the binary operand when dot's outer dims
// are flattened/unflattened with respect to the binary operand dims.
// Adjusting the operand shape to the dot's shape enables fusion in oneDNN.
absl::StatusOr<Shape> AdjustBinaryOperandShape(
    const HloInstruction* operand_instr, const Shape& dot_shape) {
  if (ShapeUtil::ElementsIn(operand_instr->shape()) !=
      ShapeUtil::ElementsIn(dot_shape)) {
    return absl::CancelledError(
        "Number of elements in operand and dot instruction do not match.");
  }
  return dot_shape;
};

inline bool IsOperandFusible(HloInstruction* operand, HloInstruction* instr) {
  // Check if the operand's shape is compatible for fusion.
  // An operand is fusible if
  //    1. rank(operand) <= rank(instr) and
  //    2. Starting from the last dim in backward direction, the dimension
  //       size of operand is either 1 or same to dot.
  auto operand_dims = operand->shape().dimensions();
  auto instr_dims = instr->shape().dimensions();
  if (operand_dims.size() > instr_dims.size()) {
    return false;
  }
  auto operand_idx = static_cast<int>(operand_dims.size()) - 1;
  auto instr_idx = static_cast<int>(instr_dims.size()) - 1;
  for (; operand_idx >= 0; --operand_idx, --instr_idx) {
    if (operand_dims[operand_idx] != 1 &&
        operand_dims[operand_idx] != instr_dims[instr_idx]) {
      return false;
    }
  }
  return true;
}

template <typename Pattern>
inline auto OptionalConvertAndBitcast(HloInstruction** optional_convert,
                                      HloInstruction** optional_bitcast,
                                      Pattern pattern) {
  // Checks the presence of some intermediate operations that can be moved /
  // folded to allow dot fusion with add.
  // Try to match either of the following:
  //   1. pattern-root -> bf16/f16-to-fp32 convert -> bitcast
  //   2. pattern-root -> bf16/f16-to-fp32 convert
  //   3. pattern-root -> bitcast
  //   4. pattern-root
  auto common = m::AnyOf<HloInstruction>(
      pu::SupportedConvert(optional_convert, std::move(pattern).WithOneUser())
          .WithElementType(PrimitiveType::F32),
      std::move(pattern).WithOneUser());
  return m::AnyOf<HloInstruction>(
      BitcastWithReshapeSemantics(optional_bitcast, common), common);
}

}  // namespace

bool OneDnnContractionRewriter::ShouldRewriteDot(
    const HloInstruction* dot_instr, bool before_layout_assignment) {
  if (dot_instr->opcode() != HloOpcode::kDot) {
    return false;
  }
  // Blocking control dependencies.
  if (dot_instr->HasControlDependencies() ||
      !IsSupportedType(dot_instr->shape().element_type()) ||
      dot_instr->operands().size() != 2) {
    return false;
  }

  // We rewrite when the data type is F32 or BF16. We do not need to check
  // equality of contraction dim-size of the operands. HLO verifier already does
  // the job. We, however, need to check if contraction is over only 1 dimension
  // (i.e., K dimension in matrix-multiplication parlance). We also restrict
  // that batch dimensions of the operands match.
  const Shape& lhs_shape = dot_instr->operand(0)->shape();
  const Shape& rhs_shape = dot_instr->operand(1)->shape();
  const Shape& output_shape = dot_instr->shape();

  // None of the operands and result should be ZeroElementArray.
  if (ShapeUtil::IsZeroElementArray(lhs_shape) ||
      ShapeUtil::IsZeroElementArray(rhs_shape) ||
      ShapeUtil::IsZeroElementArray(output_shape)) {
    return false;
  }

  // OneDNN only supports rank <= kOneDnnMaxNDims and singular non-contracting
  // dimensions. We should not rewrite if any of these conditions are violated.
  if (lhs_shape.dimensions().empty() ||
      lhs_shape.dimensions().size() > kOneDnnMaxNDims ||
      rhs_shape.dimensions().empty() ||
      rhs_shape.dimensions().size() > kOneDnnMaxNDims ||
      output_shape.dimensions().size() >
          std::min<uint64_t>({lhs_shape.dimensions().size(),
                              rhs_shape.dimensions().size(),
                              kOneDnnMaxNDims})) {
    return false;
  }

  // Layout should be row-major, contraction dimensions capture transpose
  // scenarios in the last two dimensions. Col-major layouts are corrected to
  // row-major for BatchDot operation as part of the layout-assignment pass.
  // Skip row-major layout check before layout-assignment pass.
  if (!before_layout_assignment) {
    bool row_major = IsRowMajor(lhs_shape) && IsRowMajor(rhs_shape) &&
                     IsRowMajor(output_shape);
    if (!row_major) {
      return false;
    }
  }

  auto dot_dim_numbers = dot_instr->dot_dimension_numbers();
  int64_t lhs_dim_k = dot_dim_numbers.lhs_contracting_dimensions(0);
  int64_t rhs_dim_k = dot_dim_numbers.rhs_contracting_dimensions(0);

  // Supported contraction is only in one of last two dimensions.
  if (lhs_dim_k + 2 < lhs_shape.dimensions().size() ||
      rhs_dim_k + 2 < rhs_shape.dimensions().size()) {
    return false;
  }

  // OneDNN matmul has scratch allocation and copy overheads. The overheads
  // can be amortized if there is sufficient number of flops. We don't rewrite
  // for small cases (determined empirically).
  // TODO(intel-tf): Relax the condition when more optimizations in oneDNN
  // matmul is achieved.
  auto num_flops = xla::HloCostAnalysis::GetDotFlops(lhs_shape, output_shape,
                                                     dot_dim_numbers);
  auto rank = output_shape.dimensions().size();
  auto flops_threshold = (rank <= 2) ? (1 << 24) : (1 << 19);
  return (num_flops >= flops_threshold);
}

bool OneDnnContractionRewriter::ShouldRewriteConv(
    const HloInstruction* conv_instr) {
  if (conv_instr->opcode() != HloOpcode::kConvolution ||
      conv_instr->HasControlDependencies() ||
      !IsSupportedType(conv_instr->shape().element_type()) ||
      conv_instr->batch_group_count() != 1) {
    return false;
  }

  // TODO(intel-tf): Remove this restriction after enabling backward weights
  // support
  if (conv_instr->operand(1)->opcode() == HloOpcode::kReverse) {
    return false;
  }

  const Shape& inp_shape = conv_instr->operand(0)->shape();
  const Shape& ker_shape = conv_instr->operand(1)->shape();
  const Shape& out_shape = conv_instr->shape();
  if (ShapeUtil::IsZeroElementArray(inp_shape) ||
      ShapeUtil::IsZeroElementArray(ker_shape) ||
      ShapeUtil::IsZeroElementArray(out_shape)) {
    return false;
  }

  auto dims = conv_instr->window().dimensions().size();
  if (dims >= 4 || dims <= 0) {
    return false;
  }

  if (inp_shape.dimensions().size() != ker_shape.dimensions().size() ||
      inp_shape.dimensions().size() != out_shape.dimensions().size()) {
    return false;
  }

  return true;
}

inline bool S32ToF32(const HloInstruction* instr) {
  return instr->shape().element_type() == F32 &&
         instr->operand(0)->shape().element_type() == S32;
}

inline bool Int8ToS32(const HloInstruction* instr) {
  auto input_type = instr->operand(0)->shape().element_type();
  return instr->shape().element_type() == S32 &&
         (input_type == S8 || input_type == U8);
}

inline bool F32ToInt8(const HloInstruction* instr) {
  auto output_type = instr->shape().element_type();
  return instr->operand(0)->shape().element_type() == F32 &&
         (output_type == S8 || output_type == U8);
}

inline auto Int8ToS32Pattern(HloInstruction** quant_input) {
  return m::Convert(m::Op(quant_input)).WithPredicate(Int8ToS32);
}

inline auto AddZPPattern(HloInstruction** quant_input, HloInstruction** zp) {
  return m::AddAnyOrder(Int8ToS32Pattern(quant_input),
                        m::Broadcast(m::Constant(zp)));
}

inline auto S32ToF32WithOptionalAddZP(HloInstruction** quant_input,
                                      HloInstruction** zp) {
  return m::Convert(m::AnyOf<HloInstruction>(AddZPPattern(quant_input, zp),
                                             Int8ToS32Pattern(quant_input)))
      .WithPredicate(S32ToF32);
}

template <typename Pattern>
auto OptionalCopyAndBitcastPattern(Pattern pattern, HloInstruction** copy,
                                   HloInstruction** bitcast) {
  return m::AnyOf<HloInstruction>(m::Copy(copy, m::Bitcast(bitcast, pattern)),
                                  m::Bitcast(bitcast, pattern), pattern);
}

// The Dequantize pattern (the Bitcast -> Copy nodes are not part of the
// original Dequantize pattern but are added by the layout assignment pass):
// Convert(INT8->S32)   Broadcast(Negated-ZP-Constant)
//                   | |
//                   Add
//                    |
//             Convert(S32->F32)   Broadcast(Scale-Constant)
//                            |    |
//                           Multiply
//                              |
//                           Bitcast
//                              |
//                            Copy
auto DequantizePattern(HloInstruction** quant_input, HloInstruction** scale,
                       HloInstruction** zp, HloInstruction** copy,
                       HloInstruction** bitcast) {
  auto deq_pattern = m::AnyOf<HloInstruction>(
      m::MultiplyAnyOrder(S32ToF32WithOptionalAddZP(quant_input, zp),
                          m::Broadcast(m::Constant(scale))),
      S32ToF32WithOptionalAddZP(quant_input, zp));
  // Layout assignment pass may insert Transpose | Bitcast -> Copy pattern.
  // For now we only handle Bitcast -> Copy assuming Transpose was replaced with
  // Bitcast by AlgebraicSimplifier.
  return OptionalCopyAndBitcastPattern(deq_pattern, copy, bitcast);
}

auto OptionalAddZPAndMultiplyScale(HloInstruction** scale, HloInstruction** zp,
                                   HloInstruction** input_custom_call) {
  auto multiply_scale = m::MultiplyAnyOrder(m::Op(input_custom_call),
                                            m::Broadcast(m::Constant(scale)));
  auto add_zp = m::AddAnyOrder(
      m::AnyOf<HloInstruction>(multiply_scale, m::Op(input_custom_call)),
      m::Broadcast(m::Constant(zp)));
  return m::AnyOf<HloInstruction>(add_zp, multiply_scale,
                                  m::Op(input_custom_call));
}

// The Quantize pattern
//                         Input Broadcast(Inversed-Scale-Constant)
//                           |    |
//                          Multiply Broadcast(ZP-Constant)
//                               \   /
// Broadcast(Clamp-Min-Constant)  Add   Broadcast(Clamp-Max-Constant)
//                             \   |   /
//                               Clamp
//                                 |
//                         RoundNearestEven
//                                 |
//                         Convert(F32->INT8)
auto QuantizePattern(HloInstruction** scale, HloInstruction** zp,
                     HloInstruction** input_custom_call,
                     HloInstruction** clamp_min, HloInstruction** clamp_max) {
  auto quant_pattern =
      m::Convert(
          m::Op()
              .WithOpcode(HloOpcode::kRoundNearestEven)
              .WithOperand(0, m::Clamp(m::Broadcast(m::Constant(clamp_min)),
                                       OptionalAddZPAndMultiplyScale(
                                           scale, zp, input_custom_call),
                                       m::Broadcast(m::Constant(clamp_max)))))
          .WithPredicate(F32ToInt8);
  return quant_pattern;
}

void PrepareQuantizationInstrs(
    HloInstruction* contraction_instr, HloInstruction* quant_src,
    HloInstruction* quant_wei, HloInstruction* copy_src,
    HloInstruction* copy_wei, HloInstruction* bitcast_src,
    HloInstruction* bitcast_wei, HloInstruction** src_scale,
    HloInstruction** src_zp, HloInstruction** wei_scale,
    HloInstruction** wei_zp, HloInstruction** new_copy_src,
    HloInstruction** new_copy_wei) {
  auto create_bitcast_copy = [&](HloInstruction* quant_operand,
                                 HloInstruction* bitcast, HloInstruction* copy,
                                 HloInstruction*& new_bitcast,
                                 HloInstruction*& new_copy) {
    auto quant_type = quant_operand->shape().element_type();
    if (bitcast) {
      new_bitcast =
          contraction_instr->AddInstruction(HloInstruction::CreateBitcast(
              ShapeUtil::ChangeElementType(bitcast->shape(), quant_type),
              quant_operand));
      if (copy) {
        new_copy =
            contraction_instr->AddInstruction(HloInstruction::CreateUnary(
                ShapeUtil::ChangeElementType(copy->shape(), quant_type),
                HloOpcode::kCopy, new_bitcast));
      } else {
        // There is no copy after bitcast.
        new_copy = new_bitcast;
      }
    } else {
      new_copy = contraction_instr->AddInstruction(HloInstruction::CreateUnary(
          ShapeUtil::ChangeElementType(copy->shape(), quant_type),
          HloOpcode::kCopy, quant_operand));
    }
  };

  HloInstruction *new_bitcast_src = nullptr, *new_bitcast_wei = nullptr;
  // We need to add bitcast and copy instructions that were in the original
  // pattern after each operand. It is possible that bitcast is not followed
  // by copy, we will handle that inside create_bitcast_copy also.
  if (copy_src != nullptr || bitcast_src != nullptr) {
    create_bitcast_copy(quant_src, bitcast_src, copy_src, new_bitcast_src,
                        *new_copy_src);
  } else {
    *new_copy_src = quant_src;
  }
  if (copy_wei != nullptr || bitcast_wei != nullptr) {
    create_bitcast_copy(quant_wei, bitcast_wei, copy_wei, new_bitcast_wei,
                        *new_copy_wei);
  } else {
    *new_copy_wei = quant_wei;
  }

  const float scale = 1.0;
  const int zp = 0;
  auto set_default_value = [&]<typename T>(HloInstruction*& instr, T value) {
    instr = contraction_instr->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<T>(value)));
  };
  // If scale and/or zp are not found. That means they had a default values
  // that were optimized out. Hence, we set a default value of 1.0 and 0 for
  // scale and zp respectively.
  if (*src_scale == nullptr) {
    set_default_value(*src_scale, scale);
  }
  if (*src_zp == nullptr) {
    set_default_value(*src_zp, zp);
  }
  if (*wei_scale == nullptr) {
    set_default_value(*wei_scale, scale);
  }
  if (*wei_zp == nullptr) {
    set_default_value(*wei_zp, zp);
  }
}

// Check if the Quantize and Dequantize pairs have the same scale and zero point
// values.
bool ValidateQDQParams(HloInstruction* dequant_input,
                       HloInstruction* dequant_scale,
                       HloInstruction* dequant_zp) {
  HloInstruction *quant_zp = nullptr, *quant_scale = nullptr, *input = nullptr,
                 *clamp_min = nullptr, *clamp_max = nullptr;

  bool quant_pattern = Match(
      dequant_input,
      QuantizePattern(&quant_scale, &quant_zp, &input, &clamp_min, &clamp_max));

  // We expect scale/zp to match between Quantize and Dequantize ops. We
  // validate that only if Quantize op exists, otherwise fusing only Dequantize
  // op is functionally correct although not a typical use case.
  if (quant_pattern) {
    // If scale is missing, it may be optimized out. Assume default value of
    // 1.0.
    auto quant_scale_value =
        quant_scale ? GetScalarConstantValue(quant_scale) : 1.0;
    auto dequant_scale_value =
        dequant_scale ? GetScalarConstantValue(dequant_scale) : 1.0;

    if (quant_scale_value && dequant_scale_value) {
      // Assume that scale for quantize has been inverted by some earlier pass.
      float quant_scale_native_value = 1.0 / quant_scale_value.value();
      float dequant_scale_native_value = dequant_scale_value.value();
      if (quant_scale_native_value != dequant_scale_native_value) {
        return false;
      }
    } else {
      return false;
    }

    // If zp is missing, it may be optimized out. Assume default value of 0.
    auto quant_zp_value = quant_zp ? GetScalarConstantValue(quant_zp) : 0;
    auto dequant_zp_value = dequant_zp ? GetScalarConstantValue(dequant_zp) : 0;
    if (quant_zp_value && dequant_zp_value) {
      int32_t quant_zp_native_value = quant_zp_value.value();
      // Assume that zp for dequantize has been negated by some earlier pass.
      int32_t dequant_zp_native_value = -dequant_zp_value.value();
      if (quant_zp_native_value != dequant_zp_native_value) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

bool ValidateQuantizationParams(HloInstruction* quant_src,
                                HloInstruction* quant_wei,
                                HloInstruction* src_scale,
                                HloInstruction* src_zp,
                                HloInstruction* wei_scale,
                                HloInstruction* wei_zp) {
  if (!ValidateQDQParams(quant_src, src_scale, src_zp) ||
      !ValidateQDQParams(quant_wei, wei_scale, wei_zp)) {
    return false;
  }

  if (!src_scale || !src_zp || !wei_zp) return false;
  // oneDNN only supports scalar scale and zero-point for src and dst.
  if (!IsScalar(src_scale) || !IsScalar(src_zp)) return false;

  // Weights scale can only be scalar or vector
  if (!IsScalar(wei_scale) && !IsVector(wei_scale)) return false;

  // oneDNN only supports zero-point=0 for weights.
  auto wei_zp_value = GetScalarConstantValue(wei_zp);
  if (!wei_zp_value || wei_zp_value.value() != 0) return false;

  return true;
}

class OneDnnContractionRewriteVisitor : public DfsHloRewriteVisitor {
 public:
  OneDnnContractionRewriteVisitor(bool graph_enabled, int intra_op_parallelism,
                           const tsl::thread::ThreadPool* compile_threadpool)
      : graph_enabled_(graph_enabled), intra_op_parallelism_(intra_op_parallelism > 0
                                  ? intra_op_parallelism
                                  : tsl::port::MaxParallelism()),
        evaluator_(/*max_loop_iterations=*/0) {
    if (compile_threadpool) {
      threadpool_device_.reset(
          new Eigen::ThreadPoolDevice(compile_threadpool->AsEigenThreadPool(),
                                      compile_threadpool->NumThreads()));
    } else {
      threadpool_handle_.reset(new tsl::thread::ThreadPool(
          tsl::Env::Default(), "XLACpuCompile", tsl::port::MaxParallelism()));
      threadpool_device_.reset(
          new Eigen::ThreadPoolDevice(threadpool_handle_->AsEigenThreadPool(),
                                      threadpool_handle_->NumThreads()));
    }
    // Set oneDNN concurrency settings (which is thread-local)
    OneDnnThreadPool::set_onednn_max_threads(intra_op_parallelism_);
  }
  // Matches patterns for possible MatMul fusions that are supported by oneDNN
  // library. Matched HLO instruction(s) are replaced by custom call.
  absl::Status HandleDot(HloInstruction* instr) override {
    // When oneDNN graph is enabled, dot will be handled via DotLibraryRewriter
    if (graph_enabled_) return absl::OkStatus();

    HloInstruction* dot_instr;
    auto pattern = m::Op(&dot_instr).WithOpcode(HloOpcode::kDot);
    if (!Match(instr, pattern)) {
      return absl::OkStatus();
    }

    TF_RETURN_IF_ERROR(
        ValidateDotDimensionNumbers(dot_instr->dot_dimension_numbers()));
    if (!OneDnnContractionRewriter::ShouldRewriteDot(dot_instr)) {
      TF_RETURN_IF_ERROR(UpcastDotToF32(dot_instr));
      return absl::OkStatus();
    }
    TF_ASSIGN_OR_RETURN(dot_instr, ReconfigureDotDimensions(dot_instr));
    auto dot_dim_numbers = dot_instr->dot_dimension_numbers();
    const Shape& lhs_shape = dot_instr->operand(0)->shape();
    const Shape& rhs_shape = dot_instr->operand(1)->shape();
    const Shape& output_shape = dot_instr->shape();

    int64_t lhs_dim_k = dot_dim_numbers.lhs_contracting_dimensions(0);
    int64_t rhs_dim_k = dot_dim_numbers.rhs_contracting_dimensions(0);

    HloInstruction *quant_src = nullptr, *src_scale = nullptr,
                      *src_zp = nullptr, *quant_wei = nullptr,
                      *wei_scale = nullptr, *wei_zp = nullptr, *copy_src = nullptr,
                      *bitcast_src = nullptr, *copy_wei = nullptr,
                      *bitcast_wei = nullptr, *matmul_call = nullptr,
                      *new_copy_src = nullptr, *new_copy_wei = nullptr;

    // Try to match uniform_dequantize_pattern -> dot.
    // This will be replaced with onednn_custom_call[int8 in, f32 out].
    bool quant_lhs =
        Match(instr, m::Op().WithOperand(
                         0, DequantizePattern(&quant_src, &src_scale, &src_zp,
                                              &copy_src, &bitcast_src)));

    bool quant_rhs =
        quant_lhs
            ? Match(instr,
                    m::Op().WithOperand(
                        1, DequantizePattern(&quant_wei, &wei_scale, &wei_zp,
                                             &copy_wei, &bitcast_wei)))
            : false;

    bool should_rewrite_int8 = false;
    if (quant_lhs) {
      should_rewrite_int8 = true;
      if (!quant_rhs) {
        // Weights (in QDQ form) are consatnt folded into F32 by earlier
        // optimization passes, we need to quantize them.
        if (rhs_dim_k == rhs_shape.dimensions_size() - 2) {
          // Channel dim is expected to be the last dimension in the shape.
          // TODO(intel-tf): Add support for the transposed case.
          const int channel_dim = rhs_shape.dimensions_size() - 1;
          absl::Status status = QuantizeWeights(instr, channel_dim, &quant_wei,
                                                &wei_scale, &wei_zp);
          should_rewrite_int8 = status.ok();
        } else {
          should_rewrite_int8 = false;
        }
      }
      if (should_rewrite_int8) {
        PrepareQuantizationInstrs(instr, quant_src, quant_wei, copy_src,
                                  copy_wei, bitcast_src, bitcast_wei,
                                  &src_scale, &src_zp, &wei_scale, &wei_zp,
                                  &new_copy_src, &new_copy_wei);
        should_rewrite_int8 =
            should_rewrite_int8 &&
            ValidateQuantizationParams(quant_src, quant_wei, src_scale, src_zp,
                                       wei_scale, wei_zp);
      }
    }
    BackendConfig backend_config;
    OneDnnMatMulConfig* matmul_config =
        backend_config.mutable_onednn_matmul_config();

    if (should_rewrite_int8) {
      // The uniform_dequantize pattern being matched assumes the zp of src was
      // negated by some earlier optimization pass.
      matmul_config->mutable_quant_config()->set_negated_src_zp(quant_lhs);
      matmul_call = instr->AddInstruction(HloInstruction::CreateCustomCall(
          output_shape,
          {new_copy_src, new_copy_wei, src_scale, src_zp, wei_scale, wei_zp},
          "__onednn$matmul"));
    } else {
      matmul_call = dot_instr->AddInstruction(HloInstruction::CreateCustomCall(
          output_shape,
          {dot_instr->mutable_operand(0), dot_instr->mutable_operand(1)},
          "__onednn$matmul"));
    }

    // Set additional info via config, e.g., transpose and fusion info.
    bool transpose_a = (lhs_dim_k + 1 != lhs_shape.dimensions().size());
    bool transpose_b = (rhs_dim_k + 2 != rhs_shape.dimensions().size());
    matmul_config->set_transpose_a(transpose_a);
    matmul_config->set_transpose_b(transpose_b);
    TF_RETURN_IF_ERROR(matmul_call->set_backend_config(backend_config));
    TF_RETURN_IF_ERROR(ReplaceInstruction(dot_instr, matmul_call));
    return absl::OkStatus();
  }

  absl::Status HandleConvolution(HloInstruction* conv) override {
    if (!OneDnnContractionRewriter::ShouldRewriteConv(conv)) {
      return absl::OkStatus();
    }

    const Shape& conv_shape = conv->shape();
    auto dims = conv->window().dimensions().size();
    const ConvolutionDimensionNumbers& conv_dims =
        conv->convolution_dimension_numbers();

    BackendConfig backend_config;
    OneDnnConvolutionConfig* conv_config =
        backend_config.mutable_onednn_conv_config();

    conv_config->set_dims(conv_shape.dimensions().size());
    conv_config->set_feature_groups(conv->feature_group_count());
    conv_config->mutable_input()->mutable_data()->set_batch_dim(
        conv_dims.input_batch_dimension());
    conv_config->mutable_kernel()->mutable_filter()->set_input_feature_dim(
        conv_dims.kernel_input_feature_dimension());
    conv_config->mutable_output()->mutable_data()->set_batch_dim(
        conv_dims.output_batch_dimension());
    conv_config->mutable_input()->mutable_data()->set_feature_dim(
        conv_dims.input_feature_dimension());
    conv_config->mutable_kernel()->mutable_filter()->set_output_feature_dim(
        conv_dims.kernel_output_feature_dimension());
    conv_config->mutable_output()->mutable_data()->set_feature_dim(
        conv_dims.output_feature_dimension());

    const Shape& output_shape = conv->shape();

    for (auto it = conv->window().dimensions().begin();
         it != conv->window().dimensions().end(); it++) {
      if ((*it).padding_low() < 0 || (*it).padding_high() < 0 ||
          (*it).stride() < 0 || (*it).base_dilation() != 1 ||
          (*it).window_reversal()) {
        return absl::OkStatus();
      }
      conv_config->mutable_window()->add_pad_left((*it).padding_low());
      conv_config->mutable_window()->add_pad_right((*it).padding_high());
      conv_config->mutable_window()->add_strides((*it).stride());
      conv_config->mutable_window()->add_window_dilations(
          (*it).window_dilation());
    }

    for (int i = 0; i < dims; i++) {
      conv_config->mutable_input()->mutable_data()->add_spatial_dims(
          conv_dims.input_spatial_dimensions()[i]);
      conv_config->mutable_kernel()->mutable_filter()->add_spatial_dims(
          conv_dims.kernel_spatial_dimensions()[i]);
      conv_config->mutable_output()->mutable_data()->add_spatial_dims(
          conv_dims.output_spatial_dimensions()[i]);
    }

    HloInstruction *quant_src = nullptr, *src_scale = nullptr,
                   *src_zp = nullptr, *quant_wei = nullptr,
                   *wei_scale = nullptr, *wei_zp = nullptr, *copy_src = nullptr,
                   *bitcast_src = nullptr, *copy_wei = nullptr,
                   *bitcast_wei = nullptr, *custom_call = nullptr,
                   *new_copy_src = nullptr, *new_copy_wei = nullptr;

    // Try to match uniform_dequantize_pattern -> convolution.
    // This will be replaced with onednn_custom_call[int8 in, f32 out].
    bool quant_lhs =
        Match(conv, m::Op().WithOperand(
                        0, DequantizePattern(&quant_src, &src_scale, &src_zp,
                                             &copy_src, &bitcast_src)));

    bool quant_rhs =
        quant_lhs
            ? Match(conv,
                    m::Op().WithOperand(
                        1, DequantizePattern(&quant_wei, &wei_scale, &wei_zp,
                                             &copy_wei, &bitcast_wei)))
            : false;

    bool should_rewrite_int8 = false;
    if (quant_lhs) {
      should_rewrite_int8 = true;
      if (!quant_rhs) {
        // Weights (in QDQ form) are consatnt folded into F32 by earlier
        // optimization passes, we need to quantize them.
        const int channel_dim = conv_dims.output_feature_dimension();
        absl::Status status =
            QuantizeWeights(conv, channel_dim, &quant_wei, &wei_scale, &wei_zp);
        should_rewrite_int8 = status.ok();
      }
      if (should_rewrite_int8) {
        PrepareQuantizationInstrs(conv, quant_src, quant_wei, copy_src,
                                  copy_wei, bitcast_src, bitcast_wei,
                                  &src_scale, &src_zp, &wei_scale, &wei_zp,
                                  &new_copy_src, &new_copy_wei);
        should_rewrite_int8 =
            should_rewrite_int8 &&
            ValidateQuantizationParams(quant_src, quant_wei, src_scale, src_zp,
                                       wei_scale, wei_zp);
      }
    }

    if (should_rewrite_int8) {
      // The uniform_dequantize pattern being matched assumes the zp of src was
      // negated by some earlier optimization pass.
      conv_config->mutable_quant_config()->set_negated_src_zp(quant_lhs);
      custom_call = conv->AddInstruction(HloInstruction::CreateCustomCall(
          output_shape,
          {new_copy_src, new_copy_wei, src_scale, src_zp, wei_scale, wei_zp},
          "__onednn$convolution"));
    } else {
      custom_call = conv->AddInstruction(HloInstruction::CreateCustomCall(
          output_shape, {conv->mutable_operand(0), conv->mutable_operand(1)},
          "__onednn$convolution"));
    }

    TF_RETURN_IF_ERROR(custom_call->set_backend_config(backend_config));
    TF_RETURN_IF_ERROR(ReplaceInstruction(conv, custom_call));
    return absl::OkStatus();
  }

  absl::Status HandleAdd(HloInstruction* instr) override {
    // Try to fuse Add to the instr. However,
    // HLO Add instruction might receive the addends after additional
    // processing like Broadcast, Bitcast, Convert, etc. is applied to the raw
    // addends. Here, the following possible pattern is matched.
    //
    // clang-format off
    //
    //         Dot / Conv                           addend
    //             |                                  |
    //             v                                  v
    //     optional instructions            optional instructions
    //    (e.g, Convert, Bitcast)         (e.g, Convert, Broadcast)
    //             |                                  |
    //             +--------------+-------------------+
    //                            |
    //                            v
    //                           Add
    //
    // clang-format on

    HloInstruction *addend_intermediate, *contraction;
    HloInstruction* optional_contraction_bitcast = nullptr;
    HloInstruction* optional_contraction_convert = nullptr;

    auto pattern = m::AddAnyOrder(
        &instr,
        OptionalConvertAndBitcast(&optional_contraction_convert,
                                  &optional_contraction_bitcast,
                                  OneDnnFusibleInstr(&contraction))
            .WithOneUser(),
        m::Op(&addend_intermediate));

    if (Match(instr, pattern)) {
      if (!IsSupportedType(contraction->shape().element_type())) {
        return absl::OkStatus();
      }

      std::vector<HloInstruction*> new_operands;
      auto contraction_operand = contraction->operand(0);
      bool quant_contraction =
          contraction_operand->shape().element_type() == S8 ||
          contraction_operand->shape().element_type() == U8;
      const int kNumQuantOperands = 4;
      const auto contraction_num_operands = contraction->operands().size();
      if (quant_contraction) {
        // Start with only adding the non int8-related operands to make sure
        // int8-related operands are always at the end.
        for (int i = 0; i < contraction_num_operands - kNumQuantOperands; ++i) {
          new_operands.push_back(contraction->operands()[i]);
        }
      } else {
        for (auto operand : contraction->operands()) {
          new_operands.push_back(operand);
        }
      }

      // At this point, the addend could have one of the following
      // possiblities that the current fusion can handle:
      //
      //   - addend -> Convert -> Broadcast -> Add
      //   - addend -> Broadcast -> Convert -> Add
      //   - addend -> Convert
      //   - addend -> Broadcast
      //   - addend
      //
      // Hunt for addend through possible sequences above and check the addend
      // is compatible for onednn fusion.
      HloInstruction* addend = nullptr;
      HloInstruction* optional_addend_broadcast = nullptr;
      auto addend_pattern = m::AnyOf<HloInstruction>(
          m::Broadcast(&optional_addend_broadcast,
                       m::Convert(&addend, m::Op())),
          m::Convert(m::Broadcast(&optional_addend_broadcast, m::Op(&addend))),
          m::Convert(&addend, m::Op()),
          m::Broadcast(&optional_addend_broadcast, m::Op(&addend)),
          m::Op(&addend));
      if (!Match(addend_intermediate, addend_pattern)) return absl::OkStatus();

      // oneDNN library requires Convolution biases to always have rank 1.
      // Therefore, these bias shapes should remain unchanged.
      if (IsOneDnnMatmulInstr(contraction) ||
          addend->shape().dimensions().size() != 1) {
        auto new_shape =
            AdjustAddendShape(contraction, addend, optional_addend_broadcast);
        if (!new_shape.ok()) {
          VLOG(2) << new_shape.status();
          return absl::OkStatus();
        }
        if (!ShapeUtil::Equal(*new_shape, addend->shape())) {
          addend = addend->AddInstruction(
              HloInstruction::CreateBitcast(new_shape.value(), addend));
        }
      }

      // If addend dtype is different from contraction dtype, add a convert
      // after the addend.
      if (addend->shape().element_type() !=
          contraction->shape().element_type()) {
        addend = addend->AddInstruction(HloInstruction::CreateConvert(
            ShapeUtil::ChangeElementType(addend->shape(),
                                         contraction->shape().element_type()),
            addend));
      }

      // For cases where the dot is followed by a reshape, the binary operands
      // shape can be adjusted, making sure the number of elements match, to
      // enable the fusion. For example:
      //      dot = f32[6304,3072] dot(...)
      //      reshape = f32[32,197,3072] reshape(dot)
      //      constant = f32[32,197,3072] constant(..)
      //      add = f32[32,197,3072] add(reshape, constant)
      // can become
      //      dot = f32[6304,3072] dot(...)
      //      constant = f32[32,197,3072] constant(..)
      //      reshape1 = f32[6304,3072] reshape(constant)
      //      add = f32[6304,3072] add(dot, reshape1)
      //      reshape2 = f32[32,197,3072] reshape(add)
      // and be replaced with the fusion
      //      fused = f32[6304,3072] custom-call(..)
      //      bitcast = f32[32,197,3072] bitcast(fused)
      auto addend_dims = addend->shape().dimensions();
      auto contraction_dims = contraction->shape().dimensions();
      if (optional_contraction_bitcast &&
          addend_dims.size() != contraction_dims.size()) {
        auto new_addend_shape =
            AdjustBinaryOperandShape(addend, contraction->shape());
        if (new_addend_shape.ok()) {
          addend = addend->AddInstruction(
              HloInstruction::CreateBitcast(new_addend_shape.value(), addend));
        }
      }

      // Validate addend for fusion.
      auto addend_user_count = addend->user_count();
      auto addend_idx = -1;
      if (IsSupportedType(addend->shape().element_type()) &&
          IsOperandFusible(addend, contraction)) {
        addend_idx = new_operands.size();
        new_operands.push_back(addend);
      } else {
        return absl::OkStatus();
      }

      // Add back the int8-related operands
      if (quant_contraction) {
        for (int i = contraction_num_operands - kNumQuantOperands;
             i < contraction_num_operands; ++i) {
          new_operands.push_back(contraction->operands()[i]);
        }
      }

      auto custom_call = Cast<HloCustomCallInstruction>(
          instr->AddInstruction(contraction->CloneWithNewOperands(
              contraction->shape(), new_operands)));

      auto backend_config = custom_call->backend_config<BackendConfig>();
      // SUM post-op does not work for BF16 because element-wise addition
      // is not allowed due to precision constraints by oneDNN.
      // Hence, the SUM use case is fused as BINARY_ADD for BF16.
      // This is verified by checking if the output shape of the
      // custom call matches the addend shape.
      bool can_fuse_sum =
          (ShapeUtil::Equal(custom_call->shape(), addend->shape()) &&
           addend_user_count == 1 &&
           custom_call->output_operand_aliasing().empty());
      auto fusions_config = GetFusionsConfig(&backend_config);
      auto optimization_config = GetOptimizationsConfig(&backend_config);
      // TODO(intel-tf): Here, we allow 1D addends only when they are the first
      // fused op. Remove this restriction once oneDNN has an optimized
      // implementation for broadcasted add across all dimensions.
      OneDnnFusionConfig_FusionKind kind =
          (ShapeUtil::TrueNumDimensions(addend->shape()) == 1)
              ? (fusions_config->ops().empty() ? OneDnnFusionConfig::BIAS
                                               : OneDnnFusionConfig::UNDEFINED)
          : can_fuse_sum ? OneDnnFusionConfig::SUM
                         : OneDnnFusionConfig::BINARY_ADD;
      if (kind == OneDnnFusionConfig::UNDEFINED) return absl::OkStatus();

      // Alias output buffers to addend for in-place accumulation
      if (kind == OneDnnFusionConfig::SUM) {
        custom_call->set_output_to_operand_aliasing({{{}, {addend_idx, {}}}});
      }

      fusions_config->add_ops(kind);

      if (optional_addend_broadcast) {
        optimization_config->set_bias_broadcast(true);
      }
      TF_RETURN_IF_ERROR(custom_call->set_backend_config(*backend_config));

      HloInstruction* new_instr;
      // If matched pattern has custom-call -> bitcast -> add, then we need to
      // insert bitcast after the new fusion to maintain the correct shape
      // (new-custom-call -> bitcast). Also, this will optionally be followed
      // by -> convert for bf16 case to avoid datatype mismatch.
      if (optional_contraction_bitcast != nullptr &&
          optional_contraction_bitcast->opcode() == HloOpcode::kBitcast) {
        if (optional_contraction_convert != nullptr &&
            optional_contraction_convert->opcode() == HloOpcode::kConvert) {
          auto bitcast_call =
              custom_call->AddInstruction(HloInstruction::CreateBitcast(
                  ShapeUtil::ChangeElementType(
                      instr->shape(), custom_call->shape().element_type()),
                  custom_call));
          new_instr =
              bitcast_call->AddInstruction(HloInstruction::CreateConvert(
                  ShapeUtil::ChangeElementType(
                      bitcast_call->shape(),
                      optional_contraction_convert->shape().element_type()),
                  bitcast_call));
        } else {
          new_instr = custom_call->AddInstruction(
              HloInstruction::CreateBitcast(instr->shape(), custom_call));
        }
      } else {
        if (optional_contraction_convert != nullptr &&
            optional_contraction_convert->opcode() == HloOpcode::kConvert) {
          new_instr = custom_call->AddInstruction(HloInstruction::CreateConvert(
              ShapeUtil::ChangeElementType(
                  custom_call->shape(),
                  optional_contraction_convert->shape().element_type()),
              custom_call));
        } else {
          new_instr = custom_call;
        }
      }
      TF_RETURN_IF_ERROR(ReplaceInstruction(instr, new_instr));
    }
    return absl::OkStatus();
  }

  absl::Status HandleMaximum(HloInstruction* instr) override {
    HloInstruction* contraction;
    HloInstruction* intermediate_instr = nullptr;
    HloInstruction* optional_bitcast = nullptr;
    // Attempt to elide maximum and fuse ReLU activation into GEMM / Conv,
    // including when slicing or bitcasting is applied to the result.
    if (Match(instr,
              m::MaximumAnyOrder(ElementwiseSafeIntermediates(
                                     &intermediate_instr, &optional_bitcast,
                                     OneDnnFusibleInstr(&contraction))
                                     .WithOneUser(),
                                 BcastConstScalar(0)))) {
      return FuseActivation(OneDnnFusionConfig::RELU, instr, contraction,
                            intermediate_instr, optional_bitcast);
    }
    return absl::OkStatus();
  }

  auto ELUActivation(HloInstruction* instr, HloInstruction** src) {
    //  Reference: tensorflow/compiler/tf2xla/kernels/elu_op.cc
    //  const auto zero = ScalarLike(x, 0);
    //  const auto pred = Gt(x, zero);
    //  const auto expm1 = Expm1(x);
    //  return Select(pred, x, expm1);
    auto pattern = m::Select(
        m::Gt(pu::OptionalConvert(m::Op(src)), BcastConvertConstScalar(0)),
        m::Op(src),
        pu::OptionalConvert(m::Expm1(pu::OptionalConvert(m::Op(src)))));
    return Match(instr, pattern);
  }

  absl::Status HandleSelect(HloInstruction* instr) override {
    HloInstruction* contraction;
    HloInstruction* intermediate_instr = nullptr;
    HloInstruction* optional_bitcast = nullptr;
    HloInstruction* src;
    // Attempt to elide ELU subgraph and fuse ELU activation into GEMM / Conv,
    // including when slicing or bitcasting is applied to the result.
    if (ELUActivation(instr, &src)) {
      if (Match(src, ElementwiseSafeIntermediates(
                         &intermediate_instr, &optional_bitcast,
                         OneDnnFusibleInstr(&contraction)))) {
        return FuseActivation(OneDnnFusionConfig::ELU, instr, contraction,
                              intermediate_instr, optional_bitcast);
      }
    }
    return absl::OkStatus();
  }

  absl::Status HandleTanh(HloInstruction* instr) override {
    HloInstruction* contraction;
    HloInstruction* intermediate_instr = nullptr;
    HloInstruction* optional_bitcast = nullptr;
    // Attempt to elide Tanh and fuse Tanh activation into GEMM / Conv,
    // including when slicing or bitcasting is applied to the result.
    if (Match(instr, m::Tanh(ElementwiseSafeIntermediates(
                                 &intermediate_instr, &optional_bitcast,
                                 OneDnnFusibleInstr(&contraction))
                                 .WithOneUser()))) {
      return FuseActivation(OneDnnFusionConfig::TANH, instr, contraction,
                            intermediate_instr, optional_bitcast);
    }
    return absl::OkStatus();
  }

  absl::Status HandleClamp(HloInstruction* instr) override {
    HloInstruction* contraction;
    HloInstruction* intermediate_instr = nullptr;
    HloInstruction* optional_bitcast = nullptr;
    // Attempt to elide RELU6 and fuse RELU6 activation into GEMM / Conv,
    // including when slicing or bitcasting is applied to the result.
    if (Match(instr, m::Clamp(BcastConstScalar(0),
                              ElementwiseSafeIntermediates(
                                  &intermediate_instr, &optional_bitcast,
                                  OneDnnFusibleInstr(&contraction))
                                  .WithOneUser(),
                              BcastConstScalar(6)))) {
      return FuseActivation(OneDnnFusionConfig::RELU6, instr, contraction,
                            intermediate_instr, optional_bitcast);
    }
    return absl::OkStatus();
  }

  // This is a heuristic to check if Multiply op is possibly part of Quantize
  // pattern.
  bool MultiplyInQuantizePattern(HloInstruction* instr) {
    auto user_instr = [&](HloInstruction* instr) {
      return (instr && instr->user_count() == 1) ? instr->users()[0] : nullptr;
    };
    auto level1 = user_instr(instr);
    auto level2 = user_instr(level1);
    auto level3 = user_instr(level2);
    auto level4 = user_instr(level3);
    if (level1 && level2 && level3 && level4) {
      std::vector<HloOpcode> ops{level1->opcode(), level2->opcode(),
                                 level3->opcode(), level4->opcode()};
      // We don't check for Add instr as it is optional in the pattern.
      if (std::find(ops.begin(), ops.end(), HloOpcode::kClamp) != ops.end() &&
          std::find(ops.begin(), ops.end(), HloOpcode::kRoundNearestEven) !=
              ops.end() &&
          std::find(ops.begin(), ops.end(), HloOpcode::kConvert) != ops.end()) {
        return true;
      }
    }
    return false;
  }

  auto SigmoidActivation(HloInstruction* instr, HloInstruction** src) {
    return Match(instr,
                 m::Divide(BcastConstScalar(1.0),
                           m::AddAnyOrder(BcastConstScalar(1.0),
                                          m::Exp(m::Negate(m::Op(src))))));
  }

  absl::Status HandleMultiply(HloInstruction* instr) override {
    HloInstruction* contraction;
    HloInstruction* intermediate_instr = nullptr;
    HloInstruction* optional_bitcast = nullptr;
    HloInstruction *src, *multiplier, *new_src;
    auto activation = GELUActivation(instr, &src);
    if (activation != OneDnnFusionConfig::UNDEFINED) {
      if (Match(src, ElementwiseSafeIntermediates(
                         &intermediate_instr, &optional_bitcast,
                         OneDnnFusibleInstr(&contraction)))) {
        return FuseActivation(activation, instr, contraction,
                              intermediate_instr, optional_bitcast);
      }
    }

    if (Match(instr, m::MultiplyAnyOrder(
                         m::Op(&multiplier).WithOpcode(HloOpcode::kDivide),
                         m::Op(&src))) &&
        SigmoidActivation(multiplier, &new_src) &&
        Match(src, ElementwiseSafeIntermediates(
                       &intermediate_instr, &optional_bitcast,
                       OneDnnFusibleInstr(&contraction))
                       .WithPredicate([&new_src](const HloInstruction* root) {
                         return root == new_src;
                       }))) {
      return FuseActivation(OneDnnFusionConfig::SWISH, instr, contraction,
                            intermediate_instr, optional_bitcast);
    }

    HloInstruction* constant;
    HloInstruction* optional_convert = nullptr;
    auto pattern =
        m::Op(&instr)
            .WithOpcode(HloOpcode::kMultiply)
            .WithBinaryOperandsAnyOrder(
                m::AnyOf<HloInstruction>(
                    pu::SupportedConvert(&optional_convert,
                                         OneDnnFusibleInstr(&contraction))
                        .WithElementType(PrimitiveType::F32),
                    OneDnnFusibleInstr(&contraction))
                    .WithOneUser(),
                m::Broadcast(m::Constant(&constant)));

    // Don't fuse Multiply if it is part of Quantize pattern. The pattern will
    // be handled later by Requantize fusion.
    if (MultiplyInQuantizePattern(instr)) return absl::OkStatus();

    if (Match(instr, pattern)) {
      std::vector<HloInstruction*> new_operands;
      auto constant_value = GetScalarConstantValue(constant);
      if (!constant_value) {
        return absl::OkStatus();
      }

      for (auto operand : contraction->operands()) {
        new_operands.push_back(operand);
      }
      auto custom_call = Cast<HloCustomCallInstruction>(instr->AddInstruction(
          contraction->CloneWithNewOperands(instr->shape(), new_operands)));
      auto backend_config = custom_call->backend_config<BackendConfig>();
      auto fusions_config = GetFusionsConfig(&backend_config);
      fusions_config->add_ops(OneDnnFusionConfig::LINEAR);
      fusions_config->add_alpha(constant_value.value());
      TF_RETURN_IF_ERROR(custom_call->set_backend_config(*backend_config));
      HloInstruction* new_instr;
      if (optional_convert != nullptr &&
          optional_convert->opcode() == HloOpcode::kConvert) {
        new_instr = custom_call->AddInstruction(HloInstruction::CreateConvert(
            ShapeUtil::ChangeElementType(
                custom_call->shape(), optional_convert->shape().element_type()),
            custom_call));
      } else {
        new_instr = custom_call;
      }

      TF_RETURN_IF_ERROR(ReplaceInstruction(instr, new_instr));
    }
    return absl::OkStatus();
  }

  absl::Status HandleDivide(HloInstruction* instr) override {
    HloInstruction* contraction;
    HloInstruction* intermediate_instr = nullptr;
    HloInstruction* optional_bitcast = nullptr;
    HloInstruction* src;
    if (SigmoidActivation(instr, &src)) {
      if (Match(src, ElementwiseSafeIntermediates(
                         &intermediate_instr, &optional_bitcast,
                         OneDnnFusibleInstr(&contraction))
                         .WithOneUser())) {
        return FuseActivation(OneDnnFusionConfig::SIGMOID, instr, contraction,
                              intermediate_instr, optional_bitcast);
      }
    }
    return absl::OkStatus();
  }

  absl::Status HandleCopy(HloInstruction* instr) override {
    HloInstruction *copy, *transpose, *custom_call;
    if (Match(instr,
              m::Copy(&copy, m::Transpose(&transpose,
                                          OneDnnMatmulInstr(&custom_call))))) {
      auto rank = transpose->dimensions().size();
      // oneDNN does not dispatch to optimized algorithm in these cases, hence
      // avoid folding transposes here.
      // TODO(intel-tf): Reevaluate this condition with future oneDNN releases.
      if (rank >= 5 || transpose->dimensions()[rank - 1] != rank - 1) {
        return absl::OkStatus();
      }
      auto backend_config = custom_call->backend_config<BackendConfig>();
      auto dimensions = backend_config->mutable_onednn_matmul_config()
                            ->mutable_result()
                            ->mutable_tensor()
                            ->mutable_dimensions();
      dimensions->Resize(transpose->dimensions().size(), 0);
      // Configure inverse transpose dimensions
      int counter = 1;
      for (auto x : transpose->dimensions()) {
        dimensions->Set(x, counter++);
      }
      auto matmul_call = Cast<HloCustomCallInstruction>(
          custom_call->AddInstruction(custom_call->CloneWithNewOperands(
              copy->shape(), custom_call->mutable_operands())));
      TF_RETURN_IF_ERROR(matmul_call->set_backend_config(*backend_config));
      TF_RETURN_IF_ERROR(ReplaceInstruction(copy, matmul_call));
    }
    return absl::OkStatus();
  }

  absl::Status FuseActivation(OneDnnFusionConfig_FusionKind kind,
                              HloInstruction* activation,
                              HloInstruction* contraction,
                              HloInstruction* intermediate_instr = nullptr,
                              HloInstruction* optional_bitcast = nullptr) {
    auto backend_config = contraction->backend_config<BackendConfig>();
    auto fusions_config = GetFusionsConfig(&backend_config);
    fusions_config->add_ops(kind);
    TF_RETURN_IF_ERROR(contraction->set_backend_config(*backend_config));
    std::unique_ptr<HloInstruction> output = contraction->Clone();
    if (optional_bitcast != nullptr &&
        optional_bitcast->opcode() == HloOpcode::kBitcast) {
      HloInstruction* new_instr = nullptr;
      if (intermediate_instr != nullptr &&
          intermediate_instr->opcode() == HloOpcode::kConvert) {
        auto bitcast_call =
            contraction->AddInstruction(HloInstruction::CreateBitcast(
                ShapeUtil::ChangeElementType(
                    optional_bitcast->shape(),
                    contraction->shape().element_type()),
                contraction));
        new_instr = bitcast_call->AddInstruction(HloInstruction::CreateConvert(
            ShapeUtil::ChangeElementType(
                bitcast_call->shape(),
                intermediate_instr->shape().element_type()),
            bitcast_call));
        return ReplaceInstruction(activation, new_instr);
      }
    } else if (intermediate_instr) {
      output = intermediate_instr->CloneWithNewOperands(
          intermediate_instr->shape(),
          {contraction->parent()->AddInstruction(std::move(output))});
    }

    return ReplaceWithNewInstruction(activation, std::move(output));
  }

  // This function changes dot instruction for supported matrix
  // multiplication scenarios. In particular, it changes the shape
  // of lhs, rhs and result arrays.
  //    - lhs configuration scenario
  //      lhs:    [batch_dims,contracting_dim] to [batch_dims,1,contracting_dim]
  //      result: [batch_dims,feature_dim] to [batch_dims,1,feature_dim]
  //
  //    - rhs configuration scenario
  //      rhs:    [batch_dims,contracting_dim] to [batch_dims,contracting_dim,1]
  //      result: [batch_dims,feature_dim] to [batch_dims,feature_dim, 1]
  //
  //    - both lhs and rhs configuration scenario
  //      lhs:    [batch_dims,contracting_dim] to [batch_dims,1,contracting_dim]
  //      rhs:    [batch_dims,contracting_dim] to [batch_dims,contracting_dim,1]
  //      result: [batch_dims] to [batch_dims,1,1]
  absl::StatusOr<HloInstruction*> ReconfigureDotDimensions(
      HloInstruction* dot_instr) {
    HloInstruction* lhs = dot_instr->mutable_operand(0);
    HloInstruction* rhs = dot_instr->mutable_operand(1);
    DotDimensionNumbers dim_numbers = dot_instr->dot_dimension_numbers();

    auto lhs_batch_dims = dim_numbers.lhs_batch_dimensions();
    auto lhs_contraction_dims = dim_numbers.lhs_contracting_dimensions();
    bool is_lhs_vector = lhs->shape().dimensions().size() ==
                         (lhs_batch_dims.size() + lhs_contraction_dims.size());

    auto rhs_batch_dims = dim_numbers.rhs_batch_dimensions();
    auto rhs_contraction_dims = dim_numbers.rhs_contracting_dimensions();
    bool is_rhs_vector = rhs->shape().dimensions().size() ==
                         (rhs_batch_dims.size() + rhs_contraction_dims.size());

    if (!is_lhs_vector && !is_rhs_vector) return dot_instr;

    std::vector<int64_t> adjusted_lhs_dims(lhs->shape().dimensions().begin(),
                                           lhs->shape().dimensions().end());
    std::vector<int64_t> adjusted_rhs_dims(rhs->shape().dimensions().begin(),
                                           rhs->shape().dimensions().end());
    std::vector<int64_t> adjusted_dot_dims(
        dot_instr->shape().dimensions().begin(),
        dot_instr->shape().dimensions().end());

    if (is_lhs_vector) {
      auto lhs_it = adjusted_lhs_dims.begin() + lhs_batch_dims.size();
      adjusted_lhs_dims.insert(lhs_it, 1, 1);
      auto result_it = adjusted_dot_dims.begin() + lhs_batch_dims.size();
      adjusted_dot_dims.insert(result_it, 1, 1);
      auto lhs_contraction_dim =
          dot_instr->dot_dimension_numbers().lhs_contracting_dimensions(0);
      dim_numbers.set_lhs_contracting_dimensions(0, lhs_contraction_dim + 1);
      lhs = lhs->AddInstruction(HloInstruction::CreateBitcast(
          ShapeUtil::MakeShape(lhs->shape().element_type(), adjusted_lhs_dims),
          lhs));
    }

    if (is_rhs_vector) {
      auto it = adjusted_rhs_dims.end();
      adjusted_rhs_dims.insert(it, 1, 1);
      auto result_it = adjusted_dot_dims.end();
      adjusted_dot_dims.insert(result_it, 1, 1);
      rhs = rhs->AddInstruction(HloInstruction::CreateBitcast(
          ShapeUtil::MakeShape(rhs->shape().element_type(), adjusted_rhs_dims),
          rhs));
    }

    HloInstruction* adjusted_dot =
        dot_instr->AddInstruction(HloInstruction::CreateDot(
            ShapeUtil::MakeShape(dot_instr->shape().element_type(),
                                 adjusted_dot_dims),
            lhs, rhs, dim_numbers, dot_instr->precision_config()));

    HloInstruction* replacement_instr = adjusted_dot->AddInstruction(
        HloInstruction::CreateBitcast(dot_instr->shape(), adjusted_dot));

    TF_RETURN_IF_ERROR(ReplaceInstruction(dot_instr, replacement_instr));
    return adjusted_dot;
  }

  // This function upcasts BF16 dots to F32 if we are unable to rewrite them to
  // oneDNN custom calls.
  absl::Status UpcastDotToF32(HloInstruction* dot_instr) {
    if (dot_instr->shape().element_type() != BF16) return absl::OkStatus();
    std::vector<HloInstruction*> new_operands;
    auto bf16_operands = dot_instr->operands();

    std::for_each(
        bf16_operands.begin(), bf16_operands.end(),
        [&new_operands](HloInstruction* instr) {
          new_operands.push_back(
              instr->AddInstruction(HloInstruction::CreateConvert(
                  ShapeUtil::ChangeElementType(instr->shape(), F32), instr)));
        });

    HloInstruction* f32_dot =
        dot_instr->AddInstruction(dot_instr->CloneWithNewOperands(
            ShapeUtil::ChangeElementType(dot_instr->shape(), F32),
            new_operands));

    HloInstruction* replacement_instr =
        f32_dot->AddInstruction(HloInstruction::CreateConvert(
            ShapeUtil::ChangeElementType(f32_dot->shape(), BF16), f32_dot));

    TF_RETURN_IF_ERROR(ReplaceInstruction(dot_instr, replacement_instr));
    return absl::OkStatus();
  }

absl::Status QuantizeWeights(HloInstruction* instr, const int channel_dim,
                               HloInstruction** quant_wei,
                               HloInstruction** wei_scale,
                               HloInstruction** wei_zp) {
    auto rhs = instr->operand(1);
    auto rhs_shape = rhs->shape();
    const auto rhs_type = rhs_shape.element_type();
    if (rhs_type != F32) {
      return absl::FailedPreconditionError(
          "Could not quantize weights! F32 weights type was expected.");
    }

    Literal weights_literal;
    if (!evaluator_.TryEvaluate(rhs, &weights_literal)) {
      return absl::FailedPreconditionError(
          "Could not quantize weights! Constant evaluation failed.");
    }

    // Channel dim is expected to be the last dimension in the shape.
    if (channel_dim != rhs_shape.dimensions().size() - 1) {
      return absl::FailedPreconditionError(
          "Could not quantize weights! Quantization axis is not supported.");
    }

    const int channel_dim_size = rhs_shape.dimensions(channel_dim);
    std::vector<float> scales(channel_dim_size);
    const int rhs_size = ShapeUtil::ElementsIn(rhs_shape);
    using NativeT = typename primitive_util::PrimitiveTypeToNative<F32>::type;
    NativeT* data = (NativeT*)(weights_literal.untyped_data());
    // Compute the weights' scales per channel_dim
    // TODO(intel-tf): Parallelize this loop to reduce compilation time.
    for (int i = 0; i < channel_dim_size; i++) {
      NativeT max = 0;
      for (int j = 0; j < rhs_size; j += channel_dim_size) {
        const NativeT abs_value = std::abs(data[j + i]);
        if (abs_value > max) {
          max = abs_value;
        }
      }
      scales[i] = max / 127.0;
    }
    dnnl::engine cpu_engine(dnnl::engine::kind::cpu, 0);
    auto src_md = ShapeToMemDesc(rhs_shape);
    auto dst_md = dnnl::memory::desc(
        src_md.get_dims(), dnnl::memory::data_type::s8, src_md.get_strides());
    auto src_mem =
        dnnl::memory(src_md, cpu_engine, weights_literal.untyped_data());
    auto scaled_rhs = Literal(ShapeUtil::ChangeElementType(rhs_shape, S8));
    auto dst_mem = dnnl::memory(dst_md, cpu_engine, scaled_rhs.untyped_data());
    auto dst_scales_mem = dnnl::memory({{channel_dim_size},
                                        dnnl::memory::data_type::f32,
                                        dnnl::memory::format_tag::x},
                                       cpu_engine, scales.data());

    dnnl::primitive_attr reorder_attr;
    auto mask = 1 << channel_dim;
    reorder_attr.set_scales_mask(DNNL_ARG_DST, mask);
    auto reorder_desc = dnnl::reorder::primitive_desc(
        cpu_engine, src_md, cpu_engine, dst_md, reorder_attr);
    dnnl::reorder reorder_prim(reorder_desc);
    std::unordered_map<int, dnnl::memory> reorder_args;
    reorder_args.insert({DNNL_ARG_SRC, src_mem});
    reorder_args.insert({DNNL_ARG_DST, dst_mem});
    reorder_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scales_mem});
    auto onednn_threadpool = std::make_unique<OneDnnThreadPool>(
        threadpool_device_->getPool(), /*is_async=*/true);
    auto onednn_stream = dnnl::threadpool_interop::make_stream(
        cpu_engine, onednn_threadpool.get());
    reorder_prim.execute(onednn_stream, reorder_args);
    onednn_stream.wait();

    *quant_wei = instr->AddInstruction(
        HloInstruction::CreateConstant(std::move(scaled_rhs)));
    *wei_scale = instr->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(scales)));
    std::vector<int> zp{0};
    *wei_zp = instr->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<int>(zp)));
    return absl::OkStatus();
  }

 private:
  bool graph_enabled_;
  int intra_op_parallelism_;
  HloEvaluator evaluator_;
  std::unique_ptr<tsl::thread::ThreadPool> threadpool_handle_;
  std::unique_ptr<Eigen::ThreadPoolDevice> threadpool_device_;
};

class OneDnnContractionRequantizeVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleCustomCall(HloInstruction* custom_call) override {
    HloInstruction *contraction = nullptr, *input_custom_call = nullptr,
                   *scale = nullptr, *zp = nullptr, *clamp_min = nullptr,
                   *clamp_max = nullptr;

    if (Match(custom_call, OneDnnFusibleInstr(&contraction))) {
      // Try to match the requantize case:
      // onednn_custom_call[int8 in, f32 out] -> uniform_quantize_pattern ->
      //        onednn_custom_call[int8 in, f32 out].
      // This will be replaced by
      // onednn_custom_call[int8 in, int8 out] -> onednn_custom_call[int8 in,
      // f32 out]
      bool requant_contraction = Match(
          custom_call,
          m::Op()
              .WithOpcode(HloOpcode::kCustomCall)
              .WithOperand(0, QuantizePattern(&scale, &zp, &input_custom_call,
                                              &clamp_min, &clamp_max)));
      if (!Match(input_custom_call, OneDnnFusibleInstr(&input_custom_call))) {
        return absl::OkStatus();
      }

      if (requant_contraction && input_custom_call) {
        auto input_custom_call_operands = input_custom_call->operands();
        auto src_type = input_custom_call_operands[0]->shape().element_type();
        auto wei_type = input_custom_call_operands[1]->shape().element_type();
        // Make sure the operands of input_custom_call are quantized.
        if ((src_type != S8 && src_type != U8) || wei_type != S8) {
          return absl::OkStatus();
        }

        const float default_scale = 1.0;
        const int default_zp = 0;
        auto set_default_value = [&]<typename T>(HloInstruction*& instr,
                                                 T value) {
          instr = contraction->AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::CreateR0<T>(value)));
        };
        // If scale and/or zp are not found. That means they had a default
        // values that were optimized out. Hence, we set a default value of 1.0
        // and 0 for scale and zp respectively.
        if (scale == nullptr) {
          set_default_value(scale, default_scale);
        }
        if (zp == nullptr) {
          set_default_value(zp, default_zp);
        }

        std::vector<HloInstruction*> requant_call_operands;
        for (auto operand : input_custom_call_operands) {
          requant_call_operands.push_back(operand);
        }
        requant_call_operands.push_back(scale);
        requant_call_operands.push_back(zp);

        auto quant_type = custom_call->operands()[0]->shape().element_type();
        auto clamp_min_value = GetScalarConstantValue(clamp_min);
        auto clamp_max_value = GetScalarConstantValue(clamp_max);
        if (!clamp_min_value || !clamp_max_value) {
          return absl::OkStatus();
        }
        // Currently we only expect below clamp values.
        // For S8, clamp_min = -128 or -127, clamp_max = 127
        // For U8, clamp_min = 0, clamp_max = 255 or 254
        auto min = clamp_min_value.value();
        auto max = clamp_max_value.value();
        if (quant_type == S8 &&
            (!(min == -128.0 || min == -127.0) || max != 127.0)) {
          return absl::OkStatus();
        }
        if (quant_type == U8 && (min != 0 || !(max == 254.0 || max == 255.0))) {
          return absl::OkStatus();
        }
        auto requant_contraction_call =
            Cast<HloCustomCallInstruction>(input_custom_call->AddInstruction(
                input_custom_call->CloneWithNewOperands(
                    ShapeUtil::ChangeElementType(input_custom_call->shape(),
                                                 quant_type),
                    requant_call_operands)));

        TF_ASSIGN_OR_RETURN(
            auto backend_config,
            requant_contraction_call->backend_config<BackendConfig>());

        // The uniform_quantize pattern being matched assumes the scale of dst
        // was inversed by some earlier optimization pass.
        if (backend_config.has_onednn_conv_config()) {
          OneDnnConvolutionConfig* conv_config =
              backend_config.mutable_onednn_conv_config();
          conv_config->mutable_quant_config()->set_inversed_dst_scale(
              requant_contraction);
        } else if (backend_config.has_onednn_matmul_config()) {
          OneDnnMatMulConfig* matmul_config =
              backend_config.mutable_onednn_matmul_config();
          matmul_config->mutable_quant_config()->set_inversed_dst_scale(
              requant_contraction);
        } else {
          return absl::OkStatus();
        }

        requant_contraction_call->set_backend_config(backend_config);

        const int size = custom_call->operands().size();
        std::vector<HloInstruction*> new_operands(size);
        new_operands[0] = requant_contraction_call;
        for (int i = 1; i < size; ++i) {
          new_operands[i] = custom_call->operands()[i];
        }
        auto new_contraction_call = Cast<HloCustomCallInstruction>(
            custom_call->AddInstruction(custom_call->CloneWithNewOperands(
                custom_call->shape(), new_operands)));

        TF_RETURN_IF_ERROR(
            ReplaceInstruction(custom_call, new_contraction_call));
      }
    }
    return absl::OkStatus();
  }
};

class OneDnnPostRewriteVisitor : public DfsHloRewriteVisitor {
 public:
  OneDnnPostRewriteVisitor(int intra_op_parallelism,
                           const tsl::thread::ThreadPool* compile_threadpool)
      : intra_op_parallelism_(intra_op_parallelism > 0
                                  ? intra_op_parallelism
                                  : tsl::port::MaxParallelism()),
        evaluator_(/*max_loop_iterations=*/0) {
    if (compile_threadpool) {
      threadpool_device_.reset(
          new Eigen::ThreadPoolDevice(compile_threadpool->AsEigenThreadPool(),
                                      compile_threadpool->NumThreads()));
    } else {
      threadpool_handle_.reset(new tsl::thread::ThreadPool(
          tsl::Env::Default(), "XLACpuCompile", tsl::port::MaxParallelism()));
      threadpool_device_.reset(
          new Eigen::ThreadPoolDevice(threadpool_handle_->AsEigenThreadPool(),
                                      threadpool_handle_->NumThreads()));
    }

#ifndef ENABLE_ONEDNN_OPENMP
    // Set oneDNN concurrency settings (which is thread-local)
    OneDnnThreadPool::set_onednn_max_threads(intra_op_parallelism_);
#endif
  }

  void UpdateTransposeDimensions(
      HloInstruction* matmul, absl::InlinedVector<HloInstruction*, 2>& new_ops,
      int operand_idx, absl::StatusOr<BackendConfig>* backend_config) {
    HloInstruction *transpose, *operand;
    // Update the dimensions only when the transpose does not involve the batch
    // dimension, as modifying it could significantly impact the performance.
    if (Match(matmul->mutable_operand(operand_idx),
              m::Copy(m::Transpose(&transpose, m::Op(&operand)))) &&
        transpose->dimensions()[0] == 0) {
      new_ops[operand_idx] = operand;
      for (auto x : transpose->dimensions()) {
        (*GetOperandTensor(operand_idx, backend_config))->Add(x + 1);
      }
    }
  }

  absl::Status HandleCustomCall(HloInstruction* custom_call) override {
    HloInstruction* contraction;
    if (Match(custom_call, OneDnnMatmulInstr(&contraction))) {
      auto backend_config = contraction->backend_config<BackendConfig>();

      auto new_ops = contraction->mutable_operands();

      UpdateTransposeDimensions(contraction, new_ops, 0, &backend_config);
      UpdateTransposeDimensions(contraction, new_ops, 1, &backend_config);

      auto matmul_call = Cast<HloCustomCallInstruction>(
          contraction->AddInstruction(contraction->CloneWithNewOperands(
              contraction->shape(), new_ops)));
      TF_RETURN_IF_ERROR(matmul_call->set_backend_config(*backend_config));
      TF_RETURN_IF_ERROR(ReplaceInstruction(contraction, matmul_call));
      return HandleCustomCallInternal<dnnl::matmul::primitive_desc>(
          matmul_call);
    } else if (Match(custom_call, OneDnnConvolutionInstr(&contraction))) {
      return HandleCustomCallInternal<
          dnnl::convolution_forward::primitive_desc>(custom_call);
    }
    return DefaultAction(custom_call);
  }

  template <typename PrimDesc>
  absl::Status HandleCustomCallInternal(HloInstruction* custom_call) {
    auto scratch_add = AddScratch<PrimDesc>(custom_call);
    if (scratch_add.ok()) {
      custom_call = *scratch_add;
      auto aliases = custom_call->output_operand_aliasing();
      if (!aliases.empty()) {
        custom_call->set_output_to_operand_aliasing({{{0}, aliases[0].second}});
      }
    } else {
      VLOG(2) << scratch_add.status();
    }
    // TODO(intel-tf): Remove this condition after enabling weights prepacking
    // for convolutions
    if constexpr (std::is_same_v<PrimDesc, dnnl::matmul::primitive_desc>) {
      auto weights_prepack = PrepackWeights<PrimDesc>(custom_call);
      if (!weights_prepack.ok()) {
        VLOG(2) << weights_prepack.status();
      }
    }
    return absl::OkStatus();
  }

  template <typename>
  absl::Status SetWeightsPrepack(HloInstruction*, bool);

  template <typename>
  absl::Status SetUserScratch(HloInstruction*, bool);

  template <typename>
  bool GetWeightsPrepack(HloInstruction*);

  template <typename>
  bool GetUserScratch(HloInstruction*);

  // Add scratch for matmul and convolution by changing the result of
  // custom-call to tuple(result, scratch)
  template <typename PrimDesc>
  absl::StatusOr<HloInstruction*> AddScratch(HloInstruction* custom_call) {
    if (GetUserScratch<PrimDesc>(custom_call)) {
      return custom_call;
    }
    TF_RETURN_IF_ERROR(SetUserScratch<PrimDesc>(custom_call, true));
    auto prim_desc = CreateOneDnnPrimDesc<PrimDesc>(custom_call);
    int64_t scratch_size = prim_desc->scratchpad_desc().get_size();
    Shape scratch_shape = ShapeUtil::MakeShape(U8, {scratch_size});
    Shape tuple_shape =
        ShapeUtil::MakeTupleShape({custom_call->shape(), scratch_shape});
    auto new_custom_call = custom_call->AddInstruction(
        custom_call->CloneWithNewShape(tuple_shape));
    HloInstruction* gte =
        new_custom_call->AddInstruction(HloInstruction::CreateGetTupleElement(
            custom_call->shape(), new_custom_call, 0));
    auto status = ReplaceInstruction(custom_call, gte);
    if (!status.ok()) {
      TF_RETURN_IF_ERROR(SetUserScratch<PrimDesc>(custom_call, false));
      return absl::CancelledError("Adding scratch is unsuccessful.");
    }
    return new_custom_call;
  }

  template <typename PrimDesc>
  absl::StatusOr<HloInstruction*> PrepackWeights(HloInstruction* custom_call) {
    if (GetWeightsPrepack<PrimDesc>(custom_call)) {
      return custom_call;
    }
    auto weights = custom_call->operand(1);
    auto weights_shape = weights->shape();
    Literal weights_literal;
    if (!(weights_shape.dimensions().size() == 2 &&
          evaluator_.TryEvaluate(weights, &weights_literal, true))) {
      return absl::CancelledError(
          "Cannot prepack weights. Not constant 2D weights.");
    }
    auto plain_weights_md = ShapeToMemDesc(weights_shape);
    if constexpr (std::is_same<PrimDesc, dnnl::matmul::primitive_desc>::value) {
      TF_ASSIGN_OR_RETURN(auto backend_config,
                          custom_call->backend_config<BackendConfig>());
      TRANSPOSE_LAST_TWO_DIMS_IF(
          backend_config.onednn_matmul_config().transpose_b(),
          plain_weights_md);
    }
    TF_RETURN_IF_ERROR(SetWeightsPrepack<PrimDesc>(custom_call, true));
    auto prim_desc = CreateOneDnnPrimDesc<PrimDesc>(custom_call);
    auto packed_weights_md = prim_desc->weights_desc();
    auto packed_weights_shape = MemDescToXlaShapeFlattened(packed_weights_md);
    auto packed_weights_literal = Literal(packed_weights_shape);
    ReorderWeight(plain_weights_md, weights_literal.untyped_data(),
                  packed_weights_md, packed_weights_literal.untyped_data());
    HloInstruction* reordered_weight = custom_call->AddInstruction(
        HloInstruction::CreateConstant(std::move(packed_weights_literal)));
    auto status =
        custom_call->ReplaceOperandWithDifferentShape(1, reordered_weight);
    if (!status.ok()) {
      TF_RETURN_IF_ERROR(SetWeightsPrepack<PrimDesc>(custom_call, false));
      return absl::CancelledError(
          "Cannot replace plain weights with prepacked weights.");
    } else {
      return custom_call;
    }
  }

  absl::StatusOr<tsl::protobuf::RepeatedField<uint64_t>*> GetOperandTensor(
      int operand_idx, absl::StatusOr<BackendConfig>* backend_config) {
    if (operand_idx > 1) {
      return absl::CancelledError("Operand index must be either 0 or 1");
    }
    auto operand =
        (operand_idx == 0)
            ? (*backend_config)->mutable_onednn_matmul_config()->mutable_lhs()
            : (*backend_config)->mutable_onednn_matmul_config()->mutable_rhs();
    return operand->mutable_tensor()->mutable_dimensions();
  }

  void ReorderWeight(const dnnl::memory::desc& src_md, void* src_buf,
                     const dnnl::memory::desc& dst_md, void* dst_buf) {
    auto onednn_threadpool = std::make_unique<OneDnnThreadPool>(
        threadpool_device_->getPool(), /*is_async=*/true);
    dnnl::engine cpu_engine(dnnl::engine::kind::cpu, 0);
    auto onednn_stream = dnnl::threadpool_interop::make_stream(
        cpu_engine, onednn_threadpool.get());
    auto src_mem = dnnl::memory(src_md, cpu_engine, src_buf);
    auto dst_mem = dnnl::memory(dst_md, cpu_engine, dst_buf);
    dnnl::reorder reorder_prim{src_mem, dst_mem};
    reorder_prim.execute(onednn_stream, src_mem, dst_mem);
    // Wait for the reorder to finish before destroying the threadpool.
    onednn_stream.wait();
  }

 private:
  int intra_op_parallelism_;
  HloEvaluator evaluator_;
  std::unique_ptr<tsl::thread::ThreadPool> threadpool_handle_;
  std::unique_ptr<Eigen::ThreadPoolDevice> threadpool_device_;
};

#define EMIT_GET_BACKEND_CONFIG_SPECIALIZATION(GETTER, PRIM_DESC, CONFIG,      \
                                               SUB_CONFIG, FIELD)              \
  template <>                                                                  \
  inline bool OneDnnPostRewriteVisitor::GETTER<PRIM_DESC>(HloInstruction *     \
                                                          custom_call) {       \
    auto backend_config = custom_call->backend_config<BackendConfig>();        \
    return backend_config.ok() ? backend_config->CONFIG().SUB_CONFIG().FIELD() \
                               : false;                                        \
  }

EMIT_GET_BACKEND_CONFIG_SPECIALIZATION(GetUserScratch,
                                       dnnl::matmul::primitive_desc,
                                       onednn_matmul_config,
                                       optimization_config, user_scratchpad);
EMIT_GET_BACKEND_CONFIG_SPECIALIZATION(GetWeightsPrepack,
                                       dnnl::matmul::primitive_desc,
                                       onednn_matmul_config,
                                       optimization_config, weights_prepacked);
EMIT_GET_BACKEND_CONFIG_SPECIALIZATION(
    GetUserScratch, dnnl::convolution_forward::primitive_desc,
    onednn_conv_config, optimization_config, user_scratchpad);

#define EMIT_SET_BACKEND_CONFIG_SPECIALIZATION(SETTER, PRIM_DESC, CONFIG_TYPE, \
                                               CONFIG, SUB_CONFIG, FIELD)      \
  template <>                                                                  \
  inline absl::Status OneDnnPostRewriteVisitor::SETTER<PRIM_DESC>(             \
      HloInstruction * custom_call, bool value) {                              \
    TF_ASSIGN_OR_RETURN(auto backend_config,                                   \
                        custom_call->backend_config<BackendConfig>());         \
    CONFIG_TYPE* config = backend_config.mutable_##CONFIG();                   \
    config->mutable_##SUB_CONFIG()->set_##FIELD(value);                        \
    return custom_call->set_backend_config(backend_config);                    \
  }

EMIT_SET_BACKEND_CONFIG_SPECIALIZATION(SetWeightsPrepack,
                                       dnnl::matmul::primitive_desc,
                                       OneDnnMatMulConfig, onednn_matmul_config,
                                       optimization_config, weights_prepacked);
EMIT_SET_BACKEND_CONFIG_SPECIALIZATION(SetUserScratch,
                                       dnnl::matmul::primitive_desc,
                                       OneDnnMatMulConfig, onednn_matmul_config,
                                       optimization_config, user_scratchpad);
EMIT_SET_BACKEND_CONFIG_SPECIALIZATION(
    SetUserScratch, dnnl::convolution_forward::primitive_desc,
    OneDnnConvolutionConfig, onednn_conv_config, optimization_config,
    user_scratchpad);

absl::StatusOr<bool> OneDnnContractionRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      3, "OneDnnContractionRewriter::Run(), before:\n" + module->ToString());
  OneDnnContractionRewriteVisitor visitor(graph_enabled_, intra_op_parallelism_,
                                           compile_threadpool_);
  TF_ASSIGN_OR_RETURN(auto result,
                      visitor.RunOnModule(module, execution_threads));

  OneDnnContractionRequantizeVisitor requantize_visitor;
  TF_ASSIGN_OR_RETURN(auto requantize_result, requantize_visitor.RunOnModule(
                                                  module, execution_threads));

  OneDnnPostRewriteVisitor reorder_visitor(intra_op_parallelism_,
                                           compile_threadpool_);
  TF_ASSIGN_OR_RETURN(auto result2,
                      reorder_visitor.RunOnModule(module, execution_threads));
  XLA_VLOG_LINES(
      3, "OneDnnContractionRewriter::Run(), after:\n" + module->ToString());
  return {result || requantize_result || result2};
}

}  // namespace cpu
}  // namespace xla
