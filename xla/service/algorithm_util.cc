/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/algorithm_util.h"

#include <cstdint>
#include <iterator>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "tsl/platform/protobuf.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace algorithm_util {

namespace {
namespace se = stream_executor;
}  // namespace

absl::StatusOr<se::blas::ComputationType> GetBlasComputationType(
    PrecisionConfig::Algorithm algorithm) {
  // Note: If we will support other algorithm & storage type combinations, such
  // as ALG_DOT_BF16_BF16_F32 with F32 input and output storage types, then
  // we'll have to also depend on the storage types here. For the mentioned
  // example, the computation type would be kBF16AsF32.
  // Only the currently supported algorithms are listed here.
  switch (algorithm) {
    case PrecisionConfig::ALG_DOT_F16_F16_F16:
      return se::blas::ComputationType::kF16;
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
      return se::blas::ComputationType::kBF16AsF32;
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM:
    case PrecisionConfig::ALG_DOT_F16_F16_F32:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
    case PrecisionConfig::ALG_DOT_BF16_BF16_FP8X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_FP8X4:

    case PrecisionConfig::ALG_DOT_F32_F32_F32:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
      return se::blas::ComputationType::kF32;
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32:
      return se::blas::ComputationType::kTF32AsF32;
    case PrecisionConfig::ALG_DOT_F64_F64_F64:
      return se::blas::ComputationType::kF64;
    default:
      return absl::InternalError(
          absl::StrFormat("GetBlasComputationType: unsupported algorithm %s",
                          xla::PrecisionConfig::Algorithm_Name(algorithm)));
  }
}

absl::StatusOr<std::vector<PrimitiveType>> GetAllowedOperandsTypeForAlgorithm(
    PrecisionConfig::Algorithm algorithm) {
  switch (algorithm) {
    case PrecisionConfig::ALG_UNSET:
      break;
    case PrecisionConfig::ALG_DOT_F16_F16_F16:
    case PrecisionConfig::ALG_DOT_F16_F16_F32:
      return std::vector<PrimitiveType>{F16};
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
      return std::vector<PrimitiveType>{F32};
    case PrecisionConfig::ALG_DOT_F64_F64_F64:
      return std::vector<PrimitiveType>{F64};
    case PrecisionConfig::ALG_DOT_BF16_BF16_BF16:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
      return std::vector<PrimitiveType>{BF16};
    case PrecisionConfig::ALG_DOT_BF16_BF16_FP8X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_FP8X4:
      return std::vector<PrimitiveType>{BF16, F32};
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
      return std::vector<PrimitiveType>{F32};  // This is not a typo.
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
      return std::vector<PrimitiveType>{F32};
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM: {
      std::vector<PrimitiveType> f8_types;
      const tsl::protobuf::EnumDescriptor* desc =
          tsl::protobuf::GetEnumDescriptor<PrimitiveType>();
      for (int i = 0; i < desc->value_count(); ++i) {
        auto ty = static_cast<PrimitiveType>(desc->value(i)->number());
        if (primitive_util::IsF8Type(ty)) {
          f8_types.push_back(ty);
        }
      }
      return f8_types;
    }
    default:
      break;
  }
  return absl::InternalError(
      absl::StrFormat("GetDotAccumulatorType: unsupported algorithm %s",
                      xla::PrecisionConfig::Algorithm_Name(algorithm)));
}

absl::StatusOr<PrimitiveType> GetDotAccumulatorType(
    PrecisionConfig::Algorithm algorithm) {
  // All dot algorithms should be listed here.
  switch (algorithm) {
    case PrecisionConfig::ALG_DOT_F16_F16_F16:
      return F16;
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM:
    case PrecisionConfig::ALG_DOT_F16_F16_F32:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
    case PrecisionConfig::ALG_DOT_BF16_BF16_FP8X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_FP8X4:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
      return F32;
    case PrecisionConfig::ALG_DOT_BF16_BF16_BF16:
      return BF16;
    case PrecisionConfig::ALG_DOT_F64_F64_F64:
      return F64;
    case PrecisionConfig::ALG_UNSET:
    default:
      return absl::InternalError(
          absl::StrFormat("GetDotAccumulatorType: unsupported algorithm %s",
                          xla::PrecisionConfig::Algorithm_Name(algorithm)));
  }
}

absl::StatusOr<PrimitiveType> GetDefaultGemmAlgorithmAccumulatorType(
    const HloInstruction* dot) {
  TF_RET_CHECK(dot != nullptr);
  TF_RET_CHECK(dot->opcode() == HloOpcode::kDot ||
               dot->opcode() == HloOpcode::kRaggedDot);

  PrimitiveType lhs_type = dot->operand(0)->shape().element_type();
  PrimitiveType rhs_type = dot->operand(1)->shape().element_type();
  PrimitiveType output_type = dot->shape().element_type();

  if (primitive_util::IsF8Type(lhs_type) &&
      primitive_util::IsF8Type(rhs_type)) {
    return F32;
  }

  if ((lhs_type == S8 || lhs_type == U8) &&
      (rhs_type == S8 || rhs_type == U8) &&
      (output_type == S32 || output_type == U32)) {
    return S32;
  }

  if (lhs_type == F64 && output_type == F64) {
    return F64;
  }

  return F32;
}

absl::StatusOr<PrimitiveType> GetDotAccumulatorType(const HloInstruction* dot) {
  TF_RET_CHECK(dot != nullptr);
  TF_RET_CHECK(dot->opcode() == HloOpcode::kDot);

  if (dot->precision_config().algorithm() == PrecisionConfig::ALG_UNSET) {
    return GetDefaultGemmAlgorithmAccumulatorType(dot);
  }
  return GetDotAccumulatorType(dot->precision_config().algorithm());
}

bool HasTf32InputType(PrecisionConfig::Algorithm algorithm) {
  return algorithm == PrecisionConfig::ALG_DOT_TF32_TF32_F32 ||
         algorithm == PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3;
}

bool HasFastAccum(PrecisionConfig::Algorithm algorithm) {
  return algorithm == PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM;
}

// It's clear that those libraries could support more, but we only list the ones
// which we explicitly test for now.
bool IsSupportedByCublasOrCublasLt(
    PrecisionConfig::Algorithm algorithm,
    stream_executor::GpuComputeCapability gpu_compute_capability,
    const HloDotInstruction* dot, const int64_t rhs_contracting_index) {
  // 8-bit x 8-bit GEMMs with contracting dim < 4 are not supported by cuBLAS.
  // As this was determined through a failing test, I'm eering on the side of
  // caution and not generalizing this further.
  if (dot) {
    auto lhs_type = dot->operand(0)->shape().element_type();
    auto rhs_type = dot->operand(1)->shape().element_type();
    auto contracting_dim_size =
        dot->operand(1)->shape().dimensions(rhs_contracting_index);
    if (primitive_util::Is8BitIntegralType(lhs_type) &&
        primitive_util::Is8BitIntegralType(rhs_type) &&
        contracting_dim_size < 4) {
      return false;
    }
  }

  switch (algorithm) {
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
    case PrecisionConfig::ALG_UNSET:
    case PrecisionConfig::ALG_DOT_F16_F16_F32:
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
    case PrecisionConfig::ALG_DOT_F64_F64_F64:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM:
      return true;
    default:
      return false;
  }
}

// Checks if we support the given algorithm using cuDNN.
bool IsSupportedByCudnn(PrecisionConfig::Algorithm algorithm) {
  switch (algorithm) {
    // When the CuDnn backend starts supporting specific algorithms, then
    // those should be listed here.
    case PrecisionConfig::ALG_UNSET:
      return true;
    default:
      return false;
  }
}

bool IsSupportedByElementalIrEmitter(PrecisionConfig::Algorithm algorithm) {
  switch (algorithm) {
    // Probably more can be added.
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
    case PrecisionConfig::ALG_UNSET:
      return true;
    default:
      return false;
  }
}

// Is the given algorithm supported on GPU with the given compute capability and
// input/output storage types.
bool IsSupportedDotAlgorithmOnGpu(
    PrecisionConfig::Algorithm algorithm,
    const stream_executor::GpuComputeCapability& gpu_compute_capability,
    PrimitiveType lhs_storage_type, PrimitiveType rhs_storage_type,
    PrimitiveType output_storage_type) {
  // Note: We may want to add some complex types here if people request that.
  const bool is_cuda_ge_ampere =
      gpu_compute_capability.IsCuda() &&
      gpu_compute_capability.cuda_compute_capability()->IsAtLeastAmpere();

  const bool is_cuda_ge_ada =
      gpu_compute_capability.IsCuda() &&
      gpu_compute_capability.cuda_compute_capability()->IsAtLeast(8, 9);

  const bool is_rocm_mi100_and_above =
      gpu_compute_capability.IsRocm() &&
      gpu_compute_capability.rocm_compute_capability()->gfx9_mi100_or_later();

  const bool is_rocm_bf16 = gpu_compute_capability.IsRocm() &&
                            gpu_compute_capability.rocm_compute_capability()
                                ->has_bf16_dtype_support();

  const bool is_sycl = gpu_compute_capability.IsOneAPI();

  const bool has_nanoo_fp8_support =
      gpu_compute_capability.IsRocm() &&
      gpu_compute_capability.rocm_compute_capability()->has_nanoo_fp8_support();

  switch (algorithm) {
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32:
    case PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM:
      if (!is_cuda_ge_ada && !is_rocm_mi100_and_above) {
        return false;
      }
      if (output_storage_type != BF16 && output_storage_type != F16 &&
          output_storage_type != F32 && output_storage_type != F8E4M3FN &&
          output_storage_type != F8E5M2 && output_storage_type != F8E4M3FNUZ &&
          output_storage_type != F8E5M2FNUZ) {
        return false;
      }
      // Other F8 types are actually not supported by NVIDIA GPUs.
      // Reference: https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
      if (lhs_storage_type == F8E5M2 && rhs_storage_type == F8E4M3FN) {
        return true;
      }
      if (lhs_storage_type == F8E4M3FN &&
          (rhs_storage_type == F8E5M2 || rhs_storage_type == F8E4M3FN)) {
        return true;
      }
      // FNUZ types support (ROCm)
      if (has_nanoo_fp8_support) {
        if (lhs_storage_type == F8E5M2FNUZ && rhs_storage_type == F8E4M3FNUZ) {
          return true;
        }
        if (lhs_storage_type == F8E4M3FNUZ &&
            (rhs_storage_type == F8E5M2FNUZ ||
             rhs_storage_type == F8E4M3FNUZ)) {
          return true;
        }
      }
      return false;
    case PrecisionConfig::ALG_DOT_BF16_BF16_FP8X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_FP8X4:
      // DotAlgorithmRewriter lowers these to F8E4M3FN dots run by cuBLASLt or
      // Triton.
      return is_cuda_ge_ada &&
             (lhs_storage_type == BF16 || lhs_storage_type == F32) &&
             (rhs_storage_type == BF16 || rhs_storage_type == F32) &&
             (output_storage_type == BF16 || output_storage_type == F32);
    case PrecisionConfig::ALG_DOT_F16_F16_F32:
      return lhs_storage_type == rhs_storage_type && lhs_storage_type == F16 &&
             (output_storage_type == F16 || output_storage_type == F32);
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32:
      if (!is_cuda_ge_ampere && !is_rocm_bf16 && !is_sycl) {
        return false;
      }
      if (lhs_storage_type != rhs_storage_type) {
        return false;
      }
      switch (lhs_storage_type) {
        case BF16:
          return output_storage_type == BF16 || output_storage_type == F32;
        case F32:
          return output_storage_type == F32;
        default:
          return false;
      }
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
    case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9:
      return (is_cuda_ge_ampere || is_rocm_bf16 || is_sycl) &&
             lhs_storage_type == rhs_storage_type && lhs_storage_type == F32 &&
             output_storage_type == F32;
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3:
    case PrecisionConfig::ALG_DOT_TF32_TF32_F32:
      return (is_cuda_ge_ampere || is_rocm_mi100_and_above || is_sycl) &&
             lhs_storage_type == rhs_storage_type && lhs_storage_type == F32 &&
             output_storage_type == F32;
    case PrecisionConfig::ALG_DOT_F32_F32_F32:
      return lhs_storage_type == rhs_storage_type && lhs_storage_type == F32 &&
             output_storage_type == F32;
    case PrecisionConfig::ALG_DOT_F64_F64_F64:
      return lhs_storage_type == rhs_storage_type && lhs_storage_type == F64 &&
             output_storage_type == F64;
    default:
      return false;
  }
}

bool IsBf16ToF32AlgorithmRequested(const HloInstruction* instr) {
  return instr->precision_config().algorithm() ==
             PrecisionConfig::ALG_DOT_BF16_BF16_F32 &&
         instr->operand_count() >= 2 &&
         instr->operand(0)->shape().element_type() == F32 &&
         instr->operand(1)->shape().element_type() == F32 &&
         instr->shape().element_type() == F32;
}

namespace {

// Maps each kept non-contracting dimension of a reduced operand scale tensor
// to its destination dimension index in dot_shape, inserting a transpose only
// when the target output dimensions are not strictly increasing.
absl::StatusOr<HloInstruction*> BroadcastScaleToDotOutput(
    HloInstruction* scale, absl::Span<const int64_t> operand_batch_dims,
    absl::Span<const int64_t> operand_contracting_dims, int64_t operand_rank,
    int64_t non_contracting_out_offset, const Shape& dot_shape) {
  if (dot_shape.dimensions().empty()) {
    return scale;
  }
  if (scale->shape().dimensions().empty()) {
    return MakeBroadcastHlo(scale, {}, dot_shape);
  }

  std::vector<int64_t> target_out_dims;
  target_out_dims.reserve(scale->shape().dimensions().size());
  int64_t nc_index = 0;
  for (int64_t d = 0; d < operand_rank; ++d) {
    if (absl::c_linear_search(operand_contracting_dims, d)) {
      continue;
    }
    auto batch_it = absl::c_find(operand_batch_dims, d);
    if (batch_it != operand_batch_dims.end()) {
      target_out_dims.push_back(
          std::distance(operand_batch_dims.begin(), batch_it));
    } else {
      target_out_dims.push_back(non_contracting_out_offset + nc_index);
      ++nc_index;
    }
  }

  if (!absl::c_is_sorted(target_out_dims)) {
    std::vector<int64_t> perm(target_out_dims.size());
    absl::c_iota(perm, 0);
    absl::c_sort(perm, [&](int64_t a, int64_t b) {
      return target_out_dims[a] < target_out_dims[b];
    });
    ABSL_ASSIGN_OR_RETURN(scale, MakeTransposeHlo(scale, perm));
    absl::c_sort(target_out_dims);
  }
  return MakeBroadcastHlo(scale, target_out_dims, dot_shape);
}

}  // namespace

// Rewrites BF16 or F32 dots with ALG_DOT_BF16_BF16_FP8X3/4 into per channel
// power of two scaling, two slice F8E4M3FN decomposition, and 3 or 4 F8E4M3FN
// matmuls accumulated in FP32.
absl::StatusOr<bool> RewriteFp8xNDot(
    HloInstruction* dot, PrecisionConfig::Precision f8_dot_precision) {
  const PrecisionConfig::Algorithm algorithm =
      dot->precision_config().algorithm();
  if (algorithm != PrecisionConfig::ALG_DOT_BF16_BF16_FP8X3 &&
      algorithm != PrecisionConfig::ALG_DOT_BF16_BF16_FP8X4) {
    return false;
  }

  HloInstruction* lhs = dot->mutable_operand(0);
  HloInstruction* rhs = dot->mutable_operand(1);
  const PrimitiveType lhs_type = lhs->shape().element_type();
  const PrimitiveType rhs_type = rhs->shape().element_type();
  if ((lhs_type != BF16 && lhs_type != F32) ||
      (rhs_type != BF16 && rhs_type != F32)) {
    return false;
  }
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions().empty() ||
      dnums.rhs_contracting_dimensions().empty()) {
    return false;
  }

  // Mask for IEEE 754 binary32 biased exponent bits (bits 30..23).
  static constexpr uint32_t kF32ExponentMask = 0x7F800000;
  // Combined biased exponent sum for scale_neg: (127 + 134) << 23 = 0x82800000.
  static constexpr uint32_t kNegScaleBiasedExponentSumBits = 261u << 23;
  // Combined descaling shift factor undoing both 2^7 scales: 2^-14.
  static constexpr float kDescaleShiftFactor = 6.103515625e-5f;
  // Positive floor clamp preventing log2(0) on all zero channels. It also keeps
  // the biased exponent of amax at least 7, so 2^-S stays a normal F32.
  static constexpr float kMinClampF32 = 1e-30f;
  // Binade shift (2^4 = 16.0) elevating residual mantissa bits into normal
  // F8E4M3FN range.
  static constexpr float kLowSliceShiftScale = 16.0f;
  // Scale factor (2^-4 = 0.0625) undoing the 16x low slice shift on first order
  // cross terms.
  static constexpr float kFirstOrderCrossTermScale = 0.0625f;
  // Scale factor (2^-8 = 0.00390625) undoing the 16x shift on both operands for
  // second order p11.
  static constexpr float kSecondOrderResidualTermScale = 0.00390625f;

  HloComputation* comp = dot->parent();

  struct QuantizedOperand {
    HloInstruction* high;
    HloInstruction* low;
    HloInstruction* exponent;
  };

  auto quantize_operand = [&](HloInstruction* op,
                              absl::Span<const int64_t> contracting_dims)
      -> absl::StatusOr<QuantizedOperand> {
    std::vector<int64_t> reduce_dims(contracting_dims.begin(),
                                     contracting_dims.end());
    absl::c_sort(reduce_dims);
    std::vector<int64_t> kept_dims;
    const int64_t op_rank = op->shape().dimensions().size();
    for (int64_t d = 0; d < op_rank; ++d) {
      if (!absl::c_linear_search(reduce_dims, d)) {
        kept_dims.push_back(d);
      }
    }

    HloInstruction* op_f32 = MakeConvertToHlo(op, F32);
    ABSL_ASSIGN_OR_RETURN(HloInstruction * abs_op,
                          MakeUnaryHlo(HloOpcode::kAbs, op_f32));

    // Dynamic exponent shift S = floor(log2(amax)) - 7 centers max magnitude in
    // [128.0, 256.0).
    HloInstruction* zero_f32 = MakeR0ConstantHlo<float>(comp, 0.0f);
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * max_op,
        MakeReduceHlo(abs_op, zero_f32, reduce_dims, HloOpcode::kMaximum));
    ABSL_ASSIGN_OR_RETURN(HloInstruction * max_clamped,
                          MakeBinaryHlo(HloOpcode::kMaximum, max_op,
                                        MakeScalarLike(max_op, kMinClampF32)));

    // Extract IEEE 754 biased exponent field and construct exact power of two
    // scale 2^(-S) via single unsigned subtract (261 << 23) - exp_op.
    HloInstruction* bits_op = MakeBitcastConvertToHlo(max_clamped, U32);
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * exp_op,
        MakeBinaryHlo(HloOpcode::kAnd, bits_op,
                      MakeScalarLike(bits_op, kF32ExponentMask)));
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * neg_s_bits,
        MakeBinaryHlo(HloOpcode::kSubtract,
                      MakeScalarLike(exp_op, kNegScaleBiasedExponentSumBits),
                      exp_op));
    HloInstruction* scale_neg = MakeBitcastConvertToHlo(neg_s_bits, F32);
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * scaled,
        MakeBinaryHlo(HloOpcode::kMultiply, op_f32,
                      MakeBroadcastHlo(scale_neg, kept_dims, op_f32->shape())));

    // High slice x_high: convert scaled value to F8E4M3FN using round to
    // nearest even, then convert back to F32 for residual subtraction.
    HloInstruction* high = MakeConvertToHlo(scaled, F8E4M3FN);
    HloInstruction* high_f32 = MakeConvertToHlo(high, F32);

    // Residual r = x_scaled - x_high cancels top bits exactly by Sterbenz
    // Lemma, leaving signed lower mantissa bits centered around zero.
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * residual_f32,
        MakeBinaryHlo(HloOpcode::kSubtract, scaled, high_f32));

    // Low slice x_low: shift residual up by 4 binades (* 16.0) into normal
    // F8E4M3FN exponent range.
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * low_f32,
        MakeBinaryHlo(HloOpcode::kMultiply, residual_f32,
                      MakeScalarLike(residual_f32, kLowSliceShiftScale)));

    return QuantizedOperand{high, MakeConvertToHlo(low_f32, F8E4M3FN),
                            MakeBitcastConvertToHlo(exp_op, F32)};
  };

  ABSL_ASSIGN_OR_RETURN(
      QuantizedOperand lhs_quant,
      quantize_operand(lhs, dnums.lhs_contracting_dimensions()));
  ABSL_ASSIGN_OR_RETURN(
      QuantizedOperand rhs_quant,
      quantize_operand(rhs, dnums.rhs_contracting_dimensions()));
  auto [a_high, a_low, exp_a_f32] = lhs_quant;
  auto [b_high, b_low, exp_b_f32] = rhs_quant;

  PrecisionConfig f8_pc;
  f8_pc.add_operand_precision(f8_dot_precision);
  f8_pc.add_operand_precision(f8_dot_precision);
  // Keep the dot's layout: on GPU this runs after layout assignment.
  const Shape dot_shape_f32 = ShapeUtil::ChangeElementType(dot->shape(), F32);
  auto make_f8_dot = [&](HloInstruction* a, HloInstruction* b) {
    return comp->AddInstruction(
        HloInstruction::CreateDot(dot_shape_f32, a, b, dnums, f8_pc),
        &dot->metadata());
  };

  // Cartesian cross terms: A * B = A_high * B_high + (A_high * B_low + A_low *
  // B_high) * 2^-4 + (A_low * B_low) * 2^-8.
  HloInstruction* p00 = make_f8_dot(a_high, b_high);
  HloInstruction* p01 = make_f8_dot(a_high, b_low);
  HloInstruction* p10 = make_f8_dot(a_low, b_high);

  // Scale first order cross terms (p01 + p10) by 2^-4 (0.0625) to undo the 16x
  // low slice shift.
  ABSL_ASSIGN_OR_RETURN(HloInstruction * p01_p10_raw,
                        MakeBinaryHlo(HloOpcode::kAdd, p01, p10));
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * p01_p10,
      MakeBinaryHlo(HloOpcode::kMultiply, p01_p10_raw,
                    MakeScalarLike(p01_p10_raw, kFirstOrderCrossTermScale)));

  HloInstruction* sum_f32 = p00;
  if (algorithm == PrecisionConfig::ALG_DOT_BF16_BF16_FP8X4) {
    // Scale second order residual term p11 by 2^-8 (0.00390625) for FP8x4.
    HloInstruction* p11_raw = make_f8_dot(a_low, b_low);
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * p11,
        MakeBinaryHlo(HloOpcode::kMultiply, p11_raw,
                      MakeScalarLike(p11_raw, kSecondOrderResidualTermScale)));
    ABSL_ASSIGN_OR_RETURN(sum_f32,
                          MakeBinaryHlo(HloOpcode::kAdd, sum_f32, p11));
  }
  ABSL_ASSIGN_OR_RETURN(sum_f32,
                        MakeBinaryHlo(HloOpcode::kAdd, sum_f32, p01_p10));

  // Restore true output dynamic range by multiplying accumulated FP32 sum by
  // per channel descale factor 2^(S_A + S_B) = exp_a_f32 * (exp_b_f32 * 2^-14).
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * scale_b,
      MakeBinaryHlo(HloOpcode::kMultiply, exp_b_f32,
                    MakeScalarLike(exp_b_f32, kDescaleShiftFactor)));
  const int64_t lhs_rank = lhs->shape().dimensions().size();
  const int64_t rhs_rank = rhs->shape().dimensions().size();
  const int64_t num_batch_dims = dnums.lhs_batch_dimensions_size();
  const int64_t num_lhs_nc =
      lhs_rank - num_batch_dims - dnums.lhs_contracting_dimensions_size();
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * s_a_out,
      BroadcastScaleToDotOutput(exp_a_f32, dnums.lhs_batch_dimensions(),
                                dnums.lhs_contracting_dimensions(), lhs_rank,
                                /*non_contracting_out_offset=*/num_batch_dims,
                                dot_shape_f32));
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * s_b_out,
      BroadcastScaleToDotOutput(
          scale_b, dnums.rhs_batch_dimensions(),
          dnums.rhs_contracting_dimensions(), rhs_rank,
          /*non_contracting_out_offset=*/num_batch_dims + num_lhs_nc,
          dot_shape_f32));
  ABSL_ASSIGN_OR_RETURN(HloInstruction * descaled_a,
                        MakeBinaryHlo(HloOpcode::kMultiply, sum_f32, s_a_out));
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * descaled_f32,
      MakeBinaryHlo(HloOpcode::kMultiply, descaled_a, s_b_out));
  HloInstruction* final_out =
      dot->shape().element_type() == F32
          ? descaled_f32
          : MakeConvertToHlo(descaled_f32, dot->shape().element_type());
  ABSL_RETURN_IF_ERROR(comp->ReplaceInstruction(dot, final_out));
  return true;
}

}  // namespace algorithm_util

}  // namespace xla
