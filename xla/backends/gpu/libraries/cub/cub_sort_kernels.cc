/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/libraries/cub/cub_sort_kernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_kernels_impl.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_policy.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_elf_utils.h"
#include "xla/stream_executor/cuda/cuda_fatbin.h"
#include "xla/stream_executor/cuda/cudart_kernel_registry.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

absl::StatusOr<internal::KernelHostRef> GetHistogramHostRef(
    PrimitiveType key_type, bool descending) {
  switch (key_type) {
    case BF16:
      return internal::GetCubSortHistogramRef<BF16>(descending);
    case F16:
      return internal::GetCubSortHistogramRef<F16>(descending);
    case F32:
      return internal::GetCubSortHistogramRef<F32>(descending);
    case F64:
      return internal::GetCubSortHistogramRef<F64>(descending);
    case S8:
      return internal::GetCubSortHistogramRef<S8>(descending);
    case S16:
      return internal::GetCubSortHistogramRef<S16>(descending);
    case S32:
      return internal::GetCubSortHistogramRef<S32>(descending);
    case S64:
      return internal::GetCubSortHistogramRef<S64>(descending);
    case U8:
      return internal::GetCubSortHistogramRef<U8>(descending);
    case U16:
      return internal::GetCubSortHistogramRef<U16>(descending);
    case U32:
      return internal::GetCubSortHistogramRef<U32>(descending);
    case U64:
      return internal::GetCubSortHistogramRef<U64>(descending);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported key type for CUB sort: ",
                       primitive_util::LowercasePrimitiveTypeName(key_type)));
  }
}

absl::StatusOr<internal::CubSortTypeKernelRefs> GetKeysHostRefs(
    PrimitiveType key_type, bool descending) {
  switch (key_type) {
    case BF16:
      return internal::GetCubSortKeysRefs<BF16>(descending);
    case F16:
      return internal::GetCubSortKeysRefs<F16>(descending);
    case F32:
      return internal::GetCubSortKeysRefs<F32>(descending);
    case F64:
      return internal::GetCubSortKeysRefs<F64>(descending);
    case S8:
      return internal::GetCubSortKeysRefs<S8>(descending);
    case S16:
      return internal::GetCubSortKeysRefs<S16>(descending);
    case S32:
      return internal::GetCubSortKeysRefs<S32>(descending);
    case S64:
      return internal::GetCubSortKeysRefs<S64>(descending);
    case U8:
      return internal::GetCubSortKeysRefs<U8>(descending);
    case U16:
      return internal::GetCubSortKeysRefs<U16>(descending);
    case U32:
      return internal::GetCubSortKeysRefs<U32>(descending);
    case U64:
      return internal::GetCubSortKeysRefs<U64>(descending);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported key type for CUB sort: ",
                       primitive_util::LowercasePrimitiveTypeName(key_type)));
  }
}

template <PrimitiveType kKeyType>
absl::StatusOr<internal::CubSortTypeKernelRefs> GetPairsHostRefsForKey(
    int value_bit_width, bool descending) {
  switch (value_bit_width) {
    case 16:
      return internal::GetCubSortPairsRefs<kKeyType, 16>(descending);
    case 32:
      return internal::GetCubSortPairsRefs<kKeyType, 32>(descending);
    case 64:
      return internal::GetCubSortPairsRefs<kKeyType, 64>(descending);
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported value bit width for CUB sort pairs: ", value_bit_width));
  }
}

absl::StatusOr<internal::CubSortTypeKernelRefs> GetPairsHostRefs(
    PrimitiveType key_type, int value_bit_width, bool descending) {
  switch (key_type) {
    case U8:
      return GetPairsHostRefsForKey<U8>(value_bit_width, descending);
    case U16:
      return GetPairsHostRefsForKey<U16>(value_bit_width, descending);
    case S32:
      return GetPairsHostRefsForKey<S32>(value_bit_width, descending);
    case U32:
      return GetPairsHostRefsForKey<U32>(value_bit_width, descending);
    case U64:
      return GetPairsHostRefsForKey<U64>(value_bit_width, descending);
    case F32:
      return GetPairsHostRefsForKey<F32>(value_bit_width, descending);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported key type for CUB sort pairs: ",
                       primitive_util::LowercasePrimitiveTypeName(key_type)));
  }
}

absl::StatusOr<internal::CubSortTypeKernelRefs> GetTypeKernelRefs(
    PrimitiveType key_type, std::optional<PrimitiveType> value_type,
    bool descending) {
  if (!value_type.has_value()) {
    return GetKeysHostRefs(key_type, descending);
  }
  if (!primitive_util::IsArrayType(*value_type)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported value type for CUB sort pairs: ",
                     primitive_util::LowercasePrimitiveTypeName(*value_type)));
  }
  return GetPairsHostRefs(key_type, primitive_util::BitWidth(*value_type),
                          descending);
}

// Verifies that `fatbin` contains a JIT-eligible PTX entry for
// `compute_capability` whose highest matching PTX arch maps to the same
// canonical SM policy tier as `CanonicalCubSortPolicySmVersion(cc.major)`.
// Because `XlaChainedPolicy` binds `ActivePolicy` at PTX compile time via
// `CUB_PTX_ARCH`, JITting PTX from a different major policy tier would mismatch
// the host planner's block geometry and cause launch failures or corruption.
absl::Status ValidateFatbinPtxPolicyTier(
    absl::Span<const uint8_t> fatbin,
    const stream_executor::CudaComputeCapability& compute_capability) {
  namespace sec = ::stream_executor::cuda;
  if (compute_capability.major < 5) {
    return absl::FailedPreconditionError(
        absl::StrCat("CUDA CUB sort requires SM >= 5.0, got ",
                     compute_capability.ToString()));
  }
  if (fatbin.size() < sizeof(sec::FatHeader)) {
    return absl::FailedPreconditionError(
        absl::StrCat("No matching CUBIN or PTX fatbinary for ",
                     compute_capability.ToString()));
  }
  sec::FatHeader header;
  std::memcpy(&header, fatbin.data(), sizeof(header));
  if ((header.magic != sec::kFatbinMagicUncompressed &&
       header.magic != sec::kFatbinMagicCompressed &&
       header.magic != sec::kFatbinMagicRelocatable) ||
      header.header_size < sizeof(sec::FatHeader) ||
      header.header_size > fatbin.size()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Kernel binary for ", compute_capability.ToString(),
                     " is not a CUDA fatbinary container with PTX fallback"));
  }

  const size_t header_size = static_cast<size_t>(header.header_size);
  const size_t entries_size = std::min(static_cast<size_t>(header.fat_size),
                                       fatbin.size() - header_size);
  absl::Span<const uint8_t> entries = fatbin.subspan(header_size, entries_size);

  const uint32_t target_arch = static_cast<uint32_t>(
      compute_capability.major * 10 + compute_capability.minor);
  std::optional<uint32_t> best_ptx_arch;
  std::vector<std::string> found_ptx_archs;
  while (entries.size() >= sizeof(sec::FatEntryHeader)) {
    sec::FatEntryHeader entry;
    std::memcpy(&entry, entries.data(), sizeof(entry));
    if (entry.header_size < sizeof(sec::FatEntryHeader) ||
        entry.header_size > entries.size() ||
        entry.size > entries.size() - entry.header_size) {
      break;
    }
    if (entry.kind == sec::kFatbinKindPtx) {
      found_ptx_archs.push_back(absl::StrCat("compute_", entry.arch));
      if (entry.arch <= target_arch &&
          (!best_ptx_arch.has_value() || entry.arch > *best_ptx_arch)) {
        best_ptx_arch = entry.arch;
      }
    }
    const size_t total_size = static_cast<size_t>(entry.header_size) +
                              static_cast<size_t>(entry.size);
    if (total_size == 0) {
      break;
    }
    entries = entries.subspan(total_size);
  }

  if (!best_ptx_arch.has_value()) {
    return absl::FailedPreconditionError(
        absl::StrCat("No matching CUBIN or JIT-compatible PTX for ",
                     compute_capability.ToString(),
                     " in embedded CUB sort fatbinary "
                     "(embedded PTX archs: [",
                     absl::StrJoin(found_ptx_archs, ", "), "])"));
  }

  const int ptx_tier =
      CanonicalCubSortPolicySmVersion(static_cast<int>(*best_ptx_arch / 10));
  const int target_tier =
      CanonicalCubSortPolicySmVersion(compute_capability.major);
  if (ptx_tier != target_tier) {
    return absl::FailedPreconditionError(
        absl::StrCat("Embedded CUB sort fatbinary has no CUBIN for ",
                     compute_capability.ToString(),
                     " and its highest JIT-compatible PTX (compute_",
                     *best_ptx_arch, ", policy tier SM", ptx_tier,
                     ") does not match host policy tier SM", target_tier));
  }
  return absl::OkStatus();
}

absl::StatusOr<stream_executor::KernelLoaderSpec> LoadCubinSpec(
    internal::KernelHostRef ref,
    const stream_executor::CudaComputeCapability& compute_capability) {
  absl::StatusOr<stream_executor::cuda::CudaRuntimeKernel> kernel =
      stream_executor::cuda::FindCudaRuntimeKernel(ref.host_fun,
                                                   compute_capability);
  if (absl::IsNotFound(kernel.status())) {
    // Fall back to the full fatbinary (which embeds PTX for driver JIT) when
    // no precompiled single-arch CUBIN matches `compute_capability` (for
    // example, targeting GB300 sm_103a or sm_120 with compute_100/compute_120
    // PTX), after verifying the JIT-eligible PTX matches the host policy tier.
    ABSL_ASSIGN_OR_RETURN(
        stream_executor::cuda::CudaRuntimeKernel fatbin_kernel,
        stream_executor::cuda::FindCudaRuntimeKernel(ref.host_fun));
    ABSL_RETURN_IF_ERROR(
        ValidateFatbinPtxPolicyTier(fatbin_kernel.cubin, compute_capability));
    kernel = fatbin_kernel;
  }
  ABSL_RETURN_IF_ERROR(kernel.status());
  return stream_executor::KernelLoaderSpec::CreateCudaCubinInMemorySpec(
      kernel->cubin, std::string(kernel->name), ref.arity);
}

absl::StatusOr<std::optional<stream_executor::cuda::CudaKernelFuncAttributes>>
LoadFuncAttributes(
    internal::KernelHostRef ref,
    const stream_executor::CudaComputeCapability& compute_capability) {
  absl::StatusOr<stream_executor::cuda::CudaKernelFuncAttributes> attrs =
      stream_executor::cuda::FindCudaRuntimeKernelFuncAttributes(
          ref.host_fun, compute_capability);
  if (!absl::IsNotFound(attrs.status())) {
    ABSL_RETURN_IF_ERROR(attrs.status());
    return *attrs;
  }
  // Fall back to the minor-0 accelerated CUBIN of the same major architecture
  // (e.g. sm_100a for sm_103a / sm_107) for occupancy attributes.
  const stream_executor::CudaComputeCapability base_accel(
      compute_capability.major, 0,
      stream_executor::CudaComputeCapability::FeatureExtension::
          kAcceleratedFeatures);
  if (compute_capability != base_accel) {
    attrs = stream_executor::cuda::FindCudaRuntimeKernelFuncAttributes(
        ref.host_fun, base_accel);
    if (!absl::IsNotFound(attrs.status())) {
      ABSL_RETURN_IF_ERROR(attrs.status());
      return *attrs;
    }
  }
  // No matching CUBIN ELF in fatbinary; return nullopt so callers use their
  // conservative occupancy fallback.
  return std::nullopt;
}

}  // namespace

absl::StatusOr<stream_executor::KernelLoaderSpec>
GetCubSortSingleTileKernelSpec(
    PrimitiveType key_type, std::optional<PrimitiveType> value_type,
    bool descending,
    const stream_executor::CudaComputeCapability& compute_capability) {
  ABSL_ASSIGN_OR_RETURN(internal::CubSortTypeKernelRefs refs,
                        GetTypeKernelRefs(key_type, value_type, descending));
  return LoadCubinSpec(refs.single_tile, compute_capability);
}

absl::StatusOr<CubSortOnesweepKernelSpecs> GetCubSortOnesweepKernelSpecs(
    PrimitiveType key_type, std::optional<PrimitiveType> value_type,
    bool descending,
    const stream_executor::CudaComputeCapability& compute_capability) {
  ABSL_ASSIGN_OR_RETURN(internal::CubSortTypeKernelRefs refs,
                        GetTypeKernelRefs(key_type, value_type, descending));
  ABSL_ASSIGN_OR_RETURN(internal::KernelHostRef histo_ref,
                        GetHistogramHostRef(key_type, descending));

  ABSL_ASSIGN_OR_RETURN(stream_executor::KernelLoaderSpec histogram,
                        LoadCubinSpec(histo_ref, compute_capability));
  ABSL_ASSIGN_OR_RETURN(
      std::optional<stream_executor::cuda::CudaKernelFuncAttributes>
          histo_attrs,
      LoadFuncAttributes(histo_ref, compute_capability));
  ABSL_ASSIGN_OR_RETURN(
      stream_executor::KernelLoaderSpec exclusive_sum,
      LoadCubinSpec(internal::GetCubSortExclusiveSumRef(), compute_capability));
  ABSL_ASSIGN_OR_RETURN(stream_executor::KernelLoaderSpec onesweep,
                        LoadCubinSpec(refs.onesweep, compute_capability));

  return CubSortOnesweepKernelSpecs{
      /*histogram=*/std::move(histogram),
      /*histogram_attrs=*/histo_attrs,
      /*exclusive_sum=*/std::move(exclusive_sum),
      /*onesweep=*/std::move(onesweep),
  };
}

absl::StatusOr<CubSortSegmentedKernelSpecs> GetCubSortSegmentedKernelSpecs(
    PrimitiveType key_type, std::optional<PrimitiveType> value_type,
    bool descending,
    const stream_executor::CudaComputeCapability& compute_capability) {
  ABSL_ASSIGN_OR_RETURN(internal::CubSortTypeKernelRefs refs,
                        GetTypeKernelRefs(key_type, value_type, descending));

  ABSL_ASSIGN_OR_RETURN(
      stream_executor::KernelLoaderSpec primary,
      LoadCubinSpec(refs.segmented_primary, compute_capability));
  ABSL_ASSIGN_OR_RETURN(stream_executor::KernelLoaderSpec alt,
                        LoadCubinSpec(refs.segmented_alt, compute_capability));

  return CubSortSegmentedKernelSpecs{
      /*primary=*/std::move(primary),
      /*alt=*/std::move(alt),
  };
}

}  // namespace xla::gpu
