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

#include "xla/backends/gpu/libraries/cub/cub_sort_thunk_emitter.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/functional/function_ref.h"
#include "absl/functional/overload.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/include/cuda_occupancy.h"
#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_kernels.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_planner.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_policy.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_emitter_context.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/ffi/attributes.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_elf_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel_args_packing_spec.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;

ShapedSlice MakeScratchSubSlice(const ShapedSlice& scratch,
                                const CubSortScratchRange& range) {
  return ShapedSlice{
      BufferAllocation::Slice(scratch.slice.allocation(),
                              scratch.slice.offset() + range.offset,
                              range.size),
      ShapeUtil::MakeShape(U8, {range.size}),
  };
}

ShapedSlice MakeSegmentChunkSlice(const ShapedSlice& base,
                                  int64_t segment_offset, int32_t grid_x,
                                  int64_t batch_size, int64_t segment_size,
                                  int64_t elem_byte_size) {
  if (segment_offset == 0 && grid_x == batch_size) {
    return base;
  }
  const int64_t byte_offset = segment_offset * segment_size * elem_byte_size;
  const int64_t byte_size =
      static_cast<int64_t>(grid_x) * segment_size * elem_byte_size;
  return ShapedSlice{
      BufferAllocation::Slice(base.slice.allocation(),
                              base.slice.offset() + byte_offset, byte_size),
      ShapeUtil::MakeShape(U8, {byte_size}),
  };
}

void AddBufferArg(const ShapedSlice& slice, bool written,
                  std::vector<emitters::KernelArgument>& args,
                  se::KernelArgsPackingSpec& packing) {
  packing.AddAddressArgument(static_cast<int>(args.size()));
  emitters::KernelArgument& arg = args.emplace_back(slice.shape, slice.slice);
  arg.set_written(written);
  arg.set_invariant(!written);
}

absl::StatusOr<std::unique_ptr<CustomKernelThunk>> MakeCustomKernelThunk(
    int devices_in_process,
    absl::FunctionRef<Thunk::ThunkInfo()> make_thunk_info,
    absl::string_view kernel_name, se::KernelLoaderSpec loader_spec,
    se::KernelArgsPackingSpec packing_spec,
    std::vector<emitters::KernelArgument> args, int32_t grid_x,
    int32_t block_threads,
    std::vector<int64_t> zeroed_output_buffer_indices = {}) {
  TF_RET_CHECK(packing_spec.size() == loader_spec.arity())
      << "Packing spec arity (" << packing_spec.size()
      << ") does not match kernel '" << kernel_name << "' arity ("
      << loader_spec.arity() << ")";
  ABSL_RETURN_IF_ERROR(packing_spec.Validate(args.size()));

  loader_spec.set_kernel_args_packing(std::move(packing_spec));
  emitters::KernelArguments kernel_arguments(std::move(args));
  GpuKernelSpec gpu_spec(
      kernel_name,
      LaunchDimensions(se::BlockDim(grid_x), se::ThreadDim(block_threads)),
      kernel_arguments, /*scratch_bytes=*/0);
  CustomKernel custom_kernel(gpu_spec, std::move(loader_spec));
  return std::make_unique<CustomKernelThunk>(
      make_thunk_info(), std::move(custom_kernel), kernel_arguments,
      devices_in_process, /*use_pdl=*/false,
      std::move(zeroed_output_buffer_indices));
}

std::optional<int> TryComputeStaticHistogramBlocksPerSm(
    const se::DeviceDescription& device_desc,
    const se::cuda::CudaKernelFuncAttributes& attrs, int block_threads) {
  if (attrs.num_regs <= 0 || device_desc.threads_per_block_limit() <= 0 ||
      device_desc.threads_per_core_limit() <= 0 ||
      device_desc.registers_per_block_limit() <= 0 ||
      device_desc.registers_per_core_limit() <= 0 ||
      device_desc.threads_per_warp() <= 0 ||
      device_desc.shared_memory_per_block() <= 0 ||
      device_desc.shared_memory_per_core() <= 0) {
    return std::nullopt;
  }

  cudaOccDeviceProp prop;
  prop.maxThreadsPerBlock =
      static_cast<int>(device_desc.threads_per_block_limit());
  prop.maxThreadsPerMultiprocessor =
      static_cast<int>(device_desc.threads_per_core_limit());
  prop.regsPerBlock = static_cast<int>(device_desc.registers_per_block_limit());
  prop.regsPerMultiprocessor =
      static_cast<int>(device_desc.registers_per_core_limit());
  prop.warpSize = static_cast<int>(device_desc.threads_per_warp());
  prop.sharedMemPerBlock =
      static_cast<size_t>(device_desc.shared_memory_per_block());
  prop.sharedMemPerMultiprocessor =
      static_cast<size_t>(device_desc.shared_memory_per_core());
  prop.numSms = std::max(1, device_desc.core_count());
  // DeviceDescription initializes opt-in / reserved shared memory to -1 when
  // unpopulated (e.g. in TestGpuDeviceInfo factories); clamp to 0 to avoid
  // unsigned SIZE_MAX wrap-around inside cuda_occupancy.h.
  prop.sharedMemPerBlockOptin = static_cast<size_t>(
      std::max<int64_t>(0, device_desc.shared_memory_per_block_optin()));
  prop.reservedSharedMemPerBlock = static_cast<size_t>(
      std::max<int64_t>(0, device_desc.reserved_shared_memory_per_block()));

  cudaOccFuncAttributes func_attrs;
  func_attrs.maxThreadsPerBlock =
      attrs.max_threads_per_block.value_or(prop.maxThreadsPerBlock);
  func_attrs.numRegs = attrs.num_regs;
  func_attrs.partitionedGCConfig = PARTITIONED_GC_OFF;
  func_attrs.shmemLimitConfig = FUNC_SHMEM_LIMIT_DEFAULT;
  func_attrs.maxDynamicSharedSizeBytes = 0;
  func_attrs.numBlockBarriers = attrs.num_barriers;

  // Try the target compute capability first, then retry with the matched
  // CUBIN's compute capability if it differs, and finally `(cc.major, 0)` for
  // newer minor variants not yet known to cuda_occupancy.h (for example,
  // sm_103 / sm_107 on CUDA 12.8).
  const se::CudaComputeCapability cc = device_desc.cuda_compute_capability();
  std::array<std::pair<int, int>, 3> candidates;
  size_t num_candidates = 0;
  auto add_candidate = [&](int major, int minor) {
    if (major <= 0) {
      return;
    }
    for (size_t i = 0; i < num_candidates; ++i) {
      if (candidates[i].first == major && candidates[i].second == minor) {
        return;
      }
    }
    candidates[num_candidates++] = {major, minor};
  };
  add_candidate(cc.major, cc.minor);
  add_candidate(attrs.compute_capability.major, attrs.compute_capability.minor);
  add_candidate(cc.major, 0);

  for (size_t i = 0; i < num_candidates; ++i) {
    const auto [major, minor] = candidates[i];
    prop.computeMajor = major;
    prop.computeMinor = minor;

    // On Hopper+ (SM90+), CUBIN ELF `.nv.shared.<kernel>` already includes
    // `reservedSharedMemPerBlock`, whereas `cuda_occupancy.h` adds
    // `properties->reservedSharedMemPerBlock` to `attributes->sharedSizeBytes`.
    // Subtract it so reserved shared memory is not double-counted.
    size_t user_static_smem = attrs.static_shared_size_bytes;
    if (major >= 9 && user_static_smem >= prop.reservedSharedMemPerBlock) {
      user_static_smem -= prop.reservedSharedMemPerBlock;
    }
    func_attrs.sharedSizeBytes = user_static_smem;

    cudaOccDeviceState state;
    cudaOccResult result{};
    if (cudaOccMaxActiveBlocksPerMultiprocessor(
            &result, &prop, &func_attrs, &state, block_threads,
            /*dynamicSmemSize=*/0) == CUDA_OCC_SUCCESS &&
        result.activeBlocksPerMultiprocessor > 0) {
      return result.activeBlocksPerMultiprocessor;
    }
  }
  return std::nullopt;
}

// Computes the max active blocks per SM for the Onesweep histogram kernel
// statically from DeviceDescription + CUBIN ELF attributes (deviceless).
int ComputeHistogramBlocksPerSm(
    const se::DeviceDescription& device_desc,
    const std::optional<se::cuda::CudaKernelFuncAttributes>& attrs,
    int block_threads) {
  if (attrs.has_value()) {
    if (std::optional<int> occ = TryComputeStaticHistogramBlocksPerSm(
            device_desc, *attrs, block_threads);
        occ.has_value()) {
      return *occ;
    }
  }
  // Conservative fallback for PTX-only targets or incomplete synthetic test
  // DeviceDescriptions.
  if (device_desc.threads_per_core_limit() > 0 && block_threads > 0) {
    int by_threads = std::max(
        1,
        static_cast<int>(device_desc.threads_per_core_limit() / block_threads));
    if (device_desc.max_blocks_per_multiprocessor() > 0) {
      by_threads = std::min(
          by_threads,
          static_cast<int>(device_desc.max_blocks_per_multiprocessor()));
    }
    return std::min(by_threads, 4);
  }
  return 1;
}

const ShapedSlice& ResolveBufferSlice(CubSortBufferRole role,
                                      const ShapedSlice& in,
                                      const ShapedSlice& out,
                                      const ShapedSlice& tmp) {
  switch (role) {
    case CubSortBufferRole::kIn:
      return in;
    case CubSortBufferRole::kOut:
      return out;
    case CubSortBufferRole::kTmp:
      return tmp;
  }
  ABSL_UNREACHABLE();
}

std::string MakeKernelNameSuffix(PrimitiveType key_type,
                                 std::optional<PrimitiveType> value_type,
                                 bool descending) {
  std::string suffix =
      absl::StrCat("_", primitive_util::LowercasePrimitiveTypeName(key_type));
  if (value_type.has_value()) {
    absl::StrAppend(&suffix, "_b", primitive_util::BitWidth(*value_type));
  }
  absl::StrAppend(&suffix, descending ? "_desc" : "_asc");
  return suffix;
}

absl::Status ValidateSortShape(const Shape& shape, absl::string_view label) {
  if (!shape.IsArray()) {
    return absl::InvalidArgumentError(
        absl::StrCat("CUB sort ", label, " shape must be an array, got: ",
                     ShapeUtil::HumanStringWithLayout(shape)));
  }
  if (shape.is_dynamic()) {
    return absl::InvalidArgumentError(
        absl::StrCat("CUB sort does not support dynamic ", label,
                     " shape: ", ShapeUtil::HumanStringWithLayout(shape)));
  }
  if (LayoutUtil::HasLayout(shape) &&
      !LayoutUtil::IsMonotonicWithDim0Major(
          ShapeUtil::DropDegenerateDimensions(shape).layout())) {
    return absl::InvalidArgumentError(
        absl::StrCat("CUB sort requires dim-0-major layout for ", label,
                     ", got: ", ShapeUtil::HumanStringWithLayout(shape)));
  }
  return absl::OkStatus();
}

absl::Status ValidateBufferPair(const CubSortBufferPair& pair,
                                int64_t expected_num_items,
                                absl::string_view label) {
  ABSL_RETURN_IF_ERROR(
      ValidateSortShape(pair.in.shape, absl::StrCat(label, ".in")));
  ABSL_RETURN_IF_ERROR(
      ValidateSortShape(pair.out.shape, absl::StrCat(label, ".out")));
  if (pair.in.shape.element_type() != pair.out.shape.element_type()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "CUB sort ", label, " input/output element type mismatch: ",
        PrimitiveType_Name(pair.in.shape.element_type()), " vs ",
        PrimitiveType_Name(pair.out.shape.element_type())));
  }
  if (ShapeUtil::ElementsIn(pair.in.shape) != expected_num_items ||
      ShapeUtil::ElementsIn(pair.out.shape) != expected_num_items) {
    return absl::InvalidArgumentError(absl::StrCat(
        "CUB sort ", label, " element count mismatch: expected ",
        expected_num_items, ", got in=", ShapeUtil::ElementsIn(pair.in.shape),
        " out=", ShapeUtil::ElementsIn(pair.out.shape)));
  }
  if (pair.in.slice.size() < ShapeUtil::ByteSizeOf(pair.in.shape)) {
    return absl::InvalidArgumentError(
        absl::StrCat("CUB sort ", label, ".in slice (", pair.in.slice.size(),
                     " B) is smaller than shape size (",
                     ShapeUtil::ByteSizeOf(pair.in.shape), " B)"));
  }
  if (pair.out.slice.size() < ShapeUtil::ByteSizeOf(pair.out.shape)) {
    return absl::InvalidArgumentError(
        absl::StrCat("CUB sort ", label, ".out slice (", pair.out.slice.size(),
                     " B) is smaller than shape size (",
                     ShapeUtil::ByteSizeOf(pair.out.shape), " B)"));
  }
  return absl::OkStatus();
}

absl::Status CheckDisjoint(const BufferAllocation::Slice& a,
                           absl::string_view a_name,
                           const BufferAllocation::Slice& b,
                           absl::string_view b_name) {
  if (a.OverlapsWith(b)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "CUB sort buffers ", a_name, " and ", b_name, " must not overlap"));
  }
  return absl::OkStatus();
}

absl::StatusOr<CubSortExecutionPlan> ValidateAndPlan(
    const se::DeviceDescription& device_description,
    const CubSortThunkEmitParams& params,
    const CubSortPlanOptions& plan_options) {
  if (!device_description.gpu_compute_capability().IsCuda()) {
    return absl::FailedPreconditionError(
        "EmitCudaCubSortThunkSequence requires a CUDA device description");
  }

  ABSL_RETURN_IF_ERROR(ValidateSortShape(params.keys.in.shape, "keys.in"));
  const int64_t num_items = ShapeUtil::ElementsIn(params.keys.in.shape);
  ABSL_RETURN_IF_ERROR(ValidateBufferPair(params.keys, num_items, "keys"));

  const PrimitiveType key_type = params.keys.in.shape.element_type();
  std::optional<PrimitiveType> value_type;
  if (params.values.has_value()) {
    ABSL_RETURN_IF_ERROR(
        ValidateBufferPair(*params.values, num_items, "values"));
    value_type = params.values->in.shape.element_type();
  }

  const se::CudaComputeCapability cc =
      device_description.cuda_compute_capability();
  ABSL_ASSIGN_OR_RETURN(
      CubSortExecutionPlan plan,
      PlanCubSortExecution(cc, key_type, value_type, num_items,
                           params.batch_size, plan_options));

  if (plan.scratch_bytes > 0 &&
      params.scratch.slice.size() < plan.scratch_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Allocated CUB sort scratch buffer (", params.scratch.slice.size(),
        " B) is smaller than required (", plan.scratch_bytes, " B)"));
  }

  // Output and scratch slices must be disjoint from inputs and from each other
  // (read-only input aliasing such as keys.in == values->in is allowed).
  ABSL_RETURN_IF_ERROR(CheckDisjoint(params.keys.out.slice, "keys.out",
                                     params.keys.in.slice, "keys.in"));
  if (params.values.has_value()) {
    ABSL_RETURN_IF_ERROR(CheckDisjoint(params.keys.out.slice, "keys.out",
                                       params.values->in.slice, "values.in"));
    ABSL_RETURN_IF_ERROR(CheckDisjoint(params.keys.out.slice, "keys.out",
                                       params.values->out.slice, "values.out"));
    ABSL_RETURN_IF_ERROR(CheckDisjoint(params.values->out.slice, "values.out",
                                       params.keys.in.slice, "keys.in"));
    ABSL_RETURN_IF_ERROR(CheckDisjoint(params.values->out.slice, "values.out",
                                       params.values->in.slice, "values.in"));
  }
  if (plan.scratch_bytes > 0) {
    ABSL_RETURN_IF_ERROR(CheckDisjoint(params.scratch.slice, "scratch",
                                       params.keys.in.slice, "keys.in"));
    ABSL_RETURN_IF_ERROR(CheckDisjoint(params.scratch.slice, "scratch",
                                       params.keys.out.slice, "keys.out"));
    if (params.values.has_value()) {
      ABSL_RETURN_IF_ERROR(CheckDisjoint(params.scratch.slice, "scratch",
                                         params.values->in.slice, "values.in"));
      ABSL_RETURN_IF_ERROR(CheckDisjoint(params.scratch.slice, "scratch",
                                         params.values->out.slice,
                                         "values.out"));
    }
  }

  return plan;
}

absl::StatusOr<ThunkSequence> EmitSingleTileThunks(
    const CubSortThunkEmitParams& params, const CubSortExecutionPlan& plan,
    se::CudaComputeCapability cc, absl::string_view suffix,
    absl::FunctionRef<Thunk::ThunkInfo()> make_thunk_info,
    const CubSortSingleTilePlan& mode_plan) {
  ABSL_ASSIGN_OR_RETURN(
      se::KernelLoaderSpec base_spec,
      GetCubSortSingleTileKernelSpec(plan.key_type, plan.value_type,
                                     params.descending, cc));
  const std::string kernel_name = absl::StrCat("cub_sort_single_tile", suffix);
  const int32_t segment_size = static_cast<int32_t>(plan.segment_size);
  const int64_t key_byte_size =
      ShapeUtil::ByteSizeOfPrimitiveType(plan.key_type);
  const int64_t val_byte_size =
      plan.value_type.has_value()
          ? ShapeUtil::ByteSizeOfPrimitiveType(*plan.value_type)
          : 0;

  ThunkSequence thunks;
  thunks.reserve(mode_plan.launches.size());
  for (const CubSortSingleTileLaunch& launch : mode_plan.launches) {
    const ShapedSlice chunk_keys_in = MakeSegmentChunkSlice(
        params.keys.in, launch.segment_offset, launch.grid_x, plan.batch_size,
        plan.segment_size, key_byte_size);
    const ShapedSlice chunk_keys_out = MakeSegmentChunkSlice(
        params.keys.out, launch.segment_offset, launch.grid_x, plan.batch_size,
        plan.segment_size, key_byte_size);
    std::vector<emitters::KernelArgument> args;
    se::KernelArgsPackingSpec packing;
    AddBufferArg(chunk_keys_in, /*written=*/false, args, packing);
    AddBufferArg(chunk_keys_out, /*written=*/true, args, packing);
    if (params.values.has_value()) {
      const ShapedSlice chunk_vals_in = MakeSegmentChunkSlice(
          params.values->in, launch.segment_offset, launch.grid_x,
          plan.batch_size, plan.segment_size, val_byte_size);
      const ShapedSlice chunk_vals_out = MakeSegmentChunkSlice(
          params.values->out, launch.segment_offset, launch.grid_x,
          plan.batch_size, plan.segment_size, val_byte_size);
      AddBufferArg(chunk_vals_in, /*written=*/false, args, packing);
      AddBufferArg(chunk_vals_out, /*written=*/true, args, packing);
    }
    packing.AddConstantArgument<int32_t>(segment_size);

    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<CustomKernelThunk> thunk,
        MakeCustomKernelThunk(params.devices_in_process, make_thunk_info,
                              kernel_name, base_spec, std::move(packing),
                              std::move(args), launch.grid_x,
                              plan.policy.single_tile.block_threads));
    thunks.push_back(std::move(thunk));
  }
  return thunks;
}

absl::StatusOr<ThunkSequence> EmitOnesweepThunks(
    const se::DeviceDescription& device_description,
    const CubSortThunkEmitParams& params, const CubSortExecutionPlan& plan,
    se::CudaComputeCapability cc, absl::string_view suffix,
    absl::FunctionRef<Thunk::ThunkInfo()> make_thunk_info,
    const CubSortOnesweepPlan& mode_plan) {
  TF_RET_CHECK(mode_plan.num_portions <= std::numeric_limits<int32_t>::max());
  ABSL_ASSIGN_OR_RETURN(
      CubSortOnesweepKernelSpecs specs,
      GetCubSortOnesweepKernelSpecs(plan.key_type, plan.value_type,
                                    params.descending, cc));

  const ShapedSlice bins_raw =
      MakeScratchSubSlice(params.scratch, mode_plan.bins_raw);
  const ShapedSlice bins_portion0 =
      MakeScratchSubSlice(params.scratch, mode_plan.bins_portion0);
  const ShapedSlice counters_all =
      MakeScratchSubSlice(params.scratch, mode_plan.ctrs_all);
  const ShapedSlice keys_tmp =
      mode_plan.keys_tmp.size > 0
          ? MakeScratchSubSlice(params.scratch, mode_plan.keys_tmp)
          : ShapedSlice{};
  const ShapedSlice values_tmp =
      mode_plan.vals_tmp.size > 0
          ? MakeScratchSubSlice(params.scratch, mode_plan.vals_tmp)
          : ShapedSlice{};

  ThunkSequence thunks;
  thunks.reserve(2 + mode_plan.launches.size());

  // 1. Histogram kernel (zeros `bins_raw` before launch).
  const int histogram_blocks_per_sm =
      ComputeHistogramBlocksPerSm(device_description, specs.histogram_attrs,
                                  plan.policy.histogram.block_threads);
  const int64_t sm_count = std::max(1, device_description.core_count());
  const int64_t max_histogram_tiles =
      CeilOfRatio<int64_t>(plan.num_items, plan.policy.histogram.tile_items());
  const int32_t histogram_grid = static_cast<int32_t>(std::min<int64_t>(
      {static_cast<int64_t>(histogram_blocks_per_sm) * sm_count,
       max_histogram_tiles, std::numeric_limits<int32_t>::max()}));
  const int32_t key_bits = static_cast<int32_t>(
      ShapeUtil::ByteSizeOfPrimitiveType(plan.key_type) * 8);
  {
    std::vector<emitters::KernelArgument> args;
    se::KernelArgsPackingSpec packing;
    AddBufferArg(bins_raw, /*written=*/true, args, packing);
    AddBufferArg(params.keys.in, /*written=*/false, args, packing);
    packing.AddConstantArgument<CubGlobalOffset>(
        static_cast<CubGlobalOffset>(plan.num_items));
    packing.AddConstantArgument<int32_t>(0);
    packing.AddConstantArgument<int32_t>(key_bits);

    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<CustomKernelThunk> thunk,
        MakeCustomKernelThunk(params.devices_in_process, make_thunk_info,
                              absl::StrCat("cub_sort_histogram", suffix),
                              std::move(specs.histogram), std::move(packing),
                              std::move(args), histogram_grid,
                              plan.policy.histogram.block_threads,
                              /*zeroed_output_buffer_indices=*/{0}));
    thunks.push_back(std::move(thunk));
  }

  // 2. Out-of-place exclusive sum over 256 bins per pass, plus
  // zero-initialization of all per-(portion, pass) dynamic tile counters.
  {
    std::vector<emitters::KernelArgument> args;
    se::KernelArgsPackingSpec packing;
    AddBufferArg(bins_raw, /*written=*/false, args, packing);
    AddBufferArg(bins_portion0, /*written=*/true, args, packing);
    AddBufferArg(counters_all, /*written=*/true, args, packing);
    packing.AddConstantArgument<int32_t>(
        static_cast<int32_t>(mode_plan.num_portions));

    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<CustomKernelThunk> thunk,
        MakeCustomKernelThunk(
            params.devices_in_process, make_thunk_info,
            "cub_sort_exclusive_sum", std::move(specs.exclusive_sum),
            std::move(packing), std::move(args), mode_plan.num_passes,
            plan.policy.exclusive_sum.block_threads));
    thunks.push_back(std::move(thunk));
  }

  // 3. Onesweep scatter passes x portions.
  const std::string onesweep_name = absl::StrCat("cub_sort_onesweep", suffix);
  for (const CubSortOnesweepPassLaunch& step : mode_plan.launches) {
    const ShapedSlice lookback =
        MakeScratchSubSlice(params.scratch, step.lookback);
    const ShapedSlice counter = MakeScratchSubSlice(params.scratch, step.ctr);
    const ShapedSlice bins_in =
        MakeScratchSubSlice(params.scratch, step.bins_in);
    const ShapedSlice& src_keys = ResolveBufferSlice(
        step.src_role, params.keys.in, params.keys.out, keys_tmp);
    const ShapedSlice& dst_keys = ResolveBufferSlice(
        step.dst_role, params.keys.in, params.keys.out, keys_tmp);

    std::vector<emitters::KernelArgument> args;
    se::KernelArgsPackingSpec packing;

    // Arg 0: lookback (written, zeroed before launch).
    AddBufferArg(lookback, /*written=*/true, args, packing);
    // Arg 1: counters (written; already zeroed once by ExclusiveSum).
    AddBufferArg(counter, /*written=*/true, args, packing);

    // Arg 2 (device): bins_out (next portion's bins or nullptr).
    if (step.bins_out.has_value()) {
      AddBufferArg(MakeScratchSubSlice(params.scratch, *step.bins_out),
                   /*written=*/true, args, packing);
    } else {
      packing.AddConstantArgument<uint64_t>(0);
    }

    // bins_in (read-only), keys_out (written), keys_in (read-only).
    AddBufferArg(bins_in, /*written=*/false, args, packing);
    AddBufferArg(dst_keys, /*written=*/true, args, packing);
    AddBufferArg(src_keys, /*written=*/false, args, packing);

    // Optional values_out (written) and values_in (read-only).
    if (params.values.has_value()) {
      const ShapedSlice& src_values = ResolveBufferSlice(
          step.src_role, params.values->in, params.values->out, values_tmp);
      const ShapedSlice& dst_values = ResolveBufferSlice(
          step.dst_role, params.values->in, params.values->out, values_tmp);
      AddBufferArg(dst_values, /*written=*/true, args, packing);
      AddBufferArg(src_values, /*written=*/false, args, packing);
    }

    packing.AddConstantArgument<int64_t>(step.portion_item_offset);
    packing.AddConstantArgument<int32_t>(step.portion_num_items);
    packing.AddConstantArgument<int32_t>(step.current_bit);
    packing.AddConstantArgument<int32_t>(step.num_bits);

    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<CustomKernelThunk> thunk,
        MakeCustomKernelThunk(params.devices_in_process, make_thunk_info,
                              onesweep_name, specs.onesweep, std::move(packing),
                              std::move(args), step.grid_x,
                              plan.policy.onesweep.block_threads,
                              /*zeroed_output_buffer_indices=*/{0}));
    thunks.push_back(std::move(thunk));
  }
  return thunks;
}

absl::StatusOr<ThunkSequence> EmitSegmentedThunks(
    const CubSortThunkEmitParams& params, const CubSortExecutionPlan& plan,
    se::CudaComputeCapability cc, absl::string_view suffix,
    absl::FunctionRef<Thunk::ThunkInfo()> make_thunk_info,
    const CubSortSegmentedMultiPassPlan& mode_plan) {
  ABSL_ASSIGN_OR_RETURN(
      CubSortSegmentedKernelSpecs specs,
      GetCubSortSegmentedKernelSpecs(plan.key_type, plan.value_type,
                                     params.descending, cc));
  const ShapedSlice keys_tmp =
      MakeScratchSubSlice(params.scratch, mode_plan.keys_tmp);
  const ShapedSlice vals_tmp =
      params.values.has_value()
          ? MakeScratchSubSlice(params.scratch, mode_plan.vals_tmp)
          : ShapedSlice{};
  const int32_t segment_size = static_cast<int32_t>(plan.segment_size);
  const int64_t key_byte_size =
      ShapeUtil::ByteSizeOfPrimitiveType(plan.key_type);
  const int64_t val_byte_size =
      plan.value_type.has_value()
          ? ShapeUtil::ByteSizeOfPrimitiveType(*plan.value_type)
          : 0;
  const std::string primary_name =
      absl::StrCat("cub_sort_segmented_primary", suffix);
  const std::string alt_name = absl::StrCat("cub_sort_segmented_alt", suffix);

  ThunkSequence thunks;
  thunks.reserve(mode_plan.launches.size());
  for (const CubSortSegmentedPassLaunch& step : mode_plan.launches) {
    const ShapedSlice src_keys =
        MakeSegmentChunkSlice(ResolveBufferSlice(step.src_role, params.keys.in,
                                                 params.keys.out, keys_tmp),
                              step.segment_offset, step.grid_x, plan.batch_size,
                              plan.segment_size, key_byte_size);
    const ShapedSlice dst_keys =
        MakeSegmentChunkSlice(ResolveBufferSlice(step.dst_role, params.keys.in,
                                                 params.keys.out, keys_tmp),
                              step.segment_offset, step.grid_x, plan.batch_size,
                              plan.segment_size, key_byte_size);

    std::vector<emitters::KernelArgument> args;
    se::KernelArgsPackingSpec packing;
    AddBufferArg(src_keys, /*written=*/false, args, packing);
    AddBufferArg(dst_keys, /*written=*/true, args, packing);
    if (params.values.has_value()) {
      const ShapedSlice src_vals = MakeSegmentChunkSlice(
          ResolveBufferSlice(step.src_role, params.values->in,
                             params.values->out, vals_tmp),
          step.segment_offset, step.grid_x, plan.batch_size, plan.segment_size,
          val_byte_size);
      const ShapedSlice dst_vals = MakeSegmentChunkSlice(
          ResolveBufferSlice(step.dst_role, params.values->in,
                             params.values->out, vals_tmp),
          step.segment_offset, step.grid_x, plan.batch_size, plan.segment_size,
          val_byte_size);
      AddBufferArg(src_vals, /*written=*/false, args, packing);
      AddBufferArg(dst_vals, /*written=*/true, args, packing);
    }
    packing.AddConstantArgument<int32_t>(segment_size);
    packing.AddConstantArgument<int32_t>(step.current_bit);
    packing.AddConstantArgument<int32_t>(step.pass_bits);

    const int block_threads = step.use_alt_policy
                                  ? plan.policy.alt_segmented.block_threads
                                  : plan.policy.segmented.block_threads;
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<CustomKernelThunk> thunk,
        MakeCustomKernelThunk(params.devices_in_process, make_thunk_info,
                              step.use_alt_policy ? alt_name : primary_name,
                              step.use_alt_policy ? specs.alt : specs.primary,
                              std::move(packing), std::move(args), step.grid_x,
                              block_threads));
    thunks.push_back(std::move(thunk));
  }
  return thunks;
}

}  // namespace

absl::StatusOr<ThunkSequence> EmitCudaCubSortThunkSequence(
    const se::DeviceDescription& device_description,
    const CubSortThunkEmitParams& params,
    absl::FunctionRef<Thunk::ThunkInfo()> make_thunk_info,
    CubSortPlanOptions plan_options) {
  ABSL_ASSIGN_OR_RETURN(
      CubSortExecutionPlan plan,
      ValidateAndPlan(device_description, params, plan_options));
  const se::CudaComputeCapability cc =
      device_description.cuda_compute_capability();
  const std::string suffix =
      MakeKernelNameSuffix(plan.key_type, plan.value_type, params.descending);

  return std::visit(
      absl::Overload{
          [&](const CubSortSingleTilePlan& mode_plan) {
            return EmitSingleTileThunks(params, plan, cc, suffix,
                                        make_thunk_info, mode_plan);
          },
          [&](const CubSortOnesweepPlan& mode_plan) {
            return EmitOnesweepThunks(device_description, params, plan, cc,
                                      suffix, make_thunk_info, mode_plan);
          },
          [&](const CubSortSegmentedMultiPassPlan& mode_plan) {
            return EmitSegmentedThunks(params, plan, cc, suffix,
                                       make_thunk_info, mode_plan);
          },
      },
      plan.mode_plan);
}

absl::StatusOr<ThunkSequence> EmitCudaCubSortNativeCustomCall(
    const HloCustomCallInstruction& instr,
    const NativeCustomCallEmitterContext& ctx) {
  const bool is_pairs =
      instr.custom_call_target() == kCubDeviceRadixSortPairsTarget;
  if (!is_pairs &&
      instr.custom_call_target() != kCubDeviceRadixSortKeysTarget) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected custom call target for CUB sort handler: ",
                     instr.custom_call_target()));
  }
  if (instr.operand_count() != (is_pairs ? 2 : 1)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected operand count for CUB sort custom call: ",
                     instr.ToString()));
  }
  if (!instr.shape().IsTuple() ||
      instr.shape().tuple_shapes().size() != (is_pairs ? 3 : 2)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected result shape for CUB sort custom call: ",
                     instr.ToString()));
  }

  ABSL_ASSIGN_OR_RETURN(xla::ffi::Attributes attrs, ctx.GetFfiAttributes());
  ABSL_ASSIGN_OR_RETURN(bool descending, attrs.Get<bool>("descending"));
  ABSL_ASSIGN_OR_RETURN(int64_t batch_size, attrs.Get<int64_t>("batch_size"));

  // Pair emitter-resolved allocation slices with the instruction's logical HLO
  // shapes (GetOperandShapedSlice would return the defining buffer's shape,
  // which can differ across bitcast/bitcast-convert).
  const HloInstruction* keys_operand = instr.operand(0);
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice keys_in_slice,
                        ctx.GetOperandAllocationSlice(0, {}));
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice keys_out_slice,
                        ctx.GetResultAllocationSlice({0}));
  CubSortBufferPair keys{
      .in = ShapedSlice{keys_in_slice, keys_operand->shape()},
      .out = ShapedSlice{keys_out_slice, instr.shape().tuple_shapes(0)},
  };
  if (keys.in.shape.element_type() != keys.out.shape.element_type()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Mismatched key element types for CUB sort custom call: ",
                     instr.ToString()));
  }

  std::optional<CubSortBufferPair> values;
  if (is_pairs) {
    const HloInstruction* values_operand = instr.operand(1);
    ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice values_in_slice,
                          ctx.GetOperandAllocationSlice(1, {}));
    ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice values_out_slice,
                          ctx.GetResultAllocationSlice({1}));
    values = CubSortBufferPair{
        .in = ShapedSlice{values_in_slice, values_operand->shape()},
        .out = ShapedSlice{values_out_slice, instr.shape().tuple_shapes(1)},
    };
    if (values->in.shape.element_type() != values->out.shape.element_type()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Mismatched value element types for CUB sort custom call: ",
          instr.ToString()));
    }
  }

  const int64_t scratch_index = is_pairs ? 2 : 1;
  ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice scratch_slice,
                        ctx.GetResultAllocationSlice({scratch_index}));
  ShapedSlice scratch{scratch_slice, instr.shape().tuple_shapes(scratch_index)};
  if (scratch.shape.element_type() != U8) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected U8 scratch buffer for CUB sort custom call: ",
                     instr.ToString()));
  }

  CubSortThunkEmitParams emit_params{
      .devices_in_process = ctx.GetTargetTopology().num_devices_per_process(),
      .descending = descending,
      .batch_size = batch_size,
      .keys = keys,
      .values = values,
      .scratch = scratch,
  };
  return EmitCudaCubSortThunkSequence(
      ctx.GetDeviceDescription(), emit_params,
      [&]() { return ctx.GenerateThunkInfo(); });
}

}  // namespace xla::gpu
