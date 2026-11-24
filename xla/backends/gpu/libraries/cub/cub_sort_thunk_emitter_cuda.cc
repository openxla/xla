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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
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
#include "xla/backends/gpu/libraries/cub/cub_sort_thunk_emitter.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/shaped_slice.h"
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

emitters::KernelArgument MakeKernelArg(const ShapedSlice& slice, bool written) {
  emitters::KernelArgument arg(slice.shape, slice.slice);
  arg.set_written(written);
  arg.set_invariant(!written);
  return arg;
}

absl::StatusOr<std::unique_ptr<CustomKernelThunk>> MakeCustomKernelThunk(
    const CubSortThunkEmitParams& params, absl::string_view kernel_name,
    se::KernelLoaderSpec loader_spec, se::KernelArgsPackingSpec packing_spec,
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
      params.make_thunk_info(), std::move(custom_kernel), kernel_arguments,
      params.devices_in_process, /*use_pdl=*/false,
      std::move(zeroed_output_buffer_indices));
}

// Computes the max active blocks per SM for the Onesweep histogram kernel
// statically from DeviceDescription + CUBIN ELF attributes (deviceless).
int ComputeHistogramBlocksPerSm(
    const se::DeviceDescription& device_desc,
    const std::optional<se::cuda::CudaKernelFuncAttributes>& attrs,
    int block_threads) {
  const se::CudaComputeCapability cc = device_desc.cuda_compute_capability();
  if (attrs.has_value() && attrs->num_regs > 0) {
    auto try_occ = [&](int major, int minor) -> std::optional<int> {
      if (major <= 0 || device_desc.threads_per_block_limit() <= 0 ||
          device_desc.threads_per_core_limit() <= 0 ||
          device_desc.registers_per_block_limit() <= 0 ||
          device_desc.registers_per_core_limit() <= 0 ||
          device_desc.threads_per_warp() <= 0 ||
          device_desc.shared_memory_per_block() <= 0 ||
          device_desc.shared_memory_per_core() <= 0) {
        return std::nullopt;
      }

      cudaOccDeviceProp prop;
      prop.computeMajor = major;
      prop.computeMinor = minor;
      prop.maxThreadsPerBlock =
          static_cast<int>(device_desc.threads_per_block_limit());
      prop.maxThreadsPerMultiprocessor =
          static_cast<int>(device_desc.threads_per_core_limit());
      prop.regsPerBlock =
          static_cast<int>(device_desc.registers_per_block_limit());
      prop.regsPerMultiprocessor =
          static_cast<int>(device_desc.registers_per_core_limit());
      prop.warpSize = static_cast<int>(device_desc.threads_per_warp());
      prop.sharedMemPerBlock =
          static_cast<size_t>(device_desc.shared_memory_per_block());
      prop.sharedMemPerMultiprocessor =
          static_cast<size_t>(device_desc.shared_memory_per_core());
      prop.numSms = std::max(1, device_desc.core_count());
      // DeviceDescription initializes opt-in / reserved shared memory to -1
      // when unpopulated (e.g. in TestGpuDeviceInfo factories); clamp to 0 to
      // avoid unsigned SIZE_MAX wrap-around inside cuda_occupancy.h.
      prop.sharedMemPerBlockOptin = static_cast<size_t>(
          std::max<int64_t>(0, device_desc.shared_memory_per_block_optin()));
      prop.reservedSharedMemPerBlock = static_cast<size_t>(
          std::max<int64_t>(0, device_desc.reserved_shared_memory_per_block()));

      // On Hopper+ (SM90+), CUBIN ELF `.nv.shared.<kernel>` already includes
      // `reservedSharedMemPerBlock`, whereas `cuda_occupancy.h` adds
      // `properties->reservedSharedMemPerBlock` to
      // `attributes->sharedSizeBytes`. Subtract it so reserved shared memory is
      // not double-counted.
      size_t user_static_smem = attrs->static_shared_size_bytes;
      if (major >= 9 && user_static_smem >= prop.reservedSharedMemPerBlock) {
        user_static_smem -= prop.reservedSharedMemPerBlock;
      }

      cudaOccFuncAttributes func_attrs;
      func_attrs.maxThreadsPerBlock =
          attrs->max_threads_per_block.value_or(prop.maxThreadsPerBlock);
      func_attrs.numRegs = attrs->num_regs;
      func_attrs.sharedSizeBytes = user_static_smem;
      func_attrs.partitionedGCConfig = PARTITIONED_GC_OFF;
      func_attrs.shmemLimitConfig = FUNC_SHMEM_LIMIT_DEFAULT;
      func_attrs.maxDynamicSharedSizeBytes = 0;
      func_attrs.numBlockBarriers = attrs->num_barriers;

      cudaOccDeviceState state;
      cudaOccResult result{};
      if (cudaOccMaxActiveBlocksPerMultiprocessor(
              &result, &prop, &func_attrs, &state, block_threads,
              /*dynamicSmemSize=*/0) == CUDA_OCC_SUCCESS &&
          result.activeBlocksPerMultiprocessor > 0) {
        return result.activeBlocksPerMultiprocessor;
      }
      return std::nullopt;
    };

    if (std::optional<int> occ = try_occ(cc.major, cc.minor); occ.has_value()) {
      return *occ;
    }
    // Retry with the matched CUBIN's compute capability if it differs from
    // `cc`, then fall back to `(cc.major, 0)` for newer minor variants not yet
    // known to cuda_occupancy.h (for example, sm_103 / sm_107 on CUDA 12.8).
    if (attrs->compute_capability.major > 0 &&
        (attrs->compute_capability.major != cc.major ||
         attrs->compute_capability.minor != cc.minor)) {
      if (std::optional<int> occ = try_occ(attrs->compute_capability.major,
                                           attrs->compute_capability.minor);
          occ.has_value()) {
        return *occ;
      }
    }
    if (cc.minor != 0) {
      if (std::optional<int> occ = try_occ(cc.major, 0); occ.has_value()) {
        return *occ;
      }
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

}  // namespace

absl::StatusOr<ThunkSequence> EmitCudaCubSortThunkSequence(
    const CubSortThunkEmitParams& params) {
  TF_RET_CHECK(params.device_description.gpu_compute_capability().IsCuda())
      << "EmitCudaCubSortThunkSequence requires a CUDA device description";

  const bool is_pairs = params.value_type.has_value();
  TF_RET_CHECK(is_pairs == params.values_in.has_value() &&
               is_pairs == params.values_out.has_value())
      << "values_in and values_out must be provided iff value_type is set";

  TF_RET_CHECK(params.keys_in.shape.IsArray());
  TF_RET_CHECK(params.keys_out.shape.IsArray());
  TF_RET_CHECK(params.keys_in.shape.element_type() == params.key_type);
  TF_RET_CHECK(params.keys_out.shape.element_type() == params.key_type);
  const int64_t num_items = ShapeUtil::ElementsIn(params.keys_in.shape);
  TF_RET_CHECK(ShapeUtil::ElementsIn(params.keys_out.shape) == num_items);
  TF_RET_CHECK(params.keys_in.slice.size() >=
               ShapeUtil::ByteSizeOf(params.keys_in.shape));
  TF_RET_CHECK(params.keys_out.slice.size() >=
               ShapeUtil::ByteSizeOf(params.keys_out.shape));
  if (is_pairs) {
    TF_RET_CHECK(params.values_in->shape.IsArray());
    TF_RET_CHECK(params.values_out->shape.IsArray());
    TF_RET_CHECK(params.values_in->shape.element_type() == *params.value_type);
    TF_RET_CHECK(params.values_out->shape.element_type() == *params.value_type);
    TF_RET_CHECK(ShapeUtil::ElementsIn(params.values_in->shape) == num_items);
    TF_RET_CHECK(ShapeUtil::ElementsIn(params.values_out->shape) == num_items);
    TF_RET_CHECK(params.values_in->slice.size() >=
                 ShapeUtil::ByteSizeOf(params.values_in->shape));
    TF_RET_CHECK(params.values_out->slice.size() >=
                 ShapeUtil::ByteSizeOf(params.values_out->shape));
  }

  const se::CudaComputeCapability cc =
      params.device_description.cuda_compute_capability();
  ABSL_ASSIGN_OR_RETURN(
      CubSortExecutionPlan plan,
      PlanCubSortExecution(cc, params.key_type, params.value_type, num_items,
                           params.batch_size, params.portion_size_override,
                           params.max_grid_x_override));

  if (plan.scratch_bytes > 0) {
    TF_RET_CHECK(params.scratch.slice.size() >= plan.scratch_bytes)
        << "Allocated CUB sort scratch buffer (" << params.scratch.slice.size()
        << " B) is smaller than required (" << plan.scratch_bytes << " B)";
  }

  // Output and scratch slices must be disjoint from inputs and from each other
  // (read-only input aliasing such as keys_in == values_in is allowed).
  if (num_items > 0) {
    TF_RET_CHECK(!params.keys_out.slice.OverlapsWith(params.keys_in.slice));
    if (is_pairs) {
      TF_RET_CHECK(
          !params.keys_out.slice.OverlapsWith(params.values_in->slice));
      TF_RET_CHECK(
          !params.keys_out.slice.OverlapsWith(params.values_out->slice));
      TF_RET_CHECK(
          !params.values_out->slice.OverlapsWith(params.keys_in.slice));
      TF_RET_CHECK(
          !params.values_out->slice.OverlapsWith(params.values_in->slice));
    }
    if (plan.scratch_bytes > 0) {
      TF_RET_CHECK(!params.scratch.slice.OverlapsWith(params.keys_in.slice));
      TF_RET_CHECK(!params.scratch.slice.OverlapsWith(params.keys_out.slice));
      if (is_pairs) {
        TF_RET_CHECK(
            !params.scratch.slice.OverlapsWith(params.values_in->slice));
        TF_RET_CHECK(
            !params.scratch.slice.OverlapsWith(params.values_out->slice));
      }
    }
  }

  ThunkSequence thunks;
  const std::string suffix = MakeKernelNameSuffix(
      params.key_type, params.value_type, params.descending);

  switch (plan.mode) {
    case CubSortMode::kEmpty:
      return thunks;

    case CubSortMode::kTrivialCopy: {
      thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
          params.make_thunk_info(), params.keys_in, params.keys_out,
          ShapeUtil::ByteSizeOf(params.keys_in.shape)));
      if (is_pairs) {
        thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
            params.make_thunk_info(), *params.values_in, *params.values_out,
            ShapeUtil::ByteSizeOf(params.values_in->shape)));
      }
      return thunks;
    }

    case CubSortMode::kSingleTile: {
      ABSL_ASSIGN_OR_RETURN(
          se::KernelLoaderSpec base_spec,
          GetCubSortSingleTileKernelSpec(params.key_type, params.value_type,
                                         params.descending, cc));
      const std::string kernel_name =
          absl::StrCat("cub_sort_single_tile", suffix);
      const int32_t segment_size = static_cast<int32_t>(plan.segment_size);
      const int64_t key_byte_size =
          ShapeUtil::ByteSizeOfPrimitiveType(params.key_type);
      const int64_t val_byte_size =
          is_pairs ? ShapeUtil::ByteSizeOfPrimitiveType(*params.value_type) : 0;
      thunks.reserve(plan.single_tile_launches.size());
      for (const CubSortSingleTileLaunch& launch : plan.single_tile_launches) {
        const ShapedSlice chunk_keys_in = MakeSegmentChunkSlice(
            params.keys_in, launch.segment_offset, launch.grid_x,
            plan.batch_size, plan.segment_size, key_byte_size);
        const ShapedSlice chunk_keys_out = MakeSegmentChunkSlice(
            params.keys_out, launch.segment_offset, launch.grid_x,
            plan.batch_size, plan.segment_size, key_byte_size);
        std::vector<emitters::KernelArgument> args;
        se::KernelArgsPackingSpec packing;
        args.push_back(MakeKernelArg(chunk_keys_in, /*written=*/false));
        args.push_back(MakeKernelArg(chunk_keys_out, /*written=*/true));
        packing.AddAddressArgument(0);
        packing.AddAddressArgument(1);
        if (is_pairs) {
          const ShapedSlice chunk_vals_in = MakeSegmentChunkSlice(
              *params.values_in, launch.segment_offset, launch.grid_x,
              plan.batch_size, plan.segment_size, val_byte_size);
          const ShapedSlice chunk_vals_out = MakeSegmentChunkSlice(
              *params.values_out, launch.segment_offset, launch.grid_x,
              plan.batch_size, plan.segment_size, val_byte_size);
          args.push_back(MakeKernelArg(chunk_vals_in, /*written=*/false));
          args.push_back(MakeKernelArg(chunk_vals_out, /*written=*/true));
          packing.AddAddressArgument(2);
          packing.AddAddressArgument(3);
        }
        packing.AddConstantArgument<int32_t>(segment_size);

        ABSL_ASSIGN_OR_RETURN(
            std::unique_ptr<CustomKernelThunk> thunk,
            MakeCustomKernelThunk(params, kernel_name, base_spec,
                                  std::move(packing), std::move(args),
                                  launch.grid_x,
                                  plan.policy.single_tile.block_threads));
        thunks.push_back(std::move(thunk));
      }
      return thunks;
    }

    case CubSortMode::kOnesweep: {
      TF_RET_CHECK(plan.num_portions <= std::numeric_limits<int32_t>::max());
      thunks.reserve(2 + plan.onesweep_launches.size());
      ABSL_ASSIGN_OR_RETURN(
          CubSortOnesweepKernelSpecs specs,
          GetCubSortOnesweepKernelSpecs(params.key_type, params.value_type,
                                        params.descending, cc));
      const ShapedSlice bins_raw =
          MakeScratchSubSlice(params.scratch, plan.bins_raw);
      const ShapedSlice bins_portion0 =
          MakeScratchSubSlice(params.scratch, plan.bins_portion0);
      const ShapedSlice ctrs_all =
          MakeScratchSubSlice(params.scratch, plan.ctrs_all);
      const ShapedSlice keys_tmp =
          plan.keys_tmp.size > 0
              ? MakeScratchSubSlice(params.scratch, plan.keys_tmp)
              : ShapedSlice{};
      const ShapedSlice vals_tmp =
          plan.vals_tmp.size > 0
              ? MakeScratchSubSlice(params.scratch, plan.vals_tmp)
              : ShapedSlice{};

      // 1. Histogram kernel (zeros `bins_raw` before launch).
      const int histo_blocks_per_sm = ComputeHistogramBlocksPerSm(
          params.device_description, specs.histogram_attrs,
          plan.policy.histogram.block_threads);
      const int64_t sm_count =
          std::max(1, params.device_description.core_count());
      const int64_t max_histo_tiles =
          CeilOfRatio<int64_t>(num_items, plan.policy.histogram.tile_items());
      const int32_t histo_grid = static_cast<int32_t>(std::min<int64_t>(
          {static_cast<int64_t>(histo_blocks_per_sm) * sm_count,
           max_histo_tiles, std::numeric_limits<int32_t>::max()}));
      const int32_t key_bits = static_cast<int32_t>(
          ShapeUtil::ByteSizeOfPrimitiveType(params.key_type) * 8);
      {
        std::vector<emitters::KernelArgument> args;
        args.push_back(MakeKernelArg(bins_raw, /*written=*/true));
        args.push_back(MakeKernelArg(params.keys_in, /*written=*/false));
        se::KernelArgsPackingSpec packing;
        packing.AddAddressArgument(0);
        packing.AddAddressArgument(1);
        packing.AddConstantArgument<CubGlobalOffset>(
            static_cast<CubGlobalOffset>(num_items));
        packing.AddConstantArgument<int32_t>(0);
        packing.AddConstantArgument<int32_t>(key_bits);

        ABSL_ASSIGN_OR_RETURN(
            std::unique_ptr<CustomKernelThunk> thunk,
            MakeCustomKernelThunk(
                params, absl::StrCat("cub_sort_histogram", suffix),
                std::move(specs.histogram), std::move(packing), std::move(args),
                histo_grid, plan.policy.histogram.block_threads,
                /*zeroed_output_buffer_indices=*/{0}));
        thunks.push_back(std::move(thunk));
      }

      // 2. Out-of-place exclusive sum over 256 bins per pass, plus
      // zero-initialization of all per-(portion, pass) dynamic tile counters.
      {
        std::vector<emitters::KernelArgument> args;
        args.push_back(MakeKernelArg(bins_raw, /*written=*/false));
        args.push_back(MakeKernelArg(bins_portion0, /*written=*/true));
        args.push_back(MakeKernelArg(ctrs_all, /*written=*/true));
        se::KernelArgsPackingSpec packing;
        packing.AddAddressArgument(0);
        packing.AddAddressArgument(1);
        packing.AddAddressArgument(2);
        packing.AddConstantArgument<int32_t>(
            static_cast<int32_t>(plan.num_portions));

        ABSL_ASSIGN_OR_RETURN(
            std::unique_ptr<CustomKernelThunk> thunk,
            MakeCustomKernelThunk(params, "cub_sort_exclusive_sum",
                                  std::move(specs.exclusive_sum),
                                  std::move(packing), std::move(args),
                                  plan.num_passes,
                                  plan.policy.exclusive_sum.block_threads));
        thunks.push_back(std::move(thunk));
      }

      // 3. Onesweep scatter passes x portions.
      const std::string onesweep_name =
          absl::StrCat("cub_sort_onesweep", suffix);
      for (const CubSortOnesweepPassLaunch& step : plan.onesweep_launches) {
        const ShapedSlice lookback =
            MakeScratchSubSlice(params.scratch, step.lookback);
        const ShapedSlice ctr = MakeScratchSubSlice(params.scratch, step.ctr);
        const ShapedSlice bins_in =
            MakeScratchSubSlice(params.scratch, step.bins_in);
        const ShapedSlice& src_keys = ResolveBufferSlice(
            step.src_role, params.keys_in, params.keys_out, keys_tmp);
        const ShapedSlice& dst_keys = ResolveBufferSlice(
            step.dst_role, params.keys_in, params.keys_out, keys_tmp);

        std::vector<emitters::KernelArgument> args;
        se::KernelArgsPackingSpec packing;

        // Arg 0: d_lookback (written, zeroed before launch).
        args.push_back(MakeKernelArg(lookback, /*written=*/true));
        packing.AddAddressArgument(0);
        // Arg 1: d_ctrs (written; already zeroed once by ExclusiveSum).
        args.push_back(MakeKernelArg(ctr, /*written=*/true));
        packing.AddAddressArgument(1);

        // Arg 2 (device): d_bins_out (next portion's bins or nullptr).
        if (step.bins_out.has_value()) {
          const int idx = static_cast<int>(args.size());
          args.push_back(
              MakeKernelArg(MakeScratchSubSlice(params.scratch, *step.bins_out),
                            /*written=*/true));
          packing.AddAddressArgument(idx);
        } else {
          packing.AddConstantArgument<uint64_t>(0);
        }

        // d_bins_in (read-only).
        {
          const int idx = static_cast<int>(args.size());
          args.push_back(MakeKernelArg(bins_in, /*written=*/false));
          packing.AddAddressArgument(idx);
        }

        // d_keys_out (written) and d_keys_in (read-only).
        {
          const int out_idx = static_cast<int>(args.size());
          args.push_back(MakeKernelArg(dst_keys, /*written=*/true));
          packing.AddAddressArgument(out_idx);
          const int in_idx = static_cast<int>(args.size());
          args.push_back(MakeKernelArg(src_keys, /*written=*/false));
          packing.AddAddressArgument(in_idx);
        }

        // Optional d_values_out (written) and d_values_in (read-only).
        if (is_pairs) {
          const ShapedSlice& src_vals = ResolveBufferSlice(
              step.src_role, *params.values_in, *params.values_out, vals_tmp);
          const ShapedSlice& dst_vals = ResolveBufferSlice(
              step.dst_role, *params.values_in, *params.values_out, vals_tmp);
          const int out_idx = static_cast<int>(args.size());
          args.push_back(MakeKernelArg(dst_vals, /*written=*/true));
          packing.AddAddressArgument(out_idx);
          const int in_idx = static_cast<int>(args.size());
          args.push_back(MakeKernelArg(src_vals, /*written=*/false));
          packing.AddAddressArgument(in_idx);
        }

        packing.AddConstantArgument<int64_t>(step.portion_item_offset);
        packing.AddConstantArgument<int32_t>(step.portion_num_items);
        packing.AddConstantArgument<int32_t>(step.current_bit);
        packing.AddConstantArgument<int32_t>(step.num_bits);

        ABSL_ASSIGN_OR_RETURN(
            std::unique_ptr<CustomKernelThunk> thunk,
            MakeCustomKernelThunk(params, onesweep_name, specs.onesweep,
                                  std::move(packing), std::move(args),
                                  step.grid_x,
                                  plan.policy.onesweep.block_threads,
                                  /*zeroed_output_buffer_indices=*/{0}));
        thunks.push_back(std::move(thunk));
      }
      return thunks;
    }

    case CubSortMode::kSegmentedMultiPass: {
      ABSL_ASSIGN_OR_RETURN(
          CubSortSegmentedKernelSpecs specs,
          GetCubSortSegmentedKernelSpecs(params.key_type, params.value_type,
                                         params.descending, cc));
      const ShapedSlice keys_tmp =
          MakeScratchSubSlice(params.scratch, plan.keys_tmp);
      const ShapedSlice vals_tmp =
          is_pairs ? MakeScratchSubSlice(params.scratch, plan.vals_tmp)
                   : ShapedSlice{};
      const int32_t segment_size = static_cast<int32_t>(plan.segment_size);
      const int64_t key_byte_size =
          ShapeUtil::ByteSizeOfPrimitiveType(params.key_type);
      const int64_t val_byte_size =
          is_pairs ? ShapeUtil::ByteSizeOfPrimitiveType(*params.value_type) : 0;
      const std::string primary_name =
          absl::StrCat("cub_sort_segmented_primary", suffix);
      const std::string alt_name =
          absl::StrCat("cub_sort_segmented_alt", suffix);

      thunks.reserve(plan.segmented_launches.size());
      for (const CubSortSegmentedPassLaunch& step : plan.segmented_launches) {
        const ShapedSlice src_keys = MakeSegmentChunkSlice(
            ResolveBufferSlice(step.src_role, params.keys_in, params.keys_out,
                               keys_tmp),
            step.segment_offset, step.grid_x, plan.batch_size,
            plan.segment_size, key_byte_size);
        const ShapedSlice dst_keys = MakeSegmentChunkSlice(
            ResolveBufferSlice(step.dst_role, params.keys_in, params.keys_out,
                               keys_tmp),
            step.segment_offset, step.grid_x, plan.batch_size,
            plan.segment_size, key_byte_size);

        std::vector<emitters::KernelArgument> args;
        se::KernelArgsPackingSpec packing;
        args.push_back(MakeKernelArg(src_keys, /*written=*/false));
        args.push_back(MakeKernelArg(dst_keys, /*written=*/true));
        packing.AddAddressArgument(0);
        packing.AddAddressArgument(1);
        if (is_pairs) {
          const ShapedSlice src_vals = MakeSegmentChunkSlice(
              ResolveBufferSlice(step.src_role, *params.values_in,
                                 *params.values_out, vals_tmp),
              step.segment_offset, step.grid_x, plan.batch_size,
              plan.segment_size, val_byte_size);
          const ShapedSlice dst_vals = MakeSegmentChunkSlice(
              ResolveBufferSlice(step.dst_role, *params.values_in,
                                 *params.values_out, vals_tmp),
              step.segment_offset, step.grid_x, plan.batch_size,
              plan.segment_size, val_byte_size);
          args.push_back(MakeKernelArg(src_vals, /*written=*/false));
          args.push_back(MakeKernelArg(dst_vals, /*written=*/true));
          packing.AddAddressArgument(2);
          packing.AddAddressArgument(3);
        }
        packing.AddConstantArgument<int32_t>(segment_size);
        packing.AddConstantArgument<int32_t>(step.current_bit);
        packing.AddConstantArgument<int32_t>(step.pass_bits);

        const int block_threads = step.use_alt_policy
                                      ? plan.policy.alt_segmented.block_threads
                                      : plan.policy.segmented.block_threads;
        ABSL_ASSIGN_OR_RETURN(
            std::unique_ptr<CustomKernelThunk> thunk,
            MakeCustomKernelThunk(
                params, step.use_alt_policy ? alt_name : primary_name,
                step.use_alt_policy ? specs.alt : specs.primary,
                std::move(packing), std::move(args), step.grid_x,
                block_threads));
        thunks.push_back(std::move(thunk));
      }
      return thunks;
    }
  }
  return absl::InternalError("Unhandled CubSortMode");
}

}  // namespace xla::gpu
