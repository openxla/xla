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

#include "xla/service/gpu/autotuning/gemm_fusion_autotuner.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/stream_executor/rocm/rocm_blas.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/primitive_util.h"
#include "xla/service/algorithm_util.h"
#include "xla/service/call_inliner.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/autotuner_compile_util.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/buffer_comparator.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion_pattern.h"
#include "xla/service/gpu/matmul_indexing_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/split_k_gemm_rewriter.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/transforms/custom_kernel_fusion_rewriter.h"
#include "xla/service/gpu/transforms/dot_algorithm_rewriter.h"
#include "xla/service/gpu/transforms/fusion_wrapper.h"
#include "xla/service/gpu/transforms/gemm_rewriter.h"
#include "xla/service/gpu/transforms/priority_fusion.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/lib/core/bits.h"
#include "xla/tsl/util/proto/proto_utils.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tsl/profiler/lib/scoped_annotation.h"


namespace xla {
namespace gpu {

const int64_t BLAS_GEMM_DEFAULT = HIPBLAS_GEMM_DEFAULT;

using BackendConfig = GemmFusionAutotunerImpl::BackendConfig;
using BackendConfigs = GemmFusionAutotunerImpl::BackendConfigs;
using ProfilingOutput = AutotunerCompileUtil::ProfilingOutput;

// Default tiling when autotuning is disabled.
constexpr TritonGemmConfig kDefaultGemmTiling = {32, 32, 32, 1, 1, 4};

// Split-K is enabled when the estimate number of waves is lower than the limit.
constexpr int kMaxWavesForSplitK = 5;

// Search space for exhaustive matmul autotuning.
constexpr std::array<int, 6> kBlockSizes = {16, 32, 64, 128, 256, 512};
constexpr std::array<int, 4> kNumStages = {1, 2, 3, 4};
constexpr std::array<int, 4> kNumWarps = {2, 4, 8, 16};
constexpr std::array<int, 5> kSplitK = {1, 2, 4, 8, 16};
constexpr std::array<int, 5> kNumCtas = {1, 2, 4, 8, 16};

std::vector<TritonGemmConfig>
GemmFusionAutotunerImpl::GetExhaustiveTritonConfigs() const {
  std::vector<TritonGemmConfig> configs;
  se::GpuComputeCapability cc = GetComputeCapability();

  for (int num_stages : kNumStages) {
    for (int tile_m : kBlockSizes) {
      for (int tile_n : kBlockSizes) {
        for (int tile_k : kBlockSizes) {
          const int tile_lhs = tile_m * tile_k;
          const int tile_rhs = tile_k * tile_n;
          for (int num_warps : kNumWarps) {
            // Each thread should read at least one input element.
            if (num_warps * WarpSize() > std::min(tile_lhs, tile_rhs)) {
              break;
            }
            for (int split_k : kSplitK) {
              // Split-K autotuning may be disabled by a flag.
              if (!debug_options_.xla_gpu_enable_split_k_autotuning() &&
                  split_k > 1) {
                break;
              }
            }
          }
        }
      }
    }
  }
  return configs;
}

std::vector<TritonGemmConfig> GemmFusionAutotunerImpl::GetDefaultTritonConfigs()
    const {
  using Config = TritonGemmConfig;
  std::vector<Config> configs = {
        Config(32, 32, 256, 1, 1, 4), Config(64, 32, 32, 16, 1, 4),
        Config(32, 64, 64, 4, 1, 4),  Config(128, 128, 64, 4, 1, 4),
        Config(16, 16, 256, 1, 1, 4), Config(16, 128, 32, 16, 1, 4),
  };
  return configs;
}

absl::StatusOr<std::vector<BackendConfig>>
GemmFusionAutotunerImpl::GenerateConfigs(const HloFusionInstruction& fusion) {
  const HloDotInstruction* dot =
      Cast<HloDotInstruction>(hlo_query::GetFirstInstructionWithOpcode(
          *fusion.called_computations().at(0), HloOpcode::kDot));
  std::vector<BackendConfig> configs;

  if (!debug_options_.xla_gpu_experimental_disable_binary_libraries()) {
    // Add cuBLAS reference config, if available.
    if (algorithm_util::IsSupportedByCublasOrCublasLt(
            dot->precision_config().algorithm(), GetComputeCapability()) &&
        !dot->sparse_operands() && IsAutotuningEnabled()) {
      configs.push_back(CuBlasConfig{});
    }
  }

  // Add CustomKernelFusion (Cutlass) configs, if available.
  // Go through all the instructions in the fusion body try to match them to
  // a custom kernel fusion pattern.
  if ((IsFusionKind(fusion, kCustomFusionKind) ||
       IsFusionKind(fusion, kTritonGemmFusionKind)) &&
      IsAutotuningEnabled() && !config_.IsDeviceless()) {
    std::vector<BackendConfig> custom_kernel_fusion_configs =
        GenerateCustomKernelFusionConfigs(
            fusion, config_.GetExecutor()->GetDeviceDescription());
    configs.insert(configs.end(), custom_kernel_fusion_configs.begin(),
                   custom_kernel_fusion_configs.end());
  }

  // Add triton configs.
  TF_ASSIGN_OR_RETURN(std::vector<TritonGemmConfig> triton_configs,
                      GenerateTritonConfigs(*dot));
  for (TritonGemmConfig& config : triton_configs) {
    configs.push_back(std::move(config));
  }
  return configs;
}

absl::StatusOr<std::vector<TritonGemmConfig>>
GemmFusionAutotunerImpl::GenerateTritonConfigs(const HloDotInstruction& dot) {
  // Retrieve the minimum bit-width participating in the dot. This is needed
  // to avoid autotuning configurations that are not supported by Triton. This
  // is used to restrict the values for tile_k.
  std::vector<const HloInstruction*> converts =
      HloBfsFindAll({&dot}, [&](const HloInstruction* node) {
        return node->opcode() == HloOpcode::kConvert;
      });
  int minBitWidth = primitive_util::BitWidth(dot.shape().element_type());
  for (auto convert : converts) {
    auto in_type = convert->operand(0)->shape().element_type();
    auto out_type = convert->shape().element_type();
    minBitWidth = std::min({minBitWidth, primitive_util::BitWidth(in_type),
                            primitive_util::BitWidth(out_type)});
  }

  std::vector<TritonGemmConfig> result_configs;
  TF_ASSIGN_OR_RETURN(TileSizeLimit limits, GetLimits(dot));

  // Generate the list of configurations (once).
  if (triton_configs_.empty()) {
    triton_configs_ = !IsAutotuningEnabled()
                          ? std::vector(1, kDefaultGemmTiling)
                      : debug_options_.xla_gpu_exhaustive_tiling_search()
                          ? GetExhaustiveTritonConfigs()
                          : GetDefaultTritonConfigs();
  }

  // Avoid autotuning tiny fusions.
  constexpr int kMinGemmElements = 32 * 32;
  bool small_dot =
      ShapeUtil::ElementsIn(dot.operand(0)->shape()) <= kMinGemmElements &&
      ShapeUtil::ElementsIn(dot.operand(1)->shape()) <= kMinGemmElements;
  std::vector<TritonGemmConfig> triton_configs =
      small_dot ? std::vector(1, kDefaultGemmTiling) : triton_configs_;

  // Split-K optimization enables more even utilization of a GPU in cases
  // where tiling just the non-contracting dimensions of a GEMM does not create
  // a sufficient number of thread block programs to occupy all available cores.
  // Around 5 full waves completely avoid the need for split-K.
  // n_tiles = split_k * (M * N) / (block_m * block_n)
  const int kCoreCount =
      !config_.IsDeviceless()
          ? config_.GetExecutor()->GetDeviceDescription().core_count()
          : 100;  // some sensible default
  const int64_t kSufficientNumberOfTiles = kMaxWavesForSplitK * kCoreCount;
  const int64_t result_size = ShapeUtil::ElementsIn(dot.shape());

  // Triton configurations are adjusted and deduplicated.
  absl::flat_hash_set<TritonGemmConfig> added;
  for (TritonGemmConfig& config : triton_configs) {
    config.block_m = std::min(config.block_m, limits.block_m);
    config.block_n = std::min(config.block_n, limits.block_n);
    config.block_k = std::min(config.block_k, limits.block_k);
    int max_split_k = 1;
    if (debug_options_.xla_gpu_enable_split_k_autotuning()) {
      int64_t ratio = kSufficientNumberOfTiles * config.block_m *
                      config.block_n / result_size;
      max_split_k = 1 << std::max<int>(tsl::Log2Floor64(ratio), 0);
    }
    config.split_k = std::min(config.split_k, max_split_k);

    // TODO(b/337839570): Triton currently has a limitation where it crashes
    // on small block_k values depending on the bit-width of the inputs to the
    // dot. The logic below accounts for this limitation.
    constexpr int kLdmatrixGranularity = 256;
    if (config.block_k < kLdmatrixGranularity / minBitWidth) {
      config.block_k = kLdmatrixGranularity / minBitWidth;
    }

    // Sparse meta should have at least one element per thread.
    // Note: only 2:4 structured sparsity is currently supported.
    if (dot.sparse_operands()) {
      config.block_m = std::max(config.block_m, 64);
      config.num_warps = std::max(config.num_warps, 4);
      config.block_k = std::max(
          config.block_k,
          2 * std::max(kMinTileSize, kLdmatrixGranularity / minBitWidth));
      int meta_elements = config.block_m * config.block_k / 16;
      config.num_warps =
          std::min<int>(config.num_warps, meta_elements / WarpSize());
    }

    if (added.insert(config).second) {
      result_configs.push_back(config);
    }
  }
  return result_configs;
}

}
}
