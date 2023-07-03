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

#include "xla/service/gpu/row_reduction_autotuner.h"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/autotuner_compile_util.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/gpu_asm_opts_util.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tsl/util/proto/proto_utils.h"

namespace xla {
namespace gpu {

using ProfilingOutput = AutotunerCompileUtil::ProfilingOutput;

namespace {

// Constructs an autotuning key for a row reduction.
static AutotuneResult::RowReductionKey RowReductionKey(int64_t tile_x) {
  AutotuneResult::RowReductionKey key;
  key.set_tile_x(tile_x);
  return key;
}

struct CompilationKey {
  template <typename H>
  friend H AbslHashValue(H h, const CompilationKey& k) {
    return H::combine(std::move(h), k.key.SerializeAsString());
  }

  bool operator==(const CompilationKey& k) const {
    return key.SerializeAsString() == k.key.SerializeAsString();
  }

  AutotuneResult::RowReductionKey key;
};

class RowReductionAutotunerVisitor : public DfsHloRewriteVisitor {
 public:
  RowReductionAutotunerVisitor(
      const AutotuneConfig& config, tsl::thread::ThreadPool* thread_pool,
      std::optional<AutotunerCompileUtil> autotuner_compile_util)
      : config_(config),
        thread_pool_(thread_pool),
        autotuner_compile_util_(autotuner_compile_util) {}

  Status HandleFusion(HloInstruction* hlo) override {
    HloComputation* fused_computation = hlo->called_computations()[0];
    if (!HasAnyUnnestedReductionRoot(fused_computation)) {
      return OkStatus();
    }

    auto roots = GetFusionRoots(fused_computation);
    HloInstruction* reduce =
        *absl::c_find_if(roots, [&](const HloInstruction* instr) {
          return IsReductionFromOrToContiguousDimensions(*instr);
        });
    ReductionDimensions reduction_dimensions =
        GetReductionKindAndContiguousComponents(*reduce);
    if (!reduction_dimensions.is_row_reduction) {
      return OkStatus();
    }
    TF_ASSIGN_OR_RETURN(auto backend_config,
                        hlo->backend_config<FusionBackendConfig>());

    VLOG(1) << "Tuning " << hlo->ToString();
    TF_ASSIGN_OR_RETURN(AutotuneResult autotune_result,
                        AutotunerUtil::Autotune(hlo, config_, [&] {
                          return AutotuneRowReductionNoCache(hlo);
                        }));
    VLOG(1) << "Result: " << autotune_result.ShortDebugString();

    TF_RET_CHECK(autotune_result.has_row_reduction());
    AutotuneResult::RowReductionKey tiling = autotune_result.row_reduction();

    *backend_config.mutable_row_reduction_config() = tiling;
    TF_RETURN_IF_ERROR(hlo->set_backend_config(backend_config));
    MarkAsChanged();
    return OkStatus();
  }

 private:
  // Autotunes a matmul without using the autotuning cache.
  StatusOr<AutotuneResult> AutotuneRowReductionNoCache(
      const HloInstruction* instr) {
    const HloComputation& fusion = *instr->called_computations()[0];
    se::StreamExecutor* stream_exec = config_.GetExecutor();
    if (!stream_exec->SynchronizeAllActivity()) {
      return InternalError("Failed to synchronize GPU for autotuning.");
    }
    se::DeviceMemoryAllocator* allocator = config_.GetAllocator();
    if (allocator == nullptr) {
      allocator = stream_exec->GetAllocator();
    }

    HloInstruction* root = fusion.root_instruction();
    TF_ASSIGN_OR_RETURN(se::Stream* const stream,
                        allocator->GetStream(stream_exec->device_ordinal()));

    const DebugOptions debug_opts = fusion.parent()->config().debug_options();

    se::RedzoneAllocator rz_allocator(
        stream, allocator, PtxOptsFromDebugOptions(debug_opts),
        /*memory_limit=*/std::numeric_limits<int64_t>::max(),
        /*redzone_size=*/0);

    const std::vector<AutotuneResult::RowReductionKey> configurations =
        GetPossibleRowReductionAutotuneConfigs();

    absl::Mutex executables_mutex;
    absl::flat_hash_map<CompilationKey, std::unique_ptr<Executable>>
        executables;

    auto compile = [&](const AutotuneResult::RowReductionKey& conf) {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                          autotuner_compile_util_->Compile([&] {
                            return RowReductionAutotuneExtractor(conf, fusion);
                          }));
      absl::MutexLock lock(&executables_mutex);
      CHECK(executables.emplace(CompilationKey{conf}, std::move(executable))
                .second);
      return OkStatus();
    };

    // Pre-compile all versions first using the thread pool.
    if (thread_pool_ &&
        debug_opts.xla_gpu_force_compilation_parallelism() != 1) {
      tsl::BlockingCounter counter(configurations.size());
      for (const AutotuneResult::RowReductionKey& conf : configurations) {
        thread_pool_->Schedule([&] {
          TF_CHECK_OK(compile(conf));
          counter.DecrementCount();
        });
      }
      counter.Wait();
    } else {
      for (const AutotuneResult::RowReductionKey& conf : configurations) {
        TF_RETURN_IF_ERROR(compile(conf));
      }
    }

    std::vector<se::DeviceMemoryBase> inputs;
    int64_t rng_state = 0;
    for (const HloInstruction* param : fusion.parameter_instructions()) {
      TF_ASSIGN_OR_RETURN(
          se::DeviceMemoryBase param_buffer,
          AutotunerUtil::CreateBuffer(rz_allocator, param->shape(), config_,
                                      rng_state));
      inputs.push_back(param_buffer);
    }

    std::vector<AutotuneResult> results;
    for (const AutotuneResult::RowReductionKey& conf : configurations) {
      VLOG(1) << "Trying reduction tiling: " << conf.ShortDebugString();

      AutotuneResult res;
      *res.mutable_row_reduction() = conf;

      auto it = executables.find(CompilationKey{conf});
      if (it == executables.end() || it->second == nullptr) {
        VLOG(1) << "Skipping this tiling.";
        continue;
      }
      TF_ASSIGN_OR_RETURN(std::optional<ProfilingOutput> profiling_output,
                          autotuner_compile_util_->ProfileExecutable(
                              it->second.get(), stream, inputs));

      if (!profiling_output) {
        VLOG(1) << "Skipping this tiling.";
        continue;
      }

      VLOG(1) << "Running the kernel took: " << profiling_output->duration;
      *res.mutable_run_time() =
          tsl::proto_utils::ToDurationProto(profiling_output->duration);

      results.push_back(res);
    }

    TF_ASSIGN_OR_RETURN(
        AutotuneResult best,
        PickBestResult(results, root->ToString(), root->GetModule()->config()));
    return best;
  }

  StatusOr<std::unique_ptr<HloModule>> RowReductionAutotuneExtractor(
      const AutotuneResult::RowReductionKey& key,
      const HloComputation& original_computation) {
    std::unique_ptr<HloModule> new_module =
        AutotunerUtil::ExtractInstructionIntoNewModule(
            *original_computation.FusionInstruction());
    HloComputation* entry_computation = new_module->entry_computation();
    HloInstruction* cloned_reduce_fusion =
        entry_computation->root_instruction();

    std::vector<HloInstruction*> roots =
        GetFusionRoots(cloned_reduce_fusion->called_computations()[0]);
    HloInstruction* reduction =
        FindHeroReduction(absl::Span<HloInstruction*>(roots));

    ReductionDimensions dims =
        GetReductionKindAndContiguousComponents(*reduction);

    TF_ASSIGN_OR_RETURN(
        auto backend_config,
        cloned_reduce_fusion->backend_config<FusionBackendConfig>());
    *backend_config.mutable_row_reduction_config() = key;
    TF_RETURN_IF_ERROR(
        cloned_reduce_fusion->set_backend_config(backend_config));

    if (ReductionIsRaceFree(new_module->config(), dims) &&
        !ReductionIsRaceFree(new_module->config(), dims, backend_config)) {
      Status s = absl::CancelledError("Smaller tiling forces atomics");
      s.SetPayload(kUncompilableFusion, absl::Cord("Tiling too small"));
      return s;
    }

    return new_module;
  }

  AutotuneConfig config_;
  tsl::thread::ThreadPool* thread_pool_;
  std::optional<AutotunerCompileUtil> autotuner_compile_util_;
};

constexpr std::array<int, 4> TILE_X = {2, 4, 8, 16};

}  // anonymous namespace

std::vector<AutotuneResult::RowReductionKey>
GetPossibleRowReductionAutotuneConfigs() {
  std::vector<AutotuneResult::RowReductionKey> configs;
  configs.reserve(TILE_X.size());
  for (int tile_x : TILE_X) {
    configs.push_back(RowReductionKey(tile_x));
  }
  return configs;
}

StatusOr<bool> RowReductionAutotuner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (module->config().debug_options().xla_gpu_autotune_level() == 0) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(
      std::optional<AutotunerCompileUtil> autotuner_compile_util,
      AutotunerCompileUtil::Create(config_, module->config().debug_options()));

  auto res = RowReductionAutotunerVisitor{config_, thread_pool_,
                                          autotuner_compile_util}
                 .RunOnModule(module, execution_threads);
  return res;
}

}  // namespace gpu
}  // namespace xla
