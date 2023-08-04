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

#include "xla/service/gpu/hlo_emitter_parameters_autotuner.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "third_party/protobuf/io/coded_stream.h"
#include "third_party/protobuf/io/zero_copy_stream_impl_lite.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuner_compile_util.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tsl/util/proto/proto_utils.h"

namespace xla::gpu {

namespace {

using ProfilingOutput = AutotunerCompileUtil::ProfilingOutput;

const Shape& GetElementShape(const HloInstruction* fusion) {
  const Shape* shape = &fusion->shape();
  while (shape->IsTuple()) {
    shape = &shape->tuple_shapes(0);
  }
  return *shape;
}

constexpr std::array<int, 5> kUnrollFactors = {1, 2, 4, 8, 16};

// Returns a list of possible unroll factors for a loop fusion.
std::vector<AutotuneResult::LoopFusionKey> GetPossibleLoopFusionAutotuneConfigs(
    const HloInstruction* fusion) {
  std::vector<AutotuneResult::LoopFusionKey> configs;
  configs.reserve(kUnrollFactors.size());
  for (int unroll_factor : kUnrollFactors) {
    int64_t num_elements = ShapeUtil::ElementsIn(GetElementShape(fusion));
    if (num_elements % unroll_factor != 0) {
      continue;
    }
    AutotuneResult::LoopFusionKey key;
    key.set_unroll_factor(unroll_factor);
    configs.push_back(key);
  }
  return configs;
}

class CompilationKey {
 public:
  explicit CompilationKey(AutotuneResult::LoopFusionKey key) {
    proto2::io::StringOutputStream string_stream(&serialized_proto_);
    proto2::io::CodedOutputStream coded_stream(&string_stream);
    coded_stream.SetSerializationDeterministic(true);
    CHECK(key.SerializeToCodedStream(&coded_stream));
  }

  template <typename H>
  friend H AbslHashValue(H h, const CompilationKey& k) {
    return H::combine(std::move(h), k.serialized_proto_);
  }

  bool operator==(const CompilationKey& k) const {
    return serialized_proto_ == k.serialized_proto_;
  }

 private:
  std::string serialized_proto_;
};

class HloEmitterParametersAutotunerVisitor : public DfsHloRewriteVisitor {
 public:
  HloEmitterParametersAutotunerVisitor(
      const AutotuneConfig& config, tsl::thread::ThreadPool* thread_pool,
      std::optional<AutotunerCompileUtil> autotuner_compile_util)
      : config_(config),
        thread_pool_(thread_pool),
        autotuner_compile_util_(autotuner_compile_util) {}

  Status HandleFusion(HloInstruction* hlo) override {
    se::StreamExecutor* stream_exec = config_.GetExecutor();
    GpuDeviceInfo gpu_device_info = GetGpuDeviceInfo(stream_exec);
    TF_ASSIGN_OR_RETURN(
        auto fusion_analysis,
        HloFusionAnalysis::Create(
            Cast<HloFusionInstruction>(hlo), &gpu_device_info,
            stream_exec->GetDeviceDescription().cuda_compute_capability()));
    if (fusion_analysis.GetEmitterFusionKind() !=
        HloFusionAnalysis::EmitterFusionKind::kLoop) {
      return OkStatus();
    }

    VLOG(1) << "Processing " << hlo->ToString();
    TF_ASSIGN_OR_RETURN(AutotuneResult autotune_result,
                        AutotunerUtil::Autotune(hlo, config_, [&] {
                          return AutotuneLoopFusionNoCache(hlo);
                        }));
    VLOG(1) << "Result: " << autotune_result.ShortDebugString();

    TF_RET_CHECK(autotune_result.has_loop_fusion() &&
                 autotune_result.loop_fusion().unroll_factor() > 0);

    TF_ASSIGN_OR_RETURN(auto backend_config,
                        hlo->backend_config<FusionBackendConfig>());

    *backend_config.mutable_loop_fusion_config() =
        autotune_result.loop_fusion();
    TF_RETURN_IF_ERROR(hlo->set_backend_config(backend_config));

    MarkAsChanged();
    return OkStatus();
  }

 private:
  // Autotunes a loop fusion without using the autotuning cache.
  StatusOr<AutotuneResult> AutotuneLoopFusionNoCache(
      const HloInstruction* instr) {
    if (config_.IsDeviceless()) {
      return InternalError(
          "Expect autotune result cache hit for deviceless compilation.");
    }

    const HloComputation& fusion = *instr->fused_instructions_computation();
    se::StreamExecutor* stream_exec = config_.GetExecutor();
    se::DeviceMemoryAllocator* allocator = config_.GetAllocator();
    if (allocator == nullptr) {
      allocator = stream_exec->GetAllocator();
    }

    TF_ASSIGN_OR_RETURN(se::Stream* const stream,
                        allocator->GetStream(stream_exec->device_ordinal()));

    const DebugOptions& debug_opts = fusion.parent()->config().debug_options();

    TF_ASSIGN_OR_RETURN(
        se::RedzoneAllocator rz_allocator,
        AutotunerUtil::CreateRedzoneAllocator(config_, debug_opts));

    const std::vector<AutotuneResult::LoopFusionKey> configurations =
        GetPossibleLoopFusionAutotuneConfigs(instr);

    absl::Mutex executables_mutex;
    absl::flat_hash_map<CompilationKey, std::unique_ptr<Executable>>
        executables;

    auto compile = [&](const AutotuneResult::LoopFusionKey& conf) {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                          autotuner_compile_util_->Compile([&] {
                            return ExtractLoopFusionForAutotuning(conf, fusion);
                          }));
      absl::MutexLock lock(&executables_mutex);
      CHECK(executables.emplace(CompilationKey(conf), std::move(executable))
                .second);
      return OkStatus();
    };

    // Pre-compile all versions first.
    if (thread_pool_ &&
        debug_opts.xla_gpu_force_compilation_parallelism() != 1) {
      tsl::BlockingCounter counter(configurations.size());
      for (const AutotuneResult::LoopFusionKey& conf : configurations) {
        thread_pool_->Schedule([&, conf] {
          TF_CHECK_OK(compile(conf));
          counter.DecrementCount();
        });
      }
      counter.Wait();
    } else {
      for (const AutotuneResult::LoopFusionKey& conf : configurations) {
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

    if (!stream_exec->SynchronizeAllActivity()) {
      return InternalError("Failed to synchronize GPU for autotuning.");
    }
    std::vector<AutotuneResult> results;
    for (const AutotuneResult::LoopFusionKey& conf : configurations) {
      VLOG(1) << "Trying unroll factor: " << conf.ShortDebugString();

      auto it = executables.find(CompilationKey(conf));
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

      auto& res = results.emplace_back();
      *res.mutable_loop_fusion() = conf;
      *res.mutable_run_time() =
          tsl::proto_utils::ToDurationProto(profiling_output->duration);
    }

    const HloInstruction& root = *fusion.root_instruction();
    TF_ASSIGN_OR_RETURN(
        AutotuneResult best,
        PickBestResult(results, root.ToString(), root.GetModule()->config()));

    return best;
  }

  StatusOr<std::unique_ptr<HloModule>> ExtractLoopFusionForAutotuning(
      const AutotuneResult::LoopFusionKey& key,
      const HloComputation& original_computation) {
    std::unique_ptr<HloModule> new_module =
        AutotunerUtil::ExtractInstructionIntoNewModule(
            *original_computation.FusionInstruction());
    HloComputation* entry_computation = new_module->entry_computation();
    HloInstruction* cloned_loop_fusion = entry_computation->root_instruction();

    TF_ASSIGN_OR_RETURN(
        auto backend_config,
        cloned_loop_fusion->backend_config<FusionBackendConfig>());
    *backend_config.mutable_loop_fusion_config() = key;
    TF_RETURN_IF_ERROR(cloned_loop_fusion->set_backend_config(backend_config));

    return new_module;
  }

  AutotuneConfig config_;
  tsl::thread::ThreadPool* thread_pool_;
  std::optional<AutotunerCompileUtil> autotuner_compile_util_;
};

}  // anonymous namespace

StatusOr<bool> HloEmitterParametersAutotuner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_SCOPED_LOGGING_TIMER("HLO emitter parameters autotuner");
  if (!module->config().debug_options().xla_gpu_unroll_factor_autotune()) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(
      std::optional<AutotunerCompileUtil> autotuner_compile_util,
      AutotunerCompileUtil::Create(config_, module->config().debug_options()));
  return HloEmitterParametersAutotunerVisitor{config_, thread_pool_,
                                              autotuner_compile_util}
      .RunOnModule(module, execution_threads);
}

}  // namespace xla::gpu
