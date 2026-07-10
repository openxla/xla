/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_FISSION_BACKEND_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_FISSION_BACKEND_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/runtime/object_pool.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

inline autotuner::Backend GetFissionBackend(autotuner::Backend backend) {
  absl::string_view backend_name = autotuner::Backend_Name(backend);
  std::string fission_name = absl::StrCat(backend_name, "_FISSION");
  autotuner::Backend fission_backend;
  if (autotuner::Backend_Parse(fission_name, &fission_backend)) {
    return fission_backend;
  }
  LOG(FATAL) << "Could not parse fission backend name: " << fission_name;
}

// A proxy backend that wraps an actual codegen backend. The `rewriter_pipeline`
// is used to transform unfused instructions to retarget them for the underlying
// codegen backend.
// If multiple supported instructions are found, the first one is profiled, then
// we use its config for the rest of the supported instructions, provided they
// are identical. E.g. three 'dots' resulting from "_X3" or "_X6" algorithms are
// identical.
class FissionBackend : public GpuCodegenBackend {
 public:
  using RewriterPipelineFactory =
      std::function<std::unique_ptr<HloPassPipeline>()>;

  FissionBackend(
      const DebugOptions* debug_options, Compiler* compiler,
      const Compiler::GpuTargetConfig* target_config,
      std::unique_ptr<GpuCodegenBackend> backend,
      RewriterPipelineFactory rewriter_pipeline_factory,
      const AliasInfo* alias_info,
      ObjectPool<std::unique_ptr<mlir::MLIRContext>>* mlir_context_pool,
      stream_executor::StreamExecutor* stream_executor = nullptr)
      : GpuCodegenBackend(GetFissionBackend(backend->backend()), debug_options,
                          compiler, target_config, stream_executor),
        rewriter_pipeline_factory_(std::move(rewriter_pipeline_factory)),
        codegen_backend_(std::move(backend)),
        alias_info_(alias_info),
        mlir_context_pool_(mlir_context_pool) {}
  ~FissionBackend() override = default;

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) override;

  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) override;

  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> hlo_module,
      const Compiler::CompileOptions& options) const override;

  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) const override;

  bool IsSupported(const HloInstruction& instr) const override;
  std::string version() const override { return codegen_backend_->version(); }

 private:
  absl::StatusOr<std::unique_ptr<HloModule>> GetFissionedAndRewrittenModule(
      const HloInstruction& fusion_instr) const;
  absl::StatusOr<std::vector<HloInstruction*>> FindSupportedInstructions(
      const HloModule* module) const;
  // Runs priority fusion to fuse prologues and epilogue after the fissioned
  // module has been generated.
  absl::Status RunPriorityFusion(HloModule* module) const;

  const RewriterPipelineFactory rewriter_pipeline_factory_;
  const std::unique_ptr<GpuCodegenBackend> codegen_backend_;
  const AliasInfo* const alias_info_;
  ObjectPool<std::unique_ptr<mlir::MLIRContext>>* const mlir_context_pool_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_FISSION_BACKEND_H_
