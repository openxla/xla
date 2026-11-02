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

#ifndef XLA_PJRT_GPU_SE_GPU_EXECUTABLE_METADATA_PROVIDER_H_
#define XLA_PJRT_GPU_SE_GPU_EXECUTABLE_METADATA_PROVIDER_H_

#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/se/stream_executor_executable.h"
#include "xla/service/compiled_module.h"
#include "xla/service/executable.h"

namespace xla {

// Serves the metadata of `StreamExecutorExecutable`s that wrap XLA:GPU
// executables. `gpu::GpuExecutable`s can build their HLO module on demand, and
// `gpu::GpuAotCompilationResult`s store the module metadata in their proto, so
// neither needs the HLO module up front.
//
// Linking this library registers the provider with
// `StreamExecutorGpuCompiler`. It's linked into the GPU compiler registrations
// only when XLA:GPU is available.
class GpuExecutableMetadataProvider final
    : public StreamExecutorExecutableMetadataProvider {
 public:
  // Returns a shared instance. The provider is stateless.
  static std::shared_ptr<const GpuExecutableMetadataProvider> Get();

  // Returns the module of `executable`, which `gpu::GpuExecutable`s build on
  // demand (see `Executable::GetOrCreateHloModule`).
  absl::StatusOr<std::shared_ptr<HloModule>> GetHloModule(
      const Executable& executable, absl::string_view reason) const override;

  // Reads the metadata of a `gpu::GpuAotCompilationResult` from its proto.
  // Returns std::nullopt for other compiled modules.
  std::optional<CompiledModuleMetadata> GetCompiledModuleMetadata(
      const CompiledModule& compiled_module) const override;

  // Returns the SPMD shardings that a `gpu::GpuExecutable` stores. Returns
  // std::nullopt for other executables.
  std::optional<SpmdShardings> GetSpmdShardings(
      const Executable& executable) const override;
};

}  // namespace xla

#endif  // XLA_PJRT_GPU_SE_GPU_EXECUTABLE_METADATA_PROVIDER_H_
