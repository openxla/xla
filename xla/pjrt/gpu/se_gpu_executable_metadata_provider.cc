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

#include "xla/pjrt/gpu/se_gpu_executable_metadata_provider.h"

#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_compiler.h"
#include "xla/service/compiled_module.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_aot_compilation_result.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/stream_executor/platform/initialize.h"

namespace xla {

std::shared_ptr<const GpuExecutableMetadataProvider>
GpuExecutableMetadataProvider::Get() {
  static const auto* const kProvider =
      new std::shared_ptr<const GpuExecutableMetadataProvider>(
          std::make_shared<GpuExecutableMetadataProvider>());
  return *kProvider;
}

absl::StatusOr<std::shared_ptr<HloModule>>
GpuExecutableMetadataProvider::GetHloModule(const Executable& executable,
                                            absl::string_view reason) const {
  return executable.GetOrCreateHloModule(reason);
}

std::optional<StreamExecutorExecutableMetadataProvider::CompiledModuleMetadata>
GpuExecutableMetadataProvider::GetCompiledModuleMetadata(
    const CompiledModule& compiled_module) const {
  const auto* aot_result =
      dynamic_cast<const gpu::GpuAotCompilationResult*>(&compiled_module);
  if (aot_result == nullptr) {
    return std::nullopt;
  }
  const gpu::GpuExecutableProto& proto = aot_result->GetExecutableProto();
  if (!proto.has_hlo_module_with_config()) {
    return std::nullopt;
  }
  // These are the values that the HloModule built from
  // `hlo_module_with_config` returns.
  const HloModuleProtoWithConfig& module_proto = proto.hlo_module_with_config();
  CompiledModuleMetadata metadata;
  metadata.name = module_proto.hlo_module().name();
  metadata.num_replicas = module_proto.config().replica_count();
  metadata.num_partitions = module_proto.config().num_partitions();
  metadata.fingerprint = proto.fingerprint();
  return metadata;
}

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    pjrt_register_se_gpu_executable_metadata_provider, {
      StreamExecutorGpuCompiler::RegisterMetadataProvider(
          GpuExecutableMetadataProvider::Get());
    });

}  // namespace xla
