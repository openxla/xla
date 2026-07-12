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

#include "xla/backends/gpu/libraries/cutedsl/module.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "CuteDSLRuntime.h"

namespace xla::gpu::cutedsl {
namespace {

std::string FormatRuntimeError(CuteDSLRT_Error_t error) {
  const char* name = CuteDSLRT_GetErrorName(error);
  const char* description = CuteDSLRT_GetErrorString(error);
  return absl::StrFormat(
      "%s (error %d): %s", name == nullptr ? "Unknown" : name,
      static_cast<int>(error),
      description == nullptr ? "Unknown error" : description);
}

}  // namespace

LoadedModule::~LoadedModule() {
  if (module_ == nullptr) return;
  CuteDSLRT_Error_t error =
      CuteDSLRT_Module_Destroy(static_cast<CuteDSLRT_Module_t*>(module_));
  if (error != CuteDSLRT_Error_Success) {
    LOG(ERROR) << "Failed to destroy CuTeDSL runtime module: "
               << FormatRuntimeError(error);
  }
}

absl::StatusOr<std::shared_ptr<LoadedModule>> LoadedModule::Create(
    absl::string_view module_bytes) {
  CuteDSLRT_Module_t* module = nullptr;
  // Standalone static and shared runtimes register their CUDA helper symbols
  // directly with ORC. shared_libs is reserved for actual dependencies of the
  // generated module.
  CuteDSLRT_Error_t error = CuteDSLRT_Module_Create_From_Bytes(
      &module, reinterpret_cast<const unsigned char*>(module_bytes.data()),
      module_bytes.size(), /*shared_libs=*/nullptr,
      /*shared_libs_size=*/0);
  if (error != CuteDSLRT_Error_Success) {
    if (module != nullptr) {
      CuteDSLRT_Error_t destroy_error = CuteDSLRT_Module_Destroy(module);
      if (destroy_error != CuteDSLRT_Error_Success) {
        LOG(ERROR) << "Failed to destroy CuTeDSL runtime module after "
                      "creation failed: "
                   << FormatRuntimeError(destroy_error);
      }
    }
    return absl::InternalError(
        absl::StrFormat("Failed to create CuTeDSL runtime module: %s",
                        FormatRuntimeError(error)));
  }
  if (module == nullptr) {
    return absl::InternalError(
        "CuTeDSL runtime created a null module without returning an error");
  }

  return std::shared_ptr<LoadedModule>(new LoadedModule(module));
}

absl::StatusOr<LoadedModule::FunctionHandle> LoadedModule::GetFunction(
    absl::string_view function_prefix) {
  if (function_prefix.empty()) {
    return absl::InvalidArgumentError(
        "CuTeDSL function prefix must not be empty");
  }
  if (function_prefix.find('\0') != absl::string_view::npos) {
    return absl::InvalidArgumentError(
        "CuTeDSL function prefix must not contain a null byte");
  }

  std::string prefix(function_prefix);
  absl::MutexLock lock(&mutex_);
  auto it = functions_by_prefix_.find(prefix);
  if (it != functions_by_prefix_.end()) return it->second;

  CuteDSLRT_Function_t* function = nullptr;
  CuteDSLRT_Error_t error = CuteDSLRT_Module_Get_Function(
      &function, static_cast<CuteDSLRT_Module_t*>(module_), prefix.c_str());
  if (error != CuteDSLRT_Error_Success) {
    return absl::InternalError(
        absl::StrFormat("Failed to load CuTeDSL %s function: %s", prefix,
                        FormatRuntimeError(error)));
  }
  if (function == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "CuTeDSL runtime returned a null %s function without returning an "
        "error",
        prefix));
  }

  functions_by_prefix_.emplace(std::move(prefix), function);
  return function;
}

absl::Status LoadedModule::Run(FunctionHandle function, void** arguments,
                               size_t argument_count) const {
  CuteDSLRT_Error_t error =
      CuteDSLRT_Function_Run(function, arguments, argument_count);
  if (error == CuteDSLRT_Error_Success) return absl::OkStatus();
  return absl::InternalError(FormatRuntimeError(error));
}

absl::StatusOr<std::shared_ptr<LoadedModule>> ModuleLoader::GetOrLoad(
    const ModuleImage& image) {
  std::string cache_key(image.sha256());
  absl::MutexLock lock(&mutex_);
  auto it = modules_.find(cache_key);
  if (it != modules_.end()) return it->second;

  absl::StatusOr<std::shared_ptr<LoadedModule>> loaded =
      LoadedModule::Create(image.bytes());
  if (!loaded.ok()) return loaded.status();
  modules_.emplace(std::move(cache_key), *loaded);
  return *loaded;
}

}  // namespace xla::gpu::cutedsl
