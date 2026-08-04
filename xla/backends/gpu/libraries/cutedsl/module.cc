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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"
#include "xla/backends/gpu/libraries/cutedsl/runtime_api.h"

namespace xla::gpu::cutedsl {

absl::StatusOr<ModuleImage> ModuleImage::Create(absl::string_view bytes) {
  if (bytes.empty()) {
    return absl::InvalidArgumentError("`module` must not be empty");
  }

  llvm::SHA256 hasher;
  hasher.update(llvm::StringRef(bytes.data(), bytes.size()));
  std::array<uint8_t, kModuleDigestSize> digest = hasher.final();
  return ModuleImage(std::string(bytes), digest);
}

absl::StatusOr<ModuleImage> ModuleImage::Create(absl::string_view bytes,
                                                absl::string_view sha256) {
  absl::StatusOr<ModuleImage> image = Create(bytes);
  if (!image.ok()) return image.status();
  if (sha256.size() != kModuleDigestSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`key` must contain one %d-byte SHA-256 digest", kModuleDigestSize));
  }
  if (!std::equal(image->sha256().begin(), image->sha256().end(),
                  sha256.begin())) {
    return absl::InvalidArgumentError(
        "SHA-256 `key` does not match the module image");
  }
  return std::move(*image);
}

absl::string_view ModuleImage::sha256() const {
  return absl::string_view(reinterpret_cast<const char*>(sha256_.data()),
                           sha256_.size());
}

namespace {

std::string FormatRuntimeError(const RuntimeApi& runtime,
                               CuteDSLRT_Error_t error) {
  const char* name = runtime.get_error_name(error);
  const char* description = runtime.get_error_string(error);
  return absl::StrFormat(
      "%s (error %d): %s", name == nullptr ? "Unknown" : name,
      static_cast<int>(error),
      description == nullptr ? "Unknown error" : description);
}

}  // namespace

LoadedModule::~LoadedModule() {
  if (module_ == nullptr) return;
  CuteDSLRT_Error_t error =
      runtime_->module_destroy(static_cast<CuteDSLRT_Module_t*>(module_));
  if (error != CuteDSLRT_Error_Success) {
    LOG(ERROR) << "Failed to destroy CuTeDSL runtime module: "
               << FormatRuntimeError(*runtime_, error);
  }
}

absl::StatusOr<std::shared_ptr<LoadedModule>> LoadedModule::Create(
    absl::string_view module_bytes, const RuntimeApi* runtime) {
  CuteDSLRT_Module_t* module = nullptr;
  // Standalone static and shared runtimes register their CUDA helper symbols
  // directly with ORC. shared_libs is reserved for actual dependencies of the
  // generated module.
  CuteDSLRT_Error_t error = runtime->module_create_from_bytes(
      &module, reinterpret_cast<const unsigned char*>(module_bytes.data()),
      module_bytes.size(), /*shared_libs=*/nullptr,
      /*shared_libs_size=*/0);
  if (error != CuteDSLRT_Error_Success) {
    if (module != nullptr) {
      CuteDSLRT_Error_t destroy_error = runtime->module_destroy(module);
      if (destroy_error != CuteDSLRT_Error_Success) {
        LOG(ERROR) << "Failed to destroy CuTeDSL runtime module after "
                      "creation failed: "
                   << FormatRuntimeError(*runtime, destroy_error);
      }
    }
    return absl::InternalError(
        absl::StrFormat("Failed to create CuTeDSL runtime module: %s",
                        FormatRuntimeError(*runtime, error)));
  }
  if (module == nullptr) {
    return absl::InternalError(
        "CuTeDSL runtime created a null module without returning an error");
  }

  return std::shared_ptr<LoadedModule>(new LoadedModule(module, runtime));
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
  CuteDSLRT_Error_t error = runtime_->module_get_function(
      &function, static_cast<CuteDSLRT_Module_t*>(module_), prefix.c_str());
  if (error != CuteDSLRT_Error_Success) {
    return absl::InternalError(
        absl::StrFormat("Failed to load CuTeDSL %s function: %s", prefix,
                        FormatRuntimeError(*runtime_, error)));
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
      runtime_->function_run(function, arguments, argument_count);
  if (error == CuteDSLRT_Error_Success) return absl::OkStatus();
  return absl::InternalError(FormatRuntimeError(*runtime_, error));
}

absl::StatusOr<std::shared_ptr<LoadedModule>> ModuleLoader::GetOrLoad(
    const ModuleImage& image) {
  std::string cache_key(image.sha256());
  absl::MutexLock lock(&mutex_);
  auto it = modules_.find(cache_key);
  if (it != modules_.end()) return it->second;

  if (runtime_ == nullptr) {
    absl::StatusOr<const RuntimeApi*> runtime = internal::GetRuntimeApi();
    if (!runtime.ok()) return runtime.status();
    runtime_ = *runtime;
  }

  absl::StatusOr<std::shared_ptr<LoadedModule>> loaded =
      LoadedModule::Create(image.bytes(), runtime_);
  if (!loaded.ok()) return loaded.status();
  modules_.emplace(std::move(cache_key), *loaded);
  return *loaded;
}

}  // namespace xla::gpu::cutedsl
