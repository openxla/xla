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

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
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
namespace {

absl::Status ValidateModuleKey(absl::string_view module,
                               absl::string_view key) {
  if (key.size() != kModuleCacheKeySize) {
    return absl::InvalidArgumentError(
        absl::StrFormat("CuTeDSL cache key must be %d bytes; got %d",
                        kModuleCacheKeySize, key.size()));
  }

  llvm::SHA256 hasher;
  hasher.update(llvm::StringRef(module.data(), module.size()));
  std::array<uint8_t, kModuleCacheKeySize> digest = hasher.final();
  if (!std::equal(digest.begin(), digest.end(),
                  reinterpret_cast<const uint8_t*>(key.data()))) {
    return absl::InvalidArgumentError(
        "CuTeDSL cache key does not match the module SHA-256 digest");
  }

  return absl::OkStatus();
}

using ModuleCacheKey = std::pair<const void*, std::string>;

struct ModuleCache {
  absl::Mutex mutex_;
  absl::flat_hash_map<ModuleCacheKey, std::weak_ptr<LoadedModule>> cache_
      ABSL_GUARDED_BY(mutex_);
};

ModuleCache& GlobalModuleCache() {
  static auto* cache = new ModuleCache;
  return *cache;
}

}  // namespace

std::string FormatRuntimeError(const RuntimeFunctions& functions,
                               CuteDSLRT_Error_t error) {
  const char* name = functions.get_error_name(error);
  const char* description = functions.get_error_string(error);
  return absl::StrFormat(
      "%s (error %d): %s", name == nullptr ? "Unknown" : name, error,
      description == nullptr ? "Unknown error" : description);
}

LoadedModule::~LoadedModule() {
  if (module_ == nullptr) return;
  CuteDSLRT_Error_t error = functions_.module_destroy(module_);
  if (error != kCuteDslRtSuccess) {
    LOG(ERROR) << "Failed to destroy CuTeDSL runtime module: "
               << FormatRuntimeError(functions_, error);
  }
}

absl::StatusOr<std::shared_ptr<LoadedModule>> LoadedModule::Create(
    RuntimeFunctions functions, absl::string_view module_bytes) {
  CuteDSLRT_Module_t* module = nullptr;
  // Standalone static and shared runtimes register their CUDA helper symbols
  // directly with ORC. shared_libs is reserved for actual dependencies of the
  // generated module.
  CuteDSLRT_Error_t error = functions.module_create_from_bytes(
      &module, reinterpret_cast<const unsigned char*>(module_bytes.data()),
      module_bytes.size(), /*shared_libs=*/nullptr,
      /*shared_libs_size=*/0);
  if (error != kCuteDslRtSuccess) {
    if (module != nullptr) {
      CuteDSLRT_Error_t destroy_error = functions.module_destroy(module);
      if (destroy_error != kCuteDslRtSuccess) {
        LOG(ERROR) << "Failed to destroy CuTeDSL runtime module after "
                      "creation failed: "
                   << FormatRuntimeError(functions, destroy_error);
      }
    }
    return absl::InternalError(
        absl::StrFormat("Failed to create CuTeDSL runtime module: %s",
                        FormatRuntimeError(functions, error)));
  }
  if (module == nullptr) {
    return absl::InternalError(
        "CuTeDSL runtime created a null module without returning an error");
  }

  return std::shared_ptr<LoadedModule>(new LoadedModule(functions, module));
}

absl::StatusOr<CuteDSLRT_Function_t*> LoadedModule::GetFunction(
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
  CuteDSLRT_Error_t error =
      functions_.module_get_function(&function, module_, prefix.c_str());
  if (error != kCuteDslRtSuccess) {
    return absl::InternalError(
        absl::StrFormat("Failed to load CuTeDSL %s function: %s", prefix,
                        FormatRuntimeError(functions_, error)));
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

absl::StatusOr<std::shared_ptr<LoadedModule>> GetOrLoadModule(
    absl::string_view module_bytes, absl::string_view sha256_key,
    const void* cache_scope) {
  absl::Status key_status = ValidateModuleKey(module_bytes, sha256_key);
  if (!key_status.ok()) return key_status;

  absl::StatusOr<RuntimeFunctions> functions = GetRuntimeFunctions();
  if (!functions.ok()) return functions.status();

  ModuleCacheKey cache_key(cache_scope, std::string(sha256_key));
  ModuleCache& cache = GlobalModuleCache();
  absl::MutexLock lock(&cache.mutex_);
  auto it = cache.cache_.find(cache_key);
  if (it != cache.cache_.end()) {
    if (std::shared_ptr<LoadedModule> cached = it->second.lock()) {
      return cached;
    }
    cache.cache_.erase(it);
  }

  absl::StatusOr<std::shared_ptr<LoadedModule>> loaded =
      LoadedModule::Create(*functions, module_bytes);
  if (!loaded.ok()) return loaded.status();
  cache.cache_.emplace(std::move(cache_key), *loaded);
  return *loaded;
}

}  // namespace xla::gpu::cutedsl
