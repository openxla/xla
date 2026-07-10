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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_MODULE_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_MODULE_H_

#include <cstddef>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/libraries/cutedsl/runtime_api.h"

namespace xla::gpu::cutedsl {

inline constexpr size_t kModuleCacheKeySize = 32;

std::string FormatRuntimeError(const RuntimeFunctions& functions,
                               CuteDSLRT_Error_t error);

// Owns one runtime module and lazily loads its exported functions. Function
// handles remain valid only while their LoadedModule is alive.
class LoadedModule {
 public:
  LoadedModule(const LoadedModule&) = delete;
  LoadedModule& operator=(const LoadedModule&) = delete;
  ~LoadedModule();

  const RuntimeFunctions& functions() const { return functions_; }

  absl::StatusOr<CuteDSLRT_Function_t*> GetFunction(
      absl::string_view function_prefix);

 private:
  friend absl::StatusOr<std::shared_ptr<LoadedModule>> GetOrLoadModule(
      absl::string_view, absl::string_view, const void*);

  LoadedModule(RuntimeFunctions functions, CuteDSLRT_Module_t* module)
      : functions_(functions), module_(module) {}

  static absl::StatusOr<std::shared_ptr<LoadedModule>> Create(
      RuntimeFunctions functions, absl::string_view module_bytes);

  RuntimeFunctions functions_;
  CuteDSLRT_Module_t* module_;
  absl::Mutex mutex_;
  absl::flat_hash_map<std::string, CuteDSLRT_Function_t*> functions_by_prefix_
      ABSL_GUARDED_BY(mutex_);
};

// Returns a weakly cached module after validating that `sha256_key` is the
// exact SHA-256 digest of `module_bytes`. `cache_scope` is an opaque stable
// identity; nullptr preserves process-wide sharing for the buffer-only FFI.
absl::StatusOr<std::shared_ptr<LoadedModule>> GetOrLoadModule(
    absl::string_view module_bytes, absl::string_view sha256_key,
    const void* cache_scope = nullptr);

}  // namespace xla::gpu::cutedsl

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_MODULE_H_
