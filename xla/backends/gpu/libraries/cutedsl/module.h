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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/libraries/cutedsl/module_image.h"

namespace xla::gpu::cutedsl {

// Owns one runtime module and lazily loads its exported functions. Function
// handles remain valid only while their LoadedModule is alive.
class LoadedModule {
 public:
  using FunctionHandle = void*;

  LoadedModule(const LoadedModule&) = delete;
  LoadedModule& operator=(const LoadedModule&) = delete;
  ~LoadedModule();

  absl::StatusOr<FunctionHandle> GetFunction(absl::string_view function_prefix);
  absl::Status Run(FunctionHandle function, void** arguments,
                   size_t argument_count) const;

 private:
  friend class ModuleLoader;

  explicit LoadedModule(void* module) : module_(module) {}

  static absl::StatusOr<std::shared_ptr<LoadedModule>> Create(
      absl::string_view module_bytes);

  void* module_;
  absl::Mutex mutex_;
  absl::flat_hash_map<std::string, FunctionHandle> functions_by_prefix_
      ABSL_GUARDED_BY(mutex_);
};

// Loads and strongly retains modules for one FFI instance. Instance state is
// shared across concurrent executions, so all access is synchronized.
class ModuleLoader {
 public:
  absl::StatusOr<std::shared_ptr<LoadedModule>> GetOrLoad(
      const ModuleImage& image);

 private:
  absl::Mutex mutex_;
  absl::flat_hash_map<std::string, std::shared_ptr<LoadedModule>> modules_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla::gpu::cutedsl

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_MODULE_H_
