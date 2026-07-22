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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/libraries/cutedsl/runtime.h"

namespace xla::gpu::cutedsl {

inline constexpr size_t kModuleDigestSize = 32;

// An immutable module image with a validated SHA-256 digest.
class ModuleImage {
 public:
  static absl::StatusOr<ModuleImage> Create(absl::string_view bytes);
  static absl::StatusOr<ModuleImage> Create(absl::string_view bytes,
                                            absl::string_view sha256);

  ModuleImage(ModuleImage&&) = default;
  ModuleImage& operator=(ModuleImage&&) = default;
  ModuleImage(const ModuleImage&) = delete;
  ModuleImage& operator=(const ModuleImage&) = delete;

  absl::string_view bytes() const { return bytes_; }
  absl::string_view sha256() const;

 private:
  ModuleImage(std::string bytes, std::array<uint8_t, kModuleDigestSize> sha256)
      : bytes_(std::move(bytes)), sha256_(sha256) {}

  std::string bytes_;
  std::array<uint8_t, kModuleDigestSize> sha256_;
};

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

  LoadedModule(void* module, const RuntimeApi* runtime)
      : module_(module), runtime_(runtime) {}

  static absl::StatusOr<std::shared_ptr<LoadedModule>> Create(
      absl::string_view module_bytes, const RuntimeApi* runtime);

  void* module_;
  const RuntimeApi* runtime_;
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
  const RuntimeApi* runtime_ ABSL_GUARDED_BY(mutex_) = nullptr;
  absl::flat_hash_map<std::string, std::shared_ptr<LoadedModule>> modules_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla::gpu::cutedsl

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_MODULE_H_
