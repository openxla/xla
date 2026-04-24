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

#ifndef XLA_SERVICE_CPU_MULTI_MODULE_CPU_EXECUTABLE_H_
#define XLA_SERVICE_CPU_MULTI_MODULE_CPU_EXECUTABLE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/service/executable.h"
#include "xla/service/service_executable_run_options.h"

namespace xla {
namespace cpu {

// An executable that wraps a main executable and a set of sub-modules.
//
// When executed, it registers the sub-modules in a thread-local registry,
// enabling the main executable to call them via a special custom call handler.
class MultiModuleCpuExecutable : public Executable {
 public:
  MultiModuleCpuExecutable(
      std::unique_ptr<Executable> main_executable,
      absl::flat_hash_map<std::string, std::unique_ptr<Executable>>
          sub_modules);

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override;

  absl::StatusOr<ScopedShapedBuffer> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<const ShapedBuffer* const> arguments) override;

  int64_t SizeOfGeneratedCodeInBytes() const override;

 private:
  std::unique_ptr<Executable> main_executable_;
  absl::flat_hash_map<std::string, std::unique_ptr<Executable>> sub_modules_;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_MULTI_MODULE_CPU_EXECUTABLE_H_
