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

#ifndef XLA_SERVICE_CPU_CPU_MULTI_MODULE_DRIVER_H_
#define XLA_SERVICE_CPU_CPU_MULTI_MODULE_DRIVER_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace cpu {

class CpuCompiler;

// Orchestrates the splitting of an HLO module into multiple modules and their
// compilation. This is used when some computations are marked as
// non-inlineable.
class CpuMultiModuleDriver {
 public:
  explicit CpuMultiModuleDriver(CpuCompiler* compiler) : compiler_(compiler) {}

  // Returns true if the module contains any non-inlineable computations and
  // should be processed by the multi-module driver.
  static bool ShouldProcess(const HloModule& module);

  // Splits the module and compiles the resulting submodules.
  absl::StatusOr<std::unique_ptr<Executable>> Compile(
      std::unique_ptr<HloModule> module,
      const std::vector<se::StreamExecutor*>& stream_execs,
      const Compiler::CompileOptions& options);

 private:
  CpuCompiler* compiler_;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_CPU_MULTI_MODULE_DRIVER_H_
