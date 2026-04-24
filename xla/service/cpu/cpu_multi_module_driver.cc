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

#include "xla/service/cpu/cpu_multi_module_driver.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/cpu_hlo_module_splitter.h"
#include "xla/service/cpu/multi_module_cpu_executable.h"
#include "xla/service/executable.h"
#include "xla/service/llvm_compiler.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {

bool CpuMultiModuleDriver::ShouldProcess(const HloModule& module) {
  // Traverse only reachable computations starting from the entry computation
  // to match the splitter's behavior.
  absl::flat_hash_set<const HloComputation*> visited;
  std::vector<const HloComputation*> worklist;
  worklist.push_back(module.entry_computation());
  visited.insert(module.entry_computation());

  while (!worklist.empty()) {
    const HloComputation* computation = worklist.back();
    worklist.pop_back();

    for (const auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCall) {
        auto it = instruction->frontend_attributes().map().find("inlineable");
        if (it != instruction->frontend_attributes().map().end() &&
            it->second != "true") {
          return true;
        }
      }
      // Add called computations to worklist if not visited.
      for (const HloComputation* called_computation :
           instruction->called_computations()) {
        if (visited.insert(called_computation).second) {
          worklist.push_back(called_computation);
        }
      }
    }
  }
  return false;
}

absl::StatusOr<std::unique_ptr<Executable>> CpuMultiModuleDriver::Compile(
    std::unique_ptr<HloModule> module,
    const std::vector<se::StreamExecutor*>& stream_execs,
    const Compiler::CompileOptions& options) {
  CpuHloModuleSplitter splitter;
  TF_ASSIGN_OR_RETURN(bool changed, splitter.Run(module.get()));
  if (!changed) {
    // Explicitly call the base class implementation to avoid virtual dispatch
    // back to CpuCompiler::Compile and infinite recursion.
    TF_ASSIGN_OR_RETURN(auto executables,
                        compiler_->LLVMCompiler::Compile(
                            std::move(module), stream_execs, options));
    return std::move(executables[0]);
  }

  // Compile the main module.
  TF_ASSIGN_OR_RETURN(
      auto main_executables,
      compiler_->Compile(std::move(module), stream_execs, options));
  auto main_executable = std::move(main_executables[0]);

  // Compile submodules.
  absl::flat_hash_map<std::string, std::unique_ptr<Executable>> sub_executables;
  for (auto& submodule : splitter.submodules()) {
    std::string name = submodule->name();
    TF_ASSIGN_OR_RETURN(
        auto sub_execs,
        compiler_->Compile(std::move(submodule), stream_execs, options));
    sub_executables[name] = std::move(sub_execs[0]);
  }

  return std::make_unique<MultiModuleCpuExecutable>(std::move(main_executable),
                                                    std::move(sub_executables));
}

}  // namespace xla::cpu
