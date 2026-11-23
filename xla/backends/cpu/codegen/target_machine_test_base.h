/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_CODEGEN_TARGET_MACHINE_TEST_BASE_H_
#define XLA_BACKENDS_CPU_CODEGEN_TARGET_MACHINE_TEST_BASE_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm-c/Target.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/status_macros.h"

namespace xla::cpu {

class TargetMachineTestBase : public HloHardwareIndependentTestBase {
 public:
  static void InitializeTargets() {
    static absl::once_flag init_once;
    absl::call_once(init_once, []() {
      LLVMInitializeX86Target();
      LLVMInitializeX86TargetInfo();
      LLVMInitializeX86TargetMC();
      LLVMInitializeX86AsmPrinter();
      LLVMInitializeARMTarget();
      LLVMInitializeARMTargetInfo();
      LLVMInitializeARMTargetMC();
      LLVMInitializeARMAsmPrinter();
      LLVMInitializeAArch64Target();
      LLVMInitializeAArch64TargetInfo();
      LLVMInitializeAArch64TargetMC();
      LLVMInitializeAArch64AsmPrinter();
    });
  }

  static absl::StatusOr<std::unique_ptr<llvm::TargetMachine>>
  CreateTargetMachineWithOptions(
      absl::string_view triple_string, absl::string_view cpu_name,
      absl::string_view features,
      std::optional<llvm::Reloc::Model> reloc_model = std::nullopt,
      llvm::CodeGenOptLevel opt_level = llvm::CodeGenOptLevel::Default) {
    InitializeTargets();
    llvm::Triple triple(
        llvm::StringRef(triple_string.data(), triple_string.size()));
    std::string error;
    const llvm::Target* target =
        llvm::TargetRegistry::lookupTarget(triple, error);
    if (target == nullptr) {
      return absl::InternalError(
          absl::StrCat("Failed to lookup target: ", error));
    }

    llvm::TargetOptions target_options;
    std::unique_ptr<llvm::TargetMachine> tm =
        absl::WrapUnique(target->createTargetMachine(
            triple, llvm::StringRef(cpu_name.data(), cpu_name.size()),
            llvm::StringRef(features.data(), features.size()), target_options,
            reloc_model, /*CM=*/std::nullopt, opt_level));
    TF_RET_CHECK(tm != nullptr) << "createTargetMachine returned null";
    return tm;
  }

 protected:
  void SetUp() override { InitializeTargets(); }

  std::unique_ptr<llvm::TargetMachine> CreateTargetMachine(
      absl::string_view triple_string, absl::string_view cpu_name,
      absl::string_view features) {
    absl::StatusOr<std::unique_ptr<llvm::TargetMachine>> tm =
        CreateTargetMachineWithOptions(triple_string, cpu_name, features);
    if (!tm.ok()) {
      LOG(ERROR) << tm.status();
      return nullptr;
    }
    return *std::move(tm);
  }

  std::unique_ptr<TargetMachineFeatures> CreateTargetMachineFeatures(
      absl::string_view triple_string, absl::string_view cpu_name,
      absl::string_view features) {
    llvm::TargetMachine* target_machine =
        target_machines_
            .emplace_back(
                CreateTargetMachine(triple_string, cpu_name, features))
            .get();
    return std::make_unique<TargetMachineFeatures>(target_machine);
  }

 private:
  std::vector<std::unique_ptr<llvm::TargetMachine>> target_machines_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_TARGET_MACHINE_TEST_BASE_H_
