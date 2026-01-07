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

#ifndef XLA_CODEGEN_PASS_MANAGER_WITH_LOGGING_H_
#define XLA_CODEGEN_PASS_MANAGER_WITH_LOGGING_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

class PassManagerWithLogging : public mlir::PassManager {
 public:
  PassManagerWithLogging(mlir::MLIRContext* context,
                         const HloModule& hlo_module,
                         absl::string_view kernel_name,
                         absl::string_view pass_name)
      : mlir::PassManager(context),
        log_stream_(SetupPassDumping(hlo_module, kernel_name, pass_name)) {}

  ~PassManagerWithLogging() {
    if (callback_id_ != 0) {
      getContext()->getDiagEngine().eraseHandler(callback_id_);
    }
  }

 private:
  std::unique_ptr<llvm::raw_fd_ostream> SetupPassDumping(
      const HloModule& hlo_module, absl::string_view kernel_name,
      absl::string_view pass_name);

  std::unique_ptr<llvm::raw_fd_ostream> log_stream_;
  mlir::DiagnosticEngine::HandlerID callback_id_ = 0;
};

}  // namespace xla

#endif  // XLA_CODEGEN_PASS_MANAGER_WITH_LOGGING_H_
