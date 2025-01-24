/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_CODEGEN_OBJECT_LOADER_H_
#define XLA_BACKENDS_CPU_CODEGEN_OBJECT_LOADER_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/IR/DataLayout.h"
#include "xla/backends/cpu/runtime/function_library.h"

namespace xla::cpu {

class ObjectLoader {
 public:
  using Symbol = FunctionLibrary::Symbol;

  explicit ObjectLoader(size_t num_dylibs);

  absl::Status AddObjFile(const std::string& obj_file,
                          const std::string& memory_buffer_name,
                          size_t dylib_index = 0);

  absl::StatusOr<std::unique_ptr<FunctionLibrary>> Load(
      absl::Span<const Symbol> symbols, const llvm::DataLayout& data_layout) &&;

  llvm::orc::RTDyldObjectLinkingLayer* object_layer() {
    return object_layer_.get();
  }

  llvm::orc::ExecutionSession* execution_session() {
    return execution_session_.get();
  }

  absl::StatusOr<llvm::orc::JITDylib*> dylib(size_t dylib_index) {
    if (dylib_index >= dylibs_.size()) {
      return absl::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("Invalid dylib index %d (num dylibs: %d))",
                          dylib_index, dylibs_.size()));
    }
    return dylibs_[dylib_index];
  }

  ~ObjectLoader();

 private:
  std::unique_ptr<llvm::orc::RTDyldObjectLinkingLayer> object_layer_;
  std::unique_ptr<llvm::orc::ExecutionSession> execution_session_;

  // Non-owning pointers to dynamic libraries created for the execution session.
  std::vector<llvm::orc::JITDylib*> dylibs_;

  // std::shared_ptr<llvm::TargetMachine> target_machine_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_OBJECT_LOADER_H_
