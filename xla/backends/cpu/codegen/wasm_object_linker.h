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

#ifndef XLA_BACKENDS_CPU_CODEGEN_WASM_OBJECT_LINKER_H_
#define XLA_BACKENDS_CPU_CODEGEN_WASM_OBJECT_LINKER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"

namespace xla::cpu {

// Links a position-independent Wasm object into a shared module using LLD.
// Infers wasm32/wasm64 from the object and returns an owning buffer.
absl::StatusOr<std::unique_ptr<llvm::MemoryBuffer>> LinkWasmObject(
    llvm::MemoryBufferRef object);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_WASM_OBJECT_LINKER_H_
