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

#ifndef XLA_BACKENDS_CPU_CODEGEN_WASM_FUNCTION_LIBRARY_H_
#define XLA_BACKENDS_CPU_CODEGEN_WASM_FUNCTION_LIBRARY_H_

#ifdef __EMSCRIPTEN__

#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "xla/backends/cpu/runtime/function_library.h"

namespace xla::cpu {

// Links and loads a PIC Wasm object, resolving the requested functions.
// The returned library owns the loaded module.
absl::StatusOr<std::unique_ptr<FunctionLibrary>> LoadWasmFunctionLibrary(
    absl::Span<const FunctionLibrary::Symbol> symbols,
    llvm::MemoryBufferRef object);

}  // namespace xla::cpu

#endif  // __EMSCRIPTEN__

#endif  // XLA_BACKENDS_CPU_CODEGEN_WASM_FUNCTION_LIBRARY_H_
