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

#ifndef XLA_BACKENDS_CPU_CODEGEN_BUILTIN_SYMBOLS_H_
#define XLA_BACKENDS_CPU_CODEGEN_BUILTIN_SYMBOLS_H_

#include "absl/strings/string_view.h"

namespace xla::cpu {

// Returns the address of a builtin runtime symbol (libc, libm, fp16/bf16
// conversions, compiler-rt helpers) available to XLA:CPU compiled code in the
// current process, or nullptr if `name` is not a builtin symbol.
//
// `name` must not include the target's global symbol prefix (e.g. the leading
// underscore on Mac OS X). This library intentionally has no LLVM dependencies
// so it can be linked into lightweight AOT runtimes.
void* LookupBuiltinSymbol(absl::string_view name);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_BUILTIN_SYMBOLS_H_
