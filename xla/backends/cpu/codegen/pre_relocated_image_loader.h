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

#ifndef XLA_BACKENDS_CPU_CODEGEN_PRE_RELOCATED_IMAGE_LOADER_H_
#define XLA_BACKENDS_CPU_CODEGEN_PRE_RELOCATED_IMAGE_LOADER_H_

#include <cstdint>
#include <memory>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/service/cpu/executable.pb.h"

namespace xla::cpu {

// Alignment of the image base and of the RX and RO segment sizes.
inline constexpr uint64_t kPreRelocatedImageSegmentAlignment = 64 * 1024;

// Maps a `PreRelocatedImageProto` into executable memory and returns a function
// library that resolves the image's exported symbols. The returned library owns
// the mapped memory.
//
// All PC-relative relocations were resolved at compile time, so loading only
// copies the segments into a 64 KiB aligned mapping, writes the 64-bit fixup
// slots (resolving external symbols with `symbol_resolver`), and applies
// segment permissions. This intentionally has no LLVM or ELF dependencies.
absl::StatusOr<std::unique_ptr<FunctionLibrary>> LoadPreRelocatedImage(
    const PreRelocatedImageProto& image,
    absl::FunctionRef<void*(absl::string_view)> symbol_resolver);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_PRE_RELOCATED_IMAGE_LOADER_H_
