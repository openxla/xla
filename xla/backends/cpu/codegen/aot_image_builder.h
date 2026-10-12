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

#ifndef XLA_BACKENDS_CPU_CODEGEN_AOT_IMAGE_BUILDER_H_
#define XLA_BACKENDS_CPU_CODEGEN_AOT_IMAGE_BUILDER_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/service/cpu/executable.pb.h"

namespace xla::cpu {

// Statically links ELF64 relocatable object files (x86-64 or AArch64) into a
// `PreRelocatedImageProto` that `LoadPreRelocatedImage` can load without an ELF
// linker.
//
// Sections are laid out into RX, RO, and RW segments, external branch targets
// get PLT stubs that jump through GOT slots, and all PC-relative relocations
// are resolved here. Only 64-bit absolute pointers (GOT slots and data/code
// pointers) are left as fixups for the loader.
//
// Returns `Unimplemented` if the objects contain relocations that can't be
// expressed that way (e.g. AArch64 MOVW_UABS from the large code model, or
// 32-bit absolute relocations), so callers can fall back to the object files.
//
// `.eh_frame` sections aren't included in the image, so the loaded code isn't
// registered with the unwinder.
absl::StatusOr<PreRelocatedImageProto> BuildPreRelocatedImage(
    absl::Span<const absl::string_view> obj_files,
    absl::Span<const std::string> exported_symbols);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_AOT_IMAGE_BUILDER_H_
