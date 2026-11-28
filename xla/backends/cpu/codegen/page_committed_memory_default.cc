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

#include <cstddef>
#include <system_error>  // NOLINT

#include "xla/backends/cpu/codegen/page_committed_memory.h"

namespace xla::cpu {

bool UsesPageCommittedMemory() { return false; }

PageCommittedBlock AllocatePageCommittedMemory(size_t num_bytes,
                                               PageProtection protection,
                                               std::error_code& error_code) {
  error_code = std::make_error_code(std::errc::not_supported);
  return {};
}

std::error_code ProtectPageCommittedMemory(void* base, size_t size,
                                           PageProtection protection) {
  return std::make_error_code(std::errc::not_supported);
}

std::error_code ReleasePageCommittedMemory(PageCommittedBlock& block) {
  return std::make_error_code(std::errc::not_supported);
}

}  // namespace xla::cpu
