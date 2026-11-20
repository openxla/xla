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

#ifndef XLA_BACKENDS_CPU_CODEGEN_PAGE_COMMITTED_MEMORY_H_
#define XLA_BACKENDS_CPU_CODEGEN_PAGE_COMMITTED_MEMORY_H_

#include <cstddef>
#include <system_error>  // NOLINT

namespace xla::cpu {

struct PageCommittedBlock {
  void* base = nullptr;
  size_t size = 0;
};

enum class PageProtection {
  kReadOnly,
  kReadWrite,
  kReadExecute,
  kReadWriteExecute,
  kExecuteOnly,
};

// Returns true on platforms where OS virtual address reservation granularity
// exceeds page size (64 KiB vs 4 KiB on Windows) and JIT sections should
// commit only the required pages within each reserved block.
bool UsesPageCommittedMemory();

// Reserves an OS virtual memory block and commits only the pages needed to
// cover `num_bytes` with `protection`. Returns an empty block if
// `num_bytes == 0` or on failure (setting `error_code` on failure).
PageCommittedBlock AllocatePageCommittedMemory(size_t num_bytes,
                                               PageProtection protection,
                                               std::error_code& error_code);

// Updates page protection for `[base, base + size)` inside a committed range
// returned by `AllocatePageCommittedMemory`, invalidating the instruction cache
// when execute permission is granted.
std::error_code ProtectPageCommittedMemory(void* base, size_t size,
                                           PageProtection protection);

// Releases a block returned by `AllocatePageCommittedMemory` and resets
// `block`.
std::error_code ReleasePageCommittedMemory(PageCommittedBlock& block);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_PAGE_COMMITTED_MEMORY_H_
