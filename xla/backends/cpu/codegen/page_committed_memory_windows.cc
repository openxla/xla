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

#include <windows.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <system_error>  // NOLINT

#include "xla/backends/cpu/codegen/page_committed_memory.h"

namespace xla::cpu {
namespace {

struct WindowsGranularity {
  size_t page_size;
  size_t reserve_granularity;
};

WindowsGranularity GetWindowsGranularity() {
  SYSTEM_INFO info;
  ::GetSystemInfo(&info);
  const size_t page_size = static_cast<size_t>(info.dwPageSize);
  const size_t alloc_granularity =
      static_cast<size_t>(info.dwAllocationGranularity);
  return {page_size, (std::max)(page_size, alloc_granularity)};
}

bool RoundUpChecked(size_t value, size_t alignment, size_t* result) {
  if (alignment == 0) {
    return false;
  }
  const size_t remainder = value % alignment;
  if (remainder == 0) {
    *result = value;
    return true;
  }
  const size_t addend = alignment - remainder;
  if (value > (std::numeric_limits<size_t>::max)() - addend) {
    return false;
  }
  *result = value + addend;
  return true;
}

DWORD ToWindowsProtection(PageProtection protection) {
  switch (protection) {
    case PageProtection::kReadOnly:
      return PAGE_READONLY;
    case PageProtection::kReadWrite:
      return PAGE_READWRITE;
    case PageProtection::kReadExecute:
      return PAGE_EXECUTE_READ;
    case PageProtection::kReadWriteExecute:
      return PAGE_EXECUTE_READWRITE;
    case PageProtection::kExecuteOnly:
      return PAGE_EXECUTE;
  }
  return PAGE_NOACCESS;
}

bool IsExecutable(PageProtection protection) {
  return protection == PageProtection::kReadExecute ||
         protection == PageProtection::kReadWriteExecute ||
         protection == PageProtection::kExecuteOnly;
}

std::error_code LastWindowsError() {
  return std::error_code(static_cast<int>(::GetLastError()),
                         std::system_category());
}

}  // namespace

bool UsesPageCommittedMemory() { return true; }

PageCommittedBlock AllocatePageCommittedMemory(size_t num_bytes,
                                               PageProtection protection,
                                               std::error_code& error_code) {
  error_code.clear();
  if (num_bytes == 0) {
    return {};
  }

  static const WindowsGranularity granularity = GetWindowsGranularity();
  size_t commit_size = 0;
  size_t reserve_size = 0;
  if (!RoundUpChecked(num_bytes, granularity.page_size, &commit_size) ||
      !RoundUpChecked(commit_size, granularity.reserve_granularity,
                      &reserve_size)) {
    error_code = std::make_error_code(std::errc::not_enough_memory);
    return {};
  }

  const DWORD win_protect = ToWindowsProtection(protection);
  if (win_protect == PAGE_NOACCESS) {
    error_code = std::make_error_code(std::errc::invalid_argument);
    return {};
  }

  // Reserve full allocation-granularity blocks with zero pagefile commit
  // charge, then commit only the 4 KiB pages needed for `num_bytes`.
  void* base =
      ::VirtualAlloc(nullptr, reserve_size, MEM_RESERVE, PAGE_NOACCESS);
  if (base == nullptr) {
    error_code = LastWindowsError();
    return {};
  }
  if (::VirtualAlloc(base, commit_size, MEM_COMMIT, win_protect) == nullptr) {
    const std::error_code commit_error = LastWindowsError();
    ::VirtualFree(base, 0, MEM_RELEASE);
    error_code = commit_error;
    return {};
  }
  if (IsExecutable(protection)) {
    ::FlushInstructionCache(::GetCurrentProcess(), base, commit_size);
  }
  return {base, commit_size};
}

std::error_code ProtectPageCommittedMemory(void* base, size_t size,
                                           PageProtection protection) {
  if (base == nullptr || size == 0) {
    return {};
  }
  const DWORD win_protect = ToWindowsProtection(protection);
  if (win_protect == PAGE_NOACCESS) {
    return std::make_error_code(std::errc::invalid_argument);
  }
  DWORD old_protect = 0;
  if (!::VirtualProtect(base, size, win_protect, &old_protect)) {
    return LastWindowsError();
  }
  if (IsExecutable(protection)) {
    ::FlushInstructionCache(::GetCurrentProcess(), base, size);
  }
  return {};
}

std::error_code ReleasePageCommittedMemory(PageCommittedBlock& block) {
  if (block.base == nullptr || block.size == 0) {
    return {};
  }
  if (!::VirtualFree(block.base, 0, MEM_RELEASE)) {
    return LastWindowsError();
  }
  block.base = nullptr;
  block.size = 0;
  return {};
}

}  // namespace xla::cpu
