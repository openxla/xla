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

// clang-format off
#include <windows.h>
#include <psapi.h>  // NOLINT(build/include_order)
// clang-format on

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <system_error>  // NOLINT
#include <vector>

#include "xla/backends/cpu/codegen/page_committed_memory.h"

namespace xla::cpu {
namespace {

TEST(PageCommittedMemoryWindowsTest, ZeroByteAllocationIsNoOp) {
  EXPECT_TRUE(UsesPageCommittedMemory());
  std::error_code ec = std::make_error_code(std::errc::invalid_argument);
  PageCommittedBlock block =
      AllocatePageCommittedMemory(0, PageProtection::kReadWrite, ec);
  EXPECT_FALSE(ec) << ec.message();
  EXPECT_EQ(block.base, nullptr);
  EXPECT_EQ(block.size, 0);
  EXPECT_FALSE(ProtectPageCommittedMemory(block.base, block.size,
                                          PageProtection::kReadOnly));
  EXPECT_FALSE(ReleasePageCommittedMemory(block));
}

TEST(PageCommittedMemoryWindowsTest, CommitsOnlyRequestedPagesInGranularity) {
  SYSTEM_INFO info;
  ::GetSystemInfo(&info);
  const size_t page_size = static_cast<size_t>(info.dwPageSize);
  const size_t granularity = static_cast<size_t>(info.dwAllocationGranularity);
  ASSERT_GT(granularity, page_size);

  std::error_code ec;
  PageCommittedBlock block =
      AllocatePageCommittedMemory(512, PageProtection::kReadWrite, ec);
  ASSERT_FALSE(ec) << ec.message();
  ASSERT_NE(block.base, nullptr);
  EXPECT_EQ(block.size, page_size);

  MEMORY_BASIC_INFORMATION mbi_committed{};
  ASSERT_NE(::VirtualQuery(block.base, &mbi_committed, sizeof(mbi_committed)),
            0);
  EXPECT_EQ(mbi_committed.AllocationBase, block.base);
  EXPECT_EQ(mbi_committed.State, static_cast<DWORD>(MEM_COMMIT));
  EXPECT_EQ(mbi_committed.Protect, static_cast<DWORD>(PAGE_READWRITE));
  EXPECT_EQ(mbi_committed.RegionSize, page_size);

  auto* trailing = static_cast<uint8_t*>(block.base) + page_size;
  MEMORY_BASIC_INFORMATION mbi_reserved{};
  ASSERT_NE(::VirtualQuery(trailing, &mbi_reserved, sizeof(mbi_reserved)), 0);
  EXPECT_EQ(mbi_reserved.AllocationBase, block.base);
  EXPECT_EQ(mbi_reserved.State, static_cast<DWORD>(MEM_RESERVE));
  EXPECT_EQ(mbi_reserved.RegionSize, granularity - page_size);

  ec = ReleasePageCommittedMemory(block);
  EXPECT_FALSE(ec) << ec.message();
  EXPECT_EQ(block.base, nullptr);
  EXPECT_EQ(block.size, 0);
}

TEST(PageCommittedMemoryWindowsTest, ProtectsSubsectionsAndExecutesCode) {
  SYSTEM_INFO info;
  ::GetSystemInfo(&info);
  const size_t page_size = static_cast<size_t>(info.dwPageSize);

  std::error_code ec;
  PageCommittedBlock block = AllocatePageCommittedMemory(
      2 * page_size, PageProtection::kReadWrite, ec);
  ASSERT_FALSE(ec) << ec.message();
  ASSERT_EQ(block.size, 2 * page_size);

  auto* code = static_cast<uint8_t*>(block.base);
  auto* rodata = code + page_size;

  constexpr uint32_t kExpected = 0x12345678u;
  std::memcpy(rodata, &kExpected, sizeof(kExpected));

  const bool is_amd64 =
      info.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64;
  if (is_amd64) {
    // Emit x86_64 `mov eax, dword ptr [rip + disp32]; ret` (7 bytes total)
    // reading `kExpected` from the adjacent read-only page.
    const int32_t disp = static_cast<int32_t>(page_size - 6);
    code[0] = 0x8B;
    code[1] = 0x05;
    std::memcpy(code + 2, &disp, sizeof(disp));
    code[6] = 0xC3;
  }

  ASSERT_FALSE(ProtectPageCommittedMemory(code, page_size,
                                          PageProtection::kReadExecute));
  ASSERT_FALSE(
      ProtectPageCommittedMemory(rodata, page_size, PageProtection::kReadOnly));

  MEMORY_BASIC_INFORMATION mbi{};
  ASSERT_NE(::VirtualQuery(code, &mbi, sizeof(mbi)), 0);
  EXPECT_EQ(mbi.State, static_cast<DWORD>(MEM_COMMIT));
  EXPECT_EQ(mbi.Protect, static_cast<DWORD>(PAGE_EXECUTE_READ));
  EXPECT_EQ(mbi.RegionSize, page_size);

  ASSERT_NE(::VirtualQuery(rodata, &mbi, sizeof(mbi)), 0);
  EXPECT_EQ(mbi.State, static_cast<DWORD>(MEM_COMMIT));
  EXPECT_EQ(mbi.Protect, static_cast<DWORD>(PAGE_READONLY));
  EXPECT_EQ(mbi.RegionSize, page_size);

  if (is_amd64) {
    using Fn = uint32_t (*)();
    Fn fn = reinterpret_cast<Fn>(code);
    EXPECT_EQ(fn(), kExpected);
  }

  EXPECT_FALSE(ReleasePageCommittedMemory(block));
}

TEST(PageCommittedMemoryWindowsTest, AvoidsGranularityCommitInflation) {
  SYSTEM_INFO info;
  ::GetSystemInfo(&info);
  const size_t page_size = static_cast<size_t>(info.dwPageSize);
  const size_t granularity = static_cast<size_t>(info.dwAllocationGranularity);
  ASSERT_GE(granularity / page_size, 4);

  PROCESS_MEMORY_COUNTERS before{};
  before.cb = sizeof(before);
  ASSERT_TRUE(
      ::GetProcessMemoryInfo(::GetCurrentProcess(), &before, sizeof(before)));

  constexpr size_t kNumBlocks = 1024;
  std::vector<PageCommittedBlock> blocks;
  blocks.reserve(kNumBlocks);
  for (size_t i = 0; i < kNumBlocks; ++i) {
    std::error_code ec;
    PageCommittedBlock block =
        AllocatePageCommittedMemory(256, PageProtection::kReadWrite, ec);
    ASSERT_FALSE(ec) << ec.message();
    *static_cast<uint32_t*>(block.base) = static_cast<uint32_t>(i);
    blocks.push_back(block);
  }

  PROCESS_MEMORY_COUNTERS after{};
  after.cb = sizeof(after);
  ASSERT_TRUE(
      ::GetProcessMemoryInfo(::GetCurrentProcess(), &after, sizeof(after)));

  // Committing 1024 single 4 KiB pages charges ~4 MiB instead of 64 MiB.
  const size_t commit_delta = after.PagefileUsage > before.PagefileUsage
                                  ? after.PagefileUsage - before.PagefileUsage
                                  : 0;
  const size_t full_granularity_bytes = kNumBlocks * granularity;
  EXPECT_LT(commit_delta, full_granularity_bytes / 4)
      << "commit_delta=" << commit_delta
      << " full_granularity_bytes=" << full_granularity_bytes;

  for (PageCommittedBlock& block : blocks) {
    EXPECT_FALSE(ReleasePageCommittedMemory(block));
  }
}

TEST(PageCommittedMemoryWindowsTest, MultiGranularityAllocation) {
  SYSTEM_INFO info;
  ::GetSystemInfo(&info);
  const size_t page_size = static_cast<size_t>(info.dwPageSize);
  const size_t granularity = static_cast<size_t>(info.dwAllocationGranularity);

  const size_t requested = granularity + page_size + 17;
  const size_t expected_commit = granularity + 2 * page_size;
  std::error_code ec;
  PageCommittedBlock block =
      AllocatePageCommittedMemory(requested, PageProtection::kReadWrite, ec);
  ASSERT_FALSE(ec) << ec.message();
  ASSERT_EQ(block.size, expected_commit);

  auto* bytes = static_cast<uint8_t*>(block.base);
  for (size_t offset = 0; offset < block.size; offset += page_size) {
    bytes[offset] = 0x5A;
  }
  EXPECT_EQ(bytes[block.size - 1], 0);
  bytes[block.size - 1] = 0xA5;
  EXPECT_EQ(bytes[block.size - 1], 0xA5);

  MEMORY_BASIC_INFORMATION mbi{};
  ASSERT_NE(::VirtualQuery(bytes, &mbi, sizeof(mbi)), 0);
  EXPECT_EQ(mbi.State, static_cast<DWORD>(MEM_COMMIT));
  EXPECT_EQ(mbi.RegionSize, expected_commit);

  EXPECT_FALSE(ReleasePageCommittedMemory(block));
}

}  // namespace
}  // namespace xla::cpu
