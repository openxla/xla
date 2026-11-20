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

#include "xla/backends/cpu/codegen/contiguous_section_memory_manager.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <system_error>  // NOLINT
#include <utility>
#include <vector>

#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Process.h"

namespace xla::cpu {
namespace {

// Fake page-aligned MemoryMapper that allows choosing the returned base's page
// offset modulo high alignments and records allocation/protection calls.
class RecordingMemoryMapper final
    : public llvm::SectionMemoryManager::MemoryMapper {
 public:
  RecordingMemoryMapper(size_t base_modulus, size_t base_offset)
      : base_modulus_(base_modulus), base_offset_(base_offset) {}

  ~RecordingMemoryMapper() override {
    for (void* raw : raw_allocations_) {
      std::free(raw);
    }
  }

  llvm::sys::MemoryBlock allocateMappedMemory(
      llvm::SectionMemoryManager::AllocationPurpose purpose, size_t num_bytes,
      const llvm::sys::MemoryBlock* const near_block, unsigned flags,
      std::error_code& error_code) override {
    error_code.clear();
    requested_sizes_.push_back(num_bytes);
    const size_t total = num_bytes + base_modulus_;
    void* raw = std::malloc(total + base_modulus_);
    raw_allocations_.push_back(raw);
    uintptr_t addr = reinterpret_cast<uintptr_t>(raw);
    uintptr_t aligned = (addr + base_modulus_ - 1) & ~(base_modulus_ - 1);
    void* base = reinterpret_cast<void*>(aligned + base_offset_);
    allocated_block_ = llvm::sys::MemoryBlock(base, num_bytes);
    return allocated_block_;
  }

  std::error_code protectMappedMemory(const llvm::sys::MemoryBlock& block,
                                      unsigned flags) override {
    protected_blocks_.emplace_back(block, flags);
    return {};
  }

  std::error_code releaseMappedMemory(llvm::sys::MemoryBlock& m) override {
    ++release_count_;
    m = llvm::sys::MemoryBlock();
    return {};
  }

  const std::vector<size_t>& requested_sizes() const {
    return requested_sizes_;
  }
  const std::vector<std::pair<llvm::sys::MemoryBlock, unsigned>>&
  protected_blocks() const {
    return protected_blocks_;
  }
  llvm::sys::MemoryBlock allocated_block() const { return allocated_block_; }
  int release_count() const { return release_count_; }

 private:
  size_t base_modulus_;
  size_t base_offset_;
  std::vector<void*> raw_allocations_;
  std::vector<size_t> requested_sizes_;
  std::vector<std::pair<llvm::sys::MemoryBlock, unsigned>> protected_blocks_;
  llvm::sys::MemoryBlock allocated_block_;
  int release_count_ = 0;
};

TEST(ContiguousSectionMemoryManagerTest, CodeOnlyAllocatesSinglePage) {
  const size_t page_size = llvm::sys::Process::getPageSizeEstimate();
  RecordingMemoryMapper mapper(page_size, 0);
  {
    ContiguousSectionMemoryManager mgr(&mapper);
    mgr.reserveAllocationSpace(/*code_size=*/256, llvm::Align(16),
                               /*ro_data_size=*/0, llvm::Align(1),
                               /*rw_data_size=*/0, llvm::Align(1));
    ASSERT_EQ(mapper.requested_sizes().size(), 1);
    EXPECT_EQ(mapper.requested_sizes()[0], page_size);

    uint8_t* code = mgr.allocateCodeSection(128, 16, 1, ".text");
    ASSERT_NE(code, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(code) % 16, 0);

    std::string err;
    EXPECT_FALSE(mgr.finalizeMemory(&err)) << err;
    ASSERT_EQ(mapper.protected_blocks().size(), 1);
    EXPECT_EQ(mapper.protected_blocks()[0].first.allocatedSize(), page_size);
    EXPECT_EQ(mapper.protected_blocks()[0].second,
              static_cast<unsigned>(llvm::sys::Memory::MF_READ |
                                    llvm::sys::Memory::MF_EXEC));
  }
  EXPECT_EQ(mapper.release_count(), 1);
}

TEST(ContiguousSectionMemoryManagerTest, EmptyReservationSkipsAllocation) {
  const size_t page_size = llvm::sys::Process::getPageSizeEstimate();
  RecordingMemoryMapper mapper(page_size, 0);
  {
    ContiguousSectionMemoryManager mgr(&mapper);
    mgr.reserveAllocationSpace(0, llvm::Align(1), 0, llvm::Align(1), 0,
                               llvm::Align(1));
    EXPECT_TRUE(mapper.requested_sizes().empty());
    std::string err;
    EXPECT_FALSE(mgr.finalizeMemory(&err)) << err;
    EXPECT_TRUE(mapper.protected_blocks().empty());
  }
  EXPECT_EQ(mapper.release_count(), 0);
}

TEST(ContiguousSectionMemoryManagerTest,
     HighAlignmentsFitWithinReservedBoundsAcrossAllPageOffsets) {
  const size_t page_size = llvm::sys::Process::getPageSizeEstimate();
  const size_t high_align = 4 * page_size;

  for (size_t offset = 0; offset < high_align; offset += page_size) {
    RecordingMemoryMapper mapper(high_align, offset);
    ContiguousSectionMemoryManager mgr(&mapper);
    mgr.reserveAllocationSpace(/*code_size=*/300, llvm::Align(2 * page_size),
                               /*ro_data_size=*/500, llvm::Align(high_align),
                               /*rw_data_size=*/128,
                               llvm::Align(2 * page_size));

    uint8_t* code = mgr.allocateCodeSection(
        300, static_cast<unsigned>(2 * page_size), 1, ".text");
    uint8_t* ro = mgr.allocateDataSection(
        500, static_cast<unsigned>(high_align), 2, ".rodata",
        /*is_read_only=*/true);
    uint8_t* rw = mgr.allocateDataSection(
        128, static_cast<unsigned>(2 * page_size), 3, ".data",
        /*is_read_only=*/false);

    ASSERT_NE(code, nullptr);
    ASSERT_NE(ro, nullptr);
    ASSERT_NE(rw, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(code) % (2 * page_size), 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ro) % high_align, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(rw) % (2 * page_size), 0);

    const auto alloc_start =
        reinterpret_cast<uintptr_t>(mapper.allocated_block().base());
    const uintptr_t alloc_end =
        alloc_start + mapper.allocated_block().allocatedSize();
    EXPECT_GE(reinterpret_cast<uintptr_t>(code), alloc_start);
    EXPECT_LE(reinterpret_cast<uintptr_t>(code) + page_size,
              reinterpret_cast<uintptr_t>(ro));
    EXPECT_LE(reinterpret_cast<uintptr_t>(ro) + page_size,
              reinterpret_cast<uintptr_t>(rw));
    EXPECT_LE(reinterpret_cast<uintptr_t>(rw) + page_size, alloc_end);

    std::string err;
    EXPECT_FALSE(mgr.finalizeMemory(&err)) << err;
    EXPECT_EQ(mapper.protected_blocks().size(), 2);
  }
}

TEST(ContiguousSectionMemoryManagerTest, DefaultMapperAllocatesAndFinalizes) {
  ContiguousSectionMemoryManager mgr(/*mmapper=*/nullptr);
  mgr.reserveAllocationSpace(/*code_size=*/64, llvm::Align(16),
                             /*ro_data_size=*/64, llvm::Align(32),
                             /*rw_data_size=*/64, llvm::Align(16));
  uint8_t* code = mgr.allocateCodeSection(16, 16, 1, ".text");
  uint8_t* ro =
      mgr.allocateDataSection(32, 32, 2, ".rodata", /*is_read_only=*/true);
  uint8_t* rw =
      mgr.allocateDataSection(16, 16, 3, ".data", /*is_read_only=*/false);
  ASSERT_NE(code, nullptr);
  ASSERT_NE(ro, nullptr);
  ASSERT_NE(rw, nullptr);
  EXPECT_LT(code, ro);
  EXPECT_LT(ro, rw);

  std::memset(code, 0x90, 16);
  constexpr uint64_t kRoValue = 0x0123456789ABCDEFULL;
  std::memcpy(ro, &kRoValue, sizeof(kRoValue));
  rw[0] = 7;

  std::string err;
  ASSERT_FALSE(mgr.finalizeMemory(&err)) << err;

  uint64_t ro_readback = 0;
  std::memcpy(&ro_readback, ro, sizeof(ro_readback));
  EXPECT_EQ(ro_readback, kRoValue);
  rw[0] = 42;
  EXPECT_EQ(rw[0], 42);
}

}  // namespace
}  // namespace xla::cpu
