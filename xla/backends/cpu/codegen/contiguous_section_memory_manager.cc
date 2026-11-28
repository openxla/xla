/* Copyright 2024 The OpenXLA Authors.

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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>  // NOLINT

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Process.h"
#include "xla/backends/cpu/codegen/page_committed_memory.h"
#include "xla/util.h"

namespace xla::cpu {
namespace {

std::optional<PageProtection> ToPageProtection(unsigned flags) {
  switch (flags & llvm::sys::Memory::MF_RWE_MASK) {
    case llvm::sys::Memory::MF_READ:
      return PageProtection::kReadOnly;
    case llvm::sys::Memory::MF_WRITE:
    case llvm::sys::Memory::MF_READ | llvm::sys::Memory::MF_WRITE:
      return PageProtection::kReadWrite;
    case llvm::sys::Memory::MF_READ | llvm::sys::Memory::MF_EXEC:
      return PageProtection::kReadExecute;
    case llvm::sys::Memory::MF_READ | llvm::sys::Memory::MF_WRITE |
        llvm::sys::Memory::MF_EXEC:
      return PageProtection::kReadWriteExecute;
    case llvm::sys::Memory::MF_EXEC:
      return PageProtection::kExecuteOnly;
    default:
      return std::nullopt;
  }
}

class DefaultMemoryMapper final
    : public llvm::SectionMemoryManager::MemoryMapper {
 public:
  llvm::sys::MemoryBlock allocateMappedMemory(
      llvm::SectionMemoryManager::AllocationPurpose purpose, size_t num_bytes,
      const llvm::sys::MemoryBlock* const near_block, unsigned flags,
      std::error_code& error_code) override {
    if (UsesPageCommittedMemory() && near_block == nullptr &&
        (flags & llvm::sys::Memory::MF_HUGE_HINT) == 0) {
      if (std::optional<PageProtection> prot = ToPageProtection(flags)) {
        PageCommittedBlock block =
            AllocatePageCommittedMemory(num_bytes, *prot, error_code);
        return llvm::sys::MemoryBlock(block.base, block.size);
      }
    }
    return llvm::sys::Memory::allocateMappedMemory(num_bytes, near_block, flags,
                                                   error_code);
  }

  std::error_code protectMappedMemory(const llvm::sys::MemoryBlock& block,
                                      unsigned flags) override {
    if (UsesPageCommittedMemory()) {
      if (std::optional<PageProtection> prot = ToPageProtection(flags)) {
        return ProtectPageCommittedMemory(block.base(), block.allocatedSize(),
                                          *prot);
      }
    }
    return llvm::sys::Memory::protectMappedMemory(block, flags);
  }

  std::error_code releaseMappedMemory(llvm::sys::MemoryBlock& m) override {
    if (UsesPageCommittedMemory()) {
      PageCommittedBlock block{m.base(), m.allocatedSize()};
      std::error_code ec = ReleasePageCommittedMemory(block);
      if (!ec) {
        m = llvm::sys::MemoryBlock();
      }
      return ec;
    }
    return llvm::sys::Memory::releaseMappedMemory(m);
  }
};

}  // namespace

ContiguousSectionMemoryManager::ContiguousSectionMemoryManager(
    llvm::SectionMemoryManager::MemoryMapper* mmapper)
    : mmapper_(mmapper), mmapper_is_owned_(false) {
  if (mmapper_ == nullptr) {
    mmapper_ = new DefaultMemoryMapper();
    mmapper_is_owned_ = true;
  }
}

ContiguousSectionMemoryManager::~ContiguousSectionMemoryManager() {
  if (allocation_.allocatedSize() != 0) {
    auto ec = mmapper_->releaseMappedMemory(allocation_);
    if (ec) {
      LOG(ERROR) << "releaseMappedMemory failed with error: " << ec.message();
    }
  }
  if (mmapper_is_owned_) {
    delete mmapper_;
  }
}

void ContiguousSectionMemoryManager::reserveAllocationSpace(
    uintptr_t code_size, llvm::Align code_align, uintptr_t ro_data_size,
    llvm::Align ro_data_align, uintptr_t rw_data_size,
    llvm::Align rw_data_align) {
  CHECK_EQ(allocation_.allocatedSize(), 0);

  static const size_t page_size = llvm::sys::Process::getPageSizeEstimate();
  CHECK(page_size != 0 && (page_size & (page_size - 1)) == 0);
  auto section_align = [](uintptr_t size, llvm::Align align) -> size_t {
    return size > 0 ? std::max<size_t>(align.value(), page_size) : page_size;
  };
  const size_t code_align_val = section_align(code_size, code_align);
  const size_t ro_data_align_val = section_align(ro_data_size, ro_data_align);
  const size_t rw_data_align_val = section_align(rw_data_size, rw_data_align);

  code_size = RoundUpTo<uintptr_t>(code_size, page_size);
  ro_data_size = RoundUpTo<uintptr_t>(ro_data_size, page_size);
  rw_data_size = RoundUpTo<uintptr_t>(rw_data_size, page_size);

  // `allocateMappedMemory` returns a page aligned base and each section size is
  // a multiple of `page_size`, so aligning a non-empty section to `align_val`
  // consumes at most `align_val - page_size` bytes of padding.
  const uintptr_t total_size =
      code_size + ro_data_size + rw_data_size + (code_align_val - page_size) +
      (ro_data_align_val - page_size) + (rw_data_align_val - page_size);
  if (total_size == 0) {
    return;
  }

  std::error_code ec;
  allocation_ = mmapper_->allocateMappedMemory(
      llvm::SectionMemoryManager::AllocationPurpose::Code, total_size, nullptr,
      llvm::sys::Memory::MF_READ | llvm::sys::Memory::MF_WRITE, ec);
  if (ec) {
    LOG(ERROR) << "allocateMappedMemory failed with error: " << ec.message();
    return;
  }

  const auto base = reinterpret_cast<std::uintptr_t>(allocation_.base());
  CHECK_EQ(base % page_size, 0);
  const uintptr_t code_start = RoundUpTo<uintptr_t>(base, code_align_val);
  code_block_ = code_free_ =
      llvm::sys::MemoryBlock(reinterpret_cast<void*>(code_start), code_size);

  const uintptr_t ro_data_start =
      RoundUpTo<uintptr_t>(code_start + code_size, ro_data_align_val);
  ro_data_block_ = ro_data_free_ = llvm::sys::MemoryBlock(
      reinterpret_cast<void*>(ro_data_start), ro_data_size);

  const uintptr_t rw_data_start =
      RoundUpTo<uintptr_t>(ro_data_start + ro_data_size, rw_data_align_val);
  CHECK_LE(rw_data_start + rw_data_size, base + allocation_.allocatedSize());
  rw_data_block_ = rw_data_free_ = llvm::sys::MemoryBlock(
      reinterpret_cast<void*>(rw_data_start), rw_data_size);
}

uint8_t* ContiguousSectionMemoryManager::allocateDataSection(
    uintptr_t size, unsigned alignment, unsigned section_id,
    llvm::StringRef section_name, bool is_read_only) {
  if (is_read_only) {
    return Allocate(ro_data_free_, size, alignment);
  } else {
    return Allocate(rw_data_free_, size, alignment);
  }
}

uint8_t* ContiguousSectionMemoryManager::allocateCodeSection(
    uintptr_t size, unsigned alignment, unsigned section_id,
    llvm::StringRef section_name) {
  return Allocate(code_free_, size, alignment);
}

uint8_t* ContiguousSectionMemoryManager::Allocate(
    llvm::sys::MemoryBlock& free_block, std::uintptr_t size,
    unsigned alignment) {
  auto base = reinterpret_cast<uintptr_t>(free_block.base());
  auto start = RoundUpTo<uintptr_t>(base, alignment);
  uintptr_t padded_size = (start - base) + size;
  if (padded_size > free_block.allocatedSize()) {
    LOG(ERROR) << "Failed to satisfy suballocation request for " << size;
    return nullptr;
  }
  free_block =
      llvm::sys::MemoryBlock(reinterpret_cast<void*>(base + padded_size),
                             free_block.allocatedSize() - padded_size);
  return reinterpret_cast<uint8_t*>(start);
}

bool ContiguousSectionMemoryManager::finalizeMemory(std::string* err_msg) {
  std::error_code ec;

  if (code_block_.allocatedSize() > 0) {
    ec = mmapper_->protectMappedMemory(
        code_block_, llvm::sys::Memory::MF_READ | llvm::sys::Memory::MF_EXEC);
    if (ec) {
      if (err_msg) {
        *err_msg = ec.message();
      }
      return true;
    }
  }
  if (ro_data_block_.allocatedSize() > 0) {
    ec = mmapper_->protectMappedMemory(ro_data_block_,
                                       llvm::sys::Memory::MF_READ);
    if (ec) {
      if (err_msg) {
        *err_msg = ec.message();
      }
      return true;
    }
  }

  if (code_block_.allocatedSize() > 0) {
    llvm::sys::Memory::InvalidateInstructionCache(code_block_.base(),
                                                  code_block_.allocatedSize());
  }
  return false;
}

}  // namespace xla::cpu
