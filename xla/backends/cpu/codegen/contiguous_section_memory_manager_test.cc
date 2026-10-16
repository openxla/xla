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

#include <cstdint>

#include "llvm/Support/Alignment.h"
#include "llvm/Support/Process.h"

namespace xla::cpu {
namespace {

TEST(ContiguousSectionMemoryManagerTest, AddsExtraRwDataSizeOnTopOfRwData) {
  // Reserve exactly one page for .data so page-size rounding doesn't leave
  // spare room for .got; the second allocation only succeeds if
  // `extra_rw_data_size` is added before rounding.
  const uintptr_t page_size = llvm::sys::Process::getPageSizeEstimate();
  constexpr uintptr_t kExtraRwSize = 16;
  ContiguousSectionMemoryManager mem_mgr(/*mmapper=*/nullptr, kExtraRwSize);
  mem_mgr.reserveAllocationSpace(/*code_size=*/0, llvm::Align(1),
                                 /*ro_data_size=*/0, llvm::Align(1),
                                 /*rw_data_size=*/page_size, llvm::Align(8));

  EXPECT_NE(mem_mgr.allocateDataSection(page_size, /*alignment=*/8,
                                        /*section_id=*/0, ".data",
                                        /*is_read_only=*/false),
            nullptr);
  EXPECT_NE(mem_mgr.allocateDataSection(kExtraRwSize, /*alignment=*/8,
                                        /*section_id=*/1, ".got",
                                        /*is_read_only=*/false),
            nullptr);
}

}  // namespace
}  // namespace xla::cpu
