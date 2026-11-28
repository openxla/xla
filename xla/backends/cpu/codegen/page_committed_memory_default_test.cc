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

#include <gtest/gtest.h>

#include <system_error>  // NOLINT

#include "xla/backends/cpu/codegen/page_committed_memory.h"

namespace xla::cpu {
namespace {

TEST(PageCommittedMemoryDefaultTest, DisabledOnNonWindowsPlatforms) {
  EXPECT_FALSE(UsesPageCommittedMemory());
  std::error_code ec;
  PageCommittedBlock block =
      AllocatePageCommittedMemory(4096, PageProtection::kReadWrite, ec);
  EXPECT_EQ(ec, std::errc::not_supported);
  EXPECT_EQ(block.base, nullptr);
  EXPECT_EQ(block.size, 0);
  EXPECT_EQ(ProtectPageCommittedMemory(nullptr, 0, PageProtection::kReadOnly),
            std::errc::not_supported);
  EXPECT_EQ(ReleasePageCommittedMemory(block), std::errc::not_supported);
}

}  // namespace
}  // namespace xla::cpu
