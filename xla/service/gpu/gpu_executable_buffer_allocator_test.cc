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

#include "xla/service/gpu/gpu_executable_buffer_allocator.h"

#include <cstdint>

#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace vmm_internal {
namespace {

// ---- ParseVmmRemapSkipEnabled -------------------------------------------

TEST(VmmRemapSkipEnabledTest, NonRocmAlwaysDisabled) {
  EXPECT_FALSE(ParseVmmRemapSkipEnabled("CUDA", nullptr));
  EXPECT_FALSE(ParseVmmRemapSkipEnabled("CUDA", "1"));
  EXPECT_FALSE(ParseVmmRemapSkipEnabled("Host", "true"));
}

TEST(VmmRemapSkipEnabledTest, RocmDefaultsOnWhenUnset) {
  EXPECT_TRUE(ParseVmmRemapSkipEnabled("ROCM", nullptr));
}

TEST(VmmRemapSkipEnabledTest, RocmEmptyStringDefaultsOn) {
  // `export XLA_VMM_SKIP_REMAP=` (empty) is treated the same as unset,
  // matching ParseVmmCopyThresholdBytes' empty-string handling.
  EXPECT_TRUE(ParseVmmRemapSkipEnabled("ROCM", ""));
}

TEST(VmmRemapSkipEnabledTest, RocmFalseyValuesDisable) {
  // Case-insensitive for false/off/no; "0" is exact.
  for (const char* v : {"0", "false", "off", "FALSE", "OFF", "False", "Off",
                        "no", "No", "NO"}) {
    EXPECT_FALSE(ParseVmmRemapSkipEnabled("ROCM", v)) << "value=" << v;
  }
}

TEST(VmmRemapSkipEnabledTest, RocmOtherValuesEnable) {
  for (const char* v : {"1", "true", "on", "yes", "anything"}) {
    EXPECT_TRUE(ParseVmmRemapSkipEnabled("ROCM", v)) << "value=" << v;
  }
}

// ---- ParseVmmCopyThresholdBytes -----------------------------------------

TEST(VmmCopyThresholdBytesTest, NonRocmAlwaysZero) {
  EXPECT_EQ(ParseVmmCopyThresholdBytes("CUDA", "65536"), 0u);
  EXPECT_EQ(ParseVmmCopyThresholdBytes("CUDA", nullptr), 0u);
}

TEST(VmmCopyThresholdBytesTest, RocmUnsetOrEmptyIsZero) {
  EXPECT_EQ(ParseVmmCopyThresholdBytes("ROCM", nullptr), 0u);
  EXPECT_EQ(ParseVmmCopyThresholdBytes("ROCM", ""), 0u);
}

TEST(VmmCopyThresholdBytesTest, RocmParsesValidByteCount) {
  EXPECT_EQ(ParseVmmCopyThresholdBytes("ROCM", "0"), 0u);
  EXPECT_EQ(ParseVmmCopyThresholdBytes("ROCM", "4096"), 4096u);
  EXPECT_EQ(ParseVmmCopyThresholdBytes("ROCM", "65536"), 65536u);
}

TEST(VmmCopyThresholdBytesTest, RocmNonNumericIsZero) {
  // "auto" (future mode) and other junk silently disable the feature.
  EXPECT_EQ(ParseVmmCopyThresholdBytes("ROCM", "auto"), 0u);
  EXPECT_EQ(ParseVmmCopyThresholdBytes("ROCM", "abc"), 0u);
}

TEST(VmmCopyThresholdBytesTest, RocmTrailingGarbageIsZero) {
  EXPECT_EQ(ParseVmmCopyThresholdBytes("ROCM", "4096abc"), 0u);
}

}  // namespace
}  // namespace vmm_internal
}  // namespace gpu
}  // namespace xla
