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

#include "xla/backends/gpu/transforms/gpu_copy_async_wrapper.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla::gpu {
namespace {

using GpuCopyAsyncWrapperTest = HloHardwareIndependentTestBase;

// Returns an HloModule with the debug flag enabled and min_bytes override set.
void EnableAsyncCopy(HloModule* module, int64_t min_bytes = 0) {
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_async_device_to_device_copy(true);
  if (min_bytes > 0) {
    module->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_async_copy_min_bytes(min_bytes);
  }
}

TEST_F(GpuCopyAsyncWrapperTest, WrapsLargeD2DCopyWhenEnabled) {
  // A single D2D copy of 1024 floats (4096 bytes), well above any threshold.
  constexpr char kHlo[] = R"(
    ENTRY main {
      p0 = f32[1024] parameter(0)
      ROOT copy = f32[1024] copy(p0)
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EnableAsyncCopy(module.get(), /*min_bytes=*/1024);

  GpuCopyAsyncWrapper wrapper(/*min_copy_bytes=*/1024);
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(true));

  EXPECT_THAT(RunFileCheck(module->ToString({}), R"(
    ; CHECK: ENTRY %main
    ; CHECK:   [[P0:%[^ ]+]] = f32[1024]{0} parameter(0)
    ; CHECK:   [[START:%[^ ]+]] = (f32[1024]{0}, f32[1024]{0}, u32[]) copy-start([[P0]])
    ; CHECK:   ROOT {{.*}} = f32[1024]{0} copy-done([[START]])
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(GpuCopyAsyncWrapperTest, DoesNotWrapWhenFlagIsDisabled) {
  constexpr char kHlo[] = R"(
    ENTRY main {
      p0 = f32[1024] parameter(0)
      ROOT copy = f32[1024] copy(p0)
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  // Do NOT set the enable flag — it defaults to false.

  GpuCopyAsyncWrapper wrapper(/*min_copy_bytes=*/1024);
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(false));

  EXPECT_THAT(RunFileCheck(module->ToString({}), R"(
    ; CHECK: ENTRY %main
    ; CHECK-NOT: copy-start
    ; CHECK:   ROOT {{.*}} = f32[1024]{0} copy(
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(GpuCopyAsyncWrapperTest, DoesNotWrapCopyBelowSizeThreshold) {
  // 4 floats = 16 bytes — below the 1024-byte threshold.
  constexpr char kHlo[] = R"(
    ENTRY main {
      p0 = f32[4] parameter(0)
      ROOT copy = f32[4] copy(p0)
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EnableAsyncCopy(module.get(), /*min_bytes=*/1024);

  GpuCopyAsyncWrapper wrapper(/*min_copy_bytes=*/1024);
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(false));

  EXPECT_THAT(RunFileCheck(module->ToString({}), R"(
    ; CHECK: ENTRY %main
    ; CHECK-NOT: copy-start
    ; CHECK:   ROOT {{.*}} = f32[4]{0} copy(
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(GpuCopyAsyncWrapperTest, IsIdempotent) {
  constexpr char kHlo[] = R"(
    ENTRY main {
      p0 = f32[1024] parameter(0)
      ROOT copy = f32[1024] copy(p0)
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EnableAsyncCopy(module.get(), /*min_bytes=*/1024);

  GpuCopyAsyncWrapper wrapper(/*min_copy_bytes=*/1024);
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(true));
  // Second run: the kCopy is gone, only copy-start/copy-done remain.
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(false));
}

TEST_F(GpuCopyAsyncWrapperTest, ProtoMinBytesOverridesConstructorDefault) {
  // Copy of 256 bytes. Constructor threshold = 1024 bytes (would skip),
  // but the proto field sets min_bytes = 128 (would wrap).
  constexpr char kHlo[] = R"(
    ENTRY main {
      p0 = f32[64] parameter(0)
      ROOT copy = f32[64] copy(p0)
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  // Enable flag AND set a proto-level min_bytes lower than constructor default.
  EnableAsyncCopy(module.get(), /*min_bytes=*/128);

  GpuCopyAsyncWrapper wrapper(/*min_copy_bytes=*/1024);
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(true));

  EXPECT_THAT(RunFileCheck(module->ToString({}), R"(
    ; CHECK: ENTRY %main
    ; CHECK:   [[START:%[^ ]+]] = (f32[64]{0}, f32[64]{0}, u32[]) copy-start(
    ; CHECK:   ROOT {{.*}} = f32[64]{0} copy-done([[START]])
  )"),
              absl_testing::IsOkAndHolds(true));
}

}  // namespace
}  // namespace xla::gpu
