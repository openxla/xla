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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/hlo/transforms/sort_rewriter.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/stream_executor/semantic_version.h"

namespace xla {
namespace gpu {
namespace {
using ::absl_testing::IsOkAndHolds;

namespace m = ::xla::match;

class SortRewriterDevicelessTest : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    HloHardwareIndependentTestBase::SetUp();
    SortRewriter::SetSortModeForTestingOnly(SortRewriter::Mode::kAlways);
  }

  absl::StatusOr<bool> RunPass(HloModule* module,
                               bool is_early_exit_with_layouts = false) {
    stream_executor::DeviceDescription device_desc;
    device_desc.set_name("NVIDIA H100 80GB HBM3");
    device_desc.set_gpu_compute_capability(
        stream_executor::CudaComputeCapability(9, 0));
    device_desc.set_cub_version(stream_executor::SemanticVersion{3, 1, 2});
    return SortRewriter(device_desc, /*is_deviceless=*/true,
                        is_early_exit_with_layouts)
        .Run(module);
  }

  void AssertSortWasRewritten(const HloModule* module) {
    EXPECT_THAT(
        module->entry_computation()->root_instruction(),
        GmockMatch(m::GetTupleElement(
            m::CustomCall({kCubDeviceRadixSortUnassignedScratchSizeTarget},
                          m::Parameter()),
            0)));
  }
};

// F32[1000] sort, keys-only.
constexpr absl::string_view kCompatibleSortHlo = R"hlo(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[1000] parameter(0)
  ROOT %sort = f32[1000] sort(%input), dimensions={0}, to_apply=%compare
})hlo";

// Untabulated huge sort shape: on CUDA, deviceless compilation computes
// scratch size in pure C++ and always rewrites regardless of lookup tables.
constexpr absl::string_view kUntabulatedSortHlo = R"hlo(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[100000000000000] parameter(0)
  ROOT %sort = f32[100000000000000] sort(%input), dimensions={0}, to_apply=%compare
})hlo";

TEST_F(SortRewriterDevicelessTest, CudaAlwaysRewritesRegardlessOfSize) {
  for (absl::string_view hlo : {kCompatibleSortHlo, kUntabulatedSortHlo}) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    ASSERT_THAT(RunPass(module.get()), IsOkAndHolds(true));
    AssertSortWasRewritten(module.get());
  }
}

TEST_F(SortRewriterDevicelessTest, CudaUntabulatedDeviceAlwaysRewrites) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kUntabulatedSortHlo));
  stream_executor::DeviceDescription custom_gpu;
  custom_gpu.set_name("NVIDIA Custom Future GPU");
  custom_gpu.set_gpu_compute_capability(
      stream_executor::CudaComputeCapability(10, 3));
  ASSERT_THAT(
      SortRewriter(custom_gpu, /*is_deviceless=*/true).Run(module.get()),
      IsOkAndHolds(true));
  AssertSortWasRewritten(module.get());
}

TEST_F(SortRewriterDevicelessTest, EarlyExitWithLayoutsForcesRewrite) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kUntabulatedSortHlo));

  ASSERT_THAT(RunPass(module.get(), /*is_early_exit_with_layouts=*/true),
              IsOkAndHolds(true));
  AssertSortWasRewritten(module.get());
}

TEST_F(SortRewriterDevicelessTest, CudaBatchedSegmentOverInt32MaxFallsBack) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[2,2147483648] parameter(0)
  ROOT %sort = f32[2,2147483648] sort(%input), dimensions={1}, to_apply=%compare
})hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(false));
}

TEST_F(SortRewriterDevicelessTest, CudaUnbatchedOverInt32MaxRewrites) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[1,2147483648] parameter(0)
  ROOT %sort = f32[1,2147483648] sort(%input), dimensions={1}, to_apply=%compare
})hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_THAT(RunPass(module.get()), IsOkAndHolds(true));
  AssertSortWasRewritten(module.get());
}

TEST_F(SortRewriterDevicelessTest, NonCudaDevicelessFallsBackUnlessEarlyExit) {
  stream_executor::DeviceDescription rocm_gpu;
  rocm_gpu.set_name("AMD Instinct MI300X");
  rocm_gpu.set_gpu_compute_capability(stream_executor::GpuComputeCapability(
      stream_executor::RocmComputeCapability("gfx942")));

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kCompatibleSortHlo));
  ASSERT_THAT(SortRewriter(rocm_gpu, /*is_deviceless=*/true,
                           /*is_early_exit_with_layouts=*/false)
                  .Run(module.get()),
              IsOkAndHolds(false));

  ASSERT_THAT(SortRewriter(rocm_gpu, /*is_deviceless=*/true,
                           /*is_early_exit_with_layouts=*/true)
                  .Run(module.get()),
              IsOkAndHolds(true));
  AssertSortWasRewritten(module.get());

  // Live-device non-CUDA compilation still rewrites compatible sorts.
  ASSERT_OK_AND_ASSIGN(auto live_module,
                       ParseAndReturnVerifiedModule(kCompatibleSortHlo));
  ASSERT_THAT(SortRewriter(rocm_gpu, /*is_deviceless=*/false,
                           /*is_early_exit_with_layouts=*/false)
                  .Run(live_module.get()),
              IsOkAndHolds(true));
  AssertSortWasRewritten(live_module.get());
}
}  // namespace
}  // namespace gpu
}  // namespace xla
