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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/backends/gpu/hlo/transforms/estimate_cub_sort_scratch_size.h"
#include "xla/backends/gpu/hlo/transforms/sort_rewriter.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_planner.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

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

  absl::StatusOr<bool> RunPass(HloModule* module) {
    stream_executor::DeviceDescription device_desc;
    device_desc.set_name("NVIDIA H100 80GB HBM3");
    device_desc.set_gpu_compute_capability(
        stream_executor::CudaComputeCapability(9, 0));
    device_desc.set_core_count(132);
    device_desc.set_cub_version(stream_executor::SemanticVersion{3, 1, 2});
    return SortRewriter(device_desc, /*is_deviceless=*/true).Run(module);
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

TEST_F(SortRewriterDevicelessTest, CudaAlwaysRewritesRegardlessOfModeOrSize) {
  struct NamedHlo {
    absl::string_view name;
    absl::string_view hlo;
  };
  constexpr NamedHlo kCases[] = {
      {"compatible", kCompatibleSortHlo},
      {"untabulated", kUntabulatedSortHlo},
  };
  for (DebugOptions::DevicelessCubMode mode :
       {DebugOptions::DEVICELESS_CUB_DISABLED,
        DebugOptions::DEVICELESS_CUB_WITH_FALLBACK,
        DebugOptions::DEVICELESS_CUB_NO_FALLBACK,
        DebugOptions::DEVICELESS_CUB_FORCE_ON_NO_FALLBACK}) {
    SCOPED_TRACE(DebugOptions::DevicelessCubMode_Name(mode));
    for (const NamedHlo& test_case : kCases) {
      SCOPED_TRACE(test_case.name);
      ASSERT_OK_AND_ASSIGN(auto module,
                           ParseAndReturnVerifiedModule(test_case.hlo));
      module->mutable_config()
          .mutable_debug_options()
          .set_xla_gpu_deviceless_cub_mode(mode);

      EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(true));
      AssertSortWasRewritten(module.get());
    }
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
  stream_executor::DeviceDescription rocm_gpu;
  rocm_gpu.set_rocm_compute_capability("gfx90a");

  ASSERT_OK_AND_ASSIGN(auto without_early_exit,
                       ParseAndReturnVerifiedModule(kUntabulatedSortHlo));
  without_early_exit->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_deviceless_cub_mode(DebugOptions::DEVICELESS_CUB_DISABLED);
  EXPECT_THAT(SortRewriter(rocm_gpu, /*is_deviceless=*/true,
                           /*is_early_exit_with_layouts=*/false)
                  .Run(without_early_exit.get()),
              IsOkAndHolds(false));

  ASSERT_OK_AND_ASSIGN(auto with_early_exit,
                       ParseAndReturnVerifiedModule(kUntabulatedSortHlo));
  with_early_exit->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_deviceless_cub_mode(DebugOptions::DEVICELESS_CUB_DISABLED);
  ASSERT_THAT(SortRewriter(rocm_gpu, /*is_deviceless=*/true,
                           /*is_early_exit_with_layouts=*/true)
                  .Run(with_early_exit.get()),
              IsOkAndHolds(true));
  AssertSortWasRewritten(with_early_exit.get());
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

TEST_F(SortRewriterDevicelessTest, CudaBatchedSegmentAtInt32MaxFallsBack) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[2,2147483647] parameter(0)
  ROOT %sort = f32[2,2147483647] sort(%input), dimensions={1}, to_apply=%compare
})hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(false));
}

TEST_F(SortRewriterDevicelessTest, CudaBatchedSegmentAtPlannerLimitHandoff) {
  const stream_executor::CudaComputeCapability sm90(9, 0);
  ASSERT_OK_AND_ASSIGN(
      const int64_t max_segment_size,
      MaxCudaCubSortSegmentedSegmentSize(sm90, F32, std::nullopt));

  constexpr absl::string_view kHloTemplate = R"hlo(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[2,$0] parameter(0)
  ROOT %sort = f32[2,$0] sort(%input), dimensions={1}, to_apply=%compare
})hlo";

  ASSERT_OK_AND_ASSIGN(auto at_limit,
                       ParseAndReturnVerifiedModule(
                           absl::Substitute(kHloTemplate, max_segment_size)));
  ASSERT_THAT(RunPass(at_limit.get()), IsOkAndHolds(true));
  AssertSortWasRewritten(at_limit.get());
  EXPECT_THAT(EstimateCubSortScratchSize(
                  "CUDA", stream_executor::GpuComputeCapability(sm90))
                  .Run(at_limit.get()),
              IsOkAndHolds(true));

  ASSERT_OK_AND_ASSIGN(auto over_limit,
                       ParseAndReturnVerifiedModule(absl::Substitute(
                           kHloTemplate, max_segment_size + 1)));
  EXPECT_THAT(RunPass(over_limit.get()), IsOkAndHolds(false));
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

TEST_F(SortRewriterDevicelessTest,
       CudaUnbatchedOverInt32MaxRewritesInAutoMode) {
  SortRewriter::SetSortModeForTestingOnly(SortRewriter::Mode::kAuto);
  struct NamedHlo {
    absl::string_view name;
    absl::string_view hlo;
  };
  constexpr NamedHlo kCases[] = {
      {"f32_2e31", R"hlo(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[1,2147483648] parameter(0)
  ROOT %sort = f32[1,2147483648] sort(%input), dimensions={1}, to_apply=%compare
})hlo"},
      {"u8_2e32", R"hlo(
HloModule TestModule

%compare {
  %lhs = u8[] parameter(0)
  %rhs = u8[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = u8[4294967296] parameter(0)
  ROOT %sort = u8[4294967296] sort(%input), dimensions={0}, to_apply=%compare
})hlo"},
  };
  for (const NamedHlo& test_case : kCases) {
    SCOPED_TRACE(test_case.name);
    ASSERT_OK_AND_ASSIGN(auto module,
                         ParseAndReturnVerifiedModule(test_case.hlo));
    EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(true));
    AssertSortWasRewritten(module.get());
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla
