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

#include "xla/backends/gpu/codegen/triton/fused_splitk.h"

#include <cstdint>
#include <memory>
#include <optional>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_module_config.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

// The RTXA6000 test device has 84 SMs, 1536 threads/SM and no block limit,
// so the heuristic assumes 1536 / (4*32) = 12 resident programs per SM,
// i.e. a saturation target of 1008 programs.
class FusedSplitKTest : public HloHardwareIndependentTestBase {
 protected:
  const se::DeviceDescription device_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();

  // Returns the root dot of a single-dot computation built from $0 = lhs
  // shape, $1 = rhs shape, $2 = out shape (with a contraction Tile of 64).
  std::unique_ptr<HloModule> MakeDotModule(absl::string_view lhs,
                                           absl::string_view rhs,
                                           absl::string_view out) {
    return ParseAndReturnVerifiedModule(
               absl::Substitute(R"(
      ENTRY e {
        p0 = $0 parameter(0)
        p1 = $1 parameter(1)
        ROOT dot = $2 dot(p0, p1),
          lhs_batch_dims={0}, lhs_contracting_dims={2},
          rhs_batch_dims={0}, rhs_contracting_dims={1},
          backend_config={sizes:[64]}
      })",
                                lhs, rhs, out))
        .value();
  }

  const HloDotInstruction* Root(const HloModule& module) {
    return Cast<HloDotInstruction>(
        module.entry_computation()->root_instruction());
  }
};

TEST_F(FusedSplitKTest, QualifyingNarrowDotReturnsSizes) {
  auto module = MakeDotModule("bf16[256,2,65536]", "bf16[256,65536,2]",
                              "f32[256,2,2]");
  std::optional<NarrowDotSizes> sizes =
      FusedSplitKQualifyingSizes(Root(*module));
  ASSERT_TRUE(sizes.has_value());
  EXPECT_EQ(sizes->m, 2);
  EXPECT_EQ(sizes->n, 2);
  EXPECT_EQ(sizes->k, 65536);
}

TEST_F(FusedSplitKTest, NonF32OutputDoesNotQualify) {
  auto module = MakeDotModule("bf16[256,2,65536]", "bf16[256,65536,2]",
                              "bf16[256,2,2]");
  EXPECT_FALSE(FusedSplitKQualifyingSizes(Root(*module)).has_value());
}

TEST_F(FusedSplitKTest, WideDotDoesNotQualify) {
  auto module = MakeDotModule("f32[256,32,65536]", "f32[256,65536,2]",
                              "f32[256,32,2]");
  EXPECT_FALSE(FusedSplitKQualifyingSizes(Root(*module)).has_value());
}

TEST_F(FusedSplitKTest, ShortContractionDoesNotQualify) {
  auto module =
      MakeDotModule("f32[256,2,2048]", "f32[256,2048,2]", "f32[256,2,2]");
  EXPECT_FALSE(FusedSplitKQualifyingSizes(Root(*module)).has_value());
}

TEST_F(FusedSplitKTest, ZeroSizedDimensionDoesNotQualifyOrCrash) {
  auto module =
      MakeDotModule("f32[8,0,4096]", "f32[8,4096,2]", "f32[8,0,2]");
  EXPECT_FALSE(FusedSplitKQualifyingSizes(Root(*module)).has_value());
  EXPECT_EQ(ChooseFusedSplitKForFusionRoot(*Root(*module), device_), 1);
}

TEST_F(FusedSplitKTest, ChoosesSaturatingSplit) {
  // 256 output tiles; target = ceil(1008/256) = 4, which divides the 1024
  // contraction tiles.
  auto module = MakeDotModule("bf16[256,2,65536]", "bf16[256,65536,2]",
                              "f32[256,2,2]");
  EXPECT_EQ(ChooseFusedSplitKForFusionRoot(*Root(*module), device_), 4);
}

TEST_F(FusedSplitKTest, SmallBatchGetsLargerSplit) {
  // 8 output tiles; target = 126 -> 128, decays to 32 to stay a proper
  // divisor of the 64 contraction tiles.
  auto module =
      MakeDotModule("bf16[8,2,4096]", "bf16[8,4096,2]", "f32[8,2,2]");
  EXPECT_EQ(ChooseFusedSplitKForFusionRoot(*Root(*module), device_), 32);
}

TEST_F(FusedSplitKTest, IndivisibleTileCountDecaysToOne) {
  // 65 = 5 * 13 contraction tiles have no power-of-two divisor.
  auto module =
      MakeDotModule("f32[8,2,4160]", "f32[8,4160,2]", "f32[8,2,2]");
  EXPECT_EQ(ChooseFusedSplitKForFusionRoot(*Root(*module), device_), 1);
}

TEST_F(FusedSplitKTest, NonRootDotIsNotSplit) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p0 = bf16[256,2,65536] parameter(0)
      p1 = bf16[256,65536,2] parameter(1)
      dot = f32[256,2,2] dot(p0, p1),
        lhs_batch_dims={0}, lhs_contracting_dims={2},
        rhs_batch_dims={0}, rhs_contracting_dims={1},
        backend_config={sizes:[64]}
      ROOT neg = f32[256,2,2] negate(dot)
    })"));
  const HloInstruction* dot =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_EQ(ChooseFusedSplitK(Cast<HloDotInstruction>(dot), /*block_k=*/64,
                              device_),
            1);
}

TEST_F(FusedSplitKTest, MissingTileConfigMeansNoSplit) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p0 = bf16[256,2,65536] parameter(0)
      p1 = bf16[256,65536,2] parameter(1)
      ROOT dot = f32[256,2,2] dot(p0, p1),
        lhs_batch_dims={0}, lhs_contracting_dims={2},
        rhs_batch_dims={0}, rhs_contracting_dims={1}
    })"));
  EXPECT_EQ(ChooseFusedSplitKForFusionRoot(
                *module->entry_computation()->root_instruction(), device_),
            1);
}

TEST_F(FusedSplitKTest, EnabledByDefaultAndDisabledByFlags) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  EXPECT_TRUE(FusedSplitKEnabled(config));

  HloModuleConfig disabled = config;
  disabled.mutable_debug_options().set_xla_gpu_enable_fused_split_k(false);
  EXPECT_FALSE(FusedSplitKEnabled(disabled));

  HloModuleConfig deterministic = config;
  deterministic.mutable_debug_options().set_xla_gpu_deterministic_ops(true);
  EXPECT_FALSE(FusedSplitKEnabled(deterministic));

  HloModuleConfig experimental = config;
  experimental.mutable_debug_options()
      .set_xla_gpu_experimental_enable_tiling_propagation(true);
  EXPECT_FALSE(FusedSplitKEnabled(experimental));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
