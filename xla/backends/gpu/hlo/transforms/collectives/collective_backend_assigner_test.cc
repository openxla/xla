/* Copyright 2025 The OpenXLA Authors.

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
#include "xla/backends/gpu/hlo/transforms/collectives/collective_backend_assigner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

class CollectiveBackendAssignerTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<bool> RunCollectiveBackendAssigner(HloModule* module) {
    return RunHloPass(CollectiveBackendAssigner(), module);
  }

  absl::StatusOr<DebugOptions::CollectivesMode> GetCollectivesMode(
      const HloInstruction* instr) {
    ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                          instr->backend_config<GpuBackendConfig>());
    return gpu_config.collective_backend_config().collectives_mode();
  }
};

TEST_F(CollectiveBackendAssignerTest,
       CollectivePermuteSymmetricMemorySetsMode) {
  absl::string_view kHloText = R"(
    HloModule m

    ENTRY main {
      p0 = u32[8,8] parameter(0)
      ROOT result = u32[8,8] collective-permute(p0), channel_id=10,
        source_target_pairs={{0,1},{1,0}}
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_collective_permute_mode(
          DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY);

  EXPECT_THAT(RunCollectiveBackendAssigner(module.get()),
              absl_testing::IsOkAndHolds(true));

  const HloInstruction* permute =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(
      GetCollectivesMode(permute),
      absl_testing::IsOkAndHolds(DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY));
}

TEST_F(CollectiveBackendAssignerTest,
       CollectivePermutePrivateMemoryLeavesDefault) {
  absl::string_view kHloText = R"(
    HloModule m

    ENTRY main {
      p0 = u32[8,8] parameter(0)
      ROOT result = u32[8,8] collective-permute(p0), channel_id=12,
        source_target_pairs={{0,1},{1,0}}
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  // Default is COLLECTIVES_MODE_INVALID — collectives_mode should not be set.

  ASSERT_THAT(RunCollectiveBackendAssigner(module.get()), absl_testing::IsOk());

  const HloInstruction* permute =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(
      GetCollectivesMode(permute),
      absl_testing::IsOkAndHolds(DebugOptions::COLLECTIVES_MODE_INVALID));
}

TEST_F(CollectiveBackendAssignerTest,
       AllReduceUnaffectedByCollectivePermuteMode) {
  absl::string_view kHloText = R"(
    HloModule m

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY main {
      p0 = f32[8,8] parameter(0)
      ROOT result = f32[8,8] all-reduce(p0), to_apply=add,
        replica_groups={{0,1}}, channel_id=13
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_collective_permute_mode(
          DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY);

  EXPECT_THAT(RunCollectiveBackendAssigner(module.get()),
              absl_testing::IsOkAndHolds(false));

  const HloInstruction* all_reduce =
      module->entry_computation()->root_instruction();
  // collectives_mode should remain COLLECTIVES_MODE_INVALID (unaffected by
  // permute flag).
  EXPECT_THAT(
      GetCollectivesMode(all_reduce),
      absl_testing::IsOkAndHolds(DebugOptions::COLLECTIVES_MODE_INVALID));
}

TEST_F(CollectiveBackendAssignerTest, AllGatherSymmetricMemorySetsMode) {
  absl::string_view kHloText = R"(
    HloModule m

    ENTRY main {
      p0 = u32[4,8] parameter(0)
      ROOT result = u32[8,8] all-gather(p0), dimensions={0},
        replica_groups={{0,1}}, channel_id=20
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  module->mutable_config().mutable_debug_options().set_xla_gpu_all_gather_mode(
      DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY);

  EXPECT_THAT(RunCollectiveBackendAssigner(module.get()),
              absl_testing::IsOkAndHolds(true));

  const HloInstruction* all_gather =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(
      GetCollectivesMode(all_gather),
      absl_testing::IsOkAndHolds(DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY));
}

TEST_F(CollectiveBackendAssignerTest, AllGatherPrivateMemoryLeavesDefault) {
  absl::string_view kHloText = R"(
    HloModule m

    ENTRY main {
      p0 = u32[4,8] parameter(0)
      ROOT result = u32[8,8] all-gather(p0), dimensions={0},
        replica_groups={{0,1}}, channel_id=21
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  ASSERT_THAT(RunCollectiveBackendAssigner(module.get()), absl_testing::IsOk());

  const HloInstruction* all_gather =
      module->entry_computation()->root_instruction();
  EXPECT_THAT(
      GetCollectivesMode(all_gather),
      absl_testing::IsOkAndHolds(DebugOptions::COLLECTIVES_MODE_INVALID));
}

TEST_F(CollectiveBackendAssignerTest,
       SymmetricMemoryRulesRecordSelectionAndKeepModeFlags) {
  absl::string_view kHloText = R"(
    HloModule m

    add {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT s = f32[] add(a, b)
    }

    ENTRY main {
      p0 = f32[512] parameter(0)
      p1 = f32[4096] parameter(1)
      ag-start = (f32[512], f32[1024]) all-gather-start(p0), dimensions={0},
        replica_groups={{0,1}}
      ag = f32[1024] all-gather-done(ag-start)
      cp-start = (f32[512], f32[512]) collective-permute-start(p0),
        source_target_pairs={{0,1},{1,0}}
      cp = f32[512] collective-permute-done(cp-start)
      a2a = f32[512] all-to-all(p0), dimensions={0}, replica_groups={{0,1}}
      rs = f32[256] reduce-scatter(p0), dimensions={0}, to_apply=add,
        replica_groups={{0,1}}
      cb = f32[512] collective-broadcast(p0), replica_groups={{0,1}}
      ar = f32[4096] all-reduce(p1), to_apply=add, replica_groups={{0,1}}
      ROOT result = (f32[1024], f32[512], f32[512], f32[256], f32[512],
        f32[4096]) tuple(ag, cp, a2a, rs, cb, ar)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  DebugOptions& opts = module->mutable_config().mutable_debug_options();
  opts.set_xla_gpu_all_gather_mode(DebugOptions::COLLECTIVES_ONE_SIDED);
  opts.set_xla_gpu_collective_permute_mode(
      DebugOptions::COLLECTIVES_PEER_MEMORY);
  auto* filter = opts.add_xla_enable_nccl_symmetric_buffers_for_collectives();
  filter->set_collective(DebugOptions::ALLCOLLECTIVES);
  filter->set_max_size_bytes(8192);

  EXPECT_THAT(RunCollectiveBackendAssigner(module.get()),
              absl_testing::IsOkAndHolds(true));

  auto mode_of = [&](absl::string_view name) {
    return GetCollectivesMode(
        module->entry_computation()->GetInstructionWithName(name));
  };
  EXPECT_THAT(mode_of("ag-start"),
              absl_testing::IsOkAndHolds(DebugOptions::COLLECTIVES_ONE_SIDED));
  EXPECT_THAT(mode_of("cp-start"), absl_testing::IsOkAndHolds(
                                       DebugOptions::COLLECTIVES_PEER_MEMORY));
  EXPECT_THAT(mode_of("a2a"), absl_testing::IsOkAndHolds(
                                  DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY));
  EXPECT_THAT(mode_of("rs"), absl_testing::IsOkAndHolds(
                                 DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY));
  EXPECT_THAT(mode_of("cb"), absl_testing::IsOkAndHolds(
                                 DebugOptions::COLLECTIVES_MODE_INVALID));
  EXPECT_THAT(mode_of("ar"), absl_testing::IsOkAndHolds(
                                 DebugOptions::COLLECTIVES_MODE_INVALID));
}

TEST_F(CollectiveBackendAssignerTest,
       SymmetricMemoryRulesSkipCombinedPermutes) {
  absl::string_view kHloText = R"(
    HloModule m

    ENTRY main {
      p0 = f32[512] parameter(0)
      p1 = f32[512] parameter(1)
      single = f32[512] collective-permute(p0),
        source_target_pairs={{0,1},{1,0}}
      combined = (f32[512], f32[512]) collective-permute(p0, p1),
        source_target_pairs={{0,1},{1,0}}
      ROOT result = (f32[512], (f32[512], f32[512])) tuple(single, combined)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  module->mutable_config()
      .mutable_debug_options()
      .add_xla_enable_nccl_symmetric_buffers_for_collectives()
      ->set_collective(DebugOptions::COLLECTIVEPERMUTE);

  EXPECT_THAT(RunCollectiveBackendAssigner(module.get()),
              absl_testing::IsOkAndHolds(true));

  auto mode_of = [&](absl::string_view name) {
    return GetCollectivesMode(
        module->entry_computation()->GetInstructionWithName(name));
  };
  EXPECT_THAT(
      mode_of("single"),
      absl_testing::IsOkAndHolds(DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY));
  EXPECT_THAT(mode_of("combined"), absl_testing::IsOkAndHolds(
                                       DebugOptions::COLLECTIVES_MODE_INVALID));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
