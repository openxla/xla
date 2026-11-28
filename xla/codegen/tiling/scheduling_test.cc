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

#include "xla/codegen/tiling/scheduling.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/tile.h"
#include "xla/codegen/tiling/tiled_hlo.h"
#include "xla/codegen/tiling/tiling_space.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_traversal.h"

namespace xla::gpu {
namespace {

using absl_testing::IsOkAndHolds;
using absl_testing::StatusIs;
using mlir::MLIRContext;
using testing::HasSubstr;

}  // namespace

MATCHER_P(MatchSchedule, schedule_string, "") {
  return ExplainMatchResult(
      true, ApproximateMatch(schedule_string, arg.ToString()), result_listener);
}

class SchedulingTest : public HloHardwareIndependentTestBase {
 public:
  SchedulingTest() : HloHardwareIndependentTestBase() {
    RegisterSymbolicExprStorage(&mlir_context_);
  }

  HloInstruction* ParseAndGetRoot(absl::string_view hlo_string) {
    auto module_or = ParseAndReturnVerifiedModule(hlo_string);
    CHECK_OK(module_or);
    module_ = std::move(module_or.value());
    return module_->entry_computation()->root_instruction();
  }

  absl::StatusOr<TiledHloComputation> ParseAndTile(
      absl::string_view hlo_string, absl::Span<const int64_t> tile_sizes = {}) {
    HloInstruction* root = ParseAndGetRoot(hlo_string);
    auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
    ABSL_ASSIGN_OR_RETURN(auto tiling_space,
                          TilingSpace::Create(*fusion_adaptor, &mlir_context_));
    if (!tile_sizes.empty()) {
      ABSL_RETURN_IF_ERROR(tiling_space->AssignTileSizes(tile_sizes));
    }
    ABSL_ASSIGN_OR_RETURN(
        TiledHloComputation tiled_computation,
        TiledHloComputation::Tile(*fusion_adaptor, std::move(tiling_space)));
    tiled_computation.Simplify();
    tiled_computation.SortInstructionsPostOrder();
    return tiled_computation;
  }

  mlir::MLIRContext mlir_context_;
  std::unique_ptr<VerifiedHloModule> module_;
};

TEST_F(SchedulingTest, OnlyParallelDimensionsTwoTilesPerProgramId) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    fusion {
      p0 = f32[2,97]{1,0} parameter(0)
      p1 = f32[2,97]{1,0} parameter(1)
      ROOT subtract = f32[2,97]{1,0} subtract(p0, p1)
    }
    ENTRY main {
      p0 = f32[2,97]{1,0} parameter(0)
      p1 = f32[2,97]{1,0} parameter(1)
      ROOT fusion = f32[2,97]{1,0} fusion(p0, p1), kind=kLoop, calls=fusion
    })",
                                    {1, 32}));
  auto scheduling = GetSchedule(tiled_computation, 2);
  EXPECT_THAT(scheduling,
              IsOkAndHolds(MatchSchedule("d0 -> (pid * 2 + tid) / 4, "
                                         "d1 -> (pid * 2 + tid) mod 4, "
                                         "num_pids=4, num_tiles=8")));
}

TEST_F(SchedulingTest, OnlyParallelDimensionsThreeTilesPerProgramId) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    fusion {
      p0 = f32[2,97]{1,0} parameter(0)
      p1 = f32[2,97]{1,0} parameter(1)
      ROOT subtract = f32[2,97]{1,0} subtract(p0, p1)
    }
    ENTRY main {
      p0 = f32[2,97]{1,0} parameter(0)
      p1 = f32[2,97]{1,0} parameter(1)
      ROOT fusion = f32[2,97]{1,0} fusion(p0, p1), kind=kLoop, calls=fusion
    })",
                                    {1, 32}));
  auto scheduling = GetSchedule(tiled_computation, 3);
  EXPECT_THAT(scheduling,
              IsOkAndHolds(MatchSchedule("d0 -> (pid * 3 + tid) / 4, "
                                         "d1 -> (pid * 3 + tid) mod 4, "
                                         "num_pids=3, num_tiles=8")));
}

TEST_F(SchedulingTest, OnlyParallelDimensions) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    fusion {
      p0 = f32[2,97]{1,0} parameter(0)
      p1 = f32[2,97]{1,0} parameter(1)
      ROOT subtract = f32[2,97]{1,0} subtract(p0, p1)
    }
    ENTRY main {
      p0 = f32[2,97]{1,0} parameter(0)
      p1 = f32[2,97]{1,0} parameter(1)
      ROOT fusion = f32[2,97]{1,0} fusion(p0, p1), kind=kLoop, calls=fusion
    })",
                                    {1, 32}));
  auto scheduling = GetSchedule(tiled_computation);
  EXPECT_THAT(scheduling,
              IsOkAndHolds(MatchSchedule("d0 -> pid / 4, d1 -> pid mod 4, "
                                         "num_pids=8, num_tiles=8")));
}

TEST_F(SchedulingTest, ReductionsAndContractionsAreNotSupported) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    max {
      p1 = f32[] parameter(1)
      p0 = f32[] parameter(0)
      ROOT m = f32[] maximum(p0, p1)
    }
    fusion {
      p0 = f32[2,97]{1,0} parameter(0)
      constant = f32[] constant(-inf)
      reduce = f32[2] reduce(p0, constant), dimensions={1}, to_apply=max
      broadcast = f32[2,97]{1,0} broadcast(reduce), dimensions={0}
      ROOT subtract = f32[2,97]{1,0} subtract(p0, broadcast)
    }
    ENTRY main {
      p0 = f32[2,97]{1,0} parameter(0)
      ROOT fusion = f32[2,97]{1,0} fusion(p0), kind=kLoop, calls=fusion
    })",
                                    {1, 32, /*reduction_tile_size=*/8}));
  EXPECT_THAT(GetSchedule(tiled_computation),
              IsOkAndHolds(MatchSchedule("d0 -> pid / 4, d1 -> pid mod 4, "
                                         "num_pids=8, num_tiles=8")));
}

TEST_F(SchedulingTest, GetDotPermutationMultipleBatchDims) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    fusion {
      p0 = f32[2,3,16,128]{3,2,1,0} parameter(0)
      p1 = f32[2,3,128,4096]{3,2,1,0} parameter(1)
      ROOT dot = f32[2,3,16,4096]{3,2,1,0} dot(p0, p1),
        lhs_batch_dims={0,1}, lhs_contracting_dims={3},
        rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    }
    ENTRY main {
      p0 = f32[2,3,16,128]{3,2,1,0} parameter(0)
      p1 = f32[2,3,128,4096]{3,2,1,0} parameter(1)
      ROOT fusion = f32[2,3,16,4096]{3,2,1,0} fusion(p0, p1), kind=kLoop, calls=fusion
    })",
                                    {1, 1, 8, 64, 32}));
  // d0, d1 are batch. d2 is m, d3 is n.
  EXPECT_THAT(
      GetSchedule(tiled_computation),
      IsOkAndHolds(MatchSchedule("d0 -> pid / 384, d1 -> (pid mod 384) / 128, "
                                 "d2 -> pid mod 384 mod 128 mod 2, "
                                 "d3 -> (pid mod 384 mod 128) / 2, "
                                 "num_pids=768, num_tiles=768")));
}

// kRaggedNonContracting M/N swap heuristic:
// When M_avg = M_total/G < N, the scheduler traverses N more slowly
// so that LHS activation tiles (size M_avg) stay in L2 cache.
//
// Shapes: LHS f32[64,256] (M=64,K=256), RHS f32[8,256,128] (G=8,K=256,N=128)
//   group_sizes s32[8], output f32[64,128]
//   M_avg = 64/8 = 8 < N = 128  →  swap M and N traversal order.
//
// TilingSpace:
//   dim 0: M=64   kParallel
//   dim 1: N=128  kParallel
//   dim 2: G=8    kSequential  (G outer loop, tile_size=1)
//   dim 3: K=256  kSequential  (K inner loop, tile_size=32)
//
// Tile sizes: [BLOCK_M=8, BLOCK_N=16, tile_G=1, BLOCK_K=32]
//   M blocks = 64/8 = 8, N blocks = 128/16 = 8, total = 64 tiles.
//
// Without swap: d0(M) is outer (slower), d1(N) is inner (faster).
// With swap:    d1(N) is outer (slower), d0(M) is inner (faster).
//   → d0 → pid % 8, d1 → pid / 8.
TEST_F(SchedulingTest, GetRaggedDotNonContractingPermutationSwapsMN) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    fusion {
      lhs = f32[64,256] parameter(0)
      rhs = f32[8,256,128] parameter(1)
      gs  = s32[8] parameter(2)
      ROOT rd = f32[64,128] ragged-dot(lhs, rhs, gs),
          lhs_contracting_dims={1}, rhs_contracting_dims={1},
          lhs_ragged_dims={0}, rhs_group_dims={0}
    }
    ENTRY main {
      p0 = f32[64,256] parameter(0)
      p1 = f32[8,256,128] parameter(1)
      p2 = s32[8] parameter(2)
      ROOT fusion = f32[64,128] fusion(p0, p1, p2), kind=kLoop, calls=fusion
    })",
                                    {8, 16, 1, 32}));
  // M_avg = 8 < N = 128 → swap: N outer (slower), M inner (faster).
  EXPECT_THAT(
      GetSchedule(tiled_computation),
      IsOkAndHolds(MatchSchedule(
          "d0 -> pid mod 8, d1 -> pid / 8, num_pids=64, num_tiles=64")));
}

// When M_avg >= N, no swap is beneficial (LHS is the larger operand).
// Shapes: LHS f32[512,256] (M=512,K=256), RHS f32[8,256,64] (G=8,K=256,N=64)
//   M_avg = 512/8 = 64 >= N = 64  →  no swap (condition is M_avg < N).
// Tile sizes: [BLOCK_M=32, BLOCK_N=32, tile_G=1, BLOCK_K=32]
//   M blocks = 512/32 = 16, N blocks = 64/32 = 2, total = 32 tiles.
TEST_F(SchedulingTest, GetRaggedDotNonContractingPermutationNoSwapWhenMGeN) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    fusion {
      lhs = f32[512,256] parameter(0)
      rhs = f32[8,256,64] parameter(1)
      gs  = s32[8] parameter(2)
      ROOT rd = f32[512,64] ragged-dot(lhs, rhs, gs),
          lhs_contracting_dims={1}, rhs_contracting_dims={1},
          lhs_ragged_dims={0}, rhs_group_dims={0}
    }
    ENTRY main {
      p0 = f32[512,256] parameter(0)
      p1 = f32[8,256,64] parameter(1)
      p2 = s32[8] parameter(2)
      ROOT fusion = f32[512,64] fusion(p0, p1, p2), kind=kLoop, calls=fusion
    })",
                                    {32, 32, 1, 32}));
  // M_avg = 64 >= N = 64 → no swap: d0(M) outer (slower), d1(N) inner (faster).
  EXPECT_THAT(
      GetSchedule(tiled_computation),
      IsOkAndHolds(MatchSchedule(
          "d0 -> pid / 2, d1 -> pid mod 2, num_pids=32, num_tiles=32")));
}

// d0 and d4 share the peer index (pid / 4); output 1 is padded on d5.
TEST_F(SchedulingTest, VariadicAllGatherSharesPeerIndex) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    f {
      p0 = f32[1,2,8,16] parameter(0)
      p1 = s32[1,2] parameter(1)
      ag = (f32[4,2,8,16], s32[4,2]) all-gather(p0, p1),
        replica_groups={{0,1,2,3}}, dimensions={0}
      gte0 = f32[4,2,8,16] get-tuple-element(ag), index=0
      gte1 = s32[4,2] get-tuple-element(ag), index=1
      ROOT t = (f32[4,2,8,16], s32[4,2]) tuple(gte0, gte1)
    }

    ENTRY e {
      arg0 = f32[1,2,8,16] parameter(0)
      arg1 = s32[1,2] parameter(1)
      ROOT fusion = (f32[4,2,8,16], s32[4,2]) fusion(arg0, arg1),
        kind=kCustom, calls=f
    })",
                                    {1, 1, 4, 16, 1, 1}));
  EXPECT_THAT(GetSchedule(tiled_computation),
              IsOkAndHolds(MatchSchedule(
                  "d0 -> pid / 4, d1 -> (pid mod 4) / 2, "
                  "d2 -> pid mod 4 mod 2, d3 -> 0, "
                  "d4 -> pid / 4 + ((pid mod 4) / 2) * 8, d5 -> pid mod 4, "
                  "num_pids=16, num_tiles=16")));
}

TEST_F(SchedulingTest, VariadicAllGatherNonZeroGatherDim) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    f {
      p0 = f32[2,1,16] parameter(0)
      p1 = s32[2,1] parameter(1)
      ag = (f32[2,4,16], s32[2,4]) all-gather(p0, p1),
        replica_groups={{0,1,2,3}}, dimensions={1}
      gte0 = f32[2,4,16] get-tuple-element(ag), index=0
      gte1 = s32[2,4] get-tuple-element(ag), index=1
      ROOT t = (f32[2,4,16], s32[2,4]) tuple(gte0, gte1)
    }

    ENTRY e {
      arg0 = f32[2,1,16] parameter(0)
      arg1 = s32[2,1] parameter(1)
      ROOT fusion = (f32[2,4,16], s32[2,4]) fusion(arg0, arg1),
        kind=kCustom, calls=f
    })",
                                    {1, 1, 8, 1, 1}));
  EXPECT_THAT(
      GetSchedule(tiled_computation),
      IsOkAndHolds(MatchSchedule("d0 -> (pid mod 4) / 2, d1 -> pid / 4, "
                                 "d2 -> pid mod 4 mod 2, d3 -> pid mod 4, "
                                 "d4 -> pid / 4 + ((pid mod 4) / 2) * 8, "
                                 "num_pids=16, num_tiles=16")));
}

// Output 1 has no private dims; padded tiles push d2 past the last peer.
TEST_F(SchedulingTest, VariadicAllGatherOutputWithoutPrivateDimsIsPadded) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    f {
      p0 = s32[1,2] parameter(0)
      p1 = s32[1] parameter(1)
      ag = (s32[4,2], s32[4]) all-gather(p0, p1),
        replica_groups={{0,1,2,3}}, dimensions={0}
      gte0 = s32[4,2] get-tuple-element(ag), index=0
      gte1 = s32[4] get-tuple-element(ag), index=1
      ROOT t = (s32[4,2], s32[4]) tuple(gte0, gte1)
    }

    ENTRY e {
      arg0 = s32[1,2] parameter(0)
      arg1 = s32[1] parameter(1)
      ROOT fusion = (s32[4,2], s32[4]) fusion(arg0, arg1), kind=kCustom,
        calls=f
    })",
                                    {1, 1, 1}));
  EXPECT_THAT(GetSchedule(tiled_computation),
              IsOkAndHolds(MatchSchedule("d0 -> pid / 2, d1 -> pid mod 2, "
                                         "d2 -> pid / 2 + (pid mod 2) * 4, "
                                         "num_pids=8, num_tiles=8")));
}

// Output 1's single-tile private dim (d3) cannot mask; d2 is pushed out of
// bounds.
TEST_F(SchedulingTest, VariadicAllGatherOutputWithSingleTileIsPadded) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    f {
      p0 = s32[1,2] parameter(0)
      p1 = s32[1,1] parameter(1)
      ag = (s32[4,2], s32[4,1]) all-gather(p0, p1),
        replica_groups={{0,1,2,3}}, dimensions={0}
      gte0 = s32[4,2] get-tuple-element(ag), index=0
      gte1 = s32[4,1] get-tuple-element(ag), index=1
      ROOT t = (s32[4,2], s32[4,1]) tuple(gte0, gte1)
    }

    ENTRY e {
      arg0 = s32[1,2] parameter(0)
      arg1 = s32[1,1] parameter(1)
      ROOT fusion = (s32[4,2], s32[4,1]) fusion(arg0, arg1), kind=kCustom,
        calls=f
    })",
                                    {1, 1, 1, 1}));
  EXPECT_THAT(GetSchedule(tiled_computation),
              IsOkAndHolds(MatchSchedule("d0 -> pid / 2, d1 -> pid mod 2, "
                                         "d2 -> pid / 2 + (pid mod 2) * 4, "
                                         "d3 -> pid mod 2, "
                                         "num_pids=8, num_tiles=8")));
}

// Six outputs with 2..10 tiles/peer share peer index `pid / 10`.
TEST_F(SchedulingTest, VariadicAllGatherWithMixedTileCounts) {
  ASSERT_OK_AND_ASSIGN(
      const TiledHloComputation tiled_computation,
      ParseAndTile(R"(
    f {
      p0 = f32[1,163840] parameter(0)
      p1 = pred[1,256] parameter(1)
      p2 = s32[1,512] parameter(2)
      p3 = pred[1,2] parameter(3)
      p4 = pred[1,2] parameter(4)
      p5 = s32[1,16] parameter(5)
      ag = (f32[4,163840], pred[4,256], s32[4,512], pred[4,2], pred[4,2],
        s32[4,16]) all-gather(p0, p1, p2, p3, p4, p5),
        replica_groups={{0,1,2,3}}, dimensions={0}
      gte0 = f32[4,163840] get-tuple-element(ag), index=0
      gte1 = pred[4,256] get-tuple-element(ag), index=1
      gte2 = s32[4,512] get-tuple-element(ag), index=2
      gte3 = pred[4,2] get-tuple-element(ag), index=3
      gte4 = pred[4,2] get-tuple-element(ag), index=4
      gte5 = s32[4,16] get-tuple-element(ag), index=5
      ROOT t = (f32[4,163840], pred[4,256], s32[4,512], pred[4,2], pred[4,2],
        s32[4,16]) tuple(gte0, gte1, gte2, gte3, gte4, gte5)
    }

    ENTRY e {
      p0 = f32[1,163840] parameter(0)
      p1 = pred[1,256] parameter(1)
      p2 = s32[1,512] parameter(2)
      p3 = pred[1,2] parameter(3)
      p4 = pred[1,2] parameter(4)
      p5 = s32[1,16] parameter(5)
      ROOT fusion = (f32[4,163840], pred[4,256], s32[4,512], pred[4,2],
        pred[4,2], s32[4,16]) fusion(p0, p1, p2, p3, p4, p5), kind=kCustom,
        calls=f
    })",
                   {1, 16384, 1, 64, 1, 128, 1, 1, 1, 1, 1, 4}));
  EXPECT_THAT(
      GetSchedule(tiled_computation),
      IsOkAndHolds(MatchSchedule(
          "d0 -> pid / 10, d1 -> pid mod 10, "
          "d2 -> pid / 10 + ((pid mod 10) / 4) * 16, d3 -> pid mod 10, "
          "d4 -> pid / 10 + ((pid mod 10) / 4) * 16, d5 -> pid mod 10, "
          "d6 -> pid / 10 + ((pid mod 10) / 2) * 8, d7 -> pid mod 10, "
          "d8 -> pid / 10 + ((pid mod 10) / 2) * 8, d9 -> pid mod 10, "
          "d10 -> pid / 10 + ((pid mod 10) / 4) * 16, d11 -> pid mod 10, "
          "num_pids=40, num_tiles=40")));
}

TEST_F(SchedulingTest, VariadicAllGatherTileSpanningPeersIsRejected) {
  HloInstruction* root = ParseAndGetRoot(R"(
    f {
      p0 = f32[1,8,16] parameter(0)
      p1 = s32[1,8] parameter(1)
      ag = (f32[4,8,16], s32[4,8]) all-gather(p0, p1),
        replica_groups={{0,1,2,3}}, dimensions={0}
      gte0 = f32[4,8,16] get-tuple-element(ag), index=0
      gte1 = s32[4,8] get-tuple-element(ag), index=1
      ROOT t = (f32[4,8,16], s32[4,8]) tuple(gte0, gte1)
    }

    ENTRY e {
      arg0 = f32[1,8,16] parameter(0)
      arg1 = s32[1,8] parameter(1)
      ROOT fusion = (f32[4,8,16], s32[4,8]) fusion(arg0, arg1),
        kind=kCustom, calls=f
    })");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  ASSERT_OK_AND_ASSIGN(auto tiling_space,
                       TilingSpace::Create(*fusion_adaptor, &mlir_context_));
  ASSERT_OK(tiling_space->AssignTileSizes({2, 8, 16, 1, 8}));
  EXPECT_THAT(
      TiledHloComputation::Tile(*fusion_adaptor, std::move(tiling_space)),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("expected tile size to divide per-rank size")));
}

// Both outputs have 2 gather tiles/peer; output 1 is padded past the last peer.
TEST_F(SchedulingTest, VariadicAllGatherMultipleTilesPerPeerOnGatherDim) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    f {
      p0 = f32[4,16] parameter(0)
      p1 = s32[2] parameter(1)
      ag = (f32[16,16], s32[8]) all-gather(p0, p1),
        replica_groups={{0,1,2,3}}, dimensions={0}
      gte0 = f32[16,16] get-tuple-element(ag), index=0
      gte1 = s32[8] get-tuple-element(ag), index=1
      ROOT t = (f32[16,16], s32[8]) tuple(gte0, gte1)
    }

    ENTRY e {
      arg0 = f32[4,16] parameter(0)
      arg1 = s32[2] parameter(1)
      ROOT fusion = (f32[16,16], s32[8]) fusion(arg0, arg1),
        kind=kCustom, calls=f
    })",
                                    {2, 8, 1}));
  EXPECT_THAT(
      GetSchedule(tiled_computation),
      IsOkAndHolds(MatchSchedule(
          "d0 -> (pid / 4) * 2 + pid mod 4 mod 2, d1 -> (pid mod 4) / 2, "
          "d2 -> (pid / 4) * 2 + pid mod 4 + ((pid mod 4) / 2) * 8, "
          "num_pids=16, num_tiles=16")));
}

TEST_F(SchedulingTest,
       InteriorVariadicAllGatherConsumedByElementwiseSharesPeerIndex) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    f {
      p0 = f32[2,16] parameter(0)
      p1 = s32[4] parameter(1)
      ag = (f32[8,16], s32[16]) all-gather(p0, p1),
        replica_groups={{0,1,2,3}}, dimensions={0}
      gte0 = f32[8,16] get-tuple-element(ag), index=0
      gte1 = s32[16] get-tuple-element(ag), index=1
      c0 = bf16[8,16] convert(gte0)
      c1 = f32[16] convert(gte1)
      ROOT t = (bf16[8,16], f32[16]) tuple(c0, c1)
    }

    ENTRY e {
      arg0 = f32[2,16] parameter(0)
      arg1 = s32[4] parameter(1)
      ROOT fusion = (bf16[8,16], f32[16]) fusion(arg0, arg1),
        kind=kCustom, calls=f
    })",
                                    {2, 8, 2}));
  EXPECT_THAT(GetSchedule(tiled_computation),
              IsOkAndHolds(MatchSchedule("d0 -> pid / 2, d1 -> pid mod 2, "
                                         "d2 -> (pid / 2) * 2 + pid mod 2, "
                                         "num_pids=8, num_tiles=8")));
}

TEST_F(SchedulingTest, VariadicAllGatherWithDotProducersSharesPeerIndex) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    f {
      lhs0 = f32[2,8] parameter(0)
      rhs0 = f32[8,16] parameter(1)
      lhs1 = f32[4,8] parameter(2)
      rhs1 = f32[8,32] parameter(3)
      dot0 = f32[2,16] dot(lhs0, rhs0),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot1 = f32[4,32] dot(lhs1, rhs1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ag = (f32[8,16], f32[16,32]) all-gather(dot0, dot1),
        replica_groups={{0,1,2,3}}, dimensions={0}
      gte0 = f32[8,16] get-tuple-element(ag), index=0
      gte1 = f32[16,32] get-tuple-element(ag), index=1
      ROOT t = (f32[8,16], f32[16,32]) tuple(gte0, gte1)
    }

    ENTRY e {
      lhs0 = f32[2,8] parameter(0)
      rhs0 = f32[8,16] parameter(1)
      lhs1 = f32[4,8] parameter(2)
      rhs1 = f32[8,32] parameter(3)
      ROOT fusion = (f32[8,16], f32[16,32]) fusion(lhs0, rhs0, lhs1, rhs1),
        kind=kCustom, calls=f
    })",
                                    {2, 8, 2, 16, 8, 8}));
  EXPECT_THAT(
      GetSchedule(tiled_computation),
      IsOkAndHolds(MatchSchedule(
          "d0 -> pid / 4 + ((pid mod 4) / 2) * 8, d1 -> pid mod 4, "
          "d2 -> (pid / 4) * 2 + pid mod 4 mod 2, d3 -> (pid mod 4) / 2, "
          "num_pids=16, num_tiles=16")));
}

// Replica dims d4 and d5 share `pid / 2`; root 1 is padded on d5.
TEST_F(SchedulingTest, VariadicReduceScatterSharesShardIndex) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    f {
      p0 = f32[4,64] parameter(0)
      p1 = f32[4,1] parameter(1)
      rs = (f32[2,64], f32[2,1]) reduce-scatter(p0, p1),
        replica_groups={{0,1}}, dimensions={0}, to_apply=add
      gte0 = f32[2,64] get-tuple-element(rs), index=0
      gte1 = f32[2,1] get-tuple-element(rs), index=1
      ROOT t = (f32[2,64], f32[2,1]) tuple(gte0, gte1)
    }

    ENTRY e {
      arg0 = f32[4,64] parameter(0)
      arg1 = f32[4,1] parameter(1)
      ROOT fusion = (f32[2,64], f32[2,1]) fusion(arg0, arg1),
        kind=kCustom, calls=f
    })",
                                    {1, 32, 1, 1}));
  EXPECT_THAT(
      GetSchedule(tiled_computation),
      IsOkAndHolds(MatchSchedule(
          "d0 -> (pid mod 2) / 2, d1 -> pid mod 2 mod 2, d2 -> pid mod 2, "
          "d3 -> 0, d4 -> pid / 2, d5 -> pid / 2 + (pid mod 2) * 2, "
          "num_pids=4, num_tiles=4")));
}

// Output 0 has 2 scattered tiles/shard on d0; d4 and d5 carry the peer index.
TEST_F(SchedulingTest, VariadicReduceScatterMultipleTilesPerShard) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    f {
      p0 = f32[8,64] parameter(0)
      p1 = f32[8,1] parameter(1)
      rs = (f32[4,64], f32[4,1]) reduce-scatter(p0, p1),
        replica_groups={{0,1}}, dimensions={0}, to_apply=add
      gte0 = f32[4,64] get-tuple-element(rs), index=0
      gte1 = f32[4,1] get-tuple-element(rs), index=1
      ROOT t = (f32[4,64], f32[4,1]) tuple(gte0, gte1)
    }

    ENTRY e {
      arg0 = f32[8,64] parameter(0)
      arg1 = f32[8,1] parameter(1)
      ROOT fusion = (f32[4,64], f32[4,1]) fusion(arg0, arg1),
        kind=kCustom, calls=f
    })",
                                    {1, 64, 2, 1}));
  EXPECT_THAT(GetSchedule(tiled_computation),
              IsOkAndHolds(MatchSchedule(
                  "d0 -> pid mod 2, d1 -> 0, d2 -> pid mod 2, d3 -> 0, "
                  "d4 -> pid / 2, d5 -> pid / 2 + (pid mod 2) * 2, "
                  "num_pids=4, num_tiles=4")));
}

TEST_F(SchedulingTest, VariadicReduceScatterWithDotProducers) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    f {
      lhs0 = f32[16,8] parameter(0)
      rhs0 = f32[8,16] parameter(1)
      lhs1 = f32[32,8] parameter(2)
      rhs1 = f32[8,32] parameter(3)
      dot0 = f32[16,16] dot(lhs0, rhs0),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot1 = f32[32,32] dot(lhs1, rhs1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      rs = (f32[4,16], f32[8,32]) reduce-scatter(dot0, dot1),
        replica_groups={{0,1,2,3}}, dimensions={0}, to_apply=add
      gte0 = f32[4,16] get-tuple-element(rs), index=0
      gte1 = f32[8,32] get-tuple-element(rs), index=1
      ROOT t = (f32[4,16], f32[8,32]) tuple(gte0, gte1)
    }

    ENTRY e {
      lhs0 = f32[16,8] parameter(0)
      rhs0 = f32[8,16] parameter(1)
      lhs1 = f32[32,8] parameter(2)
      rhs1 = f32[8,32] parameter(3)
      ROOT fusion = (f32[4,16], f32[8,32]) fusion(lhs0, rhs0, lhs1, rhs1),
        kind=kCustom, calls=f
    })",
                                    {1, 8, 1, 16, 8, 8}));
  EXPECT_THAT(GetSchedule(tiled_computation),
              IsOkAndHolds(MatchSchedule(
                  "d0 -> (pid mod 4) / 2, d1 -> pid mod 4 mod 2, "
                  "d2 -> (pid mod 4) / 2, d3 -> pid mod 4 mod 2, "
                  "d4 -> pid / 4 + ((pid mod 4) / 2) * 8, d5 -> pid / 4, "
                  "num_pids=16, num_tiles=16")));
}

TEST_F(SchedulingTest, TwoShotAllGatherSchedulesReplicaDimension) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    f {
      p0 = f32[6,2] parameter(0)
      ROOT ag = f32[6,8] all-gather(p0),
        replica_groups={{0,1,2,3}}, dimensions={1},
        backend_config={"collective_backend_config":{"kernel_strategy":"KERNEL_STRATEGY_TRITON_TWO_SHOT"}}
    }

    ENTRY e {
      arg0 = f32[6,2] parameter(0)
      ROOT fusion = f32[6,8] fusion(arg0), kind=kCustom, calls=f
    })",
                                    {2, 1}));
  EXPECT_THAT(GetSchedule(tiled_computation),
              IsOkAndHolds(
                  MatchSchedule("d0 -> (pid mod 6) / 2, d1 -> pid mod 6 mod 2, "
                                "d2 -> pid / 6, num_pids=24, num_tiles=24")));
}

TEST_F(SchedulingTest, SingleOperandReduceScatterReplicaDimension) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    f {
      p0 = f32[8,64] parameter(0)
      ROOT rs = f32[4,64] reduce-scatter(p0),
        replica_groups={{0,1}}, dimensions={0}, to_apply=add
    }

    ENTRY e {
      arg0 = f32[8,64] parameter(0)
      ROOT fusion = f32[4,64] fusion(arg0), kind=kCustom, calls=f
    })",
                                    {1, 32}));
  EXPECT_THAT(GetSchedule(tiled_computation),
              IsOkAndHolds(
                  MatchSchedule("d0 -> (pid mod 4) / 2, d1 -> pid mod 4 mod 2, "
                                "d2 -> pid / 4, num_pids=8, num_tiles=8")));
}

// Output 1 (`f32[1,4]`, tile `[1, 1]`) has 2 tiles/peer along scatter dim 1;
// padded programs push d5 by `((pid mod 4) / 2) * 4` past the last peer.
TEST_F(SchedulingTest,
       VariadicReduceScatterPaddedOutputWithTiledScatterDimIsMasked) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    f {
      p0 = f32[1,32] parameter(0)
      p1 = f32[1,8] parameter(1)
      rs = (f32[1,16], f32[1,4]) reduce-scatter(p0, p1),
        replica_groups={{0,1}}, dimensions={1}, to_apply=add
      gte0 = f32[1,16] get-tuple-element(rs), index=0
      gte1 = f32[1,4] get-tuple-element(rs), index=1
      ROOT t = (f32[1,16], f32[1,4]) tuple(gte0, gte1)
    }

    ENTRY e {
      arg0 = f32[1,32] parameter(0)
      arg1 = f32[1,8] parameter(1)
      ROOT fusion = (f32[1,16], f32[1,4]) fusion(arg0, arg1),
        kind=kCustom, calls=f
    })",
                                    {1, 2, 1, 1}));
  EXPECT_THAT(GetSchedule(tiled_computation),
              IsOkAndHolds(MatchSchedule(
                  "d0 -> (pid mod 4) / 4, d1 -> pid mod 4 mod 4, "
                  "d2 -> (pid mod 4) / 2, d3 -> pid mod 4 mod 2, "
                  "d4 -> pid / 4, d5 -> pid / 4 + ((pid mod 4) / 2) * 4, "
                  "num_pids=8, num_tiles=8")));
}

// Output 1 (`f32[4,1]`, tile `[1, 1]`) has 2 tiles/peer along scatter dim 0;
// padded programs push d4 by `((pid mod 4) / 2) * 4` past the last peer.
TEST_F(SchedulingTest,
       VariadicReduceScatterPaddedInputWithTiledScatterDimIsMasked) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    f {
      p0 = f32[32] parameter(0)
      p1 = f32[8,1] parameter(1)
      rs = (f32[16], f32[4,1]) reduce-scatter(p0, p1),
        replica_groups={{0,1}}, dimensions={0}, to_apply=add
      gte0 = f32[16] get-tuple-element(rs), index=0
      gte1 = f32[4,1] get-tuple-element(rs), index=1
      ROOT t = (f32[16], f32[4,1]) tuple(gte0, gte1)
    }

    ENTRY e {
      arg0 = f32[32] parameter(0)
      arg1 = f32[8,1] parameter(1)
      ROOT fusion = (f32[16], f32[4,1]) fusion(arg0, arg1),
        kind=kCustom, calls=f
    })",
                                    {2, 1, 1}));
  EXPECT_THAT(GetSchedule(tiled_computation),
              IsOkAndHolds(MatchSchedule(
                  "d0 -> pid mod 4, d1 -> pid mod 4, d2 -> 0, "
                  "d3 -> pid / 4, d4 -> pid / 4 + ((pid mod 4) / 2) * 4, "
                  "num_pids=8, num_tiles=8")));
}

TEST_F(SchedulingTest,
       VariadicTwoShotAllGatherPaddedSingleTileGatherDimIsMasked) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    f {
      p0 = f32[4,2] parameter(0)
      p1 = s32[1,2] parameter(1)
      ag = (f32[4,4], s32[1,4]) all-gather(p0, p1),
        replica_groups={{0,1}}, dimensions={1},
        backend_config={"collective_backend_config":{"kernel_strategy":"KERNEL_STRATEGY_TRITON_TWO_SHOT"}}
      gte0 = f32[4,4] get-tuple-element(ag), index=0
      gte1 = s32[1,4] get-tuple-element(ag), index=1
      ROOT t = (f32[4,4], s32[1,4]) tuple(gte0, gte1)
    }

    ENTRY e {
      arg0 = f32[4,2] parameter(0)
      arg1 = s32[1,2] parameter(1)
      ROOT fusion = (f32[4,4], s32[1,4]) fusion(arg0, arg1),
        kind=kCustom, calls=f
    })",
                                    {2, 1, 1, 2}));
  EXPECT_THAT(GetSchedule(tiled_computation),
              IsOkAndHolds(
                  MatchSchedule("d0 -> (pid mod 4) / 2, d1 -> pid mod 4 mod 2, "
                                "d2 -> pid mod 4, d3 -> (pid mod 4) * 2, "
                                "d4 -> pid / 4, d5 -> pid / 4, "
                                "num_pids=8, num_tiles=8")));
}

TEST_F(SchedulingTest, EffectiveTilesPerPidStridePartitionsTilesBijectively) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation four_tiles,
                       ParseAndTile(R"(
    fusion {
      p0 = f32[4,4]{1,0} parameter(0)
      ROOT neg = f32[4,4]{1,0} negate(p0)
    }
    ENTRY main {
      p0 = f32[4,4]{1,0} parameter(0)
      ROOT fusion = f32[4,4]{1,0} fusion(p0), kind=kLoop, calls=fusion
    })",
                                    {1, 4}));
  ASSERT_OK_AND_ASSIGN(Schedule s4, GetSchedule(four_tiles, 3));
  EXPECT_EQ(s4.num_tiles, 4);
  EXPECT_EQ(s4.num_pids, 2);
  EXPECT_EQ(s4.GetNumTilesPerPid(), 2);
  std::vector<int64_t> visited;
  for (int64_t pid = 0; pid < s4.num_pids; ++pid) {
    int64_t stride = s4.GetNumTilesPerPid();
    int64_t ub = std::min(s4.num_tiles - pid * stride, stride);
    for (int64_t tid = 0; tid < ub; ++tid) {
      visited.push_back(s4.GetPidExpr(0).Evaluate({pid, tid}));
    }
  }
  EXPECT_THAT(visited, testing::ElementsAre(0, 1, 2, 3));

  ASSERT_OK_AND_ASSIGN(const TiledHloComputation thirty_two_tiles,
                       ParseAndTile(R"(
    fusion {
      p0 = f32[32,4]{1,0} parameter(0)
      ROOT neg = f32[32,4]{1,0} negate(p0)
    }
    ENTRY main {
      p0 = f32[32,4]{1,0} parameter(0)
      ROOT fusion = f32[32,4]{1,0} fusion(p0), kind=kLoop, calls=fusion
    })",
                                    {1, 4}));
  EXPECT_THAT(GetSchedule(thirty_two_tiles, 9),
              IsOkAndHolds(MatchSchedule(
                  "d0 -> pid * 8 + tid, d1 -> 0, num_pids=4, num_tiles=32")));

  ASSERT_OK_AND_ASSIGN(const TiledHloComputation single_tile,
                       ParseAndTile(R"(
    fusion {
      p0 = f32[1,4]{1,0} parameter(0)
      ROOT neg = f32[1,4]{1,0} negate(p0)
    }
    ENTRY main {
      p0 = f32[1,4]{1,0} parameter(0)
      ROOT fusion = f32[1,4]{1,0} fusion(p0), kind=kLoop, calls=fusion
    })",
                                    {1, 4}));
  ASSERT_OK_AND_ASSIGN(Schedule s1, GetSchedule(single_tile, 2));
  EXPECT_EQ(s1.GetNumTilesPerPid(), 1);
  EXPECT_THAT(s1, MatchSchedule("d0 -> pid, d1 -> 0, num_pids=1, num_tiles=1"));
}

TEST_F(SchedulingTest, VariadicEffectiveTilesPerPidStrideCoversAllPeers) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation tiled_computation,
                       ParseAndTile(R"(
    f {
      p0 = f32[1,2] parameter(0)
      p1 = s32[1,1] parameter(1)
      ag = (f32[2,2], s32[2,1]) all-gather(p0, p1),
        replica_groups={{0,1}}, dimensions={0}
      gte0 = f32[2,2] get-tuple-element(ag), index=0
      gte1 = s32[2,1] get-tuple-element(ag), index=1
      ROOT t = (f32[2,2], s32[2,1]) tuple(gte0, gte1)
    }
    ENTRY e {
      arg0 = f32[1,2] parameter(0)
      arg1 = s32[1,1] parameter(1)
      ROOT fusion = (f32[2,2], s32[2,1]) fusion(arg0, arg1),
        kind=kCustom, calls=f
    })",
                                    {1, 1, 1, 1}));
  ASSERT_OK_AND_ASSIGN(Schedule schedule, GetSchedule(tiled_computation, 3));
  EXPECT_EQ(schedule.num_tiles, 4);
  EXPECT_EQ(schedule.num_pids, 2);
  EXPECT_EQ(schedule.GetNumTilesPerPid(), 2);
  std::vector<std::pair<int64_t, int64_t>> coords;
  for (int64_t pid = 0; pid < schedule.num_pids; ++pid) {
    int64_t stride = schedule.GetNumTilesPerPid();
    int64_t ub = std::min(schedule.num_tiles - pid * stride, stride);
    for (int64_t tid = 0; tid < ub; ++tid) {
      coords.push_back({schedule.GetPidExpr(0).Evaluate({pid, tid}),
                        schedule.GetPidExpr(1).Evaluate({pid, tid})});
    }
  }
  EXPECT_THAT(coords, testing::ElementsAre(std::pair<int64_t, int64_t>{0, 0},
                                           std::pair<int64_t, int64_t>{0, 1},
                                           std::pair<int64_t, int64_t>{1, 0},
                                           std::pair<int64_t, int64_t>{1, 1}));
}

TEST_F(SchedulingTest, Large64BitNumTilesPerPidAndInvalidInputs) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation large,
                       ParseAndTile(R"(
    fusion {
      p0 = f32[1,32,65536,65536]{3,2,1,0} parameter(0)
      ROOT exp = f32[1,32,65536,65536]{3,2,1,0} exponential(p0)
    }
    ENTRY main {
      p0 = f32[1,32,65536,65536]{3,2,1,0} parameter(0)
      ROOT fusion = f32[1,32,65536,65536]{3,2,1,0} fusion(p0),
        kind=kLoop, calls=fusion
    })",
                                    {1, 1, 1, 8}));
  ASSERT_OK_AND_ASSIGN(Schedule schedule, GetSchedule(large, int64_t{1} << 32));
  EXPECT_EQ(schedule.num_tiles, int64_t{1} << 34);
  EXPECT_EQ(schedule.num_pids, 4);
  EXPECT_EQ(schedule.GetNumTilesPerPid(), int64_t{1} << 32);

  EXPECT_THAT(GetSchedule(large, 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetSchedule(large, -1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetSchedule(large, std::numeric_limits<int64_t>::min()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(SchedulingTest, ZeroSizedAndOverflowingTileCounts) {
  for (absl::string_view hlo : {
           R"(
    fusion {
      p0 = f32[4,0]{1,0} parameter(0)
      ROOT neg = f32[4,0]{1,0} negate(p0)
    }
    ENTRY main {
      p0 = f32[4,0]{1,0} parameter(0)
      ROOT fusion = f32[4,0]{1,0} fusion(p0), kind=kLoop, calls=fusion
    })",
           R"(
    fusion {
      p0 = f32[0,4]{1,0} parameter(0)
      ROOT neg = f32[0,4]{1,0} negate(p0)
    }
    ENTRY main {
      p0 = f32[0,4]{1,0} parameter(0)
      ROOT fusion = f32[0,4]{1,0} fusion(p0), kind=kLoop, calls=fusion
    })"}) {
    ASSERT_OK_AND_ASSIGN(const TiledHloComputation empty,
                         ParseAndTile(hlo, {1, 1}));
    ASSERT_OK_AND_ASSIGN(Schedule schedule, GetSchedule(empty, 2));
    EXPECT_EQ(schedule.num_tiles, 0);
    EXPECT_EQ(schedule.num_pids, 0);
    EXPECT_EQ(schedule.GetNumTilesPerPid(), 0);
    EXPECT_THAT(schedule,
                MatchSchedule("d0 -> 0, d1 -> 0, num_pids=0, num_tiles=0"));
  }

  HloInstruction* root = ParseAndGetRoot(R"(
    fusion {
      p0 = f32[3037000500]{0} parameter(0)
      ROOT neg = f32[3037000500]{0} negate(p0)
    }
    ENTRY main {
      p0 = f32[3037000500]{0} parameter(0)
      ROOT fusion = f32[3037000500]{0} fusion(p0), kind=kLoop, calls=fusion
    })");
  auto adaptor = HloFusionAdaptor::ForInstruction(root);
  ASSERT_OK_AND_ASSIGN(auto zero_ts,
                       TilingSpace::Create(*adaptor, &mlir_context_));
  zero_ts->AppendDimension(root, 1, 3037000501,
                           TilingSpace::DimensionSemantics::kParallel);
  zero_ts->AppendDimension(root, 2, 0,
                           TilingSpace::DimensionSemantics::kParallel);
  ASSERT_THAT(zero_ts->AssignTileSizes({1, 1, 1}), absl_testing::IsOk());
  ASSERT_OK_AND_ASSIGN(TiledHloComputation zero_with_large_dims,
                       TiledHloComputation::Tile(*adaptor, std::move(zero_ts)));
  EXPECT_THAT(GetSchedule(zero_with_large_dims),
              IsOkAndHolds(MatchSchedule(
                  "d0 -> 0, d1 -> 0, d2 -> 0, num_pids=0, num_tiles=0")));
}

TEST_F(SchedulingTest, OverflowingTileCountsRejected) {
  HloInstruction* root = ParseAndGetRoot(R"(
    fusion {
      p0 = f32[3037000500]{0} parameter(0)
      ROOT neg = f32[3037000500]{0} negate(p0)
    }
    ENTRY main {
      p0 = f32[3037000500]{0} parameter(0)
      ROOT fusion = f32[3037000500]{0} fusion(p0), kind=kLoop, calls=fusion
    })");
  auto adaptor = HloFusionAdaptor::ForInstruction(root);
  ASSERT_OK_AND_ASSIGN(auto overflow_ts,
                       TilingSpace::Create(*adaptor, &mlir_context_));
  overflow_ts->AppendDimension(root, 1, 3037000501,
                               TilingSpace::DimensionSemantics::kParallel);
  ASSERT_THAT(overflow_ts->AssignTileSizes({1, 1}), absl_testing::IsOk());
  ASSERT_OK_AND_ASSIGN(
      TiledHloComputation overflow,
      TiledHloComputation::Tile(*adaptor, std::move(overflow_ts)));
  EXPECT_THAT(
      GetSchedule(overflow),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("overflow")));
}

TEST_F(SchedulingTest, VariadicAllGatherHandlesAllEmptyAndMixedEmptyOutputs) {
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation all_empty,
                       ParseAndTile(R"(
    f {
      p0 = f32[1,0] parameter(0)
      p1 = s32[1,0] parameter(1)
      ag = (f32[4,0], s32[4,0]) all-gather(p0, p1),
        replica_groups={{0,1,2,3}}, dimensions={0}
      gte0 = f32[4,0] get-tuple-element(ag), index=0
      gte1 = s32[4,0] get-tuple-element(ag), index=1
      ROOT t = (f32[4,0], s32[4,0]) tuple(gte0, gte1)
    }
    ENTRY e {
      arg0 = f32[1,0] parameter(0)
      arg1 = s32[1,0] parameter(1)
      ROOT fusion = (f32[4,0], s32[4,0]) fusion(arg0, arg1),
        kind=kCustom, calls=f
    })",
                                    {1, 1, 1, 1}));
  ASSERT_OK_AND_ASSIGN(Schedule empty_schedule, GetSchedule(all_empty));
  EXPECT_EQ(empty_schedule.GetNumTilesPerPid(), 0);
  EXPECT_THAT(
      empty_schedule,
      MatchSchedule(
          "d0 -> 0, d1 -> 0, d2 -> 0, d3 -> 0, num_pids=0, num_tiles=0"));

  ASSERT_OK_AND_ASSIGN(const TiledHloComputation mixed_empty,
                       ParseAndTile(R"(
    f {
      p0 = f32[1,2] parameter(0)
      p1 = s32[1,0] parameter(1)
      ag = (f32[4,2], s32[4,0]) all-gather(p0, p1),
        replica_groups={{0,1,2,3}}, dimensions={0}
      gte0 = f32[4,2] get-tuple-element(ag), index=0
      gte1 = s32[4,0] get-tuple-element(ag), index=1
      ROOT t = (f32[4,2], s32[4,0]) tuple(gte0, gte1)
    }
    ENTRY e {
      arg0 = f32[1,2] parameter(0)
      arg1 = s32[1,0] parameter(1)
      ROOT fusion = (f32[4,2], s32[4,0]) fusion(arg0, arg1),
        kind=kCustom, calls=f
    })",
                                    {1, 1, 1, 1}));
  ASSERT_OK_AND_ASSIGN(Schedule mixed_schedule, GetSchedule(mixed_empty));
  EXPECT_THAT(mixed_schedule,
              MatchSchedule("d0 -> pid / 2, d1 -> pid mod 2, d2 -> 0, d3 -> 0, "
                            "num_pids=8, num_tiles=8"));
  for (int64_t dim_id = 0; dim_id < 4; ++dim_id) {
    EXPECT_TRUE(static_cast<bool>(mixed_schedule.GetPidExpr(dim_id)));
  }
}

TEST_F(SchedulingTest, DegenerateDotAndRaggedContractingPermutation) {
  // Zero-contracting, zero-LHS-non-contracting batched dot has output rank
  // equal to num_batch + 1; must not query out-of-range output dimension.
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation no_lhs_nc_dot,
                       ParseAndTile(R"(
    fusion {
      lhs = f32[4]{0} parameter(0)
      rhs = f32[4,4096]{1,0} parameter(1)
      ROOT dot = f32[4,4096]{1,0} dot(lhs, rhs),
        lhs_batch_dims={0}, rhs_batch_dims={0}
    }
    ENTRY main {
      p0 = f32[4]{0} parameter(0)
      p1 = f32[4,4096]{1,0} parameter(1)
      ROOT fusion = f32[4,4096]{1,0} fusion(p0, p1), kind=kLoop, calls=fusion
    })",
                                    {1, 64}));
  EXPECT_THAT(
      GetSchedule(no_lhs_nc_dot),
      IsOkAndHolds(MatchSchedule(
          "d0 -> pid / 64, d1 -> pid mod 64, num_pids=256, num_tiles=256")));

  // Batched kRaggedContracting with group_sizes s32[2, 32] (G=32, M_avg=2 < 16)
  // swaps K (d2) and N (d3) using the trailing group count and parallel ranks.
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation ragged_contracting,
                       ParseAndTile(R"(
    fusion {
      lhs = f32[2,64,32]{2,1,0} parameter(0)
      rhs = f32[2,64,16]{2,1,0} parameter(1)
      gs = s32[2,32]{1,0} parameter(2)
      ROOT rd = f32[32,2,32,16]{3,2,1,0} ragged-dot(lhs, rhs, gs),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={1}, rhs_contracting_dims={1},
        lhs_ragged_dims={1}
    }
    ENTRY main {
      p0 = f32[2,64,32]{2,1,0} parameter(0)
      p1 = f32[2,64,16]{2,1,0} parameter(1)
      p2 = s32[2,32]{1,0} parameter(2)
      ROOT fusion = f32[32,2,32,16]{3,2,1,0} fusion(p0, p1, p2),
        kind=kLoop, calls=fusion
    })",
                                    {1, 1, 8, 8, 16}));
  ASSERT_OK_AND_ASSIGN(Schedule rc_schedule, GetSchedule(ragged_contracting));
  EXPECT_EQ(rc_schedule.num_tiles, 512);
  EXPECT_EQ(rc_schedule.num_pids, 512);
  // After swapping K (4 tiles) and N (2 tiles), K advances fastest (mod 4).
  EXPECT_EQ(rc_schedule.GetPidExpr(2).Evaluate({1}), 1);
  EXPECT_EQ(rc_schedule.GetPidExpr(3).Evaluate({1}), 0);
  EXPECT_EQ(rc_schedule.GetPidExpr(2).Evaluate({4}), 0);
  EXPECT_EQ(rc_schedule.GetPidExpr(3).Evaluate({4}), 1);
}

TEST_F(SchedulingTest, BatchedRaggedDotUsesTrailingGroupCountAndGuardsRank) {
  // group_sizes = s32[32, 2]: G = 2, M_avg = 64 / 2 = 32 >= N = 16 -> no swap.
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation no_swap,
                       ParseAndTile(R"(
    fusion {
      lhs = f32[32,64,32] parameter(0)
      rhs = f32[32,2,32,16] parameter(1)
      gs = s32[32,2] parameter(2)
      ROOT rd = f32[32,64,16] ragged-dot(lhs, rhs, gs),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={2},
        lhs_ragged_dims={1}, rhs_group_dims={1}
    }
    ENTRY main {
      p0 = f32[32,64,32] parameter(0)
      p1 = f32[32,2,32,16] parameter(1)
      p2 = s32[32,2] parameter(2)
      ROOT fusion = f32[32,64,16] fusion(p0, p1, p2), kind=kLoop, calls=fusion
    })",
                                    {1, 16, 8, 1, 16}));
  EXPECT_THAT(
      GetSchedule(no_swap),
      IsOkAndHolds(MatchSchedule(
          "d0 -> pid / 8, d1 -> (pid mod 8) / 2, d2 -> pid mod 8 mod 2, "
          "num_pids=256, num_tiles=256")));

  // group_sizes = s32[2, 32]: G = 32, M_avg = 64 / 32 = 2 < N = 16 -> swap.
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation swaps,
                       ParseAndTile(R"(
    fusion {
      lhs = f32[2,64,32] parameter(0)
      rhs = f32[2,32,32,16] parameter(1)
      gs = s32[2,32] parameter(2)
      ROOT rd = f32[2,64,16] ragged-dot(lhs, rhs, gs),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={2},
        lhs_ragged_dims={1}, rhs_group_dims={1}
    }
    ENTRY main {
      p0 = f32[2,64,32] parameter(0)
      p1 = f32[2,32,32,16] parameter(1)
      p2 = s32[2,32] parameter(2)
      ROOT fusion = f32[2,64,16] fusion(p0, p1, p2), kind=kLoop, calls=fusion
    })",
                                    {1, 16, 8, 1, 16}));
  EXPECT_THAT(
      GetSchedule(swaps),
      IsOkAndHolds(MatchSchedule(
          "d0 -> pid / 8, d1 -> pid mod 8 mod 4, d2 -> (pid mod 8) / 4, "
          "num_pids=16, num_tiles=16")));

  // Zero RHS non-contracting dims: output rank == num_batch + 1 (no swap/OOB).
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation zero_rhs_nc,
                       ParseAndTile(R"(
    fusion {
      lhs = f32[2,64,32] parameter(0)
      rhs = f32[2,8,32] parameter(1)
      gs = s32[2,8] parameter(2)
      ROOT rd = f32[2,64] ragged-dot(lhs, rhs, gs),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={2},
        lhs_ragged_dims={1}, rhs_group_dims={1}
    }
    ENTRY main {
      p0 = f32[2,64,32] parameter(0)
      p1 = f32[2,8,32] parameter(1)
      p2 = s32[2,8] parameter(2)
      ROOT fusion = f32[2,64] fusion(p0, p1, p2), kind=kLoop, calls=fusion
    })",
                                    {1, 16, 1, 16}));
  EXPECT_THAT(GetSchedule(zero_rhs_nc),
              IsOkAndHolds(MatchSchedule(
                  "d0 -> pid / 4, d1 -> pid mod 4, num_pids=8, num_tiles=8")));

  // Multiple LHS non-contracting dims: must not swap M with second LHS dim.
  ASSERT_OK_AND_ASSIGN(const TiledHloComputation multi_lhs_nc,
                       ParseAndTile(R"(
    fusion {
      lhs = f32[64,4,32] parameter(0)
      rhs = f32[8,32,128] parameter(1)
      gs = s32[8] parameter(2)
      ROOT rd = f32[64,4,128] ragged-dot(lhs, rhs, gs),
        lhs_contracting_dims={2}, rhs_contracting_dims={1},
        lhs_ragged_dims={0}, rhs_group_dims={0}
    }
    ENTRY main {
      p0 = f32[64,4,32] parameter(0)
      p1 = f32[8,32,128] parameter(1)
      p2 = s32[8] parameter(2)
      ROOT fusion = f32[64,4,128] fusion(p0, p1, p2), kind=kLoop, calls=fusion
    })",
                                    {16, 2, 32, 1, 16}));
  EXPECT_THAT(
      GetSchedule(multi_lhs_nc),
      IsOkAndHolds(MatchSchedule(
          "d0 -> pid / 8, d1 -> (pid mod 8) / 4, d2 -> pid mod 8 mod 4, "
          "num_pids=32, num_tiles=32")));
}

}  // namespace xla::gpu
