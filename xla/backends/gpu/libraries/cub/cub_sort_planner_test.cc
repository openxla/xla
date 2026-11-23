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

#include "xla/backends/gpu/libraries/cub/cub_sort_planner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_policy.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

testing::AssertionResult IsAlignedAndInBounds(const CubSortScratchRange& range,
                                              int64_t total_scratch) {
  if (range.size == 0) {
    return testing::AssertionSuccess();
  }
  if (range.offset % kCubSortScratchAlignmentBytes != 0) {
    return testing::AssertionFailure()
           << "Range offset " << range.offset << " is not aligned to "
           << kCubSortScratchAlignmentBytes;
  }
  if (range.offset < 0 || range.offset + range.size > total_scratch) {
    return testing::AssertionFailure()
           << "Range [" << range.offset << ", " << (range.offset + range.size)
           << ") exceeds total scratch [0, " << total_scratch << ")";
  }
  return testing::AssertionSuccess();
}

testing::AssertionResult AreDisjoint(const CubSortScratchRange& a,
                                     const CubSortScratchRange& b) {
  if (a.size == 0 || b.size == 0) {
    return testing::AssertionSuccess();
  }
  if (a.offset + a.size <= b.offset || b.offset + b.size <= a.offset) {
    return testing::AssertionSuccess();
  }
  return testing::AssertionFailure()
         << "Ranges [" << a.offset << ", " << (a.offset + a.size) << ") and ["
         << b.offset << ", " << (b.offset + b.size) << ") overlap";
}

TEST(CubSortPolicyTest, SmTiersSmallKeyTablesAndRegBoundScaling) {
  // Verify SM50 Maxwell segmented tuning vs SM60+ (and non-canonical SM52).
  const CubRadixSortPolicySpec sm50 =
      GetCubRadixSortPolicySpec(500, 4, 0, /*is_float32_key=*/false);
  const CubRadixSortPolicySpec sm52 =
      GetCubRadixSortPolicySpec(520, 4, 0, /*is_float32_key=*/false);
  const CubRadixSortPolicySpec sm60 =
      GetCubRadixSortPolicySpec(600, 4, 0, /*is_float32_key=*/false);
  EXPECT_EQ(sm50.segmented_nominal_items, 31);
  EXPECT_EQ(sm50.alt_segmented_nominal_threads, 256);
  EXPECT_TRUE(sm50.segmented_use_warp_transpose);
  EXPECT_EQ(sm52.segmented_nominal_items, 31);
  EXPECT_EQ(sm52.alt_segmented_nominal_threads, 256);
  EXPECT_TRUE(sm52.segmented_use_warp_transpose);
  EXPECT_EQ(sm60.segmented_nominal_items, 39);
  EXPECT_EQ(sm60.alt_segmented_nominal_threads, 384);
  EXPECT_FALSE(sm60.segmented_use_warp_transpose);

  // Verify Histogram & Onesweep tier transitions across SM60..SM100.
  const CubRadixSortPolicySpec sm70 =
      GetCubRadixSortPolicySpec(700, 4, 4, /*is_float32_key=*/false);
  const CubRadixSortPolicySpec sm80 =
      GetCubRadixSortPolicySpec(800, 4, 0, /*is_float32_key=*/false);
  const CubRadixSortPolicySpec sm90_u32 =
      GetCubRadixSortPolicySpec(900, 4, 0, /*is_float32_key=*/false);
  const CubRadixSortPolicySpec sm90_f32 =
      GetCubRadixSortPolicySpec(900, 4, 0, /*is_float32_key=*/true);
  EXPECT_EQ(sm70.histogram.block_threads, 256);
  EXPECT_EQ(sm70.histogram_nominal_4b_num_parts, 8);
  EXPECT_EQ(sm70.onesweep_nominal_items, 46);
  EXPECT_EQ(sm70.onesweep_rank_num_parts, 4);
  EXPECT_EQ(sm80.histogram.block_threads, 128);
  EXPECT_EQ(sm80.histogram_nominal_4b_num_parts, 1);
  EXPECT_EQ(sm80.onesweep.block_threads, 384);
  EXPECT_TRUE(sm80.onesweep_use_raking_memoize);
  EXPECT_EQ(sm90_u32.onesweep_nominal_items, 19);
  EXPECT_EQ(sm90_f32.onesweep_nominal_items, 18);

  // Verify SM90 vs SM100 small-key tables (including 16-byte value entries).
  const CubRadixSortPolicySpec sm90_u16_u16 =
      GetCubRadixSortPolicySpec(900, 2, 2, /*is_float32_key=*/false);
  const CubRadixSortPolicySpec sm100_u16_u16 =
      GetCubRadixSortPolicySpec(1000, 2, 2, /*is_float32_key=*/false);
  EXPECT_EQ(sm90_u16_u16.onesweep_nominal_threads, 448);
  EXPECT_EQ(sm90_u16_u16.onesweep_nominal_items, 16);
  EXPECT_EQ(sm100_u16_u16.onesweep_nominal_threads, 576);
  EXPECT_EQ(sm100_u16_u16.onesweep_nominal_items, 15);
  EXPECT_EQ(GetCubRadixSortPolicySpec(1000, 2, 16, /*is_float32_key=*/false)
                .onesweep_nominal_threads,
            576);
  EXPECT_EQ(GetCubRadixSortPolicySpec(1000, 1, 16, /*is_float32_key=*/false)
                .onesweep_nominal_threads,
            512);

  // Verify 8-byte RegBoundScaling halves single-tile items per thread.
  const CubRadixSortPolicySpec sm90_u64 =
      GetCubRadixSortPolicySpec(900, 8, 0, /*is_float32_key=*/false);
  EXPECT_EQ(sm90_u32.single_tile.tile_items(), 256 * 19);
  EXPECT_EQ(sm90_u64.single_tile.tile_items(), 256 * 9);
}

TEST(CubSortPlannerTest, EmptyAndTrivialCopyModes) {
  const se::CudaComputeCapability sm90(9, 0);

  for (int64_t batch_size : {int64_t{0}, int64_t{1}, int64_t{8}}) {
    ASSERT_OK_AND_ASSIGN(CubSortExecutionPlan empty_plan,
                         PlanCubSortExecution(sm90, F32, std::nullopt,
                                              /*num_items=*/0, batch_size));
    EXPECT_EQ(empty_plan.mode, CubSortMode::kEmpty);
    EXPECT_EQ(empty_plan.segment_size, 0);
    EXPECT_EQ(empty_plan.scratch_bytes, 0);
    EXPECT_TRUE(empty_plan.single_tile_launches.empty());
    EXPECT_TRUE(empty_plan.onesweep_launches.empty());
    EXPECT_TRUE(empty_plan.segmented_launches.empty());
  }

  ASSERT_OK_AND_ASSIGN(
      CubSortExecutionPlan copy_unbatched,
      PlanCubSortExecution(sm90, F32, U32, /*num_items=*/1, /*batch_size=*/1));
  EXPECT_EQ(copy_unbatched.mode, CubSortMode::kTrivialCopy);
  EXPECT_EQ(copy_unbatched.scratch_bytes, 0);

  ASSERT_OK_AND_ASSIGN(
      CubSortExecutionPlan copy_batched,
      PlanCubSortExecution(sm90, S32, std::nullopt, /*num_items=*/64,
                           /*batch_size=*/64));
  EXPECT_EQ(copy_batched.mode, CubSortMode::kTrivialCopy);
  EXPECT_EQ(copy_batched.scratch_bytes, 0);
  EXPECT_EQ(copy_batched.segment_size, 1);
}

TEST(CubSortPlannerTest, SingleTileBoundaryAndGridChunking) {
  const se::CudaComputeCapability sm80(8, 0);
  const CubRadixSortPolicySpec f32_policy =
      GetCubRadixSortPolicySpec(800, 4, 0, /*is_float32_key=*/true);
  const int32_t tile = f32_policy.single_tile.tile_items();
  EXPECT_EQ(tile, 256 * 19);

  // Exact tile boundary stays in kSingleTile with 0 scratch.
  ASSERT_OK_AND_ASSIGN(
      CubSortExecutionPlan single_unbatched,
      PlanCubSortExecution(sm80, F32, std::nullopt, tile, /*batch_size=*/1));
  EXPECT_EQ(single_unbatched.mode, CubSortMode::kSingleTile);
  EXPECT_EQ(single_unbatched.scratch_bytes, 0);
  ASSERT_EQ(single_unbatched.single_tile_launches.size(), 1);
  EXPECT_EQ(single_unbatched.single_tile_launches[0].segment_offset, 0);
  EXPECT_EQ(single_unbatched.single_tile_launches[0].grid_x, 1);

  // One past tile boundary switches to kOnesweep for batch_size == 1.
  ASSERT_OK_AND_ASSIGN(CubSortExecutionPlan onesweep,
                       PlanCubSortExecution(sm80, F32, std::nullopt, tile + 1,
                                            /*batch_size=*/1));
  EXPECT_EQ(onesweep.mode, CubSortMode::kOnesweep);
  EXPECT_GT(onesweep.scratch_bytes, 0);

  // Batched boundary switches from kSingleTile at `tile` to kSegmentedMultiPass
  // at `tile + 1`, and 8-byte dominant types use the halved 2304-item tile.
  ASSERT_OK_AND_ASSIGN(
      CubSortExecutionPlan batched_at_tile,
      PlanCubSortExecution(sm80, F32, std::nullopt, /*num_items=*/3LL * tile,
                           /*batch_size=*/3));
  EXPECT_EQ(batched_at_tile.mode, CubSortMode::kSingleTile);
  ASSERT_OK_AND_ASSIGN(
      CubSortExecutionPlan batched_past_tile,
      PlanCubSortExecution(sm80, F32, std::nullopt,
                           /*num_items=*/3LL * (tile + 1), /*batch_size=*/3));
  EXPECT_EQ(batched_past_tile.mode, CubSortMode::kSegmentedMultiPass);

  for (const auto& [key_type, val_type] :
       std::vector<std::pair<PrimitiveType, std::optional<PrimitiveType>>>{
           {U64, std::nullopt},
           {F32, U64},
       }) {
    ASSERT_OK_AND_ASSIGN(
        CubSortExecutionPlan single_8b,
        PlanCubSortExecution(sm80, key_type, val_type, 2304, /*batch_size=*/1));
    EXPECT_EQ(single_8b.mode, CubSortMode::kSingleTile);
    ASSERT_OK_AND_ASSIGN(
        CubSortExecutionPlan onesweep_8b,
        PlanCubSortExecution(sm80, key_type, val_type, 2305, /*batch_size=*/1));
    EXPECT_EQ(onesweep_8b.mode, CubSortMode::kOnesweep);
  }

  // Batched segments <= tile stay in kSingleTile and honor max_grid_x chunking.
  ASSERT_OK_AND_ASSIGN(
      CubSortExecutionPlan batched_chunked,
      PlanCubSortExecution(sm80, F32, U32, /*num_items=*/10 * 512,
                           /*batch_size=*/10, std::nullopt,
                           /*max_grid_x_override=*/4));
  EXPECT_EQ(batched_chunked.mode, CubSortMode::kSingleTile);
  EXPECT_EQ(batched_chunked.scratch_bytes, 0);
  ASSERT_EQ(batched_chunked.single_tile_launches.size(), 3);
  EXPECT_EQ(batched_chunked.single_tile_launches[0].segment_offset, 0);
  EXPECT_EQ(batched_chunked.single_tile_launches[0].grid_x, 4);
  EXPECT_EQ(batched_chunked.single_tile_launches[1].segment_offset, 4);
  EXPECT_EQ(batched_chunked.single_tile_launches[1].grid_x, 4);
  EXPECT_EQ(batched_chunked.single_tile_launches[2].segment_offset, 8);
  EXPECT_EQ(batched_chunked.single_tile_launches[2].grid_x, 2);
}

TEST(CubSortPlannerTest, OnesweepScratchAlignmentDisjointnessAndPingPong) {
  const se::CudaComputeCapability sm90(9, 0);

  // 8-bit keys-only uses 1 pass -> no keys_tmp needed, writes directly to kOut.
  ASSERT_OK_AND_ASSIGN(
      CubSortExecutionPlan u8_plan,
      PlanCubSortExecution(sm90, U8, std::nullopt, 20000, /*batch_size=*/1));
  EXPECT_EQ(u8_plan.mode, CubSortMode::kOnesweep);
  EXPECT_EQ(u8_plan.num_passes, 1);
  EXPECT_EQ(u8_plan.keys_tmp.size, 0);
  EXPECT_EQ(u8_plan.vals_tmp.size, 0);
  ASSERT_EQ(u8_plan.onesweep_launches.size(), 1);
  EXPECT_EQ(u8_plan.onesweep_launches[0].src_role, CubSortBufferRole::kIn);
  EXPECT_EQ(u8_plan.onesweep_launches[0].dst_role, CubSortBufferRole::kOut);
  EXPECT_EQ(u8_plan.onesweep_launches[0].bins_out, std::nullopt);

  // Multi-pass + multi-portion pairs sort (F32 keys, U64 values -> 4 passes).
  const CubRadixSortPolicySpec policy =
      GetCubRadixSortPolicySpec(900, 4, 8, /*is_float32_key=*/true);
  const int32_t portion_size = 2 * policy.onesweep.tile_items();
  const int64_t num_items = 2LL * portion_size + 513;

  ASSERT_OK_AND_ASSIGN(CubSortExecutionPlan plan,
                       PlanCubSortExecution(sm90, F32, U64, num_items,
                                            /*batch_size=*/1, portion_size));
  EXPECT_EQ(plan.mode, CubSortMode::kOnesweep);
  EXPECT_EQ(plan.num_passes, 4);
  EXPECT_EQ(plan.num_portions, 3);
  EXPECT_EQ(plan.onesweep_launches.size(), 12);
  EXPECT_EQ(plan.scratch_bytes % kCubSortScratchAlignmentBytes, 0);

  // Verify alignment and disjointness of all top-level scratch buffers.
  const std::vector<CubSortScratchRange> top_ranges = {
      plan.bins_raw, plan.bins_portion0, plan.lookback_max,
      plan.ctrs_all, plan.keys_tmp,      plan.vals_tmp,
  };
  for (const auto& r : top_ranges) {
    EXPECT_GT(r.size, 0);
    EXPECT_TRUE(IsAlignedAndInBounds(r, plan.scratch_bytes));
  }
  for (size_t i = 0; i < top_ranges.size(); ++i) {
    for (size_t j = i + 1; j < top_ranges.size(); ++j) {
      EXPECT_TRUE(AreDisjoint(top_ranges[i], top_ranges[j]));
    }
  }

  // Verify each Onesweep step gets a unique aligned counter slot inside
  // `ctrs_all`.
  for (size_t i = 0; i < plan.onesweep_launches.size(); ++i) {
    const CubSortScratchRange& ci = plan.onesweep_launches[i].ctr;
    EXPECT_TRUE(IsAlignedAndInBounds(ci, plan.scratch_bytes));
    EXPECT_GE(ci.offset, plan.ctrs_all.offset);
    EXPECT_LE(ci.offset + ci.size, plan.ctrs_all.offset + plan.ctrs_all.size);
    for (size_t j = i + 1; j < plan.onesweep_launches.size(); ++j) {
      EXPECT_TRUE(AreDisjoint(ci, plan.onesweep_launches[j].ctr));
    }
  }

  // Verify ping-pong continuity, portion partitioning, and inter-portion bin
  // chaining & disjointness across all portions.
  const std::vector<CubSortScratchRange> non_bin_ranges = {
      plan.bins_raw, plan.lookback_max, plan.ctrs_all,
      plan.keys_tmp, plan.vals_tmp,
  };
  for (int pass = 0; pass < plan.num_passes; ++pass) {
    int64_t expected_item_offset = 0;
    for (int64_t p = 0; p < plan.num_portions; ++p) {
      const CubSortOnesweepPassLaunch& step =
          plan.onesweep_launches[pass * plan.num_portions + p];
      EXPECT_EQ(step.pass, pass);
      EXPECT_EQ(step.portion, p);
      EXPECT_NE(step.src_role, step.dst_role);
      EXPECT_NE(step.dst_role, CubSortBufferRole::kIn);
      if (pass == 0) {
        EXPECT_EQ(step.src_role, CubSortBufferRole::kIn);
      } else {
        const CubSortOnesweepPassLaunch& prev_pass =
            plan.onesweep_launches[(pass - 1) * plan.num_portions + p];
        EXPECT_EQ(step.src_role, prev_pass.dst_role);
      }
      if (pass == plan.num_passes - 1) {
        EXPECT_EQ(step.dst_role, CubSortBufferRole::kOut);
      }
      EXPECT_EQ(step.current_bit, pass * 8);
      EXPECT_EQ(step.num_bits, 8);
      EXPECT_EQ(step.portion_item_offset, expected_item_offset);
      const int32_t expected_items =
          p + 1 < plan.num_portions ? portion_size : static_cast<int32_t>(513);
      EXPECT_EQ(step.portion_num_items, expected_items);
      const int32_t expected_grid_x =
          (expected_items + policy.onesweep.tile_items() - 1) /
          policy.onesweep.tile_items();
      EXPECT_EQ(step.grid_x, expected_grid_x);
      EXPECT_EQ(step.lookback.size, static_cast<int64_t>(expected_grid_x) *
                                        kCubSortOnesweepRadixDigits *
                                        static_cast<int64_t>(sizeof(int32_t)));
      EXPECT_LE(step.lookback.size, plan.lookback_max.size);
      expected_item_offset += step.portion_num_items;

      EXPECT_TRUE(IsAlignedAndInBounds(step.bins_in, plan.scratch_bytes));
      for (const auto& other : non_bin_ranges) {
        EXPECT_TRUE(AreDisjoint(step.bins_in, other));
      }
      if (p + 1 < plan.num_portions) {
        ASSERT_TRUE(step.bins_out.has_value());
        EXPECT_TRUE(IsAlignedAndInBounds(*step.bins_out, plan.scratch_bytes));
        EXPECT_TRUE(AreDisjoint(step.bins_in, *step.bins_out));
        for (const auto& other : non_bin_ranges) {
          EXPECT_TRUE(AreDisjoint(*step.bins_out, other));
        }
        const CubSortOnesweepPassLaunch& next =
            plan.onesweep_launches[pass * plan.num_portions + p + 1];
        EXPECT_EQ(step.bins_out->offset, next.bins_in.offset);
        EXPECT_EQ(step.bins_out->size, next.bins_in.size);
      } else {
        EXPECT_FALSE(step.bins_out.has_value());
      }
    }
    EXPECT_EQ(expected_item_offset, num_items);
  }
}

TEST(CubSortPlannerTest, OnesweepLargeInputAutomaticMultiPortion) {
  const se::CudaComputeCapability sm100(10, 0);
  // > 2^28 elements triggers multi-portion automatically without override.
  const int64_t large_n = (1LL << 28) + 100000;
  ASSERT_OK_AND_ASSIGN(CubSortExecutionPlan plan,
                       PlanCubSortExecution(sm100, U32, std::nullopt, large_n,
                                            /*batch_size=*/1));
  EXPECT_EQ(plan.mode, CubSortMode::kOnesweep);
  EXPECT_EQ(plan.num_portions, 2);
  EXPECT_EQ(plan.onesweep_launches.size(), 8);
  ASSERT_OK_AND_ASSIGN(
      int64_t scratch_size,
      ComputeCudaCubSortScratchSize(sm100, U32, std::nullopt, large_n, 1));
  EXPECT_EQ(scratch_size, plan.scratch_bytes);
}

TEST(CubSortPlannerTest, SegmentedMultiPassBitCoverageAndChunking) {
  const se::CudaComputeCapability sm90(9, 0);
  // Exercise 1B, 2B, 4B, and 8B key widths to cover all primary/alt schedules.
  for (const auto& [key_type, val_type, expected_passes] : std::vector<
           std::tuple<PrimitiveType, std::optional<PrimitiveType>, int>>{
           {U8, std::nullopt, 2},
           {U16, U16, 3},
           {F32, U32, 6},
           {U64, U64, 11},
       }) {
    ASSERT_OK_AND_ASSIGN(
        CubSortExecutionPlan plan,
        PlanCubSortExecution(sm90, key_type, val_type,
                             /*num_items=*/5 * 6000, /*batch_size=*/5,
                             std::nullopt, /*max_grid_x_override=*/2));
    EXPECT_EQ(plan.mode, CubSortMode::kSegmentedMultiPass);
    EXPECT_EQ(plan.num_passes, expected_passes);
    // 5 segments split by max_grid_x=2 -> 3 chunks per pass.
    ASSERT_EQ(plan.segmented_launches.size(), expected_passes * 3);
    EXPECT_TRUE(IsAlignedAndInBounds(plan.keys_tmp, plan.scratch_bytes));
    if (val_type.has_value()) {
      EXPECT_TRUE(IsAlignedAndInBounds(plan.vals_tmp, plan.scratch_bytes));
      EXPECT_TRUE(AreDisjoint(plan.keys_tmp, plan.vals_tmp));
    }

    const int expected_bits =
        key_type == U8 ? 8
                       : (key_type == U16 ? 16 : (key_type == F32 ? 32 : 64));
    int covered_bits = 0;
    const int64_t expected_offsets[3] = {0, 2, 4};
    const int32_t expected_grids[3] = {2, 2, 1};
    for (int pass = 0; pass < expected_passes; ++pass) {
      const CubSortSegmentedPassLaunch& first_chunk =
          plan.segmented_launches[pass * 3];
      EXPECT_EQ(first_chunk.current_bit, covered_bits);
      EXPECT_NE(first_chunk.src_role, first_chunk.dst_role);
      EXPECT_NE(first_chunk.dst_role, CubSortBufferRole::kIn);
      const int expected_radix_bits = first_chunk.use_alt_policy
                                          ? plan.policy.alt_segmented.radix_bits
                                          : plan.policy.segmented.radix_bits;
      EXPECT_EQ(first_chunk.pass_bits,
                std::min(expected_radix_bits, expected_bits - covered_bits));
      if (pass == 0) {
        EXPECT_EQ(first_chunk.src_role, CubSortBufferRole::kIn);
      } else {
        EXPECT_EQ(first_chunk.src_role,
                  plan.segmented_launches[(pass - 1) * 3].dst_role);
      }
      if (pass == expected_passes - 1) {
        EXPECT_EQ(first_chunk.dst_role, CubSortBufferRole::kOut);
      }
      for (int c = 0; c < 3; ++c) {
        const CubSortSegmentedPassLaunch& chunk =
            plan.segmented_launches[pass * 3 + c];
        EXPECT_EQ(chunk.pass, pass);
        EXPECT_EQ(chunk.segment_offset, expected_offsets[c]);
        EXPECT_EQ(chunk.grid_x, expected_grids[c]);
        EXPECT_EQ(chunk.use_alt_policy, first_chunk.use_alt_policy);
        EXPECT_EQ(chunk.src_role, first_chunk.src_role);
        EXPECT_EQ(chunk.dst_role, first_chunk.dst_role);
        EXPECT_EQ(chunk.current_bit, first_chunk.current_bit);
        EXPECT_EQ(chunk.pass_bits, first_chunk.pass_bits);
      }
      covered_bits += first_chunk.pass_bits;
    }
    EXPECT_EQ(covered_bits, expected_bits);
  }
}

TEST(CubSortPlannerTest, CanonicalPolicyTiersIgnoreMinorVersion) {
  for (int major : {5, 6, 7, 8, 9, 10, 12}) {
    ASSERT_OK_AND_ASSIGN(
        int64_t size_minor0,
        ComputeCudaCubSortScratchSize(se::CudaComputeCapability(major, 0), F32,
                                      U32, 100000, 1));
    ASSERT_OK_AND_ASSIGN(
        int64_t size_minor7,
        ComputeCudaCubSortScratchSize(se::CudaComputeCapability(major, 7), F32,
                                      U32, 100000, 1));
    EXPECT_EQ(size_minor0, size_minor7) << "major=" << major;
  }
}

TEST(CubSortPlannerTest, InvalidInputsReturnInvalidArgument) {
  const se::CudaComputeCapability sm90(9, 0);
  EXPECT_THAT(
      PlanCubSortExecution(sm90, PRED, std::nullopt, 100, 1),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Unsupported")));
  EXPECT_THAT(
      PlanCubSortExecution(sm90, F16, U32, 100, 1),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Unsupported")));
  EXPECT_THAT(
      PlanCubSortExecution(sm90, U32, TUPLE, 100, 1),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("non-array")));
  EXPECT_THAT(
      PlanCubSortExecution(sm90, U32, U8, 100, 1),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("bit width")));
  EXPECT_THAT(
      PlanCubSortExecution(sm90, F32, std::nullopt, -1, 1),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("non-negative")));
  EXPECT_THAT(
      PlanCubSortExecution(sm90, F32, std::nullopt, 0, -1),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("batch_size")));
  EXPECT_THAT(
      PlanCubSortExecution(sm90, F32, std::nullopt, 100, 0),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("positive")));
  EXPECT_THAT(
      PlanCubSortExecution(sm90, F32, std::nullopt, 100, 3),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("divisible")));
  EXPECT_THAT(PlanCubSortExecution(sm90, F32, std::nullopt, 100, 1,
                                   std::nullopt, /*max_grid_x_override=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("max_grid_x_override")));
  EXPECT_THAT(PlanCubSortExecution(sm90, F32, std::nullopt, 20000, 1,
                                   /*portion_size_override=*/123),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("portion_size_override")));
  EXPECT_THAT(PlanCubSortExecution(sm90, F32, std::nullopt, 100, 1,
                                   /*portion_size_override=*/-1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("portion_size_override")));
  const CubRadixSortPolicySpec f32_policy =
      GetCubRadixSortPolicySpec(900, 4, 0, /*is_float32_key=*/true);
  const int32_t oversized_portion =
      ((kCubSortMaxPortionLimit / f32_policy.onesweep.tile_items()) + 1) *
      f32_policy.onesweep.tile_items();
  EXPECT_THAT(PlanCubSortExecution(sm90, F32, std::nullopt, 20000, 1,
                                   oversized_portion),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("portion_size_override")));
  const int64_t huge_seg =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  EXPECT_THAT(
      PlanCubSortExecution(sm90, F32, std::nullopt, huge_seg * 2, 2),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("int32_t")));
  // Overflowing scratch byte computations must fail rather than wrap to 0.
  EXPECT_THAT(
      PlanCubSortExecution(sm90, F32, std::nullopt,
                           /*num_items=*/int64_t{1} << 61,
                           /*batch_size=*/int64_t{1} << 31),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("overflow")));
  EXPECT_THAT(
      PlanCubSortExecution(sm90, U64, U64,
                           /*num_items=*/int64_t{1} << 59,
                           /*batch_size=*/int64_t{1} << 29),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("overflow")));
}

}  // namespace
}  // namespace xla::gpu
