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

#include "xla/codegen/xtile/codegen/fusion_emitter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/ArithDialect.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCFDialect.h"
#include "mlir/Dialect/Tensor/IR/TensorDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"
#include "xla/codegen/tiling/tiled_hlo.h"
#include "xla/codegen/tiling/tiling_space.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/xtile_dialect.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"

namespace xla::xtile {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Not;

class FusionEmitterTest : public HloHardwareIndependentTestBase {
 protected:
  FusionEmitterTest() {
    mlir_context_.loadDialect<
        XTileDialect, XlaDialect, mlir::arith::ArithDialect,
        mlir::func::FuncDialect, mlir::math::MathDialect, mlir::scf::SCFDialect,
        mlir::tensor::TensorDialect, mlir::stablehlo::StablehloDialect>();
    RegisterSymbolicExprStorage(&mlir_context_);
  }

  absl::StatusOr<std::string> EmitToString(
      absl::string_view hlo_text, absl::Span<const int64_t> tile_sizes,
      int64_t num_tiles_per_pid,
      const CollectiveEntrySpec& collective_spec = {}, bool is_cpu = false) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
    const auto* fusion = Cast<HloFusionInstruction>(
        module->entry_computation()->root_instruction());
    auto adaptor = HloFusionAdaptor::ForInstruction(fusion);
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<gpu::TilingSpace> tiling_space,
                          gpu::TilingSpace::Create(*adaptor, &mlir_context_));
    ABSL_RETURN_IF_ERROR(tiling_space->AssignTileSizes(tile_sizes));
    ABSL_ASSIGN_OR_RETURN(
        gpu::TiledHloComputation tiled_computation,
        gpu::TiledHloComputation::Tile(*adaptor, std::move(tiling_space)));
    tiled_computation.Simplify();
    tiled_computation.SortInstructionsPostOrder();
    ABSL_ASSIGN_OR_RETURN(
        mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
        EmitXTileModule("xtile_fn", *fusion, tiled_computation, mlir_context_,
                        collective_spec, /*gpu_cc=*/std::nullopt,
                        num_tiles_per_pid, is_cpu));
    std::string out;
    llvm::raw_string_ostream os(out);
    mlir_module->print(os);
    return out;
  }

  mlir::MLIRContext mlir_context_;
};

TEST_F(FusionEmitterTest, Supports64BitNumTilesPerPidWithoutTruncation) {
  // Regression for b/589641911: 2^34 parallel tiles over 4 work groups yields
  // num_tiles_per_pid == 2^32, which previously narrowed to int 0 and divided
  // by zero in GetSchedule.
  ASSERT_OK_AND_ASSIGN(std::string ir_pow32,
                       EmitToString(R"(
    f {
      p0 = f32[1,32,65536,65536]{3,2,1,0} parameter(0)
      ROOT exp = f32[1,32,65536,65536]{3,2,1,0} exponential(p0)
    }
    ENTRY main {
      p0 = f32[1,32,65536,65536]{3,2,1,0} parameter(0)
      ROOT fusion = f32[1,32,65536,65536]{3,2,1,0} fusion(p0),
        kind=kLoop, calls=f
    })",
                                    {1, 1, 1, 8}, int64_t{1} << 32, {},
                                    /*is_cpu=*/true));
  EXPECT_THAT(RunFileCheck(ir_pow32, R"(
    CHECK: xtile.entry_func @xtile_fn
    CHECK-SAME: xla.range = [0 : index, 3 : index]
    CHECK: scf.for
    CHECK: xtile.insert
  )"),
              IsOkAndHolds(true));
  EXPECT_THAT(ir_pow32, HasSubstr("4294967296"));

  // num_tiles_per_pid == 2^31 previously narrowed to -2147483648, emitting
  // negative xla.range and dropping the tile loop.
  ASSERT_OK_AND_ASSIGN(std::string ir_pow31,
                       EmitToString(R"(
    f {
      p0 = f32[1,16,65536,65536]{3,2,1,0} parameter(0)
      ROOT exp = f32[1,16,65536,65536]{3,2,1,0} exponential(p0)
    }
    ENTRY main {
      p0 = f32[1,16,65536,65536]{3,2,1,0} parameter(0)
      ROOT fusion = f32[1,16,65536,65536]{3,2,1,0} fusion(p0),
        kind=kLoop, calls=f
    })",
                                    {1, 1, 1, 8}, int64_t{1} << 31, {},
                                    /*is_cpu=*/true));
  EXPECT_THAT(RunFileCheck(ir_pow31, R"(
    CHECK: xtile.entry_func @xtile_fn
    CHECK-SAME: xla.range = [0 : index, 3 : index]
    CHECK: scf.for
  )"),
              IsOkAndHolds(true));
  EXPECT_THAT(ir_pow31, HasSubstr("2147483648"));

  // num_tiles_per_pid == 2^32 + 1 previously narrowed to 1 and dropped upper
  // tiles.
  ASSERT_OK_AND_ASSIGN(std::string ir_pow32_plus_1,
                       EmitToString(R"(
    f {
      p0 = f32[17179869188]{0} parameter(0)
      ROOT neg = f32[17179869188]{0} negate(p0)
    }
    ENTRY main {
      p0 = f32[17179869188]{0} parameter(0)
      ROOT fusion = f32[17179869188]{0} fusion(p0), kind=kLoop, calls=f
    })",
                                    {1}, (int64_t{1} << 32) + 1, {},
                                    /*is_cpu=*/true));
  EXPECT_THAT(ir_pow32_plus_1, HasSubstr("4294967297"));
  EXPECT_THAT(ir_pow32_plus_1, HasSubstr("scf.for"));
}

TEST_F(FusionEmitterTest, ValidatesCollectiveSpecAndNumTilesPerPid) {
  constexpr absl::string_view kHlo = R"(
    f {
      p0 = f32[16]{0} parameter(0)
      ROOT neg = f32[16]{0} negate(p0)
    }
    ENTRY main {
      p0 = f32[16]{0} parameter(0)
      ROOT fusion = f32[16]{0} fusion(p0), kind=kLoop, calls=f
    })";
  EXPECT_THAT(EmitToString(kHlo, {8}, 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(EmitToString(kHlo, {8}, -3),
              StatusIs(absl::StatusCode::kInvalidArgument));

  CollectiveEntrySpec zero_world;
  zero_world.has_signal_and_rank = true;
  zero_world.world_size = 0;
  EXPECT_THAT(EmitToString(kHlo, {8}, 1, zero_world),
              StatusIs(absl::StatusCode::kInvalidArgument));

  CollectiveEntrySpec neg_world;
  neg_world.has_signal_and_rank = true;
  neg_world.world_size = -2;
  EXPECT_THAT(EmitToString(kHlo, {8}, 1, neg_world),
              StatusIs(absl::StatusCode::kInvalidArgument));

  CollectiveEntrySpec neg_scratch;
  neg_scratch.has_signal_and_rank = true;
  neg_scratch.world_size = 4;
  neg_scratch.num_scratch_buffers = -1;
  EXPECT_THAT(EmitToString(kHlo, {8}, 1, neg_scratch),
              StatusIs(absl::StatusCode::kInvalidArgument));

  CollectiveEntrySpec orphan_scratch;
  orphan_scratch.has_signal_and_rank = false;
  orphan_scratch.num_scratch_buffers = 1;
  EXPECT_THAT(EmitToString(kHlo, {8}, 1, orphan_scratch),
              StatusIs(absl::StatusCode::kInvalidArgument));

  CollectiveEntrySpec valid_spec;
  valid_spec.has_signal_and_rank = true;
  valid_spec.world_size = 4;
  valid_spec.num_scratch_buffers = 2;
  ASSERT_OK_AND_ASSIGN(std::string ir, EmitToString(kHlo, {8}, 1, valid_spec));
  EXPECT_THAT(ir, HasSubstr("xla.range = [0 : index, 3 : index]"));
}

TEST_F(FusionEmitterTest, ReconcilesEffectiveStrideAndHandlesZeroTileFusion) {
  ASSERT_OK_AND_ASSIGN(std::string ir_4_3, EmitToString(R"(
    f {
      p0 = f32[4,4]{1,0} parameter(0)
      ROOT neg = f32[4,4]{1,0} negate(p0)
    }
    ENTRY main {
      p0 = f32[4,4]{1,0} parameter(0)
      ROOT fusion = f32[4,4]{1,0} fusion(p0), kind=kLoop, calls=f
    })",
                                                        {1, 4}, 3));
  EXPECT_THAT(RunFileCheck(ir_4_3, R"(
    CHECK-DAG: (pid) -> (min(pid * -2 + 4, 2)), domain: pid in [0, 1]
    CHECK-DAG: (pid, tid) -> (pid * 2 + tid), domain: pid in [0, 1], tid in [0, 1]
    CHECK: scf.for
  )"),
              IsOkAndHolds(true));

  ASSERT_OK_AND_ASSIGN(std::string ir_1_2, EmitToString(R"(
    f {
      p0 = f32[1,4]{1,0} parameter(0)
      ROOT neg = f32[1,4]{1,0} negate(p0)
    }
    ENTRY main {
      p0 = f32[1,4]{1,0} parameter(0)
      ROOT fusion = f32[1,4]{1,0} fusion(p0), kind=kLoop, calls=f
    })",
                                                        {1, 4}, 2));
  EXPECT_THAT(ir_1_2, Not(HasSubstr("scf.for")));

  ASSERT_OK_AND_ASSIGN(std::string ir_empty, EmitToString(R"(
    f {
      p0 = f32[4,0]{1,0} parameter(0)
      ROOT neg = f32[4,0]{1,0} negate(p0)
    }
    ENTRY main {
      p0 = f32[4,0]{1,0} parameter(0)
      ROOT fusion = f32[4,0]{1,0} fusion(p0), kind=kLoop, calls=f
    })",
                                                          {2, 1}, 1, {},
                                                          /*is_cpu=*/true));
  EXPECT_THAT(RunFileCheck(ir_empty, R"(
    CHECK: xtile.entry_func @xtile_fn
    CHECK-NOT: xla.range
    CHECK-NOT: xtile.extract
    CHECK-NOT: xtile.insert
    CHECK: xtile.return
  )"),
              IsOkAndHolds(true));
}

TEST_F(FusionEmitterTest, GroupSizeRemapsProgramIdOnlyForValidSingleTileGrid) {
  constexpr absl::string_view kRaggedNonContracting = R"(
    f {
      lhs = f32[64,32] parameter(0)
      rhs = f32[4,32,32] parameter(1)
      gs = s32[4] parameter(2)
      ROOT rd = f32[64,32] ragged-dot(lhs, rhs, gs),
        lhs_contracting_dims={1}, rhs_contracting_dims={1},
        lhs_ragged_dims={0}, rhs_group_dims={0}
    }
    ENTRY main {
      p0 = f32[64,32] parameter(0)
      p1 = f32[4,32,32] parameter(1)
      p2 = s32[4] parameter(2)
      ROOT fusion = f32[64,32] fusion(p0, p1, p2), kind=kCustom, calls=f,
        backend_config={"fusion_backend_config":{"kind":"__triton",
          "block_level_fusion_config":{"output_tiles":[{"sizes":["16","16"]}],
            "group_size":"2","num_warps":"1","num_ctas":"1","num_stages":"1"}}}
    })";
  // Single tile per pid: remapped pid must drive xla.apply_indexing.
  ASSERT_OK_AND_ASSIGN(std::string ir_single,
                       EmitToString(kRaggedNonContracting, {16, 16, 1, 16},
                                    /*num_tiles_per_pid=*/1));
  EXPECT_THAT(RunFileCheck(ir_single, R"(
    CHECK: %[[MIN:.*]] = arith.minsi
    CHECK: arith.remsi %{{.*}}, %[[MIN]]
    CHECK: arith.divsi %{{.*}}, %[[MIN]]
    CHECK: %[[REMAPPED:.*]] = arith.index_cast %{{.*}} : i32 to index
    CHECK: xla.apply_indexing #{{.*}}(%[[REMAPPED]]
  )"),
              IsOkAndHolds(true));

  // Multiple tiles per pid: must not remap local scf.for induction variable.
  ASSERT_OK_AND_ASSIGN(std::string ir_multi,
                       EmitToString(kRaggedNonContracting, {16, 16, 1, 16},
                                    /*num_tiles_per_pid=*/2));
  EXPECT_THAT(ir_multi, HasSubstr("scf.for"));
  EXPECT_THAT(ir_multi, Not(HasSubstr("arith.minsi")));

  // kRaggedContracting with BLOCK_G == 1 remaps program_id; BLOCK_G > 1 fails
  // cleanly instead of crashing or emitting out-of-bounds group indices.
  constexpr absl::string_view kRaggedContracting = R"(
    f {
      lhs = f32[64,32] parameter(0)
      rhs = f32[64,32] parameter(1)
      gs = s32[4] parameter(2)
      ROOT rd = f32[4,32,32] ragged-dot(lhs, rhs, gs),
        lhs_contracting_dims={0}, rhs_contracting_dims={0},
        lhs_ragged_dims={0}
    }
    ENTRY main {
      p0 = f32[64,32] parameter(0)
      p1 = f32[64,32] parameter(1)
      p2 = s32[4] parameter(2)
      ROOT fusion = f32[4,32,32] fusion(p0, p1, p2), kind=kCustom, calls=f,
        backend_config={"fusion_backend_config":{"kind":"__triton",
          "block_level_fusion_config":{"output_tiles":[{"sizes":["1","16","16"]}],
            "group_size":"2","num_warps":"1","num_ctas":"1","num_stages":"1"}}}
    })";
  ASSERT_OK_AND_ASSIGN(std::string ir_contracting,
                       EmitToString(kRaggedContracting, {1, 16, 16, 16},
                                    /*num_tiles_per_pid=*/1));
  EXPECT_THAT(RunFileCheck(ir_contracting, R"(
    CHECK: arith.minsi
    CHECK: %[[REMAPPED:.*]] = arith.index_cast %{{.*}} : i32 to index
    CHECK: xla.apply_indexing #{{.*}}(%[[REMAPPED]]
  )"),
              IsOkAndHolds(true));
  EXPECT_THAT(EmitToString(kRaggedContracting, {2, 16, 16, 16},
                           /*num_tiles_per_pid=*/1),
              Not(absl_testing::IsOk()));
}

TEST_F(FusionEmitterTest, GroupSizeRemappingGuardsDegenerateAndOverflowGrids) {
  // BLOCK_M == 0 (padded to 1 in tiling_space): must not divide by zero.
  ASSERT_OK_AND_ASSIGN(std::string ir_zero_block,
                       EmitToString(R"(
    f {
      lhs = f32[16,16] parameter(0)
      rhs = f32[2,16,16] parameter(1)
      gs = s32[2] parameter(2)
      ROOT rd = f32[16,16] ragged-dot(lhs, rhs, gs),
        lhs_contracting_dims={1}, rhs_contracting_dims={1},
        lhs_ragged_dims={0}, rhs_group_dims={0}
    }
    ENTRY main {
      p0 = f32[16,16] parameter(0)
      p1 = f32[2,16,16] parameter(1)
      p2 = s32[2] parameter(2)
      ROOT fusion = f32[16,16] fusion(p0, p1, p2), kind=kCustom, calls=f,
        backend_config={"fusion_backend_config":{"kind":"__triton",
          "block_level_fusion_config":{"output_tiles":[{"sizes":["0","16"]}],
            "group_size":"2","num_warps":"1","num_ctas":"1","num_stages":"1"}}}
    })",
                                    {1, 16, 1, 16}, 1));
  EXPECT_THAT(ir_zero_block, Not(HasSubstr("arith.minsi")));

  // Non-power-of-two BLOCK_M == 3 (padded to 4 in tiling_space): tile counts
  // disagree, so remapping must bail out rather than producing OOB indices.
  ASSERT_OK_AND_ASSIGN(std::string ir_non_pow2,
                       EmitToString(R"(
    f {
      lhs = f32[16,16] parameter(0)
      rhs = f32[2,16,32] parameter(1)
      gs = s32[2] parameter(2)
      ROOT rd = f32[16,32] ragged-dot(lhs, rhs, gs),
        lhs_contracting_dims={1}, rhs_contracting_dims={1},
        lhs_ragged_dims={0}, rhs_group_dims={0}
    }
    ENTRY main {
      p0 = f32[16,16] parameter(0)
      p1 = f32[2,16,32] parameter(1)
      p2 = s32[2] parameter(2)
      ROOT fusion = f32[16,32] fusion(p0, p1, p2), kind=kCustom, calls=f,
        backend_config={"fusion_backend_config":{"kind":"__triton",
          "block_level_fusion_config":{"output_tiles":[{"sizes":["3","16"]}],
            "group_size":"2","num_warps":"1","num_ctas":"1","num_stages":"1"}}}
    })",
                                    {4, 16, 1, 16}, 1));
  EXPECT_THAT(ir_non_pow2, Not(HasSubstr("arith.minsi")));

  // group_size * num_pid_n == 65536 * 65536 > INT32_MAX: must bail out instead
  // of wrapping the 32-bit divisor to 0.
  ASSERT_OK_AND_ASSIGN(std::string ir_i32_overflow,
                       EmitToString(R"(
    f {
      lhs = f32[64,16] parameter(0)
      rhs = f32[2,16,1048576] parameter(1)
      gs = s32[2] parameter(2)
      ROOT rd = f32[64,1048576] ragged-dot(lhs, rhs, gs),
        lhs_contracting_dims={1}, rhs_contracting_dims={1},
        lhs_ragged_dims={0}, rhs_group_dims={0}
    }
    ENTRY main {
      p0 = f32[64,16] parameter(0)
      p1 = f32[2,16,1048576] parameter(1)
      p2 = s32[2] parameter(2)
      ROOT fusion = f32[64,1048576] fusion(p0, p1, p2), kind=kCustom, calls=f,
        backend_config={"fusion_backend_config":{"kind":"__triton",
          "block_level_fusion_config":{"output_tiles":[{"sizes":["16","16"]}],
            "group_size":"65536","num_warps":"1","num_ctas":"1",
            "num_stages":"1"}}}
    })",
                                    {16, 16, 1, 16}, 1));
  EXPECT_THAT(ir_i32_overflow, Not(HasSubstr("arith.minsi")));

  // Non-root ragged-dot with epilogue must not remap unless verified at root.
  ASSERT_OK_AND_ASSIGN(std::string ir_non_root,
                       EmitToString(R"(
    f {
      lhs = f32[64,16] parameter(0)
      rhs = f32[2,16,32] parameter(1)
      gs = s32[2] parameter(2)
      rd = f32[64,32] ragged-dot(lhs, rhs, gs),
        lhs_contracting_dims={1}, rhs_contracting_dims={1},
        lhs_ragged_dims={0}, rhs_group_dims={0}
      ROOT neg = f32[64,32] negate(rd)
    }
    ENTRY main {
      p0 = f32[64,16] parameter(0)
      p1 = f32[2,16,32] parameter(1)
      p2 = s32[2] parameter(2)
      ROOT fusion = f32[64,32] fusion(p0, p1, p2), kind=kCustom, calls=f,
        backend_config={"fusion_backend_config":{"kind":"__triton",
          "block_level_fusion_config":{"output_tiles":[{"sizes":["16","16"]}],
            "group_size":"2","num_warps":"1","num_ctas":"1","num_stages":"1"}}}
    })",
                                    {16, 16, 1, 16}, 1));
  EXPECT_THAT(ir_non_root, Not(HasSubstr("arith.minsi")));
}

}  // namespace
}  // namespace xla::xtile
