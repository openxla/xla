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

#include "xla/backends/gpu/transforms/custom_call_scratch_assigner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_constants.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registry.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_scratch_context.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/ffi/attributes.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/gpu/gpu_memory_space_assignment.h"
#include "xla/service/gpu_topology.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace op = ::xla::testing::opcode_matchers;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Optional;
using ::testing::Pair;

constexpr absl::string_view kScratchTarget = "xla.gpu.test_scratch_assigner";
constexpr absl::string_view kNoScratchTarget = "xla.gpu.test_no_scratch";

// A static array shape with the default layout in `memory_space`.
Shape Scratch(PrimitiveType type, absl::Span<const int64_t> dims,
              int64_t memory_space = 0) {
  Shape shape = ShapeUtil::MakeShapeWithDescendingLayout(type, dims);
  shape.mutable_layout()->set_memory_space(memory_space);
  return shape;
}

using TestScratchHandler = std::function<absl::StatusOr<std::vector<Shape>>(
    const HloCustomCallInstruction&, const NativeCustomCallScratchContext&)>;

// The scratch handler registered for `kScratchTarget` forwards to this, so
// every test can install its own behavior.
TestScratchHandler& CurrentScratchHandler() {
  static auto* handler = new TestScratchHandler();
  return *handler;
}

void RegisterTestHandlers() {
  static const bool registered = [] {
    auto emit_thunks = [](const HloCustomCallInstruction&, const auto&) {
      return absl::UnimplementedError("not used in this test");
    };
    CHECK_OK(NativeCustomCallHandlerRegistry::GetGlobal().Register(
        kScratchTarget, NativeCustomCallHandlerBundle{
                            /*emit_thunks=*/emit_thunks,
                            /*request_scratch_buffers=*/
                            [](const HloCustomCallInstruction& instr,
                               const NativeCustomCallScratchContext& ctx) {
                              return CurrentScratchHandler()(instr, ctx);
                            }}));
    CHECK_OK(NativeCustomCallHandlerRegistry::GetGlobal().Register(
        kNoScratchTarget, emit_thunks));
    return true;
  }();
  (void)registered;
}

class CustomCallScratchAssignerTest : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    HloHardwareIndependentTestBase::SetUp();
    RegisterTestHandlers();
    ASSERT_OK_AND_ASSIGN(stream_executor::GpuTargetConfigProto proto,
                         GetGpuTargetConfig(GpuModel::H100_SXM));
    ASSERT_OK_AND_ASSIGN(GpuTargetConfig target_config,
                         GpuTargetConfig::FromProto(proto));
    topology_ = std::make_unique<GpuTopology>(
        GetSingleDeviceGpuTopology(target_config.platform_name, target_config));
    // By default, every custom call asks for an f32[16] buffer in default
    // memory and an s32[4] buffer in collective memory.
    SetScratchShapes(
        {Scratch(F32, {16}),
         Scratch(S32, {4},
                 static_cast<int64_t>(MemorySpaceColor::kCollective))});
  }

  void TearDown() override {
    CurrentScratchHandler() = nullptr;
    HloHardwareIndependentTestBase::TearDown();
  }

  void SetScratchShapes(std::vector<Shape> shapes) {
    CurrentScratchHandler() = [shapes = std::move(shapes)](
                                  const HloCustomCallInstruction&,
                                  const NativeCustomCallScratchContext&) {
      return shapes;
    };
  }

  absl::StatusOr<bool> RunPass(HloModule* module) {
    CustomCallScratchAssigner pass(topology_.get(), &mlir_context_);
    return RunHloPass(&pass, module);
  }

  static const HloCustomCallInstruction* FindCustomCall(
      const HloModule& module, absl::string_view name) {
    const HloInstruction* instr = FindInstruction(&module, name);
    return instr == nullptr ? nullptr
                            : DynCast<HloCustomCallInstruction>(instr);
  }

  std::unique_ptr<GpuTopology> topology_;
  mlir::MLIRContext mlir_context_;
};

TEST_F(CustomCallScratchAssignerTest, AppendsScratchToArrayResult) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p = f32[8] parameter(0)
      cc = f32[8] custom-call(p),
        custom_call_target="xla.gpu.test_scratch_assigner"
      ROOT neg = f32[8] negate(cc)
    })"));
  EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(true));

  const HloCustomCallInstruction* cc = FindCustomCall(*module, "cc");
  ASSERT_NE(cc, nullptr);
  EXPECT_EQ(cc->shape().ToString(/*print_layout=*/true),
            "(f32[8]{0}, f32[16]{0}, s32[4]{0:S(7)})");
  EXPECT_THAT(
      cc->get_frontend_attribute(kNativeCustomCallNumScratchBuffersAttr),
      Optional(std::string("2")));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Negate(op::GetTupleElement(cc, 0)));
  EXPECT_THAT(module->entry_computation()->root_instruction()->shape(),
              ShapeUtil::MakeShape(F32, {8}));
}

TEST_F(CustomCallScratchAssignerTest, AppendsScratchToTupleResult) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p = f32[8] parameter(0)
      cc = (f32[8], s8[2]) custom-call(p),
        custom_call_target="xla.gpu.test_scratch_assigner"
      a = f32[8] get-tuple-element(cc), index=0
      b = s8[2] get-tuple-element(cc), index=1
      ROOT t = (f32[8], s8[2]) tuple(a, b)
    })"));
  EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(true));

  const HloCustomCallInstruction* cc = FindCustomCall(*module, "cc");
  ASSERT_NE(cc, nullptr);
  EXPECT_EQ(cc->shape().ToString(/*print_layout=*/true),
            "((f32[8]{0}, s8[2]{0}), f32[16]{0}, s32[4]{0:S(7)})");
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::GetTupleElement(op::GetTupleElement(cc, 0), 0),
                        op::GetTupleElement(op::GetTupleElement(cc, 0), 1)));
}

TEST_F(CustomCallScratchAssignerTest,
       WholeTupleConsumerReceivesGetTupleElementZero) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p = f32[8] parameter(0)
      ROOT cc = (f32[8], s8[2]) custom-call(p),
        custom_call_target="xla.gpu.test_scratch_assigner"
    })"));
  EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(true));

  const HloCustomCallInstruction* cc = FindCustomCall(*module, "cc");
  ASSERT_NE(cc, nullptr);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::GetTupleElement(cc, 0));
  EXPECT_EQ(module->entry_computation()->root_instruction()->shape(),
            ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {8}),
                                       ShapeUtil::MakeShape(S8, {2})}));
}

TEST_F(CustomCallScratchAssignerTest, HandlesRootCustomCallInWhileBody) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    body {
      p = f32[8] parameter(0)
      ROOT cc = f32[8] custom-call(p),
        custom_call_target="xla.gpu.test_scratch_assigner"
    }
    cond {
      p = f32[8] parameter(0)
      ROOT c = pred[] constant(false)
    }
    ENTRY e {
      p = f32[8] parameter(0)
      ROOT w = f32[8] while(p), condition=cond, body=body
    })"));
  EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(true));

  const HloCustomCallInstruction* cc = FindCustomCall(*module, "cc");
  ASSERT_NE(cc, nullptr);
  EXPECT_TRUE(cc->shape().IsTuple());
  const HloComputation* body = cc->parent();
  EXPECT_EQ(body->name(), "body");
  EXPECT_THAT(body->root_instruction(), op::GetTupleElement(cc, 0));
  EXPECT_THAT(body->root_instruction()->shape(),
              ShapeUtil::MakeShape(F32, {8}));
}

TEST_F(CustomCallScratchAssignerTest, KeepsLayoutsAndDefaultsMissingLayout) {
  Shape with_layout = ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3}, {0, 1});
  with_layout.mutable_layout()->set_memory_space(
      static_cast<int64_t>(MemorySpaceColor::kCollective));
  Shape without_layout = ShapeUtil::MakeShape(U8, {4, 5});
  without_layout.clear_layout();
  SetScratchShapes({with_layout, without_layout});

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      ROOT cc = f32[8] custom-call(),
        custom_call_target="xla.gpu.test_scratch_assigner"
    })"));
  EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(true));

  const HloCustomCallInstruction* cc = FindCustomCall(*module, "cc");
  ASSERT_NE(cc, nullptr);
  EXPECT_EQ(cc->shape().ToString(/*print_layout=*/true),
            "(f32[8]{0}, f32[2,3]{0,1:S(7)}, u8[4,5]{1,0})");
}

TEST_F(CustomCallScratchAssignerTest, RemapsAliasingOfArrayResult) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p = f32[8] parameter(0)
      ROOT cc = f32[8] custom-call(p),
        custom_call_target="xla.gpu.test_scratch_assigner",
        output_to_operand_aliasing={{}: (0, {})}
    })"));
  EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(true));

  const HloCustomCallInstruction* cc = FindCustomCall(*module, "cc");
  ASSERT_NE(cc, nullptr);
  EXPECT_THAT(cc->output_operand_aliasing(),
              ElementsAre(Pair(ShapeIndex{0}, Pair(0, ShapeIndex{}))));
}

TEST_F(CustomCallScratchAssignerTest, KeepsAliasingOfTupleResult) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p = f32[8] parameter(0)
      cc = (f32[8], s8[2]) custom-call(p),
        custom_call_target="xla.gpu.test_scratch_assigner",
        output_to_operand_aliasing={{0}: (0, {})}
      ROOT a = f32[8] get-tuple-element(cc), index=0
    })"));
  EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(true));

  const HloCustomCallInstruction* cc = FindCustomCall(*module, "cc");
  ASSERT_NE(cc, nullptr);
  EXPECT_THAT(cc->output_operand_aliasing(),
              ElementsAre(Pair(ShapeIndex{0, 0}, Pair(0, ShapeIndex{}))));
}

TEST_F(CustomCallScratchAssignerTest, PreservesControlDependencies) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      p = f32[8] parameter(0)
      before = f32[8] negate(p)
      cc = f32[8] custom-call(p),
        custom_call_target="xla.gpu.test_scratch_assigner",
        control-predecessors={before}
      after = f32[8] negate(cc), control-predecessors={cc}
      ROOT t = (f32[8], f32[8]) tuple(before, after)
    })"));
  EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(true));

  const HloCustomCallInstruction* cc = FindCustomCall(*module, "cc");
  ASSERT_NE(cc, nullptr);
  EXPECT_THAT(cc->control_predecessors(),
              ElementsAre(FindInstruction(module.get(), "before")));
  EXPECT_THAT(cc->control_successors(),
              ElementsAre(FindInstruction(module.get(), "after")));
}

TEST_F(CustomCallScratchAssignerTest, IsIdempotent) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      ROOT cc = f32[8] custom-call(),
        custom_call_target="xla.gpu.test_scratch_assigner"
    })"));
  EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(true));
  const Shape shape_after_first_run = FindCustomCall(*module, "cc")->shape();
  EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(false));
  EXPECT_EQ(FindCustomCall(*module, "cc")->shape(), shape_after_first_run);
}

TEST_F(CustomCallScratchAssignerTest, PassesContextToHandler) {
  std::string device_name;
  absl::StatusOr<int32_t> attr;
  CurrentScratchHandler() = [&](const HloCustomCallInstruction&,
                                const NativeCustomCallScratchContext& ctx)
      -> absl::StatusOr<std::vector<Shape>> {
    device_name = ctx.GetDeviceDescription().name();
    ABSL_ASSIGN_OR_RETURN(xla::ffi::Attributes attrs, ctx.GetFfiAttributes());
    attr = attrs.Get<int32_t>("num_elements");
    if (!attr.ok()) {
      return attr.status();
    }
    return std::vector<Shape>{Scratch(F32, {*attr})};
  };

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      ROOT cc = f32[8] custom-call(),
        custom_call_target="xla.gpu.test_scratch_assigner",
        backend_config="{num_elements = 42 : i32}"
    })"));
  EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(device_name, HasSubstr("H100"));
  EXPECT_THAT(attr, IsOkAndHolds(42));
  EXPECT_EQ(FindCustomCall(*module, "cc")->shape().ToString(),
            "(f32[8], f32[42])");
}

TEST_F(CustomCallScratchAssignerTest, LeavesCustomCallsWithoutScratchAlone) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      a = f32[8] custom-call(), custom_call_target="xla.gpu.test_no_scratch"
      b = f32[8] custom-call(), custom_call_target="xla.gpu.test_unregistered"
      ROOT t = (f32[8], f32[8]) tuple(a, b)
    })"));
  EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(false));
}

TEST_F(CustomCallScratchAssignerTest, EmptyScratchListIsANoOp) {
  SetScratchShapes({});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      ROOT cc = f32[8] custom-call(),
        custom_call_target="xla.gpu.test_scratch_assigner"
    })"));
  EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(false));
  EXPECT_FALSE(
      FindCustomCall(*module, "cc")
          ->get_frontend_attribute(kNativeCustomCallNumScratchBuffersAttr)
          .has_value());
}

TEST_F(CustomCallScratchAssignerTest, SkipsCustomCallsInsideFusions) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    fused {
      ROOT cc = f32[8] custom-call(),
        custom_call_target="xla.gpu.test_scratch_assigner"
    }
    ENTRY e {
      ROOT f = f32[8] fusion(), kind=kCustom, calls=fused
    })"));
  EXPECT_THAT(RunPass(module.get()), IsOkAndHolds(false));
}

TEST_F(CustomCallScratchAssignerTest, PropagatesHandlerErrors) {
  CurrentScratchHandler() = [](const HloCustomCallInstruction&,
                               const NativeCustomCallScratchContext&) {
    return absl::StatusOr<std::vector<Shape>>(
        absl::InternalError("handler failed"));
  };
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      ROOT cc = f32[8] custom-call(),
        custom_call_target="xla.gpu.test_scratch_assigner"
    })"));
  EXPECT_THAT(RunPass(module.get()), StatusIs(absl::StatusCode::kInternal,
                                              HasSubstr("handler failed")));
}

TEST_F(CustomCallScratchAssignerTest, RejectsTupleScratchShape) {
  SetScratchShapes(
      {ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {4})})});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      ROOT cc = f32[8] custom-call(),
        custom_call_target="xla.gpu.test_scratch_assigner"
    })"));
  EXPECT_THAT(RunPass(module.get()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be an array shape")));
}

TEST_F(CustomCallScratchAssignerTest, RejectsDynamicScratchShape) {
  SetScratchShapes({ShapeUtil::MakeShape(F32, {4}, {true})});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      ROOT cc = f32[8] custom-call(),
        custom_call_target="xla.gpu.test_scratch_assigner"
    })"));
  EXPECT_THAT(RunPass(module.get()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must have static dimensions")));
}

TEST_F(CustomCallScratchAssignerTest, RejectsUnsupportedMemorySpace) {
  SetScratchShapes(
      {Scratch(F32, {4}, static_cast<int64_t>(MemorySpaceColor::kTempBuffer))});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      ROOT cc = f32[8] custom-call(),
        custom_call_target="xla.gpu.test_scratch_assigner"
    })"));
  EXPECT_THAT(RunPass(module.get()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unsupported memory space 2")));
}

TEST_F(CustomCallScratchAssignerTest, PreservesArraySharding) {
  SetScratchShapes({ShapeUtil::MakeShape(F32, {4})});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      ROOT cc = f32[8] custom-call(),
        custom_call_target="xla.gpu.test_scratch_assigner",
        sharding={replicated}
    })"));
  ASSERT_OK_AND_ASSIGN(bool changed, RunPass(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_TRUE(root->has_sharding());
  EXPECT_EQ(root->sharding(), HloSharding::Replicate());

  const HloInstruction* new_cc = root->operand(0);
  EXPECT_EQ(new_cc->opcode(), HloOpcode::kCustomCall);
  EXPECT_TRUE(new_cc->has_sharding());
  EXPECT_EQ(new_cc->sharding().GetSubSharding(new_cc->shape(), {0}),
            HloSharding::Replicate());
  EXPECT_EQ(new_cc->sharding().GetSubSharding(new_cc->shape(), {1}),
            HloSharding::Replicate());
}

TEST_F(CustomCallScratchAssignerTest, PreservesTupleSharding) {
  SetScratchShapes({ShapeUtil::MakeShape(F32, {4})});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY e {
      ROOT cc = (f32[8], f32[4]) custom-call(),
        custom_call_target="xla.gpu.test_scratch_assigner",
        sharding={{maximal device=0}, {replicated}}
    })"));
  ASSERT_OK_AND_ASSIGN(bool changed, RunPass(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_TRUE(root->has_sharding());
  EXPECT_EQ(root->sharding().GetSubSharding(root->shape(), {0}),
            HloSharding::SingleDevice(0));
  EXPECT_EQ(root->sharding().GetSubSharding(root->shape(), {1}),
            HloSharding::Replicate());

  const HloInstruction* new_cc = root->operand(0);
  EXPECT_EQ(new_cc->opcode(), HloOpcode::kCustomCall);
  EXPECT_TRUE(new_cc->has_sharding());
  EXPECT_EQ(new_cc->sharding().GetSubSharding(new_cc->shape(), {0, 0}),
            HloSharding::SingleDevice(0));
  EXPECT_EQ(new_cc->sharding().GetSubSharding(new_cc->shape(), {0, 1}),
            HloSharding::Replicate());
  EXPECT_EQ(new_cc->sharding().GetSubSharding(new_cc->shape(), {1}),
            HloSharding::Replicate());
}

}  // namespace
}  // namespace xla::gpu
