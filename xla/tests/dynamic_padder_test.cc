/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/transforms/dynamic_padder.h"

#include <gmock/gmock.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/dynamic_dimension_simplifier.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/dynamic_dimension_inference.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/xla_test_backend_predicates.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace m = ::xla::match;
namespace op = xla::testing::opcode_matchers;
using ::absl_testing::IsOkAndHolds;

OpDynamismSupport OpHasDynamismSupport(HloInstruction* hlo) {
  if (hlo->opcode() != HloOpcode::kCustomCall) {
    return OpDynamismSupport::kNoSupport;
  }
  if (hlo->custom_call_target() == "OpWithDynamicLowering") {
    return OpDynamismSupport::kRequired;
  }
  return OpDynamismSupport::kNoSupport;
}

bool CustomCallDynamicDimensionInference(
    HloInstruction* hlo, DynamicDimensionInference* inferencer) {
  if (hlo->custom_call_target() == "OpWithDynamicLowering") {
    if (hlo->shape().IsTuple()) {
      // Use the operand's dynamic size as output dynamic size.
      HloInstruction* dynamic_size =
          inferencer->GetDynamicSize(hlo->mutable_operand(0), {1}, 0);
      inferencer->SetDynamicSize(hlo, {1}, 0, dynamic_size);
      return true;
    } else {
      // Use the operand's dynamic size as output dynamic size.
      HloInstruction* dynamic_size =
          inferencer->GetDynamicSize(hlo->mutable_operand(0), {}, 0);
      inferencer->SetDynamicSize(hlo, {}, 0, dynamic_size);
      return true;
    }
  }

  return false;
}

class DynamicPadderTest : public HloTestBase {
 protected:
  DynamicPadderTest() { module_ = CreateNewVerifiedModule(); }

  std::unique_ptr<HloModule> GetHloModule(const std::string& hlo_text) {
    std::unique_ptr<HloModule> module =
        ParseAndReturnVerifiedModule(hlo_text).value();
    return module;
  }

  absl::StatusOr<bool> RunPadder(
      bool slice_dynamic_output = false,
      OpSupportsDynamismHandler op_supports_dynamism_handler =
          OpHasDynamismSupport,
      DynamicDimensionInference::CustomCallInferenceHandler
          custom_call_handler = CustomCallDynamicDimensionInference) {
    DynamicPadderOptions options;
    options.slice_dynamic_output = slice_dynamic_output;
    options.op_supports_dynamism_handler =
        std::move(op_supports_dynamism_handler);
    options.custom_call_handler = std::move(custom_call_handler);
    DynamicPadder padder(std::move(options));
    ABSL_ASSIGN_OR_RETURN(bool changed, RunHloPass(&padder, module_.get()));
    if (!changed) return false;
    // Dynamic padder can add redundant tuple/get-tuple-element and copy
    // instructions.
    TupleSimplifier tuple_simplifier;
    ABSL_RETURN_IF_ERROR(RunHloPass(&tuple_simplifier, module_.get()).status());
    AlgebraicSimplifier alg_simplifier(AlgebraicSimplifierOptions{});
    ABSL_RETURN_IF_ERROR(RunHloPass(&alg_simplifier, module_.get()).status());
    return true;
  }

  void ExpectPadded(const HloInstruction* inst) {
    EXPECT_THAT(inst,
                op::Select(op::Lt(op::Iota(), op::Broadcast(op::Parameter())),
                           ::testing::_, op::Broadcast()));
  }

  HloComputation* GetScalarAddComputation() {
    auto embedded_builder = HloComputation::Builder("add");
    auto lhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeShape(F32, {}), "lhs"));
    auto rhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
        1, ShapeUtil::MakeShape(F32, {}), "rhs"));
    embedded_builder.AddInstruction(
        HloInstruction::CreateBinary(lhs->shape(), HloOpcode::kAdd, lhs, rhs));
    return module_->AddEmbeddedComputation(embedded_builder.Build());
  }

  std::unique_ptr<HloModule> module_;
  const Shape scalar_shape_ = ShapeUtil::MakeShape(S32, {});
};

TEST_F(DynamicPadderTest, SliceToDynamicSizeIsClamped) {
  // Regression test for https://github.com/openxla/xla/issues/44940: the
  // size operand of set-dimension-size is not otherwise validated, so the
  // materializing SliceToDynamic must clamp it to [0, bound].
  const std::string hlo_text = R"(
HloModule ClampSize

ENTRY main {
  data = s32[4] parameter(0)
  size = s32[] parameter(1)
  ROOT dyn = s32[<=4] set-dimension-size(data, size), dimensions={0}
}
)";
  module_ = GetHloModule(hlo_text);
  TF_ASSERT_OK(RunPadder(/*slice_dynamic_output=*/true).status());
  EXPECT_THAT(module_->entry_computation()->root_instruction(),
              op::CustomCall(
                  {"SliceToDynamic"}, op::Parameter(0),
                  op::Clamp(op::Constant(), op::Parameter(1), op::Constant())));
}

class MemoryAlignmentTest : public HloInterpreterReferenceMixin<HloTestBase> {};

// Test that dynamic padder will not cause memory misalignment in CUDA
// when the read or write address is not aligned with 32 bits.
// TODO(b/203599920): Disabled on CPU due to ASAN test failure.
TEST_F(MemoryAlignmentTest, TestDataTypeFP16) {
  if (test::DeviceIs(test::kCpu)) {
    GTEST_SKIP();
  }
  const std::string hlo_text = R"(
    HloModule TestDataTypeFP16

    update_add (p0: f16[], p1: f16[]) -> f16[] {
      p0 = f16[] parameter(0)
      p1 = f16[] parameter(1)
      ROOT out = f16[] add(p0, p1)
    }

    ENTRY main () -> f16[<=1,1] {
      c1 = s32[1]{0} constant({1})
      c2 = f16[1,1]{1,0} constant({ {0.099976} })
      shape = s32[] reshape(s32[1]{0} c1)
      dim_size = f16[<=1,1]{1,0} set-dimension-size(f16[1,1]{1,0} c2, s32[] shape),
          dimensions={0}
      ROOT out = f16[<=1,1]{1,0} scatter(f16[<=1,1]{1,0} dim_size, s32[1]{0} c1, f16[1,1]{1,0} c2),
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=update_add
    }
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(DynamicPadderTest, ReduceTest) {
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {1, 2, 2});
  auto reduce_shape = ShapeUtil::MakeShape(F32, {2});
  auto dynamic_shape =
      ShapeUtil::MakeShape(F32, {1, 2, 2}, {false, false, true});

  auto data_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "data_param"));
  auto* size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));

  data_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, data_param, size_param, 2));

  auto negate = builder.AddInstruction(HloInstruction::CreateUnary(
      dynamic_shape, HloOpcode::kNegate, data_param));

  auto init = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0)));

  auto reduce = builder.AddInstruction(HloInstruction::CreateReduce(
      reduce_shape, negate, init, {0, 2}, GetScalarAddComputation()));
  EXPECT_FALSE(module_->is_dynamic());
  module_->AddEntryComputation(builder.Build());

  ASSERT_OK(RunPadder().status());

  ExpectPadded(reduce->operand(0));
  EXPECT_TRUE(module_->is_dynamic());
}

TEST_F(DynamicPadderTest, OptimizationBarrierTest) {
  const std::string hlo_text = R"(
HloModule OptimizationBarrierTest

ENTRY %OptimizationBarrierTest (data_param: f32[1,2,2], size_param: s32[]) -> f32[1,2,<=2] {
  %data_param = f32[1,2,2]{2,1,0} parameter(0)
  %size_param = s32[] parameter(1)
  %set-dimension-size = f32[1,2,<=2]{2,1,0} set-dimension-size(%data_param, %size_param), dimensions={2}
  %opt-barrier = f32[1,2,<=2]{2,1,0} opt-barrier(%set-dimension-size)
  ROOT %negate = f32[1,2,<=2]{2,1,0} negate(%opt-barrier)
}
)";

  module_ = GetHloModule(hlo_text);
  ASSERT_OK(RunPadder().status());
}

TEST_F(DynamicPadderTest, DynamicLoweringTest) {
  const std::string hlo_text = R"(
HloModule DynamicLowering

ENTRY main {
  param = s32[5] parameter(0)
  const = s32[] constant(3)
  param_padded = s32[<=5] set-dimension-size(param, const),
                dimensions={0}
  custom-call.1 = s32[<=5] custom-call(param_padded),
    custom_call_target="OpWithDynamicLowering"
  custom-call.2 = s32[<=5] custom-call(custom-call.1),
    custom_call_target="OpWithDynamicLowering"
  // Negate doesn't support dynamic lowering.
  ROOT negate = s32[<=5] negate(custom-call.2)
}
)";

  module_ = GetHloModule(hlo_text);

  ASSERT_OK(RunPadder(/*slice_dynamic_output=*/true).status());
  // After rewrite, we should have :
  //
  //   param
  //     |
  //  SliceToDynamic
  //     |
  //  OpWithDynamicLowering (custom_call_1)
  //     |
  //  OpWithDynamicLowering (custom_call_2)
  //     |
  //  PadToStatic
  //     |     \
  //   Negate  Clamp(0, size, bound)
  //      \     /
  //   SliceToDynamic // Root require dynamic form tensor.
  auto custom_call_1 =
      module_->entry_computation()->GetInstructionWithName("custom-call.1");
  auto custom_call_2 =
      module_->entry_computation()->GetInstructionWithName("custom-call.2");
  // Test that the input to custom call
  HloInstruction* slice_to_dynamic = custom_call_1->mutable_operand(0);
  ASSERT_THAT(slice_to_dynamic->opcode(), HloOpcode::kCustomCall);
  ASSERT_THAT(slice_to_dynamic->custom_call_target(), "SliceToDynamic");
  ASSERT_EQ(custom_call_2->user_count(), 1);
  HloInstruction* pad_to_static = custom_call_2->users()[0];
  ASSERT_THAT(pad_to_static->opcode(), HloOpcode::kCustomCall);
  ASSERT_THAT(pad_to_static->custom_call_target(), "PadToStatic");
  slice_to_dynamic = module_->entry_computation()->root_instruction();
  ASSERT_THAT(slice_to_dynamic->opcode(), HloOpcode::kCustomCall);
  ASSERT_THAT(slice_to_dynamic->custom_call_target(), "SliceToDynamic");
}

TEST_F(DynamicPadderTest, DynamicLoweringTestTupleInput) {
  const std::string hlo_text = R"(
HloModule DynamicLowering

ENTRY main {
  param = s32[5] parameter(0)
  const = s32[] constant(3)
  param_padded = s32[<=5] set-dimension-size(param, const),
                dimensions={0}
  // Create a tuple with static and dynamic componenet.
  tuple_arg = (s32[], s32[<=5]) tuple(const, param_padded)
  custom-call.1 = (s32[], s32[<=5]) custom-call(tuple_arg),
    custom_call_target="OpWithDynamicLowering"
  custom-call.2 = (s32[], s32[<=5]) custom-call(custom-call.1),
    custom_call_target="OpWithDynamicLowering"
  data = s32[<=5]{0} get-tuple-element(custom-call.2), index=1
  // Negate doesn't support dynamic lowering.
  ROOT negate = s32[<=5] negate(data)
}
)";

  module_ = GetHloModule(hlo_text);

  ASSERT_OK(RunPadder(/*slice_dynamic_output=*/true).status());
  // After rewrite, we should have :
  //
  //   param
  //     |
  //  SliceToDynamic
  //     |
  //    Tuple
  //     |
  //  OpWithDynamicLowering (custom_call_1)
  //     |
  //  OpWithDynamicLowering (custom_call_2)
  //     |
  //   GTE
  //     |
  //  PadToStatic
  //     |     \
  //   Negate  Clamp(0, size, bound)
  //      \     /
  //   SliceToDynamic // Root require dynamic form tensor.

  auto* root = module_->entry_computation()->root_instruction();
  // The final result should use the dynamic size provided by PadToStatic,
  // clamped to the dimension bound.
  EXPECT_THAT(
      root, op::CustomCall(
                {"SliceToDynamic"}, op::Negate(),
                op::Clamp(op::Constant(),
                          op::GetTupleElement(op::CustomCall({"PadToStatic"})),
                          op::Constant())));
  HloInstruction* negate = root->mutable_operand(0);
  EXPECT_THAT(
      negate,
      op::Negate(op::GetTupleElement(op::CustomCall(
          {"PadToStatic"}, op::GetTupleElement(op::CustomCall(
                               {"OpWithDynamicLowering"}, ::testing::_))))));
  auto custom_call_1 =
      module_->entry_computation()->GetInstructionWithName("custom-call.1");
  EXPECT_THAT(custom_call_1,
              op::CustomCall({"OpWithDynamicLowering"},
                             op::Tuple(op::Constant(),
                                       op::CustomCall({"SliceToDynamic"}))));
}

TEST_F(DynamicPadderTest, DynamicOutputNestedTuple) {
  const std::string hlo_text = R"(
HloModule DynamicLowering

ENTRY main {
  param = s32[5] parameter(0)
  const = s32[] constant(3)
  const2 = s32[] constant(4)
  param_padded = s32[<=5] set-dimension-size(param, const),
                dimensions={0}
  // Create a tuple with static and dynamic componenet.
  tuple0 = (s32[], s32[<=5]) tuple(const, param_padded)
  ROOT tuple1 = (s32[], (s32[], s32[<=5])) tuple(const2, tuple0)
}
)";

  module_ = GetHloModule(hlo_text);

  ASSERT_OK(RunPadder(/*slice_dynamic_output=*/true).status());
  ASSERT_OK(TupleSimplifier().Run(module_.get()).status());
  XLA_LOG_LINES(INFO, module_->ToString());

  auto* root = module_->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::Constant(), op::Tuple()));
  HloInstruction* nested_tuple = root->mutable_operand(1);
  EXPECT_THAT(nested_tuple,
              op::Tuple(op::Constant(), op::CustomCall({"SliceToDynamic"})));
}

TEST_F(DynamicPadderTest, BitcastConvertWideningWithDynamicDimension) {
  // Regression test for https://github.com/openxla/xla/issues/45967:
  // a bit-width-changing bitcast-convert on an operand with a dynamic
  // dimension made the dynamic padder fail with an Unimplemented error.
  const std::string hlo_text = R"(
HloModule BitcastConvertWidening

ENTRY main {
  param = s64[16,1] parameter(0)
  size = s32[] parameter(1)
  param_dynamic = s64[<=16,1] set-dimension-size(param, size), dimensions={0}
  ROOT bitcast = f32[<=16,1,2] bitcast-convert(param_dynamic)
}
)";

  module_ = GetHloModule(hlo_text);

  TF_ASSERT_OK(RunPadder(/*slice_dynamic_output=*/true).status());
  XLA_LOG_LINES(INFO, module_->ToString());

  auto* root = module_->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::CustomCall({"SliceToDynamic"}));
}

TEST_F(DynamicPadderTest, BitcastConvertNarrowingWithDynamicDimension) {
  // Same as above, in the narrowing direction: the major dimensions of the
  // consumed shape may be dynamic (the contracted minor dimension is static).
  const std::string hlo_text = R"(
HloModule BitcastConvertNarrowing

ENTRY main {
  param = f32[16,2] parameter(0)
  size = s32[] parameter(1)
  param_dynamic = f32[<=16,2] set-dimension-size(param, size), dimensions={0}
  ROOT bitcast = s64[<=16] bitcast-convert(param_dynamic)
}
)";

  module_ = GetHloModule(hlo_text);

  TF_ASSERT_OK(RunPadder(/*slice_dynamic_output=*/true).status());
  XLA_LOG_LINES(INFO, module_->ToString());

  auto* root = module_->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::CustomCall({"SliceToDynamic"}));
}

TEST_F(DynamicPadderTest, ConvolutionTest) {
  auto builder = HloComputation::Builder(TestName());
  constexpr int xdim = 3;
  constexpr int ydim = 2;
  constexpr int zdim = 1;
  auto xy_shape = ShapeUtil::MakeShape(F32, {xdim, ydim});
  auto yz_shape = ShapeUtil::MakeShape(F32, {ydim, zdim});
  auto zx_shape = ShapeUtil::MakeShape(F32, {zdim, xdim});
  auto xy_shape_dynamic =
      ShapeUtil::MakeShape(F32, {xdim, ydim}, {false, true});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, xy_shape, "A"));
  auto* b_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, yz_shape, "B"));
  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, scalar_shape_, "size_param"));

  auto dnums = XlaBuilder::CreateDefaultConvDimensionNumbers(0);

  dnums.set_kernel_input_feature_dimension(0);
  dnums.set_kernel_output_feature_dimension(1);
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(1);
  dnums.set_output_feature_dimension(0);

  Window window;

  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      xy_shape_dynamic, a_param, size_param, 1));

  auto* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      zx_shape, {a_param, b_param}, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, DefaultPrecisionConfig(2)));

  module_->AddEntryComputation(builder.Build());

  ASSERT_OK(RunPadder().status());

  ExpectPadded(conv->operand(0));
}

TEST_F(DynamicPadderTest, ConvolutionNoPad) {
  auto builder = HloComputation::Builder(TestName());
  constexpr int xdim = 3;
  constexpr int ydim = 2;
  constexpr int zdim = 1;
  auto xy_shape = ShapeUtil::MakeShape(F32, {xdim, ydim});
  auto yz_shape = ShapeUtil::MakeShape(F32, {ydim, zdim});
  auto zx_shape = ShapeUtil::MakeShape(F32, {zdim, xdim}, {false, true});

  auto dynamic_shape = ShapeUtil::MakeShape(F32, {xdim, ydim}, {true, false});

  auto* a_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, xy_shape, "A"));
  auto* b_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, yz_shape, "B"));
  auto* size_param = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/2, scalar_shape_, "size_param"));

  a_param = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, a_param, size_param, 0));

  auto dnums = XlaBuilder::CreateDefaultConvDimensionNumbers(0);

  dnums.set_kernel_input_feature_dimension(0);
  dnums.set_kernel_output_feature_dimension(1);
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(1);
  dnums.set_output_feature_dimension(0);

  Window window;

  auto* conv = builder.AddInstruction(HloInstruction::CreateConvolve(
      zx_shape, {a_param, b_param}, /*feature_group_count=*/1,
      /*batch_group_count=*/1, window, dnums, DefaultPrecisionConfig(2)));

  module_->AddEntryComputation(builder.Build());

  ASSERT_OK(RunPadder().status());

  EXPECT_THAT(conv->operand(0), op::Parameter());
}

TEST_F(DynamicPadderTest, ReduceWindowNoPadForTrivialWindow) {
  auto builder = HloComputation::Builder(TestName());
  auto input_shape = ShapeUtil::MakeShape(F32, {4, 5});
  auto reduce_shape = ShapeUtil::MakeShape(F32, {3, 5}, {false, true});
  auto dynamic_shape = ShapeUtil::MakeShape(F32, {4, 5}, {false, true});

  auto input = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "input"));
  auto* size_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape_, "size_param"));

  input = builder.AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_shape, input, size_param, 1));

  auto init = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0)));
  ASSERT_OK_AND_ASSIGN(Window window, ParseWindow("size=2x1 pad=0_0x0_0"));
  auto output = builder.AddInstruction(HloInstruction::CreateReduceWindow(
      reduce_shape, input, init, window, GetScalarAddComputation()));

  module_->AddEntryComputation(builder.Build());

  ASSERT_OK(RunPadder().status());

  EXPECT_THAT(output->operand(0), op::Parameter());
}

TEST_F(DynamicPadderTest, VariadicReduceWindowNoPadForTrivialWindow) {
  const std::string hlo_text = R"(
HloModule VariadicReduceWindowNoPadForTrivialWindow

add_f32 (a: f32[], b: s32[], c: f32[], d: s32[]) -> (f32[], s32[]) {
  a = f32[] parameter(0)
  b = s32[] parameter(1)
  c = f32[] parameter(2)
  d = s32[] parameter(3)
  add.0 = f32[] add(a, c)
  add.1 = s32[] add(b, d)
  ROOT out = tuple(add.0, add.1)
}

ENTRY main {
  input.0 = f32[4, 5] parameter(0)
  input.1 = s32[4, 5] parameter(1)
  size_param.0 = s32[] parameter(2)
  size_param.1 = s32[] parameter(3)
  input_dynamic.0 = f32[4,<=5] set-dimension-size(input.0, size_param.0), dimensions={1}
  input_dynamic.1 = s32[4,<=5] set-dimension-size(input.1, size_param.0), dimensions={1}
  init.0 = f32[] constant(0.0)
  init.1 = s32[] constant(0)
  ROOT output = (f32[3, <=5], s32[3, <=5]) reduce-window(input_dynamic.0, input_dynamic.1, init.0, init.1), window={size=2x1 pad=0_0x0_0}, to_apply=add_f32
}
)";

  const int kNumParams = 2;
  module_ = ParseAndReturnVerifiedModule(hlo_text).value();

  ASSERT_OK(RunPadder().status());

  for (int i = 0; i < kNumParams; ++i) {
    EXPECT_THAT(module_->entry_computation()->root_instruction()->operand(i),
                op::Parameter());
  }
}

TEST_F(DynamicPadderTest, PadS8ToS32Dot) {
  const std::string hlo_text = R"(
HloModule test
ENTRY test {
  a = s8[<=16,32] parameter(0)
  b = s8[32,64] parameter(1)
  ROOT root = s32[<=16,64] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  module_ = GetHloModule(hlo_text);
  ASSERT_OK(RunPadder(/*slice_dynamic_output=*/true).status());

  EXPECT_THAT(module_->entry_computation()->root_instruction(),
              GmockMatch(m::CustomCall({"SliceToDynamic"},
                                       m::Dot(m::Op().WithShape(S8, {16, 32}),
                                              m::Op().WithShape(S8, {32, 64}))
                                           .WithShape(S32, {16, 64}),
                                       m::Op(), m::Op())));
}

TEST_F(DynamicPadderTest, PadToStaticForCustomCall) {
  const std::string hlo_text = R"(
HloModule test
ENTRY test {
  a = f32[64] parameter(0)
  ROOT c = f32[<=128] custom-call(a),
    custom_call_target="UnknownOp"
}
)";

  module_ = GetHloModule(hlo_text);
  ASSERT_OK(RunPadder(/*slice_dynamic_output=*/true).status());

  EXPECT_THAT(module_->entry_computation()->root_instruction(),
              GmockMatch(m::CustomCall({"UnknownOp"})));
}

TEST_F(DynamicPadderTest, WhileLoopDynamicShapeChangeToStatic) {
  const std::string hlo_text = R"(
HloModule WhileLoopDynamicShapeChangeToStatic

 %cond_wrapper.19447 {
  param = (s32[], s32[], f32[], f32[<=32,216]{1,0}) parameter(0)
  %get-tuple-element.184 = s32[] get-tuple-element(param), index=0
  %get-tuple-element.185 = s32[] get-tuple-element(param), index=1
  ROOT %compare.28 = pred[] compare(s32[] %get-tuple-element.184, s32[] %get-tuple-element.185), direction=LT
 }

%while_body_78894_grad_83711__.18882 {
  param = (s32[], s32[], f32[], f32[<=32,216]{1,0}) parameter(0)
  %get-tuple-element.184 = s32[] get-tuple-element(param), index=0
  %get-tuple-element.185 = s32[] get-tuple-element(param), index=1
  %add.1 = s32[] add(get-tuple-element.184, get-tuple-element.184)
  %gte.2 = f32[] get-tuple-element(param), index=2
  %broadcast.19389 = f32[32,216]{1,0} broadcast(f32[] %gte.2), dimensions={}
  %constant.32 = s32[] constant(32)
  %set-dimension-size = f32[<=32,216]{1,0} set-dimension-size(f32[32,216]{1,0} %broadcast.19389, s32[] %constant.32), dimensions={0}
  ROOT tuple = (s32[], s32[], f32[], f32[<=32,216]{1,0}) tuple(add.1, %get-tuple-element.185, %gte.2, %set-dimension-size)
}

ENTRY main {
  param = f32[] parameter(0)
  param.1 = f32[<=32,216]{1,0} parameter(1)
  const = s32[] constant(3)
  const2 = s32[] constant(4)
  %tuple.18877 = (s32[], s32[], f32[], f32[<=32,216]{1,0}) tuple(const, const2, param, param.1)
  %while.19451 = (s32[], s32[], f32[], f32[<=32,216]{1,0})
    while((s32[], s32[], f32[], f32[<=32,216]{1,0})
     %tuple.18877), condition=%cond_wrapper.19447, body=%while_body_78894_grad_83711__.18882
  ROOT result = f32[<=32,216]{1,0} get-tuple-element(while.19451), index=3
 }
)";

  module_ = GetHloModule(hlo_text);

  ASSERT_OK(RunPadder(/*slice_dynamic_output=*/true).status());
  XLA_LOG_LINES(INFO, module_->ToString());
  auto* root = module_->entry_computation()->root_instruction();
  EXPECT_EQ(root->shape(), ShapeUtil::MakeShape(F32, {32, 216}, {true, false}));
  // Find the while loop and ensure that the dynamic dimension size was added to
  // its operand and output.
  HloInstruction* while_inst = nullptr;
  for (HloInstruction* inst :
       module_->entry_computation()->MakeInstructionPostOrder()) {
    if (inst->opcode() == HloOpcode::kWhile) {
      ASSERT_EQ(while_inst, nullptr)
          << "while_inst: " << while_inst->name() << ", inst: " << inst->name();
      while_inst = inst;
    }
  }
  EXPECT_EQ(while_inst->shape(),
            ShapeUtil::MakeTupleShape({ShapeUtil::MakeScalarShape(S32),
                                       ShapeUtil::MakeScalarShape(S32),
                                       ShapeUtil::MakeScalarShape(F32),
                                       ShapeUtil::MakeShape(F32, {32, 216}),
                                       ShapeUtil::MakeScalarShape(S32)}));
}

TEST_F(DynamicPadderTest, WhileLoopCarriesRequiredDynamicShape) {
  // Test a while loop that carries dynamic shapes.
  // This module is similar to an on-device training loop with gradients delayed
  // by a step. Dynamic shapes are only touched by ops with dynamic lowerings,
  // so they should not be padded.
  const std::string hlo_text = R"(
HloModule WhileLoopCarriesRequiredDynamicShape

%cond {
  param = (f32[1024], f32[<=64], f32[32], f32[<=64], f32[32], s32[], s32[], token[]) parameter(0)
  current = s32[] get-tuple-element(param), index=5
  last = s32[] get-tuple-element(param), index=6
  ROOT result = pred[] compare(current, last), direction=LT
}

%body {
  param = (f32[1024], f32[<=64], f32[32], f32[<=64], f32[32], s32[], s32[], token[]) parameter(0)
  var = f32[1024] get-tuple-element(param), index=0
  input0 = f32[<=64] get-tuple-element(param), index=1
  grad0 = f32[32] get-tuple-element(param), index=2
  input1 = f32[<=64] get-tuple-element(param), index=3
  act1 = f32[32] get-tuple-element(param), index=4

  grad1 = f32[32] custom-call(act1), custom_call_target="ComputeGradients"

  var1 = f32[1024] custom-call(var, input0, grad0), custom_call_target="ApplyGradients", output_to_operand_aliasing={{}: (0, {})}

  token2 = token[] get-tuple-element(param), index=7
  infeed2 = (f32[<=64], token[]) infeed(token2)
  input2 = f32[<=64] get-tuple-element(infeed2), index=0
  act2 = f32[32] custom-call(var1, input2), custom_call_target="ComputeActivations"

  current = s32[] get-tuple-element(param), index=5
  constant1 = s32[] constant(1)
  add = s32[] add(current, constant1)

  last = s32[] get-tuple-element(param), index=6
  token3 = token[] get-tuple-element(infeed2), index=1
  ROOT result = (f32[1024], f32[<=64], f32[32], f32[<=64], f32[32], s32[], s32[], token[]) tuple(var1, input1, grad1, input2, act2, add, last, token3)
}

ENTRY main {
  last = s32[] parameter(0)
  var = f32[1024] parameter(1)

  token0 = token[] after-all()
  infeed0 = (f32[<=64], token[]) infeed(token0)
  input0 = f32[<=64] get-tuple-element(infeed0), index=0
  act0 = f32[32] custom-call(var, input0), custom_call_target="ComputeActivations"

  grad0 = f32[32] custom-call(act0), custom_call_target="ComputeGradients"
  token1 = token[] get-tuple-element(infeed0), index=1
  infeed1 = (f32[<=64], token[]) infeed(token1)
  input1 = f32[<=64] get-tuple-element(infeed1), index=0
  act1 = f32[32] custom-call(var, input1), custom_call_target="ComputeActivations"

  token2 = token[] get-tuple-element(infeed1), index=1

  zero = s32[] constant(0)
  tuple = (f32[1024], f32[<=64], f32[32]{0}, f32[<=64], f32[32]{0}, s32[], s32[], token[]) tuple(var, input0, grad0, input1, act1, zero, last, token2)
  while = (f32[1024], f32[<=64], f32[32]{0}, f32[<=64], f32[32]{0}, s32[], s32[], token[]) while(tuple), condition=%cond, body=%body

  ROOT result = f32[1024] get-tuple-element(while), index=0
}
)";

  module_ = GetHloModule(hlo_text);

  auto op_supports_dynamism = [](HloInstruction* hlo) {
    if (hlo->opcode() != HloOpcode::kCustomCall) {
      return OpDynamismSupport::kNoSupport;
    }
    if (hlo->custom_call_target() == "ComputeActivations" ||
        hlo->custom_call_target() == "ApplyGradients") {
      return OpDynamismSupport::kRequired;
    }
    return OpDynamismSupport::kNoSupport;
  };
  auto custom_call_handler = [](HloInstruction* hlo,
                                DynamicDimensionInference* inference) {
    return false;
  };
  ASSERT_OK(
      RunPadder(
          /*slice_dynamic_output=*/true,
          /*op_supports_dynamism_handler=*/std::move(op_supports_dynamism),
          /*custom_call_handler=*/std::move(custom_call_handler))
          .status());
  XLA_VLOG_LINES(1, module_->ToString());

  for (HloComputation* computation : module_->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCustomCall) {
        EXPECT_NE(instruction->custom_call_target(), "PadToStatic");
        EXPECT_NE(instruction->custom_call_target(), "SliceToDynamic");
        if (instruction->custom_call_target() == "ComputeActivations") {
          EXPECT_TRUE(instruction->operand(1)->shape().is_dynamic());
        } else if (instruction->custom_call_target() == "ApplyGradients") {
          EXPECT_TRUE(instruction->operand(1)->shape().is_dynamic());
        }
      } else if (instruction->opcode() == HloOpcode::kWhile) {
        const Shape& shape = instruction->shape();
        EXPECT_TRUE(shape.tuple_shapes(1).is_dynamic());
        EXPECT_TRUE(shape.tuple_shapes(3).is_dynamic());
      }
    }
  }
}

TEST_F(DynamicPadderTest, HandleReshapeCheckPastReshape) {
  // Two different sizes.
  auto hlo_text = R"(
HloModule ReshapeDynamicDimension
ENTRY main {
  p0 = f32[4,511,432]{2,1,0} parameter(0)
  p1 = s32[] parameter(1)
  p2 = f32[432,337]{1,0:T(8,128)} parameter(2)
  p0_dynamic = f32[<=4,511,432] set-dimension-size(p0, p1), dimensions={0}
  reshape.4179 = f32[<=2044,432]{1,0} reshape(p0_dynamic)
  dot.4180 = f32[<=2044,337]{1,0} dot(reshape.4179, p2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  transpose.4181 = f32[<=2044,337]{1,0} transpose(dot.4180), dimensions={0,1}
  ROOT reshape.4183 = f32[<=4,511,337]{2,1,0} reshape(transpose.4181)
})";
  module_ = GetHloModule(hlo_text);
  // Set up dynamic parameter binding.
  ASSERT_OK(RunPadder(/*slice_dynamic_output=*/true).status());
  VLOG(3) << module_->ToString();
  CHECK(module_->is_dynamic());
  CHECK(module_->entry_computation()
            ->root_instruction()
            ->shape()
            .is_dynamic_dimension(0));
}

TEST_F(DynamicPadderTest, ConcatToPad) {
  const std::string hlo_text = R"(
HloModule ConcatToPad

ENTRY main {
  param_0 = f32[4] parameter(0)
  param_1 = f32[4] parameter(1)
  size_0 = s32[] parameter(2)
  size_1 = s32[] parameter(3)
  padded_0 = f32[<=4] set-dimension-size(param_0, size_0), dimensions={0}
  padded_1 = f32[<=4] set-dimension-size(param_1, size_1), dimensions={0}
  ROOT concat = f32[<=8] concatenate(padded_0, padded_1), dimensions={0}
}
)";
  module_ = GetHloModule(hlo_text);
  ASSERT_OK(RunPadder().status());

  auto root = module_->entry_computation()->root_instruction();
  ASSERT_THAT(root->opcode(), HloOpcode::kDynamicUpdateSlice);

  auto pad = root->operand(0);
  ASSERT_THAT(pad->opcode(), HloOpcode::kPad);
  EXPECT_EQ(pad->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(pad->operand(0)->parameter_number(), 0);

  // Check padding config
  EXPECT_EQ(pad->padding_config().dimensions_size(), 1);
  EXPECT_EQ(pad->padding_config().dimensions(0).edge_padding_low(), 0);
  EXPECT_EQ(pad->padding_config().dimensions(0).edge_padding_high(), 4);
  EXPECT_EQ(pad->padding_config().dimensions(0).interior_padding(), 0);

  // Check DUS operands
  auto operand_1_static = root->operand(1);
  EXPECT_EQ(operand_1_static->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(operand_1_static->parameter_number(), 1);

  auto offset = root->operand(2);
  EXPECT_EQ(offset->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(offset->parameter_number(), 2);
}

TEST_F(DynamicPadderTest, ZeroSizedDynamicBinaryOpNotRewritten) {
  // Regression test for https://github.com/openxla/xla/issues/49925.
  //
  // Both operands of an elementwise binary op have a dynamic dimension whose
  // static bound is zero, so RewriteDynamicBinaryOp must skip the implicit
  // broadcast rewrite and leave the compare untouched.
  const std::string hlo_text = R"(
HloModule module

ENTRY entry {
  a = f32[0, 2] parameter(0)
  b = f32[0, 2] parameter(1)
  size_0 = s32[] parameter(2)
  size_1 = s32[] parameter(3)
  a_dynamic = f32[<=0, 2] set-dimension-size(a, size_0), dimensions={0}
  b_dynamic = f32[<=0, 2] set-dimension-size(b, size_1), dimensions={0}
  ROOT cmp = pred[<=0, 2] compare(a_dynamic, b_dynamic), direction=EQ
}
)";
  module_ = GetHloModule(hlo_text);

  DynamicPadderOptions options;
  options.slice_dynamic_output = false;
  DynamicPadder padder(options);
  ASSERT_OK_AND_ASSIGN(bool changed, padder.Run(module_.get()));
  // DynamicDimensionInference and ReplaceSetSize strip the set-dimension-size
  // ops and clear the dynamic dimensions, so the pass still reports changed.
  EXPECT_TRUE(changed);
  ASSERT_OK(verifier().Run(module_.get()).status());

  EXPECT_THAT(module_->entry_computation()->root_instruction(),
              op::Compare(op::Parameter(0), op::Parameter(1)));
  for (const HloInstruction* inst :
       module_->entry_computation()->instructions()) {
    EXPECT_NE(inst->opcode(), HloOpcode::kSlice);
    EXPECT_NE(inst->opcode(), HloOpcode::kSelect);
    EXPECT_NE(inst->opcode(), HloOpcode::kReshape);
  }
}

// Test that dynamic padder has the same result as if not padded.
class ExecutionTest : public HloTestBase {
 protected:
  std::unique_ptr<HloModule> GetHloModule(const std::string& hlo_text) {
    std::unique_ptr<HloModule> module =
        ParseAndReturnVerifiedModule(hlo_text).value();
    return module;
  }
  absl::StatusOr<Literal> PadAndExecute(std::unique_ptr<HloModule> module,
                                        absl::Span<Literal* const> arguments,
                                        bool slice_dynamic_output = true) {
    if (!slice_dynamic_output) {
      auto new_config = module->config();
      new_config.mutable_entry_computation_layout()
          ->mutable_result_layout()
          ->ClearDynamicShape();
      module->set_config(new_config);
    }
    DynamicPadderOptions options;
    options.slice_dynamic_output = slice_dynamic_output;
    DynamicPadder padder(options);
    CHECK_OK(padder.Run(module.get()).status());
    HloDCE dce;
    CHECK_OK(dce.Run(module.get()).status());
    return Execute(std::move(module), {arguments});
  }
};

TEST_F(ExecutionTest, ScatterUpdate) {
  // Test that scattering on indices=[2] is same as scattering on indices=[4]
  // and dynamic dimension = 2
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[INDICES_BOUND] parameter(1)
  updates = s32[INDICES_BOUND,3] parameter(2)
  dynamic_size = s32[] parameter(3)
  indices_dynamic = s32[<=INDICES_BOUND] set-dimension-size(indices, dynamic_size), dimensions={0}
  updates_dynamic = s32[<=INDICES_BOUND,3] set-dimension-size(updates, dynamic_size), dimensions={0}
  ROOT scatter = s32[3,3] scatter(operand, indices_dynamic, updates_dynamic),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1

}
)";
  const std::string hlo_text_not_padded =
      absl::StrReplaceAll(hlo_text, {{"INDICES_BOUND", "2"}});
  auto module_not_padded = GetHloModule(hlo_text_not_padded);

  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  Literal dynamic_size = LiteralUtil::CreateR0<int32_t>(2);

  ASSERT_OK_AND_ASSIGN(Literal not_padded, Execute(std::move(module_not_padded),
                                                   {&operand, &scatter_indices,
                                                    &updates, &dynamic_size}));

  // Pad input to 4.
  const std::string hlo_text_padded =
      absl::StrReplaceAll(hlo_text, {{"INDICES_BOUND", "4"}});
  auto module_padded = GetHloModule(hlo_text_padded);
  // Pad the rest of input with garbage data.
  Literal scatter_indices_padded = LiteralUtil::CreateR1<int32_t>({0, 2, 0, 4});
  Literal updates_padded = LiteralUtil::CreateR2<int32_t>(
      {{10, 20, 30}, {70, 80, 90}, {30, 22, 11}, {-1, 20, -1}});
  DynamicPadder padder;
  CHECK_OK(padder.Run(module_padded.get()).status());
  ASSERT_OK_AND_ASSIGN(Literal padded,
                       PadAndExecute(std::move(module_padded),
                                     {&operand, &scatter_indices_padded,
                                      &updates_padded, &dynamic_size}));

  EXPECT_EQ(padded, not_padded);
}

TEST_F(ExecutionTest, ScatterUpdateWindowDim) {
  const std::string hlo_text = R"(
HloModule ScatterUpdateWindowDim

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[1,2,3] parameter(0)
  indices = s32[1] parameter(1)
  updates = s32[2,3,1] parameter(2)
  dynamic_size = s32[] constant(1)
  operand_dynamic = s32[1, <=2, 3] set-dimension-size(operand, dynamic_size),
      dimensions={1}
  updates_dynamic = s32[<=2, 3, 1] set-dimension-size(updates, dynamic_size),
      dimensions={0}
  ROOT scatter = s32[1, <=2, 3] scatter(operand_dynamic, indices, updates_dynamic),
      to_apply=update_s32,
      update_window_dims={0, 1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1

}
)";
  auto hlo_module = GetHloModule(hlo_text);

  Literal operand = LiteralUtil::CreateR3<int32_t>({{{0, 0, 0}, {0, 0, 0}}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0});
  Literal updates =
      LiteralUtil::CreateR3<int32_t>({{{10}, {20}, {30}}, {{70}, {80}, {90}}});

  ASSERT_OK_AND_ASSIGN(
      Literal padded,
      PadAndExecute(std::move(hlo_module),
                    {&operand, &scatter_indices, &updates}, false));
  Literal expected =
      LiteralUtil::CreateR3<int32_t>({{{10, 20, 30}, {70, 80, 90}}});
  EXPECT_EQ(padded, expected);
}

TEST_F(ExecutionTest, ScatterUpdateF32) {
  // Test that scattering on indices=[2] is same as scattering on indices=[4]
  // and dynamic dimension = 2
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

update_f32 (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  ROOT rhs = f32[] parameter(1)
}

ENTRY main {
  operand = f32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = f32[2,3] parameter(2)
  dynamic_size = s32[] parameter(3)
  indices_dynamic = s32[<=2] set-dimension-size(indices, dynamic_size), dimensions={0}
  updates_dynamic = f32[<=2,3] set-dimension-size(updates, dynamic_size), dimensions={0}
  ROOT scatter = f32[3,3] scatter(operand, indices_dynamic, updates_dynamic),
      to_apply=update_f32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1

}
)";

  auto module_not_padded = GetHloModule(hlo_text);

  Literal operand = LiteralUtil::CreateR2<float>(
      {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<float>({{10.0, 20.0, 30.0}, {70.0, 80.0, 90.0}});
  // Dynamic Size is 1, pad to 2
  Literal dynamic_size = LiteralUtil::CreateR0<int32_t>(1);

  auto module_padded = GetHloModule(hlo_text);
  DynamicPadder padder;
  CHECK_OK(padder.Run(module_padded.get()).status());
  ASSERT_OK_AND_ASSIGN(
      Literal not_padded,
      PadAndExecute(std::move(module_padded),
                    {&operand, &scatter_indices, &updates, &dynamic_size}));
  // Although we have two indices, only the first element is updated because of
  // padding.
  EXPECT_EQ(LiteralUtil::CreateR2<float>(
                {{10.0, 20.0, 30.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}),
            not_padded);
}

TEST_F(ExecutionTest, WholeDimensionGather) {
  // Second dimension (size 2) is dynamic, assuming real size is 1 and padded to
  // 2:
  //
  // [[1, 2]
  //  [3, 4]
  //  [5, 6]]
  //
  // Gathering the second dimension out creates:
  //
  // [3, 4]
  //
  // Reducing this gives us 3 (4 is padded value so ignored)
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[3, 2, 1] parameter(0)
  size = s32[] constant(1)
  param_padded = s32[3, <=2, 1] set-dimension-size(param, size), dimensions={1}
  index = s32[] constant(1)
  gather = s32[<=2,1]{1,0} gather(param_padded, index),
              offset_dims={0,1},
              collapsed_slice_dims={0},
              start_index_map={0},
              index_vector_dim=0,
              slice_sizes={1,2,1}
  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(gather, init),
      dimensions={0, 1},
      to_apply=update_s32
}
)";
  // Slicing out entire dimension propagates the dimension
  Literal operand =
      LiteralUtil::CreateR3<int32_t>({{{1}, {2}}, {{3}, {4}}, {{5}, {6}}});
  auto module = GetHloModule(hlo_text);
  DynamicPadder padder;
  CHECK_OK(padder.Run(module.get()).status());
  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}));

  // Only first element will be reduced.
  Literal expected = LiteralUtil::CreateR0<int32_t>(3);

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, TwoDimensionReduce) {
  // Test that reducing on operand=[2,2] is same as reducing on operand=[4,4]
  // and dynamic dimension = 2
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[INDICES_BOUND, INDICES_BOUND] parameter(0)
  dynamic_size = s32[] parameter(1)
  param_0 = s32[<=INDICES_BOUND,INDICES_BOUND] set-dimension-size(param, dynamic_size), dimensions={0}
  param_1 = s32[<=INDICES_BOUND,INDICES_BOUND] set-dimension-size(param_0, dynamic_size), dimensions={1}
  const = s32[] constant(0)
  ROOT reduce = s32[] reduce(param_1, const),
      dimensions={0, 1},
      to_apply=update_s32
}
)";
  const std::string hlo_text_not_padded =
      absl::StrReplaceAll(hlo_text, {{"INDICES_BOUND", "2"}});
  auto module_not_padded = GetHloModule(hlo_text_not_padded);

  Literal operand = LiteralUtil::CreateR2<int32_t>({{1, 2}, {4, 5}});
  Literal dynamic_size = LiteralUtil::CreateR0<int32_t>(2);

  ASSERT_OK_AND_ASSIGN(Literal not_padded, Execute(std::move(module_not_padded),
                                                   {&operand, &dynamic_size}));

  // Pad input to 4.
  const std::string hlo_text_padded =
      absl::StrReplaceAll(hlo_text, {{"INDICES_BOUND", "4"}});
  auto module_padded = GetHloModule(hlo_text_padded);
  // Pad the rest of input with garbage data.
  Literal operand_padded = LiteralUtil::CreateR2<int32_t>(
      {{1, 2, 3, 4}, {4, 5, 6, 7}, {1, 2, 3, 4}, {4, 5, 6, 7}});
  DynamicPadder padder;
  CHECK_OK(padder.Run(module_padded.get()).status());
  ASSERT_OK_AND_ASSIGN(Literal padded,
                       PadAndExecute(std::move(module_padded),
                                     {&operand_padded, &dynamic_size}));

  EXPECT_EQ(padded, not_padded);
}

TEST_F(ExecutionTest, DynamicDimensionClamp) {
  const std::string hlo_text = R"(
HloModule TensorFlowTenaryV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[5] parameter(0)
  const = s32[] constant(3)
  param_padded = s32[<=5] set-dimension-size(param, const), dimensions={0}
  clamp = s32[<=5] clamp(param_padded, param_padded, param_padded)
  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(clamp, init),
      dimensions={0},
      to_apply=update_s32
}
)";

  // Input has upper bound of 5, dynamic dimension is 3.
  Literal operand = LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4, 5});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}));

  // only first 3 elements will be reduced.
  Literal expected = LiteralUtil::CreateR0<int32_t>(6);

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicCumulativeProductScan) {
  // Regression test for https://github.com/openxla/xla/issues/44944: a
  // cumprod scan over a dynamic dimension used to crash the CPU executable.
  const std::string hlo_text = R"(
HloModule DynamicScan

mul_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT mul = s32[] multiply(lhs, rhs)
}

ENTRY main {
  param = s32[8] parameter(0)
  size = s32[] constant(3)
  param_dynamic = s32[<=8] set-dimension-size(param, size), dimensions={0}
  init = s32[] constant(1)
  ROOT cumprod = s32[<=8] reduce-window(param_dynamic, init), window={size=8 pad=7_0}, to_apply=mul_s32
}
)";

  // Input has upper bound of 8, dynamic size is 3; the rest is garbage.
  Literal operand =
      LiteralUtil::CreateR1<int32_t>({2, 3, 4, -1, -1, -1, -1, -1});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}, false));
  result.SetDynamicSize(0, 3);

  // Cumulative products of the valid prefix {2, 3, 4}.
  Literal expected = LiteralUtil::CreateR1<int32_t>({2, 6, 24});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, OutOfBoundsDynamicDimensionSizeClamped) {
  // Regression test for https://github.com/openxla/xla/issues/44940: a
  // runtime size beyond the dimension bound used to become an unchecked
  // copy bound in SliceToDynamic, corrupting memory.
  // The interpreter cannot execute the SliceToDynamic custom call.
  if (test::DeviceIs(test::kInterpreter)) {
    GTEST_SKIP();
  }
  const std::string hlo_text = R"(
HloModule OversizeDynamicSize

ENTRY main {
  data = s32[4] parameter(0)
  size = s32[] parameter(1)
  ROOT dyn = s32[<=4] set-dimension-size(data, size), dimensions={0}
}
)";
  Literal data = LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4});

  // An oversize runtime size is clamped to the bound.
  Literal oversize = LiteralUtil::CreateR0<int32_t>(100000);
  auto module = GetHloModule(hlo_text);
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          PadAndExecute(std::move(module), {&data, &oversize}));
  EXPECT_EQ(result.ToStatic(), LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4}));

  // A negative runtime size is clamped to zero.
  Literal negative = LiteralUtil::CreateR0<int32_t>(-5);
  module = GetHloModule(hlo_text);
  TF_ASSERT_OK_AND_ASSIGN(Literal empty_result,
                          PadAndExecute(std::move(module), {&data, &negative}));
  EXPECT_EQ(empty_result.ToStatic(), LiteralUtil::CreateR1<int32_t>({}));
}

TEST_F(ExecutionTest, DynamicConcat) {
  // Concatting a list of {dynamic_operand, static_operand, dynamic_operand}.
  const std::string hlo_text = R"(
HloModule DynamicConcat

ENTRY main {
  param_0 = s32[3] parameter(0)
  param_1 = s32[3] parameter(1)
  param_2 = s32[3] parameter(2)
  size = s32[] constant(2)
  param_padded_0 = s32[<=3] set-dimension-size(param_0, size), dimensions={0}
  param_padded_2 = s32[<=3] set-dimension-size(param_2, size), dimensions={0}
  ROOT %concatenate = s32[<=9]
    concatenate(s32[<=3] param_padded_0, s32[<=3] param_1, s32[<=3] param_padded_2),
    dimensions={0}
}
)";

  // Input has upper bound of 3, dynamic dimension is 2. Using -1 as padding.
  Literal operand_0 =
      LiteralUtil::CreateR1<int32_t>({1, 2, -1});  // Dynamic operand.
  Literal operand_1 =
      LiteralUtil::CreateR1<int32_t>({3, 4, 5});  // Static operand.
  Literal operand_2 =
      LiteralUtil::CreateR1<int32_t>({6, 7, -1});  // Dynamic operand.
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(
      Literal result,
      PadAndExecute(std::move(module), {&operand_0, &operand_1, &operand_2},
                    false));
  result.SetDynamicSize(0, 7);
  Literal expected = LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4, 5, 6, 7});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, BitcastConvertWideningWithDynamicDimension) {
  const std::string hlo_text = R"(
HloModule BitcastConvertWidening

ENTRY main {
  param_0 = s64[3] parameter(0)
  size = s32[] constant(2)
  param_padded = s64[<=3] set-dimension-size(param_0, size), dimensions={0}
  ROOT bitcast = s32[<=3,2] bitcast-convert(param_padded)
}
)";

  Literal operand_0 = LiteralUtil::CreateR1<int64_t>({1, 2, -1});
  auto module = GetHloModule(hlo_text);

  TF_ASSERT_OK_AND_ASSIGN(
      Literal result, PadAndExecute(std::move(module), {&operand_0}, false));
  result.SetDynamicSize(0, 2);
  Literal expected = LiteralUtil::CreateR2<int32_t>({{1, 0}, {2, 0}});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, BitcastConvertNarrowingWithDynamicDimension) {
  const std::string hlo_text = R"(
HloModule BitcastConvertNarrowing

ENTRY main {
  param_0 = s32[3,2] parameter(0)
  size = s32[] constant(2)
  param_padded = s32[<=3,2] set-dimension-size(param_0, size), dimensions={0}
  ROOT bitcast = s64[<=3] bitcast-convert(param_padded)
}
)";

  Literal operand_0 =
      LiteralUtil::CreateR2<int32_t>({{1, 0}, {2, 0}, {-1, -1}});
  auto module = GetHloModule(hlo_text);

  TF_ASSERT_OK_AND_ASSIGN(
      Literal result, PadAndExecute(std::move(module), {&operand_0}, false));
  result.SetDynamicSize(0, 2);
  Literal expected = LiteralUtil::CreateR1<int64_t>({1, 2});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicReverseSingleDim) {
  const std::string hlo_text = R"(
HloModule DynamicConcat

ENTRY main {
  param_0 = s32[3] parameter(0)
  size = s32[] constant(2)
  param_padded_0 = s32[<=3] set-dimension-size(param_0, size), dimensions={0}
  ROOT %reverse = s32[<=3]
    reverse(s32[<=3] param_padded_0),
    dimensions={0}
}
)";

  // Input has upper bound of 3, dynamic dimension is 2. Using -1 as padding.
  Literal operand_0 =
      LiteralUtil::CreateR1<int32_t>({1, 2, -1});  // Dynamic operand.
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand_0}, false));
  result.SetDynamicSize(0, 2);
  Literal expected = LiteralUtil::CreateR1<int32_t>({2, 1});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicReverseMultiDims) {
  const std::string hlo_text = R"(
HloModule DynamicConcat

ENTRY main {
  param_0 = s32[3, 3] parameter(0)
  size = s32[] constant(2)
  param_padded_0 = s32[<=3, 3] set-dimension-size(param_0, size), dimensions={0}
  param_padded_1 = s32[<=3, <=3] set-dimension-size(param_padded_0, size),
    dimensions={1}
  ROOT %reverse = s32[<=3, <=3]
    reverse(s32[<=3, <=3] param_padded_1),
    dimensions={0, 1}
}
)";

  // Input has upper bound of 3, dynamic dimension is 2. Using -1 as padding.
  Literal operand_0 = LiteralUtil::CreateR2<int32_t>(
      {{1, 2, -1}, {3, 4, -1}, {-1, -1, -1}});  // Dynamic operand.
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand_0}, false));
  result.SetDynamicSize(0, 2);
  result.SetDynamicSize(1, 2);
  Literal expected = LiteralUtil::CreateR2<int32_t>({{4, 3}, {2, 1}});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicDimensionReduce) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[5] parameter(0)
  const = s32[] constant(3)
  param_padded = s32[<=5] set-dimension-size(param, const), dimensions={0}
  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(param_padded, init),
      dimensions={0},
      to_apply=update_s32
}
)";

  // Input has upper bound of 5, dynamic dimension is 3.
  Literal operand = LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4, 5});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}));

  // only first 3 elements will be reduced.
  Literal expected = LiteralUtil::CreateR0<int32_t>(6);

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, InputMinorDimensionReshape) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[1, 2, 5, 1] parameter(0)
  const = s32[] constant(3)
  param_padded = s32[1, 2, <=5, 1] set-dimension-size(param, const), dimensions={2}
  reshaped = s32[<=10] reshape(param_padded)
  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(reshaped, init),
      dimensions={0},
      to_apply=update_s32
}
)";

  // The third dimension has upper bound of 5, dynamic dimension is 3.
  Literal operand = LiteralUtil::CreateR4<int32_t>(
      {{{{1}, {2}, {3}, {4}, {5}}, {{2}, {4}, {6}, {7}, {8}}}});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}));

  // Only the first 6 elements will be reduced.
  Literal expected = LiteralUtil::CreateR0<int32_t>(18);

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, SliceSingleElement) {
  // Slicing out a single element is supported.
  const std::string hlo_text = R"(
HloModule Slicing

ENTRY main {
  param = s32[5] parameter(0)
  const = s32[] constant(3)
  param_padded = s32[<=5] set-dimension-size(param, const), dimensions={0}
  ROOT slice = s32[1]{0} slice(param_padded), slice={[0:1]}
}
)";

  // The dynamic dimension has upper bound of 5, dynamic dimension is 3.
  Literal operand = LiteralUtil::CreateR1<int32_t>({0, 1, 2, 3, 4});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}));

  Literal expected = LiteralUtil::CreateR1<int32_t>({0});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, OutputMinorDimensionReshape) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[12] parameter(0)
  const = s32[] constant(8)
  param_padded = s32[<=12] set-dimension-size(param, const), dimensions={0}
  // Second dimension is dynamic.
  reshaped = s32[2, <=3, 2] reshape(param_padded), inferred_dimension=1
  init = s32[] constant(0)
  ROOT reduce = s32[2, 2] reduce(reshaped, init),
      dimensions={1},
      to_apply=update_s32
}
)";

  // The third dimension has upper bound of 5, dynamic dimension is 3.
  Literal operand =
      LiteralUtil::CreateR1<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}));

  // After padding and reshape we have
  //
  // [[[0, 1],
  //   [2, 3]
  //   [P, P]]
  //  [[4, 5],
  //   [6, 7],
  //   [P, P]]]
  // Reducing on the second dimension gives us
  //  [0+2, 1+3]
  //  [4+6, 5+7]
  //
  Literal expected = LiteralUtil::CreateR2<int32_t>({{2, 4}, {10, 12}});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, OutputMinorDimensionReshapeWithUnchangedDimMajor) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[2, 6] parameter(0)
  const = s32[] constant(4)
  param_padded = s32[2, <=6] set-dimension-size(param, const), dimensions={1}
  // Third dimension is dynamic.
  reshaped = s32[2, 2, <=3] reshape(param_padded), inferred_dimension=2
  init = s32[] constant(0)
  ROOT reduce = s32[2, 2] reduce(reshaped, init),
      dimensions={2},
      to_apply=update_s32
}
)";

  // The third dimension has upper bound of 5, dynamic dimension is 3.
  Literal operand = LiteralUtil::CreateR2<int32_t>(
      {{0, 1, 2, 3, 4, 5}, {6, 7, 8, 9, 10, 11}});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}));

  // After padding and reshape we have
  //
  // [[[0, 1, P],
  //   [2, 3, P]],
  //  [[6, 7, P],
  //   [8, 9, P]]]
  // Reducing on the third dimension gives us
  //  [0+1, 2+3]
  //  [6+7, 8+9]
  //
  Literal expected = LiteralUtil::CreateR2<int32_t>({{1, 5}, {13, 17}});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, OutputMinorDimensionReshapeWithUnchangedDimMinor) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[6, 2] parameter(0)
  const = s32[] constant(4)
  param_padded = s32[<=6, 2] set-dimension-size(param, const), dimensions={0}
  // Second dimension is dynamic.
  reshaped = s32[2, <=3, 2] reshape(param_padded), inferred_dimension=1
  init = s32[] constant(0)
  ROOT reduce = s32[2, 2] reduce(reshaped, init),
      dimensions={1},
      to_apply=update_s32
}
)";

  // The third dimension has upper bound of 5, dynamic dimension is 3.
  Literal operand = LiteralUtil::CreateR2<int32_t>(
      {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}));

  // After padding and reshape we have
  //
  // [[[0, 1],
  //   [2, 3]
  //   [P, P]],
  //  [[4, 5],
  //   [6, 7],
  //   [P, P]]]
  // Reducing on the second dimension gives us
  //  [0+2, 1+3]
  //  [4+6, 5+7]
  //
  Literal expected = LiteralUtil::CreateR2<int32_t>({{2, 4}, {10, 12}});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicInputFeature) {
  const std::string hlo_text = R"(
HloModule DynamicInputFeature

ENTRY main {
  param = f32[1, 1, 5] parameter(0)
  const = s32[] constant(5)
  one = f32[] constant(1)
  kernel = f32[1,5,1]{2,1,0} broadcast(f32[] one), dimensions={}
  param_dynamic = f32[1,1,<=5] set-dimension-size(param, const), dimensions={2}
  ROOT conv = f32[1, 1, 1]{2,1,0} custom-call(f32[1, 1, <=5] param_dynamic, f32[1,<=5,1]{2,1,0} kernel),
                             window={size=1 pad=0_0},
                             dim_labels=b0f_0io->b0f,
                             padding_type=PADDING_VALID,
                             custom_call_target="DynamicConvolutionForward"
}
)";

  Literal operand = LiteralUtil::CreateR3<float>({{{1, 2, 3, 4, 5}}});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}));

  Literal expected = LiteralUtil::CreateR3<float>({{{15}}});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicDimensionReshapeUnchanged) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[1, 2, 5, 1] parameter(0)
  const = s32[] constant(3)
  param_padded = s32[1, 2, <=5, 1] set-dimension-size(param, const), dimensions={2}
  reshaped = s32[2, <=5] reshape(param_padded)
  init = s32[] constant(0)
  ROOT reduce = s32[2] reduce(reshaped, init),
      dimensions={1},
      to_apply=update_s32
}
)";

  // Test dynamic padder in unchanged dimension reshape.
  Literal operand = LiteralUtil::CreateR4<int32_t>(
      {{{{1}, {2}, {3}, {4}, {5}}, {{2}, {4}, {6}, {7}, {8}}}});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}));

  Literal expected = LiteralUtil::CreateR1<int32_t>({6, 12});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DegeneratedDimension) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[1, 2, 5, 1] parameter(0)
  size = s32[] constant(0)
// First dimension is dynamic.
  param_padded = s32[<=1, 2, 5, 1] set-dimension-size(param, size),
    dimensions={0}
  reshaped = s32[<=10] reshape(param_padded)
  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(reshaped, init),
      dimensions={0},
      to_apply=update_s32
}
)";

  // First dimension (1) is dynamic. Since dynamic size is 0, result is also 0.
  Literal operand = LiteralUtil::CreateR4<int32_t>(
      {{{{1}, {2}, {3}, {4}, {5}}, {{2}, {4}, {6}, {7}, {8}}}});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}));

  Literal expected = LiteralUtil::CreateR0<int32_t>(0);

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, ReshapeSplitCombineSameTime) {
  // [<=4, 2, <=2]
  //       |
  //    Reshape
  //       |
  // [2, <=2, <=4]
  //
  // Split one input dynamic dim to multiple output dims while combining two
  // dimensions together.
  //
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[4, 2, 2] parameter(0)
  two = s32[] constant(2)
  one = s32[] constant(1)
  param_padded_partial = s32[<=4, 2, 2] set-dimension-size(param, two),
    dimensions={0}

  param_padded_dynamic = s32[<=4, 2, <=2] set-dimension-size(param_padded_partial,
                                                             one),
    dimensions={2}
  reshaped = s32[2, <=2, <=4] reshape(param_padded_dynamic),
    inferred_dimension=1
  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(reshaped, init),
      dimensions={0, 1, 2},
      to_apply=update_s32
}
)";

  // First and last dims are dynamic. Padded data are expressed as -1.
  Literal operand = LiteralUtil::CreateR3<int32_t>({{{0, -1}, {1, -1}},
                                                    {{2, -1}, {3, -1}},
                                                    {{-1, -1}, {-1, -1}},
                                                    {{-1, -1}, {-1, -1}}});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}));

  // Reshaping (with correct reshape rewriting) produces:
  // [[[0, 1, -1, -1], [-1, -1, -1, -1]], [[2, 3, -1, -1], [-1, -1, -1, -1]]]
  //
  //  Dynamic padder auto pads -1 with 0.
  //
  // Reducing it produces 0 + 1 + 2 + 3 = 6

  Literal expected = LiteralUtil::CreateR0<int32_t>(6);

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicDimensionReduceInCall) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

call_with_dynamic_dimension {
  param = s32[<=5] parameter(0)
  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(param, init),
      dimensions={0},
      to_apply=update_s32
}

ENTRY main {
  param0 = s32[5] parameter(0)
  param1 = s32[5] parameter(1)
  const0 = s32[] constant(3)
  const1 = s32[] constant(4)
  param0_padded = s32[<=5] set-dimension-size(param0, const0), dimensions={0}
  param1_padded = s32[<=5] set-dimension-size(param1, const1), dimensions={0}
  reduced0 = s32[] call(param0_padded), to_apply=call_with_dynamic_dimension
  reduced1 = s32[] call(param1_padded), to_apply=call_with_dynamic_dimension
  ROOT add = s32[] add(reduced0, reduced1)
}
)";

  // Input has upper bound of 5, dynamic dimension is 3.
  Literal operand0 = LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4, 5});
  Literal operand1 = LiteralUtil::CreateR1<int32_t>({6, 7, 8, 9, 10});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(
      Literal result, PadAndExecute(std::move(module), {&operand0, &operand1}));

  Literal expected = LiteralUtil::CreateR0<int32_t>(36);

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, ReshapeComplicated) {
  // [2, <=4, 4]
  //       |
  //    Reshape
  //       |
  // [<=16, 2]
  //
  // Reshape that is not a composition of splitting one input dim to multiple
  // output dims or combining multiple input dimensions to one output dimension.
  //
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[2, 4, 4] parameter(0)
  two = s32[] constant(2)
  param_padded_dynamic = s32[2, <=4, 4] set-dimension-size(param, two),
    dimensions={1}
  reshaped = s32[<=16, 2] reshape(param_padded_dynamic), inferred_dimension=0
  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(reshaped, init),
      dimensions={0, 1},
      to_apply=update_s32
}
)";

  // First and last dims are dynamic. Padded data are expressed as -1.
  Literal operand = LiteralUtil::CreateR3<int32_t>(
      {{{1, 2, 3, 4}, {5, 6, 7, 8}, {-1, -1, -1, -1}, {-1, -1, -1, -1}},
       {{9, 10, 11, 12},
        {13, 14, 15, 16},
        {-1, -1, -1, -1},
        {-1, -1, -1, -1}}});
  auto module = GetHloModule(hlo_text);
  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}));

  // Reshaping (with correct reshape rewriting) produces:
  // [[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]],
  //  [[-1, -1], [-1, -1], ...]]
  //
  //  Dynamic padder auto pads -1 with 0.
  //
  // Reducing it produces 1 + 2 + 3 + ... + 16 = 136

  Literal expected = LiteralUtil::CreateR0<int32_t>(136);
  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, WhileLoopStack) {
  // Push into a dynamic sized stack with iteration number:
  // init:
  // [[P, P],
  //  [P, P],
  //  [P, P],
  //  [P, P]]
  // First iteration i = 0:
  // [[0, 0],
  //  [P, P],
  //  [P, P],
  //  [P, P]]
  // Second iteration i = 1:
  // [[0, 0],
  //  [1, 1],
  //  [P, P],
  //  [P, P]]
  // Third iteration i = 2:
  // [[0, 0],
  //  [1, 1],
  //  [2, 2],
  //  [P, P]]

  const std::string hlo_text = R"(
HloModule module

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

body {
  stack = (s32[<=4,2]) parameter(0)
  stack_buffer = s32[<=4, 2] get-tuple-element(stack), index=0
  stack_size = s32[] get-dimension-size(stack_buffer), dimensions={0}
  zero = s32[] constant(0)
  one = s32[] constant(1)
  // content of the stack is the stack index broadcasted.
  new_data = s32[1, 2] broadcast(s32[] stack_size), dimensions={}
  new_stack_size = s32[] add(stack_size, one)
  new_stack_buffer = s32[<=4, 2] set-dimension-size(stack_buffer, new_stack_size), dimensions={0}
  new_stack = s32[<=4, 2] dynamic-update-slice(new_stack_buffer, new_data, stack_size, zero)
  ROOT new_stack_tuple = (s32[<=4,2]) tuple(new_stack)
}

condition {
  stack = (s32[<=4,2]) parameter(0)
  stack_buffer = s32[<=4, 2] get-tuple-element(stack), index=0
  stack_size = s32[] get-dimension-size(stack_buffer), dimensions={0}
  three = s32[] constant(3)
  ROOT less-than = pred[] compare(s32[] stack_size, s32[] three), direction=LT
}

ENTRY entry {
  zero = s32[] constant(0)
  pad = s32[] constant(-1)
  stack_buffer_input = s32[4, 2] broadcast(s32[] pad), dimensions={}
  stack_buffer_input_dynamic = s32[<=4, 2] set-dimension-size(stack_buffer_input, zero), dimensions={0}
  input_tuple = (s32[<=4 ,2]) tuple(stack_buffer_input_dynamic)
  while = (s32[<=4, 2]) while(input_tuple), body=body, condition=condition
  stack_buffer = s32[<=4, 2] get-tuple-element(while), index=0
  ROOT reduce = s32[2] reduce(stack_buffer, zero),
    dimensions={0},
    to_apply=update_s32
}
)";

  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result, PadAndExecute(std::move(module), {}));

  // Stack has three valid items in it:
  // [[0, 0],
  //  [1, 1],
  //  [2, 2],
  //  [P, P]]
  //
  // Reducing along major dimension gives us [3, 3]
  Literal expected = LiteralUtil::CreateR1<int32_t>({{3, 3}});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicAddWithImplicitBroadcast) {
  const std::string hlo_text = R"(
HloModule module

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY entry {
  zero = s32[] constant(0)
  one = s32[] constant(1)
  two = s32[] constant(2)
  three = s32[] constant(3)
  input1 = s32[4, 2] iota(), iota_dimension=0
  ones = s32[4, 2] broadcast(one), dimensions={}
  input1_added = s32[4, 2] add(input1, ones)
  input1_dynamic = s32[<=4, 2] set-dimension-size(input1_added, one), dimensions={0}
  input2 = s32[4, 2] broadcast(two), dimensions={}
  input2_dynamic = s32[<=4, 2] set-dimension-size(input2, three), dimensions={0}
  add = s32[<=4, 2] add(input1_dynamic, input2_dynamic)
  ROOT reduce = s32[2] reduce(add, zero),
    dimensions={0},
    to_apply=update_s32
}
)";

  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result, PadAndExecute(std::move(module), {}));

  // Array has two valid items in it:
  // [[3, 3],
  //  [3, 3],
  //  [3, 3],
  //  [P, P]]
  // Reducing them gives us [9, 9]
  Literal expected = LiteralUtil::CreateR1<int32_t>({{9, 9}});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicAddWithImplicitSlice) {
  const std::string hlo_text = R"(
HloModule module

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY entry {
  zero = s32[] constant(0)
  one = s32[] constant(1)
  two = s32[] constant(2)
  three = s32[] constant(3)
  input1 = s32[4, 2] broadcast(one), dimensions={}
  input1_dynamic = s32[<=4, 2] set-dimension-size(input1, three), dimensions={0}
  input2 = s32[4, 2] broadcast(two), dimensions={}
  input2_dynamic = s32[<=4, 2] set-dimension-size(input2, two), dimensions={0}
  add = s32[<=4, 2] add(input1_dynamic, input2_dynamic)
  ROOT reduce = s32[2] reduce(add, zero),
    dimensions={0},
    to_apply=update_s32
}
)";

  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result, PadAndExecute(std::move(module), {}));

  // Array has two valid items in it:
  // [[3, 3],
  //  [3, 3],
  //  [P, P],
  //  [P, P]]
  // Reducing them gives us [6, 6]
  Literal expected = LiteralUtil::CreateR1<int32_t>({{6, 6}});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicBinaryOpOverZeroSizedDimension) {
  // Regression test for https://github.com/openxla/xla/issues/49925.
  //
  // Both operands of an elementwise binary op have a dynamic dimension whose
  // static bound is zero, so the dimension is always empty and the two dynamic
  // sizes can never differ. The implicit broadcast rewrite used to slice a
  // size-1 window out of the empty dimension anyway, which produced an
  // out-of-bounds slice that failed HLO verification.
  const std::string hlo_text = R"(
HloModule module

and_pred (lhs: pred[], rhs: pred[]) -> pred[] {
  lhs = pred[] parameter(0)
  rhs = pred[] parameter(1)
  ROOT and = pred[] and(lhs, rhs)
}

ENTRY entry {
  zero_f32 = f32[] constant(0)
  init = pred[] constant(true)
  size_0 = s32[] parameter(0)
  size_1 = s32[] parameter(1)
  a = f32[0, 2] broadcast(zero_f32), dimensions={}
  b = f32[0, 2] broadcast(zero_f32), dimensions={}
  a_dynamic = f32[<=0, 2] set-dimension-size(a, size_0), dimensions={0}
  b_dynamic = f32[<=0, 2] set-dimension-size(b, size_1), dimensions={0}
  cmp = pred[<=0, 2] compare(a_dynamic, b_dynamic), direction=EQ
  ROOT result = pred[2] reduce(cmp, init), dimensions={0}, to_apply=and_pred
}
)";

  auto module = GetHloModule(hlo_text);

  Literal size_0 = LiteralUtil::CreateR0<int32_t>(0);
  Literal size_1 = LiteralUtil::CreateR0<int32_t>(0);
  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&size_0, &size_1}));

  Literal expected = LiteralUtil::CreateR1<bool>({true, true});
  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicStackPop) {
  // This tests the case where a static sized stack is popped by a dynamic
  // number of times.

  // In the beginning the stack has static size that has 4 elements:
  // [[1, 1],
  //  [1, 1],
  //  [1, 1],
  //  [1, 1]]
  //
  // Popping this stack using set-dimension-size in a loop creates a dynamic
  // result depending on how many times we pop it (in this test, two times).

  const std::string hlo_text = R"(
HloModule module

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

body {
  param_tuple = (s32[<=4,2]) parameter(0)
  param = s32[<=4, 2] get-tuple-element(param_tuple), index=0
  one = s32[] constant(1)
  size = s32[] get-dimension-size(param), dimensions={0}
  new_size = s32[] subtract(size, one)
  output = s32[<=4, 2] set-dimension-size(param, new_size), dimensions={0}
  ROOT root = (s32[<=4, 2]) tuple(output)
}

condition {
  stack = (s32[<=4,2]) parameter(0)
  stack_buffer = s32[<=4,2] get-tuple-element(stack), index=0
  stack_size = s32[] get-dimension-size(stack_buffer), dimensions={0}
  two = s32[] constant(2)
  ROOT greater-than = pred[] compare(s32[] stack_size, s32[] two), direction=GT
}

ENTRY entry {
  one = s32[] constant(1)
  zero = s32[] constant(0)
  four = s32[] constant(4)
  stack_buffer_input = s32[4, 2] broadcast(s32[] one), dimensions={}
  stack_buffer_dynamic = s32[<=4, 2] set-dimension-size(stack_buffer_input, four), dimensions={0}
  input_tuple = (s32[<=4, 2]) tuple(stack_buffer_dynamic)
  while = (s32[<=4, 2]) while(input_tuple), body=body, condition=condition
  stack_buffer = s32[<=4, 2] get-tuple-element(while), index=0
  ROOT reduce = s32[2] reduce(stack_buffer, zero),
    dimensions={0},
    to_apply=update_s32
}
)";

  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result, PadAndExecute(std::move(module), {}));

  // Stack has two valid items in it:
  // [[1, 1],
  //  [1, 1],
  //  [P, P],
  //  [P, P]]
  // Reducing them gives us [2, 2]
  Literal expected = LiteralUtil::CreateR1<int32_t>({{2, 2}});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DoubleDynamicDimension) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[2, 3, 3] parameter(0)
  size = s32[] constant(2)
  param_padded_partial = s32[2, <=3, 3] set-dimension-size(param, size),
    dimensions={1}
  param_padded = s32[2, 3, <=3] set-dimension-size(param_padded_partial, size),
    dimensions={2}
  reshaped = s32[<=18] reshape(param_padded)
  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(reshaped, init),
      dimensions={0},
      to_apply=update_s32
}
)";

  // First dimension (1) is dynamic. Since dynamic size is 0, result is also 0.
  Literal operand = LiteralUtil::CreateR3<int32_t>(
      {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}, {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}));

  // Padded data looks like this (P is padding which is ignored).
  // [[0, 1, P]
  // [3, 4, P]
  // [P, P, P]]
  //
  // [[0, 1, P]
  // [3, 4, P]
  // [P, P, P]]
  //
  // Reshaping (with correct reshape rewriting) produces:
  // [0, 1, 3, 4, 0, 1, 3, 4, P, P, P, P, P, P, P, P, P, P]
  //
  // Reducing it produces 16

  Literal expected = LiteralUtil::CreateR0<int32_t>(16);

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicReshapeDoubleDynamicDimensions) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

ENTRY main {
  param = s32[2, 3, 3] parameter(0)
  size = s32[] constant(2)
  param_padded_partial = s32[2, <=3, 3] set-dimension-size(param, size),
    dimensions={1}
  param_padded = s32[2, <=3, <=3] set-dimension-size(param_padded_partial, size),
    dimensions={2}
  result_size = s32[] constant(8)
  ROOT reshaped = s32[<=18] dynamic-reshape(param_padded, result_size)
}
)";

  // First dimension (1) is dynamic. Since dynamic size is 0, result is also 0.
  Literal operand = LiteralUtil::CreateR3<int32_t>(
      {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}, {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}, false));
  result.SetDynamicSize(0, 8);
  // Padded data looks like this (P is padding which is ignored).
  // [[0, 1, P]
  // [3, 4, P]
  // [P, P, P]]
  //
  // [[0, 1, P]
  // [3, 4, P]
  // [P, P, P]]
  //
  // Reshaping (with correct reshape rewriting) produces:
  // [0, 1, 3, 4, 0, 1, 3, 4]
  Literal expected = LiteralUtil::CreateR1<int32_t>({0, 1, 3, 4, 0, 1, 3, 4});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicReshapeOutputDoubleDynamicDimensions) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

ENTRY main {
  param = s32[18] parameter(0)
  eight = s32[] constant(8)
  param_dynamic = s32[<=18] set-dimension-size(param, eight), dimensions={0}
  two = s32[] constant(2)
  // every dimension has dynamic size two.
  ROOT reshaped = s32[2, <=3, <=3] dynamic-reshape(param_dynamic, two, two, two)
}
)";
  Literal operand = LiteralUtil::CreateR1<int32_t>(
      {0, 1, 3, 4, 0, 1, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1});

  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}, false));
  VLOG(1) << " result: " << result.ToString();
  result.SetDynamicSize(1, 2);
  result.SetDynamicSize(2, 2);
  // Padded operand is:
  // [0, 1, 3, 4, 0, 1, 3, 4, P, P ....]
  //
  // Reshaping it should produce:
  // [[0, 1, P]
  // [3, 4, P]
  // [P, P, P]]
  //
  // [[0, 1, P]
  // [3, 4, P]
  // [P, P, P]]
  Literal expected =
      LiteralUtil::CreateR3<int32_t>({{{0, 1}, {3, 4}}, {{0, 1}, {3, 4}}});
  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicReshapeComplicated) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

ENTRY main {
  param = s32[3, 4, 4] parameter(0)
  two = s32[] constant(2)
  param_dynamic = s32[<=3, 4, 4] set-dimension-size(param, two), dimensions={0}
  three = s32[] constant(3)
  param_dynamic1 = s32[<=3, <=4, 4] set-dimension-size(param_dynamic, three), dimensions={1}
  param_dynamic2 = s32[<=3, <=4, <=4] set-dimension-size(param_dynamic1, three), dimensions={2}
  six = s32[] constant(6)

  // Static reshape is from [3, 4, 4] to [6, 8].
  // Dynamic reshape is from [2, 3, 3] to [3, 6].
  ROOT reshaped = s32[<=6, <=8] dynamic-reshape(param_dynamic2, three, six)
}
)";
  Literal operand = LiteralUtil::CreateR3<int32_t>(
      {{{0, 1, 2, -1}, {3, 4, 5, -1}, {6, 7, 8, -1}, {-1, -1, -1, -1}},
       {{9, 8, 7, -1}, {6, 5, 4, -1}, {3, 2, 1, -1}, {-1, -1, -1, -1}},
       {{-1, -1, -1, -1},
        {-1, -1, -1, -1},
        {-1, -1, -1, -1},
        {-1, -1, -1, -1}}});

  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}, false));
  result.SetDynamicSize(0, 3);
  result.SetDynamicSize(1, 6);
  Literal expected = LiteralUtil::CreateR2<int32_t>(
      {{0, 1, 2, 3, 4, 5}, {6, 7, 8, 9, 8, 7}, {6, 5, 4, 3, 2, 1}});
  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, SetGetDimensionSize) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

ENTRY main {
  param = s32[3] parameter(0)
  size = s32[] constant(2)
  param_dynamic_size = s32[3] set-dimension-size(param, size),
    dimensions={0}
  ROOT gds = s32[] get-dimension-size(param_dynamic_size),
    dimensions={0}
}
)";

  // First dimension (1) is dynamic. Since dynamic size is 0, result is also 0.
  Literal operand = LiteralUtil::CreateR1<int32_t>({1, 2, 3});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand}));

  // Should return the size 2 instead of 3.
  Literal expected = LiteralUtil::CreateR0<int32_t>(2);

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicSort) {
  const std::string hlo_text = R"(
HloModule TEST

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

%compare-greater-than (lhs: s32[], rhs: s32[]) -> pred[] {
  %lhs = s32[] parameter(0)
  %rhs = s32[] parameter(1)
  ROOT %compare = pred[] compare(s32[] %lhs, s32[] %rhs), direction=GT
}

ENTRY main {
  param = s32[4] parameter(0)
  size = s32[] constant(3)
  param_dynamic_size = s32[<=4] set-dimension-size(param, size),
    dimensions={0}
  ROOT sort = s32[<=4]{0} sort(s32[4]{0} %param_dynamic_size),
    dimensions={0}, is_stable=false, to_apply=%compare-greater-than
}
)";

  Literal operand = LiteralUtil::CreateR1<int32_t>({1, 4, 3, 2});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand},
                                     /*slice_dynamic_output=*/false));
  Literal expected = LiteralUtil::CreateR1<int32_t>({4, 3, 1, 2});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicPad) {
  const std::string hlo_text = R"(
HloModule TEST

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[4] parameter(0)
  size = s32[] constant(3)
  padding = s32[] constant(2)
  param_dynamic = s32[<=4] set-dimension-size(param, size),
    dimensions={0}
  // Pad head and tail with 2
  pad = s32[<=6] pad(param_dynamic, padding), padding=1_1

  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(pad, init),
    dimensions={0},
    to_apply=update_s32
}
)";

  Literal operand = LiteralUtil::CreateR1<int32_t>({1, 4, 3, 5});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  // After padding head and tail with "2", the effective data will be [2, 1, 4,
  // 3, 2]

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand},
                                     /*slice_dynamic_output=*/false));
  Literal expected = LiteralUtil::CreateR0<int32_t>(12);

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicPadInteriorPadding) {
  const std::string hlo_text = R"(
HloModule TEST

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[4] parameter(0)
  size = s32[] constant(3)
  padding = s32[] constant(2)
  param_dynamic = s32[<=4] set-dimension-size(param, size),
    dimensions={0}
  // Pad interior with constant 2.
  pad = s32[<=7] pad(param_dynamic, padding), padding=0_0_1

  init = s32[] constant(0)
  ROOT reduce = s32[] reduce(pad, init),
    dimensions={0},
    to_apply=update_s32
}
)";

  // Only the first 3 elements are effective: 1, 4, 3
  Literal operand = LiteralUtil::CreateR1<int32_t>({1, 4, 3, 5});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  // After interior padding with "2", the effective data will be
  // [1, 2, 4, 2, 3]
  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand},
                                     /*slice_dynamic_output=*/false));
  Literal expected = LiteralUtil::CreateR0<int32_t>(12);

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicConditionalDimension) {
  const std::string hlo_text = R"(
HloModule module

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

true_branch {
  true_param = (s32[<=3,2]) parameter(0)
  param = s32[<=3, 2] get-tuple-element(true_param), index=0
  add = s32[<=3,2] add(param, param)
  ROOT true_tuple = (s32[<=3,2], s32[<=3,2]) tuple(add, add)
}

false_branch {
  false_param = (s32[<=3,2]) parameter(0)
  param = s32[<=3, 2] get-tuple-element(false_param), index=0
  add = s32[<=3,2] add(param, param)
  ROOT false_tuple = (s32[<=3,2], s32[<=3,2]) tuple(add, add)
}

ENTRY entry {
  param0 = s32[3,2] parameter(0)
  size = s32[] constant(2)
  branch = pred[] constant(false)
  param_dynamic = s32[<=3, 2] set-dimension-size(param0, size), dimensions={0}
  param_tuple = (s32[<=3 ,2]) tuple(param_dynamic)
  conditional = (s32[<=3, 2], s32[<=3, 2]) conditional(branch, param_tuple, param_tuple),
    true_computation=true_branch, false_computation=false_branch
  gte0 = s32[<=3,2] get-tuple-element(conditional), index=1
  init = s32[] constant(0)
  ROOT reduce = s32[2] reduce(gte0, init),
    dimensions={0},
    to_apply=update_s32
}
)";

  Literal operand = LiteralUtil::CreateR2<int32_t>({{0, 1}, {2, 3}, {4, 5}});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand},
                                     /*slice_dynamic_output=*/false));
  Literal expected = LiteralUtil::CreateR1<int32_t>({4, 8});

  EXPECT_EQ(result, expected);
}

TEST_F(ExecutionTest, DynamicTupleSort) {
  const std::string hlo_text = R"(
HloModule TEST

%compare-greater-than (lhs: s32[], rhs: s32[], lhs_2: s32[], lhs_2: s32[]) -> pred[] {
  %lhs = s32[] parameter(0)
  %rhs = s32[] parameter(1)
  %lhs_2 = s32[] parameter(2)
  %rhs_2 = s32[] parameter(3)
  ROOT %compare = pred[] compare(s32[] %lhs, s32[] %rhs), direction=GT
}

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  param = s32[3] parameter(0)
  size = s32[] constant(2)
  param_dynamic_size = s32[<=3] set-dimension-size(param, size),
    dimensions={0}
  sort = (s32[<=3]{0}, s32[<=3]{0}) sort(s32[<=3]{0} %param_dynamic_size,
                                         s32[<=3]{0} %param_dynamic_size),
    dimensions={0}, is_stable=true, to_apply=%compare-greater-than
  ROOT get-tuple-element = s32[<=3]{0} get-tuple-element((s32[<=3]{0}, s32[<=3]{0}) %sort),
    index=0
}
)";

  Literal operand = LiteralUtil::CreateR1<int32_t>({0, 4, 2});
  auto module = GetHloModule(hlo_text);

  ASSERT_OK_AND_ASSIGN(Literal result,
                       PadAndExecute(std::move(module), {&operand},
                                     /*slice_dynamic_output=*/false));
  Literal expected = LiteralUtil::CreateR1<int32_t>({4, 0, 2});

  EXPECT_EQ(result, expected);
}

namespace op = xla::testing::opcode_matchers;

class HloDimensionSizeLegalizerTest : public HloTestBase {
 protected:
  HloDimensionSizeLegalizerTest() {}
};

TEST_F(HloDimensionSizeLegalizerTest, Ok) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule _
ENTRY gds {
  p = s32[3,4] parameter(0)
  size0 = s32[] get-dimension-size(p), dimensions={0}
  size1 = s32[] get-dimension-size(p), dimensions={1}
  ROOT mul = s32[] multiply(size0, size1)
})")
                    .value();
  DynamicPadder pass;
  EXPECT_TRUE(pass.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Multiply(op::Constant(), op::Constant()));
}

TEST_F(HloDimensionSizeLegalizerTest, GetSetSetDimensionSizeRewriter) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule _
ENTRY gds {
  p = s32[3,4] parameter(0)
  size0 = s32[] get-dimension-size(p), dimensions={0}
  p_copy = s32[3,4] copy(p)
  p_copy_dynamic = s32[<=3, 4] set-dimension-size(p_copy, size0), dimensions={0}
  size1 = s32[] get-dimension-size(p_copy_dynamic), dimensions={0}
  ROOT mul = s32[] multiply(size0, size1)
})")
                    .value();
  DynamicPadder pass;
  EXPECT_TRUE(pass.Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Multiply(op::Constant(), op::Constant()));
}

TEST_F(HloDimensionSizeLegalizerTest, IllegalType) {
  auto module = ParseAndReturnUnverifiedModule(R"(
HloModule _
ENTRY gds {
  p = s32[3]{0} parameter(0)
  ROOT gds = s64[] get-dimension-size(p), dimensions={0}
})")
                    .value();
  DynamicPadder pass;
  EXPECT_FALSE(pass.Run(module.get()).ok());
}

TEST_F(HloDimensionSizeLegalizerTest, IllegalDimension) {
  auto module = ParseAndReturnUnverifiedModule(R"(
HloModule _
ENTRY gds {
  p = f32[2,5] parameter(0)
  ROOT gds = s32[] get-dimension-size(p), dimensions={2}
})")
                    .value();
  DynamicPadder pass;
  EXPECT_FALSE(pass.Run(module.get()).ok());
}

class SizeCheckTest : public HloTestBase {
 protected:
  SizeCheckTest() {}
};

TEST_F(SizeCheckTest, CompileTimeCheckBinaryOpFail) {
  auto module = ParseAndReturnUnverifiedModule(R"(
HloModule _
ENTRY gds {
  size_0 = s32[] parameter(0)
  size_1 = s32[] parameter(1)
  arg = s32[4]{0} parameter(2)
  dynamic_arg_0 = s32[<=4] set-dimension-size(arg, size_0), dimensions={0}
  dynamic_arg_1 = s32[<=4] set-dimension-size(arg, size_1), dimensions={0}
  ROOT add = s32[<=4] add(dynamic_arg_0, dynamic_arg_1)
})")
                    .value();
  auto options = DynamicPadderOptions();
  options.shape_check_mode =
      DynamicDimensionInference::ShapeCheckMode::kCompileTime;
  DynamicPadder pass(options);
  auto status = pass.Run(module.get()).status();
  EXPECT_THAT(status.code(), tsl::error::INVALID_ARGUMENT);
}

TEST_F(SizeCheckTest, CompileTimeCheckBinaryOpPass) {
  // Two different sizes.
  auto module = ParseAndReturnUnverifiedModule(R"(
HloModule _
ENTRY gds {
  size_0 = s32[] parameter(0)
  size_0_reshape = s32[1] reshape(size_0)
  size_1 = s32[] reshape(size_0_reshape)
  arg = s32[4]{0} parameter(1)
  dynamic_arg_0 = s32[<=4] set-dimension-size(arg, size_0), dimensions={0}
  dynamic_arg_1 = s32[<=4] set-dimension-size(arg, size_1), dimensions={0}
  ROOT add = s32[<=4] add(dynamic_arg_0, dynamic_arg_1)
})")
                    .value();
  auto options = DynamicPadderOptions();
  options.shape_check_mode =
      DynamicDimensionInference::ShapeCheckMode::kCompileTime;
  DynamicDimensionSimplifier simplifier;
  EXPECT_TRUE(simplifier.Run(module.get()).ok());
  DynamicPadder pass(options);
  auto status = pass.Run(module.get()).status();
  EXPECT_TRUE(status.ok());
}

TEST_F(DynamicPadderTest, ShardingDynamicShapes) {
  constexpr absl::string_view hlo = R"(
HloModule main

ENTRY main {
  token.0 = after-all()
  infeed_tuple.0 = (s32[<=32], token[]) infeed(token.0), sharding={{manual}, {manual}}
  infeed.0 = get-tuple-element(infeed_tuple.0), index=0
  ROOT sharding.0 = s32[<=32] custom-call(infeed.0), custom_call_target="Sharding", sharding={manual}
}
)";
  ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(bool changed, RunPadder(/*slice_dynamic_output=*/true));
  EXPECT_FALSE(changed);
}

TEST_F(DynamicPadderTest, ReplaceSetBoundIgnoresLayout) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule main

    ENTRY main {
      p0 = s32[2,4]{0,1} parameter(0)
      r = s32[2,4]{1,0} reshape(p0)
      ROOT sb = s32[2,4]{1,0} custom-call(r), custom_call_target="SetBound", literal=s32[] 2
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(kHlo));
  DynamicDimensionSimplifier simplifier;
  EXPECT_THAT(simplifier.Run(module_.get()), IsOkAndHolds(true));
  EXPECT_THAT(RunPadder(), IsOkAndHolds(true));
  EXPECT_THAT(module_->entry_computation()->root_instruction(),
              op::Parameter(0));
}

TEST_F(DynamicPadderTest, ReplaceSetSizeIgnoresLayout) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule main

    ENTRY main {
      p0 = s32[2,4]{0,1} parameter(0)
      r = s32[2,4]{1,0} reshape(p0)
      sb = s32[2,4]{1,0} custom-call(r), custom_call_target="SetBound", literal=s32[] 2
      size = s32[] constant(2)
      ROOT sds = s32[<=2,4]{1,0} set-dimension-size(sb, size), dimensions={0}
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(kHlo));
  DynamicDimensionSimplifier simplifier;
  EXPECT_THAT(simplifier.Run(module_.get()), IsOkAndHolds(true));
  EXPECT_THAT(RunPadder(), IsOkAndHolds(true));
  EXPECT_THAT(module_->entry_computation()->root_instruction(),
              op::Parameter(0));
}

TEST_F(ExecutionTest, VariadicDynamicScatterMasksIndicesAndPreservesOperands) {
  constexpr absl::string_view kVariadicScatter = R"hlo(
HloModule VariadicScatter

add_pair {
  a0 = s32[] parameter(0)
  b0 = s32[] parameter(1)
  a1 = s32[] parameter(2)
  b1 = s32[] parameter(3)
  ROOT t = (s32[], s32[]) tuple(s32[] add(a0, a1), s32[] add(b0, b1))
}

ENTRY main {
  op0 = s32[4] parameter(0)
  op1 = s32[4] parameter(1)
  idx = s32[4] parameter(2)
  upd0 = s32[4] parameter(3)
  upd1 = s32[4] parameter(4)
  dsz = s32[] parameter(5)
  idx_d = s32[<=4] set-dimension-size(idx, dsz), dimensions={0}
  u0_d = s32[<=4] set-dimension-size(upd0, dsz), dimensions={0}
  u1_d = s32[<=4] set-dimension-size(upd1, dsz), dimensions={0}
  ROOT sc = (s32[4], s32[4]) scatter(op0, op1, idx_d, u0_d, u1_d),
    update_window_dims={}, inserted_window_dims={0},
    scatter_dims_to_operand_dims={0}, index_vector_dim=1,
    indices_are_sorted=true, unique_indices=true, to_apply=add_pair
}
)hlo";
  auto mod_check = GetHloModule(std::string(kVariadicScatter));
  ASSERT_OK(DynamicPadder().Run(mod_check.get()).status());
  const HloInstruction* sc = FindInstruction(mod_check.get(), "sc");
  ASSERT_NE(sc, nullptr);
  EXPECT_FALSE(sc->ToString().find("indices_are_sorted=true") !=
               std::string::npos);
  EXPECT_TRUE(sc->unique_indices());

  Literal op0 = LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4});
  Literal op1 = LiteralUtil::CreateR1<int32_t>({10, 20, 30, 40});
  Literal idx = LiteralUtil::CreateR1<int32_t>({1, 3, 0, 2});
  Literal u0 = LiteralUtil::CreateR1<int32_t>({100, 300, 999, 888});
  Literal u1 = LiteralUtil::CreateR1<int32_t>({1000, 3000, 9999, 8888});
  Literal dsz = LiteralUtil::CreateR0<int32_t>(2);
  ASSERT_OK_AND_ASSIGN(
      Literal out, PadAndExecute(GetHloModule(std::string(kVariadicScatter)),
                                 {&op0, &op1, &idx, &u0, &u1, &dsz}));
  Literal expected = LiteralUtil::MakeTupleOwned(
      LiteralUtil::CreateR1<int32_t>({1, 102, 3, 304}),
      LiteralUtil::CreateR1<int32_t>({10, 1020, 30, 3040}));
  EXPECT_EQ(out, expected);

  // Dynamic complex scatter_operands[1] must not be padded with MaxValue.
  constexpr absl::string_view kComplexOperand1 = R"hlo(
HloModule ComplexVariadicOperand

upd_fn {
  a0 = c64[] parameter(0)
  b0 = c64[] parameter(1)
  a1 = c64[] parameter(2)
  b1 = c64[] parameter(3)
  ROOT t = (c64[], c64[]) tuple(a1, b1)
}

ENTRY main {
  op0 = c64[4] parameter(0)
  op1 = c64[4] parameter(1)
  dsz = s32[] parameter(2)
  op1_d = c64[<=4] set-dimension-size(op1, dsz), dimensions={0}
  idx = s32[1] constant({0})
  u = c64[1] constant({(7, 8)})
  ROOT sc = (c64[4], c64[<=4]) scatter(op0, op1_d, idx, u, u),
    update_window_dims={}, inserted_window_dims={0},
    scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=upd_fn
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto cmod,
                       ParseAndReturnVerifiedModule(kComplexOperand1));
  EXPECT_THAT(DynamicPadder().Run(cmod.get()), IsOkAndHolds(true));
  ASSERT_OK(verifier().Run(cmod.get()).status());
}

TEST_F(ExecutionTest, NarrowSignedAndUnsignedScatterIndicesOutOfBounds) {
  const bool run_narrow_exec =
      test::DeviceIs(test::kCpu) || test::DeviceIs(test::kInterpreter);
  constexpr absl::string_view kNarrowS8 = R"hlo(
HloModule NarrowS8Scatter

add_s32 {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  ROOT s = s32[] add(a, b)
}

ENTRY main {
  base = s32[200] broadcast(s32[] constant(0)), dimensions={}
  idx = s8[4] parameter(0)
  upd = s32[4] parameter(1)
  dsz = s32[] parameter(2)
  idx_d = s8[<=4] set-dimension-size(idx, dsz), dimensions={0}
  upd_d = s32[<=4] set-dimension-size(upd, dsz), dimensions={0}
  ROOT sc = s32[200] scatter(base, idx_d, upd_d),
    update_window_dims={}, inserted_window_dims={0},
    scatter_dims_to_operand_dims={0}, index_vector_dim=1,
    indices_are_sorted=true, to_apply=add_s32
}
)hlo";
  auto s8_check = GetHloModule(std::string(kNarrowS8));
  EXPECT_THAT(DynamicPadder().Run(s8_check.get()), IsOkAndHolds(true));
  ASSERT_OK(verifier().Run(s8_check.get()).status());
  const HloInstruction* s8_sc = FindInstruction(s8_check.get(), "sc");
  ASSERT_NE(s8_sc, nullptr);
  EXPECT_EQ(s8_sc->ToString().find("indices_are_sorted=true"),
            std::string::npos);
  ASSERT_EQ(s8_sc->operand(1)->opcode(), HloOpcode::kSelect);
  ASSERT_EQ(s8_sc->operand(1)->operand(2)->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(s8_sc->operand(1)->operand(2)->operand(0)->literal(),
            LiteralUtil::CreateR0<int8_t>(-1));
  Literal dsz2 = LiteralUtil::CreateR0<int32_t>(2);
  if (run_narrow_exec) {
    Literal s8_idx = LiteralUtil::CreateR1<int8_t>({2, 5, 0, 1});
    Literal s8_upd = LiteralUtil::CreateR1<int32_t>({11, 22, 777, 888});
    ASSERT_OK_AND_ASSIGN(Literal s8_out,
                         PadAndExecute(GetHloModule(std::string(kNarrowS8)),
                                       {&s8_idx, &s8_upd, &dsz2}));
    EXPECT_EQ(s8_out.Get<int32_t>({2}), 11);
    EXPECT_EQ(s8_out.Get<int32_t>({5}), 22);
    EXPECT_EQ(s8_out.Get<int32_t>({127}), 0);
    EXPECT_EQ(s8_out.Get<int32_t>({0}), 0);
  }

  // u8 into s32[255] is strictly OOB at 255 and preserves indices_are_sorted.
  constexpr absl::string_view kSafeU8 = R"hlo(
HloModule SafeU8Scatter

add_s32 {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  ROOT s = s32[] add(a, b)
}

ENTRY main {
  base = s32[255] broadcast(s32[] constant(0)), dimensions={}
  idx = u8[3] parameter(0)
  upd = s32[3] parameter(1)
  dsz = s32[] parameter(2)
  idx_d = u8[<=3] set-dimension-size(idx, dsz), dimensions={0}
  upd_d = s32[<=3] set-dimension-size(upd, dsz), dimensions={0}
  ROOT sc = s32[255] scatter(base, idx_d, upd_d),
    update_window_dims={}, inserted_window_dims={0},
    scatter_dims_to_operand_dims={0}, index_vector_dim=1,
    indices_are_sorted=true, to_apply=add_s32
}
)hlo";
  auto u8_check = GetHloModule(std::string(kSafeU8));
  EXPECT_THAT(DynamicPadder().Run(u8_check.get()), IsOkAndHolds(true));
  ASSERT_OK(verifier().Run(u8_check.get()).status());
  const HloInstruction* u8_sc = FindInstruction(u8_check.get(), "sc");
  ASSERT_NE(u8_sc, nullptr);
  EXPECT_NE(u8_sc->ToString().find("indices_are_sorted=true"),
            std::string::npos);
  ASSERT_EQ(u8_sc->operand(1)->opcode(), HloOpcode::kSelect);
  ASSERT_EQ(u8_sc->operand(1)->operand(2)->opcode(), HloOpcode::kBroadcast);
  EXPECT_EQ(u8_sc->operand(1)->operand(2)->operand(0)->literal(),
            LiteralUtil::CreateR0<uint8_t>(255));
  Literal dsz1 = LiteralUtil::CreateR0<int32_t>(1);
  if (run_narrow_exec) {
    Literal u8_idx = LiteralUtil::CreateR1<uint8_t>({3, 0, 254});
    Literal u8_upd = LiteralUtil::CreateR1<int32_t>({42, 99, 88});
    ASSERT_OK_AND_ASSIGN(Literal u8_out,
                         PadAndExecute(GetHloModule(std::string(kSafeU8)),
                                       {&u8_idx, &u8_upd, &dsz1}));
    EXPECT_EQ(u8_out.Get<int32_t>({3}), 42);
    EXPECT_EQ(u8_out.Get<int32_t>({254}), 0);
  }

  // 2D u8 into s32[300,100] is strictly OOB via second coordinate (255 > 99).
  constexpr absl::string_view kSafe2DU8 = R"hlo(
HloModule Safe2DU8Scatter

add_s32 {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  ROOT s = s32[] add(a, b)
}

ENTRY main {
  base = s32[300,100] broadcast(s32[] constant(0)), dimensions={}
  idx = u8[2,2] constant({{1, 2}, {0, 0}})
  upd = s32[2] constant({9, 55})
  dsz = s32[] parameter(0)
  idx_d = u8[<=2,2] set-dimension-size(idx, dsz), dimensions={0}
  upd_d = s32[<=2] set-dimension-size(upd, dsz), dimensions={0}
  ROOT sc = s32[300,100] scatter(base, idx_d, upd_d),
    update_window_dims={}, inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1}, index_vector_dim=1, to_apply=add_s32
}
)hlo";
  auto u8_2d_check = GetHloModule(std::string(kSafe2DU8));
  EXPECT_THAT(DynamicPadder().Run(u8_2d_check.get()), IsOkAndHolds(true));
  ASSERT_OK(verifier().Run(u8_2d_check.get()).status());
  if (run_narrow_exec) {
    ASSERT_OK_AND_ASSIGN(
        Literal u8_2d,
        PadAndExecute(GetHloModule(std::string(kSafe2DU8)), {&dsz1}));
    EXPECT_EQ(u8_2d.Get<int32_t>({1, 2}), 9);
    EXPECT_EQ(u8_2d.Get<int32_t>({0, 0}), 0);
  }

  // u8 into s32[256] (where 255 is in bounds), dynamic index_vector_dim, and
  // empty scatter_dims_to_operand_dims must return Unimplemented.
  for (absl::string_view bad_hlo : {absl::string_view(R"hlo(
HloModule InBoundsU8
add_s32 {
  ROOT s = s32[] add(s32[] parameter(0), s32[] parameter(1))
}
ENTRY main {
  base = s32[256] parameter(0)
  dsz = s32[] parameter(3)
  idx_d = u8[<=2] set-dimension-size(u8[2] parameter(1), dsz), dimensions={0}
  upd_d = s32[<=2] set-dimension-size(s32[2] parameter(2), dsz), dimensions={0}
  ROOT sc = s32[256] scatter(base, idx_d, upd_d), update_window_dims={},
    inserted_window_dims={0}, scatter_dims_to_operand_dims={0},
    index_vector_dim=1, to_apply=add_s32
})hlo"),
                                    absl::string_view(R"hlo(
HloModule DynIndexVec
add_s32 {
  ROOT s = s32[] add(s32[] parameter(0), s32[] parameter(1))
}
ENTRY main {
  base = s32[8,8] parameter(0)
  dsz = s32[] parameter(3)
  idx_d = s32[2,<=2] set-dimension-size(s32[2,2] parameter(1), dsz), dimensions={1}
  ROOT sc = s32[8,8] scatter(base, idx_d, s32[2] parameter(2)),
    update_window_dims={}, inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1}, index_vector_dim=1, to_apply=add_s32
})hlo"),
                                    absl::string_view(R"hlo(
HloModule EmptyScatterDims
add_s32 {
  ROOT s = s32[] add(s32[] parameter(0), s32[] parameter(1))
}
ENTRY main {
  base = s32[4] parameter(0)
  dsz = s32[] parameter(3)
  idx_d = s32[<=2,0] set-dimension-size(s32[2,0] parameter(1), dsz), dimensions={0}
  upd_d = s32[<=2,4] set-dimension-size(s32[2,4] parameter(2), dsz), dimensions={0}
  ROOT sc = s32[4] scatter(base, idx_d, upd_d), update_window_dims={1},
    inserted_window_dims={}, scatter_dims_to_operand_dims={},
    index_vector_dim=1, to_apply=add_s32
})hlo")}) {
    ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(bad_hlo));
    EXPECT_EQ(DynamicPadder().Run(m.get()).status().code(),
              absl::StatusCode::kUnimplemented);
  }
}

TEST_F(ExecutionTest, SelectAndScatterReversedGeAndLeSelectsValidMin) {
  auto run_sas = [&](absl::string_view ty, absl::string_view cmp_expr,
                     Literal op, Literal src, Literal init) {
    const std::string hlo = absl::StrReplaceAll(
        R"hlo(
HloModule SasTest
sel {
  p0 = $T$[] parameter(0)
  p1 = $T$[] parameter(1)
  ROOT c = pred[] $CMP$
}
add_fn {
  ROOT s = $T$[] add($T$[] parameter(0), $T$[] parameter(1))
}
ENTRY main {
  op = $T$[4] parameter(0)
  src = $T$[1] parameter(1)
  init = $T$[] parameter(2)
  dsz = s32[] parameter(3)
  op_d = $T$[<=4] set-dimension-size(op, dsz), dimensions={0}
  ROOT r = $T$[<=4] select-and-scatter(op_d, src, init),
    window={size=4 stride=4}, select=sel, scatter=add_fn
}
)hlo",
        {{"$T$", ty}, {"$CMP$", cmp_expr}});
    Literal dsz = LiteralUtil::CreateR0<int32_t>(2);
    return PadAndExecute(GetHloModule(hlo), {&op, &src, &init, &dsz},
                         /*slice_dynamic_output=*/false);
  };

  // Reversed GE (argmin), forward LE (argmin), and reversed LE (argmax) on f32
  // and s32 (where Compare::order() is kTotal).
  ASSERT_OK_AND_ASSIGN(
      Literal r_rev_ge,
      run_sas("f32", "compare(p1, p0), direction=GE",
              LiteralUtil::CreateR1<float>({3.0f, 7.0f, -100.0f, 1.0f}),
              LiteralUtil::CreateR1<float>({5.0f}),
              LiteralUtil::CreateR0<float>(0.0f)));
  EXPECT_EQ(r_rev_ge.Get<float>({0}), 5.0f);
  EXPECT_EQ(r_rev_ge.Get<float>({2}), 0.0f);

  ASSERT_OK_AND_ASSIGN(Literal r_fwd_le,
                       run_sas("s32", "compare(p0, p1), direction=LE",
                               LiteralUtil::CreateR1<int32_t>({9, 4, -50, 2}),
                               LiteralUtil::CreateR1<int32_t>({11}),
                               LiteralUtil::CreateR0<int32_t>(0)));
  EXPECT_EQ(r_fwd_le.Get<int32_t>({1}), 11);
  EXPECT_EQ(r_fwd_le.Get<int32_t>({2}), 0);

  ASSERT_OK_AND_ASSIGN(Literal r_rev_le,
                       run_sas("s32", "compare(p1, p0), direction=LE",
                               LiteralUtil::CreateR1<int32_t>({4, 9, 999, 2}),
                               LiteralUtil::CreateR1<int32_t>({13}),
                               LiteralUtil::CreateR0<int32_t>(0)));
  EXPECT_EQ(r_rev_le.Get<int32_t>({1}), 13);
  EXPECT_EQ(r_rev_le.Get<int32_t>({2}), 0);

  // Reflexive compare(p0, p0) GE and float TOTALORDER must be rejected.
  for (absl::string_view bad_cmp :
       {"compare(p0, p0), direction=GE",
        "compare(p0, p1), direction=GE, type=TOTALORDER"}) {
    const std::string hlo = absl::StrReplaceAll(
        R"hlo(
HloModule BadSas
sel {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT c = pred[] $CMP$
}
add_fn {
  ROOT s = f32[] add(f32[] parameter(0), f32[] parameter(1))
}
ENTRY main {
  op_d = f32[<=4] set-dimension-size(f32[4] parameter(0), s32[] parameter(1)), dimensions={0}
  ROOT r = f32[<=4] select-and-scatter(op_d, f32[1] constant({1}), f32[] constant(0)),
    window={size=4 stride=4}, select=sel, scatter=add_fn
}
)hlo",
        {{"$CMP$", bad_cmp}});
    ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
    EXPECT_EQ(DynamicPadder().Run(m.get()).status().code(),
              absl::StatusCode::kUnimplemented);
  }
}

TEST_F(ExecutionTest, MixedPrecisionAndAuxiliaryOperandDotConvPadding) {
  constexpr absl::string_view kMixedFloatDot = R"hlo(
HloModule MixedFloatDot
ENTRY main {
  lhs = f32[2,4] parameter(0)
  rhs = bf16[4,2] parameter(1)
  dsz = s32[] parameter(2)
  lhs_d = f32[2,<=4] set-dimension-size(lhs, dsz), dimensions={1}
  rhs_d = bf16[<=4,2] set-dimension-size(rhs, dsz), dimensions={0}
  ROOT d = f32[2,2] dot(lhs_d, rhs_d),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto fdot, ParseAndReturnVerifiedModule(kMixedFloatDot));
  EXPECT_THAT(DynamicPadder().Run(fdot.get()), IsOkAndHolds(true));
  ASSERT_OK(verifier().Run(fdot.get()).status());

  constexpr absl::string_view kMixedIntDot = R"hlo(
HloModule MixedIntDot
ENTRY main {
  lhs = s8[2,4] constant({{1, 2, 99, 99}, {3, 4, 99, 99}})
  rhs = s16[4,2] constant({{10, 20}, {30, 40}, {999, 999}, {999, 999}})
  dsz = s32[] parameter(0)
  lhs_d = s8[2,<=4] set-dimension-size(lhs, dsz), dimensions={1}
  rhs_d = s16[<=4,2] set-dimension-size(rhs, dsz), dimensions={0}
  ROOT d = s32[2,2] dot(lhs_d, rhs_d),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)hlo";
  auto idot_check = GetHloModule(std::string(kMixedIntDot));
  EXPECT_THAT(DynamicPadder().Run(idot_check.get()), IsOkAndHolds(true));
  ASSERT_OK(verifier().Run(idot_check.get()).status());
  if (test::DeviceIs(test::kCpu) || test::DeviceIs(test::kInterpreter)) {
    Literal dsz2 = LiteralUtil::CreateR0<int32_t>(2);
    ASSERT_OK_AND_ASSIGN(
        Literal idot_out,
        PadAndExecute(GetHloModule(std::string(kMixedIntDot)), {&dsz2}));
    EXPECT_EQ(idot_out,
              LiteralUtil::CreateR2<int32_t>({{70, 100}, {150, 220}}));
  }

  // F8E8M0FNU has no zero representation and must return Unimplemented.
  constexpr absl::string_view kE8M0Dot = R"hlo(
HloModule E8M0Dot
ENTRY main {
  dsz = s32[] parameter(2)
  a = f8e8m0fnu[2,<=4] set-dimension-size(f8e8m0fnu[2,4] parameter(0), dsz), dimensions={1}
  b = f8e8m0fnu[<=4,2] set-dimension-size(f8e8m0fnu[4,2] parameter(1), dsz), dimensions={0}
  ROOT d = f32[2,2] dot(a, b),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto e8m0, ParseAndReturnUnverifiedModule(kE8M0Dot));
  EXPECT_EQ(DynamicPadder().Run(e8m0.get()).status().code(),
            absl::StatusCode::kUnimplemented);

  // Auxiliary operand >= 2 (e.g. block-scaled dot) must return Unimplemented.
  constexpr absl::string_view kScaledDot = R"hlo(
HloModule AuxOperandDot
ENTRY main {
  a = f32[2,4] parameter(0)
  b = f32[4,2] parameter(1)
  s = f32[<=4,2] set-dimension-size(f32[4,2] parameter(2), s32[] parameter(3)), dimensions={0}
  ROOT d = f32[2,2] dot(a, b, s, b),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto sdot, ParseAndReturnUnverifiedModule(kScaledDot));
  EXPECT_EQ(DynamicPadder().Run(sdot.get()).status().code(),
            absl::StatusCode::kUnimplemented);
}

TEST_F(DynamicPadderTest,
       DynamicConvInputGradForwardKernelGradPrecisionAndMixedTypes) {
  constexpr absl::string_view kInputGrad = R"hlo(
HloModule DynConvInputGrad
ENTRY main {
  sizes = s32[3] constant({1, 3, 1})
  grad_d = bf16[1,<=4,1] set-dimension-size(bf16[1,4,1] parameter(0), s32[] parameter(2)), dimensions={1}
  kernel = bf16[2,1,1] parameter(1)
  ROOT conv = f32[1,4,1] custom-call(sizes, grad_d, kernel),
    window={size=2 pad=0_1}, dim_labels=b0f_0io->b0f, padding_type=PADDING_SAME,
    operand_precision={default, high, highest},
    custom_call_target="DynamicConvolutionInputGrad"
}
)hlo";
  ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(kInputGrad));
  for (HloInstruction* inst : module_->entry_computation()->instructions()) {
    if (inst->IsCustomCall("DynamicConvolutionInputGrad")) {
      inst->mutable_precision_config()->set_algorithm(
          PrecisionConfig::ALG_DOT_BF16_BF16_F32);
    }
  }
  EXPECT_THAT(RunPadder(), IsOkAndHolds(true));
  ASSERT_OK(verifier().Run(module_.get()).status());
  const HloInstruction* conv = module_->entry_computation()->root_instruction();
  ASSERT_EQ(conv->opcode(), HloOpcode::kConvolution);
  EXPECT_EQ(conv->precision_config().algorithm(),
            PrecisionConfig::ALG_DOT_BF16_BF16_F32);
  ASSERT_EQ(conv->precision_config().operand_precision_size(), 2);
  EXPECT_EQ(conv->precision_config().operand_precision(0),
            PrecisionConfig::HIGH);
  EXPECT_EQ(conv->precision_config().operand_precision(1),
            PrecisionConfig::HIGHEST);

  constexpr absl::string_view kForwardS8ToS32 = R"hlo(
HloModule DynConvFwdS8
ENTRY main {
  in_d = s8[1,<=4,1] set-dimension-size(s8[1,4,1] parameter(0), s32[] parameter(2)), dimensions={1}
  k = s8[2,1,1] parameter(1)
  ROOT conv = s32[1,4,1] custom-call(in_d, k),
    window={size=2 pad=0_1}, dim_labels=b0f_0io->b0f, padding_type=PADDING_SAME,
    custom_call_target="DynamicConvolutionForward"
}
)hlo";
  ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(kForwardS8ToS32));
  EXPECT_THAT(RunPadder(), IsOkAndHolds(true));
  ASSERT_OK(verifier().Run(module_.get()).status());
  EXPECT_EQ(module_->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kConvolution);

  constexpr absl::string_view kKernelGradMixed = R"hlo(
HloModule DynConvKernelGrad
ENTRY main {
  dsz = s32[] parameter(2)
  act = bf16[1,<=4,1] set-dimension-size(bf16[1,4,1] parameter(0), dsz), dimensions={1}
  grd = f16[1,<=4,1] set-dimension-size(f16[1,4,1] parameter(1), dsz), dimensions={1}
  ROOT conv = f32[1,1,1] custom-call(act, grd),
    window={size=4 pad=0_0}, dim_labels=f0b_i0o->0bf, padding_type=PADDING_SAME,
    custom_call_target="DynamicConvolutionKernelGrad"
}
)hlo";
  ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(kKernelGradMixed));
  EXPECT_THAT(DynamicPadder().Run(module_.get()), IsOkAndHolds(true));
  ASSERT_OK(verifier().Run(module_.get()).status());
  EXPECT_EQ(module_->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kConvolution);
}

TEST_F(ExecutionTest,
       DynamicVariadicSortAsRootWithShardingControlDepsAndSubByte) {
  constexpr absl::string_view kVariadicSort = R"hlo(
HloModule VariadicRootSort
cmp {
  k0 = s32[] parameter(0)
  k1 = s32[] parameter(1)
  v0 = f32[] parameter(2)
  v1 = f32[] parameter(3)
  ROOT lt = pred[] compare(k0, k1), direction=LT
}
ENTRY main {
  keys = s32[4] parameter(0)
  vals = f32[4] parameter(1)
  dsz = s32[] parameter(2)
  kd = s32[<=4] set-dimension-size(keys, dsz), dimensions={0}
  vd = f32[<=4] set-dimension-size(vals, dsz), dimensions={0}
  ROOT s = (s32[<=4], f32[<=4]) sort(kd, vd), dimensions={0}, to_apply=cmp
}
)hlo";
  Literal keys = LiteralUtil::CreateR1<int32_t>({30, 10, -99, 5});
  Literal vals = LiteralUtil::CreateR1<float>({3.0f, 1.0f, 9.0f, 5.0f});
  Literal dsz = LiteralUtil::CreateR0<int32_t>(2);
  for (bool slice_out : {false, true}) {
    if (slice_out && test::DeviceIs(test::kInterpreter)) {
      DynamicPadderOptions opts;
      opts.slice_dynamic_output = true;
      auto sm = GetHloModule(std::string(kVariadicSort));
      EXPECT_THAT(DynamicPadder(opts).Run(sm.get()), IsOkAndHolds(true));
      ASSERT_OK(verifier().Run(sm.get()).status());
      continue;
    }
    ASSERT_OK_AND_ASSIGN(Literal out,
                         PadAndExecute(GetHloModule(std::string(kVariadicSort)),
                                       {&keys, &vals, &dsz}, slice_out));
    const auto& tup = out.DecomposeTuple();
    ASSERT_EQ(tup.size(), 2);
    EXPECT_EQ(tup[0].Get<int32_t>({0}), 10);
    EXPECT_EQ(tup[0].Get<int32_t>({1}), 30);
    EXPECT_EQ(tup[1].Get<float>({0}), 1.0f);
    EXPECT_EQ(tup[1].Get<float>({1}), 3.0f);
  }

  // Verify tuple/array sharding (including shard_group), control dependencies,
  // and sub-byte s4 {:E(4)} layout preservation.
  constexpr absl::string_view kShardedCtrlSort = R"hlo(
HloModule ShardedCtrlSort
cmp_s4 {
  a = s4[] parameter(0)
  b = s4[] parameter(1)
  c = s32[] parameter(2)
  d = s32[] parameter(3)
  ROOT lt = pred[] compare(a, b), direction=LT
}
cmp_1 {
  ROOT lt = pred[] compare(s32[] parameter(0), s32[] parameter(1)), direction=LT
}
ENTRY main {
  k = s4[4]{0:E(4)} parameter(0)
  v = s32[4]{0} parameter(1)
  dsz = s32[] parameter(2)
  kd = s4[<=4]{0:E(4)} set-dimension-size(k, dsz), dimensions={0}
  vd = s32[<=4]{0} set-dimension-size(v, dsz), dimensions={0}
  s_tup = (s4[<=4]{0:E(4)}, s32[<=4]{0}) sort(kd, vd), dimensions={0},
    to_apply=cmp_s4, control-predecessors={dsz},
    sharding={{devices=[2]0,1 shard_as 1}, {replicated}}
  s_arr = s32[<=4]{0} sort(vd), dimensions={0}, to_apply=cmp_1,
    control-predecessors={s_tup}, sharding={devices=[2]0,1}
  ROOT out = ((s4[<=4]{0:E(4)}, s32[<=4]{0}), s32[<=4]{0}) tuple(s_tup, s_arr)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto smod,
                       ParseAndReturnVerifiedModule(kShardedCtrlSort));
  EXPECT_THAT(DynamicPadder().Run(smod.get()), IsOkAndHolds(true));
  ASSERT_OK(verifier().Run(smod.get()).status());
  for (const HloInstruction* inst : smod->entry_computation()->instructions()) {
    if (inst->shape().IsArray() && inst->shape().has_layout() &&
        (inst->shape().element_type() == S32 ||
         inst->shape().element_type() == PRED)) {
      EXPECT_EQ(inst->shape().layout().element_size_in_bits(), 0);
    }
  }
}

TEST_F(ExecutionTest, DynamicUpdateSliceRightEdgeClamping1DAnd2D) {
  constexpr absl::string_view kDus1D = R"hlo(
HloModule DusRightEdge1D
ENTRY main {
  base = s32[5] parameter(0)
  upd = s32[3] parameter(1)
  dsz = s32[] parameter(2)
  start = s32[] parameter(3)
  upd_d = s32[<=3] set-dimension-size(upd, dsz), dimensions={0}
  ROOT r = s32[5] dynamic-update-slice(base, upd_d, start)
}
)hlo";
  Literal base = LiteralUtil::CreateR1<int32_t>({10, 20, 30, 40, 50});
  Literal upd = LiteralUtil::CreateR1<int32_t>({7, 8, 9});
  auto run_1d = [&](int32_t d, int32_t s) {
    Literal ld = LiteralUtil::CreateR0<int32_t>(d);
    Literal ls = LiteralUtil::CreateR0<int32_t>(s);
    return PadAndExecute(GetHloModule(std::string(kDus1D)),
                         {&base, &upd, &ld, &ls});
  };
  EXPECT_EQ(run_1d(1, 3).value(),
            LiteralUtil::CreateR1<int32_t>({10, 20, 30, 7, 50}));
  EXPECT_EQ(run_1d(1, 4).value(),
            LiteralUtil::CreateR1<int32_t>({10, 20, 30, 40, 7}));
  EXPECT_EQ(run_1d(1, -2).value(),
            LiteralUtil::CreateR1<int32_t>({7, 20, 30, 40, 50}));
  EXPECT_EQ(run_1d(0, 4).value(),
            LiteralUtil::CreateR1<int32_t>({10, 20, 30, 40, 50}));
  EXPECT_EQ(run_1d(3, 4).value(),
            LiteralUtil::CreateR1<int32_t>({10, 20, 7, 8, 9}));

  // U64 start > INT64_MAX clamps to N - d without signed overflow, and dynamic
  // base clamps to base_dyn - d.
  constexpr absl::string_view kDusU64DynBase = R"hlo(
HloModule DusU64DynBase
ENTRY main {
  base_d = s32[<=5] set-dimension-size(s32[5] parameter(0), s32[] parameter(3)), dimensions={0}
  upd_d = s32[<=3] set-dimension-size(s32[3] parameter(1), s32[] parameter(2)), dimensions={0}
  start = u64[] parameter(4)
  ROOT r = s32[<=5] dynamic-update-slice(base_d, upd_d, start)
}
)hlo";
  Literal d1 = LiteralUtil::CreateR0<int32_t>(1);
  Literal b4 = LiteralUtil::CreateR0<int32_t>(4);
  auto u64_check = GetHloModule(std::string(kDusU64DynBase));
  EXPECT_THAT(DynamicPadder().Run(u64_check.get()), IsOkAndHolds(true));
  ASSERT_OK(verifier().Run(u64_check.get()).status());
  if (test::DeviceIs(test::kCpu) || test::DeviceIs(test::kInterpreter)) {
    Literal u64_big = LiteralUtil::CreateR0<uint64_t>(uint64_t{1} << 63);
    ASSERT_OK_AND_ASSIGN(
        Literal u64_out,
        PadAndExecute(GetHloModule(std::string(kDusU64DynBase)),
                      {&base, &upd, &d1, &b4, &u64_big},
                      /*slice_dynamic_output=*/false));
    EXPECT_EQ(u64_out, LiteralUtil::CreateR1<int32_t>({10, 20, 30, 7, 50}));
  }

  // Also test U32 start > INT32_MAX with dynamic base across all backends.
  constexpr absl::string_view kDusU32DynBase = R"hlo(
HloModule DusU32DynBase
ENTRY main {
  base_d = s32[<=5] set-dimension-size(s32[5] parameter(0), s32[] parameter(3)), dimensions={0}
  upd_d = s32[<=3] set-dimension-size(s32[3] parameter(1), s32[] parameter(2)), dimensions={0}
  start = u32[] parameter(4)
  ROOT r = s32[<=5] dynamic-update-slice(base_d, upd_d, start)
}
)hlo";
  Literal u32_big = LiteralUtil::CreateR0<uint32_t>(uint32_t{1} << 31);
  ASSERT_OK_AND_ASSIGN(Literal u32_out,
                       PadAndExecute(GetHloModule(std::string(kDusU32DynBase)),
                                     {&base, &upd, &d1, &b4, &u32_big},
                                     /*slice_dynamic_output=*/false));
  EXPECT_EQ(u32_out, LiteralUtil::CreateR1<int32_t>({10, 20, 30, 7, 50}));

  // 2D partial update and sub-byte s4 {:E(4)} layout verification.
  constexpr absl::string_view kDus2D = R"hlo(
HloModule Dus2D
ENTRY main {
  base = s32[3,4] constant({{1,2,3,4},{5,6,7,8},{9,10,11,12}})
  upd = s32[2,3] constant({{91,92,93},{94,95,96}})
  d0 = s32[] parameter(0)
  d1 = s32[] parameter(1)
  upd0 = s32[<=2,3] set-dimension-size(upd, d0), dimensions={0}
  upd1 = s32[<=2,<=3] set-dimension-size(upd0, d1), dimensions={1}
  ROOT r = s32[3,4] dynamic-update-slice(base, upd1, s32[] constant(2), s32[] constant(3))
}
)hlo";
  ASSERT_OK_AND_ASSIGN(
      Literal out_2d,
      PadAndExecute(GetHloModule(std::string(kDus2D)), {&d1, &d1}));
  EXPECT_EQ(out_2d, LiteralUtil::CreateR2<int32_t>(
                        {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 91}}));

  constexpr absl::string_view kDusS4 = R"hlo(
HloModule DusS4
ENTRY main {
  base = s4[5]{0:E(4)} parameter(0)
  upd_d = s4[<=3]{0:E(4)} set-dimension-size(s4[3]{0:E(4)} parameter(1), s32[] parameter(2)), dimensions={0}
  ROOT r = s4[5]{0:E(4)} dynamic-update-slice(base, upd_d, s4[]{:E(4)} parameter(3))
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto s4_mod, ParseAndReturnVerifiedModule(kDusS4));
  EXPECT_THAT(DynamicPadder().Run(s4_mod.get()), IsOkAndHolds(true));
  ASSERT_OK(verifier().Run(s4_mod.get()).status());
  EXPECT_EQ(s4_mod->entry_computation()
                ->root_instruction()
                ->shape()
                .layout()
                .element_size_in_bits(),
            4);
  for (const HloInstruction* inst :
       s4_mod->entry_computation()->instructions()) {
    if (inst->shape().IsArray() && inst->shape().has_layout() &&
        inst->shape().element_type() != S4) {
      EXPECT_EQ(inst->shape().layout().element_size_in_bits(), 0);
    }
  }
}

TEST_F(DynamicPadderTest,
       ZeroBoundDynamicReshapeReverseAndUnboundedGetDimensionSize) {
  for (absl::string_view ok_hlo : {absl::string_view(R"hlo(
HloModule ZeroSplit
ENTRY main {
  x = f32[<=0] set-dimension-size(f32[0] parameter(0), s32[] constant(0)), dimensions={0}
  ROOT r = f32[0,<=2] dynamic-reshape(x, s32[] constant(0), s32[] parameter(1))
})hlo"),
                                   absl::string_view(R"hlo(
HloModule ZeroCombine
ENTRY main {
  x = f32[0,<=2] set-dimension-size(f32[0,2] parameter(0), s32[] parameter(1)), dimensions={1}
  ROOT r = f32[<=0] dynamic-reshape(x, s32[] constant(0))
})hlo"),
                                   absl::string_view(R"hlo(
HloModule ZeroFlattenUnflatten
ENTRY main {
  x = f32[<=0,2] set-dimension-size(f32[0,2] parameter(0), s32[] constant(0)), dimensions={0}
  ROOT r = f32[2,<=0] dynamic-reshape(x, s32[] constant(2), s32[] constant(0))
})hlo"),
                                   absl::string_view(R"hlo(
HloModule ZeroReverse
ENTRY main {
  x = f32[<=0] set-dimension-size(f32[0] parameter(0), s32[] constant(0)), dimensions={0}
  ROOT r = f32[<=0] reverse(x), dimensions={0}
})hlo")}) {
    ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(ok_hlo));
    ASSERT_OK(DynamicPadder().Run(m.get()).status());
  }

  // Dynamic reverse bound > INT32_MAX and get-dimension-size on an unbounded
  // dimension must return Unimplemented without S32 truncation or overflow.
  ASSERT_OK_AND_ASSIGN(auto huge_rev, ParseAndReturnVerifiedModule(R"hlo(
HloModule HugeZeroByteReverse
ENTRY main {
  x = s32[<=2147483648,0] parameter(0)
  ROOT r = s32[<=2147483648,0] reverse(x), dimensions={0}
})hlo"));
  EXPECT_EQ(DynamicPadder().Run(huge_rev.get()).status().code(),
            absl::StatusCode::kUnimplemented);

  ASSERT_OK_AND_ASSIGN(auto unbounded_gds, ParseAndReturnUnverifiedModule(R"hlo(
HloModule UnboundedGetDimSize
ENTRY main {
  p = s32[?,0] iota(), iota_dimension=0
  ROOT g = s32[] get-dimension-size(p), dimensions={0}
})hlo"));
  EXPECT_EQ(DynamicPadder().Run(unbounded_gds.get()).status().code(),
            absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace xla
