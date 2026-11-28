/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/batchnorm_expander.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/error_spec.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/transforms/dynamic_padder.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/dynamic_dimension_inference.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class BatchNormExpanderTest : public HloInterpreterReferenceMixin<HloTestBase> {
 protected:
  // BatchNorm should have a dynamic sized divider for mean operations.
  int64_t CountGetDimensionSize(const HloModule& module) {
    int64_t count = 0;
    for (HloComputation* comp : module.computations()) {
      for (HloInstruction* inst : comp->instructions()) {
        if (inst->opcode() == HloOpcode::kGetDimensionSize) {
          count++;
        }
      }
    }
    return count;
  }
};

// Test that we expand BatchNormTraining.
TEST_F(BatchNormExpanderTest, BatchNormTraining) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 2, 2, 2});
  Shape scale_shape = ShapeUtil::MakeShape(F32, {2});
  Shape offset_shape = ShapeUtil::MakeShape(F32, {2});

  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "activation"));

  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scale_shape, "scale"));

  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, offset_shape, "offset"));

  builder.AddInstruction(HloInstruction::CreateBatchNormTraining(
      ShapeUtil::MakeTupleShape({input_shape, scale_shape, offset_shape}),
      param0, param1, param2,
      /*epsilon=*/0.001, /*feature_index=*/3));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBatchNormTraining);
  BatchNormExpander rewriter(/*rewrite_training_op=*/true,
                             /*rewrite_inference_op=*/true,
                             /*rewrite_grad_op=*/true);
  ASSERT_TRUE(rewriter.Run(module.get()).value());
  root = computation->root_instruction();
  EXPECT_EQ(CountGetDimensionSize(*module), 3);
  // Make sure this operation is expanded.
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
}

// Test that we expand BatchNormGrad.
TEST_F(BatchNormExpanderTest, BatchNormGrad) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 2, 2, 2});
  Shape scale_shape = ShapeUtil::MakeShape(F32, {2});
  Shape mean_shape = ShapeUtil::MakeShape(F32, {2});
  Shape var_shape = ShapeUtil::MakeShape(F32, {2});
  Shape grad_output_shape = ShapeUtil::MakeShape(F32, {2, 2, 2, 2});

  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "activation"));

  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scale_shape, "scale"));

  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(2, mean_shape, "mean"));

  HloInstruction* param3 = builder.AddInstruction(
      HloInstruction::CreateParameter(3, var_shape, "var"));

  HloInstruction* param4 = builder.AddInstruction(
      HloInstruction::CreateParameter(4, grad_output_shape, "grad_output"));

  builder.AddInstruction(HloInstruction::CreateBatchNormGrad(
      ShapeUtil::MakeTupleShape({input_shape, scale_shape, mean_shape}), param0,
      param1, param2, param3, param4,
      /*epsilon=*/0.001, /*feature_index=*/3));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kBatchNormGrad);
  BatchNormExpander rewriter(/*rewrite_training_op=*/true,
                             /*rewrite_inference_op=*/true,
                             /*rewrite_grad_op=*/true);
  ASSERT_TRUE(rewriter.Run(module.get()).value());
  root = computation->root_instruction();
  EXPECT_EQ(CountGetDimensionSize(*module), 3);
  // Make sure this operation is expanded.
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
}

TEST_F(BatchNormExpanderTest, BatchNormTrainingSharding) {
  const char* module_str = R"(
HloModule module
ENTRY entry {
  %param.0 = f32[8,4] parameter(0)
  %param.1 = f32[4] parameter(1)
  %param.2 = f32[4] parameter(2)
  ROOT %batch-norm-training = (f32[8,4], f32[4], f32[4])
    batch-norm-training(f32[8,4] %param.0, f32[4] %param.1, f32[4] %param.2),
    epsilon=0.001, feature_index=1, sharding={{maximal device=1},{maximal device=1},{maximal device=1}}
})";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  BatchNormExpander rewriter(/*rewrite_training_op=*/true,
                             /*rewrite_inference_op=*/true,
                             /*rewrite_grad_op=*/true);
  ASSERT_TRUE(rewriter.Run(m.get()).value());

  for (auto* instruction : m->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      continue;
    }
    auto device = instruction->sharding_unique_device();
    ASSERT_TRUE(device);
    EXPECT_EQ(*device, 1);
  }
}

// Test that we expand BatchNormInference.
TEST_F(BatchNormExpanderTest, BatchNormInference) {
  const char* module_str = R"(
HloModule module
ENTRY entry {
  %param.0 = f32[2,2,2,2] parameter(0)
  %param.1 = f32[2] parameter(1)
  %param.2 = f32[2] parameter(2)
  %param.3 = f32[2] parameter(3)
  %param.4 = f32[2] parameter(4)
  ROOT %batch-norm-inference = f32[2,2,2,2]
    batch-norm-inference(f32[2,2,2,2] %param.0, f32[2] %param.1, f32[2] %param.2,
                         f32[2] %param.3, f32[2] %param.4),
    epsilon=0.001, feature_index=3
})";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  BatchNormExpander rewriter(/*rewrite_training_op=*/false,
                             /*rewrite_inference_op=*/true,
                             /*rewrite_grad_op=*/false);
  ASSERT_THAT(rewriter.Run(m.get()), absl_testing::IsOkAndHolds(true));
  // BatchNormInference expands to a single result (not a tuple).
  EXPECT_NE(m->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kBatchNormInference);
}

TEST_F(BatchNormExpanderTest, BatchNormInferenceSharding) {
  const char* module_str = R"(
HloModule module
ENTRY entry {
  %param.0 = f32[8,4] parameter(0)
  %param.1 = f32[4] parameter(1)
  %param.2 = f32[4] parameter(2)
  %param.3 = f32[4] parameter(3)
  %param.4 = f32[4] parameter(4)
  ROOT %batch-norm-inference = f32[8,4]
    batch-norm-inference(f32[8,4] %param.0, f32[4] %param.1, f32[4] %param.2,
                         f32[4] %param.3, f32[4] %param.4),
    epsilon=0.001, feature_index=1, sharding={maximal device=1}
})";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  BatchNormExpander rewriter(/*rewrite_training_op=*/false,
                             /*rewrite_inference_op=*/true,
                             /*rewrite_grad_op=*/false);
  ASSERT_THAT(rewriter.Run(m.get()), absl_testing::IsOkAndHolds(true));

  for (auto* instruction : m->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      continue;
    }
    auto device = instruction->sharding_unique_device();
    ASSERT_TRUE(device);
    EXPECT_EQ(*device, 1);
  }
}

TEST_F(BatchNormExpanderTest, BatchNormGradSharding) {
  const char* module_str = R"(
HloModule module
ENTRY entry {
  %param.0 = f32[8,4] parameter(0)
  %param.1 = f32[4] parameter(1)
  %param.2 = f32[4] parameter(2)
  %param.3 = f32[4] parameter(3)
  %param.4 = f32[8,4] parameter(4)
  ROOT %batch-norm-grad = (f32[8,4], f32[4], f32[4])
    batch-norm-grad(f32[8,4] %param.0, f32[4] %param.1, f32[4] %param.2,
                    f32[4] %param.3, f32[8,4] %param.4),
    epsilon=0.001, feature_index=1, sharding={{maximal device=1},{maximal device=1},{maximal device=1}}
})";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  BatchNormExpander rewriter(/*rewrite_training_op=*/false,
                             /*rewrite_inference_op=*/false,
                             /*rewrite_grad_op=*/true);
  ASSERT_THAT(rewriter.Run(m.get()), absl_testing::IsOkAndHolds(true));

  for (auto* instruction : m->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      continue;
    }
    auto device = instruction->sharding_unique_device();
    ASSERT_TRUE(device);
    EXPECT_EQ(*device, 1);
  }
}

TEST_F(BatchNormExpanderTest, DisabledFlagsAreNoop) {
  const char* module_str = R"(
HloModule module
ENTRY entry {
  %param.0 = f32[2,2,2,2] parameter(0)
  %param.1 = f32[2] parameter(1)
  %param.2 = f32[2] parameter(2)
  ROOT %batch-norm-training = (f32[2,2,2,2], f32[2], f32[2])
    batch-norm-training(f32[2,2,2,2] %param.0, f32[2] %param.1, f32[2] %param.2),
    epsilon=0.001, feature_index=3
})";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(module_str));
  // All rewrite flags disabled — pass should not change the module.
  BatchNormExpander rewriter(/*rewrite_training_op=*/false,
                             /*rewrite_inference_op=*/false,
                             /*rewrite_grad_op=*/false);
  EXPECT_THAT(rewriter.Run(m.get()), absl_testing::IsOkAndHolds(false));
  EXPECT_EQ(m->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kBatchNormTraining);
}

TEST_F(BatchNormExpanderTest, Execution) {
  const char* module_str = R"(
HloModule module
ENTRY entry {
  %param.0 = f32[8,4] parameter(0)
  %param.1 = f32[4] parameter(1)
  %param.2 = f32[4] parameter(2)
  ROOT %batch-norm-training = (f32[8,4], f32[4], f32[4])
    batch-norm-training(f32[8,4] %param.0, f32[4] %param.1, f32[4] %param.2),
    epsilon=0.001, feature_index=1, sharding={{maximal device=1},{maximal device=1},{maximal device=1}}
})";
  EXPECT_TRUE(RunAndCompare(module_str, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(BatchNormExpanderTest,
       WidensLargeElementCountAndRejectsOverInt32DynamicBound) {
  BatchNormExpander rewriter(/*rewrite_training_op=*/true,
                             /*rewrite_inference_op=*/true,
                             /*rewrite_grad_op=*/true);
  auto has_scalar_op = [](const HloModule& m, HloOpcode op, PrimitiveType ty) {
    return absl::c_any_of(
        m.entry_computation()->instructions(), [&](const HloInstruction* h) {
          return h->opcode() == op &&
                 ShapeUtil::Equal(h->shape(), ShapeUtil::MakeShape(ty, {}));
        });
  };

  // N = 2^31 accumulates in S64 without signed S32 wraparound.
  ASSERT_OK_AND_ASSIGN(auto m_s64, ParseAndReturnVerifiedModule(R"(
HloModule s64_count
ENTRY entry {
  x = f32[65536,32768,2] parameter(0)
  s = f32[2] parameter(1)
  o = f32[2] parameter(2)
  ROOT bn = (f32[65536,32768,2], f32[2], f32[2])
    batch-norm-training(x, s, o), epsilon=0.001, feature_index=2
})"));
  ASSERT_THAT(rewriter.Run(m_s64.get()), absl_testing::IsOkAndHolds(true));
  ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                          /*allow_mixed_precision=*/false)
                  .Run(m_s64.get()),
              absl_testing::IsOk());
  EXPECT_TRUE(has_scalar_op(*m_s64, HloOpcode::kMultiply, S64));
  EXPECT_FALSE(has_scalar_op(*m_s64, HloOpcode::kMultiply, S32));

  // Static axis > INT32_MAX emits an S64 constant instead of illegal
  // kGetDimensionSize.
  ASSERT_OK_AND_ASSIGN(auto m_big_axis, ParseAndReturnVerifiedModule(R"(
HloModule big_axis
ENTRY entry {
  x = f32[2147483648,2] parameter(0)
  s = f32[2] parameter(1)
  m = f32[2] parameter(2)
  v = f32[2] parameter(3)
  dy = f32[2147483648,2] parameter(4)
  ROOT bn = (f32[2147483648,2], f32[2], f32[2])
    batch-norm-grad(x, s, m, v, dy), epsilon=0.001, feature_index=1
})"));
  ASSERT_THAT(rewriter.Run(m_big_axis.get()), absl_testing::IsOkAndHolds(true));
  ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                          /*allow_mixed_precision=*/false)
                  .Run(m_big_axis.get()),
              absl_testing::IsOk());
  EXPECT_EQ(CountGetDimensionSize(*m_big_axis), 0);
  EXPECT_TRUE(has_scalar_op(*m_big_axis, HloOpcode::kMultiply, S64));

  // Near-INT64_MAX count (2^32 * INT32_MAX) widens to S64 and F32 without
  // signed overflow, and a zero-sized axis alongside a 2^64 (> INT64_MAX)
  // sibling product short-circuits cleanly to S64 zero.
  ASSERT_OK_AND_ASSIGN(auto m_near_s64, ParseAndReturnVerifiedModule(R"(
HloModule near_s64
ENTRY entry {
  x = f8e4m3fn[65536,65536,2147483647,1] parameter(0)
  s = f8e4m3fn[1] parameter(1)
  o = f8e4m3fn[1] parameter(2)
  ROOT bn = (f8e4m3fn[65536,65536,2147483647,1], f8e4m3fn[1], f8e4m3fn[1])
    batch-norm-training(x, s, o), epsilon=0.5, feature_index=3
})"));
  ASSERT_THAT(rewriter.Run(m_near_s64.get()), absl_testing::IsOkAndHolds(true));
  ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                          /*allow_mixed_precision=*/false)
                  .Run(m_near_s64.get()),
              absl_testing::IsOk());
  EXPECT_TRUE(has_scalar_op(*m_near_s64, HloOpcode::kMultiply, S64));
  EXPECT_TRUE(has_scalar_op(*m_near_s64, HloOpcode::kConvert, F32));

  ASSERT_OK_AND_ASSIGN(auto m_zero_gt_s64, ParseAndReturnVerifiedModule(R"(
HloModule zero_gt_s64
ENTRY entry {
  x = f8e4m3fn[0,65536,65536,65536,65536,2] parameter(0)
  s = f8e4m3fn[2] parameter(1)
  o = f8e4m3fn[2] parameter(2)
  ROOT bn = (f8e4m3fn[0,65536,65536,65536,65536,2], f8e4m3fn[2], f8e4m3fn[2])
    batch-norm-training(x, s, o), epsilon=0.5, feature_index=5
})"));
  ASSERT_THAT(rewriter.Run(m_zero_gt_s64.get()),
              absl_testing::IsOkAndHolds(true));
  ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                          /*allow_mixed_precision=*/false)
                  .Run(m_zero_gt_s64.get()),
              absl_testing::IsOk());
  EXPECT_EQ(CountGetDimensionSize(*m_zero_gt_s64), 0);
  EXPECT_TRUE(has_scalar_op(*m_zero_gt_s64, HloOpcode::kConstant, S64));

  // Zero-sized axis alongside >INT32_MAX siblings short-circuits to 0 without
  // S64 overflow or +Inf * 0 = NaN.
  ASSERT_OK_AND_ASSIGN(auto m_zero, ParseAndReturnVerifiedModule(R"(
HloModule zero_huge
ENTRY entry {
  x = f32[0,2147483648,2,2] parameter(0)
  s = f32[2] parameter(1)
  o = f32[2] parameter(2)
  ROOT bn = (f32[0,2147483648,2,2], f32[2], f32[2])
    batch-norm-training(x, s, o), epsilon=0.001, feature_index=3
})"));
  ASSERT_THAT(rewriter.Run(m_zero.get()), absl_testing::IsOkAndHolds(true));
  ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                          /*allow_mixed_precision=*/false)
                  .Run(m_zero.get()),
              absl_testing::IsOk());
  EXPECT_EQ(CountGetDimensionSize(*m_zero), 0);
  EXPECT_TRUE(has_scalar_op(*m_zero, HloOpcode::kConstant, S64));

  // Dynamic non-feature dimension > INT32_MAX returns InvalidArgument cleanly.
  ASSERT_OK_AND_ASSIGN(auto m_bad_dyn, ParseAndReturnVerifiedModule(R"(
HloModule bad_dyn
ENTRY entry {
  x = f32[<=2147483648,2] parameter(0)
  s = f32[2] parameter(1)
  o = f32[2] parameter(2)
  ROOT bn = (f32[<=2147483648,2], f32[2], f32[2])
    batch-norm-training(x, s, o), epsilon=0.001, feature_index=1
})"));
  EXPECT_THAT(rewriter.Run(m_bad_dyn.get()),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(BatchNormExpanderTest,
       PreventsNarrowFloatElementCountOverflowInTrainingAndGrad) {
  BatchNormExpander rewriter(/*rewrite_training_op=*/true,
                             /*rewrite_inference_op=*/true,
                             /*rewrite_grad_op=*/true);
  auto eval_with_ddi =
      [](HloModule* m,
         absl::Span<const Literal* const> args) -> absl::StatusOr<Literal> {
    ABSL_ASSIGN_OR_RETURN(auto ddi, DynamicDimensionInference::Run(m));
    HloEvaluator evaluator;
    evaluator.set_dynamic_dimension_inference(&ddi);
    return evaluator.Evaluate(*m, args);
  };

  auto make_narrow = [](PrimitiveType ty, absl::Span<const int64_t> dims,
                        float even_val,
                        float odd_val) -> absl::StatusOr<Literal> {
    Literal f32(ShapeUtil::MakeShape(F32, dims));
    auto data = f32.data<float>();
    const int64_t c = dims.back();
    for (size_t idx = 0; idx < data.size(); ++idx) {
      const size_t sample = idx / static_cast<size_t>(c);
      data[idx] = (sample % 2 == 0) ? even_val : odd_val;
    }
    return f32.Convert(ty);
  };

  auto check_training_and_grad = [&](PrimitiveType ty,
                                     absl::Span<const int64_t> dims) {
    const std::string tn = primitive_util::LowercasePrimitiveTypeName(ty);
    std::string dstr;
    for (size_t i = 0; i < dims.size(); ++i) {
      absl::StrAppend(&dstr, i == 0 ? "" : ",", dims[i]);
    }
    const int64_t fi = static_cast<int64_t>(dims.size()) - 1;
    const int64_t c = dims.back();
    ASSERT_OK_AND_ASSIGN(
        auto mt,
        ParseAndReturnVerifiedModule(absl::StrCat(
            "HloModule t_", tn, "\nENTRY e {\n  x = ", tn, "[", dstr,
            "] parameter(0)\n  s = ", tn, "[", c, "] parameter(1)\n  o = ", tn,
            "[", c, "] parameter(2)\n  ROOT bn = (", tn, "[", dstr, "], ", tn,
            "[", c, "], ", tn, "[", c,
            "]) batch-norm-training(x, s, o), epsilon=1, "
            "feature_index=",
            fi, "\n}")));
    ASSERT_THAT(rewriter.Run(mt.get()), absl_testing::IsOkAndHolds(true));
    ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                            /*allow_mixed_precision=*/false)
                    .Run(mt.get()),
                absl_testing::IsOk());
    ASSERT_OK_AND_ASSIGN(Literal x, make_narrow(ty, dims, 0.25f, 0.25f));
    ASSERT_OK_AND_ASSIGN(Literal s, make_narrow(ty, {c}, 1.0f, 1.0f));
    ASSERT_OK_AND_ASSIGN(Literal z, make_narrow(ty, {c}, 0.0f, 0.0f));
    ASSERT_OK_AND_ASSIGN(Literal tout, eval_with_ddi(mt.get(), {&x, &s, &z}));
    auto tparts = tout.DecomposeTuple();
    ASSERT_OK_AND_ASSIGN(Literal mean_f32, tparts[1].Convert(F32));
    ASSERT_OK_AND_ASSIGN(Literal var_f32, tparts[2].Convert(F32));
    EXPECT_FLOAT_EQ(mean_f32.data<float>()[0], 0.25f) << tn;
    EXPECT_FLOAT_EQ(var_f32.data<float>()[0], 0.0f) << tn;

    ASSERT_OK_AND_ASSIGN(
        auto mg,
        ParseAndReturnVerifiedModule(absl::StrCat(
            "HloModule g_", tn, "\nENTRY e {\n  x = ", tn, "[", dstr,
            "] parameter(0)\n  s = ", tn, "[", c, "] parameter(1)\n  m = ", tn,
            "[", c, "] parameter(2)\n  v = ", tn, "[", c,
            "] parameter(3)\n  dy = ", tn, "[", dstr,
            "] parameter(4)\n  ROOT bn = (", tn, "[", dstr, "], ", tn, "[", c,
            "], ", tn, "[", c,
            "]) batch-norm-grad(x, s, m, v, dy), epsilon=1, "
            "feature_index=",
            fi, "\n}")));
    ASSERT_THAT(rewriter.Run(mg.get()), absl_testing::IsOkAndHolds(true));
    ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                            /*allow_mixed_precision=*/false)
                    .Run(mg.get()),
                absl_testing::IsOk());
    ASSERT_OK_AND_ASSIGN(Literal m, make_narrow(ty, {c}, 0.25f, 0.25f));
    ASSERT_OK_AND_ASSIGN(Literal dy, make_narrow(ty, dims, 0.125f, -0.125f));
    ASSERT_OK_AND_ASSIGN(Literal gout,
                         eval_with_ddi(mg.get(), {&x, &s, &m, &z, &dy}));
    auto gparts = gout.DecomposeTuple();
    ASSERT_OK_AND_ASSIGN(Literal gx_f32, gparts[0].Convert(F32));
    EXPECT_FLOAT_EQ(gx_f32.data<float>()[0], 0.125f) << tn;
    EXPECT_FLOAT_EQ(gx_f32.data<float>()[c], -0.125f) << tn;
  };

  check_training_and_grad(F16, {16, 64, 64, 2});   // N = 65536 > 65504
  check_training_and_grad(F8E4M3, {16, 16, 2});    // N = 256 > 240
  check_training_and_grad(F8E4M3FN, {32, 16, 2});  // N = 512 > 448

  // Correlated F16 grad (x = +/-2, dy = +/-0.375, sum_dy_x = 49152) divides 1D
  // reductions before broadcasting so centered * sum_dy_x does not overflow.
  ASSERT_OK_AND_ASSIGN(auto m_corr, ParseAndReturnVerifiedModule(R"(
HloModule f16_corr
ENTRY e {
  x = f16[16,64,64,2] parameter(0)
  s = f16[2] parameter(1)
  m = f16[2] parameter(2)
  v = f16[2] parameter(3)
  dy = f16[16,64,64,2] parameter(4)
  ROOT bn = (f16[16,64,64,2], f16[2], f16[2])
    batch-norm-grad(x, s, m, v, dy), epsilon=0, feature_index=3
})"));
  ASSERT_THAT(rewriter.Run(m_corr.get()), absl_testing::IsOkAndHolds(true));
  ASSERT_OK_AND_ASSIGN(Literal xc,
                       make_narrow(F16, {16, 64, 64, 2}, 2.0f, -2.0f));
  ASSERT_OK_AND_ASSIGN(Literal sc, make_narrow(F16, {2}, 1.0f, 1.0f));
  ASSERT_OK_AND_ASSIGN(Literal mc, make_narrow(F16, {2}, 0.0f, 0.0f));
  ASSERT_OK_AND_ASSIGN(Literal vc, make_narrow(F16, {2}, 4.0f, 4.0f));
  ASSERT_OK_AND_ASSIGN(Literal dyc,
                       make_narrow(F16, {16, 64, 64, 2}, 0.375f, -0.375f));
  ASSERT_OK_AND_ASSIGN(Literal corr_out,
                       eval_with_ddi(m_corr.get(), {&xc, &sc, &mc, &vc, &dyc}));
  ASSERT_OK_AND_ASSIGN(Literal corr_gx,
                       corr_out.DecomposeTuple()[0].Convert(F32));
  EXPECT_FLOAT_EQ(corr_gx.data<float>()[0], 0.0f);
  EXPECT_FLOAT_EQ(corr_gx.data<float>()[2], 0.0f);

  // All 12 narrow float types widen strictly above floor(max finite value),
  // while in-range narrow shapes and BF16 emit zero F32 converts.
  auto has_f32 = [](const HloModule& m) {
    return absl::c_any_of(m.entry_computation()->instructions(),
                          [](const HloInstruction* h) {
                            return h->shape().element_type() == F32;
                          });
  };
  constexpr std::pair<PrimitiveType, int64_t> kThresholds[] = {
      {F8E5M3FNU, 114688}, {F16, 65504},        {F8E5M2, 57344},
      {F8E5M2FNUZ, 57344}, {F8E4M3FN, 448},     {F8E4M3, 240},
      {F8E4M3FNUZ, 240},   {F8E4M3B11FNUZ, 30}, {F6E3M2FN, 28},
      {F8E3M4, 15},        {F6E2M3FN, 7},       {F4E2M1FN, 6}};
  for (auto [ty, limit] : kThresholds) {
    const std::string tn = primitive_util::LowercasePrimitiveTypeName(ty);
    for (int64_t n : {limit, limit + 1}) {
      ASSERT_OK_AND_ASSIGN(
          auto m,
          ParseAndReturnVerifiedModule(absl::StrCat(
              "HloModule lim_", tn, "\nENTRY e {\n  x = ", tn, "[", n,
              ",2] parameter(0)\n  s = ", tn, "[2] parameter(1)\n  m = ", tn,
              "[2] parameter(2)\n  v = ", tn, "[2] parameter(3)\n  dy = ", tn,
              "[", n, ",2] parameter(4)\n  ROOT bn = (", tn, "[", n, ",2], ",
              tn, "[2], ", tn,
              "[2]) batch-norm-grad(x, s, m, v, dy), epsilon=0.5, "
              "feature_index=1\n}")));
      ASSERT_THAT(rewriter.Run(m.get()), absl_testing::IsOkAndHolds(true));
      ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                              /*allow_mixed_precision=*/false)
                      .Run(m.get()),
                  absl_testing::IsOk());
      EXPECT_EQ(has_f32(*m), n > limit) << tn << " n=" << n;
    }
  }

  for (absl::string_view hlo : {absl::string_view(R"(
HloModule f16_in_range
ENTRY e {
  x = f16[16,64,63,2] parameter(0)
  s = f16[2] parameter(1)
  o = f16[2] parameter(2)
  ROOT bn = (f16[16,64,63,2], f16[2], f16[2])
    batch-norm-training(x, s, o), epsilon=0.001, feature_index=3
})"),
                                absl::string_view(R"(
HloModule bf16_unwidened
ENTRY e {
  x = bf16[16,64,64,2] parameter(0)
  s = bf16[2] parameter(1)
  o = bf16[2] parameter(2)
  ROOT bn = (bf16[16,64,64,2], bf16[2], bf16[2])
    batch-norm-training(x, s, o), epsilon=0.001, feature_index=3
})")}) {
    ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
    ASSERT_THAT(rewriter.Run(m.get()), absl_testing::IsOkAndHolds(true));
    EXPECT_FALSE(has_f32(*m));
  }
}

TEST_F(BatchNormExpanderTest, RejectsF8E8M0FnuWithUnimplementedWhenEnabled) {
  BatchNormExpander disabled(/*rewrite_training_op=*/false,
                             /*rewrite_inference_op=*/false,
                             /*rewrite_grad_op=*/false);
  BatchNormExpander enabled(/*rewrite_training_op=*/true,
                            /*rewrite_inference_op=*/true,
                            /*rewrite_grad_op=*/true);
  for (absl::string_view hlo : {absl::string_view(R"(
HloModule e8m0_train
ENTRY e {
  x = f8e8m0fnu[4,2] parameter(0)
  s = f8e8m0fnu[2] parameter(1)
  o = f8e8m0fnu[2] parameter(2)
  ROOT bn = (f8e8m0fnu[4,2], f8e8m0fnu[2], f8e8m0fnu[2])
    batch-norm-training(x, s, o), epsilon=1, feature_index=1
})"),
                                absl::string_view(R"(
HloModule e8m0_infer
ENTRY e {
  x = f8e8m0fnu[4,2] parameter(0)
  s = f8e8m0fnu[2] parameter(1)
  o = f8e8m0fnu[2] parameter(2)
  m = f8e8m0fnu[2] parameter(3)
  v = f8e8m0fnu[2] parameter(4)
  ROOT bn = f8e8m0fnu[4,2]
    batch-norm-inference(x, s, o, m, v), epsilon=1, feature_index=1
})"),
                                absl::string_view(R"(
HloModule e8m0_grad
ENTRY e {
  x = f8e8m0fnu[4,2] parameter(0)
  s = f8e8m0fnu[2] parameter(1)
  m = f8e8m0fnu[2] parameter(2)
  v = f8e8m0fnu[2] parameter(3)
  dy = f8e8m0fnu[4,2] parameter(4)
  ROOT bn = (f8e8m0fnu[4,2], f8e8m0fnu[2], f8e8m0fnu[2])
    batch-norm-grad(x, s, m, v, dy), epsilon=1, feature_index=1
})")}) {
    ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
    const int64_t before = m->entry_computation()->instruction_count();
    EXPECT_THAT(disabled.Run(m.get()), absl_testing::IsOkAndHolds(false));
    EXPECT_THAT(enabled.Run(m.get()),
                absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
    EXPECT_EQ(m->entry_computation()->instruction_count(), before);
  }
}

TEST_F(BatchNormExpanderTest, ExcludesPaddingInAsymmetricDynamicBatchNormGrad) {
  BatchNormExpander rewriter(/*rewrite_training_op=*/true,
                             /*rewrite_inference_op=*/true,
                             /*rewrite_grad_op=*/true);
  for (bool dyn_act : {false, true}) {
    const absl::string_view setup =
        dyn_act ? "  x = f32[<=4,1] set-dimension-size(x0, n), dimensions={0}\n"
                  "  dy = f32[4,1] copy(dy0)\n"
                : "  x = f32[4,1] copy(x0)\n"
                  "  dy = f32[<=4,1] set-dimension-size(dy0, n), "
                  "dimensions={0}\n";
    ASSERT_OK_AND_ASSIGN(
        auto m,
        ParseAndReturnVerifiedModule(absl::StrCat(
            "HloModule asym\nENTRY e {\n  x0 = f32[4,1] parameter(0)\n  s = "
            "f32[1] parameter(1)\n  mean = f32[1] parameter(2)\n  var = f32[1] "
            "parameter(3)\n  dy0 = f32[4,1] parameter(4)\n  n = s32[] "
            "parameter(5)\n",
            setup,
            "  ROOT bn = (f32[<=4,1], f32[1], f32[1]) batch-norm-grad(x, s, "
            "mean, var, dy), epsilon=0, feature_index=1\n}")));
    ASSERT_THAT(rewriter.Run(m.get()), absl_testing::IsOkAndHolds(true));
    ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                            /*allow_mixed_precision=*/false)
                    .Run(m.get()),
                absl_testing::IsOk());
    DynamicPadderOptions opts;
    opts.slice_dynamic_output = false;
    ASSERT_THAT(DynamicPadder(opts).Run(m.get()), absl_testing::IsOk());
    ASSERT_OK_AND_ASSIGN(auto ddi, DynamicDimensionInference::Run(m.get()));

    Literal x = LiteralUtil::CreateR2<float>({{1.0f}, {3.0f}, {5.0f}, {5.0f}});
    Literal s = LiteralUtil::CreateR1<float>({1.0f});
    Literal mean = LiteralUtil::CreateR1<float>({2.0f});
    Literal var = LiteralUtil::CreateR1<float>({1.0f});
    Literal dy =
        LiteralUtil::CreateR2<float>({{2.0f}, {6.0f}, {100.0f}, {100.0f}});
    Literal n = LiteralUtil::CreateR0<int32_t>(2);
    HloEvaluator evaluator;
    evaluator.set_dynamic_dimension_inference(&ddi);
    ASSERT_OK_AND_ASSIGN(
        Literal out, evaluator.Evaluate(*m, {&x, &s, &mean, &var, &dy, &n}));
    auto parts = out.DecomposeTuple();
    EXPECT_FLOAT_EQ(parts[2].data<float>()[0], 8.0f) << dyn_act;
    EXPECT_FLOAT_EQ(parts[1].data<float>()[0], 4.0f) << dyn_act;
    EXPECT_FLOAT_EQ(parts[0].data<float>()[0], 0.0f) << dyn_act;
    EXPECT_FLOAT_EQ(parts[0].data<float>()[1], 0.0f) << dyn_act;
  }
}

TEST_F(BatchNormExpanderTest,
       HandlesMixedDynamicAndStaticOperandsAcrossAllModes) {
  BatchNormExpander rewriter(/*rewrite_training_op=*/true,
                             /*rewrite_inference_op=*/true,
                             /*rewrite_grad_op=*/true);
  for (absl::string_view hlo : {absl::string_view(R"(
HloModule dyn_train_feat
ENTRY e {
  x = f32[4,<=8] parameter(0)
  s = f32[8] parameter(1)
  o = f32[8] parameter(2)
  ROOT bn = (f32[4,<=8], f32[<=8], f32[<=8])
    batch-norm-training(x, s, o), epsilon=0.001, feature_index=1
})"),
                                absl::string_view(R"(
HloModule dyn_train_scale
ENTRY e {
  x = f32[4,8] parameter(0)
  s = f32[<=8] parameter(1)
  o = f32[8] parameter(2)
  ROOT bn = (f32[4,8], f32[8], f32[8])
    batch-norm-training(x, s, o), epsilon=0.001, feature_index=1
})"),
                                absl::string_view(R"(
HloModule dyn_infer
ENTRY e {
  x = f32[4,8] parameter(0)
  s = f32[<=8] parameter(1)
  o = f32[8] parameter(2)
  m = f32[8] parameter(3)
  v = f32[<=8] parameter(4)
  ROOT bn = f32[4,8]
    batch-norm-inference(x, s, o, m, v), epsilon=0.001, feature_index=1
})"),
                                absl::string_view(R"(
HloModule dyn_grad_a
ENTRY e {
  x = f32[4,<=8] parameter(0)
  s = f32[8] parameter(1)
  m = f32[8] parameter(2)
  v = f32[8] parameter(3)
  dy = f32[<=4,8] parameter(4)
  ROOT bn = (f32[<=4,<=8], f32[<=8], f32[<=8])
    batch-norm-grad(x, s, m, v, dy), epsilon=0.001, feature_index=1
})"),
                                absl::string_view(R"(
HloModule dyn_grad_b
ENTRY e {
  x = f32[4,8] parameter(0)
  s = f32[<=8] parameter(1)
  m = f32[<=8] parameter(2)
  v = f32[8] parameter(3)
  dy = f32[4,8] parameter(4)
  ROOT bn = (f32[4,8], f32[8], f32[8])
    batch-norm-grad(x, s, m, v, dy), epsilon=0.001, feature_index=1
})")}) {
    ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
    ASSERT_THAT(rewriter.Run(m.get()), absl_testing::IsOkAndHolds(true));
    ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                            /*allow_mixed_precision=*/false)
                    .Run(m.get()),
                absl_testing::IsOk());
    ASSERT_THAT(DynamicDimensionInference::Run(m.get()), absl_testing::IsOk());
  }
}

TEST_F(BatchNormExpanderTest,
       PreservesLayoutsSubByteHostSpaceMixedPrecisionThreadAndRank1) {
  BatchNormExpander rewriter(/*rewrite_training_op=*/true,
                             /*rewrite_inference_op=*/true,
                             /*rewrite_grad_op=*/true);

  // Mismatched grad_output layout {0,1} vs activation {1,0}, plus non-default
  // host-memory-space output requiring a two-step transpose + space copy.
  ASSERT_OK_AND_ASSIGN(auto m_layout, ParseAndReturnUnverifiedModule(R"(
HloModule layout_and_host
ENTRY e {
  x = f32[8,4]{1,0} parameter(0)
  s = f32[4]{0} parameter(1)
  m = f32[4]{0} parameter(2)
  v = f32[4]{0} parameter(3)
  dy = f32[8,4]{0,1} parameter(4)
  ROOT bn = (f32[8,4]{0,1:S(5)}, f32[4]{0}, f32[4]{0})
    batch-norm-grad(x, s, m, v, dy), epsilon=0.001, feature_index=1
})"));
  ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                          /*allow_mixed_precision=*/false)
                  .Run(m_layout.get()),
              absl_testing::IsOk());
  ASSERT_THAT(rewriter.Run(m_layout.get()), absl_testing::IsOkAndHolds(true));
  ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                          /*allow_mixed_precision=*/false)
                  .Run(m_layout.get()),
              absl_testing::IsOk());
  EXPECT_EQ(m_layout->entry_computation()
                ->root_instruction()
                ->operand(0)
                ->shape()
                .layout()
                .memory_space(),
            Layout::kHostMemorySpace);

  // Packed sub-byte F4E2M1FN with N = 8 > max_finite(6) and non-main thread.
  ASSERT_OK_AND_ASSIGN(auto m_f4, ParseAndReturnUnverifiedModule(R"(
HloModule subbyte_and_thread
async_comp {
  x = f4e2m1fn[4,2,2]{2,1,0:E(4)} parameter(0)
  s = f4e2m1fn[2]{0} parameter(1)
  o = f4e2m1fn[2]{0:E(4)} parameter(2)
  ROOT bn = (f4e2m1fn[4,2,2]{2,1,0:E(4)}, f4e2m1fn[2]{0:E(4)},
             f4e2m1fn[2]{0:E(4)})
    batch-norm-training(x, s, o), epsilon=0.5, feature_index=2
}
ENTRY e {
  x = f4e2m1fn[4,2,2]{2,1,0:E(4)} parameter(0)
  s = f4e2m1fn[2]{0} parameter(1)
  o = f4e2m1fn[2]{0:E(4)} parameter(2)
  st = ((f4e2m1fn[4,2,2]{2,1,0:E(4)}, f4e2m1fn[2]{0}, f4e2m1fn[2]{0:E(4)}),
        (f4e2m1fn[4,2,2]{2,1,0:E(4)}, f4e2m1fn[2]{0:E(4)}, f4e2m1fn[2]{0:E(4)}),
        s32[]) async-start(x, s, o), calls=async_comp,
    async_execution_thread="worker"
  ROOT dn = (f4e2m1fn[4,2,2]{2,1,0:E(4)}, f4e2m1fn[2]{0:E(4)},
             f4e2m1fn[2]{0:E(4)}) async-done(st)
})"));
  ASSERT_THAT(rewriter.Run(m_f4.get(), {"main", "worker"}),
              absl_testing::IsOkAndHolds(true));
  ASSERT_THAT(HloVerifier(HloVerifierOpts{}
                              .MakeLayoutSensitive()
                              .VerifyCallNestedComputationThreadName())
                  .Run(m_f4.get()),
              absl_testing::IsOk());

  // Mixed-precision BF16/F32 operands and outputs, plus rank-1 f32[4].
  ASSERT_OK_AND_ASSIGN(auto m_mp, ParseAndReturnUnverifiedModule(R"(
HloModule mixed_precision
ENTRY e {
  x = bf16[8,4]{1,0} parameter(0)
  s = f32[4]{0} parameter(1)
  o = f32[4]{0} parameter(2)
  m = f32[4]{0} parameter(3)
  v = f32[4]{0} parameter(4)
  ROOT bn = f32[8,4]{1,0}
    batch-norm-inference(x, s, o, m, v), epsilon=0.001, feature_index=1
})"));
  ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                          /*allow_mixed_precision=*/true)
                  .Run(m_mp.get()),
              absl_testing::IsOk());
  ASSERT_THAT(rewriter.Run(m_mp.get()), absl_testing::IsOkAndHolds(true));
  ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                          /*allow_mixed_precision=*/false)
                  .Run(m_mp.get()),
              absl_testing::IsOk());

  ASSERT_OK_AND_ASSIGN(auto m_r1, ParseAndReturnVerifiedModule(R"(
HloModule rank1
ENTRY e {
  x = f32[4] parameter(0)
  s = f32[4] parameter(1)
  o = f32[4] parameter(2)
  ROOT bn = (f32[4], f32[4], f32[4])
    batch-norm-training(x, s, o), epsilon=1, feature_index=0
})"));
  ASSERT_THAT(rewriter.Run(m_r1.get()), absl_testing::IsOkAndHolds(true));
  ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                          /*allow_mixed_precision=*/false)
                  .Run(m_r1.get()),
              absl_testing::IsOk());
}

TEST_F(BatchNormExpanderTest,
       PropagatesShardingsAndRelaysControlDependenciesAcrossAllModes) {
  BatchNormExpander rewriter(/*rewrite_training_op=*/true,
                             /*rewrite_inference_op=*/true,
                             /*rewrite_grad_op=*/true);

  // Dynamic operand keeps tiled sharding on static broadcasts, and matching
  // {1}=={2} propagates to 1D feature ops.
  ASSERT_OK_AND_ASSIGN(auto m_sh, ParseAndReturnVerifiedModule(R"(
HloModule feat_and_dyn_sharding
ENTRY e {
  x = f32[<=8,4] parameter(0)
  s = f32[4] parameter(1)
  o = f32[4] parameter(2)
  ROOT bn = (f32[<=8,4], f32[4], f32[4])
    batch-norm-training(x, s, o), epsilon=0.001, feature_index=1,
    sharding={{devices=[1,2]0,1},{devices=[2]0,1},{devices=[2]0,1}}
})"));
  ASSERT_THAT(rewriter.Run(m_sh.get()), absl_testing::IsOkAndHolds(true));
  ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                          /*allow_mixed_precision=*/false)
                  .Run(m_sh.get()),
              absl_testing::IsOk());
  for (const HloInstruction* h : m_sh->entry_computation()->instructions()) {
    if (h->opcode() == HloOpcode::kParameter || !h->shape().IsArray()) {
      continue;
    }
    if (h->shape().dimensions().size() == 2) {
      EXPECT_FALSE(h->sharding().IsReplicated()) << h->ToString();
    } else if (h->shape().dimensions().size() == 1) {
      EXPECT_FALSE(h->sharding().IsReplicated()) << h->ToString();
    }
  }

  // Heterogeneous maximal leaves, rank-1 tuple sharding, and Manual / Unknown /
  // Unreduced shardings.
  for (absl::string_view hlo : {absl::string_view(R"(
HloModule hetero_leaves
ENTRY e {
  x = f32[8,4] parameter(0)
  s = f32[4] parameter(1)
  o = f32[4] parameter(2)
  ROOT bn = (f32[8,4], f32[4], f32[4])
    batch-norm-training(x, s, o), epsilon=0.001, feature_index=1,
    sharding={{maximal device=0},{maximal device=1},{maximal device=2}}
})"),
                                absl::string_view(R"(
HloModule rank1_sharding
ENTRY e {
  x = f32[4] parameter(0)
  s = f32[4] parameter(1)
  o = f32[4] parameter(2)
  ROOT bn = (f32[4], f32[4], f32[4])
    batch-norm-training(x, s, o), epsilon=0.001, feature_index=0,
    sharding={{devices=[2]0,1},{replicated},{replicated}}
})"),
                                absl::string_view(R"(
HloModule manual_sharding
ENTRY e {
  x = f32[8,4] parameter(0)
  s = f32[4] parameter(1)
  m = f32[4] parameter(2)
  v = f32[4] parameter(3)
  dy = f32[8,4] parameter(4)
  ROOT bn = (f32[8,4], f32[4], f32[4])
    batch-norm-grad(x, s, m, v, dy), epsilon=0.001, feature_index=1,
    sharding={{manual},{manual},{manual}}
})"),
                                absl::string_view(R"(
HloModule unknown_sharding
ENTRY e {
  x = f32[8,4] parameter(0)
  s = f32[4] parameter(1)
  o = f32[4] parameter(2)
  m = f32[4] parameter(3)
  v = f32[4] parameter(4)
  ROOT bn = f32[8,4] batch-norm-inference(x, s, o, m, v),
    epsilon=0.001, feature_index=1, sharding={unknown}
})"),
                                absl::string_view(R"(
HloModule unreduced_sharding
ENTRY e {
  x = f32[8,4] parameter(0)
  s = f32[4] parameter(1)
  o = f32[4] parameter(2)
  m = f32[4] parameter(3)
  v = f32[4] parameter(4)
  ROOT bn = f32[8,4] batch-norm-inference(x, s, o, m, v),
    epsilon=0.001, feature_index=1, sharding={unreduced}
})")}) {
    ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo));
    ASSERT_THAT(rewriter.Run(m.get()), absl_testing::IsOkAndHolds(true));
    ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                            /*allow_mixed_precision=*/false)
                    .Run(m.get()),
                absl_testing::IsOk());
    if (m->name() == "manual_sharding" || m->name() == "unknown_sharding" ||
        m->name() == "unreduced_sharding") {
      for (const HloInstruction* h : m->entry_computation()->instructions()) {
        if (h->opcode() != HloOpcode::kParameter && h->shape().IsArray()) {
          EXPECT_FALSE(h->sharding().IsReplicated()) << h->ToString();
        }
      }
    }
  }

  // Control predecessors and successors on Training, Inference, and Grad are
  // relayed cleanly without aborting or leaving unexpanded BatchNorm ops.
  ASSERT_OK_AND_ASSIGN(auto m_ctrl, ParseAndReturnVerifiedModule(R"(
HloModule control_deps
ENTRY e {
  x = f32[4,2] parameter(0)
  s = f32[2] parameter(1)
  o = f32[2] parameter(2)
  dy = f32[4,2] parameter(3)
  tok = token[] after-all()
  tr = (f32[4,2], f32[2], f32[2]) batch-norm-training(x, s, o),
    epsilon=0.001, feature_index=1, control-predecessors={tok}
  m = f32[2] get-tuple-element(tr), index=1
  v = f32[2] get-tuple-element(tr), index=2
  bn_inf = f32[4,2] batch-norm-inference(x, s, o, m, v),
    epsilon=0.001, feature_index=1, control-predecessors={tr}
  gr = (f32[4,2], f32[2], f32[2]) batch-norm-grad(x, s, m, v, dy),
    epsilon=0.001, feature_index=1, control-predecessors={bn_inf}
  succ = token[] after-all(), control-predecessors={tr, bn_inf, gr}
  gx = f32[4,2] get-tuple-element(gr), index=0
  ROOT res = (f32[4,2], f32[4,2], token[]) tuple(bn_inf, gx, succ)
})"));
  ASSERT_THAT(rewriter.Run(m_ctrl.get()), absl_testing::IsOkAndHolds(true));
  ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                          /*allow_mixed_precision=*/false)
                  .Run(m_ctrl.get()),
              absl_testing::IsOk());
  for (const HloInstruction* h : m_ctrl->entry_computation()->instructions()) {
    EXPECT_NE(h->opcode(), HloOpcode::kBatchNormTraining);
    EXPECT_NE(h->opcode(), HloOpcode::kBatchNormInference);
    EXPECT_NE(h->opcode(), HloOpcode::kBatchNormGrad);
  }
  const HloInstruction* succ = FindInstruction(m_ctrl.get(), "succ");
  ASSERT_NE(succ, nullptr);
  EXPECT_EQ(succ->control_predecessors().size(), 3);
}

}  // namespace
}  // namespace xla
