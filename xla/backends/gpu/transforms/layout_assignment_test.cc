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

#include "xla/backends/gpu/transforms/layout_assignment.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status_matchers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/computation_layout.h"
#include "xla/service/gpu/conv_layout_normalization.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_assignment.h"
#include "xla/service/layout_normalization.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;
using ::testing::NotNull;

class LayoutAssignmentTest : public HloHardwareIndependentTestBase {
 protected:
  const se::CudaComputeCapability default_cuda_cc_ =
      se::CudaComputeCapability();
  const se::GpuComputeCapability default_gpu_cc_ = se::GpuComputeCapability();
  const se::DeviceDescription default_device_description_ =
      se::DeviceDescription();

  // Runs GpuLayoutAssignment on `module` with the layouts of the entry
  // computation.
  void AssignLayouts(HloModule* module) {
    ComputationLayout computation_layout(
        module->entry_computation()->ComputeProgramShape(),
        /*ignore_layouts=*/false);
    GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                          default_device_description_);
    EXPECT_THAT(layout_assignment.Run(module),
                absl_testing::IsOkAndHolds(true));
  }
};

TEST_F(LayoutAssignmentTest, Elementwise) {
  Shape ashape = ShapeUtil::MakeShape(F32, {42, 12});
  Shape ashape_in_row_major(ashape);
  Shape ashape_in_col_major(ashape);
  *ashape_in_row_major.mutable_layout() = LayoutUtil::MakeLayout({1, 0});
  *ashape_in_col_major.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  // Enumerate all possible combinations of layouts.
  for (const Shape& lhs_shape_with_layout :
       {ashape_in_row_major, ashape_in_col_major}) {
    for (const Shape& rhs_shape_with_layout :
         {ashape_in_row_major, ashape_in_col_major}) {
      for (const Shape& result_shape_with_layout :
           {ashape_in_row_major, ashape_in_col_major}) {
        // GpuLayoutAssignment should assign the same layout to "add" and its
        // two operands.
        auto builder = HloComputation::Builder(TestName());
        auto x = builder.AddInstruction(
            HloInstruction::CreateParameter(0, ashape, "x"));
        auto y = builder.AddInstruction(
            HloInstruction::CreateParameter(1, ashape, "y"));
        auto add = builder.AddInstruction(
            HloInstruction::CreateBinary(ashape, HloOpcode::kAdd, x, y));
        auto module = CreateNewVerifiedModule();
        HloComputation* computation =
            module->AddEntryComputation(builder.Build(add));

        ComputationLayout computation_layout(
            computation->ComputeProgramShape());
        *computation_layout.mutable_parameter_layout(0) =
            ShapeLayout(lhs_shape_with_layout);
        *computation_layout.mutable_parameter_layout(1) =
            ShapeLayout(rhs_shape_with_layout);
        *computation_layout.mutable_result_layout() =
            ShapeLayout(result_shape_with_layout);

        GpuLayoutAssignment layout_assignment(
            &computation_layout, default_gpu_cc_, default_device_description_);
        EXPECT_THAT(layout_assignment.Run(module.get()),
                    absl_testing::IsOkAndHolds(true));

        for (const HloInstruction* operand : add->operands()) {
          EXPECT_TRUE(LayoutUtil::Equal(add->shape().layout(),
                                        operand->shape().layout()));
        }
      }
    }
  }
}

TEST_F(LayoutAssignmentTest, DotLayoutUnchangedIfValid) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = f32[5,2,3]{1,2,0} parameter(0)
    p1 = f32[5,3,4]{1,2,0} parameter(1)
    ROOT dot.1330.10585 = f32[5,2,4]{2,1,0} dot(p0, p1),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={1}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);
  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Op().WithShape(F32, {5, 2, 3}, {1, 2, 0}),
                                m::Op().WithShape(F32, {5, 3, 4}, {1, 2, 0}))
                             .WithShape(F32, {5, 2, 4}, {2, 1, 0})));
}

TEST_F(LayoutAssignmentTest, DotLayoutSetToDefaultIfDefaultValid) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = f32[5,3,2] parameter(0)
    p1 = f32[5,4,3]{0,1,2} parameter(1)
    ROOT dot.1330.10585 = f32[5,2,4] dot(p0, p1),
      lhs_batch_dims={0}, lhs_contracting_dims={1},
      rhs_batch_dims={0}, rhs_contracting_dims={2}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Op().WithShape(F32, {5, 3, 2}, {2, 1, 0}),
                                m::Op().WithShape(F32, {5, 4, 3}, {2, 1, 0}))
                             .WithShape(F32, {5, 2, 4}, {2, 1, 0})));
}

TEST_F(LayoutAssignmentTest, BitcastConvertKeepsCompatibleLayouts) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
e {
  a = u4[5,4,3,2]{3,2,1,0} parameter(0)
  b = u8[5,4,3]{2,1,0} bitcast-convert(a)
  c = u8[4,5,3]{2,1,0} transpose(b), dimensions={1,0,2}
})"));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape());
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);
  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Copy(m::Transpose(
                  m::BitcastConvert(
                      m::Parameter().WithShape(U4, {5, 4, 3, 2}, {3, 2, 1, 0}))
                      .WithShape(U8, {5, 4, 3}, {2, 1, 0})))));
}

TEST_F(LayoutAssignmentTest, DotOperandLayoutSetToBatchRowsColsOtherwise) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = f32[2,3,5]{2,1,0} parameter(0)
    p1 = f32[3,4,5] parameter(1)
    ROOT dot.1330.10585 = f32[5,2,4] dot(p0, p1),
      lhs_batch_dims={2}, lhs_contracting_dims={1},
      rhs_batch_dims={2}, rhs_contracting_dims={0}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Op().WithShape(F32, {2, 3, 5}, {1, 0, 2}),
                                m::Op().WithShape(F32, {3, 4, 5}, {1, 0, 2}))));
}

TEST_F(LayoutAssignmentTest, DotOperandInconsistentDimLayouts) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = f32[5,6,2,3] parameter(0)
    p1 = f32[6,5,3,4] parameter(1)
    ROOT dot.1330.10585 = f32[5,6,2,4] dot(p0, p1),
      lhs_batch_dims={0,1}, lhs_contracting_dims={3},
      rhs_batch_dims={1,0}, rhs_contracting_dims={2}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Dot(m::Op().WithShape(F32, {5, 6, 2, 3}, {3, 2, 1, 0}),
                        m::Op().WithShape(F32, {6, 5, 3, 4}, {3, 2, 0, 1}))));
}

TEST_F(LayoutAssignmentTest, TransposedDotLayout) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = f32[5,2,3] parameter(0)
    p1 = f32[5,3,4,6] parameter(1)
    dot = f32[5,2,4,6] dot(p0, p1),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={1}
    ROOT out = f32[2,5,4,6] transpose(dot), dimensions={1,0,2,3}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Transpose(
                     m::Dot(m::Op().WithShape(F32, {5, 2, 3}, {2, 1, 0}),
                            m::Op().WithShape(F32, {5, 3, 4, 6}, {3, 2, 1, 0}))
                         .WithShape(F32, {5, 2, 4, 6}, {3, 2, 0, 1}))
                     .WithShape(F32, {2, 5, 4, 6}, {3, 2, 1, 0})));
}

TEST_F(LayoutAssignmentTest, TransposedDotOfDotLayout) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = f32[8,50] parameter(0)
    p1 = f32[2,8,4,4] parameter(1)
    p2 = f32[4,38] parameter(2)
    dot.1 = f32[50,2,4,4]{3,2,1,0} dot(p0, p1),
      lhs_contracting_dims={0}, rhs_contracting_dims={1}
    dot.2 = f32[50,2,4,38]{3,2,1,0} dot(dot.1, p2),
      lhs_contracting_dims={2}, rhs_contracting_dims={0}
    ROOT out = f32[2,50,38,4]{2,3,0,1} transpose(dot.2), dimensions={1,0,3,2}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  // The transpose layout is not supported by dot.2. Also, we need a copy
  // between dot.1 and dot.2, because the needed operand layout for the lhs of
  // dot.1 cannot be used as layout for dot.1
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Transpose(
              m::Dot(m::Copy(m::Dot(m::Op().WithShape(F32, {8, 50}, {1, 0}),
                                    m::Op().WithShape(F32, {2, 8, 4, 4},
                                                      {3, 2, 0, 1}))
                                 .WithShape(F32, {50, 2, 4, 4}, {3, 2, 1, 0}))
                         .WithShape(F32, {50, 2, 4, 4}, {3, 1, 0, 2}),
                     m::Op().WithShape(F32, {4, 38}, {1, 0}))
                  .WithShape(F32, {50, 2, 4, 38}, {3, 2, 1, 0}))
              .WithShape(F32, {2, 50, 38, 4}, {2, 3, 0, 1})));
}

TEST_F(LayoutAssignmentTest, DotLayoutS8) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY int8_t {
    p0 = s8[32,64] parameter(0)
    p1 = s8[64,96] parameter(1)
    ROOT out = s32[32,96] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Op().WithShape(S8, {32, 64}, {1, 0}),
                                m::Op().WithShape(S8, {64, 96}, {0, 1}))));
}

TEST_F(LayoutAssignmentTest, SameLayoutOnOperandsAndOutputsOfSort) {
  const char* hlo_text = R"(
  HloModule SortLayout

  compare {
    p.0.lhs = f32[] parameter(0)
    p.0.rhs = f32[] parameter(1)
    p.1.lhs = f32[] parameter(2)
    p.1.rhs = f32[] parameter(3)
    ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
  }

  ENTRY sort {
    keys = f32[3,2]{0,1} constant({{0,1},{0,1},{0,1}})
    values = f32[2,3]{1,0} parameter(0)
    transpose = f32[3,2]{1,0} transpose(values), dimensions={1,0}
    ROOT sort = (f32[3,2]{1,0}, f32[3,2]{1,0}) sort(keys, transpose),
      dimensions={1}, to_apply=compare
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Sort(m::Op().WithShape(F32, {3, 2}, {1, 0}),
                                 m::Op().WithShape(F32, {3, 2}, {1, 0}))));
}

TEST_F(LayoutAssignmentTest,
       SameLayoutOnOperandsAndOutputsOfCubDeviceRadixSort) {
  const char* hlo_text = R"(
  HloModule SortLayout

  ENTRY sort {
    keys = f32[3,2]{0,1} constant({{0,1},{0,1},{0,1}})
    values = f32[2,3]{1,0} parameter(0)
    transpose = f32[3,2]{1,0} transpose(values), dimensions={1,0}
    ROOT sort = (f32[3,2]{1,0}, f32[3,2]{1,0}, u8[128]{0})
        custom-call(keys, transpose), custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size"
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::CustomCall(m::Op().WithShape(F32, {3, 2}, {1, 0}),
                                       m::Op().WithShape(F32, {3, 2}, {1, 0}))))
      << module->ToString();
}

TEST_F(LayoutAssignmentTest, TopKLayout) {
  const char* hlo_text = R"(
  HloModule topk

  compare-greater-than {
    p.1.lhs.3 = s32[] parameter(2)
    p.1.rhs.4 = s32[] parameter(3)
    p.0.lhs.1 = f32[] parameter(0)
    bitcast-convert = s32[] bitcast-convert(p.0.lhs.1)
    constant = s32[] constant(0)
    compare = pred[] compare(bitcast-convert, constant), direction=LT
    constant.2 = s32[] constant(2147483647)
    xor = s32[] xor(constant.2, bitcast-convert)
    select = s32[] select(compare, xor, bitcast-convert)
    p.0.rhs.2 = f32[] parameter(1)
    bitcast-convert.1 = s32[] bitcast-convert(p.0.rhs.2)
    compare.1 = pred[] compare(bitcast-convert.1, constant), direction=LT
    xor.1 = s32[] xor(constant.2, bitcast-convert.1)
    select.1 = s32[] select(compare.1, xor.1, bitcast-convert.1)
    ROOT compare.5 = pred[] compare(select, select.1), direction=GT
  }

  ENTRY main {
    Arg_0.1 = f32[2048,6]{1,0} parameter(0)
    t = f32[6,2048]{0,1} transpose(Arg_0.1), dimensions={1,0}
    ROOT custom-call.1 = (f32[6,8]{1,0}, s32[6,8]{1,0}) custom-call(t), custom_call_target="__gpu$TopK", api_version=API_VERSION_TYPED_FFI, called_computations={compare-greater-than}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::CustomCall(
                  m::Transpose(m::Copy().WithShape(F32, {2048, 6}, {0, 1}))
                      .WithShape(F32, {6, 2048}, {1, 0}))));
}

TEST_F(LayoutAssignmentTest, TopKAfterAllGatherHasDefaultLayout) {
  const char* hlo_text = R"(
  HloModule topk

  compare-greater-than {
    p.1.lhs.3 = s32[] parameter(2)
    p.1.rhs.4 = s32[] parameter(3)
    p.0.lhs.1 = f32[] parameter(0)
    p.0.rhs.2 = f32[] parameter(1)
    ROOT compare.5 = pred[] compare(p.0.lhs.1, p.0.rhs.2), direction=GT
  }

  ENTRY main {
    p0 = f32[32,64]{1,0} parameter(0)
    ag = f32[32,512]{1,0} all-gather(p0), dimensions={1}, replica_groups={{0,1,2,3,4,5,6,7}}
    neg = f32[32,512]{1,0} negate(ag)
    topk = (f32[32,64]{1,0}, s32[32,64]{1,0}) custom-call(neg), custom_call_target="__gpu$TopK", api_version=API_VERSION_TYPED_FFI, called_computations={compare-greater-than}
    gte0 = f32[32,64]{1,0} get-tuple-element(topk), index=0
    gte1 = s32[32,64]{1,0} get-tuple-element(topk), index=1
    ROOT out = (f32[32,64]{1,0}, s32[32,64]{1,0}) tuple(gte0, gte1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  const HloInstruction* topk = FindInstruction(module.get(), "topk");
  ASSERT_NE(topk, nullptr);
  EXPECT_EQ(topk->operand(0)->shape().layout(), LayoutUtil::MakeLayout({1, 0}));
  EXPECT_EQ(ShapeUtil::GetSubshape(topk->shape(), {0}).layout(),
            LayoutUtil::MakeLayout({1, 0}));
  EXPECT_EQ(ShapeUtil::GetSubshape(topk->shape(), {1}).layout(),
            LayoutUtil::MakeLayout({1, 0}));
}

TEST_F(LayoutAssignmentTest,
       BitcastConvertFromNarrowerTypeGetsOptimalInputLayout) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
e {
  a = s4[3,5,2]{0,1,2:E(4)} parameter(0)
  b = s8[3,5]{0,1} bitcast-convert(a)
})"));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);
  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::BitcastConvert(
          m::Copy(m::Parameter()).WithShape(S4, {3, 5, 2}, {2, 0, 1}))));
}

TEST_F(LayoutAssignmentTest,
       BitcastConvertToNarrowerTypeGetsOptimalOutputLayout) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
e {
  a = s8[3,5] parameter(0)
  b = s4[3,5,2]{0,1,2} bitcast-convert(a)
})"));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);
  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Copy(
                  m::BitcastConvert(
                      m::Copy(m::Parameter()).WithShape(S8, {3, 5}, {0, 1}))
                      .WithShape(S4, {3, 5, 2}, {2, 0, 1}))));
}

TEST_F(LayoutAssignmentTest, FftLayout) {
  const char* hlo_text = R"(
  HloModule Fft_module

  ENTRY Fft {
    input = c64[8,32]{0,1} parameter(0)
    fft = c64[8,32] fft(input), fft_type=FFT, fft_length={32}
    ROOT transpose = c64[32,8] transpose(fft), dimensions={1,0}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Copy(
                  m::Transpose(m::Fft(m::Op().WithShape(C64, {8, 32}, {1, 0}))
                                   .WithShape(C64, {8, 32}, {1, 0})))));
}

TEST_F(LayoutAssignmentTest, CustomCallConstrainedAlias) {
  const char* module_str = R"(
HloModule TestModule

ENTRY entry {
  Arg_0 = f32[2,5,5]{2,1,0} parameter(0)
  Arg_1 = f32[2,5,5]{2,1,0} parameter(1)
  Arg_2 = f32[2,5,5]{2,1,0} parameter(2)
  dot.0 = f32[2,5,5]{2,1,0} dot(Arg_1, Arg_2), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={2}, operand_precision={highest,highest}
  custom-call.0 = (f32[2,5,5]{1,2,0}, s8[16]{0}, s8[16]{0}) custom-call(Arg_0, dot.0), custom_call_target="dummy_call", operand_layout_constraints={f32[2,5,5]{1,2,0}, f32[2,5,5]{1,2,0}}, output_to_operand_aliasing={{0}: (1, {})}
  ROOT get-tuple-element.0 = f32[2,5,5]{1,2,0} get-tuple-element(custom-call.0), index=0
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(m.get()), absl_testing::IsOkAndHolds(true));

  const HloInstruction* call_0 = FindInstruction(m.get(), "custom-call.0");
  auto expect_layout = [](const Shape& shape,
                          absl::Span<const int64_t> minor_to_major) {
    const Layout expected = LayoutUtil::MakeLayout(minor_to_major);
    EXPECT_TRUE(LayoutUtil::Equal(shape.layout(), expected))
        << "Expected layout " << expected << ", actual " << shape.layout();
  };
  expect_layout(ShapeUtil::GetSubshape(call_0->shape(), {0}), {1, 2, 0});
  expect_layout(call_0->operand(0)->shape(), {1, 2, 0});
  expect_layout(call_0->operand(1)->shape(), {1, 2, 0});
}

TEST_F(LayoutAssignmentTest, MoveToHostCustomCallConstrained) {
  const char* module_str = R"(
HloModule TestModule

ENTRY entry {
  Arg_0 = f32[2,5,5]{0,1,2} parameter(0)
  custom-call.0 = f32[2,5,5] custom-call(Arg_0), custom_call_target="MoveToHost"
  ROOT custom-call.1 = f32[2,5,5]{2, 1, 0} custom-call(custom-call.0),
      custom_call_target="fixed_call", operand_layout_constraints={f32[2,5,5]{1,2,0}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(m.get()), absl_testing::IsOkAndHolds(true));

  const HloInstruction* call_0 = FindInstruction(m.get(), "custom-call.0");
  const Layout input_layout = call_0->operand(0)->shape().layout();
  const Layout output_layout = call_0->shape().layout();
  EXPECT_EQ(input_layout, LayoutUtil::GetDefaultLayoutForR3());
  EXPECT_EQ(output_layout, LayoutUtil::GetDefaultLayoutForR3());
}

TEST_F(LayoutAssignmentTest, MoveToDeviceCustomCallConstrained) {
  const char* module_str = R"(
HloModule TestModule

ENTRY entry {
  Arg_0 = f32[2,5,5]{1,2,0} parameter(0)
  custom-call.0 = f32[2,5,5] custom-call(Arg_0), custom_call_target="MoveToDevice"
  ROOT custom-call.1 = f32[2,5,5]{2, 1, 0} custom-call(custom-call.0),
      custom_call_target="fixed_call", operand_layout_constraints={f32[2,5,5]{0,1,2}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(m.get()), absl_testing::IsOkAndHolds(true));

  const HloInstruction* call_0 = FindInstruction(m.get(), "custom-call.0");
  const Layout input_layout = call_0->operand(0)->shape().layout();
  const Layout output_layout = call_0->shape().layout();
  EXPECT_EQ(input_layout, LayoutUtil::GetDefaultLayoutForR3());
  EXPECT_EQ(output_layout, LayoutUtil::GetDefaultLayoutForR3());
}

TEST_F(LayoutAssignmentTest, MoveToHostCustomCallAutoLayoutNotConstrained) {
  const char* module_str = R"(
HloModule TestModule

ENTRY entry {
  Arg_0 = f32[2,5,5] parameter(0)
  custom-call.0 = f32[2,5,5] custom-call(Arg_0), custom_call_target="MoveToHost"
  ROOT custom-call.1 = f32[2,5,5]{2, 1, 0} custom-call(custom-call.0),
      custom_call_target="fixed_call", operand_layout_constraints={f32[2,5,5]{1,2,0}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());
  // Clear layouts to simulate JAX AUTO layout.
  computation_layout.mutable_parameter_layout(0)->Clear();
  computation_layout.mutable_result_layout()->Clear();

  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(m.get()), absl_testing::IsOkAndHolds(true));

  const HloInstruction* call_0 = FindInstruction(m.get(), "custom-call.0");
  const Layout input_layout = call_0->operand(0)->shape().layout();
  const Layout output_layout = call_0->shape().layout();

  // Since it is AUTO layout, we should not have constrained it to default.
  // It should have layout {1, 2, 0} propagated from custom-call.1 constraint.
  Layout expected_layout = LayoutUtil::MakeLayout({1, 2, 0});
  EXPECT_EQ(input_layout, expected_layout);
  EXPECT_EQ(output_layout, expected_layout);
}

TEST_F(LayoutAssignmentTest,
       MoveToHostCustomCallResultAutoLayoutNotConstrained) {
  const char* module_str = R"(
HloModule TestModule

ENTRY entry {
  Arg_0 = f32[2,5,5] parameter(0)
  Arg_1 = f32[2,5,5]{1,2,0} parameter(1)
  add = f32[2,5,5] add(Arg_0, Arg_1)
  ROOT custom-call.1 = f32[2,5,5] custom-call(add), custom_call_target="MoveToHost"
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  // Clear layouts to simulate JAX AUTO layout.
  computation_layout.mutable_parameter_layout(0)->Clear();
  computation_layout.mutable_result_layout()->Clear();

  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(m.get()), absl_testing::IsOkAndHolds(true));

  const HloInstruction* call_1 = FindInstruction(m.get(), "custom-call.1");
  const Layout input_layout = call_1->operand(0)->shape().layout();
  const Layout output_layout = call_1->shape().layout();

  Layout expected_layout = LayoutUtil::MakeLayout({1, 2, 0});
  EXPECT_EQ(input_layout, expected_layout);
  EXPECT_EQ(output_layout, expected_layout);
}

TEST_F(LayoutAssignmentTest, FP16ROCmConvolutionHasNCHWLayoutRDNA) {
  const char* hlo = R"(
ENTRY entry {
  p0 = f16[2,64,64,16]{3,2,1,0} parameter(0)
  p1 = f16[6,16,3,32]{3,2,1,0} parameter(1)
  ROOT conv = (f64[2,64,64,32]{3,2,1,0}, u8[0]{0}) custom-call(p0, p1),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b10f_o10i->b10f,
    custom_call_target="__cudnn$convForward"
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo));
  ComputationLayout computation_layout(
      hlo_module->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(
      &computation_layout,
      se::GpuComputeCapability{
          se::RocmComputeCapability::EarliestRDNASupport()},
      default_device_description_);

  EXPECT_THAT(layout_assignment.Run(hlo_module.get()),
              absl_testing::IsOkAndHolds(true));

  // We start from b10f_o10i->b10f, meaning that the inputs start out as
  // NWHC_OWHI->NWHC. Layout assignment should yield layouts of the form
  // {1,2,3,0} for both inputs and for the output, therefore, in order to get to
  // the desired NCHW_OIHW->NCHW layout.
  EXPECT_THAT(
      RunFileCheck(hlo_module->ToString(HloPrintOptions::ShortParsable()), R"(
      // CHECK-DAG: [[P0:[^ ]+]] = {{.*}} parameter(0)
      // CHECK-DAG: [[P1:[^ ]+]] = {{.*}} parameter(1)
      // CHECK-DAG: [[COPY_P0:[^ ]+]] = {{.*}}{1,2,3,0} copy([[P0]])
      // CHECK-DAG: [[COPY_P1:[^ ]+]] = {{.*}}{1,2,3,0} copy([[P1]])
      // CHECK:     [[CONV:[^ ]+]] = {{.*}}{1,2,3,0}, {{.*}} custom-call([[COPY_P0]], [[COPY_P1]])
      )"),
      absl_testing::IsOkAndHolds(true));
}

TEST_F(LayoutAssignmentTest, FP32ROCmConvolutionHasNCHWLayoutRDNA) {
  const char* hlo = R"(
ENTRY entry {
  p0 = f32[2,64,64,16]{3,2,1,0} parameter(0)
  p1 = f32[6,16,3,32]{3,2,1,0} parameter(1)
  ROOT conv = (f64[2,64,64,32]{3,2,1,0}, u8[0]{0}) custom-call(p0, p1),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b10f_o10i->b10f,
    custom_call_target="__cudnn$convForward"
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo));
  ComputationLayout computation_layout(
      hlo_module->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(
      &computation_layout,
      se::GpuComputeCapability{
          se::RocmComputeCapability::EarliestRDNASupport()},
      default_device_description_);

  EXPECT_THAT(layout_assignment.Run(hlo_module.get()),
              absl_testing::IsOkAndHolds(true));

  // We start from b10f_o10i->b10f, meaning that the inputs start out as
  // NWHC_OWHI->NWHC. Layout assignment should yield layouts of the form
  // {1,2,3,0} for both inputs and for the output, therefore, in order to get to
  // the desired NCHW_OIHW->NCHW layout.
  EXPECT_THAT(
      RunFileCheck(hlo_module->ToString(HloPrintOptions::ShortParsable()), R"(
      // CHECK-DAG: [[P0:[^ ]+]] = {{.*}} parameter(0)
      // CHECK-DAG: [[P1:[^ ]+]] = {{.*}} parameter(1)
      // CHECK-DAG: [[COPY_P0:[^ ]+]] = {{.*}}{1,2,3,0} copy([[P0]])
      // CHECK-DAG: [[COPY_P1:[^ ]+]] = {{.*}}{1,2,3,0} copy([[P1]])
      // CHECK:     [[CONV:[^ ]+]] = {{.*}}{1,2,3,0}, {{.*}} custom-call([[COPY_P0]], [[COPY_P1]])
      )"),
      absl_testing::IsOkAndHolds(true));
}

TEST_F(LayoutAssignmentTest, FP16ROCmConvolutionHasNHWCLayoutCDNA) {
  // Enable ROCm NHWC for this test
  setenv("TF_USE_ROCM_NHWC", "true", 1);

  const char* hlo = R"(
ENTRY entry {
  p0 = f16[2,64,64,16]{3,2,1,0} parameter(0)
  p1 = f16[6,16,3,32]{3,2,1,0} parameter(1)
  ROOT conv = (f64[2,64,64,32]{3,2,1,0}, u8[0]{0}) custom-call(p0, p1),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b10f_o10i->b10f,
    custom_call_target="__cudnn$convForward"
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo));
  ComputationLayout computation_layout(
      hlo_module->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(
      &computation_layout,
      se::GpuComputeCapability{
          se::RocmComputeCapability::EarliestCDNASupport()},
      default_device_description_);

  EXPECT_THAT(layout_assignment.Run(hlo_module.get()),
              absl_testing::IsOkAndHolds(true));

  // We start from b10f_o10i->b10f, meaning that the inputs start out as
  // NWHC_OWHI->NWHC. Layout assignment should yield layouts of the form
  // {3,1,2,0} (transpose the middle dimensions) for both inputs and for the
  // output, therefore, in order to get to the desired NHWC_OHWI->NHWC layout.
  EXPECT_THAT(
      RunFileCheck(hlo_module->ToString(HloPrintOptions::ShortParsable()), R"(
      // CHECK-DAG: [[P0:[^ ]+]] = {{.*}} parameter(0)
      // CHECK-DAG: [[P1:[^ ]+]] = {{.*}} parameter(1)
      // CHECK-DAG: [[COPY_P0:[^ ]+]] = {{.*}}{3,1,2,0} copy([[P0]])
      // CHECK-DAG: [[COPY_P1:[^ ]+]] = {{.*}}{3,1,2,0} copy([[P1]])
      // CHECK:     [[CONV:[^ ]+]] = {{.*}}{3,1,2,0}, {{.*}} custom-call([[COPY_P0]], [[COPY_P1]])
      )"),
      absl_testing::IsOkAndHolds(true));

  // Clean up after the test
  unsetenv("TF_USE_ROCM_NHWC");
}

TEST_F(LayoutAssignmentTest, FP32ROCmConvolutionHasNCHWLayoutCDNA) {
  const char* hlo = R"(
ENTRY entry {
  p0 = f32[2,64,64,16]{3,2,1,0} parameter(0)
  p1 = f32[6,16,3,32]{3,2,1,0} parameter(1)
  ROOT conv = (f64[2,64,64,32]{3,2,1,0}, u8[0]{0}) custom-call(p0, p1),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b10f_o10i->b10f,
    custom_call_target="__cudnn$convForward"
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo));
  ComputationLayout computation_layout(
      hlo_module->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(
      &computation_layout,
      se::GpuComputeCapability{
          se::RocmComputeCapability::EarliestCDNASupport()},
      default_device_description_);

  EXPECT_THAT(layout_assignment.Run(hlo_module.get()),
              absl_testing::IsOkAndHolds(true));

  // We start from b10f_o10i->b10f, meaning that the inputs start out as
  // NWHC_OWHI->NWHC. Layout assignment should yield layouts of the form
  // {1,2,3,0} for both inputs and for the output, therefore, in order to get to
  // the desired NCHW_OIHW->NCHW layout.
  EXPECT_THAT(
      RunFileCheck(hlo_module->ToString(HloPrintOptions::ShortParsable()), R"(
      // CHECK-DAG: [[P0:[^ ]+]] = {{.*}} parameter(0)
      // CHECK-DAG: [[P1:[^ ]+]] = {{.*}} parameter(1)
      // CHECK-DAG: [[COPY_P0:[^ ]+]] = {{.*}}{1,2,3,0} copy([[P0]])
      // CHECK-DAG: [[COPY_P1:[^ ]+]] = {{.*}}{1,2,3,0} copy([[P1]])
      // CHECK:     [[CONV:[^ ]+]] = {{.*}}{1,2,3,0}, {{.*}} custom-call([[COPY_P0]], [[COPY_P1]])
      )"),
      absl_testing::IsOkAndHolds(true));
}

TEST_F(LayoutAssignmentTest, CuDNNConvolutionHasNHWCLayoutPostHopper) {
  const char* hlo = R"(
ENTRY entry {
  p0 = f32[1,64,64,16]{3,2,1,0} parameter(0)
  p1 = f32[3,16,3,32]{3,2,1,0} parameter(1)
  ROOT conv = (f32[1,64,64,32]{3,2,1,0}, u8[0]{0}) custom-call(p0, p1),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b10f_o10i->b10f,
    custom_call_target="__cudnn$convForwardGraph"
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo));
  ComputationLayout computation_layout(
      hlo_module->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        se::CudaComputeCapability::Hopper(),
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(hlo_module.get()),
              absl_testing::IsOkAndHolds(true));

  // We start from b10f_o10i->b10f, meaning that the inputs start out as
  // NWHC_OWHI->NWHC. Layout assignment should yield layouts of the form
  // {3,1,2,0} (transpose the middle dimensions) for both inputs and for the
  // output, therefore, in order to get to the desired NHWC_OHWI->NHWC layout.
  EXPECT_THAT(
      RunFileCheck(hlo_module->ToString(HloPrintOptions::ShortParsable()), R"(
// CHECK-DAG: [[P0:[^ ]+]] = {{.*}} parameter(0)
// CHECK-DAG: [[P1:[^ ]+]] = {{.*}} parameter(1)
// CHECK-DAG: [[COPY_P0:[^ ]+]] = {{.*}}{3,1,2,0} copy([[P0]])
// CHECK-DAG: [[COPY_P1:[^ ]+]] = {{.*}}{3,1,2,0} copy([[P1]])
// CHECK:     [[CONV:[^ ]+]] = {{.*}}{3,1,2,0}, {{.*}} custom-call([[COPY_P0]], [[COPY_P1]])
)"),
      absl_testing::IsOkAndHolds(true));
}

TEST_F(LayoutAssignmentTest, RawConvolutionLayoutAssignment) {
  const char* hlo = R"(
ENTRY entry {
  p0 = f32[1,64,64,16]{3,2,1,0} parameter(0)
  p1 = f32[32,3,3,16]{3,2,1,0} parameter(1)
  ROOT conv = f32[1,64,64,32]{3,2,1,0} convolution(p0, p1),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b10f_o10i->b10f,
    feature_group_count=1, convolution_kind=fprop
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo));

  DebugOptions debug_options = hlo_module->config().debug_options();
  debug_options.set_xla_gpu_experimental_enable_conv_fusion(true);
  hlo_module->mutable_config().set_debug_options(debug_options);

  ComputationLayout computation_layout(
      hlo_module->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        se::CudaComputeCapability::Hopper(),
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(hlo_module.get()),
              absl_testing::IsOkAndHolds(true));

  EXPECT_THAT(
      RunFileCheck(hlo_module->ToString(HloPrintOptions::ShortParsable()), R"(
// We start from b10f_o10i->b10f, meaning that the inputs start out as
// NWHC_OWHI->NWHC. Layout assignment should yield layouts of the form
// {3,1,2,0} (transpose the middle dimensions) for both inputs and for the
// output, therefore, in order to get to the desired NHWC_OHWI->NHWC layout
// because Hopper supports highly optimized NHWC convolutions.
// CHECK-DAG: [[P0:[^ ]+]] = {{.*}} parameter(0)
// CHECK-DAG: [[P1:[^ ]+]] = {{.*}} parameter(1)
// CHECK-DAG: [[COPY_P0:[^ ]+]] = {{.*}}{3,1,2,0} copy([[P0]])
// CHECK-DAG: [[COPY_P1:[^ ]+]] = {{.*}}{3,1,2,0} copy([[P1]])
// CHECK:     [[CONV:[^ ]+]] = {{.*}}{3,1,2,0} convolution([[COPY_P0]], [[COPY_P1]])
)"),
      absl_testing::IsOkAndHolds(true));
}

TEST_F(LayoutAssignmentTest, F64CuDNNConvolutionHasNCHWLayoutPostHopper) {
  const char* hlo = R"(
ENTRY entry {
  p0 = f64[2,64,64,16]{3,2,1,0} parameter(0)
  p1 = f64[6,16,3,32]{3,2,1,0} parameter(1)
  ROOT conv = (f64[2,64,64,32]{3,2,1,0}, u8[0]{0}) custom-call(p0, p1),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b10f_o10i->b10f,
    custom_call_target="__cudnn$convForwardGraph"
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(hlo));
  ComputationLayout computation_layout(
      hlo_module->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        se::CudaComputeCapability::Hopper(),
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(hlo_module.get()),
              absl_testing::IsOkAndHolds(true));

  // We start from b10f_o10i->b10f, meaning that the inputs start out as
  // NWHC_OWHI->NWHC. Layout assignment should yield layouts of the form
  // {1,2,3,0} for both inputs and for the output, therefore, in order to get to
  // the desired NCHW_OIHW->NCHW layout.
  EXPECT_THAT(
      RunFileCheck(hlo_module->ToString(HloPrintOptions::ShortParsable()), R"(
// CHECK-DAG: [[P0:[^ ]+]] = {{.*}} parameter(0)
// CHECK-DAG: [[P1:[^ ]+]] = {{.*}} parameter(1)
// CHECK-DAG: [[COPY_P0:[^ ]+]] = {{.*}}{1,2,3,0} copy([[P0]])
// CHECK-DAG: [[COPY_P1:[^ ]+]] = {{.*}}{1,2,3,0} copy([[P1]])
// CHECK:     [[CONV:[^ ]+]] = {{.*}}{1,2,3,0}, {{.*}} custom-call([[COPY_P0]], [[COPY_P1]])
)"),
      absl_testing::IsOkAndHolds(true));
}

TEST_F(LayoutAssignmentTest, ReduceOperandLayout) {
  const char* module_str = R"(
scalar_add_computation {
  scalar_lhs = c64[] parameter(0)
  scalar_rhs = c64[] parameter(1)
  ROOT add.1 = c64[] add(scalar_lhs, scalar_rhs)
}

ENTRY main {
  param_0 = c64[512,64,1024,32,128]{4,3,2,1,0} parameter(0)
  negate = c64[512,64,1024,32,128]{4,3,2,1,0} negate(param_0)
  constant_7 = c64[] constant((0, 0))
  ROOT reduce.2 = c64[512,1024,128]{2,1,0} reduce(negate, constant_7), dimensions={1,3}, to_apply=scalar_add_computation
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(m.get()), absl_testing::IsOkAndHolds(true));
  auto reduce = m->entry_computation()->root_instruction();
  EXPECT_EQ(reduce->operand(0)->shape().layout().minor_to_major(),
            LayoutUtil::MakeLayout({3, 1, 4, 2, 0}).minor_to_major());
}

TEST_F(LayoutAssignmentTest, ReduceOperandLayoutSmallReductionDimension) {
  // Same as ReduceOperandLayout, but with a small reduction dimension size.
  const char* module_str = R"(
scalar_add_computation {
  scalar_lhs = c64[] parameter(0)
  scalar_rhs = c64[] parameter(1)
  ROOT add.1 = c64[] add(scalar_lhs, scalar_rhs)
}

ENTRY main {
  param_0 = c64[512,16,1024,128]{3,2,1,0} parameter(0)
  negate = c64[512,16,1024,128]{3,2,1,0} negate(param_0)
  constant_7 = c64[] constant((0, 0))
  ROOT reduce.2 = c64[512,1024,128]{2,1,0} reduce(negate, constant_7), dimensions={1}, to_apply=scalar_add_computation
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(m.get()), absl_testing::IsOkAndHolds(true));
  auto reduce = m->entry_computation()->root_instruction();
  // Expect the layout to be unchanged.
  EXPECT_EQ(reduce->operand(0)->shape().layout().minor_to_major(),
            LayoutUtil::MakeLayout({3, 2, 1, 0}).minor_to_major());
}

TEST_F(LayoutAssignmentTest, AutoLayoutE4M3ContractingMinorFirst) {
  const char* hlo = R"(

  HloModule jit_dot_general_f8e4m3fn

  ENTRY main {
    p0 = f8e4m3fn[128,5120] parameter(0)
    p1 = f8e4m3fn[5120,10240] parameter(1)
    ROOT dot = f32[128,10240] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
  )";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> m,
      ParseAndReturnUnverifiedModule(
          hlo, {}, HloParserOptions().set_fill_missing_layouts(false)));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);
  EXPECT_THAT(layout_assignment.Run(m.get()), absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::Dot(m::Parameter(0).WithShape(F8E4M3FN, {128, 5120}, {1, 0}),
                 m::Parameter(1).WithShape(F8E4M3FN, {5120, 10240}, {0, 1}))
              .WithShape(F32, {128, 10240}, {1, 0})));
}

TEST_F(LayoutAssignmentTest, AutoLayoutS4DotContractingMinorLhs) {
  const char* hlo = R"(
  HloModule AutoLayoutS4DotContractingMinorLhs

  ENTRY main {
    p0 = s4[5120,128] parameter(0)
    p0.c = bf16[5120,128] convert(p0)
    p1 = bf16[5120,10240] parameter(1)
    ROOT dot = bf16[128,10240] dot(p0.c, p1), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  })";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> m,
      ParseAndReturnUnverifiedModule(
          hlo, {}, HloParserOptions().set_fill_missing_layouts(false)));
  DebugOptions debug_options = m->config().debug_options();
  debug_options.set_xla_gpu_experimental_pack_dot_operands_along_k_dimension(
      true);
  m->mutable_config().set_debug_options(debug_options);
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);
  EXPECT_THAT(layout_assignment.Run(m.get()), absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(m->entry_computation()->parameter_instruction(0),
              GmockMatch(m::Parameter(0).WithShape(S4, {5120, 128}, {0, 1})));
  EXPECT_THAT(
      m->entry_computation()->parameter_instruction(1),
      GmockMatch(m::Parameter(1).WithShape(BF16, {5120, 10240}, {1, 0})));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot().WithShape(BF16, {128, 10240}, {1, 0})));
}

TEST_F(LayoutAssignmentTest, AutoLayoutS4DotContractingMinorRhs) {
  const char* hlo = R"(
  HloModule AutoLayoutS4DotContractingMinorRhs

  ENTRY main {
    p0 = bf16[5120,128] parameter(0)
    p1 = s4[5120,10240] parameter(1)
    p1.c = bf16[5120,10240] convert(p1)
    ROOT dot = bf16[128,10240] dot(p0, p1.c), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  })";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> m,
      ParseAndReturnUnverifiedModule(
          hlo, {}, HloParserOptions().set_fill_missing_layouts(false)));
  DebugOptions debug_options = m->config().debug_options();
  debug_options.set_xla_gpu_experimental_pack_dot_operands_along_k_dimension(
      true);
  m->mutable_config().set_debug_options(debug_options);
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);
  EXPECT_THAT(layout_assignment.Run(m.get()), absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(m->entry_computation()->parameter_instruction(0),
              GmockMatch(m::Parameter(0).WithShape(BF16, {5120, 128}, {1, 0})));
  EXPECT_THAT(m->entry_computation()->parameter_instruction(1),
              GmockMatch(m::Parameter(1).WithShape(S4, {5120, 10240}, {0, 1})));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot().WithShape(BF16, {128, 10240}, {1, 0})));
}

TEST_F(LayoutAssignmentTest, AutoLayoutS4DotFollowingTheChain) {
  const char* hlo = R"(
  HloModule AutoLayoutS4DotFollowingTheChain

  ENTRY main {
    p0 = s4[3072,128] parameter(0)
    p0.c = s8[3072,128] convert(p0)
    p0.c2 = bf16[3072,128] convert(p0.c)
    p1 = bf16[3072,128] parameter(1)
    p0.m = bf16[3072,128] multiply(p0.c2, p1)
    p2 = bf16[3072,9216] parameter(2)
    ROOT dot = bf16[128,9216] dot(p0.m, p2), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  })";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> m,
      ParseAndReturnUnverifiedModule(
          hlo, {}, HloParserOptions().set_fill_missing_layouts(false)));
  DebugOptions debug_options = m->config().debug_options();
  debug_options.set_xla_gpu_experimental_pack_dot_operands_along_k_dimension(
      true);
  m->mutable_config().set_debug_options(debug_options);
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);

  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);
  ASSERT_THAT(layout_assignment.Run(m.get()), absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(m->entry_computation()->parameter_instruction(0),
              GmockMatch(m::Parameter(0).WithShape(S4, {3072, 128}, {0, 1})));
  EXPECT_THAT(m->entry_computation()->parameter_instruction(1),
              GmockMatch(m::Parameter(1).WithShape(BF16, {3072, 128}, {0, 1})));
  EXPECT_THAT(
      m->entry_computation()->parameter_instruction(2),
      GmockMatch(m::Parameter(2).WithShape(BF16, {3072, 9216}, {1, 0})));
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Dot().WithShape(BF16, {128, 9216}, {1, 0})));
}

TEST_F(LayoutAssignmentTest, DotSetsNonMandatoryConstraintIfAutoLayout) {
  const char* hlo = R"(
  HloModule t,
     entry_computation_layout={(f16[300,20]{1,0}, f16[40,20,50])
                               ->f16[300,2000]{1,0}}
  ENTRY main {
    p0 = f16[300,20] parameter(0)

    p1 = f16[40,20,50] parameter(1)
    t1 = f16[40,50,20] transpose(p1), dimensions={0,2,1}
    r1 = f16[2000,20] reshape(t1)

    ROOT dot = f16[300,2000] dot(p0,r1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  })";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> m,
      ParseAndReturnUnverifiedModule(
          hlo, {}, HloParserOptions().set_fill_missing_layouts(false)));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);

  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);
  EXPECT_THAT(layout_assignment.Run(m.get()), absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(m->ToString(), R"(
// CHECK: p1 = f16[40,20,50]{1,2,0} parameter(1)
)"),
              absl_testing::IsOkAndHolds(true));
};

TEST_F(LayoutAssignmentTest, VariadicReduceSameOperandLayout) {
  const char* module_str = R"(
HloModule variadic_reduce

reducer {
  %Arg_0.261 = s32[] parameter(0)
  %Arg_2.263 = s32[] parameter(2)
  mul = s32[] multiply(%Arg_0.261, %Arg_2.263)
  %Arg_1.260 = f32[] parameter(1)
  %Arg_3.262 = f32[] parameter(3)
  add = f32[] add(%Arg_1.260, %Arg_3.262)
  ROOT %tuple = (s32[], f32[]) tuple(mul, add)
}

ENTRY main {
  param_0 = f32[512,16,1024,128]{3,2,1,0} parameter(0)
  param_1 = s32[128,1024,16,512]{3,2,1,0} parameter(1)
  transpose = s32[512,16,1024,128]{0,1,2,3} transpose(param_1), dimensions={3,2,1,0}
  zero = f32[] constant(0.0)
  one = s32[] constant(1)
  ROOT reduce.2 = (s32[512,1024,128]{0,1,2}, f32[512,1024,128]{2,1,0}) reduce(transpose, param_0, one, zero), dimensions={1}, to_apply=reducer
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(m.get()), absl_testing::IsOkAndHolds(true));
  auto reduce = m->entry_computation()->root_instruction();
  EXPECT_EQ(reduce->operand(0)->shape().layout().minor_to_major(),
            reduce->operand(1)->shape().layout().minor_to_major());
}

TEST_F(LayoutAssignmentTest, SendRcvLayout) {
  const char* hlo = R"(
HloModule Module

condition  {
    p = (f32[100,100], (f32[100,100], u32[], token[])) parameter(0)
    ROOT lt = pred[] constant(1)
}

body {
    p = (f32[100,100], (f32[100,100], u32[], token[])) parameter(0)

    t1 = f32[100,100] get-tuple-element(p), index=0
    t = (f32[100,100], u32[], token[]) get-tuple-element(p), index=1
    sdone = token[] send-done(t), channel_id=3, frontend_attributes={
      _xla_send_recv_pipeline="0"
    }
    tk = token[] after-all()


    rcvd = (f32[100,100]{0,1}, u32[], token[]) recv(tk), channel_id=2
    zz = (f32[100,100]{0,1}, token[]) recv-done(rcvd), channel_id=2

    rcvd_d = get-tuple-element(zz), index=0

    snd = (f32[100,100]{0,1}, u32[], token[]) send(t1, tk), channel_id=3, frontend_attributes={
      _xla_send_recv_pipeline="0"
    }
    a = add(t1, t1)

    b = add(rcvd_d, a)

    ROOT tup =  tuple(b, snd)
}

ENTRY %main {
    p0 = f32[100,100] parameter(0)
    tk = token[] after-all()
    snd = (f32[100,100]{0,1}, u32[], token[]) send(p0, tk), channel_id=1, frontend_attributes={
      _xla_send_recv_pipeline="0"
    }
    t = tuple(p0, snd)
    ROOT loop = while(t), condition=condition, body=body
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  RunAndFilecheckHloRewrite(
      hlo,
      GpuLayoutAssignment{&computation_layout, default_gpu_cc_,
                          default_device_description_},
      R"(
// CHECK: (f32[100,100]{1,0}, u32[], token[]) recv
// CHECK:  (f32[100,100]{1,0}, token[]) recv-done
// CHECK:  (f32[100,100]{1,0}, u32[], token[]) send
                                )");
}

TEST_F(LayoutAssignmentTest, RaggedAllToAllLayoutSetRaggedDimToMajor) {
  absl::string_view hlo = R"(
  HloModule module

  ENTRY main {
    input = f32[16,4,8]{0,2,1} parameter(0)
    output = f32[32,4,8]{0,1,2} parameter(1)
    input_offsets = s32[2] parameter(2)
    send_sizes = s32[2] parameter(3)
    output_offsets = s32[2] parameter(4)
    recv_sizes = s32[2] parameter(5)
    ROOT ra2a = f32[32,4,8]{0,1,2} ragged-all-to-all(input, output,
      input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,1}}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo));

  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape(), /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(m.get()), absl_testing::IsOkAndHolds(true));
  auto ragged_all_to_all = FindInstruction(m.get(), HloOpcode::kRaggedAllToAll);
  EXPECT_THAT(ragged_all_to_all, NotNull());

  // Operands 0 and 1 of ragged-all-to-all can have different shapes, but they
  // must have the same layout.
  EXPECT_EQ(ragged_all_to_all->operand(0)->shape().layout().minor_to_major(),
            ragged_all_to_all->operand(1)->shape().layout().minor_to_major());

  // Operand 1 is aliased to the output of the ragged-all-to-all, so they must
  // have the same shape.
  EXPECT_EQ(ragged_all_to_all->operand(1)->shape(), ragged_all_to_all->shape());

  // The ragged dimension (0) must be in the most major position in the layout.
  EXPECT_TRUE(ShapeUtil::IsEffectivelyMostMajorDimension(
      ragged_all_to_all->shape(), 0));
}

TEST_F(LayoutAssignmentTest, ReshapeBitcastMinimizeChanges) {
  const char* hlo_text = R"(
  HloModule ReshapeLayout
  ENTRY entry {
    p0 = f32[10,20]{1,0} parameter(0)
    ROOT reshape = f32[1,10,20]{0,2,1} reshape(p0)
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(m::Op().WithShape(F32, {10, 20}, {1, 0}))
                             .WithShape(F32, {1, 10, 20}, {0, 2, 1})));
}

TEST_F(LayoutAssignmentTest, ReshapeBitcastMinimizeChangesMultipleOnes) {
  const char* hlo_text = R"(
  HloModule ReshapeLayout
  ENTRY entry {
    p0 = f32[10,20]{1,0} parameter(0)
    ROOT reshape = f32[1,10,1,20]{0,2,3,1} reshape(p0)
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Reshape(m::Op().WithShape(F32, {10, 20}, {1, 0}))
                             .WithShape(F32, {1, 10, 1, 20}, {0, 2, 3, 1})));
}

TEST_F(LayoutAssignmentTest, ReshapeBitcastMinimizeChangesAdjustNonDegenerate) {
  const char* hlo_text = R"(
  HloModule ReshapeLayout
  ENTRY entry {
    p0 = f32[10,20]{1,0} parameter(0)
    reshape = f32[1,10,20]{0,1,2} reshape(p0)
    ROOT out = f32[1,10,20]{0,2,1} add(reshape, reshape)
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);

  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  auto reshape = FindInstruction(module.get(), HloOpcode::kReshape);
  EXPECT_THAT(reshape, NotNull());
  EXPECT_EQ(reshape->shape().layout().minor_to_major(),
            (std::vector<int64_t>{0, 2, 1}));
}

TEST_F(LayoutAssignmentTest, CuDnnFusionBodyStaysLayoutConsistent) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
fused_computation {
  a = f32[2,8,768] parameter(0)
  b = f32[2,768,768] parameter(1)
  c = f32[2,8,768] parameter(2)
  d = f32[2,8,768] dot(a, b),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
  m = f32[2,8,768] multiply(d, c)
}

main {
  p0 = f32[2,8,768] parameter(0)
  p1 = f32[2,768,768] parameter(1)
  p2 = f32[2,8,768] parameter(2)
  f = f32[2,8,768] fusion(p0, p1, p2),
    kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion"}}
})"));

  ComputationLayout* computation_layout =
      module->mutable_entry_computation_layout();
  *computation_layout->mutable_result_layout() = ShapeLayout(
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 8, 768}, {1, 0, 2}));

  GpuLayoutAssignment layout_assignment(computation_layout, default_gpu_cc_,
                                        default_device_description_);
  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  HloVerifier verifier(
      HloVerifierOpts{}.MakeLayoutSensitive().WithInstructionCanChangeLayout(
          LayoutAssignment::InstructionCanChangeLayout));
  EXPECT_THAT(verifier.Run(module.get()).status(), absl_testing::IsOk())
      << module->ToString();
}

TEST_F(LayoutAssignmentTest,
       DotWithReversedBatchDimsPreservesDefaultRowColOrder) {
  const char* hlo_text = R"(
  HloModule DotReversedBatchDims
  ENTRY main {
    lhs = f32[2,4,12,65] parameter(0)
    rhs = f32[4,2,65,32] parameter(1)
    ROOT dot = f32[4,2,12,32] dot(lhs, rhs),
      lhs_batch_dims={1,0}, lhs_contracting_dims={3},
      rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);
  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  const HloInstruction* dot = module->entry_computation()->root_instruction();
  EXPECT_THAT(dot->operand(0)->shape().layout().minor_to_major(),
              ::testing::ElementsAre(3, 2, 0, 1));
}

// Returns all instructions in `module` that produce an array of `num_elements`
// elements.
std::vector<const HloInstruction*> ArraysOfSize(const HloModule* module,
                                                int64_t num_elements) {
  std::vector<const HloInstruction*> arrays;
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->shape().IsArray() &&
          ShapeUtil::ElementsIn(instruction->shape()) == num_elements) {
        arrays.push_back(instruction);
      }
    }
  }
  return arrays;
}

// A slice of a parameter, which has a fixed layout, feeds an all-gather that
// needs dimension 2 to be the most major dimension. $0 is the slice operation.
constexpr absl::string_view kSliceOfParameterFeedingAllGather = R"(
  HloModule module

  ENTRY main {
    weights = f32[8,16,32]{2,1,0} parameter(0)
    i = s32[] parameter(1)
    zero = s32[] constant(0)
    layer = f32[1,16,32]{2,1,0} $0
    ROOT gathered = f32[1,16,64]{2,1,0} all-gather(layer), dimensions={2}, replica_groups={{0,1}}
  })";
constexpr absl::string_view kSliceOfWeights =
    "slice(weights), slice={[3:4], [0:16], [0:32]}";
constexpr absl::string_view kDynamicSliceOfWeights =
    "dynamic-slice(weights, i, zero, zero), dynamic_slice_sizes={1,16,32}";

TEST_F(LayoutAssignmentTest, SliceOfParameterChangesLayoutForAllGather) {
  // The slice provides the layout the all-gather needs and reads the parameter
  // directly: nothing converts the whole parameter.
  for (absl::string_view slice_op : {kSliceOfWeights, kDynamicSliceOfWeights}) {
    SCOPED_TRACE(slice_op);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(absl::Substitute(
                             kSliceOfParameterFeedingAllGather, slice_op)));
    AssignLayouts(module.get());

    const HloInstruction* weights =
        module->entry_computation()->parameter_instruction(0);
    EXPECT_THAT(weights->shape().layout().minor_to_major(),
                ::testing::ElementsAre(2, 1, 0));
    const HloInstruction* layer = FindInstruction(module.get(), "layer");
    ASSERT_THAT(layer, NotNull());
    EXPECT_EQ(layer->operand(0), weights) << module->ToString();
    EXPECT_THAT(layer->shape().layout().minor_to_major(),
                ::testing::ElementsAre(1, 0, 2));
    const HloInstruction* gathered = FindInstruction(module.get(), "gathered");
    ASSERT_THAT(gathered, NotNull());
    EXPECT_EQ(gathered->operand(0), layer) << module->ToString();
    // The parameter is the only array of its size: there is no copy,
    // transpose or other conversion of it.
    EXPECT_THAT(ArraysOfSize(module.get(), 8 * 16 * 32),
                ::testing::ElementsAre(weights))
        << module->ToString();
  }
}

TEST_F(LayoutAssignmentTest, SliceOfLoopStateChangesLayoutForAllGather) {
  // Same as above, but the sliced array is carried through a while loop. The
  // slice reads the loop state directly, and the array is converted neither
  // inside nor outside of the loop.
  for (absl::string_view slice_op :
       {"slice(stack), slice={[3:4], [0:16], [0:32]}",
        "dynamic-slice(stack, i, zero, zero), dynamic_slice_sizes={1,16,32}"}) {
    SCOPED_TRACE(slice_op);
    const std::string hlo_text = absl::Substitute(
        R"(
    HloModule module

    body {
      p = (s32[], f32[8,16,32]{2,1,0}, f32[1,16,64]{2,1,0}) parameter(0)
      i = s32[] get-tuple-element(p), index=0
      stack = f32[8,16,32]{2,1,0} get-tuple-element(p), index=1
      zero = s32[] constant(0)
      layer = f32[1,16,32]{2,1,0} $0
      gathered = f32[1,16,64]{2,1,0} all-gather(layer), dimensions={2}, replica_groups={{0,1}}
      one = s32[] constant(1)
      next = s32[] add(i, one)
      ROOT result = (s32[], f32[8,16,32]{2,1,0}, f32[1,16,64]{2,1,0}) tuple(next, stack, gathered)
    }

    condition {
      p = (s32[], f32[8,16,32]{2,1,0}, f32[1,16,64]{2,1,0}) parameter(0)
      i = s32[] get-tuple-element(p), index=0
      limit = s32[] constant(8)
      ROOT lt = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      weights = f32[8,16,32]{2,1,0} parameter(0)
      acc = f32[1,16,64]{2,1,0} parameter(1)
      zero = s32[] constant(0)
      init = (s32[], f32[8,16,32]{2,1,0}, f32[1,16,64]{2,1,0}) tuple(zero, weights, acc)
      ROOT loop = (s32[], f32[8,16,32]{2,1,0}, f32[1,16,64]{2,1,0}) while(init), condition=condition, body=body
    })",
        slice_op);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(hlo_text));
    AssignLayouts(module.get());

    const HloInstruction* layer = FindInstruction(module.get(), "layer");
    ASSERT_THAT(layer, NotNull());
    EXPECT_THAT(layer->shape().layout().minor_to_major(),
                ::testing::ElementsAre(1, 0, 2));
    // The slice reads element 1 of the loop state.
    const HloInstruction* stack = layer->operand(0);
    EXPECT_THAT(stack, GmockMatch(m::GetTupleElement(m::Parameter(0), 1)))
        << module->ToString();
    EXPECT_THAT(stack->shape().layout().minor_to_major(),
                ::testing::ElementsAre(2, 1, 0));
    // The array is carried through the loop in the layout of the parameter.
    const HloInstruction* loop = FindInstruction(module.get(), "loop");
    ASSERT_THAT(loop, NotNull());
    EXPECT_THAT(loop->shape().tuple_shapes(1).layout().minor_to_major(),
                ::testing::ElementsAre(2, 1, 0));
    // Every array of the size of the sliced array is the parameter or an
    // element of a tuple it was put into, in the layout of the parameter:
    // there is no copy, transpose or other conversion of it.
    std::vector<const HloInstruction*> arrays =
        ArraysOfSize(module.get(), 8 * 16 * 32);
    EXPECT_THAT(arrays,
                ::testing::Contains(
                    module->entry_computation()->parameter_instruction(0)));
    for (const HloInstruction* array : arrays) {
      EXPECT_THAT(
          array->opcode(),
          ::testing::AnyOf(HloOpcode::kParameter, HloOpcode::kGetTupleElement))
          << array->ToString() << "\n"
          << module->ToString();
      EXPECT_THAT(array->shape().layout().minor_to_major(),
                  ::testing::ElementsAre(2, 1, 0))
          << array->ToString();
    }
  }
}

TEST_F(LayoutAssignmentTest,
       LayoutChangingDynamicSliceExistsOnlyBeforeLayoutNormalization) {
  // Between GpuLayoutAssignment and LayoutNormalization the module is valid
  // according to
  // GpuLayoutAssignment::InstructionCanChangeLayoutBeforeNormalization, but not
  // according to the stricter LayoutAssignment::InstructionCanChangeLayout that
  // the layout-sensitive verifiers of the GPU compiler use. After
  // LayoutNormalization the stricter predicate holds.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(absl::Substitute(
          kSliceOfParameterFeedingAllGather, kDynamicSliceOfWeights)));
  AssignLayouts(module.get());

  const HloInstruction* layer = FindInstruction(module.get(), "layer");
  ASSERT_THAT(layer, NotNull());
  ASSERT_NE(layer->shape().layout(), layer->operand(0)->shape().layout())
      << module->ToString();

  auto verify = [&](HloPredicate instruction_can_change_layout) {
    HloVerifier verifier(
        HloVerifierOpts{}.MakeLayoutSensitive().WithInstructionCanChangeLayout(
            instruction_can_change_layout));
    return verifier.Run(module.get()).status();
  };

  EXPECT_THAT(
      verify(
          GpuLayoutAssignment::InstructionCanChangeLayoutBeforeNormalization),
      absl_testing::IsOk())
      << module->ToString();
  EXPECT_THAT(verify(LayoutAssignment::InstructionCanChangeLayout),
              absl_testing::StatusIs(
                  ::testing::_,
                  ::testing::HasSubstr("Instruction shouldn't change layouts")))
      << module->ToString();

  LayoutNormalization layout_normalization(&NormalizeLayoutForGpuCustomCalls);
  EXPECT_THAT(layout_normalization.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  EXPECT_THAT(verify(LayoutAssignment::InstructionCanChangeLayout),
              absl_testing::IsOk())
      << module->ToString();
  // After normalization a transpose performs the layout change.
  EXPECT_THAT(FindInstruction(module.get(), HloOpcode::kTranspose), NotNull())
      << module->ToString();
}

TEST_F(LayoutAssignmentTest, SliceKeepsOperandLayoutWithoutOtherRequirement) {
  const char* hlo_text = R"(
  HloModule module

  ENTRY main {
    p = f32[8,16,32]{0,1,2} parameter(0)
    s = f32[2,16,32]{0,1,2} slice(p), slice={[0:2], [0:16], [0:32]}
    ROOT neg = f32[2,16,32]{0,1,2} negate(s)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  AssignLayouts(module.get());

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Negate(
          m::Slice(m::Parameter()).WithShape(F32, {2, 16, 32}, {0, 1, 2}))))
      << module->ToString();
}

TEST_F(LayoutAssignmentTest, DynamicSliceOfFreeOperandKeepsOperandLayout) {
  // The sliced array is neither a parameter nor loop state, so the slice keeps
  // the layout of its operand and the array takes the layout the all-gather
  // needs.
  for (absl::string_view weights_op :
       {"broadcast(scalar), dimensions={}", "broadcast(vector), dimensions={1}",
        "iota(), iota_dimension=1", "constant({...})"}) {
    SCOPED_TRACE(weights_op);
    const std::string hlo_text = absl::Substitute(
        R"(
    HloModule module

    ENTRY main {
      i = s32[] parameter(0)
      scalar = f32[] parameter(1)
      vector = f32[16]{0} parameter(2)
      weights = f32[8,16,32]{2,1,0} $0
      zero = s32[] constant(0)
      layer = f32[1,16,32]{2,1,0} dynamic-slice(weights, i, zero, zero), dynamic_slice_sizes={1,16,32}
      ROOT gathered = f32[1,16,64]{2,1,0} all-gather(layer), dimensions={2}, replica_groups={{0,1}}
    })",
        weights_op);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(hlo_text));
    AssignLayouts(module.get());

    const HloInstruction* weights = FindInstruction(module.get(), "weights");
    ASSERT_THAT(weights, NotNull());
    EXPECT_THAT(weights->shape().layout().minor_to_major(),
                ::testing::ElementsAre(1, 0, 2));
    const HloInstruction* layer = FindInstruction(module.get(), "layer");
    ASSERT_THAT(layer, NotNull());
    EXPECT_EQ(layer->operand(0), weights) << module->ToString();
    EXPECT_THAT(layer->shape().layout().minor_to_major(),
                ::testing::ElementsAre(1, 0, 2));
    const HloInstruction* gathered = FindInstruction(module.get(), "gathered");
    ASSERT_THAT(gathered, NotNull());
    EXPECT_EQ(gathered->operand(0), layer) << module->ToString();
    // Neither the sliced array nor the slice is copied.
    EXPECT_THAT(ArraysOfSize(module.get(), 8 * 16 * 32),
                ::testing::ElementsAre(weights))
        << module->ToString();
    EXPECT_THAT(ArraysOfSize(module.get(), 16 * 32),
                ::testing::ElementsAre(layer))
        << module->ToString();
  }
}

TEST_F(LayoutAssignmentTest, SliceOfFreeOperandKeepsOperandLayout) {
  const char* hlo_text = R"(
  HloModule module

  ENTRY main {
    weights = f32[8,16,32]{2,1,0} constant({...})
    layer = f32[1,16,32]{2,1,0} slice(weights), slice={[3:4], [0:16], [0:32]}
    ROOT gathered = f32[1,16,64]{2,1,0} all-gather(layer), dimensions={2}, replica_groups={{0,1}}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  AssignLayouts(module.get());

  const HloInstruction* layer = FindInstruction(module.get(), "layer");
  ASSERT_THAT(layer, NotNull());
  const HloInstruction* weights = layer->operand(0);
  EXPECT_EQ(weights->opcode(), HloOpcode::kConstant) << module->ToString();
  EXPECT_THAT(weights->shape().layout().minor_to_major(),
              ::testing::ElementsAre(1, 0, 2));
  EXPECT_THAT(layer->shape().layout().minor_to_major(),
              ::testing::ElementsAre(1, 0, 2));
  // Neither the sliced array nor the slice is copied.
  EXPECT_THAT(ArraysOfSize(module.get(), 8 * 16 * 32),
              ::testing::ElementsAre(weights))
      << module->ToString();
  EXPECT_THAT(ArraysOfSize(module.get(), 16 * 32),
              ::testing::ElementsAre(layer))
      << module->ToString();
}

TEST_F(LayoutAssignmentTest,
       DynamicSliceOfElementwiseOperandKeepsOperandLayout) {
  // The sliced array is computed from a parameter. Only a slice of an array
  // with a fixed layout changes the layout, so the layout the all-gather needs
  // reaches the computed array and the slice keeps the layout of its operand.
  const char* hlo_text = R"(
  HloModule module

  ENTRY main {
    p = f32[8,16,32]{2,1,0} parameter(0)
    i = s32[] parameter(1)
    weights = f32[8,16,32]{2,1,0} negate(p)
    zero = s32[] constant(0)
    layer = f32[1,16,32]{2,1,0} dynamic-slice(weights, i, zero, zero), dynamic_slice_sizes={1,16,32}
    ROOT gathered = f32[1,16,64]{2,1,0} all-gather(layer), dimensions={2}, replica_groups={{0,1}}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  AssignLayouts(module.get());

  const HloInstruction* layer = FindInstruction(module.get(), "layer");
  ASSERT_THAT(layer, NotNull());
  const HloInstruction* weights = layer->operand(0);
  EXPECT_EQ(weights->opcode(), HloOpcode::kNegate) << module->ToString();
  EXPECT_THAT(weights->shape().layout().minor_to_major(),
              ::testing::ElementsAre(1, 0, 2));
  EXPECT_THAT(layer->shape().layout().minor_to_major(),
              ::testing::ElementsAre(1, 0, 2));
}

TEST_F(LayoutAssignmentTest, SliceOfComputedArrayKeepsOperandLayout) {
  // `probs` is computed in this computation and has two users: a slice, whose
  // result has to be in the default layout, and a reduction followed by a
  // transpose, which would prefer `probs` in the layout {3,1,2,0}. The slice
  // does not change the layout, so its layout reaches `probs` and wins over the
  // preference of the reduction. Otherwise `probs` and everything it is
  // computed from would take the layout preferred by the small reduction
  // result, and the slice would have to transpose the large array.
  const char* hlo_text = R"(
  HloModule module

  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT sum = f32[] add(x, y)
  }

  ENTRY main {
    p = f32[4,64,8,65]{3,2,1,0} parameter(0)
    logits = f32[4,8,64,65]{3,2,1,0} transpose(p), dimensions={0,2,1,3}
    probs = f32[4,8,64,65]{3,2,1,0} exponential(logits)
    sliced = f32[4,8,64,64]{3,2,1,0} slice(probs), slice={[0:4], [0:8], [0:64], [0:64]}
    zero = f32[] constant(0)
    sums = f32[4,8,64]{2,1,0} reduce(probs, zero), dimensions={3}, to_apply=add
    sums_t = f32[4,64,8]{2,1,0} transpose(sums), dimensions={0,2,1}
    ROOT result = (f32[4,8,64,64]{3,2,1,0}, f32[4,64,8]{2,1,0}) tuple(sliced, sums_t)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  AssignLayouts(module.get());

  const HloInstruction* sliced = FindInstruction(module.get(), "sliced");
  ASSERT_THAT(sliced, NotNull());
  const HloInstruction* probs = sliced->operand(0);
  EXPECT_EQ(probs->opcode(), HloOpcode::kExp) << module->ToString();
  EXPECT_THAT(probs->shape().layout().minor_to_major(),
              ::testing::ElementsAre(3, 2, 1, 0))
      << module->ToString();
  EXPECT_THAT(sliced->shape().layout().minor_to_major(),
              ::testing::ElementsAre(3, 2, 1, 0))
      << module->ToString();
}

struct ScaledDotLayoutTestCase {
  std::string name;
  // Shapes of lhs, rhs, lhs_scale, rhs_scale and the output, followed by the
  // dot dimension numbers.
  std::string lhs, rhs, lhs_scale, rhs_scale, output, dims;
  // Expected minor_to_major of lhs and rhs.
  std::string lhs_minor_to_major, rhs_minor_to_major;
};

class ScaledDotLayoutAssignmentTest
    : public LayoutAssignmentTest,
      public ::testing::WithParamInterface<ScaledDotLayoutTestCase> {};

// Expects operand `operand_no` of `dot` to have `minor_to_major`, and its scale
// (unless a no-op scalar) to have the same layout without the sub-byte element
// size of an FP4 operand.
void ExpectScaledDotSideLayout(const HloInstruction* dot, int64_t operand_no,
                               absl::string_view minor_to_major) {
  const Layout& operand_layout = dot->operand(operand_no)->shape().layout();
  EXPECT_EQ(absl::StrJoin(operand_layout.minor_to_major(), ","),
            minor_to_major);
  const Shape& scale_shape = dot->operand(operand_no + 2)->shape();
  if (ShapeUtil::IsScalar(scale_shape)) {
    return;
  }
  EXPECT_EQ(scale_shape.layout(),
            LayoutUtil::MakeLayout(operand_layout.minor_to_major()));
}

TEST_P(ScaledDotLayoutAssignmentTest, OperandsAndScalesAreContractingMinor) {
  const ScaledDotLayoutTestCase& test_case = GetParam();
  const std::string hlo_text =
      absl::Substitute(R"(
  HloModule ScaledDot
  ENTRY main {
    lhs = $0 parameter(0)
    rhs = $1 parameter(1)
    lhs_scale = $2 parameter(2)
    rhs_scale = $3 parameter(3)
    ROOT dot = $4 scaled-dot(lhs, rhs, lhs_scale, rhs_scale), $5
  })",
                       test_case.lhs, test_case.rhs, test_case.lhs_scale,
                       test_case.rhs_scale, test_case.output, test_case.dims);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout, default_gpu_cc_,
                                        default_device_description_);
  EXPECT_THAT(layout_assignment.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  const HloInstruction* dot =
      FindInstruction(module.get(), HloOpcode::kScaledDot);
  ASSERT_THAT(dot, NotNull());
  ExpectScaledDotSideLayout(dot, 0, test_case.lhs_minor_to_major);
  ExpectScaledDotSideLayout(dot, 1, test_case.rhs_minor_to_major);
  EXPECT_TRUE(LayoutUtil::IsMonotonicWithDim0Major(dot->shape().layout()));
}

INSTANTIATE_TEST_SUITE_P(
    ScaledDot, ScaledDotLayoutAssignmentTest,
    ::testing::ValuesIn<ScaledDotLayoutTestCase>({
        {"NN", "f8e4m3fn[128,64]", "f8e4m3fn[64,256]", "f8e8m0fnu[128,2]",
         "f8e8m0fnu[2,256]", "bf16[128,256]",
         "lhs_contracting_dims={1}, rhs_contracting_dims={0}", "1,0", "0,1"},
        {"NT", "f8e4m3fn[128,64]", "f8e4m3fn[256,64]", "f8e8m0fnu[128,2]",
         "f8e8m0fnu[256,2]", "bf16[128,256]",
         "lhs_contracting_dims={1}, rhs_contracting_dims={1}", "1,0", "1,0"},
        {"TN", "f8e4m3fn[64,128]", "f8e4m3fn[64,256]", "f8e8m0fnu[2,128]",
         "f8e8m0fnu[2,256]", "bf16[128,256]",
         "lhs_contracting_dims={0}, rhs_contracting_dims={0}", "0,1", "0,1"},
        {"TT", "f8e4m3fn[64,128]", "f8e4m3fn[256,64]", "f8e8m0fnu[2,128]",
         "f8e8m0fnu[256,2]", "bf16[128,256]",
         "lhs_contracting_dims={0}, rhs_contracting_dims={1}", "0,1", "1,0"},
        {"BatchedNN", "f8e4m3fn[4,128,64]", "f8e4m3fn[4,64,256]",
         "f8e8m0fnu[4,128,2]", "f8e8m0fnu[4,2,256]", "bf16[4,128,256]",
         "lhs_batch_dims={0}, lhs_contracting_dims={2}, "
         "rhs_batch_dims={0}, rhs_contracting_dims={1}",
         "2,1,0", "1,2,0"},
        {"BatchedNT", "f8e4m3fn[4,128,64]", "f8e4m3fn[4,256,64]",
         "f8e8m0fnu[4,128,2]", "f8e8m0fnu[4,256,2]", "bf16[4,128,256]",
         "lhs_batch_dims={0}, lhs_contracting_dims={2}, "
         "rhs_batch_dims={0}, rhs_contracting_dims={2}",
         "2,1,0", "2,1,0"},
        {"BatchedTN", "f8e4m3fn[4,64,128]", "f8e4m3fn[4,64,256]",
         "f8e8m0fnu[4,2,128]", "f8e8m0fnu[4,2,256]", "bf16[4,128,256]",
         "lhs_batch_dims={0}, lhs_contracting_dims={1}, "
         "rhs_batch_dims={0}, rhs_contracting_dims={1}",
         "1,2,0", "1,2,0"},
        {"BatchedTT", "f8e4m3fn[4,64,128]", "f8e4m3fn[4,256,64]",
         "f8e8m0fnu[4,2,128]", "f8e8m0fnu[4,256,2]", "bf16[4,128,256]",
         "lhs_batch_dims={0}, lhs_contracting_dims={1}, "
         "rhs_batch_dims={0}, rhs_contracting_dims={2}",
         "1,2,0", "2,1,0"},
        // rhs already has a valid contracting-minor layout and keeps it, which
        // its scale must follow; lhs does not and gets reassigned.
        {"PresetLayouts", "f8e4m3fn[64,128,4]{2,1,0}",
         "f8e4m3fn[256,4,64]{2,1,0}", "f8e8m0fnu[2,128,4]{2,1,0}",
         "f8e8m0fnu[256,4,2]{2,0,1}", "bf16[4,128,256]{0,2,1}",
         "lhs_batch_dims={2}, lhs_contracting_dims={0}, "
         "rhs_batch_dims={1}, rhs_contracting_dims={2}",
         "0,1,2", "2,1,0"},
        {"Fp4SingleScaleBlock", "f4e2m1fn[32,128]{1,0:E(4)}",
         "f4e2m1fn[32,256]{0,1:E(4)}", "f8e8m0fnu[1,128]{1,0}",
         "f8e8m0fnu[1,256]{1,0}", "bf16[128,256]{1,0}",
         "lhs_contracting_dims={0}, rhs_contracting_dims={0}", "0,1", "0,1"},
        {"ScalarScale", "bf16[64,128]{1,0}", "f8e4m3fn[256,64]{0,1}", "bf16[]",
         "f8e8m0fnu[256,2]{0,1}", "bf16[128,256]{1,0}",
         "lhs_contracting_dims={0}, rhs_contracting_dims={1}", "0,1", "1,0"},
    }),
    [](const ::testing::TestParamInfo<ScaledDotLayoutTestCase>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace gpu
}  // namespace xla
