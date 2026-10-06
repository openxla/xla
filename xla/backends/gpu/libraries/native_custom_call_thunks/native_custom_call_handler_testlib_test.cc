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

#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_testlib.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_emitter_context.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registration.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_registry.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_utils.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_scratch_context.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/ffi/attributes.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_memory_space_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

// Records what the context served the handler, so a test can assert on it.
struct HandlerObservation {
  int64_t num_kernel_arguments = 0;
  int64_t result_argument_position = -1;
  Shape operand_shape;
  int32_t val = 0;
};

// Returns a handler that records into `out` what the context served it.
//
// The registry takes an `absl::AnyInvocable` and `EmitThunksWith` takes an
// `absl::FunctionRef`, so a handler can capture test-local state instead of
// having to smuggle it out through a global.
auto ObservingHandler(HandlerObservation& out) {
  return [&out](const HloCustomCallInstruction& instr,
                const NativeCustomCallEmitterContext& ctx)
             -> absl::StatusOr<ThunkSequence> {
    ABSL_ASSIGN_OR_RETURN(emitters::KernelArguments kernel_args,
                          ctx.CreateKernelArguments());
    out.num_kernel_arguments = kernel_args.args().size();
    ABSL_ASSIGN_OR_RETURN(out.result_argument_position,
                          kernel_args.PositionOfResult({}));

    ABSL_ASSIGN_OR_RETURN(ShapedSlice operand,
                          ctx.GetOperandShapedSlice(0, {}));
    out.operand_shape = operand.shape;

    ABSL_ASSIGN_OR_RETURN(xla::ffi::Attributes attrs, ctx.GetFfiAttributes());
    ABSL_ASSIGN_OR_RETURN(out.val, attrs.Get<int32_t>("val"));

    return ThunkSequence::Empty();
  };
}

absl::StatusOr<ThunkSequence> EmptyHandler(
    const HloCustomCallInstruction&, const NativeCustomCallEmitterContext&) {
  return ThunkSequence::Empty();
}

// Registered at static-init time so that the registry dispatch path below has
// a target to resolve.
XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER("xla.gpu.test_testlib_registered",
                                            EmptyHandler);

// Asks for one scratch buffer in default memory and one in collective memory.
absl::StatusOr<std::vector<Shape>> TwoScratchBuffers(
    const HloCustomCallInstruction&, const NativeCustomCallScratchContext&) {
  ABSL_ASSIGN_OR_RETURN(Shape plain, MakeScratchShape(F32, {16}));
  ABSL_ASSIGN_OR_RETURN(
      Shape collective,
      MakeScratchShape(S32, {4}, NativeCustomCallMemorySpace::kCollective));
  return std::vector<Shape>{plain, collective};
}

XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER(
    "xla.gpu.test_testlib_scratch",
    NativeCustomCallHandlerBundle{
        /*emit_thunks=*/EmptyHandler,
        /*request_scratch_buffers=*/TwoScratchBuffers});

constexpr absl::string_view kScratchHlo = R"hlo(
  ENTRY e {
    p0 = f32[4] parameter(0)
    c0 = f32[4] custom-call(p0),
      custom_call_target="xla.gpu.test_testlib_scratch"
    neg = f32[4] negate(c0)
    ROOT c1 = f32[4] custom-call(neg),
      custom_call_target="xla.gpu.test_testlib_scratch"
  }
)hlo";

constexpr absl::string_view kHlo = R"hlo(
  ENTRY e {
    p0 = f32[2,3] parameter(0)
    ROOT c = f32[4] custom-call(p0),
      custom_call_target="xla.gpu.test_testlib_registered",
      backend_config="{val = 7 : i32}"
  }
)hlo";

TEST(NativeCustomCallHandlerTesterTest, ServesTheHandlerARealBufferAssignment) {
  ASSERT_OK_AND_ASSIGN(auto tester,
                       NativeCustomCallHandlerTester::Create(kHlo));

  HandlerObservation observation;
  ASSERT_OK(tester->EmitThunksWith(ObservingHandler(observation)));

  // One operand plus one result.
  EXPECT_EQ(observation.num_kernel_arguments, 2);
  EXPECT_EQ(observation.result_argument_position, 1);
  EXPECT_EQ(observation.operand_shape, ShapeUtil::MakeShape(F32, {2, 3}));
  EXPECT_EQ(observation.val, 7);
}

TEST(NativeCustomCallHandlerTesterTest,
     EmitThunksDispatchesThroughTheRegistry) {
  ASSERT_OK_AND_ASSIGN(auto tester,
                       NativeCustomCallHandlerTester::Create(kHlo));
  EXPECT_OK(tester->EmitThunks());
}

TEST(NativeCustomCallHandlerTesterTest, EmitThunksWithBypassesTheRegistry) {
  ASSERT_OK_AND_ASSIGN(auto tester, NativeCustomCallHandlerTester::Create(R"hlo(
    ENTRY e {
      p0 = f32[4] parameter(0)
      ROOT c = f32[4] custom-call(p0), custom_call_target="not.registered"
    }
  )hlo"));

  EXPECT_THAT(tester->EmitThunks(), StatusIs(absl::StatusCode::kNotFound,
                                             HasSubstr("not.registered")));

  EXPECT_OK(tester->EmitThunksWith(EmptyHandler));
}

TEST(NativeCustomCallHandlerTesterTest, SelectsANamedInstruction) {
  ASSERT_OK_AND_ASSIGN(auto tester, NativeCustomCallHandlerTester::Create(
                                        R"hlo(
    ENTRY e {
      p0 = f32[4] parameter(0)
      c0 = f32[4] custom-call(p0), custom_call_target="first"
      ROOT c1 = f32[4] custom-call(c0), custom_call_target="second"
    }
  )hlo",
                                        {/*gpu_model=*/GpuModel::H100_SXM,
                                         /*instruction_name=*/"c0"}));
  EXPECT_EQ(tester->instruction().custom_call_target(), "first");
}

TEST(NativeCustomCallHandlerTesterTest, RejectsANonCustomCallInstruction) {
  EXPECT_THAT(NativeCustomCallHandlerTester::Create(R"hlo(
    ENTRY e {
      p0 = f32[4] parameter(0)
      ROOT add = f32[4] add(p0, p0)
    }
  )hlo"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("is not a custom call")));
}

TEST(NativeCustomCallHandlerTesterTest, RejectsAnUnknownInstructionName) {
  EXPECT_THAT(NativeCustomCallHandlerTester::Create(
                  kHlo, {/*gpu_model=*/GpuModel::H100_SXM,
                         /*instruction_name=*/"nope"}),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("nope")));
}

TEST(NativeCustomCallHandlerTesterTest, AppendsScratchBuffersToTheResult) {
  ASSERT_OK_AND_ASSIGN(auto tester,
                       NativeCustomCallHandlerTester::Create(kScratchHlo));

  // The scratch assigner turned the array result into a tuple of the original
  // result plus the two scratch buffers. Note that buffer assignment rewrites
  // the memory space in the layout to the allocation color, so the S(8) scratch
  // buffer shows up as S(7) from here on (just like S(1) buffers do).
  const HloCustomCallInstruction& instr = tester->instruction();
  EXPECT_EQ(instr.name(), "c1");
  EXPECT_EQ(instr.shape().ToString(/*print_layout=*/true),
            "(f32[4]{0}, f32[16]{0}, s32[4]{0:S(7)})");

  ASSERT_OK_AND_ASSIGN(ShapedSlice result,
                       tester->context().GetResultShapedSlice({0}));
  ASSERT_OK_AND_ASSIGN(ShapedSlice plain,
                       tester->context().GetResultShapedSlice({1}));
  ASSERT_OK_AND_ASSIGN(ShapedSlice collective,
                       tester->context().GetResultShapedSlice({2}));
  EXPECT_EQ(result.shape, ShapeUtil::MakeShape(F32, {4}));
  EXPECT_EQ(plain.shape, ShapeUtil::MakeShape(F32, {16}));
  EXPECT_EQ(plain.slice.size(), 16 * sizeof(float));
  EXPECT_EQ(collective.shape.layout().memory_space(),
            static_cast<int64_t>(MemorySpaceColor::kCollective));
  EXPECT_EQ(collective.slice.size(), 4 * sizeof(int32_t));
  EXPECT_EQ(collective.slice.allocation()->color(),
            static_cast<int>(MemorySpaceColor::kCollective));

  // One operand plus three results.
  ASSERT_OK_AND_ASSIGN(emitters::KernelArguments kernel_args,
                       tester->context().CreateKernelArguments());
  EXPECT_EQ(kernel_args.args().size(), 4);
}

TEST(NativeCustomCallHandlerTesterTest, CanSkipTheScratchAssigner) {
  NativeCustomCallHandlerTester::Options options;
  options.run_scratch_assigner = false;
  ASSERT_OK_AND_ASSIGN(
      auto tester, NativeCustomCallHandlerTester::Create(kScratchHlo, options));
  EXPECT_EQ(tester->instruction().shape(), ShapeUtil::MakeShape(F32, {4}));
}

}  // namespace
}  // namespace xla::gpu
