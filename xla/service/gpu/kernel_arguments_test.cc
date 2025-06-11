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
#include "xla/service/gpu/kernel_arguments.h"

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_assigner.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/hlo_ordering.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class KernelArgumentsTest : public HloTestBase {
 public:
  absl::StatusOr<std::unique_ptr<BufferAssignment>> AssignBuffers(
      HloModule* module) {
    constexpr uint64_t pointer_size = 8;
    const se::DeviceDescription& gpu_device_info =
        backend().default_stream_executor()->GetDeviceDescription();
    TF_RETURN_IF_ERROR(
        ScheduleGpuModule(module, pointer_size, gpu_device_info).status());

    auto buffer_size_bytes_function =
        [](const BufferValue& buffer_value) -> int64_t {
      return ShapeSizeBytesFunction(pointer_size)(buffer_value.shape());
    };

    return BufferAssigner::Run(
        module, std::make_unique<SequentialHloOrdering>(module->schedule()),
        buffer_size_bytes_function,
        /*color_alignment=*/
        [](LogicalBuffer::Color) { return kXlaAllocatedBufferAlignBytes; });
  }
};

TEST_F(KernelArgumentsTest, InterleavedOutputIndicesTest) {
  const absl::string_view hlo_string = R"(
HloModule TestModule

ENTRY main {
  param0 = f32[10] parameter(0)
  param1 = f32[20] parameter(1)
  param2 = f32[30] parameter(2)
  
  add1 = f32[10] add(param0, param0)
  add2 = f32[20] add(param1, param1)
  
  ROOT tuple_result = (f32[10], f32[20]) tuple(add1, add2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment, AssignBuffers(module.get()));

  // Get the root instruction (tuple) and its operands
  HloInstruction* root = module->entry_computation()->root_instruction();
  std::vector<const HloInstruction*> needed_operands = {
      module->entry_computation()->parameter_instruction(0),  // param0
      module->entry_computation()->parameter_instruction(1),  // param1
      module->entry_computation()->parameter_instruction(2)   // param2
  };

  // Test case 1: Interleave outputs at positions 1 and 4
  // Expected order: input0, output0, input1, input2, output1
  std::vector<int32_t> interleaved_indices = {1, 4};

  TF_ASSERT_OK_AND_ASSIGN(
      KernelArguments kernel_args,
      KernelArguments::Create(*buffer_assignment, root, needed_operands,
                              interleaved_indices));

  const auto& args = kernel_args.args();
  ASSERT_EQ(args.size(), 5);  // 3 inputs + 2 outputs

  // Verify the interleaving:
  // Position 0: input0 (param0)
  EXPECT_EQ(args[0].shape(), ShapeUtil::MakeShape(F32, {10}));
  EXPECT_FALSE(args[0].written());

  // Position 1: output0 (first element of tuple result)
  EXPECT_EQ(args[1].shape(), ShapeUtil::MakeShape(F32, {10}));
  EXPECT_TRUE(args[1].written());

  // Position 2: input1 (param1)
  EXPECT_EQ(args[2].shape(), ShapeUtil::MakeShape(F32, {20}));
  EXPECT_FALSE(args[2].written());

  // Position 3: input2 (param2)
  EXPECT_EQ(args[3].shape(), ShapeUtil::MakeShape(F32, {30}));
  EXPECT_FALSE(args[3].written());

  // Position 4: output1 (second element of tuple result)
  EXPECT_EQ(args[4].shape(), ShapeUtil::MakeShape(F32, {20}));
  EXPECT_TRUE(args[4].written());
}

TEST_F(KernelArgumentsTest, InterleavedOutputIndicesEdgeCases) {
  const absl::string_view hlo_string = R"(
HloModule TestModule

ENTRY main {
  param0 = f32[5] parameter(0)
  param1 = f32[5] parameter(1)
  
  add_result = f32[5] add(param0, param1)
  
  ROOT tuple_result = (f32[5],) tuple(add_result)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment, AssignBuffers(module.get()));

  HloInstruction* root = module->entry_computation()->root_instruction();
  std::vector<const HloInstruction*> needed_operands = {
      module->entry_computation()->parameter_instruction(0),  // param0
      module->entry_computation()->parameter_instruction(1)   // param1
  };

  // Test case: Output at the beginning (position 0)
  std::vector<int32_t> interleaved_indices = {0};

  TF_ASSERT_OK_AND_ASSIGN(
      KernelArguments kernel_args,
      KernelArguments::Create(*buffer_assignment, root, needed_operands,
                              interleaved_indices));

  const auto& args = kernel_args.args();
  ASSERT_EQ(args.size(), 3);  // 2 inputs + 1 output

  // Expected order: output0, input0, input1
  EXPECT_TRUE(args[0].written());   // output
  EXPECT_FALSE(args[1].written());  // input0
  EXPECT_FALSE(args[2].written());  // input1
}

TEST_F(KernelArgumentsTest, InterleavedOutputIndicesErrorCases) {
  const absl::string_view hlo_string = R"(
HloModule TestModule

ENTRY main {
  param0 = f32[5] parameter(0)
  ROOT result = f32[5] add(param0, param0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment, AssignBuffers(module.get()));

  HloInstruction* root = module->entry_computation()->root_instruction();
  std::vector<const HloInstruction*> needed_operands = {
      module->entry_computation()->parameter_instruction(0)};

  // Test case: Output index out of bounds
  std::vector<int32_t> invalid_indices = {
      5};  // Only 2 total positions (1 input + 1 output)

  auto result = KernelArguments::Create(*buffer_assignment, root,
                                        needed_operands, invalid_indices);
  EXPECT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              testing::HasSubstr("Output index out of bounds"));
}

TEST_F(KernelArgumentsTest, EmptyInterleavedIndicesFallback) {
  const absl::string_view hlo_string = R"(
HloModule TestModule

ENTRY main {
  param0 = f32[5] parameter(0)
  ROOT result = f32[5] add(param0, param0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment, AssignBuffers(module.get()));

  HloInstruction* root = module->entry_computation()->root_instruction();
  std::vector<const HloInstruction*> needed_operands = {
      module->entry_computation()->parameter_instruction(0)};

  // Test case: Empty interleaved indices should fall back to regular Create
  std::vector<int32_t> empty_indices = {};

  TF_ASSERT_OK_AND_ASSIGN(
      KernelArguments kernel_args,
      KernelArguments::Create(*buffer_assignment, root, needed_operands,
                              empty_indices));

  const auto& args = kernel_args.args();
  ASSERT_EQ(args.size(), 2);  // 1 input + 1 output

  // Expected order: input, output (regular order)
  EXPECT_FALSE(args[0].written());  // input
  EXPECT_TRUE(args[1].written());   // output
}

}  // namespace
}  // namespace gpu
}  // namespace xla