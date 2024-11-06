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

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_pjrt_test_base.h"

#ifdef GOOGLE_CUDA
namespace xla {
namespace gpu {
namespace {

class PtxKernelE2ETest : public HloPjRtTestBase {};

TEST_F(PtxKernelE2ETest, ScalarAdd) {
  constexpr char kModuleStr[] = R"(
    HloModule ptx_test
    
    ENTRY main {
      a = f32[] constant(3.0)
      b = f32[] constant(4.0)
      ROOT out = f32[] custom-call(a, b), custom_call_target="__gpu$xla.gpu.ptx"
    })";

  // Parse the module
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_NE(root, nullptr);
  EXPECT_EQ(root->opcode(), HloOpcode::kCustomCall);
  auto* custom_call = static_cast<HloCustomCallInstruction*>(root);

  std::string kernel_name = "add_kernel";
  std::string kernel_content = R"(
    .version 7.0
    .target sm_70
    .address_size 64
    
    .visible .entry add_kernel(
        .param .u64 input_a,
        .param .u64 input_b,
        .param .u64 output)
    {
      .reg .f32 a, b, c;
      .reg .u64 addr_a, addr_b, addr_out;
      
      ld.param.u64 addr_a, [input_a];
      ld.param.u64 addr_b, [input_b];
      ld.param.u64 addr_out, [output];
      
      ld.global.f32 a, [addr_a];
      ld.global.f32 b, [addr_b];
      add.f32 c, a, b;
      st.global.f32 [addr_out], c;
      
      ret;
    }
  )";
  int grid_x = 1, grid_y = 1, grid_z = 1;
  int block_x = 1, block_y = 1, block_z = 1;
  int shared_mem_bytes = 0;
  std::vector<int> output_indices = {2};

  std::string backend_config = absl::StrFormat(
      R"({name = "%s", source = "%s", grid_x = %d, grid_y = %d, grid_z = %d, block_x = %d, block_y = %d, block_z = %d, shared_mem_bytes = %d, output_indices = [%s]})",
      kernel_name,
      absl::StrReplaceAll(kernel_content, {{"\"", "\\\""}, {"\n", "\\n"}}),
      grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem_bytes,
      absl::StrJoin(output_indices, ", "));

  custom_call->set_raw_backend_config_string(backend_config);
  Literal result = ExecuteAndTransfer(std::move(module), {});
  EXPECT_EQ(result.Get<float>({}), 7.0f);
}

TEST_F(PtxKernelE2ETest, TensorAdd) {
  constexpr char kModuleStr[] = R"(
    HloModule ptx_tensor_test
    
    ENTRY main {
      a = f32[4] constant({1.0, 2.0, 3.0, 4.0})
      b = f32[4] constant({5.0, 6.0, 7.0, 8.0})
      ROOT out = f32[4] custom-call(a, b), custom_call_target="__gpu$xla.gpu.ptx"
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_NE(root, nullptr);
  EXPECT_EQ(root->opcode(), HloOpcode::kCustomCall);
  auto* custom_call = static_cast<HloCustomCallInstruction*>(root);

  std::string kernel_name = "tensor_add_kernel";
  std::string kernel_content = R"(
    .version 7.0
    .target sm_70
    .address_size 64
    
    .visible .entry tensor_add_kernel(
        .param .u64 input_a,
        .param .u64 input_b,
        .param .u64 output)
    {
      // Get base pointers
      .reg .u64 a_base, b_base, out_base;
      ld.param.u64 a_base, [input_a];
      ld.param.u64 b_base, [input_b];
      ld.param.u64 out_base, [output];
      
      // Thread ID calculation - just use thread ID directly for this simple case
      .reg .u32 tid;
      mov.u32 tid, %tid.x;
      
      // Hard-coded array size = 4
      .reg .pred p;
      setp.ge.u32 p, tid, 4;
      @p bra done;
      
      // Calculate byte offset (4 bytes per float)
      .reg .u64 offset;
      cvt.u64.u32 offset, tid;  // Convert tid to 64-bit
      mul.lo.u64 offset, offset, 4;  // Each float is 4 bytes
      
      // Calculate element addresses
      .reg .u64 a_addr, b_addr, out_addr;
      add.u64 a_addr, a_base, offset;
      add.u64 b_addr, b_base, offset;
      add.u64 out_addr, out_base, offset;
      
      // Load input values
      .reg .f32 a_val, b_val, result;
      ld.global.f32 a_val, [a_addr];
      ld.global.f32 b_val, [b_addr];
      
      // Perform addition
      add.f32 result, a_val, b_val;
      
      // Store result
      st.global.f32 [out_addr], result;
      
    done:
      ret;
    }
  )";

  int grid_x = 1, grid_y = 1, grid_z = 1;
  int block_x = 4, block_y = 1, block_z = 1;
  int shared_mem_bytes = 0;
  std::vector<int> output_indices = {2};

  std::string backend_config = absl::StrFormat(
      R"({name = "%s", source = "%s", grid_x = %d, grid_y = %d, grid_z = %d, block_x = %d, block_y = %d, block_z = %d, shared_mem_bytes = %d, output_indices = [%s]})",
      kernel_name,
      absl::StrReplaceAll(kernel_content, {{"\"", "\\\""}, {"\n", "\\n"}}),
      grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem_bytes,
      absl::StrJoin(output_indices, ", "));

  custom_call->set_raw_backend_config_string(backend_config);
  Literal result = ExecuteAndTransfer(std::move(module), {});

  EXPECT_EQ(result.Get<float>({0}), 6.0f);
  EXPECT_EQ(result.Get<float>({1}), 8.0f);
  EXPECT_EQ(result.Get<float>({2}), 10.0f);
  EXPECT_EQ(result.Get<float>({3}), 12.0f);
}

TEST_F(PtxKernelE2ETest, TensorAddWithoutOutputIndices) {
  constexpr char kModuleStr[] = R"(
    HloModule ptx_tensor_test
    
    ENTRY main {
      a = f32[4] constant({1.0, 2.0, 3.0, 4.0})
      b = f32[4] constant({5.0, 6.0, 7.0, 8.0})
      ROOT out = f32[4] custom-call(a, b), custom_call_target="__gpu$xla.gpu.ptx"
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_NE(root, nullptr);
  EXPECT_EQ(root->opcode(), HloOpcode::kCustomCall);
  auto* custom_call = static_cast<HloCustomCallInstruction*>(root);

  std::string kernel_name = "tensor_add_kernel";
  std::string kernel_content = R"(
    .version 7.0
    .target sm_70
    .address_size 64
    
    .visible .entry tensor_add_kernel(
        .param .u64 input_a,
        .param .u64 input_b,
        .param .u64 output)
    {
      // Get base pointers
      .reg .u64 a_base, b_base, out_base;
      ld.param.u64 a_base, [input_a];
      ld.param.u64 b_base, [input_b];
      ld.param.u64 out_base, [output];
      
      // Thread ID calculation - just use thread ID directly for this simple case
      .reg .u32 tid;
      mov.u32 tid, %tid.x;
      
      // Hard-coded array size = 4
      .reg .pred p;
      setp.ge.u32 p, tid, 4;
      @p bra done;
      
      // Calculate byte offset (4 bytes per float)
      .reg .u64 offset;
      cvt.u64.u32 offset, tid;  // Convert tid to 64-bit
      mul.lo.u64 offset, offset, 4;  // Each float is 4 bytes
      
      // Calculate element addresses
      .reg .u64 a_addr, b_addr, out_addr;
      add.u64 a_addr, a_base, offset;
      add.u64 b_addr, b_base, offset;
      add.u64 out_addr, out_base, offset;
      
      // Load input values
      .reg .f32 a_val, b_val, result;
      ld.global.f32 a_val, [a_addr];
      ld.global.f32 b_val, [b_addr];
      
      // Perform addition
      add.f32 result, a_val, b_val;
      
      // Store result
      st.global.f32 [out_addr], result;
      
    done:
      ret;
    }
  )";

  int grid_x = 1, grid_y = 1, grid_z = 1;
  int block_x = 4, block_y = 1, block_z = 1;
  int shared_mem_bytes = 0;

  std::string backend_config = absl::StrFormat(
      R"({name = "%s", source = "%s", grid_x = %d, grid_y = %d, grid_z = %d, block_x = %d, block_y = %d, block_z = %d, shared_mem_bytes = %d})",
      kernel_name,
      absl::StrReplaceAll(kernel_content, {{"\"", "\\\""}, {"\n", "\\n"}}),
      grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem_bytes);

  custom_call->set_raw_backend_config_string(backend_config);
  Literal result = ExecuteAndTransfer(std::move(module), {});

  EXPECT_EQ(result.Get<float>({0}), 6.0f);
  EXPECT_EQ(result.Get<float>({1}), 8.0f);
  EXPECT_EQ(result.Get<float>({2}), 10.0f);
  EXPECT_EQ(result.Get<float>({3}), 12.0f);
}

TEST_F(PtxKernelE2ETest, TensorAddWithNonTrivialOutputIndices) {
  constexpr char kModuleStr[] = R"(
    HloModule ptx_tensor_test
    
    ENTRY main {
      a = f32[4] constant({1.0, 2.0, 3.0, 4.0})
      b = f32[4] constant({5.0, 6.0, 7.0, 8.0})
      ROOT out = f32[4] custom-call(a, b), custom_call_target="__gpu$xla.gpu.ptx"
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_NE(root, nullptr);
  EXPECT_EQ(root->opcode(), HloOpcode::kCustomCall);
  auto* custom_call = static_cast<HloCustomCallInstruction*>(root);

  std::string kernel_name = "tensor_add_kernel";
  std::string kernel_content = R"(
    .version 7.0
    .target sm_70
    .address_size 64
    
    .visible .entry tensor_add_kernel(
        .param .u64 input_a,
        .param .u64 output,
        .param .u64 input_b
        )
    {
      // Get base pointers
      .reg .u64 a_base, b_base, out_base;
      ld.param.u64 a_base, [input_a];
      ld.param.u64 out_base, [output];
      ld.param.u64 b_base, [input_b];
      
      // Thread ID calculation - just use thread ID directly for this simple case
      .reg .u32 tid;
      mov.u32 tid, %tid.x;
      
      // Hard-coded array size = 4
      .reg .pred p;
      setp.ge.u32 p, tid, 4;
      @p bra done;
      
      // Calculate byte offset (4 bytes per float)
      .reg .u64 offset;
      cvt.u64.u32 offset, tid;  // Convert tid to 64-bit
      mul.lo.u64 offset, offset, 4;  // Each float is 4 bytes
      
      // Calculate element addresses
      .reg .u64 a_addr, b_addr, out_addr;
      add.u64 a_addr, a_base, offset;
      add.u64 b_addr, b_base, offset;
      add.u64 out_addr, out_base, offset;
      
      // Load input values
      .reg .f32 a_val, b_val, result;
      ld.global.f32 a_val, [a_addr];
      ld.global.f32 b_val, [b_addr];
      
      // Perform addition
      add.f32 result, a_val, b_val;
      
      // Store result
      st.global.f32 [out_addr], result;
      
    done:
      ret;
    }
  )";

  int grid_x = 1, grid_y = 1, grid_z = 1;
  int block_x = 4, block_y = 1, block_z = 1;
  int shared_mem_bytes = 0;
  std::vector<int> output_indices = {1};

  std::string backend_config = absl::StrFormat(
      R"({name = "%s", source = "%s", grid_x = %d, grid_y = %d, grid_z = %d, block_x = %d, block_y = %d, block_z = %d, shared_mem_bytes = %d, output_indices = [%s]})",
      kernel_name,
      absl::StrReplaceAll(kernel_content, {{"\"", "\\\""}, {"\n", "\\n"}}),
      grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_mem_bytes,
      absl::StrJoin(output_indices, ", "));

  custom_call->set_raw_backend_config_string(backend_config);
  Literal result = ExecuteAndTransfer(std::move(module), {});

  EXPECT_EQ(result.Get<float>({0}), 6.0f);
  EXPECT_EQ(result.Get<float>({1}), 8.0f);
  EXPECT_EQ(result.Get<float>({2}), 10.0f);
  EXPECT_EQ(result.Get<float>({3}), 12.0f);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

#endif  // GOOGLE_CUDA