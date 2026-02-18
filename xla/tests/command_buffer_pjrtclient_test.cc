/* Copyright 2024 The OpenXLA Authors.

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

// End-to-end test for the Virtual Memory Management (VMM) allocator.
// This test verifies that HLO computations can be compiled and executed
// correctly when using the DeviceAddressVmmAllocator.

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/array2d.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;

class CommandBufferPjrtClientTest : public ::testing::Test {
 protected:
  void SetUp() override {
    GpuAllocatorConfig gpu_config;
    gpu_config.kind = GpuAllocatorConfig::Kind::kVmm;
    gpu_config.preallocate = false;  // VMM manages memory differently

    GpuClientOptions options;
    options.allocator_config = std::move(gpu_config);

    auto client_or = GetXlaPjrtGpuClient(options);
    if (!client_or.ok()) {
      GTEST_SKIP() << "GPU client not available: " << client_or.status();
    }
    client_ = std::move(client_or.value());

    if (client_->devices().empty()) {
      GTEST_SKIP() << "No GPU devices available";
    }

    TF_ASSERT_OK_AND_ASSIGN(
        memory_space_,
        client_->addressable_devices()[0]->default_memory_space());
  }

  // Helper to compile HLO text to an executable.
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileHlo(
      const char* hlo_text) {
    TF_ASSIGN_OR_RETURN(auto hlo_module,
                        ParseAndReturnUnverifiedModule(hlo_text, {}));
    XlaComputation computation(hlo_module->ToProto());
    CompileOptions options;
    // Disable autotuning to avoid backend-specific failures in test runs.
    options.executable_build_options.mutable_debug_options()
        ->set_xla_gpu_autotune_level(0);
    options.env_option_overrides.emplace_back(
        "xla_gpu_enable_command_buffer_va_remapping", true);
    options.env_option_overrides.emplace_back("xla_gpu_graph_min_graph_size",
                                              int64_t{1});
    return client_->CompileAndLoad(computation, options);
  }

  std::unique_ptr<PjRtClient> client_;
  PjRtMemorySpace* memory_space_ = nullptr;
};

TEST_F(CommandBufferPjrtClientTest, SimpleAdd) {
  const char* hlo_text = R"(
    HloModule simple_add

    ENTRY main {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      ROOT add = f32[4] add(p0, p1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto executable, CompileHlo(hlo_text));

  // Create input literals.
  auto input0 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  auto input1 = LiteralUtil::CreateR1<float>({5.0f, 6.0f, 7.0f, 8.0f});

  // Transfer inputs to device.
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer0,
      client_->BufferFromHostLiteral(input0, memory_space_));
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer1,
      client_->BufferFromHostLiteral(input1, memory_space_));

  // Execute.
  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      executable->Execute({{buffer0.get(), buffer1.get()}}, {}));

  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);

  // Transfer result back and verify.
  TF_ASSERT_OK_AND_ASSIGN(auto result_literal, results[0][0]->ToLiteralSync());

  auto expected = LiteralUtil::CreateR1<float>({6.0f, 8.0f, 10.0f, 12.0f});
  EXPECT_EQ(*result_literal, expected);
}

TEST_F(CommandBufferPjrtClientTest, MatrixMultiply) {
  const char* hlo_text = R"(
    HloModule matmul

    ENTRY main {
      p0 = f32[2,3] parameter(0)
      p1 = f32[3,2] parameter(1)
      ROOT dot = f32[2,2] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto executable, CompileHlo(hlo_text));

  // Create input matrices.
  // p0 = [[1, 2, 3], [4, 5, 6]]
  // p1 = [[1, 2], [3, 4], [5, 6]]
  auto input0 = LiteralUtil::CreateR2<float>({{1, 2, 3}, {4, 5, 6}});
  auto input1 = LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}, {5, 6}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer0,
      client_->BufferFromHostLiteral(input0, memory_space_));
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer1,
      client_->BufferFromHostLiteral(input1, memory_space_));

  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      executable->Execute({{buffer0.get(), buffer1.get()}}, {}));

  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(auto result_literal, results[0][0]->ToLiteralSync());

  // Expected: [[22, 28], [49, 64]]
  auto expected = LiteralUtil::CreateR2<float>({{22, 28}, {49, 64}});
  EXPECT_EQ(*result_literal, expected);
}

TEST_F(CommandBufferPjrtClientTest, LargeAllocation) {
  // Test with a larger allocation to exercise VMM.
  const char* hlo_text = R"(
    HloModule large_alloc

    ENTRY main {
      p0 = f32[1024,1024] parameter(0)
      p1 = f32[1024,1024] parameter(1)
      ROOT add = f32[1024,1024] add(p0, p1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto executable, CompileHlo(hlo_text));

  // Create large input arrays filled with 1s and 2s.
  auto input0 = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>(1024, 1024, 1.0f));
  auto input1 = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>(1024, 1024, 2.0f));

  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer0,
      client_->BufferFromHostLiteral(input0, memory_space_));
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer1,
      client_->BufferFromHostLiteral(input1, memory_space_));

  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      executable->Execute({{buffer0.get(), buffer1.get()}}, {}));

  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(auto result_literal, results[0][0]->ToLiteralSync());

  // All values should be 3.0.
  auto expected = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>(1024, 1024, 3.0f));
  EXPECT_EQ(*result_literal, expected);
}

TEST_F(CommandBufferPjrtClientTest, MultipleExecutions) {
  // Test that multiple executions work correctly with VMM allocator.
  const char* hlo_text = R"(
    HloModule multi_exec

    ENTRY main {
      p0 = f32[4] parameter(0)
      ROOT neg = f32[4] negate(p0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto executable, CompileHlo(hlo_text));

  for (int i = 0; i < 5; ++i) {
    auto input = LiteralUtil::CreateR1<float>(
        {static_cast<float>(i), static_cast<float>(i + 1),
         static_cast<float>(i + 2), static_cast<float>(i + 3)});

    TF_ASSERT_OK_AND_ASSIGN(
        auto buffer,
        client_->BufferFromHostLiteral(input, memory_space_));

    TF_ASSERT_OK_AND_ASSIGN(
        auto results,
        executable->Execute({{buffer.get()}}, {}));

    ASSERT_EQ(results.size(), 1);
    ASSERT_EQ(results[0].size(), 1);

    TF_ASSERT_OK_AND_ASSIGN(auto result_literal, results[0][0]->ToLiteralSync());

    std::vector<float> expected_data = {
        static_cast<float>(-i), static_cast<float>(-(i + 1)),
        static_cast<float>(-(i + 2)), static_cast<float>(-(i + 3))};
    if (i == 0) {
      expected_data[0] = -0.0f;
    }
    auto expected = LiteralUtil::CreateR1<float>(expected_data);
    EXPECT_EQ(*result_literal, expected);
  }
}

TEST_F(CommandBufferPjrtClientTest, ReduceSum) {
  const char* hlo_text = R"(
    HloModule reduce_sum

    add {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT sum = f32[] add(x, y)
    }

    ENTRY main {
      p0 = f32[100] parameter(0)
      init = f32[] constant(0)
      ROOT reduce = f32[] reduce(p0, init), dimensions={0}, to_apply=add
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto executable, CompileHlo(hlo_text));

  // Create input array [1, 2, 3, ..., 100].
  std::vector<float> data(100);
  for (int i = 0; i < 100; ++i) {
    data[i] = static_cast<float>(i + 1);
  }
  auto input = LiteralUtil::CreateR1<float>(data);

  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client_->BufferFromHostLiteral(input, memory_space_));

  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      executable->Execute({{buffer.get()}}, {}));

  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(auto result_literal, results[0][0]->ToLiteralSync());

  // Sum of 1 to 100 = 5050.
  auto expected = LiteralUtil::CreateR0<float>(5050.0f);
  EXPECT_EQ(*result_literal, expected);
}

TEST_F(CommandBufferPjrtClientTest, Broadcast) {
  const char* hlo_text = R"(
    HloModule broadcast

    ENTRY main {
      p0 = f32[] parameter(0)
      ROOT broadcast = f32[3,4] broadcast(p0), dimensions={}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto executable, CompileHlo(hlo_text));

  auto input = LiteralUtil::CreateR0<float>(42.0f);

  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client_->BufferFromHostLiteral(input, memory_space_));

  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      executable->Execute({{buffer.get()}}, {}));

  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(auto result_literal, results[0][0]->ToLiteralSync());

  auto expected = LiteralUtil::CreateR2<float>(
      {{42, 42, 42, 42}, {42, 42, 42, 42}, {42, 42, 42, 42}});
  EXPECT_EQ(*result_literal, expected);
}

// Test VA remapping with multiple iterations where each iteration has different
// buffer addresses. This verifies the VA remapping mechanism correctly handles
// changing physical memory addresses.
TEST_F(CommandBufferPjrtClientTest, VaRemappingWithDifferentBufferAddresses) {
  const char* hlo_text = R"(
    HloModule va_remap_test

    ENTRY main {
      p0 = f32[256] parameter(0)
      p1 = f32[256] parameter(1)
      ROOT add = f32[256] add(p0, p1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto executable, CompileHlo(hlo_text));

  constexpr int kNumIterations = 20;

  for (int iter = 0; iter < kNumIterations; ++iter) {
    SCOPED_TRACE(absl::StrCat("Iteration ", iter));

    // Create input data that varies by iteration.
    std::vector<float> data0(256), data1(256);
    for (int i = 0; i < 256; ++i) {
      data0[i] = static_cast<float>(i + iter);
      data1[i] = static_cast<float>(i * 2 + iter);
    }
    auto input0 = LiteralUtil::CreateR1<float>(data0);
    auto input1 = LiteralUtil::CreateR1<float>(data1);

    TF_ASSERT_OK_AND_ASSIGN(
        auto buffer0,
        client_->BufferFromHostLiteral(input0, memory_space_));
    TF_ASSERT_OK_AND_ASSIGN(
        auto buffer1,
        client_->BufferFromHostLiteral(input1, memory_space_));

    // Execute computation.
    TF_ASSERT_OK_AND_ASSIGN(
        auto results,
        executable->Execute({{buffer0.get(), buffer1.get()}}, {}));

    ASSERT_EQ(results.size(), 1);
    ASSERT_EQ(results[0].size(), 1);

    TF_ASSERT_OK_AND_ASSIGN(auto result_literal, results[0][0]->ToLiteralSync());

    // Compute expected result.
    std::vector<float> expected_data(256);
    for (int i = 0; i < 256; ++i) {
      expected_data[i] = data0[i] + data1[i];
    }
    auto expected = LiteralUtil::CreateR1<float>(expected_data);
    EXPECT_EQ(*result_literal, expected)
        << "Mismatch at iteration " << iter;

    // Buffers go out of scope and are freed here.
  }
}

// Test that the same executable can be run multiple times with completely
// fresh buffer allocations each time, verifying VA remapping handles
// repeated map/unmap cycles correctly.
TEST_F(CommandBufferPjrtClientTest, RepeatedExecutionsWithFreshBuffers) {
  const char* hlo_text = R"(
    HloModule repeated_exec

    ENTRY main {
      p0 = f32[512] parameter(0)
      c1 = f32[] constant(2.0)
      bcast = f32[512] broadcast(c1), dimensions={}
      ROOT mul = f32[512] multiply(p0, bcast)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto executable, CompileHlo(hlo_text));

  constexpr int kNumIterations = 50;

  for (int iter = 0; iter < kNumIterations; ++iter) {
    SCOPED_TRACE(absl::StrCat("Iteration ", iter));

    // Create fresh input data each iteration.
    std::vector<float> input_data(512);
    for (int i = 0; i < 512; ++i) {
      input_data[i] = static_cast<float>(i + iter * 100);
    }
    auto input = LiteralUtil::CreateR1<float>(input_data);

    // Create fresh buffer each iteration - this may get different addresses.
    TF_ASSERT_OK_AND_ASSIGN(
        auto buffer,
        client_->BufferFromHostLiteral(input, memory_space_));

    // Execute.
    TF_ASSERT_OK_AND_ASSIGN(
        auto results,
        executable->Execute({{buffer.get()}}, {}));

    ASSERT_EQ(results.size(), 1);
    ASSERT_EQ(results[0].size(), 1);

    TF_ASSERT_OK_AND_ASSIGN(auto result_literal, results[0][0]->ToLiteralSync());

    // Verify result: each element should be doubled.
    std::vector<float> expected_data(512);
    for (int i = 0; i < 512; ++i) {
      expected_data[i] = input_data[i] * 2.0f;
    }
    auto expected = LiteralUtil::CreateR1<float>(expected_data);
    EXPECT_EQ(*result_literal, expected)
        << "Mismatch at iteration " << iter;
  }
}

// Test many executions with negate operation to verify VA remapping
// handles repeated map/unmap cycles correctly over many iterations.
TEST_F(CommandBufferPjrtClientTest, ManyExecutionsNegate) {
  const char* hlo_text = R"(
    HloModule many_exec_negate

    ENTRY main {
      p0 = f32[128] parameter(0)
      ROOT neg = f32[128] negate(p0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto executable, CompileHlo(hlo_text));

  constexpr int kNumIterations = 100;

  for (int iter = 0; iter < kNumIterations; ++iter) {
    SCOPED_TRACE(absl::StrCat("Iteration ", iter));

    // Create input for this iteration with unique values.
    std::vector<float> input_data(128);
    for (int i = 0; i < 128; ++i) {
      input_data[i] = static_cast<float>(i + iter * 10);
    }
    auto input = LiteralUtil::CreateR1<float>(input_data);

    TF_ASSERT_OK_AND_ASSIGN(
        auto buffer,
        client_->BufferFromHostLiteral(input, memory_space_));

    // Execute.
    TF_ASSERT_OK_AND_ASSIGN(
        auto results,
        executable->Execute({{buffer.get()}}, {}));

    ASSERT_EQ(results.size(), 1);
    ASSERT_EQ(results[0].size(), 1);

    TF_ASSERT_OK_AND_ASSIGN(auto result_literal, results[0][0]->ToLiteralSync());

    // Verify result: negate each element.
    std::vector<float> expected_data(128);
    for (int i = 0; i < 128; ++i) {
      expected_data[i] = -input_data[i];
    }
    auto expected = LiteralUtil::CreateR1<float>(expected_data);
    EXPECT_EQ(*result_literal, expected)
        << "Mismatch at iteration " << iter;
  }
}

}  // namespace
}  // namespace xla
