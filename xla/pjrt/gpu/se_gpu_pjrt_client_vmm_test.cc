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

// End-to-end tests for GpuAllocatorConfig::Kind::kVmm.
// Verifies that HLO computations compile and execute correctly when the default
// GPU memory allocator is set to the VMM (Virtual Memory Management) allocator.

#include <cstdint>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/stream_executor/cuda/cuda_device_address_vmm_allocator.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

// Creates a PJRT GPU client configured to use the VMM allocator on device 0.
absl::StatusOr<std::unique_ptr<PjRtClient>> CreateVmmClient() {
  GpuAllocatorConfig allocator_config;
  allocator_config.kind = GpuAllocatorConfig::Kind::kVmm;
  allocator_config.preallocate = false;

  GpuClientOptions options;
  options.allocator_config = allocator_config;
  options.allowed_devices = std::set<int>({0});
  return GetStreamExecutorGpuClient(options);
}

// Compiles the given HLO text on `client` and returns the loaded executable.
absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileHlo(
    PjRtClient& client, const char* hlo_text) {
  TF_ASSIGN_OR_RETURN(auto hlo_module,
                      ParseAndReturnUnverifiedModule(hlo_text, {}));
  XlaComputation computation(hlo_module->ToProto());
  CompileOptions compile_options;
  compile_options.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_autotune_level(0);
  return client.CompileAndLoad(computation, compile_options);
}

// Transfers host literals to device, executes, and returns the result literal.
// `inputs` is a span of pointers to avoid copying non-copyable Literals.
absl::StatusOr<std::shared_ptr<Literal>> RunWithInputs(
    PjRtLoadedExecutable& executable, PjRtClient& client,
    PjRtMemorySpace& memory_space, absl::Span<const Literal* const> inputs) {
  std::vector<std::unique_ptr<PjRtBuffer>> input_buffers;
  input_buffers.reserve(inputs.size());
  for (const Literal* input : inputs) {
    TF_ASSIGN_OR_RETURN(auto buf,
                        client.BufferFromHostLiteral(*input, &memory_space));
    input_buffers.push_back(std::move(buf));
  }

  std::vector<PjRtBuffer*> raw_ptrs;
  raw_ptrs.reserve(input_buffers.size());
  for (auto& buf : input_buffers) raw_ptrs.push_back(buf.get());

  TF_ASSIGN_OR_RETURN(auto results,
                      executable.Execute({raw_ptrs}, /*options=*/{}));
  TF_RET_CHECK(results.size() == 1);
  TF_RET_CHECK(results[0].size() == 1);
  TF_ASSIGN_OR_RETURN(auto literal, results[0][0]->ToLiteral().Await());
  return literal;
}

class VmmAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto client_or = CreateVmmClient();
    if (!client_or.ok()) {
      GTEST_SKIP() << "VMM allocator not available: " << client_or.status();
    }
    client_ = std::move(client_or.value());

    if (client_->addressable_devices().empty()) {
      GTEST_SKIP() << "No addressable GPU devices available.";
    }
    memory_space_ =
        client_->addressable_devices()[0]->default_memory_space().value();
  }

  std::unique_ptr<PjRtClient> client_;
  PjRtMemorySpace* memory_space_ = nullptr;
};

// ---- Tests ------------------------------------------------------------------

// Verifies that the client is using DeviceAddressVmmAllocator when configured
// with kVmm allocator kind.
TEST_F(VmmAllocatorTest, AllocatorIsDeviceAddressVmmAllocator) {
  auto* se_client = dynamic_cast<PjRtStreamExecutorClient*>(client_.get());
  ASSERT_NE(se_client, nullptr);
  se::DeviceAddressAllocator* allocator = se_client->allocator();
  ASSERT_NE(allocator, nullptr);
  EXPECT_NE(dynamic_cast<se::DeviceAddressVmmAllocator*>(allocator), nullptr)
      << "Expected DeviceAddressVmmAllocator but got a different allocator "
         "type";
}

TEST_F(VmmAllocatorTest, VectorAdd) {
  constexpr char kHlo[] = R"(
    HloModule vector_add
    ENTRY main {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      ROOT add = f32[4] add(p0, p1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto executable, CompileHlo(*client_, kHlo));

  auto input0 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  auto input1 = LiteralUtil::CreateR1<float>({10.0f, 20.0f, 30.0f, 40.0f});
  const Literal* inputs[] = {&input0, &input1};
  TF_ASSERT_OK_AND_ASSIGN(auto result, RunWithInputs(*executable, *client_,
                                                     *memory_space_, inputs));

  auto expected = LiteralUtil::CreateR1<float>({11.0f, 22.0f, 33.0f, 44.0f});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, *result));
}

TEST_F(VmmAllocatorTest, VectorMultiply) {
  constexpr char kHlo[] = R"(
    HloModule vector_multiply
    ENTRY main {
      p0 = f32[8] parameter(0)
      p1 = f32[8] parameter(1)
      ROOT mul = f32[8] multiply(p0, p1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto executable, CompileHlo(*client_, kHlo));

  auto input0 = LiteralUtil::CreateR1<float>({1, 2, 3, 4, 5, 6, 7, 8});
  auto input1 = LiteralUtil::CreateR1<float>({2, 2, 2, 2, 2, 2, 2, 2});
  const Literal* inputs[] = {&input0, &input1};
  TF_ASSERT_OK_AND_ASSIGN(auto result, RunWithInputs(*executable, *client_,
                                                     *memory_space_, inputs));

  auto expected = LiteralUtil::CreateR1<float>({2, 4, 6, 8, 10, 12, 14, 16});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, *result));
}

TEST_F(VmmAllocatorTest, MatrixAdd) {
  constexpr char kHlo[] = R"(
    HloModule matrix_add
    ENTRY main {
      p0 = f32[4,4] parameter(0)
      p1 = f32[4,4] parameter(1)
      ROOT add = f32[4,4] add(p0, p1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto executable, CompileHlo(*client_, kHlo));

  auto input0 = LiteralUtil::CreateR2<float>(
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
  auto input1 = LiteralUtil::CreateR2<float>(
      {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}});
  const Literal* inputs[] = {&input0, &input1};
  TF_ASSERT_OK_AND_ASSIGN(auto result, RunWithInputs(*executable, *client_,
                                                     *memory_space_, inputs));

  auto expected = LiteralUtil::CreateR2<float>(
      {{2, 3, 4, 5}, {6, 7, 8, 9}, {10, 11, 12, 13}, {14, 15, 16, 17}});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, *result));
}

TEST_F(VmmAllocatorTest, ScalarReduce) {
  constexpr char kHlo[] = R"(
    HloModule scalar_reduce
    add {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT add = f32[] add(a, b)
    }
    ENTRY main {
      p0 = f32[16] parameter(0)
      zero = f32[] constant(0)
      ROOT sum = f32[] reduce(p0, zero), dimensions={0}, to_apply=add
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto executable, CompileHlo(*client_, kHlo));

  auto input = LiteralUtil::CreateR1<float>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  const Literal* inputs[] = {&input};
  TF_ASSERT_OK_AND_ASSIGN(auto result, RunWithInputs(*executable, *client_,
                                                     *memory_space_, inputs));

  auto expected = LiteralUtil::CreateR0<float>(136.0f);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, *result));
}

// Runs the same executable multiple times to exercise repeated allocation and
// deallocation through the VMM allocator.
TEST_F(VmmAllocatorTest, MultipleExecutions) {
  constexpr char kHlo[] = R"(
    HloModule negate
    ENTRY main {
      p0 = f32[64] parameter(0)
      ROOT neg = f32[64] negate(p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto executable, CompileHlo(*client_, kHlo));

  constexpr int kNumIterations = 10;
  for (int iter = 0; iter < kNumIterations; ++iter) {
    std::vector<float> data(64);
    for (int i = 0; i < 64; ++i) data[i] = static_cast<float>(i + iter * 100);

    auto input = LiteralUtil::CreateR1<float>(data);
    const Literal* inputs[] = {&input};
    TF_ASSERT_OK_AND_ASSIGN(auto result, RunWithInputs(*executable, *client_,
                                                       *memory_space_, inputs));

    std::vector<float> expected_data(64);
    for (int i = 0; i < 64; ++i) expected_data[i] = -data[i];
    auto expected = LiteralUtil::CreateR1<float>(expected_data);
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, *result))
        << "Mismatch at iteration " << iter;
  }
}

// Exercises a larger allocation to verify the VMM allocator handles bigger
// buffers correctly.
TEST_F(VmmAllocatorTest, LargeAllocation) {
  constexpr char kHlo[] = R"(
    HloModule large_add
    ENTRY main {
      p0 = f32[1048576] parameter(0)
      p1 = f32[1048576] parameter(1)
      ROOT add = f32[1048576] add(p0, p1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto executable, CompileHlo(*client_, kHlo));

  constexpr int kN = 1048576;
  std::vector<float> data0(kN), data1(kN);
  for (int i = 0; i < kN; ++i) {
    data0[i] = static_cast<float>(i);
    data1[i] = static_cast<float>(kN - i);
  }
  auto input0 = LiteralUtil::CreateR1<float>(data0);
  auto input1 = LiteralUtil::CreateR1<float>(data1);
  const Literal* inputs[] = {&input0, &input1};
  TF_ASSERT_OK_AND_ASSIGN(auto result, RunWithInputs(*executable, *client_,
                                                     *memory_space_, inputs));

  // Every element should be kN.
  std::vector<float> expected_data(kN, static_cast<float>(kN));
  auto expected = LiteralUtil::CreateR1<float>(expected_data);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, *result));
}

// Runs two different executables sequentially to verify that the VMM allocator
// correctly handles interleaved allocations from different computations.
TEST_F(VmmAllocatorTest, SequentialDifferentExecutables) {
  constexpr char kHloAdd[] = R"(
    HloModule add
    ENTRY main {
      p0 = f32[32] parameter(0)
      p1 = f32[32] parameter(1)
      ROOT add = f32[32] add(p0, p1)
    }
  )";
  constexpr char kHloMul[] = R"(
    HloModule mul
    ENTRY main {
      p0 = f32[32] parameter(0)
      p1 = f32[32] parameter(1)
      ROOT mul = f32[32] multiply(p0, p1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto exe_add, CompileHlo(*client_, kHloAdd));
  TF_ASSERT_OK_AND_ASSIGN(auto exe_mul, CompileHlo(*client_, kHloMul));

  std::vector<float> a(32), b(32);
  for (int i = 0; i < 32; ++i) {
    a[i] = static_cast<float>(i + 1);
    b[i] = 2.0f;
  }
  auto lit_a = LiteralUtil::CreateR1<float>(a);
  auto lit_b = LiteralUtil::CreateR1<float>(b);
  const Literal* inputs[] = {&lit_a, &lit_b};

  // add: result[i] = a[i] + b[i]
  TF_ASSERT_OK_AND_ASSIGN(
      auto add_result,
      RunWithInputs(*exe_add, *client_, *memory_space_, inputs));
  std::vector<float> expected_add(32);
  for (int i = 0; i < 32; ++i) expected_add[i] = a[i] + b[i];
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<float>(expected_add),
                                     *add_result));

  // mul: result[i] = a[i] * b[i]
  TF_ASSERT_OK_AND_ASSIGN(
      auto mul_result,
      RunWithInputs(*exe_mul, *client_, *memory_space_, inputs));
  std::vector<float> expected_mul(32);
  for (int i = 0; i < 32; ++i) expected_mul[i] = a[i] * b[i];
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<float>(expected_mul),
                                     *mul_result));
}

}  // namespace
}  // namespace xla
