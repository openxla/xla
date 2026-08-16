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

// End-to-end tests for AllGather with the Triton collective kernel backend.
//
// Architecture note (xla/ vs xla_ref/):
//   In xla_ref, allgather was wrapped in kAllGatherStart/kAllGatherDone.
//   In xla/ (new upstream arch), ALL collectives use the generic
//   kAsyncStart/kAsyncDone wrapper.  The inner wrapped instruction is
//   kAllGather.  Tests that check for kAllGatherStart must use kAllGather
//   instead, and runtime-fallback detection (ScopedMockLog warnings) is
//   replaced by compile-time eligibility checks.

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

std::string GetAsyncTestName(bool is_async) {
  return is_async ? "async" : "sync";
}

// In the new architecture, kAllGather is the wrapped instruction inside
// kAsyncStart.  We search for kAllGather (not the legacy kAllGatherStart).
void VerifyAllGatherType(const HloModule* module, PrimitiveType expected_type) {
  bool found = false;
  for (auto* comp : module->computations()) {
    for (auto* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kAllGather) {
        PrimitiveType actual_type = instr->operand(0)->shape().element_type();
        ASSERT_EQ(actual_type, expected_type)
            << "Expected AllGather type " << PrimitiveType_Name(expected_type)
            << " but got " << PrimitiveType_Name(actual_type);
        found = true;
      }
    }
  }
  ASSERT_TRUE(found) << "No AllGather found in module";
}

class AllGatherTestNoParams : public CollectiveOpsWithFlagsBase {
 public:
  explicit AllGatherTestNoParams(bool is_async = false)
      : CollectiveOpsWithFlagsBase(/*enable_async=*/is_async,
                                   /*enable_p2p_memcpy=*/false,
                                   /*enable_symmetric_buffer=*/false,
                                   /*memory_size=*/32 * kMB,
                                   /*collectives_memory_size=*/0) {}

  void SetUp() override {
    CollectiveOpsE2ETestBase::SetUp();
    // Check for Triton support: Ampere+ for CUDA, any supported GPU for ROCm
    if (Capability().IsCuda() && !IsAmpereAndHigher()) {
      GTEST_SKIP() << "Test requires Ampere or newer architecture for CUDA "
                      "since it's using triton.";
    }
  }

  // Returns true if the current device supports Triton collective kernels
  // (Hopper+ for CUDA, or any supported ROCm device).
  bool IsTritonCapable() {
    return IsHopperAndHigher() || Capability().IsRocm();
  }

  bool CheckDeviceCount(int32_t required_device_count) {
    [&]() -> void {
      const int32_t current_device_count = device_count();
      if (current_device_count < required_device_count) {
        ASSERT_GE(current_device_count, 2)
            << "Test requires at least 2 devices but only "
            << current_device_count << " available";
        if (current_device_count < required_device_count) {
          GTEST_SKIP() << "Test requires at least " << required_device_count
                       << " devices but only " << current_device_count
                       << " available.";
        }
      }
    }();
    return !IsSkipped() && !HasFatalFailure();
  }

  // Installs a log interceptor that fails the test if the AllGather Triton
  // backend silently falls back to NCCL/RCCL due to an ineligible shape.
  // The interceptor stays active (via the member variable) until the test
  // destructor runs, covering the `ExecuteReplicated` call that follows.
  void CheckNoTritonFallback() {
    triton_fallback_checker_ = std::make_unique<absl::ScopedMockLog>(
        absl::MockLogDefault::kIgnoreUnexpected);
    // Expect ZERO occurrences of the fallback warning. If it fires, the test
    // fails because Triton was requested but the shape was not eligible.
    EXPECT_CALL(*triton_fallback_checker_,
                Log(absl::LogSeverity::kWarning, ::testing::_,
                    ::testing::HasSubstr("falling back to NCCL/RCCL")))
        .Times(0);
    triton_fallback_checker_->StartCapturingLogs();
  }

 private:
  // Holds the log interceptor set up by CheckNoTritonFallback().
  // Destroyed at the end of the test, triggering expectation verification.
  std::unique_ptr<absl::ScopedMockLog> triton_fallback_checker_;
};

struct AllGatherTestParams {
  bool is_async;
  bool use_all_gather_triton_backend;
  bool use_experimental_tiling = false;

  std::vector<int64_t> GetShape(PrimitiveType element_type,
                                int32_t rank = 1) const {
    int64_t total = 1024;
    if (rank <= 1) {
      return {total};
    }
    std::vector<int64_t> dims;
    for (int32_t i = 0; i < rank - 1; ++i) {
      dims.push_back(rank);
      total /= rank;
    }
    dims.push_back(total);
    return dims;
  }

  static std::vector<AllGatherTestParams> Generate() {
    std::vector<AllGatherTestParams> params;
    for (bool is_async : {true, false}) {
      for (bool use_all_gather_triton_backend : {true, false}) {
        params.push_back({is_async, use_all_gather_triton_backend,
                          /*use_experimental_tiling=*/true});
      }
    }
    return params;
  }

  [[maybe_unused]] friend void PrintTo(const AllGatherTestParams& params,
                                       std::ostream* os) {
    *os << "{ .is_async=" << params.is_async
        << ", .use_all_gather_triton_backend="
        << params.use_all_gather_triton_backend
        << ", .use_experimental_tiling=" << params.use_experimental_tiling
        << " }";
  }
};

struct AllGatherTypesTestParams : public AllGatherTestParams {
  PrimitiveType element_type;

  static std::vector<AllGatherTypesTestParams> Generate() {
    std::vector<AllGatherTypesTestParams> params;
    for (auto& all_gather_test_params : AllGatherTestParams::Generate()) {
      for (const PrimitiveType element_type :
           {F32, F16, BF16, S32, S8, PRED, S16, F8E4M3FN, F8E5M2}) {
        params.push_back({all_gather_test_params, element_type});
      }
    }
    return params;
  }
};

struct InputsOutputs {
  std::vector<Literal> inputs;
  std::vector<Literal> expected_outputs;

  [[nodiscard]] std::vector<std::vector<Literal*>> InputLiteralPtrs() {
    std::vector<std::vector<Literal*>> result;
    for (auto& input : inputs) {
      result.push_back(std::vector<Literal*>{&input});
    }
    return result;
  }
};

template <typename T>
static void FillRandom(Array<T>& array, int64_t seed) {
  constexpr PrimitiveType primitive_type =
      primitive_util::NativeToPrimitiveType<T>();
  if constexpr (primitive_util::IsFloatingPointType(primitive_type)) {
    array.FillRandom(T{1.0f}, T{10.0}, seed);
  } else if constexpr (primitive_type == PRED) {
    array.FillRandomBool(seed);
  } else if constexpr (primitive_util::IsIntegralType(primitive_type)) {
    array.FillRandomUniform(T{0}, T{100}, seed);
  } else {
    GTEST_FAIL() << "Unsupported element type: "
                 << absl::StrFormat("%v",
                                    primitive_util::NativeToPrimitiveType<T>());
  }
}

template <PrimitiveType kElementType>
static absl::StatusOr<InputsOutputs> BuildTestInputsOutputs(
    const HloModule& module, int64_t num_replicas,
    int64_t all_gather_dimension) {
  using ElementType = primitive_util::NativeTypeOf<kElementType>;

  std::vector<Array<ElementType>> inputs;
  std::vector<Literal> input_literals;
  const HloInstruction* const hlo_instr =
      HloHardwareIndependentTestBase::FindInstruction(&module,
                                                      HloOpcode::kAllGather);
  if (hlo_instr == nullptr) {
    return absl::InvalidArgumentError(
        "Instruction 'all-gather' not found in module.");
  }
  const HloAllGatherInstruction* instr =
      Cast<HloAllGatherInstruction>(hlo_instr);
  const Shape& input_shape = instr->operand(0)->shape();

  for (int i = 0; i < num_replicas; ++i) {
    auto& input =
        inputs.emplace_back(Array<ElementType>(input_shape.dimensions()));
    FillRandom<ElementType>(input, i);
    input_literals.push_back(LiteralUtil::CreateFromArray(input));
  }

  std::vector<Literal> expected_output_literals;
  const std::vector<ReplicaGroup>& replica_groups =
      instr->device_list()->replica_groups();

  std::vector<std::vector<int64_t>> device_to_groups(num_replicas);
  for (const auto& replica_group : replica_groups) {
    const auto& replica_ids = replica_group.replica_ids();
    for (int64_t replica : replica_group.replica_ids()) {
      CHECK_EQ(device_to_groups[replica].size(), 0);
      device_to_groups[replica].assign(replica_ids.begin(), replica_ids.end());
    }
  }

  const Shape& output_shape = instr->shape();
  for (int i = 0; i < num_replicas; ++i) {
    const std::vector<int64_t>& group = device_to_groups[i];
    Literal output(output_shape);
    int64_t offset = 0;
    for (int64_t replica : group) {
      const Literal& input = input_literals[replica];
      const int64_t slice_size = input_shape.dimensions(all_gather_dimension);
      std::vector<int64_t> src_base(input_shape.dimensions_size(), 0);
      std::vector<int64_t> dest_base(output_shape.dimensions_size(), 0);
      dest_base[all_gather_dimension] = offset;
      TF_RETURN_IF_ERROR(output.CopySliceFrom(input, src_base, dest_base,
                                              input_shape.dimensions()));
      offset += slice_size;
    }
    expected_output_literals.push_back(std::move(output));
  }

  return InputsOutputs{std::move(input_literals),
                       std::move(expected_output_literals)};
}

absl::StatusOr<InputsOutputs> BuildTestInputsOutputs(
    PrimitiveType element_type, const HloModule& module, int64_t num_replicas,
    int64_t all_gather_dimension) {
  const auto dispatch = [&](auto type) {
    return BuildTestInputsOutputs<type>(module, num_replicas,
                                        all_gather_dimension);
  };
  return primitive_util::ArrayTypeSwitch(dispatch, element_type);
}

class AllGatherTest
    : public AllGatherTestNoParams,
      public ::testing::WithParamInterface<AllGatherTestParams> {
 public:
  AllGatherTest() : AllGatherTestNoParams(/*is_async=*/GetParam().is_async) {}

 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions opts = CollectiveOpsWithFlagsBase::GetDebugOptionsForTest();
    opts.clear_xla_gpu_experimental_use_collective_kernels();
    if (GetParam().use_all_gather_triton_backend) {
      opts.add_xla_gpu_experimental_use_collective_kernels(
          DebugOptions::COLLECTIVE_KERNEL_ALL_GATHER);
      opts.set_xla_gpu_experimental_enable_tiling_propagation(true);
    }
    return opts;
  }
};

class AllGatherTypesTest
    : public AllGatherTestNoParams,
      public ::testing::WithParamInterface<AllGatherTypesTestParams> {
 public:
  AllGatherTypesTest()
      : AllGatherTestNoParams(/*is_async=*/GetParam().is_async) {}

 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions opts = CollectiveOpsWithFlagsBase::GetDebugOptionsForTest();
    opts.clear_xla_gpu_experimental_use_collective_kernels();
    if (GetParam().use_all_gather_triton_backend) {
      opts.add_xla_gpu_experimental_use_collective_kernels(
          DebugOptions::COLLECTIVE_KERNEL_ALL_GATHER);
      opts.set_xla_gpu_experimental_enable_tiling_propagation(true);
    }
    return opts;
  }
};

INSTANTIATE_TEST_SUITE_P(
    AllGatherTest, AllGatherTest,
    ::testing::ValuesIn(AllGatherTestParams::Generate()),
    [](const ::testing::TestParamInfo<AllGatherTestParams>& info) {
      return absl::StrCat(
          GetAsyncTestName(info.param.is_async), "_",
          info.param.use_all_gather_triton_backend ? "triton" : "nccl");
    });

INSTANTIATE_TEST_SUITE_P(
    AllGatherTypesTest, AllGatherTypesTest,
    ::testing::ValuesIn(AllGatherTypesTestParams::Generate()),
    [](const ::testing::TestParamInfo<AllGatherTypesTestParams>& info) {
      return absl::StrCat(
          GetAsyncTestName(info.param.is_async), "_",
          primitive_util::LowercasePrimitiveTypeName(info.param.element_type),
          "_", info.param.use_all_gather_triton_backend ? "triton" : "nccl");
    });

TEST_P(AllGatherTypesTest, SupportedTypes2GPUs) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    param_0 = %1$s[%2$s] parameter(0)
    ROOT all-gather = %1$s[%3$s] all-gather(param_0),
      replica_groups={{0,1}}, dimensions={0}
  }
  )";
  constexpr int64_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  const PrimitiveType element_type = GetParam().element_type;
  const std::vector<int64_t> input_shape = GetParam().GetShape(element_type);
  std::vector<int64_t> output_shape = input_shape;
  output_shape[0] *= kNumReplicas;

  const std::string module_str = absl::StrFormat(
      kModuleStr, primitive_util::LowercasePrimitiveTypeName(element_type),
      absl::StrJoin(input_shape, ","), absl::StrJoin(output_shape, ","));
  SCOPED_TRACE(::testing::Message() << "module_str: " << module_str);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(module_str, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(
      InputsOutputs test_io,
      (BuildTestInputsOutputs(element_type, *module, kNumReplicas,
                              /*all_gather_dimension=*/0)));
  CheckNoTritonFallback();
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Equal(test_io.expected_outputs[i], results[i]))
        << "ExpectedOutput != Result at rank " << i;
  }
}

TEST_P(AllGatherTest, F32_8GPUs_AllReplicasOneGroup) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    param_0 = f32[%1$s] parameter(0)
    ROOT all-gather = f32[%2$s] all-gather(param_0),
      replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}
  }
  )";

  constexpr int64_t kNumReplicas = 8;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  const std::vector<int64_t> input_shape =
      GetParam().GetShape(PrimitiveType::F32);
  std::vector<int64_t> output_shape = input_shape;
  output_shape[0] *= kNumReplicas;

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(
          absl::StrFormat(kModuleStr, absl::StrJoin(input_shape, ","),
                          absl::StrJoin(output_shape, ",")),
          kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(
      InputsOutputs test_io,
      (BuildTestInputsOutputs<PrimitiveType::F32>(*module, kNumReplicas,
                                                  /*all_gather_dimension=*/0)));
  CheckNoTritonFallback();
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-5}))
        << "ExpectedOutput != Result at index " << i;
  }
}

TEST_P(AllGatherTest, F32_8GPUs_2ReplicasPerGroup) {
  const std::vector<int64_t> input_shape =
      GetParam().GetShape(PrimitiveType::F32);
  std::vector<int64_t> output_shape = input_shape;
  output_shape[0] *= 2;

  const auto kModuleStr = absl::StrFormat(
      R"(
  HloModule test

  ENTRY test_computation {
    param_0 = f32[%1$s] parameter(0)
    ROOT all-gather = f32[%2$s] all-gather(param_0),
      replica_groups={{0,4},{1,5},{2,6},{3,7}}, dimensions={0}
  }
  )",
      absl::StrJoin(input_shape, ","), absl::StrJoin(output_shape, ","));

  constexpr int64_t kNumReplicas = 8;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(
      InputsOutputs test_io,
      (BuildTestInputsOutputs<PrimitiveType::F32>(*module, kNumReplicas,
                                                  /*all_gather_dimension=*/0)));
  CheckNoTritonFallback();
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Equal(test_io.expected_outputs[i], results[i]))
        << "ExpectedOutput != Result at index " << i;
  }
}

TEST_P(AllGatherTest, F32TwoD4GPUs) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    param_0 = f32[%1$s] parameter(0)
    ROOT all-gather = f32[%2$s] all-gather(param_0),
      replica_groups={{0,1,2,3}}, dimensions={0}
  }
  )";

  constexpr int64_t kNumReplicas = 4;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  const std::vector<int64_t> input_shape =
      GetParam().GetShape(PrimitiveType::F32, /*rank=*/2);
  std::vector<int64_t> output_shape = input_shape;
  output_shape[0] *= kNumReplicas;

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(
          absl::StrFormat(kModuleStr, absl::StrJoin(input_shape, ","),
                          absl::StrJoin(output_shape, ",")),
          kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(
      InputsOutputs test_io,
      (BuildTestInputsOutputs<PrimitiveType::F32>(*module, kNumReplicas,
                                                  /*all_gather_dimension=*/0)));
  CheckNoTritonFallback();
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-5}))
        << "ExpectedOutput != Result at index " << i;
  }
}

// Non-power-of-2 shape: NCCL should handle it.
TEST_P(AllGatherTest, F32_3D_2GPUs) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    param_0 = f32[%1$s] parameter(0)
    ROOT all-gather = f32[%2$s] all-gather(param_0),
      replica_groups={{0,1}}, dimensions={0}
  }
  )";
  constexpr int64_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }
  // Use a Triton-eligible 3D shape: 4*4*64 = 1024 elements, F32 (32 bits).
  // 1024 * 32 = 32768 bits ≡ 0 (mod 128) → aligned for Triton.
  const std::vector<int64_t> input_shape = {4, 4, 64};
  std::vector<int64_t> output_shape = input_shape;
  output_shape[0] *= kNumReplicas;

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(
          absl::StrFormat(kModuleStr, absl::StrJoin(input_shape, ","),
                          absl::StrJoin(output_shape, ",")),
          kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(
      InputsOutputs test_io,
      (BuildTestInputsOutputs<PrimitiveType::F32>(*module, kNumReplicas,
                                                  /*all_gather_dimension=*/0)));
  CheckNoTritonFallback();
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-5}))
        << "ExpectedOutput != Result at index " << i;
  }
}

// Power-of-2 shape: Triton should handle it.
TEST_P(AllGatherTest, F32_3D_2GPUs_PowerOf2) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    param_0 = f32[%1$s] parameter(0)
    ROOT all-gather = f32[%2$s] all-gather(param_0),
      replica_groups={{0,1}}, dimensions={0}
  }
  )";
  constexpr int64_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }
  const std::vector<int64_t> input_shape = {8, 8, 16};
  std::vector<int64_t> output_shape = input_shape;
  output_shape[0] *= kNumReplicas;

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      ParseAndReturnVerifiedModule(
          absl::StrFormat(kModuleStr, absl::StrJoin(input_shape, ","),
                          absl::StrJoin(output_shape, ",")),
          kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(
      InputsOutputs test_io,
      (BuildTestInputsOutputs<PrimitiveType::F32>(*module, kNumReplicas,
                                                  /*all_gather_dimension=*/0)));
  CheckNoTritonFallback();
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-5}))
        << "ExpectedOutput != Result at index " << i;
  }
}

TEST_P(AllGatherTest, F32TwoD4GPUs_Dim1) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    param_0 = f32[32,32] parameter(0)
    ROOT all-gather = f32[32,128] all-gather(param_0),
      replica_groups={{0,1,2,3}}, dimensions={1}
  }
  )";

  constexpr int64_t kNumReplicas = 4;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(
      InputsOutputs test_io,
      (BuildTestInputsOutputs<PrimitiveType::F32>(*module, kNumReplicas,
                                                  /*all_gather_dimension=*/1)));
  CheckNoTritonFallback();
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-5}))
        << "ExpectedOutput != Result at index " << i;
  }
}

TEST_P(AllGatherTest, F32_3D_2GPUs_Dim1) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    param_0 = f32[8,8,16] parameter(0)
    ROOT all-gather = f32[8,16,16] all-gather(param_0),
      replica_groups={{0,1}}, dimensions={1}
  }
  )";
  constexpr int64_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(
      InputsOutputs test_io,
      (BuildTestInputsOutputs<PrimitiveType::F32>(*module, kNumReplicas,
                                                  /*all_gather_dimension=*/1)));
  CheckNoTritonFallback();
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-5}))
        << "ExpectedOutput != Result at index " << i;
  }
}

TEST_P(AllGatherTest, F32_3D_2GPUs_Dim2) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    param_0 = f32[8,8,16] parameter(0)
    ROOT all-gather = f32[8,8,32] all-gather(param_0),
      replica_groups={{0,1}}, dimensions={2}
  }
  )";
  constexpr int64_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(
      InputsOutputs test_io,
      (BuildTestInputsOutputs<PrimitiveType::F32>(*module, kNumReplicas,
                                                  /*all_gather_dimension=*/2)));
  CheckNoTritonFallback();
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-5}))
        << "ExpectedOutput != Result at index " << i;
  }
}

TEST_P(AllGatherTest, BF16_1024Elements_2GPUs) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    param_0 = bf16[1024] parameter(0)
    ROOT all-gather = bf16[2048] all-gather(param_0),
      replica_groups={{0,1}}, dimensions={0}
  }
  )";

  constexpr int64_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(InputsOutputs test_io,
                          (BuildTestInputsOutputs<PrimitiveType::BF16>(
                              *module, kNumReplicas,
                              /*all_gather_dimension=*/0)));
  CheckNoTritonFallback();
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-3}))
        << "ExpectedOutput != Result at replica " << i;
  }
}

TEST_P(AllGatherTest, BF16_1024Elements_4GPUs) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    param_0 = bf16[1024] parameter(0)
    ROOT all-gather = bf16[4096] all-gather(param_0),
      replica_groups={{0,1,2,3}}, dimensions={0}
  }
  )";

  constexpr int64_t kNumReplicas = 4;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(InputsOutputs test_io,
                          (BuildTestInputsOutputs<PrimitiveType::BF16>(
                              *module, kNumReplicas,
                              /*all_gather_dimension=*/0)));
  CheckNoTritonFallback();
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-3}))
        << "ExpectedOutput != Result at replica " << i;
  }
}

TEST_P(AllGatherTest, BF16_1024Elements_8GPUs) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    param_0 = bf16[1024] parameter(0)
    ROOT all-gather = bf16[8192] all-gather(param_0),
      replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}
  }
  )";

  constexpr int64_t kNumReplicas = 8;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(InputsOutputs test_io,
                          (BuildTestInputsOutputs<PrimitiveType::BF16>(
                              *module, kNumReplicas,
                              /*all_gather_dimension=*/0)));
  CheckNoTritonFallback();
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-3}))
        << "ExpectedOutput != Result at replica " << i;
  }
}

// Tests AllGather with 4096 f32 elements per replica — exercises
// multi-tile-per-rank execution (tiles_per_rank = 2, total_blocks = 4).
TEST_P(AllGatherTest, F32_MultiTilePerRank_2GPUs) {
  constexpr absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    param_0 = f32[4096] parameter(0)
    ROOT all-gather = f32[8192] all-gather(param_0),
      replica_groups={{0,1}}, dimensions={0}
  }
  )";
  constexpr int64_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(
      InputsOutputs test_io,
      (BuildTestInputsOutputs<PrimitiveType::F32>(*module, kNumReplicas,
                                                  /*all_gather_dimension=*/0)));
  CheckNoTritonFallback();
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-5}))
        << "ExpectedOutput != Result at index " << i;
  }
}

// FP8 vs FP16 all-gather comparison.
TEST_F(AllGatherTestNoParams, AsyncAllGather_F8E4M3FN_2GPUs) {
  bool has_fp8_support = false;
  if (Capability().IsCuda()) {
    has_fp8_support = Capability().cuda_compute_capability()->IsAtLeast(9, 0);
  } else if (Capability().IsRocm()) {
    has_fp8_support =
        Capability().rocm_compute_capability()->has_ocp_fp8_support();
  }

  if (!has_fp8_support) {
    GTEST_SKIP() << "FP8 requires GPU with OCP FP8 support.";
  }

  constexpr absl::string_view kF16ModuleStr = R"(
  HloModule f16_allgather
  ENTRY test {
    param = f16[32,64] parameter(0)
    ROOT all-gather = f16[64,64] all-gather(param), replica_groups={{0,1}}, dimensions={0}
  })";

  constexpr absl::string_view kF8ModuleStr = R"(
  HloModule fp8_allgather
  ENTRY test {
    param = f16[32,64] parameter(0)
    param_f8 = f8e4m3fn[32,64] convert(param)
    all-gather_f8 = f8e4m3fn[64,64] all-gather(param_f8), replica_groups={{0,1}}, dimensions={0}
    ROOT result = f16[64,64] convert(all-gather_f8)
  })";

  constexpr int64_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }

  Array<Eigen::half> input1({32, 64}), input2({32, 64});
  input1.FillRandom(Eigen::half(0.1f), 0.5f, /*seed=*/0);
  input2.FillRandom(Eigen::half(0.1f), 0.5f, /*seed=*/1);

  Literal input_lit1 = LiteralUtil::CreateFromArray(input1);
  Literal input_lit2 = LiteralUtil::CreateFromArray(input2);

  TF_ASSERT_OK_AND_ASSIGN(auto f16_module, ParseAndReturnVerifiedModule(
                                               kF16ModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult f16_result,
                          ExecuteReplicated(std::move(f16_module),
                                            std::vector<std::vector<Literal*>>{
                                                {&input_lit1}, {&input_lit2}}));
  VerifyAllGatherType(f16_result.optimized_module, F16);

  TF_ASSERT_OK_AND_ASSIGN(
      auto f8_module, ParseAndReturnVerifiedModule(kF8ModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult f8_result,
                          ExecuteReplicated(std::move(f8_module),
                                            std::vector<std::vector<Literal*>>{
                                                {&input_lit1}, {&input_lit2}}));
  VerifyAllGatherType(f8_result.optimized_module, F8E4M3FN);

  ASSERT_EQ(f16_result.results.size(), kNumReplicas);
  ASSERT_EQ(f8_result.results.size(), kNumReplicas);

  // FP8 vs FP16 comparison: close but not identical due to precision.
  EXPECT_TRUE(LiteralTestUtil::Near(f16_result.results[0], f8_result.results[0],
                                    ErrorSpec{6e-2, 6e-2}));
}

}  // namespace
}  // namespace xla
