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
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
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

void VerifyAllGatherType(const HloModule* module, PrimitiveType expected_type) {
  bool found = false;
  for (auto* comp : module->computations()) {
    for (auto* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kAllGather ||
          instr->opcode() == HloOpcode::kAllGatherStart) {
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

  // Receives a required device count and asserts or skips the test if the
  // device count is less than the required device count.
  //
  // Tap presubmits have only 2 GPUs available. This is the minimum required
  // device count below which we fail the test.
  // If the required device count is more than 2 and not enough devices are
  // available, the test is marked as skipped when running on TAP.
  //
  // Returns false if the test should be skipped or has failed. Otherwise,
  // returns true.
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

  // Returns true if the current device supports Triton collective kernels.
  // Triton collectives require Hopper+ for CUDA or any supported ROCm device.
  bool IsTritonCapable() {
    return IsHopperAndHigher() || Capability().IsRocm();
  }

  // Activates a log monitor that fails the test if a "Falling back to
  // NCCL/RCCL" WARNING is emitted while the Triton flag is set.
  // Call this just before ExecuteReplicated() in tests using
  // Triton-compatible shapes. Has no effect when use_triton_backend is false
  // or the hardware is not Triton-capable.
  //
  // The ScopedMockLog expectation is verified automatically when the fixture
  // is torn down (i.e., when triton_mock_log_ is destroyed).
  void CheckNoTritonFallback(bool use_triton_backend) {
    if (!use_triton_backend || !IsTritonCapable()) return;
    triton_mock_log_ = std::make_unique<absl::ScopedMockLog>(
        absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(*triton_mock_log_,
                Log(absl::LogSeverity::kWarning, ::testing::_,
                    ::testing::HasSubstr("Falling back to NCCL/RCCL")))
        .Times(0);
    triton_mock_log_->StartCapturingLogs();
  }

 private:
  // Holds the ScopedMockLog set up by CheckNoTritonFallback().
  // Null when not in use. Expectations are verified when this is destroyed.
  std::unique_ptr<absl::ScopedMockLog> triton_mock_log_;
};

struct AllGatherTestParams {
  // If true uses the async stream for the collective.
  bool is_async;
  // If true, uses the Triton backend for all-gather.
  // If false, uses the NCCL backend.
  bool use_all_gather_triton_backend;

  // Returns the shape for the given element type and rank.
  std::vector<int64_t> GetShape(PrimitiveType element_type,
                                int32_t rank = 1) const {
    // Use a reasonable size for testing
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
        params.push_back({is_async, use_all_gather_triton_backend});
      }
    }
    return params;
  }

  // Convert to string for test naming.
  [[maybe_unused]] friend void PrintTo(const AllGatherTestParams& params,
                                       std::ostream* os) {
    *os << "{ .is_async=" << params.is_async
        << ", .use_all_gather_triton_backend="
        << params.use_all_gather_triton_backend << " }";
  }
};

struct AllGatherTypesTestParams : public AllGatherTestParams {
  // The element type of the all-gather.
  PrimitiveType element_type;

  static std::vector<AllGatherTypesTestParams> Generate() {
    std::vector<AllGatherTypesTestParams> params;
    for (auto& all_gather_test_params : AllGatherTestParams::Generate()) {
      // Test common types used in ML workloads
      for (const PrimitiveType element_type : {F32, F16, BF16, S32, S8, PRED}) {
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

  // Create inputs for each replica
  for (int i = 0; i < num_replicas; ++i) {
    auto& input =
        inputs.emplace_back(Array<ElementType>(input_shape.dimensions()));
    FillRandom<ElementType>(input, i);
    input_literals.push_back(LiteralUtil::CreateFromArray(input));
  }

  std::vector<Literal> expected_output_literals;

  const std::vector<ReplicaGroup>& replica_groups =
      instr->device_list()->replica_groups();

  // Map each device to set of replica groups it belongs to.
  std::vector<std::vector<int64_t>> device_to_groups(num_replicas);
  for (const auto& replica_group : replica_groups) {
    const auto& replica_ids = replica_group.replica_ids();
    for (int64_t replica : replica_group.replica_ids()) {
      CHECK_EQ(device_to_groups[replica].size(), 0);
      device_to_groups[replica].assign(replica_ids.begin(), replica_ids.end());
    }
  }

  // Build expected outputs by concatenating inputs along the all-gather
  // dimension For all-gather, each replica in a group receives the
  // concatenation of all inputs from replicas in that group along the specified
  // dimension.
  const Shape& output_shape = instr->shape();
  for (int i = 0; i < num_replicas; ++i) {
    const std::vector<int64_t>& group = device_to_groups[i];

    // Create output literal with the concatenated shape
    Literal output(output_shape);

    // Manually concatenate by copying slices from each replica in the group
    // along the all-gather dimension
    int64_t offset = 0;
    for (int64_t replica : group) {
      const Literal& input = input_literals[replica];
      const int64_t slice_size = input_shape.dimensions(all_gather_dimension);

      // Set up source and destination bases
      std::vector<int64_t> src_base(input_shape.dimensions_size(), 0);
      std::vector<int64_t> dest_base(output_shape.dimensions_size(), 0);
      dest_base[all_gather_dimension] = offset;

      // Copy data from input to output at the appropriate offset
      TF_RETURN_IF_ERROR(output.CopySliceFrom(
          input, src_base, dest_base, /*copy_size=*/input_shape.dimensions()));
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
    opts.set_xla_gpu_unsupported_use_all_gather_triton_backend(
        GetParam().use_all_gather_triton_backend);
    return opts;
  }

  // Calls the base CheckNoTritonFallback() with this test's Triton flag.
  void CheckNoTritonFallback() {
    AllGatherTestNoParams::CheckNoTritonFallback(
        GetParam().use_all_gather_triton_backend);
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
    opts.set_xla_gpu_unsupported_use_all_gather_triton_backend(
        GetParam().use_all_gather_triton_backend);
    return opts;
  }

  // Calls the base CheckNoTritonFallback() with this test's Triton flag.
  void CheckNoTritonFallback() {
    AllGatherTestNoParams::CheckNoTritonFallback(
        GetParam().use_all_gather_triton_backend);
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

  // Calculate output shape (dimension 0 multiplied by num replicas in group)
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
                        /*arguments=*/test_io.InputLiteralPtrs()))
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
                        /*arguments=*/test_io.InputLiteralPtrs()))
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
  output_shape[0] *= 2;  // 2 replicas per group

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
                        /*arguments=*/test_io.InputLiteralPtrs()))
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-5}))
        << "ExpectedOutput != Result at index " << i;
  }
}

// This test uses a 3D shape with non-power-of-2 dimensions (e.g., [3,3,113] =
// 1017 elements) which doesn't meet Triton's alignment requirements (must be
// divisible by 4). The test verifies that the system correctly falls back to
// NCCL/RCCL when Triton is not supported.
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
  const std::vector<int64_t> input_shape =
      GetParam().GetShape(PrimitiveType::F32, /*rank=*/3);
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

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()))
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-5}))
        << "ExpectedOutput != Result at index " << i;
  }
}

// 3D test with power-of-2 dimensions that work with Triton tiling
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
  // Use power-of-2 dimensions: [8, 8, 16] = 1024 elements (aligned)
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
                        /*arguments=*/test_io.InputLiteralPtrs()))
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-5}))
        << "ExpectedOutput != Result at index " << i;
  }
}

// Test gathering on dimension 1 for 2D data
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
                        /*arguments=*/test_io.InputLiteralPtrs()))
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-5}))
        << "ExpectedOutput != Result at index " << i;
  }
}

// Test gathering on dimension 1 for 3D data
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
                        /*arguments=*/test_io.InputLiteralPtrs()))
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-5}))
        << "ExpectedOutput != Result at index " << i;
  }
}

// Test gathering on dimension 2 for 3D data
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
                        /*arguments=*/test_io.InputLiteralPtrs()))
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-5}))
        << "ExpectedOutput != Result at index " << i;
  }
}

// Large BF16 test matching the Python test configuration
// Tests 1024 elements (2KB) with 2 GPUs to verify Triton output correctness
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

  // Build test inputs and expected outputs
  TF_ASSERT_OK_AND_ASSIGN(InputsOutputs test_io,
                          (BuildTestInputsOutputs<PrimitiveType::BF16>(
                              *module, kNumReplicas,
                              /*all_gather_dimension=*/0)));

  CheckNoTritonFallback();
  // Execute the all-gather operation
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()));

  // Verify outputs match expected
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-3}))
        << "ExpectedOutput != Result at replica " << i
        << "\nThis test verifies that AllGather output matches the expected "
        << "concatenation of inputs. Failure indicates incorrect Triton "
           "implementation.";
  }
}

// Large BF16 test with 4 GPUs
// Tests 1024 elements per replica (2KB) -> 4096 elements output (8KB)
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

  // Build test inputs and expected outputs
  TF_ASSERT_OK_AND_ASSIGN(InputsOutputs test_io,
                          (BuildTestInputsOutputs<PrimitiveType::BF16>(
                              *module, kNumReplicas,
                              /*all_gather_dimension=*/0)));

  CheckNoTritonFallback();
  // Execute the all-gather operation
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()));

  // Verify outputs match expected
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-3}))
        << "ExpectedOutput != Result at replica " << i
        << "\nThis test verifies that AllGather output matches the expected "
        << "concatenation of inputs. Failure indicates incorrect Triton "
           "implementation.";
  }
}

// Large BF16 test with 8 GPUs
// Tests 1024 elements per replica (2KB) -> 8192 elements output (16KB)
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

  // Build test inputs and expected outputs
  TF_ASSERT_OK_AND_ASSIGN(InputsOutputs test_io,
                          (BuildTestInputsOutputs<PrimitiveType::BF16>(
                              *module, kNumReplicas,
                              /*all_gather_dimension=*/0)));

  CheckNoTritonFallback();
  // Execute the all-gather operation
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()));

  // Verify outputs match expected
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-3}))
        << "ExpectedOutput != Result at replica " << i
        << "\nThis test verifies that AllGather output matches the expected "
        << "concatenation of inputs. Failure indicates incorrect Triton "
           "implementation.";
  }
}

// Test fixture for verifying that the Triton backend actually executes when
// the Triton flag is set. Uses ScopedMockLog to detect silent NCCL fallbacks.
//
// The key insight: when xla_gpu_unsupported_use_all_gather_triton_backend=true
// but the Triton kernel cannot run (either because the shape is not supported
// at compile time or because runtime checks fail), a LOG(WARNING) containing
// "Falling back to NCCL/RCCL" is emitted. Tests in this fixture use
// ScopedMockLog to assert that this warning is absent (Triton ran) or present
// (expected fallback).
class AllGatherTritonVerificationTest : public AllGatherTestNoParams {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions opts = CollectiveOpsWithFlagsBase::GetDebugOptionsForTest();
    opts.set_xla_gpu_unsupported_use_all_gather_triton_backend(true);
    return opts;
  }
};

// Verifies that the Triton backend actually runs (no NCCL/RCCL fallback) when:
//  - The Triton flag is explicitly set, AND
//  - The all-gather shape is compatible with Triton
//    (1024 bf16 elements: aligned to kNumElementsPerThread=4, power-of-2 GPUs)
//
// If Triton silently falls back to NCCL/RCCL, this test fails because the
// "Falling back to NCCL/RCCL" WARNING would be emitted and detected.
TEST_F(AllGatherTritonVerificationTest, TritonActuallyRuns_BF16_1024Elements_2GPUs) {
  constexpr int64_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }
  if (!IsTritonCapable()) {
    GTEST_SKIP() << "Test requires Hopper+ CUDA or ROCm for Triton backend. "
                    "Skipping Triton execution verification.";
  }

  constexpr absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    param_0 = bf16[1024] parameter(0)
    ROOT all-gather = bf16[2048] all-gather(param_0),
      replica_groups={{0,1}}, dimensions={0}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(InputsOutputs test_io,
                          (BuildTestInputsOutputs<PrimitiveType::BF16>(
                              *module, kNumReplicas,
                              /*all_gather_dimension=*/0)));

  // Monitor logs during execution to verify Triton backend actually ran.
  // A "Falling back to NCCL/RCCL" WARNING indicates a silent fallback, which
  // would mean the test was not actually validating Triton behavior.
  absl::ScopedMockLog mock_log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(mock_log,
              Log(absl::LogSeverity::kWarning, ::testing::_,
                  ::testing::HasSubstr("Falling back to NCCL/RCCL")))
      .Times(0);  // No fallback expected — Triton should run.
  mock_log.StartCapturingLogs();

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), test_io.InputLiteralPtrs()));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-3}))
        << "Triton all-gather result mismatch at replica " << i;
  }
}

// Verifies that when the Triton flag is set but the shape is not supported by
// Triton (1017 elements = 3*3*113, not divisible by kNumElementsPerThread=4),
// a WARNING containing "Falling back to NCCL/RCCL" is emitted.
//
// This test serves two purposes:
//  1. Confirms the warning mechanism works correctly.
//  2. Documents that NCCL/RCCL still produces a correct result on fallback.
TEST_F(AllGatherTritonVerificationTest,
       NcclFallbackWarningEmitted_NonAlignedShape_2GPUs) {
  constexpr int64_t kNumReplicas = 2;
  if (!CheckDeviceCount(kNumReplicas)) {
    return;
  }
  if (!IsTritonCapable()) {
    GTEST_SKIP() << "Test requires Hopper+ CUDA or ROCm for Triton backend. "
                    "Skipping NCCL fallback warning verification.";
  }

  // Shape has 1017 elements (3*3*113): not divisible by kNumElementsPerThread
  // (4), so Triton cannot emit a kernel and falls back to NCCL/RCCL.
  constexpr absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    param_0 = f32[3,3,113] parameter(0)
    ROOT all-gather = f32[6,3,113] all-gather(param_0),
      replica_groups={{0,1}}, dimensions={0}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(InputsOutputs test_io,
                          (BuildTestInputsOutputs<PrimitiveType::F32>(
                              *module, kNumReplicas,
                              /*all_gather_dimension=*/0)));

  // Expect the fallback warning to be emitted exactly because Triton cannot
  // handle this shape.
  absl::ScopedMockLog mock_log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(mock_log,
              Log(absl::LogSeverity::kWarning, ::testing::_,
                  ::testing::HasSubstr("Falling back to NCCL/RCCL")))
      .Times(::testing::AtLeast(1));  // Fallback warning must be emitted.
  mock_log.StartCapturingLogs();

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module), test_io.InputLiteralPtrs()));

  // NCCL/RCCL fallback must still produce the correct result.
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-5}))
        << "NCCL/RCCL fallback result mismatch at replica " << i;
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
    GTEST_SKIP() << "FP8 requires GPU with OCP FP8 support (CUDA Hopper+ or "
                    "ROCm MI350/gfx12xx with ROCm 7.0+).";
  }

  // FP16 baseline
  constexpr absl::string_view kF16ModuleStr = R"(
  HloModule f16_allgather
  ENTRY test {
    param = f16[32,64] parameter(0)
    ROOT all-gather = f16[64,64] all-gather(param), replica_groups={{0,1}}, dimensions={0}
  })";

  // FP8 version
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
  // Verify FP16 all-gather type in optimized module
  VerifyAllGatherType(f16_result.optimized_module, F16);

  TF_ASSERT_OK_AND_ASSIGN(
      auto f8_module, ParseAndReturnVerifiedModule(kF8ModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult f8_result,
                          ExecuteReplicated(std::move(f8_module),
                                            std::vector<std::vector<Literal*>>{
                                                {&input_lit1}, {&input_lit2}}));
  // Verify FP8 all-gather type in optimized module
  VerifyAllGatherType(f8_result.optimized_module, F8E4M3FN);

  ASSERT_EQ(f16_result.results.size(), kNumReplicas);
  ASSERT_EQ(f8_result.results.size(), kNumReplicas);

  // FP8 vs FP16 comparison: should be close but not identical
  // Note: Using 6% tolerance due to cumulative FP16 and FP8 precision loss
  EXPECT_TRUE(LiteralTestUtil::Near(f16_result.results[0], f8_result.results[0],
                                    ErrorSpec{6e-2, 6e-2}));
}

}  // namespace
}  // namespace xla
