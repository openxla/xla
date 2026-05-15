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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/all_gather.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/lib/gtl/int_type.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::Field;
using ::testing::HasSubstr;

TSL_LIB_GTL_DEFINE_INT_TYPE(CollectiveKernelEnabled, bool);

class BuildAllGatherInfoTest : public HloHardwareIndependentTestBase {
 protected:
  // Builds an all-gather HLO module and invokes BuildAllGatherInfo.
  //
  // `num_elements` is the number of elements in the input operand (per
  // replica). The all-gather is 1-D along dimension 0.
  //
  // `replica_groups` is a flat list of device IDs forming a single replica
  // group. An empty list means replica_groups={} (empty) in the HLO, which
  // will trigger the "replica groups must be provided" error path.
  absl::StatusOr<AllGatherInfo> BuildInfo(
      CollectiveKernelEnabled collective_kernel_enabled,
      PrimitiveType element_type, int64_t num_elements,
      std::vector<int32_t> replica_groups) {
    const int num_replicas =
        replica_groups.empty() ? 1 : static_cast<int>(replica_groups.size());
    const std::string element_type_str =
        primitive_util::LowercasePrimitiveTypeName(element_type);
    const std::string input_shape_str = absl::StrFormat("%d", num_elements);
    const std::string output_shape_str =
        absl::StrFormat("%d", num_elements * num_replicas);
    const std::string replica_groups_str =
        replica_groups.empty()
            ? ""
            : absl::StrFormat("{%s}", absl::StrJoin(replica_groups, ","));

    // The all-gather gathers along dimension 0: output_dim0 = num_replicas *
    // input_dim0.
    constexpr absl::string_view kModuleStr = R"(
    HloModule test
    ENTRY test_computation {
      param_0 = %1$s[%2$s] parameter(0)
      ROOT all-gather = %1$s[%3$s] all-gather(param_0),
          dimensions={0}, replica_groups={%4$s}
    }
    )";
    const std::string module_str =
        absl::StrFormat(kModuleStr, element_type_str, input_shape_str,
                        output_shape_str, replica_groups_str);

    SCOPED_TRACE(testing::Message() << "module_str: " << module_str);

    se::DeviceDescription device_info =
        TestGpuDeviceInfo::H100SXMDeviceInfo();

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> module,
        ParseAndReturnVerifiedModule(module_str, num_replicas));

    const HloInstruction* hlo_instr =
        HloHardwareIndependentTestBase::FindInstruction(
            module.get(), HloOpcode::kAllGather);
    return BuildAllGatherInfo(collective_kernel_enabled.value(), device_info,
                              Cast<HloAllGatherInstruction>(hlo_instr));
  }
};

TEST_F(BuildAllGatherInfoTest, SucceedsForSupportedF32) {
  EXPECT_THAT(BuildInfo(CollectiveKernelEnabled(true), F32, /*num_elements=*/512,
                        /*replica_groups=*/{0, 1}),
              IsOkAndHolds(AllOf(
                  Field(&AllGatherInfo::num_devices, 2),
                  Field(&AllGatherInfo::num_elements, 512),
                  Field(&AllGatherInfo::element_type, F32))));
}

TEST_F(BuildAllGatherInfoTest, SucceedsForSupportedBF16) {
  EXPECT_THAT(BuildInfo(CollectiveKernelEnabled(true), BF16,
                        /*num_elements=*/512, /*replica_groups=*/{0, 1}),
              IsOkAndHolds(AllOf(
                  Field(&AllGatherInfo::num_devices, 2),
                  Field(&AllGatherInfo::num_elements, 512),
                  Field(&AllGatherInfo::element_type, BF16))));
}

TEST_F(BuildAllGatherInfoTest, SucceedsForFourDevices) {
  EXPECT_THAT(
      BuildInfo(CollectiveKernelEnabled(true), F32, /*num_elements=*/512,
                /*replica_groups=*/{0, 1, 2, 3}),
      IsOkAndHolds(AllOf(Field(&AllGatherInfo::num_devices, 4),
                         Field(&AllGatherInfo::num_elements, 512),
                         Field(&AllGatherInfo::element_type, F32))));
}

TEST_F(BuildAllGatherInfoTest, FailsIfCollectiveKernelDisabled) {
  EXPECT_THAT(
      BuildInfo(CollectiveKernelEnabled(false), F32, /*num_elements=*/512,
                /*replica_groups=*/{0, 1}),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("not enabled")));
}

TEST_F(BuildAllGatherInfoTest, FailsForNonPowerOfTwoDevices) {
  EXPECT_THAT(
      BuildInfo(CollectiveKernelEnabled(true), F32, /*num_elements=*/512,
                /*replica_groups=*/{0, 1, 2}),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("only supported for power of 2")));
}

TEST_F(BuildAllGatherInfoTest, FailsForUnsupportedUnsignedType) {
  // U32 is not supported by the Triton all-gather kernel.
  EXPECT_THAT(
      BuildInfo(CollectiveKernelEnabled(true), U32, /*num_elements=*/512,
                /*replica_groups=*/{0, 1}),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("is not supported for all-gather kernel")));
}

TEST_F(BuildAllGatherInfoTest, FailsForUnalignedElements) {
  // 7 is not divisible by kNumElementsPerThread (which is >= 2).
  EXPECT_THAT(
      BuildInfo(CollectiveKernelEnabled(true), F32, /*num_elements=*/7,
                /*replica_groups=*/{0, 1}),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("not aligned to the alignment requirement")));
}

TEST_F(BuildAllGatherInfoTest, FailsIfReplicaGroupsEmpty) {
  EXPECT_THAT(
      BuildInfo(CollectiveKernelEnabled(true), F32, /*num_elements=*/512,
                /*replica_groups=*/{}),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("Replica groups must be explicitly provided")));
}

}  // namespace
}  // namespace xla::gpu
