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

#include "xla/backends/gpu/runtime/collective_params.h"

#include <gtest/gtest.h>
#include "xla/runtime/device_id.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/gpu_executable_run_options.h"

namespace xla::gpu {
namespace {

// Devices 0..3 run two replicas of two partitions.
DeviceAssignment TwoReplicasOfTwoPartitions() {
  DeviceAssignment device_assn(/*replica_count=*/2, /*computation_count=*/2);
  device_assn.FillIota(0);
  return device_assn;
}

TEST(CountLocalParticipantsTest, AllDevicesAreLocalWithoutDeviceIdMap) {
  EXPECT_EQ(CountLocalParticipants(TwoReplicasOfTwoPartitions(),
                                   /*global_device_id_map=*/nullptr),
            4);
}

TEST(CountLocalParticipantsTest, CountsLocalDevicesOfTheAssignment) {
  // Global device 7 runs in this process but is not in the assignment.
  const GpuExecutableRunOptions::DeviceIdMap global_device_id_map = {
      {LocalDeviceId(0), GlobalDeviceId(1)},
      {LocalDeviceId(1), GlobalDeviceId(3)},
      {LocalDeviceId(2), GlobalDeviceId(7)}};
  EXPECT_EQ(CountLocalParticipants(TwoReplicasOfTwoPartitions(),
                                   &global_device_id_map),
            2);
}

}  // namespace
}  // namespace xla::gpu
