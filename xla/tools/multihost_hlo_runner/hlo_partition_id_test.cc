/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "xla/tests/literal_test_util.h"
#include "xla/tools/multihost_hlo_runner/hlo_runner.h"

namespace xla {

TEST(MultiHostHloRunnerTest, PartitionId) {
  const std::string hlo_string = R"(
    HloModule test, is_scheduled=true
    ENTRY test_computation {
      ROOT out = u32[] partition-id()
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          MultiHostHloRunner::ReadModuleFromString(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          MultiHostHloRunner::GetDeviceClient(
                              xla::MultiHostHloRunner::DeviceType::kGpu));
  const int num_devices = client->device_count();
  MultiHostHloRunner::Options options{
      .num_partitions = static_cast<size_t>(num_devices),
      .hlo_passes_mode =
          MultiHostHloRunner::HloPassesMode::kDisableAllHloPasses,
      .spmd_mode = (num_devices > 1)
                       ? MultiHostHloRunner::SpmdMode::kUseSpmdPartitioning
                       : MultiHostHloRunner::SpmdMode::kNotUseSpmdPartitioning};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiHostHloRunner> hlo_runner,
      MultiHostHloRunner::CreateMultiHostHloRunner(options, std::move(client)));
  TF_ASSERT_OK_AND_ASSIGN(auto output,
                          hlo_runner->CompileAndRun(hlo_module.get()));
  for (int device_id = 0; device_id < num_devices; ++device_id) {
    const std::vector<xla::Literal> &device_output = output[device_id];
    EXPECT_EQ(device_output.size(), 1);
    LiteralTestUtil::ExpectR0Equal<uint32_t>(device_id, device_output[0]);
  }
}
}  // namespace xla
