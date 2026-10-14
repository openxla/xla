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

#include "xla/backends/gpu/runtime/collective_execution.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/runtime/device_id.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {
namespace {

se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

class CollectiveTopologyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    se::StreamExecutor* executor = GpuExecutor();
    ASSERT_OK_AND_ASSIGN(stream_, executor->CreateStream());

    device_assignment_(0, 0) = 0;
    device_assignment_(1, 0) = 1;
    gpu_options_.set_gpu_global_device_ids(GpuExecutableRunOptions::DeviceIdMap{
        std::make_pair(LocalDeviceId(0), GlobalDeviceId(0))});
    gpu_options_.set_incarnations(
        absl::flat_hash_map<GlobalDeviceId, IncarnationId>{
            {GlobalDeviceId(0), IncarnationId(1)},
            {GlobalDeviceId(1), IncarnationId(2)}});

    ServiceExecutableRunOptions run_options;
    run_options.mutable_run_options()->set_stream(stream_.get());
    run_options.mutable_run_options()->set_device_assignment(
        &device_assignment_);
    run_options.mutable_run_options()->set_gpu_executable_run_options(
        &gpu_options_);
    ASSERT_OK_AND_ASSIGN(
        CollectiveParams params,
        CollectiveParams::Create(run_options, /*async_streams=*/{},
                                 LocalDeviceId(executor->device_ordinal())));
    params_ = std::make_unique<CollectiveParams>(std::move(params));
  }

  std::unique_ptr<se::Stream> stream_;
  DeviceAssignment device_assignment_{/*replica_count=*/2,
                                      /*computation_count=*/1};
  GpuExecutableRunOptions gpu_options_;
  std::unique_ptr<CollectiveParams> params_;
};

TEST_F(CollectiveTopologyTest, MatchesParamsItWasTakenFrom) {
  CollectiveTopology topology(*params_);
  EXPECT_TRUE(topology.Matches(*params_));
}

TEST_F(CollectiveTopologyTest, DetectsChangesBehindTheSamePointers) {
  CollectiveTopology topology(*params_);

  device_assignment_(0, 0) = 1;
  device_assignment_(1, 0) = 0;
  EXPECT_FALSE(topology.Matches(*params_));
  device_assignment_(0, 0) = 0;
  device_assignment_(1, 0) = 1;
  EXPECT_TRUE(topology.Matches(*params_));

  gpu_options_.set_gpu_global_device_ids(GpuExecutableRunOptions::DeviceIdMap{
      std::make_pair(LocalDeviceId(0), GlobalDeviceId(1))});
  ASSERT_EQ(params_->global_device_id_map,
            &*gpu_options_.gpu_global_device_ids());
  EXPECT_FALSE(topology.Matches(*params_));
  gpu_options_.set_gpu_global_device_ids(GpuExecutableRunOptions::DeviceIdMap{
      std::make_pair(LocalDeviceId(0), GlobalDeviceId(0))});
  EXPECT_TRUE(topology.Matches(*params_));

  gpu_options_.set_incarnations(
      absl::flat_hash_map<GlobalDeviceId, IncarnationId>{
          {GlobalDeviceId(0), IncarnationId(1)},
          {GlobalDeviceId(1), IncarnationId(3)}});
  ASSERT_EQ(params_->incarnations, &*gpu_options_.incarnations());
  EXPECT_FALSE(topology.Matches(*params_));
}

TEST_F(CollectiveTopologyTest, DetectsChangedFields) {
  CollectiveTopology topology(*params_);

  CollectiveParams params = *params_;
  params.global_device_id = GlobalDeviceId(1);
  EXPECT_FALSE(topology.Matches(params));

  params = *params_;
  params.collectives = nullptr;
  EXPECT_FALSE(topology.Matches(params));

  params = *params_;
  params.device_assn = nullptr;
  EXPECT_FALSE(topology.Matches(params));
  EXPECT_TRUE(CollectiveTopology(params).Matches(params));

  params = *params_;
  params.global_device_id_map = nullptr;
  EXPECT_FALSE(topology.Matches(params));

  params = *params_;
  params.incarnations = nullptr;
  EXPECT_FALSE(topology.Matches(params));
}

TEST_F(CollectiveTopologyTest, IgnoresFieldsThatDoNotDetermineCliques) {
  CollectiveTopology topology(*params_);

  CollectiveParams params = *params_;
  params.run_id = RunId(params.run_id.ToInt() + 1);
  params.collective_max_nchannels = 4;
  params.p2p_max_nchannels = 4;
  params.local_device_count = 8;
  EXPECT_TRUE(topology.Matches(params));
}

}  // namespace
}  // namespace xla::gpu
