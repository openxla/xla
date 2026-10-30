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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::cuda {
namespace {

TEST(CudaCommandBuffer128Test,
     NativeKernelPreservesPreferredClusterDimensions) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  const DeviceDescription& device = executor->GetDeviceDescription();
  if (device.driver_version() < SemanticVersion{12, 8, 0}) {
    GTEST_SKIP() << "Preferred cluster dimensions require CUDA driver >= 12.8.";
  }
  if (!device.cuda_compute_capability().IsAtLeastBlackwell()) {
    GTEST_SKIP()
        << "Preferred cluster dimensions require at least a Blackwell GPU.";
  }

  ASSERT_OK_AND_ASSIGN(KernelLoaderSpec spec,
                       gpu::GetAddI32TestKernelSpec(platform->id()));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                       executor->LoadKernel(spec));
  DeviceAddress<int32_t> a = executor->AllocateArray<int32_t>(4);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CommandBuffer> command_buffer,
      executor->CreateCommandBuffer(CommandBuffer::Mode::kPrimary));
  NativeKernel native{kernel->platform_specific_handle().kernel, "add",
                      /*pdl=*/false, /*cooperative=*/false,
                      ClusterDim(2, 1, 1)};
  ASSERT_OK_AND_ASSIGN(
      const CommandBuffer::Command* command,
      command_buffer->CreateLaunch(ThreadDim(1, 1, 1), BlockDim(4, 1, 1),
                                   ClusterDim(1, 1, 1), native,
                                   *PackKernelArgs(0, a, a, a), {}));
  const auto* gpu_command =
      dynamic_cast<const gpu::GpuCommandBuffer::GpuCommand*>(command);
  ASSERT_NE(gpu_command, nullptr);
  CUlaunchAttributeValue value{};
  ASSERT_EQ(cuGraphKernelNodeGetAttribute(
                absl::bit_cast<CUgraphNode>(gpu_command->handle),
                CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, &value),
            CUDA_SUCCESS);
  EXPECT_EQ(value.preferredClusterDim.x, 2);
  EXPECT_EQ(value.preferredClusterDim.y, 1);
  EXPECT_EQ(value.preferredClusterDim.z, 1);
  ASSERT_OK(command_buffer->Finalize());
  executor->Deallocate(&a);
}

}  // namespace
}  // namespace stream_executor::cuda
