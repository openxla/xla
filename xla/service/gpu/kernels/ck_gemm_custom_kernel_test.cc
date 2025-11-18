/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/kernels/ck_gemm_custom_kernel.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <hip/hip_fp16.h>

#include <gtest/gtest.h>
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"

namespace xla::gpu::kernel::gemm_universal {

TEST(CkGemmKernelTest, SimpleGemm) {
  se::Platform* platform =
      se::PlatformManager::PlatformWithName("ROCM").value();
  se::StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  auto stream = executor->CreateStream().value();

  // Load [8, 8] x [8, 8] gemm kernel
  TF_ASSERT_OK_AND_ASSIGN(
      auto custom_kernels,
      GetCkGemmKernels("ck_gemm", PrimitiveType::F16,
                       PrimitiveType::F16, PrimitiveType::F16, 8, 8, 8,
                       /*indices=*/{0, 1, 2},
                       executor->GetDeviceDescription()));
  auto custom_kernel = custom_kernels[0];

  TF_ASSERT_OK_AND_ASSIGN(auto gemm,
                          executor->LoadKernel(custom_kernel.kernel_spec()));

  int64_t length = 8*8;
  int64_t byte_length = sizeof(__half) * length;

  // Prepare arguments: a=2, b=2, c=0 (using FP16)
  se::DeviceMemory<__half> a = executor->AllocateArray<__half>(length, 0);
  se::DeviceMemory<__half> b = executor->AllocateArray<__half>(length, 0);
  se::DeviceMemory<__half> c = executor->AllocateArray<__half>(length, 0);

  __half value = __float2half(2.0f);
  uint16_t pattern;
  std::memcpy(&pattern, &value, sizeof(pattern));

  // For FP16, we need to set 16-bit patterns
  uint32_t pattern32 = (static_cast<uint32_t>(pattern) << 16) | pattern;

  TF_ASSERT_OK(stream->Memset32(&a, pattern32, byte_length));
  TF_ASSERT_OK(stream->Memset32(&b, pattern32, byte_length));
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // Launch gemm kernel with device memory arguments.
  se::KernelArgsDeviceMemoryArray arr(
      std::vector<se::DeviceMemoryBase>({a, b, c}),
      custom_kernel.shared_memory_bytes());
  TF_ASSERT_OK(gemm->Launch(custom_kernel.thread_dims(),
                            custom_kernel.block_dims(), stream.get(), arr));

  // Copy `c` data back to host.
  std::vector<__half> dst(length, __float2half(-1.0f));
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  std::vector<__half> expected(length, __float2half(8*4.0f));
  ASSERT_EQ(dst, expected);
}

}  // namespace xla::gpu::kernel::gemm_universal
