/* Copyright 2025 The OpenXLA Authors.

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

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/time/time.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "third_party/nccl/nccl.h"
#include "third_party/nccl/nccl_device.h"
#include "xla/backends/gpu/tests/collective_ops_ffi_kernels.h"
#include "xla/ffi/api/collectives_c_api.h"
#include "xla/service/rendezvous.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

absl::Status CommunicatorAllReduceU32(stream_executor::Stream* stream,
                                      XLA_FFI_Communicator* communicator,
                                      const void* send_buffer,
                                      void* recv_buffer, int64_t count) {
  ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(communicator);
  cudaStream_t cuda_stream =
      absl::bit_cast<cudaStream_t>(stream->platform_specific_handle().stream);

  ncclResult_t result =
      ncclAllReduce(send_buffer, recv_buffer, count, ncclUint32, ncclSum,
                    nccl_comm, cuda_stream);
  TF_RET_CHECK(result == ncclSuccess)
      << "ncclAllReduce failed: " << ncclGetErrorString(result);
  return stream->BlockHostUntilDone();
}

absl::Status WindowPeerAllReduceU32(stream_executor::Stream* stream,
                                    XLA_FFI_Window* window, void* recv_buffer,
                                    int64_t count) {
#if (NCCL_VERSION_CODE >= 22902) || defined(USE_NCCL_HOST_API)
  ncclWindow_t nccl_win = reinterpret_cast<ncclWindow_t>(window);

  void* src0 = nullptr;
  void* src1 = nullptr;
  ncclResult_t r0 = ncclGetPeerDevicePointer(nccl_win, 0, /*peer=*/0, &src0);
  TF_RET_CHECK(r0 == ncclSuccess)
      << "ncclGetPeerDevicePointer(rank=0) failed: " << ncclGetErrorString(r0);
  ncclResult_t r1 = ncclGetPeerDevicePointer(nccl_win, 0, /*peer=*/1, &src1);
  TF_RET_CHECK(r1 == ncclSuccess)
      << "ncclGetPeerDevicePointer(rank=1) failed: " << ncclGetErrorString(r1);
  TF_RET_CHECK(src0 != nullptr && src1 != nullptr);

  ABSL_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  static constexpr int32_t kKey = 0;
  const int32_t* key = &kKey;
  ABSL_RETURN_IF_ERROR(Rendezvous<const int32_t*>(
      "WindowPeerAllReduceU32", key, 2, absl::Seconds(1), absl::Seconds(5)));

  ABSL_ASSIGN_OR_RETURN(
      auto kernel, stream_executor::gpu::GpuKernelRegistry::GetGlobalRegistry()
                       .LoadKernel<Peer2AllReduce>(stream->parent()));

  const uint64_t byte_size = static_cast<uint64_t>(count) * sizeof(uint32_t);
  stream_executor::BlockDim block_dims(1);
  stream_executor::ThreadDim thread_dims(8);
  ABSL_RETURN_IF_ERROR(
      kernel.Launch(thread_dims, block_dims, stream,
                    stream_executor::DeviceAddress<uint32_t>::MakeFromByteSize(
                        src0, byte_size),
                    stream_executor::DeviceAddress<uint32_t>::MakeFromByteSize(
                        src1, byte_size),
                    stream_executor::DeviceAddress<uint32_t>::MakeFromByteSize(
                        recv_buffer, byte_size),
                    static_cast<size_t>(count)));
  return stream->BlockHostUntilDone();
#else
  return absl::UnimplementedError(
      "Window peer all-reduce requires NCCL >= 2.29.2");
#endif
}

}  // namespace xla::gpu
