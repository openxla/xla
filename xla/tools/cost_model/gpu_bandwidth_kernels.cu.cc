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

#include "xla/tools/cost_model/gpu_bandwidth_kernels.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/stream_executor/cuda/cuda_status.h"

namespace xla::gpu {
namespace {

using ::stream_executor::cuda::ToStatus;

// Size of the scratch buffer written and freed before every timed launch:
// the L2 cache size of H100.
constexpr size_t kL2CacheFlushSizeBytes = 50ULL * 1024 * 1024;
// Launch configuration: each thread block writes one `kTileBytes` tile.
constexpr int kBlockDim = 256;
constexpr int kTileBytes = 4096;
// Number of timed launches per transfer size; the fastest one is reported.
constexpr int kNumTimedLaunches = 16;

// Owns the destination buffer and the timing events of one measurement.
struct MeasurementResources {
  MeasurementResources() = default;
  MeasurementResources(const MeasurementResources&) = delete;
  MeasurementResources& operator=(const MeasurementResources&) = delete;
  ~MeasurementResources() {
    cudaFree(dst);  // No-op for nullptr.
    if (start != nullptr) cudaEventDestroy(start);
    if (end != nullptr) cudaEventDestroy(end);
  }

  void* dst = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;
};

// Writes `num_bytes` of zeros from shared memory to `dst`, one `kTileBytes`
// tile per thread block, with the Hopper tensor memory accelerator (TMA) bulk
// copy `cp.async.bulk.global.shared::cta`. The body is only compiled for
// sm_90+; the host code rejects older devices.
__global__ void WriteUblkcpKernel(char* dst, size_t num_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  __shared__ __align__(128) uint32_t tile[kTileBytes / sizeof(uint32_t)];
  for (int i = threadIdx.x; i < kTileBytes / sizeof(uint32_t);
       i += blockDim.x) {
    tile[i] = 0;
  }
  // Make the generic-proxy writes above visible to the async proxy.
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();

  const size_t offset = static_cast<size_t>(blockIdx.x) * kTileBytes;
  if (threadIdx.x != 0 || offset >= num_bytes) {
    return;
  }
  const size_t remaining_bytes = num_bytes - offset;
  const uint32_t tile_bytes = static_cast<uint32_t>(
      remaining_bytes < kTileBytes ? remaining_bytes : kTileBytes);
  asm volatile(
      "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n\t"
      "cp.async.bulk.commit_group;\n\t"
      "cp.async.bulk.wait_group 0;"
      :
      : "l"(__cvta_generic_to_global(dst + offset)),
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(tile))),
        "r"(tile_bytes)
      : "memory");
#endif
}

// Puts the device into a cold-L2 state: writes a scratch buffer larger than
// the L2 cache and frees it, which makes the driver flush the cache and
// synchronize the device before the next launch.
absl::Status FlushL2Cache() {
  void* scratch = nullptr;
  ABSL_RETURN_IF_ERROR(
      ToStatus(cudaMalloc(&scratch, kL2CacheFlushSizeBytes), "cudaMalloc"));
  const absl::Status status = ToStatus(
      cudaMemsetAsync(scratch, 0, kL2CacheFlushSizeBytes), "cudaMemsetAsync");
  cudaFree(scratch);
  return status;
}

}  // namespace

absl::StatusOr<double> MeasureWriteUblkcpBandwidthBytesPerSec(
    int ordinal, int64_t dma_size_bytes) {
  // `cp.async.bulk` requires transfer sizes that are multiples of 16 bytes.
  if (dma_size_bytes <= 0 || dma_size_bytes % 16 != 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("dma_size_bytes=%d must be a positive multiple of 16.",
                        dma_size_bytes));
  }
  if (cudaSetDevice(ordinal) != cudaSuccess) {
    cudaGetLastError();  // Clear the sticky per-thread error.
    return absl::UnavailableError(
        absl::StrFormat("Failed to set CUDA device %d.", ordinal));
  }
  int compute_capability_major = 0;
  ABSL_RETURN_IF_ERROR(ToStatus(
      cudaDeviceGetAttribute(&compute_capability_major,
                             cudaDevAttrComputeCapabilityMajor, ordinal),
      "cudaDeviceGetAttribute"));
  if (compute_capability_major < 9) {
    return absl::UnimplementedError(absl::StrFormat(
        "TMA bulk writes require compute capability 9.0+, device %d has %d.x.",
        ordinal, compute_capability_major));
  }

  const size_t num_bytes = static_cast<size_t>(dma_size_bytes);
  MeasurementResources resources;
  ABSL_RETURN_IF_ERROR(
      ToStatus(cudaMalloc(&resources.dst, num_bytes), "cudaMalloc"));
  ABSL_RETURN_IF_ERROR(
      ToStatus(cudaEventCreate(&resources.start), "cudaEventCreate"));
  ABSL_RETURN_IF_ERROR(
      ToStatus(cudaEventCreate(&resources.end), "cudaEventCreate"));
  char* const dst = static_cast<char*>(resources.dst);
  const int grid_dim =
      static_cast<int>((num_bytes + kTileBytes - 1) / kTileBytes);

  // Report the fastest of `kNumTimedLaunches` cold-L2 launches. A small
  // transfer lasts only a few microseconds, so individual timings scatter by
  // several percent with host-side launch jitter; the minimum rejects it.
  float best_elapsed_ms = 0.0f;
  for (int i = 0; i < kNumTimedLaunches; ++i) {
    ABSL_RETURN_IF_ERROR(FlushL2Cache());
    ABSL_RETURN_IF_ERROR(
        ToStatus(cudaEventRecord(resources.start), "cudaEventRecord"));
    WriteUblkcpKernel<<<grid_dim, kBlockDim>>>(dst, num_bytes);
    ABSL_RETURN_IF_ERROR(
        ToStatus(cudaGetLastError(), "WriteUblkcpKernel launch"));
    ABSL_RETURN_IF_ERROR(
        ToStatus(cudaEventRecord(resources.end), "cudaEventRecord"));
    ABSL_RETURN_IF_ERROR(
        ToStatus(cudaEventSynchronize(resources.end), "cudaEventSynchronize"));
    float elapsed_ms = 0.0f;
    ABSL_RETURN_IF_ERROR(ToStatus(
        cudaEventElapsedTime(&elapsed_ms, resources.start, resources.end),
        "cudaEventElapsedTime"));
    if (i == 0 || elapsed_ms < best_elapsed_ms) {
      best_elapsed_ms = elapsed_ms;
    }
  }
  return static_cast<double>(num_bytes) / (best_elapsed_ms * 1e-3);
}

}  // namespace xla::gpu
