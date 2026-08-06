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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_MORI_KERNELS_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_MORI_KERNELS_H_

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "mori/shmem/shmem_api.hpp"
#include "xla/primitive_util.h"

namespace xla_mori {

// using stream_executor::gpu::GpuStreamHandle;

using GpuStreamHandle = std::intptr_t;

// Zero-initialise signal flag memory (device memory / symmetric heap).
void InitSignalMemory(void* ptr, size_t bytes);

// Allocates a device int array of `count` elements and copies `host_values`
// into it. Returns nullptr on failure. Free with FreeDeviceArray.
int* AllocDeviceIntArray(const int* host_values, int count);

// Frees memory previously returned by AllocDeviceIntArray.
void FreeDeviceArray(void* device_ptr);

absl::Status SendSDMA(void* recv_buffer, void* send_buffer, size_t bytes,
                      int peer, std::intptr_t stream_handle);

// Intra-node P2P Send via MORI (put model, single kernel).
int Send(void* recv_buffer, void* send_buffer, size_t bytes, int peer,
         uint32_t* signal_flags, std::intptr_t stream_handle);

// Intra-node P2P Recv via MORI (put model – wait only).
int Recv(void* recv_buffer, void* send_buffer, size_t bytes, int peer,
         uint32_t* signal_flags, std::intptr_t stream_handle);

// Barrier on the given stream.
absl::Status BarrierOnStream(std::intptr_t stream_handle);

// All-gather over the `num_ranks` participants of this collective. `my_rank` is
// this participant's rank within the collective and `rank_to_pe` is a device
// array mapping collective rank -> global MORI PE (so arbitrary device subsets
// work, not just a contiguous prefix). `flags_buffer` is symmetric-heap memory
// with at least `num_ranks` uint64 slots used for cross-rank completion, and
// `generation` is a per-call monotonically increasing value.
absl::Status AllGather(void* send_buffer, void* recv_buffer, size_t bytes,
                       int my_rank, int num_ranks, const int* rank_to_pe,
                       void* flags_buffer, uint64_t generation,
                       std::intptr_t stream_handle);

absl::Status ReduceScatter(void* send_buffer, void* recv_buffer,
                           void* staging_buffer, void* group_counters,
                           xla::PrimitiveType dtype, size_t count, int my_rank,
                           int num_ranks, std::intptr_t stream_handle);

}  // namespace xla_mori

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_MORI_KERNELS_H_
