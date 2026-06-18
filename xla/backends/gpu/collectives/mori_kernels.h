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
#include "xla/primitive_util.h"
#include "mori/shmem/shmem_api.hpp"

namespace xla_mori {

//using stream_executor::gpu::GpuStreamHandle;

using GpuStreamHandle = std::intptr_t;

// Zero-initialise signal flag memory (device memory / symmetric heap).
void InitSignalMemory(void* ptr, size_t bytes);

absl::Status SendSDMA(void* recv_buffer, void* send_buffer, size_t bytes, int peer,
    std::intptr_t stream_handle, int device_id);
 
// Intra-node P2P Send via MORI (put model, single kernel).
int Send(void* recv_buffer, void* send_buffer, size_t bytes, int peer,
         uint32_t* signal_flags, std::intptr_t stream_handle);

// Intra-node P2P Recv via MORI (put model – wait only).
int Recv(void* recv_buffer, void* send_buffer, size_t bytes, int peer,
         uint32_t* signal_flags, std::intptr_t stream_handle);

// Barrier on the given stream.
absl::Status BarrierOnStream(std::intptr_t stream_handle);

absl::Status AllGather(void* send_buffer, void* recv_buffer, size_t bytes,
               std::intptr_t stream_handle, int device_id);

absl::Status ReduceScatter(void* send_buffer, void* recv_buffer, void* staging_buffer, 
    xla::PrimitiveType dtype, size_t count, int gen_counter, 
    std::intptr_t stream_handle, int device_id);

void RegisterMemObjPtr(void *ptr, mori::application::SymmMemObjPtr obj);
void DeregisterMemObjPtr(void* ptr);
} // namespace xla_mori

#endif // XLA_BACKENDS_GPU_COLLECTIVES_MORI_KERNELS_H_
