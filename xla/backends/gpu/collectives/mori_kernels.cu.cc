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

#include <algorithm>
#include <string>
#include <map>
#include <mutex>
#include <tuple>

#include "absl/strings/str_cat.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/collectives/mori_kernels.h"
#include "mori/shmem/shmem.hpp"
#include "mori/collective/reduce_scatter_kernels.hpp"

using namespace mori;

#ifndef MORI_UNLIKELY
#define MORI_UNLIKELY(x) (x)
#endif

#define MORI_HIP_ERROR(status) \
  if ((status) != hipSuccess) { \
    return absl::InternalError(absl::StrCat("MORI error: ", hipGetErrorString(status))); \
  }

namespace xla_mori {

static std::mutex g_memObjMapMutex;
using MemObjMap = std::map< void*, application::SymmMemObjPtr >;
static std::deque<MemObjMap> g_memObjMap(8);

void InitSignalMemory(void* ptr, size_t bytes) {
  if (ptr == nullptr || bytes == 0) {
    return;
  }
  (void)hipMemset(ptr, 0, bytes);
}

void RegisterMemObjPtr(void* ptr, mori::application::SymmMemObjPtr obj) {
  int dev_id = -1;
  (void)hipGetDevice(&dev_id);
  std::lock_guard<std::mutex> lock(g_memObjMapMutex);
  if (MORI_UNLIKELY(dev_id >= g_memObjMap.size())) {
    g_memObjMap.resize(dev_id + 1);
  }
  auto& devObjMap = g_memObjMap[dev_id];
  devObjMap[ptr] = obj;
}

void DeregisterMemObjPtr(void* ptr) {
  int dev_id = -1;
  (void)hipGetDevice(&dev_id);
  std::lock_guard<std::mutex> lock(g_memObjMapMutex);
  if (MORI_UNLIKELY(dev_id >= g_memObjMap.size())) {
    return;
  }
  auto& devObjMap = g_memObjMap[dev_id];
  devObjMap.erase(ptr);
}

std::tuple<application::SymmMemObjPtr, uintptr_t> 
          QueryMemObjPtr(void* ptr, size_t size, int device_id) {
  
  std::lock_guard<std::mutex> lock(g_memObjMapMutex);
  if (MORI_UNLIKELY(device_id >= g_memObjMap.size())) {
    fprintf(stderr, "QueryMemObjPtr: Memory wrong device id: %d\n", device_id);
    return std::make_tuple(application::SymmMemObjPtr{}, 0);
  }
  auto& devObjMap = g_memObjMap[device_id];
  auto it = devObjMap.upper_bound(ptr);
  if (MORI_UNLIKELY(it == devObjMap.begin())) {
    fprintf(stderr, "QueryMemObjPtr: Memory object not found: ptr=%p\n", ptr);
    return std::make_tuple(application::SymmMemObjPtr{}, 0);
  }
  auto& obj = (--it)->second;
  auto ofs = reinterpret_cast<uintptr_t>(ptr) - 
              reinterpret_cast<uintptr_t>(obj->localPtr);
  if (MORI_UNLIKELY(ofs + size > obj->size)) {
    fprintf(stderr, "QueryMemObjPtr: Memory object out of range: ptr=%p size=%zx ofs=%zx obj->size=%zx\n", ptr, size, ofs, obj->size);
    return std::make_tuple(application::SymmMemObjPtr{}, 0);
  }
  return std::make_tuple(obj, ofs);
}


namespace {


__global__ void BarrierKernel() {
  shmem::ShmemBarrierAllThread();
}

__global__ void SDMAPutKernel(int myPe, int destPe, int numQ, 
        application::SymmMemObjPtr inBuf, size_t inOfs,
        application::SymmMemObjPtr outBuf, size_t outOfs, size_t chunkSz) {
  const int tid = threadIdx.x;

  if (tid < numQ) {;
    int qpId = tid;
    size_t qpChunkSz = chunkSz / numQ, qpOfs = qpId * qpChunkSz;
    if (qpId == numQ - 1) {
      qpChunkSz = chunkSz - (qpId * qpChunkSz);
    }
    // printf("inBuf: %p outBuf: %p outOfs: %zx inOfs: %zx qpOfs: %zx qpChunkSz: %zx\n", 
    //   inBuf->localPtr, outBuf->localPtr, outOfs, inOfs, qpOfs, qpChunkSz);
    shmem::ShmemPutMemNbiThread(outBuf, outOfs + qpOfs, inBuf, inOfs + qpOfs, 
           qpChunkSz, destPe, qpId);
    // shmem::ShmemQuietThread(destPe, outBuf);
  }
}

__global__ void AllGatherKernel(int myPe, int npes, int numQ, 
        application::SymmMemObjPtr inBuf, size_t inOfs,
        application::SymmMemObjPtr outBuf, size_t outOfs, size_t chunkSz) {
  const int tid = threadIdx.x;

  // so we split data sending across numQ queues
  // each queue sends chunkBytes / numQ bytes
  if (tid < npes * numQ) {
    int destPe = tid / numQ;
    int qpId = tid - destPe * numQ;
    size_t qpChunkSz = chunkSz / numQ, qpOfs = qpId * qpChunkSz;
    if (qpId == numQ - 1) {
      qpChunkSz = chunkSz - (qpId * qpChunkSz);
    }

    size_t dstOfs = static_cast<size_t>(myPe) * chunkSz + qpOfs;
    // printf(" rank=%d at %d\n", myPe, __LINE__);
    {
      // printf("rank=%d sending to %d qpId=%d ofs=%zu chunkSz=%zu\n", myPe,
      //   destPe, qpId, dstOfs, qpChunkSz);
      shmem::ShmemPutMemNbiThread(outBuf, outOfs + dstOfs, inBuf, inOfs + qpOfs, 
          qpChunkSz, destPe, qpId);
    }
    // return;
    // //printf("rank=%d at %d\n", myPe, __LINE__);
    // // //    enum TransportType { RDMA = 0, P2P = 1, SDMA = 2 };
    // auto ttype = shmem::GetGlobalGpuStatesPtr()->transportTypes[destPe];
    // printf("transport type: %d numQ: %d\n", ttype, buf->sdmaNumQueue);
    // NOTE: no need to check transport type here, as the transport type is already checked in the ShmemPutMemNbiThread
    // if (ttype == application::SDMA) {
    shmem::ShmemQuietThread(destPe, outBuf);
    // }
  }
  if (tid == 0) {
    shmem::ShmemBarrierAllThread();
  }
}

}  // anonymous namespace

absl::Status SendSDMA(void* recv_buffer, void* send_buffer, size_t bytes, int peer,
         std::intptr_t stream_handle, int device_id) {
  auto stream = reinterpret_cast< hipStream_t >(stream_handle);

  using TT = std::tuple<application::SymmMemObjPtr, uintptr_t>;
  auto [inBuf, inOfs] = QueryMemObjPtr(send_buffer, bytes, device_id);
  auto [outBuf, outOfs] = QueryMemObjPtr(recv_buffer, bytes, device_id);
  if (MORI_UNLIKELY(inBuf.cpu == nullptr || outBuf.cpu == nullptr)) {
    return absl::InternalError(absl::StrCat("SendSDMA: Memory object not found"));
  }
  const uint32_t numQ = std::min(outBuf->sdmaNumQueue, 1u); // could be adapted to the data size
  SDMAPutKernel<<<1, 256, 0, stream>>>(shmem::ShmemMyPe(), peer, numQ, 
                             inBuf, inOfs, outBuf, outOfs, bytes);
  MORI_HIP_ERROR(hipGetLastError());
  return absl::OkStatus();
}

absl::Status BarrierOnStream(std::intptr_t stream_handle) {
  auto stream = reinterpret_cast<hipStream_t>(stream_handle);
  BarrierKernel<<<1, 1, 0, stream>>>();
  MORI_HIP_ERROR(hipGetLastError());
  return absl::OkStatus();
}

absl::Status AllGather(void* send_buffer, void* recv_buffer, size_t bytes,
      std::intptr_t stream_handle, int device_id) {
  auto stream = reinterpret_cast<hipStream_t>(stream_handle);
  int myPe = shmem::ShmemMyPe();
  auto [inBuf, inOfs] = QueryMemObjPtr(send_buffer, bytes, device_id);
  auto [outBuf, outOfs] = QueryMemObjPtr(recv_buffer, bytes, device_id);
  if (MORI_UNLIKELY(inBuf.cpu == nullptr || outBuf.cpu == nullptr)) {
    return absl::InternalError(absl::StrCat("AllGather: Memory object not found"));
  }
  const uint32_t numQ = std::min(outBuf->sdmaNumQueue, 8u); // could be adapted to the data size
  AllGatherKernel<<<1, 256, 0, stream>>>(myPe, shmem::ShmemNPes(), 
                    numQ, inBuf, inOfs, outBuf, outOfs, bytes);
  MORI_HIP_ERROR(hipGetLastError());
  return absl::OkStatus();
}

absl::Status ReduceScatter(void* send_buffer, void* recv_buffer, 
      void* staging_buffer, void* group_counters, xla::PrimitiveType dtype, 
      size_t chunkElems, std::intptr_t stream_handle, int device_id) {
  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  // the input count is the size of the output buffer (one shard)
  if (dtype != xla::PrimitiveType::F32) {
    return absl::InternalError(absl::StrCat("ReduceScatter: Unsupported data type: ", dtype));
  }
  auto stream = reinterpret_cast<hipStream_t>(stream_handle);

  using ElemT = float;
  // numElems is the TOTAL input element count (matches XLA's num_elems); the
  // per-rank output shard is chunkElems = numElems / npes.
  const size_t numElems = chunkElems * npes;
  const size_t chunkBytes = chunkElems * sizeof(ElemT);

  // if (info.deviceId == 0) {
  //   XPUT("reduce_scatter_test: %d PEs, %zu bytes/shard (%zu elems), %zu bytes input/PE", npes,
  //        chunkBytes, chunkElems, inBytes);
  
  // Channel sizing. With the receiver-side completion signal (Phase 2) the push
  // kernel no longer has a cross-block flag handoff or co-residency requirement,
  // so BOTH push and pull use full SM occupancy.
  const int numQ = 1;/// static_cast<int>(std::max(1u, baseObj->sdmaNumQueue));

  constexpr int kThreads = 256;
  constexpr int VecBytes = 16,  NumVecs = 8;
  constexpr int VecSize = VecBytes / sizeof(ElemT);
  size_t totalVecs = chunkElems / (VecSize * NumVecs);
  int wantBlocks = static_cast<int>(std::max<size_t>(1, (totalVecs + kThreads - 1) / kThreads));
  int blocks = std::min(wantBlocks, std::max(1, 256));

  // Push slicing: RS_PUSH_SLICES (default 1) rounded DOWN to a power of two and
  // clamped to [1,8]; also clamped so each slice has at least one vector. logS =
  // log2(S) in [0,3]. pushBlocks is rounded to a multiple of S (>= S) so each of
  // the S groups gets >= 1 block (G = pushBlocks/S).
  int pushSlices = 1;
  if (const char* s = std::getenv("RS_PUSH_SLICES")) {
    int req = std::atoi(s);
    if (req > 1) {
      int p = 1;
      while ((p << 1) <= req && p < 8) p <<= 1;
      pushSlices = p;
    }
  }
  {
    const size_t maxSlicesByData = std::max<size_t>(1, chunkElems / VecSize);
    while (pushSlices > 1 && static_cast<size_t>(pushSlices) > maxSlicesByData) pushSlices >>= 1;
  }
  int logS = 0;
  while ((1 << logS) < pushSlices) logS++;
  int pushBlocks = std::max(pushSlices, (blocks / pushSlices) * pushSlices);

  // Local-only per-group block counters for the push reset (never peer-written).
  // Owned/zeroed once by the communicator (a small tail of the staging buffer);
  // the kernel self-zeroes them after each launch, so no per-call memset here.
  auto* groupCounters = static_cast<uint64_t*>(group_counters);

  // The new push kernel relies on a cross-PE barrier between launches (its
  // per-slice flags are self-reset, not generation-stamped). Mirror the
  // reference's ShmemBarrierAll() before each launch.
  BarrierKernel<<<1, 1, 0, stream>>>();
  MORI_HIP_ERROR(hipGetLastError());

  ReduceScatterPushKernel<VecBytes, NumVecs, ElemT, SumOp><<<pushBlocks, kThreads, 0, stream>>>(
          myPe, npes, logS,
          static_cast<const ElemT*>(send_buffer), static_cast<ElemT*>(staging_buffer), 
          static_cast<ElemT*>(recv_buffer), groupCounters, chunkElems);
  MORI_HIP_ERROR(hipGetLastError());  
  MORI_HIP_ERROR(hipStreamSynchronize(stream));
  return absl::OkStatus();
}


}  // namespace xla_mori
