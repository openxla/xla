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

using namespace mori;

#ifndef MORI_UNLIKELY
#define MORI_UNLIKELY(x) (x)
#endif

#define MORI_HIP_ERROR(status) \
  if ((status) != hipSuccess) { \
    return absl::InternalError(absl::StrCat("MORI error: ", hipGetErrorString(status))); \
  }

namespace roc_mori {

static std::mutex g_memObjMapMutex;
using MemObjMap = std::map< void*, application::SymmMemObjPtr >;
static std::deque<MemObjMap> g_memObjMap(8);

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

constexpr int kWarpSize = 64, kWarpsPerBlock = 2;
constexpr int kBlockSize = kWarpSize * kWarpsPerBlock;
constexpr int kMaxBlocks = 16;
constexpr size_t kBytesPerWarp = 2*1024;

// --------------------------------------------------------------------------
// MoriPutKernel – single kernel that:
//   1. Copies data from local send_buffer to peer's recv_buffer via P2P.
//   2. Uses a retirement counter so that the LAST block to finish sets
//      a completion flag on the remote peer's signal_flags[myPe].
//
// signal_flags  – base of the per-PE signal array (in symmetric heap).
// block_counter – a single uint32_t in device memory, initialised to 0.
//                 It is used only within this kernel and reset before exit.
// --------------------------------------------------------------------------
__global__ void MoriPutKernel(void* recv_buffer, void* send_buffer,
                              size_t bytes, int peer,
                              uint32_t* signal_flags) {
  using T = uint8_t;
  T *src = static_cast<T *>(send_buffer), *dst;
  uint32_t *remote_sig;
  {
    int myPe = shmem::ShmemMyPe();
    // Translate the local symmetric address of recv_buffer to the
    // P2P-mapped address on the remote peer.
    uint64_t remote_addr = shmem::ShmemPtrP2p(
      reinterpret_cast<uint64_t>(recv_buffer), myPe, peer);
    dst = reinterpret_cast<T*>(remote_addr);
    remote_addr = shmem::ShmemPtrP2p(
      reinterpret_cast<uint64_t>(signal_flags + myPe + 1), myPe, peer);
    remote_sig = reinterpret_cast<uint32_t*>(remote_addr);
    if (threadIdx.x == 0) {
      while (core::AtomicLoadRelaxedSystem(remote_sig) != 0) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
  }
  __syncthreads();

  uint32_t warpId = blockIdx.x * blockDim.x + threadIdx.x,
           totalWarps = gridDim.x * blockDim.x;
  if (warpSize == 64) {
    warpId /= 64, totalWarps /= 64;
  } else {
    warpId /= 32, totalWarps /= 32;
  }
  for (size_t off = static_cast<size_t>(warpId) * kBytesPerWarp; off < bytes;
              off += static_cast<size_t>(totalWarps) * kBytesPerWarp) {
    size_t n = std::min(kBytesPerWarp, bytes - off);
    core::WarpCopy<T>(dst + off, src + off, n);
  }

  // Ensure all P2P writes from this thread are globally visible.
  __threadfence_system();
  // Intra-block barrier: every thread in this block has completed its
  // WarpCopy + fence before we touch the retirement counter.
  __syncthreads();

  if (threadIdx.x == 0) {
    // no need to transfer this flag => it lives on this node
    auto prev = core::AtomicAddRelaxed(signal_flags, uint32_t{1});
    if (prev + 1 == gridDim.x) { // all blocks are done => set global flag
      core::AtomicStoreRelaxed(signal_flags, uint32_t{0}); // reset counter
      core::AtomicStoreRelaxedSystem(remote_sig, uint32_t{1});
    }
  }
}

// --------------------------------------------------------------------------
// RecvWaitKernel – spins on a local signal location until the remote
// peer has written a non-zero value (via MoriPutKernel), then resets it.
// --------------------------------------------------------------------------
__global__ void RecvWaitKernel(uint32_t* signal) {
  while (core::AtomicLoadRelaxedSystem(signal) == 0) {
    __builtin_amdgcn_s_sleep(1);
  }
  // Reset for the next round.
  core::AtomicStoreRelaxedSystem(signal, uint32_t{0});
  // __threadfence_system(); ???
}

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
/*
inline __device__ void ShmemPutMemNbiThreadKernel<application::TransportType::SDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe, int qpId) {*/

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

void InitSignalMemory(void* ptr, size_t bytes) {
  hipMemset(ptr, 0, bytes);
}

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

int Send(void* recv_buffer, void* send_buffer, size_t bytes, int peer,
         uint32_t* signal_flags, std::intptr_t stream_handle) {

  size_t total = kBytesPerWarp * kWarpsPerBlock;

  // 4K - handled by 1 block
  // 8K - by 2 blocks
  // anything until 6K - also handled by 1 block
  
  int numBlocks = static_cast<int>((bytes + total / 2) / total);
  numBlocks = std::max(1, std::min(numBlocks, kMaxBlocks));

  // fprintf(stderr, "MORI send Using blocks %d\n", numBlocks);

  auto stream = reinterpret_cast< hipStream_t >(stream_handle);
  MoriPutKernel<<<numBlocks, kBlockSize, 0, stream>>>(
      recv_buffer, send_buffer, bytes, peer, signal_flags);
  return 0;
}

int Recv(void* /*recv_buffer*/, void* /*send_buffer*/, size_t /*bytes*/,
         int peer, uint32_t* signal_flags, std::intptr_t stream_handle) {

  auto stream = reinterpret_cast< hipStream_t >(stream_handle);
  RecvWaitKernel<<<1, 1, 0, stream>>>(&signal_flags[peer + 1]);
  return 0;
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

}  // namespace roc_mori
