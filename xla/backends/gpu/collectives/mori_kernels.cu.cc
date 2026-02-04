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

#include "xla/backends/gpu/collectives/mori_kernels.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mori/collective/reduce_scatter_kernels.hpp"
#include "mori/core/transport/rdma/core_device_types.hpp"
#include "mori/core/transport/sdma/device_primitives.hpp"
#include "mori/core/utils/utils.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori;

#ifndef MORI_UNLIKELY
#define MORI_UNLIKELY(x) (x)
#endif

#define MORI_HIP_ERROR(status)                                    \
  if ((status) != hipSuccess) {                                   \
    return absl::InternalError(                                   \
        absl::StrCat("MORI error: ", hipGetErrorString(status))); \
  }

namespace xla_mori {

void InitSignalMemory(void* ptr, size_t bytes) {
  if (ptr != nullptr && bytes != 0) {
    (void)hipMemset(ptr, 0, bytes);
  }
}

int* AllocDeviceIntArray(const int* host_values, int count) {
  if (host_values == nullptr || count <= 0) {
    return nullptr;
  }
  int* device_ptr = nullptr;
  if (hipMalloc(reinterpret_cast<void**>(&device_ptr), count * sizeof(int)) !=
      hipSuccess) {
    return nullptr;
  }
  if (hipMemcpy(device_ptr, host_values, count * sizeof(int),
                hipMemcpyHostToDevice) != hipSuccess) {
    (void)hipFree(device_ptr);
    return nullptr;
  }
  return device_ptr;
}

void FreeDeviceArray(void* device_ptr) {
  if (device_ptr != nullptr) {
    (void)hipFree(device_ptr);
  }
}

namespace {

__global__ void BarrierKernel() { shmem::ShmemBarrierAllThread(); }

__global__ void SDMAPutKernel(int myPe, int destPe, int numQ,
                              application::SymmMemObjPtr inBuf, size_t inOfs,
                              application::SymmMemObjPtr outBuf, size_t outOfs,
                              size_t chunkSz) {
  const int tid = threadIdx.x;

  if (tid < numQ) {
    ;
    int qpId = tid;
    size_t qpChunkSz = chunkSz / numQ, qpOfs = qpId * qpChunkSz;
    if (qpId == numQ - 1) {
      qpChunkSz = chunkSz - (qpId * qpChunkSz);
    }
    // printf("inBuf: %p outBuf: %p outOfs: %zx inOfs: %zx qpOfs: %zx qpChunkSz:
    // %zx\n",
    //   inBuf->localPtr, outBuf->localPtr, outOfs, inOfs, qpOfs, qpChunkSz);
    shmem::ShmemPutMemNbiThread(outBuf, outOfs + qpOfs, inBuf, inOfs + qpOfs,
                                qpChunkSz, destPe, qpId);
    // shmem::ShmemQuietThread(destPe, outBuf);
  }
}

// All-gather across the participants of a single collective.
//
// `nranks`/`myRank` describe the collective (NOT the global MORI clique), and
// `rankToPe` maps a collective rank to its global MORI PE. This is what makes
// the kernel work when the executable uses a subset of the locally visible
// devices: iterating over `ShmemNPes()` would target PEs that never joined this
// collective (and never allocated `outBuf`), producing illegal accesses, and a
// global `ShmemBarrierAll` would deadlock waiting on non-participants.
//
// Completion uses a participant-scoped flag handshake (mirroring MORI's
// OneShotAllGatherSdmaKernel): after draining the SDMA queue to a peer we set
// our slot in the peer's `flags` to `gen`, then wait for every participant to
// set our own slots. SDMA state is read from the global heap object so the
// drain matches the raw-pointer put.
__global__ void AllGatherKernel(int myRank, int nranks,
                                const int* __restrict__ rankToPe,
                                const void* inBuf, void* outBuf, size_t chunkSz,
                                uint64_t* flags, uint64_t gen) {
  const int tid = threadIdx.x;

  // 1. Push this rank's shard into slot [myRank] on every participant.
  if (tid < nranks) {
    int destPe = rankToPe[tid];
    size_t dstOfs = static_cast<size_t>(myRank) * chunkSz;
    shmem::ShmemPutMemNbiThreadKernel<application::TransportType::SDMA>(
        static_cast<uint8_t*>(outBuf) + dstOfs,
        static_cast<const uint8_t*>(inBuf), chunkSz, destPe, 0);
  }
  __syncthreads();

  // 2. Drain the SDMA queue to each peer, then signal completion into the
  //    peer's flags[myRank].
  if (tid < nranks) {
    int destPe = rankToPe[tid];
    application::SymmMemObj* heapObj = shmem::GetGlobalGpuStatesPtr()->heapObj;
    int intraNodePe = destPe % 8;
    HSAuint64* signals =
        heapObj->signalPtrs + intraNodePe * heapObj->sdmaNumQueue;
    HSAuint64* expected =
        heapObj->expectSignalsPtr + intraNodePe * heapObj->sdmaNumQueue;
    core::SdmaQueitThread(signals, expected, heapObj->sdmaNumQueue);

    shmem::ShmemAtomicSizeNonFetchThreadKernel<
        application::TransportType::SDMA>(
        reinterpret_cast<const void*>(flags + myRank), &gen, sizeof(uint64_t),
        core::atomicType::AMO_SET, destPe, 0);
  }
  __syncthreads();

  // 3. Wait until every participant has signalled us.
  if (tid == 0) {
    for (int r = 0; r < nranks; ++r) {
      while (core::AtomicLoadRelaxed(flags + r) < gen) {
      }
    }
  }
  __syncthreads();
}

}  // anonymous namespace

absl::Status SendSDMA(void* recv_buffer, void* send_buffer, size_t bytes,
                      int peer, std::intptr_t stream_handle) {
  // auto stream = reinterpret_cast< hipStream_t >(stream_handle);
  // const uint32_t numQ = std::min(outBuf->sdmaNumQueue, 1u); // could be
  // adapted to the data size SDMAPutKernel<<<1, 256, 0,
  // stream>>>(shmem::ShmemMyPe(), peer, numQ,
  //                            inBuf, inOfs, outBuf, outOfs, bytes);
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
                       int my_rank, int num_ranks, const int* rank_to_pe,
                       void* flags_buffer, uint64_t generation,
                       std::intptr_t stream_handle) {
  auto stream = reinterpret_cast<hipStream_t>(stream_handle);
  if (num_ranks <= 0 || rank_to_pe == nullptr || flags_buffer == nullptr) {
    return absl::InvalidArgumentError(
        "AllGather: invalid participant metadata");
  }
  AllGatherKernel<<<1, 256, 0, stream>>>(
      my_rank, num_ranks, rank_to_pe, send_buffer, recv_buffer, bytes,
      static_cast<uint64_t*>(flags_buffer), generation);
  MORI_HIP_ERROR(hipGetLastError());
  return absl::OkStatus();
}

absl::Status ReduceScatter(void* send_buffer, void* recv_buffer,
                           void* staging_buffer, void* group_counters,
                           xla::PrimitiveType dtype, size_t chunkElems,
                           int my_rank, int num_ranks,
                           std::intptr_t stream_handle) {
  // Use the collective's own rank set rather than the global MORI clique
  // (ShmemMyPe/ShmemNPes), which spans every locally visible device and would
  // make the push kernel target non-participant PEs. NOTE: the push kernel
  // still treats a peer index as a global PE, so non-contiguous device subsets
  // require threading rank_to_pe into ReduceScatterPushKernel as well.
  int myPe = my_rank;
  int npes = num_ranks;

  // the input count is the size of the output buffer (one shard)
  if (dtype != xla::PrimitiveType::F32) {
    return absl::InternalError(
        absl::StrCat("ReduceScatter: Unsupported data type: ", dtype));
  }
  auto stream = reinterpret_cast<hipStream_t>(stream_handle);

  using ElemT = float;
  // numElems is the TOTAL input element count (matches XLA's num_elems); the
  // per-rank output shard is chunkElems = numElems / npes.
  const size_t numElems = chunkElems * npes;
  const size_t chunkBytes = chunkElems * sizeof(ElemT);

  // if (info.deviceId == 0) {
  //   XPUT("reduce_scatter_test: %d PEs, %zu bytes/shard (%zu elems), %zu bytes
  //   input/PE", npes,
  //        chunkBytes, chunkElems, inBytes);

  // Channel sizing. With the receiver-side completion signal (Phase 2) the push
  // kernel no longer has a cross-block flag handoff or co-residency
  // requirement, so BOTH push and pull use full SM occupancy.
  const int numQ =
      1;  /// static_cast<int>(std::max(1u, baseObj->sdmaNumQueue));

  constexpr int kThreads = 256;
  constexpr int VecBytes = 16, NumVecs = 8;
  constexpr int VecSize = VecBytes / sizeof(ElemT);
  size_t totalVecs = chunkElems / (VecSize * NumVecs);
  int wantBlocks = static_cast<int>(
      std::max<size_t>(1, (totalVecs + kThreads - 1) / kThreads));
  int blocks = std::min(wantBlocks, std::max(1, 256));

  // Push slicing: RS_PUSH_SLICES (default 1) rounded DOWN to a power of two and
  // clamped to [1,8]; also clamped so each slice has at least one vector. logS
  // = log2(S) in [0,3]. pushBlocks is rounded to a multiple of S (>= S) so each
  // of the S groups gets >= 1 block (G = pushBlocks/S).
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
    while (pushSlices > 1 && static_cast<size_t>(pushSlices) > maxSlicesByData)
      pushSlices >>= 1;
  }
  int logS = 0;
  while ((1 << logS) < pushSlices) logS++;
  int pushBlocks = std::max(pushSlices, (blocks / pushSlices) * pushSlices);

  // Local-only per-group block counters for the push reset (never
  // peer-written). Owned/zeroed once by the communicator (a small tail of the
  // staging buffer); the kernel self-zeroes them after each launch, so no
  // per-call memset here.
  auto* groupCounters = static_cast<uint32_t*>(group_counters);

  // The new push kernel relies on a cross-PE barrier between launches (its
  // per-slice flags are self-reset, not generation-stamped). Mirror the
  // reference's ShmemBarrierAll() before each launch.
  BarrierKernel<<<1, 1, 0, stream>>>();
  MORI_HIP_ERROR(hipGetLastError());

  ReduceScatterPushKernel<VecBytes, NumVecs, ElemT, SumOp>
      <<<pushBlocks, kThreads, 0, stream>>>(
          myPe, npes, logS, static_cast<const ElemT*>(send_buffer),
          static_cast<ElemT*>(staging_buffer), static_cast<ElemT*>(recv_buffer),
          groupCounters, chunkElems);
  MORI_HIP_ERROR(hipGetLastError());
  MORI_HIP_ERROR(hipStreamSynchronize(stream));
  return absl::OkStatus();
}

}  // namespace xla_mori
