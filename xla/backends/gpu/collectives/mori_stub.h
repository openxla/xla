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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_MORI_STUB_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_MORI_STUB_H_

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

// Inert stand-in for the subset of the MORI shmem host API used by the MORI
// collectives/communicator backbone. These placeholders let the backbone
// compile and link without depending on the MORI library. All operations are
// no-ops. Replace by including the real "mori/shmem/shmem_api.hpp" once the
// MORI bindings are wired up.

#define MORI_SHMEM_UNIQUE_ID_BYTES 128

namespace mori {
namespace shmem {

using mori_shmem_uniqueid_t = std::array<uint8_t, MORI_SHMEM_UNIQUE_ID_BYTES>;

struct mori_shmem_init_attr_t {
  int32_t rank;
  int32_t nranks;
  mori_shmem_uniqueid_t uid;
  void* mpi_comm;  // Optional MPI_Comm pointer.
};

// Initialization flags.
[[maybe_unused]] constexpr unsigned int MORI_SHMEM_INIT_WITH_MPI_COMM = 0;
[[maybe_unused]] constexpr unsigned int MORI_SHMEM_INIT_WITH_UNIQUEID = 1;

inline int ShmemGetUniqueId(mori_shmem_uniqueid_t* /*uid*/) { return 0; }

inline int ShmemSetAttrUniqueIdArgs(int /*rank*/, int /*nranks*/,
                                    mori_shmem_uniqueid_t* /*uid*/,
                                    mori_shmem_init_attr_t* /*attr*/) {
  return 0;
}

inline int ShmemInitAttr(unsigned int /*flags*/,
                         mori_shmem_init_attr_t* /*attr*/) {
  return 0;
}

inline int ShmemFinalize() { return 0; }

inline int ShmemMyPe() { return 0; }

inline int ShmemNPes() { return 0; }

inline void* ShmemMalloc(size_t /*size*/) { return nullptr; }

inline void ShmemFree(void* /*ptr*/) {}

}  // namespace shmem
}  // namespace mori

// Reduction functors are named as template args by the communicator dispatch
// (::SumOp<T>, ...). The stub facade ignores Op, so forward decls suffice.
template <class T>
struct SumOp;
template <class T>
struct MaxOp;
template <class T>
struct MinOp;
template <class T>
struct ProdOp;

namespace mori {
namespace collective {

// Inert stand-in for the real MORI CollectivesFacade. Header-only, all Run* are
// no-ops returning hipSuccess. Lets the collectives/communicator wiring compile
// and link without @roc_mori. Swap for the real facade by defining
// XLA_GPU_USE_REAL_MORI (see mori_kernels.h).
class CollectivesFacade {
  CollectivesFacade() = default;

 public:
  enum class RsMode { kPush, kPull };
  using AddressVector = std::vector<std::pair<const void*, void*>>;

  CollectivesFacade(const CollectivesFacade&) = delete;
  CollectivesFacade& operator=(const CollectivesFacade&) = delete;

  static std::unique_ptr<CollectivesFacade> Create(int myPe, int nPes,
                                                   size_t /*maxStagingBytes*/) {
    auto f = std::unique_ptr<CollectivesFacade>(new CollectivesFacade());
    f->myPe_ = myPe;
    f->nPes_ = nPes;
    return f;
  }
  ~CollectivesFacade() = default;

  template <class T, class Op>
  hipError_t RunReduceScatter(const T*, T*, size_t, hipStream_t) {
    return hipSuccess;
  }
  template <class T, class Op>
  hipError_t RunAllReduce(const T*, T*, size_t, hipStream_t) {
    return hipSuccess;
  }
  template <class = void>
  hipError_t RunAllGather(const void*, void*, size_t, hipStream_t) {
    return hipSuccess;
  }
  template <class = void>
  hipError_t RunAllToAll(const AddressVector&, size_t, hipStream_t) {
    return hipSuccess;
  }
  template <class = void>
  hipError_t RunBarrier(hipStream_t) {
    return hipSuccess;
  }
  template <class = void>
  hipError_t RunSend(const void*, size_t, int, hipStream_t) {
    return hipSuccess;
  }
  template <class = void>
  hipError_t RunRecv(void*, size_t, int, hipStream_t) {
    return hipSuccess;
  }
  template <class = void>
  hipError_t RunCollectivePermute(const void*, void*, size_t, int, const int*,
                                  int, hipStream_t) {
    return hipSuccess;
  }
  template <class = void>
  hipError_t RunQuiet(hipStream_t) {
    return hipSuccess;
  }
  template <class = void>
  hipError_t RunFence() {
    return hipSuccess;
  }

 private:
  int myPe_{0};
  int nPes_{0};
};

}  // namespace collective
}  // namespace mori

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_MORI_STUB_H_
