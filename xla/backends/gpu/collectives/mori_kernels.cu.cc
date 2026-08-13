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

// This is the single device translation unit for the MORI XLA collectives. It
// is compiled as HIP, so including mori_kernels.h here pulls in the facade's
// device path (kernels + Run* definitions). The explicit float instantiations
// below emit the symbols that the host mori_communicator.cc references through
// the decl-only host path of the same header.
#include "xla/backends/gpu/collectives/mori_kernels.h"

namespace mori {
namespace collective {

#define PREFIX(NAME) template hipError_t CollectivesFacade::NAME

// Emit RunReduceScatter/RunAllReduce symbols for every (dtype, op) combo.
#define MORI_INST(PT, CT, RK, OP)                                            \
  PREFIX(RunReduceScatter)<CT, OP<CT>>(const CT*, CT*, size_t, hipStream_t); \
  PREFIX(RunAllReduce)<CT, OP<CT>>(const CT*, CT*, size_t, hipStream_t);
#define MORI_INST_DTYPE(PT, CT) MORI_FOR_EACH_OP(MORI_INST, PT, CT)
MORI_FOR_EACH_DTYPE(MORI_INST_DTYPE)

PREFIX(RunAllGather)<>(const void*, void*, size_t, hipStream_t);
PREFIX(RunAllToAll)<>(const CollectivesFacade::AddressVector&, size_t,
                      hipStream_t);
PREFIX(RunBarrier)<>(hipStream_t);
PREFIX(RunSend)<>(const void*, size_t, int, hipStream_t);
PREFIX(RunRecv)<>(void*, size_t, int, hipStream_t);
PREFIX(RunCollectivePermute)<>(const void*, void*, size_t, int, const int*, int,
                               hipStream_t);
PREFIX(RunQuiet)<>(hipStream_t);
PREFIX(RunFence)<>();

#undef MORI_INST_DTYPE
#undef MORI_INST
#undef PREFIX

}  // namespace collective
}  // namespace mori
