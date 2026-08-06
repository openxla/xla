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

// This is the single device translation unit for the MORI XLA collectives. It
// is compiled as HIP, so including mori_kernels.h here pulls in the facade's
// device path (kernels + Run* definitions). The explicit float instantiations
// below emit the symbols that the host mori_communicator.cc references through
// the decl-only host path of the same header.
#include "xla/backends/gpu/collectives/mori_kernels.h"

namespace mori {
namespace collective {

template void CollectivesFacade::RunReduceScatter<float, ::SumOp<float>>(
    const float*, float*, size_t, hipStream_t);
template void CollectivesFacade::RunAllReduce<float, ::SumOp<float>>(
    const float*, float*, size_t, hipStream_t);
template void CollectivesFacade::RunAllGather<float>(const float*, float*,
                                                     size_t, hipStream_t);
template void CollectivesFacade::RunAllToAll<float>(const float* const*,
                                                    float* const*, size_t,
                                                    hipStream_t);
template void CollectivesFacade::RunBarrier<void>(hipStream_t);

}  // namespace collective
}  // namespace mori
