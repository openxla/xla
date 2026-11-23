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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_CUMEM_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_CUMEM_H_

namespace stream_executor {

// Returns true if NCCL_CUMEM_ENABLE >= 1, which RCCL requires to use its cuMem
// allocator, and with it symmetric memory windows. It defaults to 0 on ROCm.
// Read once, like RCCL's own ncclCuMemEnable().
bool IsNcclCuMemEnabled();

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_CUMEM_H_
