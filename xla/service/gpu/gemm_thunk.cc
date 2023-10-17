/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/gemm_thunk.h"

#include <utility>

#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/precompiled_kernels.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/stream_executor/device_memory.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {

GemmThunk::GemmThunk(ThunkInfo thunk_info, GemmConfig config,
                     const BufferAllocation::Slice& lhs_buffer,
                     const BufferAllocation::Slice& rhs_buffer,
                     const BufferAllocation::Slice& output_buffer,
                     bool deterministic)
    : Thunk(Kind::kGemm, thunk_info),
      config_(std::move(config)),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      output_buffer_(output_buffer),
      deterministic_(deterministic) {}

Status GemmThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "Running GEMM thunk";
  const BufferAllocations& allocs = *params.buffer_allocations;
  TF_RETURN_IF_ERROR(RunGemm(config_, allocs.GetDeviceAddress(lhs_buffer_),
                             allocs.GetDeviceAddress(rhs_buffer_),
                             allocs.GetDeviceAddress(output_buffer_),
                             deterministic_, params.stream));
  // Cache policy reset is executed after the thunk, which it is attached to,
  // simultaneously with the next one.
  if (buffer_to_reset_.allocation() != nullptr) {
    se::Event event(params.stream->parent());
    TF_RET_CHECK(event.Init());
    params.stream->ThenRecordEvent(&event);
    params.async_comms_streams[0]->ThenWaitFor(&event);
    TF_RETURN_IF_ERROR(
        Prefetch(params.async_comms_streams[0],
                 params.buffer_allocations->GetDeviceAddress(buffer_to_reset_),
                 /*do_reset=*/true));
  }
  return OkStatus();
}

Status GemmThunk::Initialize(se::StreamExecutor* executor,
                             ExecutableSource src) {
  if (!executor->AsBlas()) {
    return absl::InternalError("Failed to initialize BLAS support");
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
