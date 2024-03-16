/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/copy_thunk.h"

#include <cstdint>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

DeviceToDeviceCopyThunk::DeviceToDeviceCopyThunk(
    ThunkInfo thunk_info, const BufferAllocation::Slice &source_buffer,
    const BufferAllocation::Slice &destination_buffer, uint64_t mem_size)
    : Thunk(Kind::kCopy, thunk_info), source_buffer_(source_buffer),
      destination_buffer_(destination_buffer), mem_size_(mem_size) {}

absl::Status
DeviceToDeviceCopyThunk::ExecuteOnStream(const ExecuteParams &params) {
  se::DeviceMemoryBase destination_data =
      params.buffer_allocations->GetDeviceAddress(destination_buffer_);
  se::DeviceMemoryBase source_data =
      params.buffer_allocations->GetDeviceAddress(source_buffer_);
  VLOG(3) << "Memcpy D2D of size " << mem_size_ << " from "
          << source_data.opaque() << " to " << destination_data.opaque();
  return params.stream->Memcpy(&destination_data, source_data, mem_size_);
}

//===----------------------------------------------------------------------===//
// CopyAsyncEvents
//===----------------------------------------------------------------------===//

// Emplace() will insert {key, event} pair into the hash map,
// and return the event in order to do RecordEvent() for async memcpy.
absl::Status CopyAsyncEvents::Emplace(se::StreamExecutor *executor,
                                      const HloInstruction *instr,
                                      se::Event &&event) {
  Key key = {executor, instr};

  absl::MutexLock lock(&mutex_);
  VLOG(3) << "Emplace event " << event.implementation();
  if (auto [it, inserted] = events_.try_emplace(key, std::move(event));
      inserted) {
    return absl::OkStatus();
  }
    VLOG(3) << "ATTN: event " << event.implementation() << "already exists!";
  return absl::InternalError("Async copy event already exists!");
}

// Retrieve a completion event started by copy-start instruction
// `instr`, and remove the event from the collection.
absl::StatusOr<se::Event>
CopyAsyncEvents::Extract(se::StreamExecutor *executor,
                         const HloInstruction *instr) {

  Key key = {executor, instr};
  absl::MutexLock lock(&mutex_);
  if (auto event = events_.extract(key)) {
    VLOG(3) << "Extract event " << event.mapped().implementation();
    return std::move(event.mapped());
  }
  return absl::InternalError("Async copy event was not found!");
}

//===----------------------------------------------------------------------===//
// DeviceHostCopyThunk
//===----------------------------------------------------------------------===//
DeviceHostCopyThunk::DeviceHostCopyThunk(
    ThunkInfo thunk_info, const BufferAllocation::Slice &source_buffer,
    const BufferAllocation::Slice &destination_buffer, uint64_t mem_size,
    std::shared_ptr<CopyAsyncEvents> async_events, const HloInstruction *instr,
    bool device_to_host)
    : DeviceToDeviceCopyThunk(thunk_info, source_buffer, destination_buffer,
                              mem_size),
      async_events_(std::move(async_events)), instr_(instr),
      device_to_host_(device_to_host) {}

absl::Status DeviceHostCopyThunk::ExecuteOnStream(const ExecuteParams &params) {
  se::DeviceMemoryBase destination_data =
      params.buffer_allocations->GetDeviceAddress(destination());
  se::DeviceMemoryBase source_data =
      params.buffer_allocations->GetDeviceAddress(source());
  void *cpu_dst = destination_data.opaque();
  void *cpu_src = source_data.opaque();
  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      GetStreamForExecution(Thunk::execution_stream_id(), params));
  if (stream == params.stream) {
    if (device_to_host_) {
      VLOG(3) << "Memcpy D2H from the main stream";
      return params.stream->Memcpy(cpu_dst, source_data, size_bytes());
    } else {
      VLOG(3) << "Memcpy H2D from the main stream";
      return params.stream->Memcpy(&destination_data, cpu_src, size_bytes());
    }
  }
  // memcpy is issued from the other stream, not the main compute stream
  if (device_to_host_) {
    VLOG(3) << "Memcpy D2H from the other stream";
    TF_RETURN_IF_ERROR(stream->Memcpy(cpu_dst, source_data, size_bytes()));
  } else {
    VLOG(3) << "Memcpy H2D from the other stream";
    TF_RETURN_IF_ERROR(
        stream->Memcpy(&destination_data, cpu_src, size_bytes()));
  }
  se::StreamExecutor *executor = params.stream->parent();
  se::Event event(executor);
  if (!event.Init()) {
    return absl::InternalError(
        "Failed to initialize copy operation async completion event!");
  }
  // Record memcpy operation completion.
  TF_RETURN_IF_ERROR(stream->RecordEvent(&event));
  VLOG(3) << "Emplace events: " << event.implementation()
          << " for inst: " << instr_->ToString();
  return async_events_->Emplace(executor, instr_, std::move(event));
}

//===----------------------------------------------------------------------===//
// DeviceHostCopyDoneThunk
//===----------------------------------------------------------------------===//
DeviceHostCopyDoneThunk::DeviceHostCopyDoneThunk(
    Thunk::Kind kind, ThunkInfo thunk_info,
    std::shared_ptr<CopyAsyncEvents> async_events,
    const HloInstruction *copy_start_instr)
    : Thunk(kind, std::move(thunk_info)),
      async_events_(std::move(async_events)),
      copy_start_instr_(copy_start_instr) {}

absl::Status
DeviceHostCopyDoneThunk::ExecuteOnStream(const ExecuteParams &params) {
  VLOG(3) << "CopyDone thunk between a host and a device for: "
          << copy_start_instr_->ToString();
  se::StreamExecutor *executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(se::Event event,
                      async_events_->Extract(executor, copy_start_instr_));
  return params.stream->WaitFor(&event);
}

} // namespace gpu
} // namespace xla
