/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/platform/statusor.h"
#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_event.h"

namespace stream_executor {
namespace gpu {

namespace sycl = ::sycl;

Event::Status GpuEvent::PollForStatus() {
  auto* event = gpu_event_->event;
  auto event_status =
      event->get_info<sycl::info::event::command_execution_status>();

  switch (event_status) {
    case sycl::info::event_command_status::submitted:
    case sycl::info::event_command_status::running:
      return Event::Status::kPending;
    case sycl::info::event_command_status::complete:
      return Event::Status::kComplete;
    default:
      return Event::Status::kUnknown;
  }
}

}  // namespace gpu
}  // namespace stream_executor
