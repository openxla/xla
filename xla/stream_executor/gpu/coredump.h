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

#ifndef XLA_STREAM_EXECUTOR_GPU_COREDUMP_H_
#define XLA_STREAM_EXECUTOR_GPU_COREDUMP_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

namespace stream_executor::gpu {

// Triggers GPU coredump generation by writing to a coredump named pipe.
//
// On CUDA, the runtime supports user-triggered coredumps when the process is
// launched with the following environment variables:
//
//   CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1
//   CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1
//   CUDA_COREDUMP_FILE=<path_to_coredump_output>
//   CUDA_COREDUMP_PIPE=<path_to_named_pipe>
//   CUDA_COREDUMP_SHOW_PROGRESS=1
//
// When these are set, the CUDA runtime creates a named pipe at the path
// specified by CUDA_COREDUMP_PIPE. Writing a single byte to this pipe signals
// the CUDA runtime to generate a GPU coredump containing the state of all
// active GPU contexts.
//
// This function is designed to be called from the hang watchdog abort handler,
// where multiple threads (one per GPU guard) may attempt to trigger a coredump
// simultaneously. Only the first caller performs the actual trigger; subsequent
// concurrent callers block until the trigger completes, then return.
//
// After writing to the pipe, the function sleeps for `wait_duration` to allow
// the GPU runtime to finish writing the coredump files before the process is
// aborted.
//
// Returns OkStatus on success. Returns an error status if the pipe cannot be
// opened or written to.
//
// `pipe_path`: Path to the GPU coredump named pipe (same as CUDA_COREDUMP_PIPE)
// `wait_duration`: How long to sleep after triggering to let the GPU runtime
//                  finish writing coredumps
absl::Status TriggerGpuCoredump(absl::string_view pipe_path,
                                absl::Duration wait_duration);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_COREDUMP_H_
