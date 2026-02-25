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

#include "xla/stream_executor/gpu/coredump.h"

#include <memory>
#include <string>

#include "absl/base/const_init.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"

namespace stream_executor::gpu {

// Performs the actual coredump trigger. Called exactly once under the mutex.
static absl::Status TriggerGpuCoredumpImpl(absl::string_view pipe_path,
                                           absl::Duration wait_duration) {
  tsl::Env* env = tsl::Env::Default();

  // Write a single byte to the named pipe to trigger GPU coredump generation.
  std::unique_ptr<tsl::WritableFile> file;

  if (absl::Status s = env->NewWritableFile(std::string(pipe_path), &file);
      !s.ok()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to open GPU coredump pipe '%s': %s. For CUDA, check that "
        "CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 and CUDA_COREDUMP_PIPE are "
        "set correctly.",
        pipe_path, s.message()));
  }

  if (absl::Status s = file->Append("1"); !s.ok()) {
    return absl::InternalError(
        absl::StrFormat("Failed to write to GPU coredump pipe '%s': %s",
                        pipe_path, s.message()));
  }

  if (absl::Status s = file->Close(); !s.ok()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to close GPU coredump pipe '%s': %s", pipe_path, s.message()));
  }

  absl::SleepFor(wait_duration);
  return absl::OkStatus();
}

absl::Status TriggerGpuCoredump(absl::string_view pipe_path,
                                absl::Duration wait_duration) {
  // Use a mutex so that only one thread triggers the coredump and all other
  // threads wait for it to complete before returning. This prevents any thread
  // from reaching LOG(FATAL) in the Abort callback while the coredump is still
  // being written.
  //
  // Use a timeout to avoid hanging forever if the coredump itself gets stuck
  // (e.g. a pipe write blocks indefinitely). After the timeout, waiting threads
  // give up and let the caller proceed to abort.
  static absl::Mutex mu(absl::kConstInit);
  static absl::Status* result = nullptr;

  // Add one second to give the trigger implementation one extra second to
  // finish, so we don't get time out errors because of the jitter.
  absl::Duration lock_wait = wait_duration + absl::Seconds(1);

  if (!mu.LockWhenWithTimeout(absl::Condition::kTrue, lock_wait)) {
    return absl::DeadlineExceededError(
        "Timed out waiting for GPU coredump trigger to complete");
  }
  absl::Cleanup unlock = [] { mu.Unlock(); };

  if (result == nullptr) {
    result = new absl::Status(TriggerGpuCoredumpImpl(pipe_path, wait_duration));
  }
  return *result;
}

}  // namespace stream_executor::gpu
