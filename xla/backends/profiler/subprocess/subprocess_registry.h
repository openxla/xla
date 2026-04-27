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

#ifndef XLA_BACKENDS_PROFILER_SUBPROCESS_SUBPROCESS_REGISTRY_H_
#define XLA_BACKENDS_PROFILER_SUBPROCESS_SUBPROCESS_REGISTRY_H_

#include <sys/types.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "third_party/gloop/util/functional/auto_function_runner.h"
#include "tsl/profiler/protobuf/profiler_service.grpc.pb.h"

namespace xla {
namespace profiler {
namespace subprocess {

// Information about a registered subprocess.
struct SubprocessInfo {
  int32_t pid;
  std::string address;
  std::shared_ptr<tensorflow::grpc::ProfilerService::Stub> profiler_stub;

  template <typename H>
  friend H AbslHashValue(H h, const SubprocessInfo& subprocess) {
    return H::combine(std::move(h), subprocess.pid);
  }

  bool operator==(const SubprocessInfo& other) const {
    return pid == other.pid;
  }

  bool operator!=(const SubprocessInfo& other) const {
    return !(*this == other);
  }

  std::string DebugString() const;
};

// Registers a subprocess that has a running HTTP server listening on the given
// port or Unix domain socket, so that it can be profiled using the
// subprocess profiler.
// It is expected to call `RegisterSubprocess` after starting the subprocess.
// This method will create a stub to the ProfilerService in the subprocess, and
// may block for a while until the stub is ready or connection times out.
// RETURNS: an error if the subprocess is already registered or if the
// subprocess stub cannot be created.
absl::StatusOr<util::functional::AutoFunctionRunner> RegisterSubprocess(
    int32_t pid, std::optional<int> port,
    std::optional<absl::string_view> unix_domain_socket);

// Returns all currently registered subprocesses.
std::vector<SubprocessInfo> GetRegisteredSubprocesses();

}  // namespace subprocess
}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_SUBPROCESS_SUBPROCESS_REGISTRY_H_
