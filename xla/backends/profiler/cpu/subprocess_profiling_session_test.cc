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

#include "xla/backends/profiler/cpu/subprocess_profiling_session.h"  // IWYU pragma: keep

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/backends/profiler/cpu/subprocess_registry.h"
#include "xla/tsl/platform/resource_loader.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/platform/test.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace cpu {
namespace {

using ::testing::IsEmpty;
using ::testing::Not;

std::unique_ptr<tsl::SubProcess> CreateSubProcess(
    const std::vector<std::string>& args) {
  auto subprocess = std::make_unique<tsl::SubProcess>();
  subprocess->SetProgram(args[0], args);
  subprocess->SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_DUPPARENT);
  subprocess->SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_DUPPARENT);
  return subprocess;
}

class SubprocessProfilingSessionTest : public ::testing::Test {
 public:
  void SetUpSubprocesses(int num_subprocesses) {
    std::string subprocess_main_path = tsl::GetDataDependencyFilepath(
        "tensorflow/compiler/xla/backends/profiler/cpu/subprocess_main");
    ASSERT_TRUE(subprocesses_.empty());
    subprocesses_.resize(num_subprocesses);
    for (int i = 0; i < num_subprocesses; ++i) {
      std::vector<std::string> args;
      args.push_back(subprocess_main_path);
      int port = tsl::testing::PickUnusedPortOrDie();
      args.push_back(absl::StrCat("--port=", port));
      args.push_back("--vmodule=subprocess_main=10");

      SubProcessRuntime& subprocess_runtime = subprocesses_[i];
      subprocess_runtime.port = port;
      subprocess_runtime.subprocess = CreateSubProcess(args);
      ASSERT_TRUE(subprocess_runtime.subprocess->Start());
      ASSERT_OK(
          RegisterSubprocess(subprocess_runtime.port, subprocess_runtime.port));
    }
    // Wait for connections to be established.
    absl::SleepFor(absl::Seconds(1));
  }

  void TearDown() override {
    for (auto& subprocess_runtime : subprocesses_) {
      ASSERT_NE(subprocess_runtime.subprocess, nullptr);
      ASSERT_TRUE(subprocess_runtime.subprocess->Kill(SIGKILL));
      ASSERT_OK(UnregisterSubprocess(subprocess_runtime.port));
    }
  }

  struct SubProcessRuntime {
    std::unique_ptr<tsl::SubProcess> subprocess;
    int port;
  };
  std::vector<SubProcessRuntime> subprocesses_;
};

TEST_F(SubprocessProfilingSessionTest, MultipleSubprocessesIntegrationTest) {
  SetUpSubprocesses(3);

  auto session =
      tsl::ProfilerSession::Create(tsl::ProfilerSession::DefaultOptions());
  absl::SleepFor(absl::Milliseconds(100));  // profile for 100ms
  tensorflow::profiler::XSpace space;
  ASSERT_OK(session->CollectData(&space));

  ASSERT_THAT(space.planes(), Not(IsEmpty()));

  absl::flat_hash_set<std::string> trace_me_names;
  for (const auto& plane : space.planes()) {
    const auto& event_metadata = plane.event_metadata();
    for (const auto& line : plane.lines()) {
      for (const auto& event : line.events()) {
        trace_me_names.insert(event_metadata.at(event.metadata_id()).name());
      }
    }
  }
  absl::flat_hash_set<std::string> expected_trace_me_names;
  for (const auto& subproc : subprocesses_) {
    // The pid is actually the port number since tsl::SubProcess does not
    // expose the pid.
    expected_trace_me_names.insert(
        absl::StrCat("subprocess_test_", subproc.port));
  }
  EXPECT_EQ(trace_me_names, expected_trace_me_names);
}

}  // namespace
}  // namespace cpu
}  // namespace profiler
}  // namespace xla
