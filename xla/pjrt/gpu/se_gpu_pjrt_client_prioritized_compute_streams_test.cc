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

// Tests for prioritized compute streams in the StreamExecutor GPU client: an
// execution carrying the `execution_priority` custom option runs on the
// compute stream with the closest priority, prioritized compute streams run
// concurrently without recycling each other's memory, and a high-priority
// execution is not stuck behind work queued on the default compute stream.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/custom_options.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client_test_helper.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/se/local_device_state.h"
#include "xla/pjrt/se/pjrt_stream_executor_client.h"
#include "xla/pjrt/se/prioritized_compute_stream.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream.h"
#include "xla/tests/literal_test_util.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::VariantWith;

// A small program whose output is cheap to verify.
constexpr absl::string_view kCriticalProgram = R"(
  HloModule critical

  ENTRY main {
    p = f32[4]{0} parameter(0)
    two = f32[] constant(2)
    bcast = f32[4]{0} broadcast(two), dimensions={}
    mul = f32[4]{0} multiply(p, bcast)
    ROOT add = f32[4]{0} add(mul, bcast)
  })";

// A program that keeps the GPU busy for a few milliseconds per run and needs
// temporaries (the dot results) that are torn down right after launch.
constexpr absl::string_view kBackgroundProgram = R"(
  HloModule background

  ENTRY main {
    p = f32[2048,2048]{1,0} parameter(0)
    d1 = f32[2048,2048]{1,0} dot(p, p), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    a1 = f32[2048,2048]{1,0} add(d1, p)
    d2 = f32[2048,2048]{1,0} dot(a1, p), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT a2 = f32[2048,2048]{1,0} add(d2, p)
  })";

GpuClientOptions ClientOptionsWithPriorities(std::vector<int> priorities) {
  GpuClientOptions options = GetTestGpuClientOptions();
  options.compute_stream_priorities = std::move(priorities);
  return options;
}

ExecuteOptions WithPriority(int64_t priority) {
  ExecuteOptions options;
  options.custom_options =
      std::make_shared<const CustomOptions>(CustomOptions::Map{
          {std::string(kExecutionPriorityCustomOption),
           CustomOptions::Value(std::in_place_type<int64_t>, priority)}});
  return options;
}

absl::StatusOr<LocalDeviceState*> GetLocalDeviceState(PjRtClient* client) {
  auto* common_client = absl::down_cast<CommonPjRtClient*>(client);
  auto* raw_client = absl::down_cast<PjRtStreamExecutorRawClient*>(
      common_client->raw_client());
  return raw_client->GetLocalDeviceState(
      client->addressable_devices().front()->local_device_id());
}

absl::StatusOr<std::intptr_t> GetDefinitionStreamHandle(PjRtBuffer* buffer) {
  ABSL_RETURN_IF_ERROR(buffer->GetReadyFuture().Await());
  auto* common_buffer = dynamic_cast<CommonPjRtBuffer*>(buffer);
  TF_RET_CHECK(common_buffer != nullptr);
  auto hold =
      common_buffer->GetBufferWithHold(CommonPjRtBuffer::ScopedHold::kUsage);
  ABSL_RETURN_IF_ERROR(hold.status());
  absl::Span<const PjRtDeviceEventRef> definition_events =
      hold.buffer()->definition_events();
  TF_RET_CHECK(definition_events.size() == 1);
  std::optional<PjRtDeviceEventPtr::DefinitionStreamInfo> stream_info =
      definition_events.front().ptr().GetDefinitionStream();
  TF_RET_CHECK(stream_info.has_value());
  return stream_info->stream;
}

std::intptr_t StreamHandle(se::Stream* stream) {
  return absl::bit_cast<std::intptr_t>(
      stream->platform_specific_handle().stream);
}

// Uploads `literal` and waits for the transfer to finish, so that callers may
// pass temporaries (BufferFromHostLiteral requires the literal to stay alive
// until the transfer completes) and so that executions below only measure
// compute.
absl::StatusOr<std::unique_ptr<PjRtBuffer>> ToDevice(PjRtClient* client,
                                                     const Literal& literal) {
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(
          literal,
          *client->addressable_devices().front()->default_memory_space()));
  ABSL_RETURN_IF_ERROR(buffer->GetReadyFuture().Await());
  return buffer;
}

TEST(PrioritizedComputeStreamsTest, NoneConfiguredKeepsDefaultBehavior) {
  ASSERT_OK_AND_ASSIGN(auto client,
                       GetStreamExecutorGpuClient(GetTestGpuClientOptions()));
  ASSERT_OK_AND_ASSIGN(LocalDeviceState * device_state,
                       GetLocalDeviceState(client.get()));
  EXPECT_FALSE(device_state->has_prioritized_compute_streams());
  EXPECT_EQ(device_state->num_prioritized_compute_streams(), 1);
  PrioritizedComputeStream* default_stream =
      device_state->prioritized_compute_stream(0);
  ASSERT_NE(default_stream, nullptr);
  EXPECT_TRUE(default_stream->is_default());
  EXPECT_EQ(default_stream->stream(), device_state->compute_stream());
  EXPECT_EQ(default_stream->priority(), 0);
  EXPECT_EQ(default_stream->allocator(), nullptr);
  EXPECT_EQ(device_state->ComputeStreamForPriority(-3), default_stream);
  EXPECT_TRUE(device_state->allocation_sync_points_enabled());

  ASSERT_OK_AND_ASSIGN(auto executable,
                       CompileExecutable(kCriticalProgram, *client));
  ASSERT_OK_AND_ASSIGN(
      auto input,
      ToDevice(client.get(), LiteralUtil::CreateR1<float>({1, 2, 3, 4})));
  // The custom option is accepted but has no effect without additional
  // prioritized compute streams.
  ASSERT_OK_AND_ASSIGN(auto results,
                       executable->Execute({{input.get()}}, WithPriority(-1)));
  ASSERT_OK_AND_ASSIGN(std::intptr_t stream,
                       GetDefinitionStreamHandle(results[0][0].get()));
  EXPECT_EQ(stream, StreamHandle(device_state->compute_stream()));
  ASSERT_OK_AND_ASSIGN(auto literal, results[0][0]->ToLiteral().Await());
  EXPECT_THAT(literal->data<float>(), ElementsAre(4, 6, 8, 10));
}

TEST(PrioritizedComputeStreamsTest, ExecutesOnStreamWithClosestPriority) {
  // Pass unsorted priorities with duplicates (including 0, which becomes the
  // default stream) to verify deduplication and descending-priority sorting.
  ASSERT_OK_AND_ASSIGN(
      auto client,
      GetStreamExecutorGpuClient(ClientOptionsWithPriorities({-3, -1, -3, 0})));
  ASSERT_OK_AND_ASSIGN(LocalDeviceState * device_state,
                       GetLocalDeviceState(client.get()));
  ASSERT_TRUE(device_state->has_prioritized_compute_streams());
  ASSERT_EQ(device_state->num_prioritized_compute_streams(), 3);
  EXPECT_FALSE(device_state->allocation_sync_points_enabled());
  PrioritizedComputeStream* default_stream =
      device_state->prioritized_compute_stream(0);
  EXPECT_EQ(device_state->default_prioritized_compute_stream(), default_stream);
  PrioritizedComputeStream* stream1 =
      device_state->prioritized_compute_stream(1);
  PrioritizedComputeStream* stream2 =
      device_state->prioritized_compute_stream(2);
  EXPECT_TRUE(default_stream->is_default());
  EXPECT_FALSE(stream1->is_default());
  EXPECT_NE(default_stream->allocator(), nullptr);
  EXPECT_NE(stream1->allocator(), nullptr);

  // The additional streams exist, are distinct, and carry the requested
  // priority.
  se::Stream* se_stream1 = stream1->stream();
  se::Stream* se_stream2 = stream2->stream();
  ASSERT_NE(se_stream1, nullptr);
  ASSERT_NE(se_stream2, nullptr);
  EXPECT_NE(se_stream1, device_state->compute_stream());
  EXPECT_NE(se_stream1, se_stream2);
  EXPECT_EQ(stream1->priority(), -1);
  EXPECT_EQ(stream2->priority(), -3);
  EXPECT_THAT(se_stream1->priority(), VariantWith<int>(-1));
  EXPECT_THAT(se_stream2->priority(), VariantWith<int>(-3));

  // Priority -> stream mapping: exact matches, clamping and ties.
  EXPECT_EQ(device_state->ComputeStreamForPriority(0), default_stream);
  EXPECT_EQ(device_state->ComputeStreamForPriority(7), default_stream);
  EXPECT_EQ(device_state->ComputeStreamForPriority(-1), stream1);
  EXPECT_EQ(device_state->ComputeStreamForPriority(-3), stream2);
  EXPECT_EQ(device_state->ComputeStreamForPriority(-5), stream2);
  // -2 is equidistant from -1 and -3; ties go to the higher priority.
  EXPECT_EQ(device_state->ComputeStreamForPriority(-2), stream2);

  ASSERT_OK_AND_ASSIGN(auto executable,
                       CompileExecutable(kCriticalProgram, *client));
  ASSERT_OK_AND_ASSIGN(
      auto input,
      ToDevice(client.get(), LiteralUtil::CreateR1<float>({1, 2, 3, 4})));

  struct Case {
    std::optional<int64_t> priority;
    se::Stream* expected_stream;
  };
  const Case cases[] = {
      {std::nullopt, device_state->compute_stream()},
      {0, device_state->compute_stream()},
      {-1, se_stream1},
      {-3, se_stream2},
      {-9, se_stream2},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.priority.has_value() ? absl::StrCat(*c.priority)
                                        : std::string("none"));
    ExecuteOptions options =
        c.priority.has_value() ? WithPriority(*c.priority) : ExecuteOptions();
    ASSERT_OK_AND_ASSIGN(auto results,
                         executable->Execute({{input.get()}}, options));
    ASSERT_OK_AND_ASSIGN(std::intptr_t stream,
                         GetDefinitionStreamHandle(results[0][0].get()));
    EXPECT_EQ(stream, StreamHandle(c.expected_stream));
    ASSERT_OK_AND_ASSIGN(auto literal, results[0][0]->ToLiteral().Await());
    EXPECT_THAT(literal->data<float>(), ElementsAre(4, 6, 8, 10));
  }
}

// Without priority 0 in the list, the lowest configured priority becomes the
// default compute stream and no priority-0 stream is created.
TEST(PrioritizedComputeStreamsTest, LowestPriorityBecomesDefaultStream) {
  ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(
                                        ClientOptionsWithPriorities({-3, -1})));
  ASSERT_OK_AND_ASSIGN(LocalDeviceState * device_state,
                       GetLocalDeviceState(client.get()));
  ASSERT_EQ(device_state->num_prioritized_compute_streams(), 2);
  PrioritizedComputeStream* default_stream =
      device_state->default_prioritized_compute_stream();
  PrioritizedComputeStream* high_stream =
      device_state->prioritized_compute_stream(1);
  EXPECT_TRUE(default_stream->is_default());
  EXPECT_EQ(default_stream->stream(), device_state->compute_stream());
  EXPECT_EQ(default_stream->priority(), -1);
  EXPECT_THAT(device_state->compute_stream()->priority(), VariantWith<int>(-1));
  EXPECT_EQ(high_stream->priority(), -3);
  // Requests at or below the lowest configured priority clamp to the default
  // stream; the H2D stream inherits the default priority (harmless, copy
  // engines ignore it).
  EXPECT_EQ(device_state->ComputeStreamForPriority(0), default_stream);
  EXPECT_EQ(device_state->ComputeStreamForPriority(-1), default_stream);
  EXPECT_EQ(device_state->ComputeStreamForPriority(-2), high_stream);
  EXPECT_THAT(device_state->host_to_device_stream()->priority(),
              VariantWith<int>(-1));

  ASSERT_OK_AND_ASSIGN(auto executable,
                       CompileExecutable(kCriticalProgram, *client));
  ASSERT_OK_AND_ASSIGN(
      auto input,
      ToDevice(client.get(), LiteralUtil::CreateR1<float>({1, 2, 3, 4})));
  ASSERT_OK_AND_ASSIGN(auto results,
                       executable->Execute({{input.get()}}, WithPriority(0)));
  ASSERT_OK_AND_ASSIGN(std::intptr_t stream,
                       GetDefinitionStreamHandle(results[0][0].get()));
  EXPECT_EQ(stream, StreamHandle(device_state->compute_stream()));
  ASSERT_OK_AND_ASSIGN(auto literal, results[0][0]->ToLiteral().Await());
  EXPECT_THAT(literal->data<float>(), ElementsAre(4, 6, 8, 10));
}

// Runs background work on the default compute stream and critical work on a
// higher-priority compute stream concurrently, many times over, and checks
// that every result is correct. The background program allocates temporaries
// that are released right after launch; without completion-ordered
// deallocation the higher-priority stream could reuse them while the
// background kernels are still reading them.
TEST(PrioritizedComputeStreamsTest, ConcurrentStreamsProduceCorrectResults) {
  ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(
                                        ClientOptionsWithPriorities({0, -1})));
  ASSERT_OK_AND_ASSIGN(auto background,
                       CompileExecutable(kBackgroundProgram, *client));
  ASSERT_OK_AND_ASSIGN(auto critical,
                       CompileExecutable(kCriticalProgram, *client));

  Literal background_input =
      LiteralUtil::CreateFull<float>({2048, 2048}, 0.001f);
  ASSERT_OK_AND_ASSIGN(auto background_arg,
                       ToDevice(client.get(), background_input));
  ASSERT_OK_AND_ASSIGN(
      auto critical_arg,
      ToDevice(client.get(), LiteralUtil::CreateR1<float>({1, 2, 3, 4})));

  // Reference background result from an uncontended run.
  ASSERT_OK_AND_ASSIGN(
      auto reference_results,
      background->Execute({{background_arg.get()}}, ExecuteOptions()));
  ASSERT_OK_AND_ASSIGN(auto reference,
                       reference_results[0][0]->ToLiteral().Await());

  constexpr int kIterations = 40;
  std::vector<std::unique_ptr<PjRtBuffer>> background_outputs;
  std::vector<std::unique_ptr<PjRtBuffer>> critical_outputs;
  for (int i = 0; i < kIterations; ++i) {
    ASSERT_OK_AND_ASSIGN(auto bg, background->Execute({{background_arg.get()}},
                                                      WithPriority(0)));
    background_outputs.push_back(std::move(bg[0][0]));
    ASSERT_OK_AND_ASSIGN(
        auto cr, critical->Execute({{critical_arg.get()}}, WithPriority(-1)));
    critical_outputs.push_back(std::move(cr[0][0]));
    // Drop every other background output immediately so that its memory is
    // released (deferred) while the stream is still busy.
    if (i % 2 == 0) {
      background_outputs.back().reset();
    }
  }

  for (int i = 0; i < kIterations; ++i) {
    SCOPED_TRACE(i);
    ASSERT_OK_AND_ASSIGN(auto literal,
                         critical_outputs[i]->ToLiteral().Await());
    EXPECT_THAT(literal->data<float>(), ElementsAre(4, 6, 8, 10));
    if (background_outputs[i] != nullptr) {
      ASSERT_OK_AND_ASSIGN(auto bg_literal,
                           background_outputs[i]->ToLiteral().Await());
      EXPECT_TRUE(LiteralTestUtil::Equal(*reference, *bg_literal));
    }
  }
}

// Queues several background runs on the default compute stream, then issues
// one critical run. On a higher-priority compute stream the critical run
// completes long before the queued background work would have let it through
// in FIFO order.
TEST(PrioritizedComputeStreamsTest, HighPriorityBypassesQueuedBackgroundWork) {
  ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(
                                        ClientOptionsWithPriorities({0, -1})));
  ASSERT_OK_AND_ASSIGN(auto background,
                       CompileExecutable(kBackgroundProgram, *client));
  ASSERT_OK_AND_ASSIGN(auto critical,
                       CompileExecutable(kCriticalProgram, *client));

  Literal background_input =
      LiteralUtil::CreateFull<float>({2048, 2048}, 0.001f);
  ASSERT_OK_AND_ASSIGN(auto background_arg,
                       ToDevice(client.get(), background_input));
  ASSERT_OK_AND_ASSIGN(
      auto critical_arg,
      ToDevice(client.get(), LiteralUtil::CreateR1<float>({1, 2, 3, 4})));

  // Warm up both executables (autotuning, command buffer instantiation).
  for (int i = 0; i < 2; ++i) {
    ASSERT_OK_AND_ASSIGN(auto bg, background->Execute({{background_arg.get()}},
                                                      ExecuteOptions()));
    ASSERT_OK(bg[0][0]->GetReadyFuture().Await());
    ASSERT_OK_AND_ASSIGN(
        auto cr, critical->Execute({{critical_arg.get()}}, ExecuteOptions()));
    ASSERT_OK(cr[0][0]->GetReadyFuture().Await());
  }

  constexpr int kQueuedBackgroundRuns = 12;
  auto measure =
      [&](int64_t critical_priority) -> absl::StatusOr<absl::Duration> {
    std::vector<std::unique_ptr<PjRtBuffer>> keep_alive;
    absl::Time queue_start = absl::Now();
    for (int i = 0; i < kQueuedBackgroundRuns; ++i) {
      ABSL_ASSIGN_OR_RETURN(
          auto bg,
          background->Execute({{background_arg.get()}}, WithPriority(0)));
      keep_alive.push_back(std::move(bg[0][0]));
    }
    absl::Time start = absl::Now();
    ABSL_ASSIGN_OR_RETURN(auto cr,
                          critical->Execute({{critical_arg.get()}},
                                            WithPriority(critical_priority)));
    ABSL_RETURN_IF_ERROR(cr[0][0]->GetReadyFuture().Await());
    absl::Duration critical_latency = absl::Now() - start;
    for (auto& buffer : keep_alive) {
      ABSL_RETURN_IF_ERROR(buffer->GetReadyFuture().Await());
    }
    LOG(INFO) << "critical priority " << critical_priority << ": latency "
              << critical_latency << ", background backlog drained in "
              << (absl::Now() - queue_start);
    return critical_latency;
  };

  ASSERT_OK_AND_ASSIGN(absl::Duration fifo_latency, measure(/*priority=*/0));
  ASSERT_OK_AND_ASSIGN(absl::Duration prioritized_latency,
                       measure(/*priority=*/-1));
  LOG(INFO) << "FIFO latency: " << fifo_latency
            << ", prioritized stream latency: " << prioritized_latency;
  // The FIFO run has to wait for the whole backlog; the prioritized run only
  // for the kernel in flight. Require a comfortable margin to stay robust.
  EXPECT_LT(prioritized_latency * 2, fifo_latency);
}

}  // namespace
}  // namespace xla
