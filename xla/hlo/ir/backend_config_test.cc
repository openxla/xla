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

#include "xla/hlo/ir/backend_config.h"

#include <cstdint>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

const int kNumThreads = 100;
const int kNumRepetitions = 100;

// This string has to be in a canonical form (without spaces and new lines)
// since the == operator does not canonicalize the raw strings before comparing
// them.
constexpr absl::string_view kRawString =
    R"({"operation_queue_id":"0","fusion_backend_config":{"kind":"__triton_gemm","triton_gemm_config":{"block_m":"256","block_n":"256","block_k":"32","split_k":"1","num_stages":"1","num_warps":"16","num_ctas":"1","is_tma_allowed":false,"is_warp_specialization_allowed":false,"waves_per_eu":"0","group_size":"0","mfma_size":"0"}},"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"})";

template <typename Input, typename CheckFn>
void RunThreaded(Input input, CheckFn check_fn) {
  for (int i = 0; i < kNumRepetitions; ++i) {
    BackendConfigWrapper source(input);

    absl::Notification all_threads_created;
    std::vector<std::unique_ptr<std::thread>> threads;

    for (int i = 0; i < kNumThreads; ++i) {
      threads.emplace_back(std::make_unique<std::thread>([&] {
        all_threads_created.WaitForNotification();
        check_fn(source);
      }));
    }
    all_threads_created.Notify();

    for (int i = 0; i < kNumThreads; ++i) {
      threads[i]->join();
    }
  }
}

TEST(BackendConfigWrapperTest, ConcurrentGetProto) {
  RunThreaded(std::string{kRawString}, [](BackendConfigWrapper& source) {
    gpu::GpuBackendConfig proto;
    TF_EXPECT_OK(source.GetProto(&proto));
    EXPECT_TRUE(proto.has_fusion_backend_config());
    BackendConfigWrapper wrapped(proto);
    EXPECT_TRUE(wrapped == source);
  });
}

TEST(BackendConfigWrapperTest, ConcurrentGetRawString) {
  BackendConfigWrapper source_json(std::string{kRawString});
  gpu::GpuBackendConfig proto;
  TF_EXPECT_OK(source_json.GetProto(&proto));

  RunThreaded(proto, [](BackendConfigWrapper& source) {
    std::string raw_string = source.GetRawString();
    EXPECT_EQ(raw_string, kRawString);
    BackendConfigWrapper wrapped(raw_string);
    EXPECT_TRUE(wrapped == source);
  });
}

TEST(BackendConfigWrapperTest, AssignmentToNonEmptyIsOK) {
  BackendConfigWrapper a(std::string{kRawString});
  BackendConfigWrapper b(std::string{kRawString});
  a = std::move(b);
  EXPECT_TRUE(a == BackendConfigWrapper(std::string{kRawString}));
}

TEST(BackendConfigWrapperTest, AssignmentDoesNotDeadlock) {
  BackendConfigWrapper source;
  BackendConfigWrapper& ref = source;
  source = std::move(ref);
}

TEST(BackendConfigWrapperTest, SelfComparisonDoesNotDeadlock) {
  BackendConfigWrapper source(std::string{kRawString});
  EXPECT_TRUE(source == source);
}

TEST(BackendConfigWrapperTest, ComparisonDoesNotDeadlock) {
  BackendConfigWrapper source_json(std::string{kRawString});
  gpu::GpuBackendConfig proto;
  TF_EXPECT_OK(source_json.GetProto(&proto));
  RunThreaded(std::string{kRawString}, [&proto](BackendConfigWrapper& source) {
    BackendConfigWrapper other_first(proto);
    EXPECT_TRUE(other_first == source);
    BackendConfigWrapper other_second(proto);
    EXPECT_TRUE(source == other_second);
  });
}

TEST(BackendConfigWrapperTest, EmptyStateIsTrackedCorrectly) {
  BackendConfigWrapper default_empty;
  EXPECT_TRUE(default_empty.empty());
  EXPECT_TRUE(default_empty == BackendConfigWrapper());

  BackendConfigWrapper empty_str("");
  EXPECT_TRUE(empty_str.empty());
  EXPECT_TRUE(default_empty == empty_str);

  BackendConfigWrapper stripped_empty(R"("wait_on_operation_queues":[],)");
  EXPECT_TRUE(stripped_empty.empty());

  BackendConfigWrapper from_str(std::string{kRawString});
  EXPECT_FALSE(from_str.empty());
  EXPECT_FALSE(default_empty == from_str);
  EXPECT_FALSE(from_str == default_empty);

  BackendConfigWrapper copied_empty(default_empty);
  EXPECT_TRUE(copied_empty.empty());

  BackendConfigWrapper copied_non_empty(from_str);
  EXPECT_FALSE(copied_non_empty.empty());

  gpu::GpuBackendConfig proto;
  TF_EXPECT_OK(from_str.GetProto(&proto));
  BackendConfigWrapper from_proto(proto);
  EXPECT_FALSE(from_proto.empty());
  EXPECT_FALSE(from_proto.GetRawString().empty());
  EXPECT_FALSE(from_proto.empty());

  // Move non-empty into empty: destination becomes non-empty.
  BackendConfigWrapper move_dst;
  move_dst = std::move(copied_non_empty);
  EXPECT_FALSE(move_dst.empty());

  // Self-move of non-empty preserves non-empty state.
  BackendConfigWrapper& move_dst_ref = move_dst;
  move_dst = std::move(move_dst_ref);
  EXPECT_FALSE(move_dst.empty());

  // Move empty into non-empty: destination becomes empty.
  move_dst = BackendConfigWrapper();
  EXPECT_TRUE(move_dst.empty());
}

TEST(BackendConfigWrapperTest, ContentIdIsStableAcrossFormatConversions) {
  BackendConfigWrapper from_string(std::string{kRawString});
  const uint64_t from_string_id = from_string.content_id();
  gpu::GpuBackendConfig proto;
  EXPECT_OK(from_string.GetProto(&proto));
  EXPECT_EQ(from_string.content_id(), from_string_id);

  BackendConfigWrapper from_proto(proto);
  const uint64_t from_proto_id = from_proto.content_id();
  EXPECT_EQ(from_proto.GetRawString(), kRawString);
  EXPECT_EQ(from_proto.content_id(), from_proto_id);
}

TEST(BackendConfigWrapperTest, ContentIdChangesWhenProtoIsMutated) {
  BackendConfigWrapper original(std::string{kRawString});
  const uint64_t original_id = original.content_id();

  BackendConfigWrapper copy(original);
  const uint64_t copy_id = copy.content_id();
  EXPECT_OK(copy.ApplyFnOnProto<gpu::GpuBackendConfig>(
      [](gpu::GpuBackendConfig* config) {
        config->set_operation_queue_id(1);
        return absl::OkStatus();
      }));
  const uint64_t mutated_id = copy.content_id();
  EXPECT_NE(mutated_id, copy_id);
  EXPECT_EQ(original.content_id(), original_id);

  // A failing function may still have mutated the proto.
  EXPECT_FALSE(copy.ApplyFnOnProto<gpu::GpuBackendConfig>(
                       [](gpu::GpuBackendConfig* config) {
                         config->set_operation_queue_id(2);
                         return absl::InternalError("failed after mutating");
                       })
                   .ok());
  EXPECT_NE(copy.content_id(), mutated_id);
}

TEST(BackendConfigWrapperTest, ContentIdChangesOnMoveAssignment) {
  BackendConfigWrapper target(std::string{kRawString});
  BackendConfigWrapper source(std::string{"{}"});
  const uint64_t target_id = target.content_id();
  const uint64_t source_id = source.content_id();
  target = std::move(source);
  EXPECT_NE(target.content_id(), target_id);
  EXPECT_NE(source.content_id(), source_id);  // NOLINT(bugprone-use-after-move)
}

TEST(BackendConfigWrapperTest, ConcurrentContentIdIsAssignedOnce) {
  BackendConfigWrapper source(std::string{kRawString});
  std::vector<uint64_t> ids(kNumThreads);
  absl::Notification all_threads_created;
  std::vector<std::unique_ptr<std::thread>> threads;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(std::make_unique<std::thread>([&, i] {
      all_threads_created.WaitForNotification();
      ids[i] = source.content_id();
    }));
  }
  all_threads_created.Notify();
  for (std::unique_ptr<std::thread>& thread : threads) {
    thread->join();
  }
  for (uint64_t id : ids) {
    EXPECT_EQ(id, ids[0]);
  }
}

}  // namespace
}  // namespace xla
