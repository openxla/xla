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

#include "xla/pjrt/common_pjrt_client.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/linearize_throttler.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/raw_pjrt_client.h"
#include "xla/pjrt/staging_buffer.h"
#include "xla/runtime/chip_id.h"
#include "xla/runtime/device_id.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

TEST(TransferIncarnationsTest, EmptySnapshotReturnsNullopt) {
  ASSERT_OK_AND_ASSIGN(auto transfer_incarnations,
                       TransferIncarnations(/*src_process_index=*/1,
                                            /*dst_process_index=*/2, {}));
  EXPECT_FALSE(transfer_incarnations.has_value());
}

TEST(TransferIncarnationsTest, SnapshotKeepsSourceThenDestination) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(2), IncarnationId(20)},
      {TaskId(1), IncarnationId(10)},
  };

  ASSERT_OK_AND_ASSIGN(
      auto forward,
      TransferIncarnations(/*src_process_index=*/2, /*dst_process_index=*/1,
                           incarnations));
  ASSERT_TRUE(forward.has_value());
  EXPECT_THAT(*forward, ElementsAre(IncarnationId(20), IncarnationId(10)));

  ASSERT_OK_AND_ASSIGN(
      auto swapped,
      TransferIncarnations(/*src_process_index=*/1, /*dst_process_index=*/2,
                           incarnations));
  ASSERT_TRUE(swapped.has_value());
  EXPECT_THAT(*swapped, ElementsAre(IncarnationId(10), IncarnationId(20)));
}

TEST(TransferIncarnationsTest, SameTaskKeepsBothIncarnations) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(3), IncarnationId(7)},
  };

  ASSERT_OK_AND_ASSIGN(
      auto transfer_incarnations,
      TransferIncarnations(/*src_process_index=*/3, /*dst_process_index=*/3,
                           incarnations));
  ASSERT_TRUE(transfer_incarnations.has_value());
  EXPECT_THAT(*transfer_incarnations,
              ElementsAre(IncarnationId(7), IncarnationId(7)));
}

TEST(TransferIncarnationsTest, MissingTaskFails) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(1), IncarnationId(10)},
  };

  EXPECT_THAT(TransferIncarnations(/*src_process_index=*/1,
                                   /*dst_process_index=*/2, incarnations),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

class CountingAsyncWorkRunner : public AsyncWorkRunner {
 public:
  void Execute(Task task) override {
    execute_count_.fetch_add(1, std::memory_order_relaxed);
    std::move(task)();
  }

  int execute_count() const {
    return execute_count_.load(std::memory_order_relaxed);
  }

  void ResetCount() { execute_count_.store(0, std::memory_order_relaxed); }

 private:
  std::atomic<int> execute_count_{0};
};

class FakeHostMemoryAllocator : public HostMemoryAllocator {
 public:
  OwnedPtr Allocate(size_t size, const AllocateOptions& options) override {
    if (size == 0) {
      Deleter nop_deleter;
      nop_deleter.deleter = +[](void*, void*) {};
      nop_deleter.arg = nullptr;
      return OwnedPtr(nullptr, nop_deleter);
    }
    Deleter array_deleter;
    array_deleter.deleter =
        +[](void* ptr, void*) { delete[] static_cast<uint8_t*>(ptr); };
    array_deleter.arg = nullptr;
    return OwnedPtr(new uint8_t[size], array_deleter);
  }
};

class FakeDeviceDescription : public PjRtDeviceDescription {
 public:
  int id() const override { return 0; }
  int process_index() const override { return 0; }
  absl::string_view device_kind() const override { return "fake"; }
  absl::string_view DebugString() const override { return "fake:0"; }
  absl::string_view ToString() const override { return "FakeDevice(id=0)"; }
  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

 private:
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_;
};

class FakeTopologyDescription : public PjRtTopologyDescription {
 public:
  PjRtPlatformId platform_id() const override { return CudaId(); }
  absl::string_view platform_name() const override { return CudaName(); }
  absl::string_view platform_version() const override { return "fake"; }
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions()
      const override {
    std::vector<std::unique_ptr<const PjRtDeviceDescription>> descriptions;
    descriptions.push_back(std::make_unique<FakeDeviceDescription>());
    return descriptions;
  }
  absl::StatusOr<uint64_t> Fingerprint() const override { return 0; }
  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }
  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override {
    return LayoutUtil::MakeDescendingLayout(dims.size());
  }

 private:
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_;
};

class FakeDeviceEventPromise : public PjRtDeviceEventPromise {
 public:
  explicit FakeDeviceEventPromise(tsl::RCReference<tsl::IndirectAsyncValue> av)
      : av_(std::move(av)) {}

  PjRtDeviceEventPtr event() const override {
    return PjRtDeviceEventPtr(tsl::AsyncValuePtr<bool>(av_.get()));
  }

  void Set(PjRtDeviceEventRef event) override {
    av_->ForwardTo(event.down_cast<bool>());
  }

  void SetError(absl::Status s) override { av_->SetError(std::move(s)); }

  void SetReady() override {
    av_->ForwardTo(tsl::MakeAvailableAsyncValueRef<bool>(true));
  }

 private:
  tsl::RCReference<tsl::IndirectAsyncValue> av_;
};

class FakeRawBuffer : public CommonPjRtRawBufferImpl {
 public:
  FakeRawBuffer(PjRtMemorySpace* memory_space, size_t size_bytes)
      : memory_space_(memory_space), storage_(size_bytes, 0) {}

  PjRtMemorySpace* memory_space() const override { return memory_space_; }

  size_t GetOnDeviceSizeInBytes() const override { return storage_.size(); }

  void* OpaqueDeviceMemoryDataPointer() const override {
    return const_cast<uint8_t*>(storage_.data());
  }

  absl::StatusOr<PjRtDeviceEventRef> CopyRawHostToDeviceAndReturnEvent(
      const void* src, int64_t offset, int64_t transfer_size,
      PjRtDeviceEventRefVector dependencies) override {
    if (offset < 0 || transfer_size < 0 ||
        static_cast<size_t>(offset + transfer_size) > storage_.size()) {
      return absl::OutOfRangeError("H2D copy out of bounds");
    }
    if (transfer_size > 0) {
      std::memcpy(storage_.data() + offset, src,
                  static_cast<size_t>(transfer_size));
    }
    return PjRtDeviceEventRef(tsl::MakeAvailableAsyncValueRef<bool>(true));
  }

  absl::StatusOr<PjRtDeviceEventRef> CopyRawDeviceToHostAndReturnEvent(
      void* dst, int64_t offset, int64_t transfer_size,
      PjRtDeviceEventRefVector dependencies) override {
    return absl::UnimplementedError("D2H not implemented");
  }

  absl::StatusOr<PjRtRawBufferRef> Slice(int64_t offset,
                                         int64_t size) override {
    return absl::UnimplementedError("Slice not implemented");
  }

  absl::StatusOr<PjRtDeviceEventRef> MakeAllocationReadyEvent() override {
    return PjRtDeviceEventRef(tsl::MakeAvailableAsyncValueRef<bool>(true));
  }

  void CopyTo(
      PjRtRawBufferRef dst_raw_buffer,
      PjRtDeviceEventPromiseRef definition_event_promise,
      PjRtDeviceEventPromiseRef src_usage_event_promise,
      absl::AnyInvocable<void(absl::Status) &&> allocation_event) override {
    definition_event_promise.SetError(
        absl::UnimplementedError("CopyTo not implemented"));
  }

  PjRtDeviceEventPtr GetRawBufferAsyncValue() override { return {}; }

  absl::Span<const uint8_t> bytes() const { return storage_; }

 private:
  PjRtMemorySpace* memory_space_;
  std::vector<uint8_t> storage_;
};

class FakeRawClient : public PjRtRawClient {
 public:
  FakeRawClient(AsyncWorkRunner* async_work_runner,
                HostMemoryAllocator* host_allocator)
      : async_work_runner_(async_work_runner),
        host_allocator_(host_allocator) {}

  AsyncWorkRunner* async_work_runner() const override {
    return async_work_runner_;
  }

  HostMemoryAllocator* GetHostMemoryAllocator() const override {
    return host_allocator_;
  }

  absl::StatusOr<PjRtRawBufferRef> AllocateRawBuffer(
      PjRtMemorySpace* memory_space, size_t on_device_bytes_count,
      bool retry_on_oom, tsl::AsyncValueRef<bool> allocate_after) override {
    return tsl::MakeRef<FakeRawBuffer>(memory_space, on_device_bytes_count);
  }

  absl::StatusOr<std::pair<PjRtDeviceEventPromiseRef, PjRtDeviceEventRef>>
  CreateLinkedEventPromise(LocalDeviceId local_device_id, int memory_kind_id,
                           absl::string_view debug_info) override {
    tsl::RCReference<tsl::IndirectAsyncValue> av =
        tsl::MakeIndirectAsyncValue();
    PjRtDeviceEventRef event = PjRtDeviceEventRef(tsl::AsyncValueRef<bool>(av));
    PjRtDeviceEventPromiseRef promise(
        tsl::MakeRef<FakeDeviceEventPromise>(std::move(av)));
    return std::make_pair(std::move(promise), std::move(event));
  }

  absl::StatusOr<PjRtDeviceEventRef> CreateDeviceEvent(
      LocalDeviceId local_device_id, int memory_kind_id,
      Future<> dependency) override {
    return absl::UnimplementedError("CreateDeviceEvent not implemented");
  }

  absl::StatusOr<PjRtRawBufferRef> ImportForeignMemory(
      PjRtMemorySpace* memory_space, void* device_ptr, size_t size,
      absl::AnyInvocable<void() &&> on_delete_callback,
      bool is_mutable) override {
    return absl::UnimplementedError("ImportForeignMemory not implemented");
  }

  tsl::AsyncValueRef<PjRtExecutable> ToAsyncExecutable(
      std::shared_ptr<PjRtExecutable> executable) const override {
    return {};
  }

  std::pair<tsl::AsyncValueRef<PjRtExecutable>,
            PjRtFulfillAsyncExecutableCallback>
  CreateLinkedExecutablePromise() const override {
    return {tsl::AsyncValueRef<PjRtExecutable>(),
            [](absl::StatusOr<tsl::AsyncValueRef<PjRtExecutable>>) {}};
  }

 private:
  AsyncWorkRunner* async_work_runner_;
  HostMemoryAllocator* host_allocator_;
};

class InlineAsyncLinearizeTest : public ::testing::Test {
 protected:
  void InitClient(size_t max_inflight_linearization_bytes) {
    PjRtPluginAttributes plugin_attributes;
    plugin_attributes.pjrt_c_api_major_version = 0;
    plugin_attributes.pjrt_c_api_minor_version = 0;
    plugin_attributes.attributes["should_stage_host_to_device_transfers"] =
        true;

    LinearizeThrottler::Options throttler_options;
    throttler_options.max_inflight_linearization_bytes =
        max_inflight_linearization_bytes;

    client_ = std::make_unique<CommonPjRtClient>(
        CudaId(), CudaName(), /*platform_version=*/"fake",
        /*process_index=*/0, std::make_shared<FakeTopologyDescription>(),
        std::make_unique<FakeRawClient>(&runner_, &host_allocator_),
        /*kv_store=*/nullptr, std::move(plugin_attributes), throttler_options);

    auto device = std::make_unique<CommonPjRtDevice>(
        std::make_unique<FakeDeviceDescription>(), LocalDeviceId(0),
        LocalChipId(0), /*is_addressable=*/true, client_.get());
    auto memory_space =
        std::make_unique<PinnedHostMemorySpace>(/*id=*/0, device.get());
    memory_space_ = memory_space.get();
    device->AttachMemorySpace(memory_space_, /*is_default=*/true);

    std::vector<std::unique_ptr<PjRtDevice>> devices;
    devices.push_back(std::move(device));
    std::vector<std::unique_ptr<PjRtMemorySpace>> memory_spaces;
    memory_spaces.push_back(std::move(memory_space));
    client_->AttachDevices(std::move(devices), std::move(memory_spaces));
  }

  CountingAsyncWorkRunner runner_;
  FakeHostMemoryAllocator host_allocator_;
  std::unique_ptr<CommonPjRtClient> client_;
  PjRtMemorySpace* memory_space_ = nullptr;
};

TEST_F(InlineAsyncLinearizeTest, SmallStagedBufferLinearizesInline) {
  InitClient(/*max_inflight_linearization_bytes=*/1024 * 1024);

  constexpr int64_t kNumElements = 1024;  // 4 KiB <= 512 KiB
  std::vector<int32_t> host_data(kNumElements);
  std::iota(host_data.begin(), host_data.end(), 1);
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {kNumElements}, {0});
  tsl::RCReference<FakeRawBuffer> raw_buffer = tsl::MakeRef<FakeRawBuffer>(
      memory_space_, static_cast<size_t>(kNumElements * sizeof(int32_t)));

  bool host_buffer_released = false;
  ASSERT_OK_AND_ASSIGN(
      PjRtDeviceEventRef event,
      client_->LinearizeHostBufferInto(
          host_data.data(), S32, shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
          [&host_buffer_released]() { host_buffer_released = true; }, shape,
          raw_buffer));

  EXPECT_EQ(runner_.execute_count(), 0);
  EXPECT_TRUE(host_buffer_released);
  ASSERT_NE(event.async_value(), nullptr);
  EXPECT_TRUE(event.async_value()->IsAvailable());
  EXPECT_FALSE(event.GetErrorIfPresent().has_value());
  EXPECT_EQ(std::memcmp(raw_buffer->bytes().data(), host_data.data(),
                        host_data.size() * sizeof(int32_t)),
            0);
}

TEST_F(InlineAsyncLinearizeTest, LargeStagedBufferDispatchesToAsyncWorkRunner) {
  InitClient(/*max_inflight_linearization_bytes=*/4 * 1024 * 1024);

  constexpr int64_t kNumElements = 262144;  // 1 MiB > 512 KiB
  std::vector<int32_t> host_data(kNumElements);
  std::iota(host_data.begin(), host_data.end(), 7);
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {kNumElements}, {0});
  tsl::RCReference<FakeRawBuffer> raw_buffer = tsl::MakeRef<FakeRawBuffer>(
      memory_space_, static_cast<size_t>(kNumElements * sizeof(int32_t)));

  bool host_buffer_released = false;
  ASSERT_OK_AND_ASSIGN(
      PjRtDeviceEventRef event,
      client_->LinearizeHostBufferInto(
          host_data.data(), S32, shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
          [&host_buffer_released]() { host_buffer_released = true; }, shape,
          raw_buffer));

  EXPECT_EQ(runner_.execute_count(), 1);
  EXPECT_TRUE(host_buffer_released);
  ASSERT_NE(event.async_value(), nullptr);
  EXPECT_TRUE(event.async_value()->IsAvailable());
  EXPECT_FALSE(event.GetErrorIfPresent().has_value());
  EXPECT_EQ(std::memcmp(raw_buffer->bytes().data(), host_data.data(),
                        host_data.size() * sizeof(int32_t)),
            0);
}

TEST_F(InlineAsyncLinearizeTest, ZeroByteBufferCompletesCleanly) {
  InitClient(/*max_inflight_linearization_bytes=*/1024 * 1024);

  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {0}, {0});
  tsl::RCReference<FakeRawBuffer> raw_buffer =
      tsl::MakeRef<FakeRawBuffer>(memory_space_, /*size_bytes=*/0);

  int32_t dummy = 0;
  bool host_buffer_released = false;
  ASSERT_OK_AND_ASSIGN(
      PjRtDeviceEventRef event,
      client_->LinearizeHostBufferInto(
          &dummy, S32, shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
          [&host_buffer_released]() { host_buffer_released = true; }, shape,
          raw_buffer));

  EXPECT_EQ(runner_.execute_count(), 1);
  EXPECT_TRUE(host_buffer_released);
  ASSERT_NE(event.async_value(), nullptr);
  EXPECT_TRUE(event.async_value()->IsAvailable());
  EXPECT_FALSE(event.GetErrorIfPresent().has_value());
}

TEST_F(InlineAsyncLinearizeTest,
       SmallBufferFallsBackToAsyncWhenThrottlerExhausted) {
  constexpr size_t kThrottlerLimitBytes = 64 * 1024;
  InitClient(kThrottlerLimitBytes);

  std::vector<uint8_t> credit_holder(kThrottlerLimitBytes, 0);
  tsl::AsyncValueRef<PjRtStagingBuffer> held_credit =
      client_->linearize_throttler()->CreateFromData(
          credit_holder.data(), credit_holder.size(), []() {});
  ASSERT_TRUE(held_credit.IsAvailable());

  constexpr int64_t kNumElements = 1024;  // 4 KiB <= 512 KiB
  std::vector<int32_t> host_data(kNumElements);
  std::iota(host_data.begin(), host_data.end(), 100);
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {kNumElements}, {0});
  tsl::RCReference<FakeRawBuffer> raw_buffer = tsl::MakeRef<FakeRawBuffer>(
      memory_space_, static_cast<size_t>(kNumElements * sizeof(int32_t)));

  bool host_buffer_released = false;
  ASSERT_OK_AND_ASSIGN(
      PjRtDeviceEventRef event,
      client_->LinearizeHostBufferInto(
          host_data.data(), S32, shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
          [&host_buffer_released]() { host_buffer_released = true; }, shape,
          raw_buffer));

  // Staging credit is unavailable, so LinearizeHostBufferInto must return
  // immediately without blocking or copying yet.
  EXPECT_EQ(runner_.execute_count(), 0);
  EXPECT_FALSE(host_buffer_released);
  ASSERT_NE(event.async_value(), nullptr);
  EXPECT_FALSE(event.async_value()->IsAvailable());

  // Releasing the held credit schedules the throttled staging allocation and
  // the H2D dispatch callback onto `runner_`.
  held_credit.reset();
  EXPECT_GT(runner_.execute_count(), 0);
  EXPECT_TRUE(host_buffer_released);
  EXPECT_TRUE(event.async_value()->IsAvailable());
  EXPECT_FALSE(event.GetErrorIfPresent().has_value());
  EXPECT_EQ(std::memcmp(raw_buffer->bytes().data(), host_data.data(),
                        host_data.size() * sizeof(int32_t)),
            0);
}

TEST_F(InlineAsyncLinearizeTest,
       SmallStagedBufferInvokesOnDoneCallbackOnLinearizeError) {
  InitClient(/*max_inflight_linearization_bytes=*/1024 * 1024);

  std::vector<int32_t> host_data = {1, 2, 3, 4};
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {4}, {0});
  const std::vector<int64_t> invalid_strides = {4, 4};  // rank 2 != dims rank 1
  tsl::RCReference<FakeRawBuffer> raw_buffer =
      tsl::MakeRef<FakeRawBuffer>(memory_space_, 4 * sizeof(int32_t));

  bool host_buffer_released = false;
  ASSERT_OK_AND_ASSIGN(
      PjRtDeviceEventRef event,
      client_->LinearizeHostBufferInto(
          host_data.data(), S32, shape.dimensions(), invalid_strides,
          PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
          [&host_buffer_released]() { host_buffer_released = true; }, shape,
          raw_buffer));

  EXPECT_EQ(runner_.execute_count(), 0);
  EXPECT_TRUE(host_buffer_released);
  ASSERT_NE(event.async_value(), nullptr);
  EXPECT_TRUE(event.async_value()->IsAvailable());
  ASSERT_TRUE(event.GetErrorIfPresent().has_value());
  EXPECT_THAT(*event.GetErrorIfPresent(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla
