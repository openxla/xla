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

#include "xla/service/gpu/gpu_executable_buffer_allocator.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/mock_platform.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/platform.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::A;
using ::testing::HasSubstr;
using ::testing::Invoke;
using ::testing::NiceMock;

class TestDeviceAddressAllocator : public se::DeviceAddressAllocator {
 public:
  struct AllocationRequest {
    int device_ordinal;
    uint64_t size;
    bool retry_on_failure;
    int64_t memory_space;
  };

  TestDeviceAddressAllocator(const se::Platform* platform, se::Stream* stream)
      : se::DeviceAddressAllocator(platform), stream_(stream) {}

  ~TestDeviceAddressAllocator() override {
    for (void* allocation : allocations_) {
      std::free(allocation);
    }
  }

  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override {
    allocation_requests_.push_back(
        {device_ordinal, size, retry_on_failure, memory_space});
    if (size == 0) {
      return se::ScopedDeviceAddress<uint8_t>();
    }
    void* allocation = std::malloc(size);
    if (allocation == nullptr) {
      return absl::ResourceExhaustedError("malloc failed");
    }
    allocations_.insert(allocation);
    return se::ScopedDeviceAddress<uint8_t>(
        se::DeviceAddressBase(allocation, size), device_ordinal, this);
  }

  absl::Status Deallocate(int /*device_ordinal*/,
                          se::DeviceAddressBase address) override {
    if (address.is_null()) {
      return absl::OkStatus();
    }
    auto it = allocations_.find(address.opaque());
    if (it == allocations_.end()) {
      return absl::InvalidArgumentError("allocation not found");
    }
    ++deallocation_attempt_count_;
    if (fail_deallocation_) {
      return absl::ResourceExhaustedError("deallocation failed");
    }
    std::free(*it);
    allocations_.erase(it);
    ++deallocation_count_;
    return absl::OkStatus();
  }

  absl::StatusOr<se::Stream*> GetStream(int /*device_ordinal*/) override {
    return stream_;
  }

  bool Owns(se::DeviceAddressBase address) const {
    return allocations_.find(address.opaque()) != allocations_.end();
  }

  const std::vector<AllocationRequest>& allocation_requests() const {
    return allocation_requests_;
  }

  int deallocation_count() const { return deallocation_count_; }
  int deallocation_attempt_count() const { return deallocation_attempt_count_; }
  void set_fail_deallocation(bool fail_deallocation) {
    fail_deallocation_ = fail_deallocation;
  }

 private:
  se::Stream* stream_;
  std::set<void*> allocations_;
  std::vector<AllocationRequest> allocation_requests_;
  int deallocation_count_ = 0;
  int deallocation_attempt_count_ = 0;
  bool fail_deallocation_ = false;
};

class GpuExecutableBufferAllocatorTest : public ::testing::Test {
 protected:
  using DonationState = GpuExecutableBufferAllocator::DonationState;
  using AliasKind = HloInputOutputAliasConfig::AliasKind;
  using OutputBufferSource = GpuExecutableBufferAllocator::OutputBufferSource;
  using OutputBufferSpec = GpuExecutableBufferAllocator::OutputBufferSpec;
  using OutputBufferSpecMap = GpuExecutableBufferAllocator::OutputBufferSpecMap;

  GpuExecutableBufferAllocatorTest() : memory_allocator_(&platform_, &stream_) {
    run_options_.mutable_run_options()->set_stream(&stream_);
    run_options_.mutable_run_options()->set_allocator(&memory_allocator_);
  }

  void Initialize(Shape result_shape, std::vector<BufferAllocation> allocations,
                  OutputBufferSpecMap output_buffer_specs) {
    buffer_allocator_.reset();
    result_shape_ = std::move(result_shape);
    allocations_ = std::move(allocations);
    allocation_ptrs_.clear();
    allocation_ptrs_.reserve(allocations_.size());
    for (const BufferAllocation& allocation : allocations_) {
      allocation_ptrs_.push_back(&allocation);
    }
    buffer_allocator_ = std::make_unique<GpuExecutableBufferAllocator>(
        "test", allocation_ptrs_, result_shape_, std::move(output_buffer_specs),
        /*debug_options=*/nullptr, /*thunk_executor=*/nullptr);
  }

  static OutputBufferSpecMap ScalarOutputSpec(
      bool passthrough, std::optional<AliasKind> alias_kind,
      BufferAllocation::Index allocation_index = 0) {
    OutputBufferSpecMap output_buffer_specs;
    output_buffer_specs.emplace(
        ShapeIndex{},
        OutputBufferSpec{allocation_index, passthrough, alias_kind});
    return output_buffer_specs;
  }

  absl::StatusOr<std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope>>
  CreateScope() {
    return buffer_allocator_->CreateExecutionScope(
        &run_options_, &memory_allocator_, /*device_ordinal=*/0);
  }

  NiceMock<se::MockPlatform> platform_;
  NiceMock<se::MockStream> stream_;
  TestDeviceAddressAllocator memory_allocator_;
  ServiceExecutableRunOptions run_options_;
  Shape result_shape_;
  std::vector<BufferAllocation> allocations_;
  std::vector<const BufferAllocation*> allocation_ptrs_;
  std::unique_ptr<GpuExecutableBufferAllocator> buffer_allocator_;
};

TEST_F(GpuExecutableBufferAllocatorTest, RejectsMissingOutputBufferSpec) {
  Initialize(ShapeUtil::MakeShape(S32, {}),
             {BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)},
             /*output_buffer_specs=*/{});
  ASSERT_OK_AND_ASSIGN(auto scope, CreateScope());

  int32_t value = 42;
  std::vector<se::DeviceAddressBase> addresses = {
      se::DeviceAddressBase(&value, sizeof(value))};
  BufferAllocations buffer_allocations(addresses, /*device_ordinal=*/0,
                                       &memory_allocator_);

  EXPECT_FALSE(buffer_allocator_->HasOutputBuffer(/*index=*/{}));
  EXPECT_THAT(scope->ResolveOutputBuffer(
                  &run_options_, buffer_allocations, /*index=*/{},
                  [](const BufferAllocation&) {
                    ADD_FAILURE()
                        << "Donation resolver called for missing output";
                    return DonationState::kUnavailable;
                  },
                  /*buffer_allocations_debug_summary=*/"debug summary"),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("No output buffer specification")));
  EXPECT_TRUE(memory_allocator_.allocation_requests().empty());
}

TEST_F(GpuExecutableBufferAllocatorTest, DetectsAllOutputLeavesAliased) {
  Shape tuple_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(S32, {}), ShapeUtil::MakeShape(S32, {})});
  OutputBufferSpecMap output_buffer_specs;
  output_buffer_specs.emplace(
      ShapeIndex{0},
      OutputBufferSpec{/*allocation_index=*/0, /*passthrough=*/false,
                       AliasKind::kMayAlias});
  output_buffer_specs.emplace(
      ShapeIndex{1},
      OutputBufferSpec{/*allocation_index=*/1, /*passthrough=*/false,
                       AliasKind::kMustAlias});
  Initialize(tuple_shape,
             {BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
              BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0)},
             std::move(output_buffer_specs));

  EXPECT_TRUE(buffer_allocator_->all_output_leaves_aliased());

  output_buffer_specs.clear();
  output_buffer_specs.emplace(
      ShapeIndex{0},
      OutputBufferSpec{/*allocation_index=*/0, /*passthrough=*/false,
                       AliasKind::kMayAlias});
  output_buffer_specs.emplace(
      ShapeIndex{1}, OutputBufferSpec{/*allocation_index=*/1,
                                      /*passthrough=*/false, std::nullopt});
  Initialize(std::move(tuple_shape),
             {BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
              BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0)},
             std::move(output_buffer_specs));

  EXPECT_FALSE(buffer_allocator_->all_output_leaves_aliased());
}

TEST_F(GpuExecutableBufferAllocatorTest, RejectsKnownMissingMustAliasDonation) {
  Initialize(ShapeUtil::MakeShape(S32, {}),
             {BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)},
             ScalarOutputSpec(/*passthrough=*/false, AliasKind::kMustAlias));
  ASSERT_OK_AND_ASSIGN(auto scope, CreateScope());

  int32_t value = 42;
  std::vector<se::DeviceAddressBase> addresses = {
      se::DeviceAddressBase(&value, sizeof(value))};
  BufferAllocations buffer_allocations(addresses, /*device_ordinal=*/0,
                                       &memory_allocator_);
  EXPECT_THAT(
      scope->ResolveOutputBuffer(
          &run_options_, buffer_allocations, /*index=*/{},
          [](const BufferAllocation&) { return DonationState::kNotDonated; },
          /*buffer_allocations_debug_summary=*/"debug summary"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("must-alias")));
  EXPECT_TRUE(memory_allocator_.allocation_requests().empty());
}

TEST_F(GpuExecutableBufferAllocatorTest,
       CopyProtectsMustAliasWhenDonationIsUnavailable) {
  constexpr int64_t kMemorySpace = 7;
  Initialize(ShapeUtil::MakeShape(S32, {}),
             {BufferAllocation(/*index=*/0, /*size=*/4,
                               /*color=*/kMemorySpace)},
             ScalarOutputSpec(/*passthrough=*/false, AliasKind::kMustAlias));
  ASSERT_OK_AND_ASSIGN(auto scope, CreateScope());

  int32_t value = 42;
  se::DeviceAddressBase original(&value, sizeof(value));
  std::vector<se::DeviceAddressBase> addresses = {original};
  BufferAllocations buffer_allocations(addresses, /*device_ordinal=*/0,
                                       &memory_allocator_);
  EXPECT_CALL(stream_, Memcpy(A<se::DeviceAddressBase*>(),
                              A<const se::DeviceAddressBase&>(), sizeof(value)))
      .WillOnce(Invoke([](se::DeviceAddressBase* destination,
                          const se::DeviceAddressBase& source, uint64_t size) {
        std::memcpy(destination->opaque(), source.opaque(), size);
        return absl::OkStatus();
      }));
  ASSERT_OK_AND_ASSIGN(
      GpuExecutableBufferAllocator::ResolvedOutputBuffer resolved,
      scope->ResolveOutputBuffer(
          &run_options_, buffer_allocations, /*index=*/{},
          [](const BufferAllocation&) { return DonationState::kUnavailable; },
          /*buffer_allocations_debug_summary=*/"debug summary"));

  EXPECT_EQ(resolved.source, OutputBufferSource::kCopyProtected);
  EXPECT_FALSE(resolved.fell_back_to_assigned_buffer);
  EXPECT_NE(resolved.buffer.opaque(), original.opaque());
  EXPECT_EQ(*static_cast<const int32_t*>(resolved.buffer.opaque()), value);
  EXPECT_EQ(buffer_allocations.GetDeviceAddress(0), resolved.buffer);
  ASSERT_EQ(memory_allocator_.allocation_requests().size(), 1);
  EXPECT_EQ(memory_allocator_.allocation_requests()[0].size, sizeof(value));
  EXPECT_TRUE(memory_allocator_.allocation_requests()[0].retry_on_failure);
  EXPECT_EQ(memory_allocator_.allocation_requests()[0].memory_space,
            kMemorySpace);
  EXPECT_THAT(
      memory_allocator_.Deallocate(/*device_ordinal=*/0, resolved.buffer),
      StatusIs(absl::StatusCode::kOk));
}

TEST_F(GpuExecutableBufferAllocatorTest, ReusesDonatedOutputBuffer) {
  Initialize(ShapeUtil::MakeShape(S32, {}),
             {BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)},
             ScalarOutputSpec(/*passthrough=*/false, AliasKind::kMayAlias));
  ASSERT_OK_AND_ASSIGN(auto scope, CreateScope());

  int32_t value = 42;
  se::DeviceAddressBase original(&value, sizeof(value));
  std::vector<se::DeviceAddressBase> addresses = {original};
  BufferAllocations buffer_allocations(addresses, /*device_ordinal=*/0,
                                       &memory_allocator_);
  ASSERT_OK_AND_ASSIGN(
      GpuExecutableBufferAllocator::ResolvedOutputBuffer resolved,
      scope->ResolveOutputBuffer(
          &run_options_, buffer_allocations, /*index=*/{},
          [](const BufferAllocation&) { return DonationState::kDonated; },
          /*buffer_allocations_debug_summary=*/"debug summary"));

  EXPECT_EQ(resolved.source, OutputBufferSource::kDonated);
  EXPECT_EQ(resolved.buffer, original);
  EXPECT_FALSE(resolved.fell_back_to_assigned_buffer);
  EXPECT_TRUE(memory_allocator_.allocation_requests().empty());
}

TEST_F(GpuExecutableBufferAllocatorTest,
       NonDonatedPassthroughUsesAssignedBuffer) {
  Initialize(ShapeUtil::MakeShape(S32, {}),
             {BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)},
             ScalarOutputSpec(/*passthrough=*/true, AliasKind::kMayAlias));
  ASSERT_OK_AND_ASSIGN(auto scope, CreateScope());

  int32_t value = 42;
  se::DeviceAddressBase original(&value, sizeof(value));
  std::vector<se::DeviceAddressBase> addresses = {original};
  BufferAllocations buffer_allocations(addresses, /*device_ordinal=*/0,
                                       &memory_allocator_);
  ASSERT_OK_AND_ASSIGN(
      GpuExecutableBufferAllocator::ResolvedOutputBuffer resolved,
      scope->ResolveOutputBuffer(
          &run_options_, buffer_allocations, /*index=*/{},
          [](const BufferAllocation&) { return DonationState::kNotDonated; },
          /*buffer_allocations_debug_summary=*/"debug summary"));

  EXPECT_EQ(resolved.source, OutputBufferSource::kAssigned);
  EXPECT_EQ(resolved.buffer, original);
  EXPECT_TRUE(resolved.fell_back_to_assigned_buffer);
  EXPECT_TRUE(memory_allocator_.allocation_requests().empty());
}

TEST_F(GpuExecutableBufferAllocatorTest,
       ExecuteAndTearDownPreservesResolvedOutputs) {
  std::vector<BufferAllocation> allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0)};
  allocations[0].set_maybe_live_out(true);
  allocations[1].set_maybe_live_out(true);
  Initialize(ShapeUtil::MakeShape(S32, {}), std::move(allocations),
             ScalarOutputSpec(/*passthrough=*/false, std::nullopt));
  ASSERT_OK_AND_ASSIGN(auto scope, CreateScope());

  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> result_allocation,
      memory_allocator_.Allocate(/*device_ordinal=*/0, /*size=*/4,
                                 /*retry_on_failure=*/true,
                                 /*memory_space=*/0));
  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> dead_allocation,
      memory_allocator_.Allocate(/*device_ordinal=*/0, /*size=*/4,
                                 /*retry_on_failure=*/true,
                                 /*memory_space=*/0));
  se::DeviceAddressBase result_address = result_allocation.Release();
  se::DeviceAddressBase dead_address = dead_allocation.Release();
  std::vector<se::DeviceAddressBase> addresses = {result_address, dead_address};
  BufferAllocations buffer_allocations(addresses, /*device_ordinal=*/0,
                                       &memory_allocator_);
  ASSERT_OK_AND_ASSIGN(
      GpuExecutableBufferAllocator::ResolvedOutputBuffer resolved,
      scope->ResolveOutputBuffer(
          &run_options_, buffer_allocations, /*index=*/{},
          [](const BufferAllocation&) {
            ADD_FAILURE() << "Donation resolver called for unaliased output";
            return DonationState::kUnavailable;
          },
          /*buffer_allocations_debug_summary=*/"debug summary"));

  bool executed = false;
  ASSERT_OK(scope->ExecuteAndTearDown(
      buffer_allocations, /*device_ordinal=*/0,
      [&](const BufferAllocations&,
          std::optional<absl::Span<const BufferAllocation::Index>>) {
        executed = true;
        return absl::OkStatus();
      }));

  EXPECT_TRUE(executed);
  EXPECT_TRUE(memory_allocator_.Owns(resolved.buffer));
  EXPECT_FALSE(memory_allocator_.Owns(dead_address));
  EXPECT_EQ(memory_allocator_.deallocation_count(), 1);
  EXPECT_THAT(
      memory_allocator_.Deallocate(/*device_ordinal=*/0, resolved.buffer),
      StatusIs(absl::StatusCode::kOk));
}

TEST_F(GpuExecutableBufferAllocatorTest,
       ExecuteErrorWinsTeardownErrorAndTearDownRuns) {
  std::vector<BufferAllocation> allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0)};
  allocations[0].set_maybe_live_out(true);
  allocations[1].set_maybe_live_out(true);
  Initialize(ShapeUtil::MakeShape(S32, {}), std::move(allocations),
             ScalarOutputSpec(/*passthrough=*/false, std::nullopt));
  ASSERT_OK_AND_ASSIGN(auto scope, CreateScope());

  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> result_allocation,
      memory_allocator_.Allocate(/*device_ordinal=*/0, /*size=*/4,
                                 /*retry_on_failure=*/true,
                                 /*memory_space=*/0));
  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> dead_allocation,
      memory_allocator_.Allocate(/*device_ordinal=*/0, /*size=*/4,
                                 /*retry_on_failure=*/true,
                                 /*memory_space=*/0));
  se::DeviceAddressBase result_address = result_allocation.Release();
  se::DeviceAddressBase dead_address = dead_allocation.Release();
  std::vector<se::DeviceAddressBase> addresses = {result_address, dead_address};
  BufferAllocations buffer_allocations(addresses, /*device_ordinal=*/0,
                                       &memory_allocator_);
  ASSERT_OK_AND_ASSIGN(
      GpuExecutableBufferAllocator::ResolvedOutputBuffer resolved,
      scope->ResolveOutputBuffer(
          &run_options_, buffer_allocations, /*index=*/{},
          [](const BufferAllocation&) {
            ADD_FAILURE() << "Donation resolver called for unaliased output";
            return DonationState::kUnavailable;
          },
          /*buffer_allocations_debug_summary=*/"debug summary"));

  memory_allocator_.set_fail_deallocation(true);
  absl::Status status = scope->ExecuteAndTearDown(
      buffer_allocations, /*device_ordinal=*/0,
      [](const BufferAllocations&,
         std::optional<absl::Span<const BufferAllocation::Index>>) {
        return absl::InternalError("execution failed");
      });

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal,
                               HasSubstr("execution failed")));
  EXPECT_EQ(memory_allocator_.deallocation_attempt_count(), 1);
  EXPECT_TRUE(memory_allocator_.Owns(resolved.buffer));
  EXPECT_TRUE(memory_allocator_.Owns(dead_address));

  memory_allocator_.set_fail_deallocation(false);
  EXPECT_THAT(
      memory_allocator_.Deallocate(/*device_ordinal=*/0, resolved.buffer),
      StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(memory_allocator_.Deallocate(/*device_ordinal=*/0, dead_address),
              StatusIs(absl::StatusCode::kOk));
}

}  // namespace
}  // namespace xla::gpu
