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

#include "xla/service/gpu/gpu_executable_va_remap_allocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/executable_run_options.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable_buffer_allocator.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_address_vmm_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/mock_platform.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::NiceMock;
using ::testing::Return;

// Matches kXlaAllocatedBufferAlignBytes so reservation slices and host-backed
// allocations satisfy CheckAlignment for every allocation kind.
constexpr uint64_t kGranularity = 256;

uint64_t RoundUpTestSize(uint64_t size) {
  return ((size + kGranularity - 1) / kGranularity) * kGranularity;
}

// Host-backed storage aligned to kGranularity.
class AlignedStorage {
 public:
  explicit AlignedStorage(uint64_t size)
      : storage_(std::make_unique<uint8_t[]>(size + kGranularity)) {}

  uint8_t* data() const {
    return reinterpret_cast<uint8_t*>(
        (reinterpret_cast<uintptr_t>(storage_.get()) + kGranularity - 1) &
        ~uintptr_t{kGranularity - 1});
  }

 private:
  std::unique_ptr<uint8_t[]> storage_;
};

class TestMemoryAllocation final : public se::MemoryAllocation {
 public:
  explicit TestMemoryAllocation(uint64_t size) : storage_(size), size_(size) {}

  se::DeviceAddressBase address() const override {
    return se::DeviceAddressBase(storage_.data(), size_);
  }

 private:
  AlignedStorage storage_;
  uint64_t size_;
};

class TestMemoryReservation final : public se::MemoryReservation {
 public:
  explicit TestMemoryReservation(uint64_t size) : storage_(size), size_(size) {}

  se::DeviceAddressBase address() const override {
    return se::DeviceAddressBase(storage_.data(), size_);
  }

  int active_mapping_count() const { return active_mapping_count_; }

 private:
  absl::Status Map(size_t reservation_offset, size_t allocation_offset,
                   size_t size, se::MemoryAllocation& allocation) override {
    if (reservation_offset > size_ || size > size_ - reservation_offset ||
        allocation_offset > allocation.address().size() ||
        size > allocation.address().size() - allocation_offset) {
      return absl::InvalidArgumentError("mapping range is out of bounds");
    }
    ++active_mapping_count_;
    return absl::OkStatus();
  }

  absl::Status SetAccess(uint64_t /*reservation_offset*/,
                         size_t /*size*/) override {
    return absl::OkStatus();
  }

  absl::Status UnMap(size_t /*reservation_offset*/, size_t /*size*/) override {
    if (active_mapping_count_ == 0) {
      return absl::FailedPreconditionError("reservation is not mapped");
    }
    --active_mapping_count_;
    return absl::OkStatus();
  }

  AlignedStorage storage_;
  uint64_t size_;
  int active_mapping_count_ = 0;
};

// Host-only VMM allocator: the 4 platform hooks are backed by heap memory, so
// the full reservation/mapping state machine runs without a GPU.
class TestVmmAllocator final : public se::DeviceAddressVmmAllocator {
 public:
  static absl::StatusOr<std::unique_ptr<TestVmmAllocator>> Create(
      const se::Platform* platform, absl::Span<const DeviceConfig> devices) {
    auto allocator =
        std::unique_ptr<TestVmmAllocator>(new TestVmmAllocator(platform));
    absl::Status status = PopulateDevices(allocator.get(), devices);
    if (!status.ok()) {
      return status;
    }
    return allocator;
  }

  TestMemoryReservation* last_reservation() const { return last_reservation_; }

 protected:
  absl::Status InitializeDeviceState(PerDeviceState& state) override {
    state.allocation_granularity = kGranularity;
    auto* timeline = new uint64_t(0);
    state.pinned_timeline = timeline;
    state.destroy_fn = [timeline] { delete timeline; };
    return absl::OkStatus();
  }

  absl::StatusOr<std::unique_ptr<se::MemoryAllocation>> CreateAllocation(
      se::StreamExecutor* /*executor*/, uint64_t size) override {
    return std::make_unique<TestMemoryAllocation>(RoundUpTestSize(size));
  }

  absl::StatusOr<std::unique_ptr<se::MemoryReservation>> CreateReservation(
      se::StreamExecutor* /*executor*/, uint64_t size) override {
    auto reservation =
        std::make_unique<TestMemoryReservation>(RoundUpTestSize(size));
    last_reservation_ = reservation.get();
    return reservation;
  }

  absl::Status EnqueueDeferredDeallocation(PerDeviceState& state,
                                           uint64_t seqno) override {
    __atomic_store_n(state.pinned_timeline, seqno, __ATOMIC_RELEASE);
    return absl::OkStatus();
  }

 private:
  explicit TestVmmAllocator(const se::Platform* platform)
      : DeviceAddressVmmAllocator(platform) {}

  TestMemoryReservation* last_reservation_ = nullptr;
};

class GpuExecutableVaRemapAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ON_CALL(executor_, device_ordinal()).WillByDefault(Return(0));
    ON_CALL(stream_, parent()).WillByDefault(Return(&executor_));
    run_options_.set_stream(&stream_);
    service_run_options_ = ServiceExecutableRunOptions(run_options_);
  }

  absl::StatusOr<std::unique_ptr<TestVmmAllocator>> CreateAllocator() {
    return TestVmmAllocator::Create(&platform_, {{&executor_, &stream_}});
  }

  NiceMock<se::MockPlatform> platform_;
  NiceMock<se::MockStreamExecutor> executor_;
  NiceMock<se::MockStream> stream_;
  ExecutableRunOptions run_options_;
  ServiceExecutableRunOptions service_run_options_;
};

TEST_F(GpuExecutableVaRemapAllocatorTest,
       RemapsParameterBufferToStableReservationAddress) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestVmmAllocator> vmm_allocator,
                       CreateAllocator());

  constexpr int64_t kBufferSize = 1024;
  BufferAllocation param_alloc(/*index=*/0, kBufferSize, /*color=*/0);
  param_alloc.set_entry_computation_parameter(
      /*parameter_number=*/0, /*param_shape_index=*/{},
      /*parameter_aliased_with_output=*/false);
  std::vector<const BufferAllocation*> allocations = {&param_alloc};

  ThunkExecutor thunk_executor{ThunkSequence{}};
  DebugOptions debug_options;
  debug_options.set_xla_gpu_command_buffer_update_mode(DebugOptions::SKIP_TEMP);
  GpuExecutableVaRemapAllocator allocator("test", allocations,
                                          ShapeUtil::MakeShape(F32, {256}),
                                          &debug_options, &thunk_executor);
  allocator.AddVaRemappedAllocationForTesting(0);
  EXPECT_EQ(allocator.command_buffer_allocation_count(), 1);

  // Caller-owned parameter buffer. It must be an exact allocator address so
  // it can be aliased into the reservation with Map().
  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> param_buffer,
      vmm_allocator->Allocate(/*device_ordinal=*/0, kBufferSize,
                              /*retry_on_failure=*/true, /*memory_space=*/0));

  auto get_parameter_buffer = [&](const BufferAllocation& allocation)
      -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
    return GpuExecutableBufferAllocator::ParameterBuffer{
        param_buffer.cref(), allocation.parameter_number()};
  };
  GpuExecutableBufferAllocator::BufferAllocToDeviceMemoryMap globals;

  const void* reservation_address = nullptr;
  for (int run = 0; run < 2; ++run) {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope> scope,
        allocator.CreateExecutionScope(&service_run_options_,
                                       vmm_allocator.get(),
                                       /*device_ordinal=*/0));
    EXPECT_TRUE(scope->va_remap_enabled());

    ASSERT_OK_AND_ASSIGN(BufferAllocations buffer_allocations,
                         scope->GenerateBufferAllocations(
                             &service_run_options_, get_parameter_buffer,
                             &globals, vmm_allocator.get(),
                             /*device_ordinal=*/0));

    // Execution sees the reservation address, not the caller-owned buffer,
    // and the address is stable across executions.
    se::DeviceAddressBase mapped = buffer_allocations.GetDeviceAddress(0);
    EXPECT_NE(mapped.opaque(), param_buffer.cref().opaque());
    if (run == 0) {
      reservation_address = mapped.opaque();
      ASSERT_NE(vmm_allocator->last_reservation(), nullptr);
      EXPECT_EQ(mapped.opaque(),
                vmm_allocator->last_reservation()->address().opaque());
    }
    EXPECT_EQ(mapped.opaque(), reservation_address);

    bool executed = false;
    ASSERT_OK(scope->ExecuteWithBufferAllocations(
        buffer_allocations, /*device_ordinal=*/0,
        [&](const BufferAllocations& execution_buffers,
            std::optional<absl::Span<const BufferAllocation::Index>>
                persistent_alloc_indices) {
          executed = true;
          EXPECT_EQ(execution_buffers.GetDeviceAddress(0).opaque(),
                    reservation_address);
          EXPECT_TRUE(persistent_alloc_indices.has_value());
          EXPECT_THAT(*persistent_alloc_indices, ElementsAre(0));
          return absl::OkStatus();
        }));
    EXPECT_TRUE(executed);

    // After the execution, the entry is rewritten to the caller-owned address
    // so TearDown and result handling see a deallocatable address.
    EXPECT_EQ(buffer_allocations.GetDeviceAddress(0).opaque(),
              param_buffer.cref().opaque());
  }

  // All reservation-address aliases are released once deferred operations
  // complete.
  ASSERT_OK(vmm_allocator->SynchronizePendingOperations(/*device_ordinal=*/0));
  EXPECT_EQ(vmm_allocator->last_reservation()->active_mapping_count(), 0);
}

TEST_F(GpuExecutableVaRemapAllocatorTest,
       NullParameterBufferFailsWhenSelectedForRemapping) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestVmmAllocator> vmm_allocator,
                       CreateAllocator());

  BufferAllocation param_alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  param_alloc.set_entry_computation_parameter(
      /*parameter_number=*/0, /*param_shape_index=*/{},
      /*parameter_aliased_with_output=*/false);
  std::vector<const BufferAllocation*> allocations = {&param_alloc};

  ThunkExecutor thunk_executor{ThunkSequence{}};
  DebugOptions debug_options;
  debug_options.set_xla_gpu_command_buffer_update_mode(DebugOptions::SKIP_TEMP);
  GpuExecutableVaRemapAllocator allocator("test", allocations,
                                          ShapeUtil::MakeShape(F32, {256}),
                                          &debug_options, &thunk_executor);
  allocator.AddVaRemappedAllocationForTesting(0);

  auto get_parameter_buffer = [&](const BufferAllocation& allocation)
      -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
    return GpuExecutableBufferAllocator::ParameterBuffer{
        se::DeviceAddressBase(), allocation.parameter_number(),
        /*allow_null_buffer=*/true};
  };
  GpuExecutableBufferAllocator::BufferAllocToDeviceMemoryMap globals;

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope> scope,
      allocator.CreateExecutionScope(&service_run_options_, vmm_allocator.get(),
                                     /*device_ordinal=*/0));
  EXPECT_FALSE(scope
                   ->GenerateBufferAllocations(
                       &service_run_options_, get_parameter_buffer, &globals,
                       vmm_allocator.get(), /*device_ordinal=*/0)
                   .ok());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
