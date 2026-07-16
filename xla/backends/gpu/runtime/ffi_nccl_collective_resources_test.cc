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

#include "xla/backends/gpu/runtime/ffi_nccl_collective_resources.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives_stub.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_nccl_collective_resources.h"
#include "xla/ffi/nccl_collective_resources_api.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/mock_platform.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

struct alignas(64) Storage {
  std::array<std::byte, 256> bytes = {};

  se::DeviceAddressBase address() {
    return se::DeviceAddressBase(bytes.data(), bytes.size());
  }
};

struct alignas(64) AddressTableStorage {
  std::array<uint64_t, 8> entries = {};

  se::DeviceAddressBase address() {
    return se::DeviceAddressBase(entries.data(), sizeof(entries));
  }
};

XLA_FFI_NcclCollectiveDeviceAddressTable MakeAddressTable(
    uint64_t* device_data, size_t address_capacity) {
  return {
      XLA_FFI_NcclCollectiveDeviceAddressTable_STRUCT_SIZE,
      /*extension_start=*/nullptr,
      device_data,
      address_capacity,
      /*address_count=*/0,
  };
}

class TestMemoryAllocation final : public se::MemoryAllocation {
 public:
  TestMemoryAllocation(
      uint64_t size, int* active_allocations,
      std::shared_ptr<absl::Notification> deallocation_notification)
      : storage_(std::make_unique<uint8_t[]>(size)),
        size_(size),
        active_allocations_(active_allocations),
        deallocation_notification_(std::move(deallocation_notification)) {
    ++*active_allocations_;
  }

  ~TestMemoryAllocation() override {
    --*active_allocations_;
    deallocation_notification_->Notify();
  }

  se::DeviceAddressBase address() const override {
    return se::DeviceAddressBase(storage_.get(), size_);
  }

 private:
  std::unique_ptr<uint8_t[]> storage_;
  uint64_t size_;
  int* active_allocations_;
  std::shared_ptr<absl::Notification> deallocation_notification_;
};

uint64_t AddressValue(void* address, uint64_t offset = 0) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address)) + offset;
}

class FakeSymmetricMemory final : public SymmetricMemory {
 public:
  FakeSymmetricMemory(
      se::DeviceAddressBase local_address,
      std::vector<se::DeviceAddressBase> peer_addresses,
      std::optional<se::DeviceAddressBase> multimem_address = std::nullopt)
      : local_address_(local_address),
        peer_addresses_(std::move(peer_addresses)),
        multimem_address_(multimem_address) {}

  se::DeviceAddressBase addr() const override { return local_address_; }

  absl::StatusOr<se::DeviceAddressBase> multimem_addr() const override {
    if (!multimem_address_.has_value()) {
      return absl::UnimplementedError("multimem is unavailable");
    }
    return *multimem_address_;
  }

  absl::StatusOr<se::DeviceAddressBase> peer_addr(RankId rank) const override {
    if (rank.value() < 0 ||
        static_cast<size_t>(rank.value()) >= peer_addresses_.size()) {
      return absl::InvalidArgumentError("peer rank is out of range");
    }
    return peer_addresses_[rank.value()];
  }

  std::string ToString() const override { return "FakeSymmetricMemory"; }

  PackedKernelArg PackKernelArg() const override {
    return local_address_.opaque();
  }

 private:
  se::DeviceAddressBase local_address_;
  std::vector<se::DeviceAddressBase> peer_addresses_;
  std::optional<se::DeviceAddressBase> multimem_address_;
};

struct GroupSpec {
  XLA_FFI_NcclCollectiveGroupMode mode =
      XLA_FFI_NCCL_COLLECTIVE_GROUP_MODE_CROSS_REPLICA;
  uint64_t communication_id = 17;
  std::vector<size_t> offsets = {0, 2};
  std::vector<int64_t> members = {0, 1};
};

struct RegionSpec {
  se::DeviceAddressBase buffer;
  size_t byte_offset;
  size_t byte_size;
  size_t required_alignment;
  XLA_FFI_NcclCollectiveMemoryKind memory_kind;
};

class CollectiveResourcesInvocation {
 public:
  CollectiveResourcesInvocation(
      int64_t replica_count, int64_t partition_count, int64_t current_device,
      std::vector<se::DeviceAddressBase> allocations = {}) {
    ON_CALL(stream_, parent()).WillByDefault(Return(&executor_));
    ON_CALL(executor_, GetPlatform()).WillByDefault(Return(&platform_));
    ON_CALL(platform_, Name()).WillByDefault(ReturnRef(platform_name_));
    ON_CALL(executor_, HostMemoryAllocate(testing::_))
        .WillByDefault(
            [this](uint64_t size)
                -> absl::StatusOr<std::unique_ptr<se::MemoryAllocation>> {
              last_host_deallocation_ = std::make_shared<absl::Notification>();
              std::unique_ptr<se::MemoryAllocation> allocation =
                  std::make_unique<TestMemoryAllocation>(
                      size, &active_host_allocations_, last_host_deallocation_);
              return allocation;
            });
    ON_CALL(stream_, Memcpy(testing::A<se::DeviceAddressBase*>(),
                            testing::A<const void*>(), testing::_))
        .WillByDefault([](se::DeviceAddressBase* destination,
                          const void* source, uint64_t size) {
          std::memcpy(destination->opaque(), source, size);
          return absl::OkStatus();
        });
    ON_CALL(stream_, DoHostCallbackWithStatus(testing::_))
        .WillByDefault([](absl::AnyInvocable<absl::Status()&&> callback) {
          return std::move(callback)();
        });

    device_assignment_.emplace(replica_count, partition_count);
    int64_t device = 0;
    for (int64_t replica = 0; replica < replica_count; ++replica) {
      for (int64_t partition = 0; partition < partition_count; ++partition) {
        (*device_assignment_)(replica, partition) = device++;
      }
    }

    gpu_options_.set_collectives(&collectives_);
    run_options_.mutable_run_options()->set_stream(&stream_);
    run_options_.mutable_run_options()->set_device_assignment(
        &*device_assignment_);
    run_options_.mutable_run_options()->set_local_device_count(replica_count *
                                                               partition_count);
    run_options_.mutable_run_options()->set_gpu_executable_run_options(
        &gpu_options_);

    absl::StatusOr<CollectiveParams> collective_params =
        CollectiveParams::Create(run_options_, /*async_streams=*/{},
                                 LocalDeviceId(current_device));
    if (!collective_params.ok()) {
      status_ = collective_params.status();
      return;
    }
    collective_params_.emplace(std::move(*collective_params));

    buffer_allocations_ = std::make_unique<BufferAllocations>(
        allocations, /*device_ordinal=*/static_cast<int>(current_device),
        /*memory_allocator=*/nullptr);
    memory_requests_ =
        std::make_unique<CollectiveMemoryRequests>(*buffer_allocations_);
    SetSymmetricMemories({});
  }

  const absl::Status& status() const { return status_; }

  absl::Status Begin(XLA_FFI_ExecutionStage stage,
                     bool include_collective_cliques = true,
                     bool include_collective_memory = true) {
    if (!status_.ok()) return status_;

    switch (stage) {
      case XLA_FFI_ExecutionStage_PREPARE:
        return resources_.BeginInvocation(
            stage, /*stream=*/nullptr, buffer_allocations_.get(),
            &*collective_params_, &clique_requests_, memory_requests_.get(),
            /*collective_cliques=*/nullptr, /*collective_memory=*/nullptr);
      case XLA_FFI_ExecutionStage_INITIALIZE:
      case XLA_FFI_ExecutionStage_EXECUTE:
        return resources_.BeginInvocation(
            stage, &stream_, buffer_allocations_.get(), &*collective_params_,
            /*collective_clique_requests=*/nullptr,
            /*collective_memory_requests=*/nullptr,
            include_collective_cliques ? &collective_cliques_ : nullptr,
            include_collective_memory ? collective_memory_.get() : nullptr);
      default:
        return resources_.BeginInvocation(
            stage, /*stream=*/nullptr, /*buffer_allocations=*/nullptr,
            /*collective_params=*/nullptr,
            /*collective_clique_requests=*/nullptr,
            /*collective_memory_requests=*/nullptr,
            /*collective_cliques=*/nullptr, /*collective_memory=*/nullptr);
    }
  }

  absl::Status Request(const GroupSpec& spec,
                       const std::vector<RegionSpec>& regions = {},
                       bool barrier_before_launch = false) {
    std::vector<XLA_FFI_NcclCollectiveRegion> region_descriptors;
    region_descriptors.reserve(regions.size());
    for (const RegionSpec& region : regions) {
      region_descriptors.push_back(XLA_FFI_NcclCollectiveRegion{
          XLA_FFI_NcclCollectiveRegion_STRUCT_SIZE,
          /*extension_start=*/nullptr, region.buffer.opaque(),
          region.buffer.size(), region.byte_offset, region.byte_size,
          region.required_alignment, region.memory_kind});
    }

    XLA_FFI_NcclCollectiveGroup group = {
        XLA_FFI_NcclCollectiveGroup_STRUCT_SIZE,
        /*extension_start=*/nullptr,
        spec.mode,
        spec.communication_id,
        spec.offsets.size() - 1,
        spec.offsets.data(),
        spec.members.size(),
        spec.members.data(),
    };
    XLA_FFI_NcclCollectiveResources_Request_Args args = {
        XLA_FFI_NcclCollectiveResources_Request_Args_STRUCT_SIZE,
        /*extension_start=*/nullptr,
        context(),
        &group,
        region_descriptors.data(),
        region_descriptors.size(),
        static_cast<uint8_t>(barrier_before_launch),
        /*resource=*/nullptr,
        /*rank=*/-1,
        /*clique_size=*/-1,
    };
    absl::Status status = resources_.Request(&args);
    if (status.ok()) {
      resource_.reset(
          reinterpret_cast<ffi::NcclCollectiveResourceHandle*>(args.resource));
      rank_ = args.rank;
      clique_size_ = args.clique_size;
    }
    return status;
  }

  absl::Status Commit(XLA_FFI_NcclCollectiveResource* resource = nullptr) {
    XLA_FFI_NcclCollectiveResources_Commit_Args args = {
        XLA_FFI_NcclCollectiveResources_Commit_Args_STRUCT_SIZE,
        /*extension_start=*/nullptr, context(),
        resource == nullptr ? this->resource() : resource};
    return resources_.Commit(&args);
  }

  absl::Status Initialize() {
    XLA_FFI_NcclCollectiveResources_Initialize_Args args = {
        XLA_FFI_NcclCollectiveResources_Initialize_Args_STRUCT_SIZE,
        /*extension_start=*/nullptr, context(), resource()};
    return resources_.Initialize(&args);
  }

  absl::Status Resolve(XLA_FFI_NcclCollectiveDeviceAddressTable* table) {
    XLA_FFI_NcclCollectiveResources_Resolve_Args args = {
        XLA_FFI_NcclCollectiveResources_Resolve_Args_STRUCT_SIZE,
        /*extension_start=*/nullptr, context(), resource(), table};
    return resources_.Resolve(&args);
  }

  absl::StatusOr<XLA_FFI_NcclCollectiveDeviceAddressTable> Resolve(
      uint64_t* device_data, size_t address_capacity) {
    XLA_FFI_NcclCollectiveDeviceAddressTable table =
        MakeAddressTable(device_data, address_capacity);
    absl::Status status = Resolve(&table);
    if (!status.ok()) return status;
    return table;
  }

  absl::Status EnqueueBarrierBeforeLaunch() {
    XLA_FFI_NcclCollectiveResources_EnqueueBarrierBeforeLaunch_Args args = {
        XLA_FFI_NcclCollectiveResources_EnqueueBarrierBeforeLaunch_Args_STRUCT_SIZE,
        /*extension_start=*/nullptr, context(), resource()};
    return resources_.EnqueueBarrierBeforeLaunch(&args);
  }

  void SetSymmetricMemories(
      absl::flat_hash_map<CollectiveMemory::Key,
                          std::shared_ptr<SymmetricMemory>>
          memories) {
    collective_memory_ = std::make_unique<CollectiveMemory>(
        *buffer_allocations_, std::move(memories),
        /*mcast_memories=*/
        absl::flat_hash_map<CollectiveMemory::Key,
                            CollectiveMemory::MulticastMemory>{},
        /*peer_memories=*/
        absl::flat_hash_map<CollectiveMemory::Key,
                            CollectiveMemory::PeerMemory>{});
  }

  XLA_FFI_NcclCollectiveResource* resource() const {
    return reinterpret_cast<XLA_FFI_NcclCollectiveResource*>(resource_.get());
  }

  int32_t rank() const { return rank_; }
  int32_t clique_size() const { return clique_size_; }
  int active_host_allocations() const { return active_host_allocations_; }
  bool WaitForHostDeallocation() {
    return last_host_deallocation_ != nullptr &&
           last_host_deallocation_->WaitForNotificationWithTimeout(
               absl::Seconds(5));
  }
  se::MockStream& stream() { return stream_; }
  void DestroyResource() { resource_.reset(); }
  const CollectiveCliqueRequests& clique_requests() const {
    return clique_requests_;
  }
  const CollectiveMemoryRequests& memory_requests() const {
    return *memory_requests_;
  }

 private:
  XLA_FFI_ExecutionContext* context() {
    return reinterpret_cast<XLA_FFI_ExecutionContext*>(this);
  }

  int active_host_allocations_ = 0;
  std::shared_ptr<absl::Notification> last_host_deallocation_;
  std::string platform_name_ = "CUDA";
  NiceMock<se::MockPlatform> platform_;
  NiceMock<se::MockStreamExecutor> executor_;
  NiceMock<se::MockStream> stream_;
  GpuCollectivesStub collectives_;
  std::optional<DeviceAssignment> device_assignment_;
  GpuExecutableRunOptions gpu_options_;
  ServiceExecutableRunOptions run_options_;
  std::optional<CollectiveParams> collective_params_;
  std::unique_ptr<BufferAllocations> buffer_allocations_;
  CollectiveCliqueRequests clique_requests_;
  std::unique_ptr<CollectiveMemoryRequests> memory_requests_;
  CollectiveCliques collective_cliques_;
  std::unique_ptr<CollectiveMemory> collective_memory_;
  FfiNcclCollectiveResources resources_;
  std::unique_ptr<ffi::NcclCollectiveResourceHandle> resource_;
  int32_t rank_ = -1;
  int32_t clique_size_ = -1;
  absl::Status status_;
};

TEST(FfiNcclCollectiveResourcesTest, ResolvesEverySupportedGroupMode) {
  struct TestCase {
    const char* name;
    GroupSpec group;
    std::vector<GlobalDeviceId> expected_clique;
  };
  std::array<TestCase, 4> test_cases = {{
      {"cross replica",
       GroupSpec{XLA_FFI_NCCL_COLLECTIVE_GROUP_MODE_CROSS_REPLICA,
                 /*communication_id=*/17,
                 /*offsets=*/{0, 2}, /*members=*/{0, 1}},
       {GlobalDeviceId(0), GlobalDeviceId(2)}},
      {"cross partition",
       GroupSpec{XLA_FFI_NCCL_COLLECTIVE_GROUP_MODE_CROSS_PARTITION,
                 /*communication_id=*/17,
                 /*offsets=*/{0, 2}, /*members=*/{0, 1}},
       {GlobalDeviceId(0), GlobalDeviceId(1)}},
      {"cross replica and partition",
       GroupSpec{XLA_FFI_NCCL_COLLECTIVE_GROUP_MODE_CROSS_REPLICA_AND_PARTITION,
                 /*communication_id=*/17,
                 /*offsets=*/{0, 2}, /*members=*/{0, 1}},
       {GlobalDeviceId(0), GlobalDeviceId(1), GlobalDeviceId(2),
        GlobalDeviceId(3)}},
      {"flattened id",
       GroupSpec{XLA_FFI_NCCL_COLLECTIVE_GROUP_MODE_FLATTENED_ID,
                 /*communication_id=*/17,
                 /*offsets=*/{0, 2, 4}, /*members=*/{0, 3, 1, 2}},
       {GlobalDeviceId(0), GlobalDeviceId(3)}},
  }};

  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    CollectiveResourcesInvocation invocation(
        /*replica_count=*/2, /*partition_count=*/2, /*current_device=*/0);
    ASSERT_THAT(invocation.status(), IsOk());
    ASSERT_THAT(invocation.Begin(XLA_FFI_ExecutionStage_PREPARE), IsOk());
    ASSERT_THAT(invocation.Request(test_case.group), IsOk());
    EXPECT_EQ(invocation.clique_size(), test_case.expected_clique.size());
    ASSERT_THAT(invocation.Commit(), IsOk());

    std::vector<CollectiveCliqueRequests::CliqueRequest> requests =
        invocation.clique_requests().OrderedRequestedCliques();
    ASSERT_EQ(requests.size(), 1);
    EXPECT_THAT(requests[0].key.devices(),
                ElementsAreArray(test_case.expected_clique));
    EXPECT_EQ(requests[0].key.communication_id(), CommunicationId(17));
  }
}

TEST(FfiNcclCollectiveResourcesTest,
     RejectsInvalidGroupBeforePublishingRequests) {
  GroupSpec group;
  group.offsets = {0, 1};
  CollectiveResourcesInvocation invocation(
      /*replica_count=*/2, /*partition_count=*/2, /*current_device=*/0);

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Begin(XLA_FFI_ExecutionStage_PREPARE), IsOk());
  EXPECT_THAT(invocation.Request(group),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("offsets must span the complete members")));
  EXPECT_EQ(invocation.clique_requests().size(), 0);
  EXPECT_EQ(invocation.memory_requests().symmetric_size(), 0);
}

TEST(FfiNcclCollectiveResourcesTest,
     RequestIsTransactionalAndCommitDeduplicatesAllocations) {
  Storage allocation0;
  Storage allocation1;
  CollectiveResourcesInvocation invocation(
      /*replica_count=*/2, /*partition_count=*/1, /*current_device=*/0,
      {allocation0.address(), allocation1.address()});
  std::vector<RegionSpec> regions = {
      {allocation0.address().GetByteSlice(16, 64), /*byte_offset=*/0,
       /*byte_size=*/16, /*required_alignment=*/16,
       XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_SYMMETRIC},
      {allocation0.address().GetByteSlice(96, 64), /*byte_offset=*/16,
       /*byte_size=*/16, /*required_alignment=*/16,
       XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_SYMMETRIC},
      {allocation1.address().GetByteSlice(32, 64), /*byte_offset=*/0,
       /*byte_size=*/16, /*required_alignment=*/16,
       XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_MULTIMEM},
  };

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Begin(XLA_FFI_ExecutionStage_PREPARE), IsOk());
  ASSERT_THAT(invocation.Request(GroupSpec{}, regions), IsOk());
  EXPECT_EQ(invocation.rank(), 0);
  EXPECT_EQ(invocation.clique_size(), 2);
  EXPECT_EQ(invocation.clique_requests().size(), 0);
  EXPECT_EQ(invocation.memory_requests().symmetric_size(), 0);

  ASSERT_THAT(invocation.Commit(), IsOk());
  EXPECT_EQ(invocation.clique_requests().size(), 1);
  EXPECT_EQ(invocation.memory_requests().symmetric_size(), 1);
  std::vector<CollectiveMemoryRequests::CollectiveAllocations> requests =
      invocation.memory_requests().OrderedSymmetricAllocations();
  ASSERT_EQ(requests.size(), 1);
  EXPECT_THAT(requests[0].allocations, ElementsAre(0, 1));
  EXPECT_THAT(invocation.Commit(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("can only be committed once")));
}

TEST(FfiNcclCollectiveResourcesTest, RejectsOutOfRangeAndMisalignedRegions) {
  Storage allocation;
  {
    CollectiveResourcesInvocation invocation(
        /*replica_count=*/2, /*partition_count=*/1, /*current_device=*/0,
        {allocation.address()});
    RegionSpec region = {allocation.address().GetByteSlice(0, 64),
                         /*byte_offset=*/48,
                         /*byte_size=*/32, /*required_alignment=*/16,
                         XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_SYMMETRIC};

    ASSERT_THAT(invocation.status(), IsOk());
    ASSERT_THAT(invocation.Begin(XLA_FFI_ExecutionStage_PREPARE), IsOk());
    EXPECT_THAT(invocation.Request(GroupSpec{}, {region}),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("exceeds containing size 64")));
  }
  {
    CollectiveResourcesInvocation invocation(
        /*replica_count=*/2, /*partition_count=*/1, /*current_device=*/0,
        {allocation.address()});
    RegionSpec region = {se::DeviceAddressBase(allocation.bytes.data() + 1, 64),
                         /*byte_offset=*/0, /*byte_size=*/16,
                         /*required_alignment=*/16,
                         XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_SYMMETRIC};

    ASSERT_THAT(invocation.status(), IsOk());
    ASSERT_THAT(invocation.Begin(XLA_FFI_ExecutionStage_PREPARE), IsOk());
    EXPECT_THAT(invocation.Request(GroupSpec{}, {region}),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("does not meet alignment 16")));
  }
}

TEST(FfiNcclCollectiveResourcesTest,
     ResolvesIntoCallerStorageAndRetainsHostStagingUntilCopyCompletes) {
  constexpr uint64_t kUnwritten = 0xdeadbeefdeadbeef;
  Storage local0;
  Storage remote0;
  Storage local1;
  Storage remote1;
  Storage multimem1;
  AddressTableStorage table_storage;
  table_storage.entries.fill(kUnwritten);
  CollectiveResourcesInvocation invocation(
      /*replica_count=*/2, /*partition_count=*/1, /*current_device=*/1,
      {local0.address(), local1.address(), table_storage.address()});
  std::vector<RegionSpec> regions = {
      {local0.address().GetByteSlice(32, 96), /*byte_offset=*/16,
       /*byte_size=*/32, /*required_alignment=*/16,
       XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_SYMMETRIC},
      {local1.address().GetByteSlice(64, 128), /*byte_offset=*/32,
       /*byte_size=*/16, /*required_alignment=*/32,
       XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_MULTIMEM},
  };

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Begin(XLA_FFI_ExecutionStage_PREPARE), IsOk());
  ASSERT_THAT(invocation.Request(GroupSpec{}, regions), IsOk());
  ASSERT_THAT(invocation.Commit(), IsOk());
  std::vector<CollectiveCliqueRequests::CliqueRequest> requests =
      invocation.clique_requests().OrderedRequestedCliques();
  ASSERT_EQ(requests.size(), 1);
  const GpuCliqueKey& clique_key = requests[0].key;

  absl::flat_hash_map<CollectiveMemory::Key, std::shared_ptr<SymmetricMemory>>
      memories;
  memories.emplace(
      std::make_pair(clique_key, 0),
      std::make_shared<FakeSymmetricMemory>(
          local0.address(), std::vector<se::DeviceAddressBase>{
                                remote0.address(), local0.address()}));
  memories.emplace(std::make_pair(clique_key, 1),
                   std::make_shared<FakeSymmetricMemory>(
                       local1.address(),
                       std::vector<se::DeviceAddressBase>{remote1.address(),
                                                          local1.address()},
                       multimem1.address()));
  invocation.SetSymmetricMemories(std::move(memories));

  ASSERT_THAT(invocation.Begin(XLA_FFI_ExecutionStage_INITIALIZE), IsOk());
  ASSERT_THAT(invocation.Initialize(), IsOk());
  std::optional<absl::AnyInvocable<absl::Status() &&>> copy_completion;
  EXPECT_CALL(invocation.stream(),
              Memcpy(testing::A<se::DeviceAddressBase*>(),
                     testing::A<const void*>(), 4 * sizeof(uint64_t)))
      .WillOnce([&](se::DeviceAddressBase* destination, const void* source,
                    uint64_t size) {
        EXPECT_EQ(destination->opaque(), table_storage.entries.data());
        EXPECT_EQ(destination->size(), 4 * sizeof(uint64_t));
        std::memcpy(destination->opaque(), source, size);
        return absl::OkStatus();
      });
  EXPECT_CALL(invocation.stream(), DoHostCallbackWithStatus(testing::_))
      .WillOnce([&](absl::AnyInvocable<absl::Status()&&> callback) {
        copy_completion.emplace(std::move(callback));
        return absl::OkStatus();
      });
  ASSERT_OK_AND_ASSIGN(XLA_FFI_NcclCollectiveDeviceAddressTable table,
                       invocation.Resolve(table_storage.entries.data(),
                                          /*address_capacity=*/6));
  EXPECT_EQ(table.device_data, table_storage.entries.data());
  EXPECT_EQ(table.address_capacity, 6);
  EXPECT_EQ(table.address_count, 4);
  EXPECT_THAT(table_storage.entries,
              ElementsAre(AddressValue(remote0.bytes.data(), 48),
                          AddressValue(local0.bytes.data(), 48),
                          AddressValue(multimem1.bytes.data(), 96),
                          AddressValue(multimem1.bytes.data(), 96), kUnwritten,
                          kUnwritten, kUnwritten, kUnwritten));
  EXPECT_EQ(invocation.active_host_allocations(), 1);
  EXPECT_THAT(invocation.Resolve(table_storage.entries.data(),
                                 /*address_capacity=*/6),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("already resolved")));
  invocation.DestroyResource();
  EXPECT_EQ(invocation.active_host_allocations(), 1);
  ASSERT_TRUE(copy_completion.has_value());
  {
    absl::AnyInvocable<absl::Status() &&> callback =
        std::move(*copy_completion);
    copy_completion.reset();
    EXPECT_THAT(std::move(callback)(), IsOk());
  }
  ASSERT_TRUE(invocation.WaitForHostDeallocation());
  EXPECT_EQ(invocation.active_host_allocations(), 0);
}

TEST(FfiNcclCollectiveResourcesTest,
     RejectsInvalidCallerStorageWithoutPublishingResolution) {
  constexpr uint64_t kUnwritten = 0xdeadbeefdeadbeef;
  Storage local;
  Storage remote;
  AddressTableStorage table_storage;
  AddressTableStorage unowned_storage;
  table_storage.entries.fill(kUnwritten);
  unowned_storage.entries.fill(kUnwritten);
  CollectiveResourcesInvocation invocation(
      /*replica_count=*/2, /*partition_count=*/1, /*current_device=*/0,
      {local.address(), table_storage.address()});
  RegionSpec region = {local.address(), /*byte_offset=*/0, /*byte_size=*/16,
                       /*required_alignment=*/16,
                       XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_SYMMETRIC};

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Begin(XLA_FFI_ExecutionStage_PREPARE), IsOk());
  ASSERT_THAT(invocation.Request(GroupSpec{}, {region}), IsOk());
  ASSERT_THAT(invocation.Commit(), IsOk());
  std::vector<CollectiveCliqueRequests::CliqueRequest> requests =
      invocation.clique_requests().OrderedRequestedCliques();
  ASSERT_EQ(requests.size(), 1);

  absl::flat_hash_map<CollectiveMemory::Key, std::shared_ptr<SymmetricMemory>>
      memories;
  memories.emplace(
      std::make_pair(requests[0].key, 0),
      std::make_shared<FakeSymmetricMemory>(
          local.address(), std::vector<se::DeviceAddressBase>{
                               local.address(), remote.address()}));
  invocation.SetSymmetricMemories(std::move(memories));

  ASSERT_THAT(invocation.Begin(XLA_FFI_ExecutionStage_INITIALIZE), IsOk());
  ASSERT_THAT(invocation.Initialize(), IsOk());
  EXPECT_CALL(invocation.stream(),
              Memcpy(testing::A<se::DeviceAddressBase*>(),
                     testing::A<const void*>(), 2 * sizeof(uint64_t)))
      .Times(1);
  EXPECT_CALL(invocation.stream(), DoHostCallbackWithStatus(testing::_))
      .Times(1);

  XLA_FFI_NcclCollectiveDeviceAddressTable undersized =
      MakeAddressTable(table_storage.entries.data(), /*address_capacity=*/1);
  EXPECT_THAT(
      invocation.Resolve(&undersized),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("capacity 1")));
  EXPECT_EQ(undersized.address_count, 0);
  EXPECT_EQ(invocation.active_host_allocations(), 0);

  XLA_FFI_NcclCollectiveDeviceAddressTable null_storage =
      MakeAddressTable(/*device_data=*/nullptr, /*address_capacity=*/2);
  EXPECT_THAT(invocation.Resolve(&null_storage),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must not be null")));
  EXPECT_EQ(null_storage.address_count, 0);
  EXPECT_EQ(invocation.active_host_allocations(), 0);

  XLA_FFI_NcclCollectiveDeviceAddressTable unowned =
      MakeAddressTable(unowned_storage.entries.data(), /*address_capacity=*/2);
  EXPECT_THAT(
      invocation.Resolve(&unowned),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("does not belong to an XLA buffer allocation")));
  EXPECT_EQ(unowned.address_count, 0);
  EXPECT_EQ(unowned_storage.entries[0], kUnwritten);
  EXPECT_EQ(invocation.active_host_allocations(), 0);

  XLA_FFI_NcclCollectiveDeviceAddressTable overlapping = MakeAddressTable(
      reinterpret_cast<uint64_t*>(local.bytes.data()), /*address_capacity=*/2);
  EXPECT_THAT(invocation.Resolve(&overlapping),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("overlaps collective region 0")));
  EXPECT_EQ(overlapping.address_count, 0);
  EXPECT_EQ(invocation.active_host_allocations(), 0);

  XLA_FFI_NcclCollectiveDeviceAddressTable nonzero_count =
      MakeAddressTable(table_storage.entries.data(), /*address_capacity=*/2);
  nonzero_count.address_count = 1;
  EXPECT_THAT(invocation.Resolve(&nonzero_count),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be initialized to zero")));
  EXPECT_EQ(invocation.active_host_allocations(), 0);

  auto* misaligned_data = reinterpret_cast<uint64_t*>(
      reinterpret_cast<std::byte*>(table_storage.entries.data()) + 1);
  XLA_FFI_NcclCollectiveDeviceAddressTable misaligned =
      MakeAddressTable(misaligned_data, /*address_capacity=*/2);
  EXPECT_THAT(invocation.Resolve(&misaligned),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("not uint64 aligned")));
  EXPECT_EQ(invocation.active_host_allocations(), 0);

  XLA_FFI_NcclCollectiveDeviceAddressTable out_of_bounds =
      MakeAddressTable(table_storage.entries.data(), /*address_capacity=*/9);
  EXPECT_THAT(invocation.Resolve(&out_of_bounds),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds containing size")));
  EXPECT_EQ(invocation.active_host_allocations(), 0);

  ASSERT_OK_AND_ASSIGN(XLA_FFI_NcclCollectiveDeviceAddressTable resolved,
                       invocation.Resolve(table_storage.entries.data(),
                                          /*address_capacity=*/2));
  EXPECT_EQ(resolved.address_count, 2);
  EXPECT_THAT(
      table_storage.entries,
      ElementsAre(AddressValue(local.bytes.data()),
                  AddressValue(remote.bytes.data()), kUnwritten, kUnwritten,
                  kUnwritten, kUnwritten, kUnwritten, kUnwritten));
  ASSERT_TRUE(invocation.WaitForHostDeallocation());
  invocation.DestroyResource();
  EXPECT_EQ(invocation.active_host_allocations(), 0);
}

TEST(FfiNcclCollectiveResourcesTest,
     CopyAndRetentionFailuresLeaveResourceRetryable) {
  constexpr uint64_t kUnwritten = 0xdeadbeefdeadbeef;
  Storage local;
  Storage remote;
  AddressTableStorage table_storage;
  table_storage.entries.fill(kUnwritten);
  CollectiveResourcesInvocation invocation(
      /*replica_count=*/2, /*partition_count=*/1, /*current_device=*/0,
      {local.address(), table_storage.address()});
  RegionSpec region = {local.address(), /*byte_offset=*/0, /*byte_size=*/16,
                       /*required_alignment=*/16,
                       XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_SYMMETRIC};

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Begin(XLA_FFI_ExecutionStage_PREPARE), IsOk());
  ASSERT_THAT(invocation.Request(GroupSpec{}, {region}), IsOk());
  ASSERT_THAT(invocation.Commit(), IsOk());
  std::vector<CollectiveCliqueRequests::CliqueRequest> requests =
      invocation.clique_requests().OrderedRequestedCliques();
  ASSERT_EQ(requests.size(), 1);

  absl::flat_hash_map<CollectiveMemory::Key, std::shared_ptr<SymmetricMemory>>
      memories;
  memories.emplace(
      std::make_pair(requests[0].key, 0),
      std::make_shared<FakeSymmetricMemory>(
          local.address(), std::vector<se::DeviceAddressBase>{
                               local.address(), remote.address()}));
  invocation.SetSymmetricMemories(std::move(memories));

  ASSERT_THAT(invocation.Begin(XLA_FFI_ExecutionStage_INITIALIZE), IsOk());
  ASSERT_THAT(invocation.Initialize(), IsOk());
  EXPECT_CALL(invocation.stream(),
              Memcpy(testing::A<se::DeviceAddressBase*>(),
                     testing::A<const void*>(), 2 * sizeof(uint64_t)))
      .WillOnce(Return(absl::InternalError("copy failed")))
      .WillOnce([](se::DeviceAddressBase* destination, const void* source,
                   uint64_t size) {
        std::memcpy(destination->opaque(), source, size);
        return absl::OkStatus();
      })
      .WillOnce([](se::DeviceAddressBase* destination, const void* source,
                   uint64_t size) {
        std::memcpy(destination->opaque(), source, size);
        return absl::OkStatus();
      });
  EXPECT_CALL(invocation.stream(), DoHostCallbackWithStatus(testing::_))
      .WillOnce(Return(absl::InternalError("callback failed")))
      .WillOnce([](absl::AnyInvocable<absl::Status()&&> callback) {
        return std::move(callback)();
      });
  EXPECT_CALL(invocation.stream(), BlockHostUntilDone())
      .WillOnce(Return(absl::OkStatus()));
  XLA_FFI_NcclCollectiveDeviceAddressTable table =
      MakeAddressTable(table_storage.entries.data(), /*address_capacity=*/2);
  EXPECT_THAT(invocation.Resolve(&table),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("copy failed")));
  EXPECT_EQ(table.address_count, 0);
  EXPECT_EQ(table_storage.entries[0], kUnwritten);
  EXPECT_EQ(invocation.active_host_allocations(), 0);

  EXPECT_THAT(
      invocation.Resolve(&table),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("callback failed")));
  EXPECT_EQ(table.address_count, 0);
  EXPECT_EQ(invocation.active_host_allocations(), 0);

  ASSERT_THAT(invocation.Resolve(&table), IsOk());
  EXPECT_EQ(table.address_count, 2);
  EXPECT_THAT(
      table_storage.entries,
      ElementsAre(AddressValue(local.bytes.data()),
                  AddressValue(remote.bytes.data()), kUnwritten, kUnwritten,
                  kUnwritten, kUnwritten, kUnwritten, kUnwritten));
  ASSERT_TRUE(invocation.WaitForHostDeallocation());
  EXPECT_EQ(invocation.active_host_allocations(), 0);
  invocation.DestroyResource();
  EXPECT_EQ(invocation.active_host_allocations(), 0);
}

TEST(FfiNcclCollectiveResourcesTest,
     RejectsMultimemWithoutCliqueWidePeerAccess) {
  Storage local;
  Storage multimem;
  AddressTableStorage table_storage;
  CollectiveResourcesInvocation invocation(
      /*replica_count=*/2, /*partition_count=*/1, /*current_device=*/0,
      {local.address(), table_storage.address()});
  RegionSpec region = {local.address(), /*byte_offset=*/0, /*byte_size=*/16,
                       /*required_alignment=*/16,
                       XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_MULTIMEM};

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Begin(XLA_FFI_ExecutionStage_PREPARE), IsOk());
  ASSERT_THAT(invocation.Request(GroupSpec{}, {region}), IsOk());
  ASSERT_THAT(invocation.Commit(), IsOk());
  std::vector<CollectiveCliqueRequests::CliqueRequest> requests =
      invocation.clique_requests().OrderedRequestedCliques();
  ASSERT_EQ(requests.size(), 1);

  absl::flat_hash_map<CollectiveMemory::Key, std::shared_ptr<SymmetricMemory>>
      memories;
  memories.emplace(std::make_pair(requests[0].key, 0),
                   std::make_shared<FakeSymmetricMemory>(
                       local.address(),
                       std::vector<se::DeviceAddressBase>{
                           local.address(), se::DeviceAddressBase()},
                       multimem.address()));
  invocation.SetSymmetricMemories(std::move(memories));

  ASSERT_THAT(invocation.Begin(XLA_FFI_ExecutionStage_INITIALIZE), IsOk());
  ASSERT_THAT(invocation.Initialize(), IsOk());
  EXPECT_THAT(invocation.Resolve(table_storage.entries.data(),
                                 /*address_capacity=*/2),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("rank 1 has no peer address")));
}

TEST(FfiNcclCollectiveResourcesTest,
     ResolvesEmptyAddressTableWithoutAllocation) {
  CollectiveResourcesInvocation invocation(
      /*replica_count=*/2, /*partition_count=*/1, /*current_device=*/0);

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Begin(XLA_FFI_ExecutionStage_PREPARE), IsOk());
  ASSERT_THAT(invocation.Request(GroupSpec{}), IsOk());
  ASSERT_THAT(invocation.Commit(), IsOk());
  ASSERT_THAT(invocation.Begin(XLA_FFI_ExecutionStage_INITIALIZE), IsOk());
  ASSERT_THAT(invocation.Initialize(), IsOk());
  EXPECT_CALL(invocation.stream(),
              Memcpy(testing::A<se::DeviceAddressBase*>(),
                     testing::A<const void*>(), testing::_))
      .Times(0);
  EXPECT_CALL(invocation.stream(), DoHostCallbackWithStatus(testing::_))
      .Times(0);
  ASSERT_OK_AND_ASSIGN(XLA_FFI_NcclCollectiveDeviceAddressTable table,
                       invocation.Resolve(/*device_data=*/nullptr,
                                          /*address_capacity=*/0));
  EXPECT_EQ(table.device_data, nullptr);
  EXPECT_EQ(table.address_capacity, 0);
  EXPECT_EQ(table.address_count, 0);
  EXPECT_EQ(invocation.active_host_allocations(), 0);
}

TEST(FfiNcclCollectiveResourcesTest, PreservesEntrySynchronizationOptIn) {
  // Creating and launching the barrier requires live GPU communicators. This
  // host test verifies that the opt-in reaches XLA's acquired-context check.
  CollectiveResourcesInvocation barrier(
      /*replica_count=*/2, /*partition_count=*/1, /*current_device=*/0);
  ASSERT_THAT(barrier.status(), IsOk());
  ASSERT_THAT(barrier.Begin(XLA_FFI_ExecutionStage_PREPARE), IsOk());
  ASSERT_THAT(barrier.Request(GroupSpec{}, /*regions=*/{},
                              /*barrier_before_launch=*/true),
              IsOk());
  ASSERT_THAT(barrier.Commit(), IsOk());
  ASSERT_THAT(barrier.Begin(XLA_FFI_ExecutionStage_INITIALIZE,
                            /*include_collective_cliques=*/false),
              IsOk());
  EXPECT_THAT(barrier.Initialize(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("requires acquired collective contexts")));

  CollectiveResourcesInvocation no_barrier(
      /*replica_count=*/2, /*partition_count=*/1, /*current_device=*/0);
  ASSERT_THAT(no_barrier.status(), IsOk());
  ASSERT_THAT(no_barrier.Begin(XLA_FFI_ExecutionStage_PREPARE), IsOk());
  ASSERT_THAT(no_barrier.Request(GroupSpec{}), IsOk());
  ASSERT_THAT(no_barrier.Commit(), IsOk());
  ASSERT_THAT(no_barrier.Begin(XLA_FFI_ExecutionStage_INITIALIZE), IsOk());
  ASSERT_THAT(no_barrier.Initialize(), IsOk());
  ASSERT_THAT(no_barrier.Begin(XLA_FFI_ExecutionStage_EXECUTE), IsOk());
  EXPECT_THAT(no_barrier.EnqueueBarrierBeforeLaunch(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("did not request entry synchronization")));
}

TEST(FfiNcclCollectiveResourcesTest, RejectsWrongStagesAndForeignResources) {
  CollectiveResourcesInvocation owner(
      /*replica_count=*/2, /*partition_count=*/1, /*current_device=*/0);
  CollectiveResourcesInvocation foreign(
      /*replica_count=*/2, /*partition_count=*/1, /*current_device=*/0);
  ASSERT_THAT(owner.status(), IsOk());
  ASSERT_THAT(foreign.status(), IsOk());

  EXPECT_THAT(owner.Begin(XLA_FFI_ExecutionStage_INSTANTIATE),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported collective resource execution "
                                 "stage")));
  EXPECT_THAT(owner.Request(GroupSpec{}),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("only be requested during FFI Prepare")));

  ASSERT_THAT(owner.Begin(XLA_FFI_ExecutionStage_PREPARE), IsOk());
  ASSERT_THAT(owner.Request(GroupSpec{}), IsOk());
  ASSERT_THAT(foreign.Begin(XLA_FFI_ExecutionStage_PREPARE), IsOk());
  EXPECT_THAT(foreign.Commit(owner.resource()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("belongs to a different execution")));

  ASSERT_THAT(owner.Begin(XLA_FFI_ExecutionStage_INITIALIZE), IsOk());
  EXPECT_THAT(owner.Commit(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("only be committed during FFI Prepare")));
  EXPECT_THAT(owner.Initialize(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("must be committed exactly once")));
}

}  // namespace
}  // namespace xla::gpu
