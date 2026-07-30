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

#include "xla/ffi/api/nccl_collective_resources.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_nccl_collective_resources.h"

namespace xla::ffi {
namespace {

struct TestState;

struct TestResource {
  TestState* state;
};

struct TestState {
  XLA_FFI_ExecutionContext* context = nullptr;
  TestResource resource = {};
  int request_count = 0;
  int commit_count = 0;
  int initialize_count = 0;
  int query_topology_count = 0;
  int resolve_count = 0;
  int resolve_host_count = 0;
  int enqueue_count = 0;
  int destroy_count = 0;
  uint64_t* expected_device_data = nullptr;
  size_t expected_address_capacity = 0;
  uint64_t* returned_device_data = nullptr;
  size_t returned_address_capacity = 0;
  size_t returned_address_count = 0;
  size_t expected_region_count = 1;
  bool expected_barrier_before_launch = true;
};

class DeviceStorage {
 public:
  DeviceStorage(uint64_t* data, int64_t capacity)
      : capacity_(capacity),
        buffer_{XLA_FFI_Buffer_STRUCT_SIZE,
                /*extension_start=*/nullptr,
                XLA_FFI_DataType_U64,
                data,
                /*rank=*/1,
                &capacity_} {}

  BufferR1<DataType::U64> buffer() const {
    return BufferR1<DataType::U64>(&buffer_);
  }

 private:
  int64_t capacity_;
  XLA_FFI_Buffer buffer_;
};

TestState* GetState(XLA_FFI_ExecutionContext* context) {
  return reinterpret_cast<TestState*>(context);
}

TestState* GetState(XLA_FFI_NcclCollectiveResource* resource) {
  return reinterpret_cast<TestResource*>(resource)->state;
}

XLA_FFI_Error* Request(XLA_FFI_NcclCollectiveResources_Request_Args* args) {
  TestState* state = GetState(args->ctx);
  ++state->request_count;
  EXPECT_EQ(args->ctx, state->context);
  EXPECT_NE(args->group, nullptr);
  if (args->group == nullptr) return nullptr;
  EXPECT_EQ(args->group->group_mode,
            XLA_FFI_NCCL_COLLECTIVE_GROUP_MODE_CROSS_REPLICA);
  EXPECT_EQ(args->group->communication_id, 7);
  EXPECT_EQ(args->group->num_groups, 1);
  EXPECT_EQ(args->group->num_members, 2);
  EXPECT_EQ(args->group->group_offsets[0], 0);
  EXPECT_EQ(args->group->group_offsets[1], 2);
  EXPECT_EQ(args->group->members[0], 0);
  EXPECT_EQ(args->group->members[1], 1);
  EXPECT_EQ(args->region_count, state->expected_region_count);
  if (args->region_count == 1) {
    EXPECT_EQ(args->regions[0].byte_offset, 16);
    EXPECT_EQ(args->regions[0].byte_size, 32);
    EXPECT_EQ(args->regions[0].required_alignment, 16);
    EXPECT_EQ(args->regions[0].memory_kind,
              XLA_FFI_NCCL_COLLECTIVE_MEMORY_KIND_SYMMETRIC);
  }
  EXPECT_EQ(args->barrier_before_launch,
            state->expected_barrier_before_launch ? 1 : 0);
  args->resource =
      reinterpret_cast<XLA_FFI_NcclCollectiveResource*>(&state->resource);
  args->rank = 1;
  args->clique_size = 2;
  return nullptr;
}

XLA_FFI_Error* Commit(XLA_FFI_NcclCollectiveResources_Commit_Args* args) {
  TestState* state = GetState(args->resource);
  ++state->commit_count;
  EXPECT_EQ(args->ctx, state->context);
  return nullptr;
}

XLA_FFI_Error* Initialize(
    XLA_FFI_NcclCollectiveResources_Initialize_Args* args) {
  TestState* state = GetState(args->resource);
  ++state->initialize_count;
  EXPECT_EQ(args->ctx, state->context);
  return nullptr;
}

XLA_FFI_Error* Resolve(XLA_FFI_NcclCollectiveResources_Resolve_Args* args) {
  TestState* state = GetState(args->resource);
  ++state->resolve_count;
  EXPECT_EQ(args->ctx, state->context);
  EXPECT_NE(args->table, nullptr);
  if (args->table == nullptr) return nullptr;
  EXPECT_EQ(args->table->device_data, state->expected_device_data);
  EXPECT_EQ(args->table->address_capacity, state->expected_address_capacity);
  EXPECT_EQ(args->table->address_count, 0);
  args->table->device_data = state->returned_device_data;
  args->table->address_capacity = state->returned_address_capacity;
  args->table->address_count = state->returned_address_count;
  return nullptr;
}

XLA_FFI_Error* ResolveHost(
    XLA_FFI_NcclCollectiveResources_ResolveHost_Args* args) {
  TestState* state = GetState(args->resource);
  ++state->resolve_host_count;
  EXPECT_EQ(args->ctx, state->context);
  EXPECT_EQ(args->address_count, state->expected_region_count * 2);
  for (size_t i = 0; i < args->address_count; ++i) {
    args->addresses[i] = 0x1000 + i;
  }
  return nullptr;
}

XLA_FFI_Error* QueryTopology(
    XLA_FFI_NcclCollectiveResources_QueryTopology_Args* args) {
  TestState* state = GetState(args->resource);
  ++state->query_topology_count;
  EXPECT_EQ(args->ctx, state->context);
  args->topology->clique_size = 2;
  args->topology->lsa_size = 2;
  args->topology->lsa_team_count = 1;
  args->topology->world_is_lsa = 1;
  args->topology->multimem_supported = 1;
  return nullptr;
}

XLA_FFI_Error* EnqueueBarrierBeforeLaunch(
    XLA_FFI_NcclCollectiveResources_EnqueueBarrierBeforeLaunch_Args* args) {
  TestState* state = GetState(args->resource);
  ++state->enqueue_count;
  EXPECT_EQ(args->ctx, state->context);
  return nullptr;
}

void Destroy(XLA_FFI_NcclCollectiveResources_Destroy_Args* args) {
  ++GetState(args->resource)->destroy_count;
}

XLA_FFI_NcclCollectiveResources_Extension TestExtension() {
  return {
      {XLA_FFI_NcclCollectiveResources_Extension_STRUCT_SIZE,
       XLA_FFI_Extension_NcclCollectiveResources, nullptr},
      XLA_FFI_NCCL_COLLECTIVE_RESOURCES_ABI_MAJOR,
      XLA_FFI_NCCL_COLLECTIVE_RESOURCES_ABI_MINOR,
      Request,
      Commit,
      Initialize,
      Resolve,
      EnqueueBarrierBeforeLaunch,
      Destroy,
      QueryTopology,
      ResolveHost,
  };
}

ErrorOr<std::unique_ptr<NcclCollectiveResource>> RequestTestResource(
    const NcclCollectiveResources& resources, bool include_region = true) {
  std::vector<size_t> group_offsets = {0, 2};
  std::vector<int64_t> group_members = {0, 1};
  NcclCollectiveGroup group = {
      NcclCollectiveGroupMode::kCrossReplica,
      /*communication_id=*/7,
      group_offsets,
      group_members,
  };
  alignas(16) std::byte allocation[64];
  std::vector<NcclCollectiveRegion> regions;
  if (include_region) {
    regions.push_back({
        allocation,
        sizeof(allocation),
        /*byte_offset=*/16,
        /*byte_size=*/32,
        /*required_alignment=*/16,
        NcclCollectiveMemoryKind::kSymmetric,
    });
  }
  return resources.Request(group, regions,
                           /*barrier_before_launch=*/include_region);
}

TEST(NcclCollectiveResourcesTest, DispatchesResourceLifecycle) {
  TestState state;
  state.context = reinterpret_cast<XLA_FFI_ExecutionContext*>(&state);
  state.resource.state = &state;
  uint64_t device_data[2] = {};
  state.expected_device_data = device_data;
  state.expected_address_capacity = 2;
  state.returned_device_data = device_data;
  state.returned_address_capacity = 2;
  state.returned_address_count = 2;

  XLA_FFI_NcclCollectiveResources_Extension extension = TestExtension();
  XLA_FFI_Api api = {};
  api.struct_size = XLA_FFI_Api_STRUCT_SIZE;
  api.extension_start = &extension.extension_base;
  NcclCollectiveResources resources(&api, state.context);

  ASSERT_TRUE(resources.available());
  ErrorOr<std::unique_ptr<NcclCollectiveResource>> requested =
      RequestTestResource(resources);
  ASSERT_TRUE(requested.has_value()) << requested.error().message();
  std::unique_ptr<NcclCollectiveResource> resource = std::move(*requested);
  EXPECT_EQ(resource->info().rank, 1);
  EXPECT_EQ(resource->info().clique_size, 2);
  EXPECT_TRUE(resources.Commit(*resource).success());
  EXPECT_TRUE(resources.Initialize(*resource).success());

  ErrorOr<NcclCollectiveTopology> topology = resources.QueryTopology(*resource);
  ASSERT_TRUE(topology.has_value()) << topology.error().message();
  EXPECT_EQ(topology->clique_size, 2);
  EXPECT_EQ(topology->lsa_size, 2);
  EXPECT_EQ(topology->lsa_team_count, 1);
  EXPECT_TRUE(topology->world_is_lsa);
  EXPECT_TRUE(topology->multimem_supported);

  DeviceStorage storage(device_data, 2);
  ErrorOr<NcclCollectiveDeviceAddressTable> table =
      resources.ResolveDeviceAddresses(*resource, storage.buffer());
  ASSERT_TRUE(table.has_value()) << table.error().message();
  EXPECT_EQ(table->device_data, device_data);
  EXPECT_EQ(table->address_count, 2);
  EXPECT_TRUE(resources.EnqueueBarrierBeforeLaunch(*resource).success());
  resource.reset();

  EXPECT_EQ(state.request_count, 1);
  EXPECT_EQ(state.commit_count, 1);
  EXPECT_EQ(state.initialize_count, 1);
  EXPECT_EQ(state.query_topology_count, 1);
  EXPECT_EQ(state.resolve_count, 1);
  EXPECT_EQ(state.enqueue_count, 1);
  EXPECT_EQ(state.destroy_count, 1);
}

TEST(NcclCollectiveResourcesTest, AcceptsOverprovisionedDeviceStorage) {
  TestState state;
  state.context = reinterpret_cast<XLA_FFI_ExecutionContext*>(&state);
  state.resource.state = &state;
  uint64_t device_data[4] = {};
  state.expected_device_data = device_data;
  state.expected_address_capacity = 4;
  state.returned_device_data = device_data;
  state.returned_address_capacity = 4;
  state.returned_address_count = 2;

  XLA_FFI_NcclCollectiveResources_Extension extension = TestExtension();
  XLA_FFI_Api api = {};
  api.struct_size = XLA_FFI_Api_STRUCT_SIZE;
  api.extension_start = &extension.extension_base;
  NcclCollectiveResources resources(&api, state.context);
  ErrorOr<std::unique_ptr<NcclCollectiveResource>> requested =
      RequestTestResource(resources);
  ASSERT_TRUE(requested.has_value()) << requested.error().message();

  DeviceStorage storage(device_data, 4);
  ErrorOr<NcclCollectiveDeviceAddressTable> table =
      resources.ResolveDeviceAddresses(**requested, storage.buffer());
  ASSERT_TRUE(table.has_value()) << table.error().message();
  EXPECT_EQ(table->device_data, device_data);
  EXPECT_EQ(table->address_count, 2);
  EXPECT_EQ(state.resolve_count, 1);
}

TEST(NcclCollectiveResourcesTest, ResolvesAddressesIntoHostStorage) {
  TestState state;
  state.context = reinterpret_cast<XLA_FFI_ExecutionContext*>(&state);
  state.resource.state = &state;
  XLA_FFI_NcclCollectiveResources_Extension extension = TestExtension();
  XLA_FFI_Api api = {};
  api.struct_size = XLA_FFI_Api_STRUCT_SIZE;
  api.extension_start = &extension.extension_base;
  NcclCollectiveResources resources(&api, state.context);
  ErrorOr<std::unique_ptr<NcclCollectiveResource>> requested =
      RequestTestResource(resources);
  ASSERT_TRUE(requested.has_value()) << requested.error().message();

  uint64_t undersized[1] = {};
  Error invalid =
      resources.ResolveAddresses(**requested, Span<uint64_t>(undersized, 1));
  EXPECT_TRUE(invalid.failure());
  EXPECT_EQ(invalid.errc(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(state.resolve_host_count, 0);

  uint64_t addresses[2] = {};
  Error error =
      resources.ResolveAddresses(**requested, Span<uint64_t>(addresses, 2));
  EXPECT_TRUE(error.success()) << error.message();
  EXPECT_EQ(addresses[0], 0x1000);
  EXPECT_EQ(addresses[1], 0x1001);
  EXPECT_EQ(state.resolve_host_count, 1);
}

TEST(NcclCollectiveResourcesTest, ResolvesEmptyHostAddressTable) {
  TestState state;
  state.context = reinterpret_cast<XLA_FFI_ExecutionContext*>(&state);
  state.resource.state = &state;
  state.expected_region_count = 0;
  state.expected_barrier_before_launch = false;
  XLA_FFI_NcclCollectiveResources_Extension extension = TestExtension();
  XLA_FFI_Api api = {};
  api.struct_size = XLA_FFI_Api_STRUCT_SIZE;
  api.extension_start = &extension.extension_base;
  NcclCollectiveResources resources(&api, state.context);
  ErrorOr<std::unique_ptr<NcclCollectiveResource>> requested =
      RequestTestResource(resources, /*include_region=*/false);
  ASSERT_TRUE(requested.has_value()) << requested.error().message();

  Error error = resources.ResolveAddresses(**requested, Span<uint64_t>());
  EXPECT_TRUE(error.success()) << error.message();
  EXPECT_EQ(state.resolve_host_count, 1);
}

TEST(NcclCollectiveResourcesTest, RejectsInvalidDeviceStorageBeforeDispatch) {
  struct TestCase {
    const char* name;
    uint64_t* device_data;
    int64_t address_capacity;
  };
  uint64_t undersized[1] = {};
  TestCase test_cases[] = {
      {"undersized", undersized, 1},
      {"null", nullptr, 2},
  };

  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    TestState state;
    state.context = reinterpret_cast<XLA_FFI_ExecutionContext*>(&state);
    state.resource.state = &state;

    XLA_FFI_NcclCollectiveResources_Extension extension = TestExtension();
    XLA_FFI_Api api = {};
    api.struct_size = XLA_FFI_Api_STRUCT_SIZE;
    api.extension_start = &extension.extension_base;
    NcclCollectiveResources resources(&api, state.context);
    ErrorOr<std::unique_ptr<NcclCollectiveResource>> requested =
        RequestTestResource(resources);
    ASSERT_TRUE(requested.has_value()) << requested.error().message();
    std::unique_ptr<NcclCollectiveResource> resource = std::move(*requested);

    DeviceStorage storage(test_case.device_data, test_case.address_capacity);
    ErrorOr<NcclCollectiveDeviceAddressTable> table =
        resources.ResolveDeviceAddresses(*resource, storage.buffer());
    ASSERT_FALSE(table.has_value());
    EXPECT_EQ(table.error().errc(), ErrorCode::kInvalidArgument);
    EXPECT_EQ(state.resolve_count, 0);
  }
}

TEST(NcclCollectiveResourcesTest, RejectsMalformedResolvedAddressTable) {
  struct TestCase {
    const char* name;
    uint64_t* returned_device_data;
    size_t returned_address_capacity;
    size_t returned_address_count;
  };
  uint64_t device_data[2] = {};
  uint64_t different_device_data[2] = {};
  TestCase test_cases[] = {
      {"changed pointer", different_device_data, 2, 2},
      {"changed capacity", device_data, 3, 2},
      {"wrong count", device_data, 2, 1},
  };

  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    TestState state;
    state.context = reinterpret_cast<XLA_FFI_ExecutionContext*>(&state);
    state.resource.state = &state;
    state.expected_device_data = device_data;
    state.expected_address_capacity = 2;
    state.returned_device_data = test_case.returned_device_data;
    state.returned_address_capacity = test_case.returned_address_capacity;
    state.returned_address_count = test_case.returned_address_count;

    XLA_FFI_NcclCollectiveResources_Extension extension = TestExtension();
    XLA_FFI_Api api = {};
    api.struct_size = XLA_FFI_Api_STRUCT_SIZE;
    api.extension_start = &extension.extension_base;
    NcclCollectiveResources resources(&api, state.context);
    ErrorOr<std::unique_ptr<NcclCollectiveResource>> requested =
        RequestTestResource(resources);
    ASSERT_TRUE(requested.has_value()) << requested.error().message();

    DeviceStorage storage(device_data, 2);
    ErrorOr<NcclCollectiveDeviceAddressTable> table =
        resources.ResolveDeviceAddresses(**requested, storage.buffer());
    ASSERT_FALSE(table.has_value());
    EXPECT_EQ(table.error().errc(), ErrorCode::kInternal);
    EXPECT_EQ(state.resolve_count, 1);
  }
}

TEST(NcclCollectiveResourcesTest, AcceptsNewerAbiMinorVersion) {
  TestState state;
  state.context = reinterpret_cast<XLA_FFI_ExecutionContext*>(&state);
  state.resource.state = &state;

  XLA_FFI_NcclCollectiveResources_Extension extension = TestExtension();
  ++extension.abi_minor_version;
  XLA_FFI_Api api = {};
  api.struct_size = XLA_FFI_Api_STRUCT_SIZE;
  api.extension_start = &extension.extension_base;
  NcclCollectiveResources resources(&api, state.context);

  ASSERT_TRUE(resources.available());
  ErrorOr<std::unique_ptr<NcclCollectiveResource>> requested =
      RequestTestResource(resources);
  ASSERT_TRUE(requested.has_value()) << requested.error().message();
  ErrorOr<NcclCollectiveTopology> topology =
      resources.QueryTopology(**requested);
  ASSERT_TRUE(topology.has_value()) << topology.error().message();
  EXPECT_EQ(state.query_topology_count, 1);
}

TEST(NcclCollectiveResourcesTest, RejectsOlderAbiMinorVersion) {
  XLA_FFI_NcclCollectiveResources_Extension extension = TestExtension();
  --extension.abi_minor_version;
  XLA_FFI_Api api = {};
  api.struct_size = XLA_FFI_Api_STRUCT_SIZE;
  api.extension_start = &extension.extension_base;

  NcclCollectiveResources resources(&api, /*ctx=*/nullptr);
  EXPECT_FALSE(resources.available());
}

TEST(NcclCollectiveResourcesTest, RejectsIncompleteAbiZeroOneTable) {
  XLA_FFI_NcclCollectiveResources_Extension extension = TestExtension();
  extension.extension_base.struct_size =
      offsetof(XLA_FFI_NcclCollectiveResources_Extension, resolve_host);
  XLA_FFI_Api api = {};
  api.struct_size = XLA_FFI_Api_STRUCT_SIZE;
  api.extension_start = &extension.extension_base;

  NcclCollectiveResources resources(&api, /*ctx=*/nullptr);
  EXPECT_FALSE(resources.available());
  EXPECT_LT(offsetof(XLA_FFI_NcclCollectiveResources_Extension, resolve_host),
            XLA_FFI_NCCL_COLLECTIVE_RESOURCES_ABI_0_1_STRUCT_SIZE);
}

TEST(NcclCollectiveResourcesTest, RejectsIncompatibleAbiMajorVersion) {
  XLA_FFI_NcclCollectiveResources_Extension extension = TestExtension();
  ++extension.abi_major_version;
  XLA_FFI_Api api = {};
  api.struct_size = XLA_FFI_Api_STRUCT_SIZE;
  api.extension_start = &extension.extension_base;

  NcclCollectiveResources resources(&api, /*ctx=*/nullptr);
  EXPECT_FALSE(resources.available());
}

TEST(NcclCollectiveResourcesTest, ReportsMissingExtension) {
  XLA_FFI_Api api = {};
  api.struct_size = XLA_FFI_Api_STRUCT_SIZE;
  NcclCollectiveResources resources(&api, /*ctx=*/nullptr);

  EXPECT_FALSE(resources.available());
  std::vector<size_t> group_offsets = {0, 1};
  std::vector<int64_t> group_members = {0};
  NcclCollectiveGroup group = {
      NcclCollectiveGroupMode::kCrossReplica,
      /*communication_id=*/0,
      group_offsets,
      group_members,
  };
  ErrorOr<std::unique_ptr<NcclCollectiveResource>> requested =
      resources.Request(group, /*regions=*/{}, false);
  ASSERT_TRUE(requested.has_error());
  EXPECT_EQ(requested.error().errc(), ErrorCode::kUnimplemented);
}

}  // namespace
}  // namespace xla::ffi
