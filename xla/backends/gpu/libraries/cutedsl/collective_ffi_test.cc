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

#include "xla/backends/gpu/libraries/cutedsl/collective_ffi.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives_stub.h"
#include "xla/backends/gpu/libraries/cutedsl/collective_config.h"
#include "xla/backends/gpu/libraries/cutedsl/runtime_api.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/ffi/invoke.h"
#include "xla/runtime/device_id.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/mock_platform.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/types.h"

namespace xla::gpu::cutedsl {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

constexpr GlobalDeviceId kDevice0(0);
constexpr GlobalDeviceId kDevice1(1);
constexpr char kCollectiveTarget[] = "__xla_gpu_cutedsl_collective_v3";

struct FakeRuntime {
  std::string module_bytes;
  std::vector<std::string> function_prefixes;
  int create_count = 0;
  int get_function_count = 0;
  int run_count = 0;
  int destroy_count = 0;
  int fail_get_function_at = -1;
};

FakeRuntime* fake_runtime = nullptr;

CuteDSLRT_Error_t ModuleCreate(CuteDSLRT_Module_t** module,
                               const unsigned char* bytes, size_t size,
                               const char**, size_t) {
  ++fake_runtime->create_count;
  fake_runtime->module_bytes.assign(reinterpret_cast<const char*>(bytes), size);
  *module = reinterpret_cast<CuteDSLRT_Module_t*>(fake_runtime);
  return kCuteDslRtSuccess;
}

CuteDSLRT_Error_t ModuleGetFunction(CuteDSLRT_Function_t** function,
                                    CuteDSLRT_Module_t* module,
                                    const char* prefix) {
  EXPECT_EQ(module, reinterpret_cast<CuteDSLRT_Module_t*>(fake_runtime));
  ++fake_runtime->get_function_count;
  fake_runtime->function_prefixes.emplace_back(prefix);
  if (fake_runtime->get_function_count == fake_runtime->fail_get_function_at) {
    return 1;
  }
  *function = reinterpret_cast<CuteDSLRT_Function_t*>(fake_runtime);
  return kCuteDslRtSuccess;
}

CuteDSLRT_Error_t FunctionRun(void*, void**, size_t) {
  ++fake_runtime->run_count;
  return kCuteDslRtSuccess;
}

CuteDSLRT_Error_t ModuleDestroy(CuteDSLRT_Module_t* module) {
  EXPECT_EQ(module, reinterpret_cast<CuteDSLRT_Module_t*>(fake_runtime));
  ++fake_runtime->destroy_count;
  return kCuteDslRtSuccess;
}

const char* GetErrorName(CuteDSLRT_Error_t) { return "FakeRuntimeError"; }
const char* GetErrorString(CuteDSLRT_Error_t) { return "fake failure"; }

const RuntimeFunctions kFakeFunctions = {
    /*module_create_from_bytes=*/ModuleCreate,
    /*module_get_function=*/ModuleGetFunction,
    /*function_run=*/FunctionRun,
    /*module_destroy=*/ModuleDestroy,
    /*get_error_name=*/GetErrorName,
    /*get_error_string=*/GetErrorString,
};

class ScopedFakeRuntime {
 public:
  explicit ScopedFakeRuntime(FakeRuntime* runtime) {
    EXPECT_EQ(fake_runtime, nullptr);
    fake_runtime = runtime;
    status_ = SetRuntimeFunctionsForTesting(&kFakeFunctions);
  }

  ~ScopedFakeRuntime() {
    ResetRuntimeFunctionsForTesting();
    fake_runtime = nullptr;
  }

  const absl::Status& status() const { return status_; }

 private:
  absl::Status status_;
};

std::string Sha256(absl::string_view bytes) {
  llvm::SHA256 hasher;
  hasher.update(llvm::StringRef(bytes.data(), bytes.size()));
  std::array<uint8_t, kCollectiveModuleDigestSizeV3> digest = hasher.final();
  return std::string(reinterpret_cast<const char*>(digest.data()),
                     digest.size());
}

struct TestAttributes {
  ffi::AttributesMap Build() const {
    std::string module_blob;
    std::string module_keys;
    std::vector<int64_t> module_offsets = {0};
    for (const std::string& module : modules) {
      module_blob.append(module);
      module_keys.append(Sha256(module));
      module_offsets.push_back(static_cast<int64_t>(module_blob.size()));
    }

    ffi::CallFrameBuilder::AttributesBuilder attributes;
    attributes.Insert("schema_version", kCollectiveCallSchemaVersionV3);
    attributes.Insert("group_mode", group_mode);
    attributes.Insert("communication_id", communication_id);
    attributes.Insert("replica_group_offsets", replica_group_offsets);
    attributes.Insert("replica_group_members", replica_group_members);
    attributes.Insert("module_blob", std::move(module_blob));
    attributes.Insert("module_offsets", std::move(module_offsets));
    attributes.Insert("module_keys", std::move(module_keys));
    attributes.Insert("module_index_by_rank", module_index_by_rank);
    attributes.Insert("peer_regions", peer_regions);
    attributes.Insert("steps", steps);
    return attributes.Build();
  }

  int64_t group_mode =
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
  int64_t communication_id = 17;
  std::vector<int64_t> replica_group_offsets = {0, 2};
  std::vector<int64_t> replica_group_members = {0, 1};
  std::vector<std::string> modules = {"collective ffi test module"};
  std::vector<int64_t> module_index_by_rank = {0, 0};
  std::vector<int64_t> peer_regions;
  std::vector<int64_t> steps = {
      static_cast<int64_t>(CollectiveStepKindV3::kLaunch), 0};
};

class FakeSymmetricMemory final : public SymmetricMemory {
 public:
  FakeSymmetricMemory(se::DeviceAddressBase local_address,
                      std::vector<se::DeviceAddressBase> peer_addresses)
      : local_address_(local_address),
        peer_addresses_(std::move(peer_addresses)) {}

  se::DeviceAddressBase addr() const override { return local_address_; }

  absl::StatusOr<se::DeviceAddressBase> peer_addr(RankId rank) const override {
    if (failing_rank_.has_value() && rank == *failing_rank_) {
      return absl::InternalError("injected peer-address failure");
    }
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

  void set_failing_rank(RankId rank) { failing_rank_ = rank; }

 private:
  se::DeviceAddressBase local_address_;
  std::vector<se::DeviceAddressBase> peer_addresses_;
  std::optional<RankId> failing_rank_;
};

struct alignas(64) Storage {
  std::array<std::byte, 256> bytes;

  se::DeviceAddressBase address() {
    return se::DeviceAddressBase(bytes.data(), bytes.size());
  }
};

PeerRegionV3 Region(int64_t offset, int64_t size, int64_t alignment = 16) {
  return PeerRegionV3{
      /*endpoint=*/PeerRegionEndpointV3::kArgument,
      /*buffer_index=*/0,
      /*byte_offset=*/offset,
      /*byte_size=*/size,
      /*required_alignment=*/alignment,
      /*memory_kind=*/PeerMemoryKindV3::kSymmetric,
  };
}

uint64_t AddressValue(void* address, uint64_t offset = 0) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address)) + offset;
}

class CollectiveFfiInvocation {
 public:
  CollectiveFfiInvocation(TestAttributes attributes, int64_t replica_count,
                          int64_t partition_count, int64_t current_device,
                          std::vector<se::DeviceAddressBase> allocations = {},
                          std::vector<se::DeviceAddressBase> arguments = {},
                          std::vector<se::DeviceAddressBase> results = {})
      : attributes_(std::move(attributes)),
        arguments_(std::move(arguments)),
        results_(std::move(results)),
        registration_(ffi::FindHandler(kCollectiveTarget, "CUDA")) {
    ON_CALL(stream_, parent()).WillByDefault(Return(&executor_));
    ON_CALL(executor_, GetPlatform()).WillByDefault(Return(&platform_));
    ON_CALL(platform_, Name()).WillByDefault(ReturnRef(platform_name_));

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

    context_.state_context = {&states_[0], &states_[1], &states_[2]};
    ffi::InvokeContext::GpuContext gpu_context;
    gpu_context.collective_params = &*collective_params_;
    gpu_context.collective_clique_requests = &clique_requests_;
    gpu_context.collective_memory_requests = memory_requests_.get();
    context_.backend_context = gpu_context;
  }

  const absl::Status& status() const { return status_; }

  absl::Status Instantiate() {
    if (!status_.ok()) return status_;
    if (!registration_.ok()) return registration_.status();
    ffi::CallFrame call_frame = BuildCallFrame();
    return ffi::Invoke(ffi::GetXlaFfiApi(), registration_->bundle.instantiate,
                       call_frame, context_,
                       XLA_FFI_ExecutionStage_INSTANTIATE);
  }

  absl::Status Prepare() {
    if (!status_.ok()) return status_;
    if (!registration_.ok()) return registration_.status();
    ffi::CallFrame call_frame = BuildCallFrame();
    return ffi::Invoke(ffi::GetXlaFfiApi(), registration_->bundle.prepare,
                       call_frame, context_, XLA_FFI_ExecutionStage_PREPARE);
  }

  const CollectiveCliqueRequests& clique_requests() const {
    return clique_requests_;
  }

  const CollectiveMemoryRequests& memory_requests() const {
    return *memory_requests_;
  }

 private:
  ffi::CallFrame BuildCallFrame() const {
    ffi::CallFrameBuilder builder(arguments_.size(), results_.size());
    for (const se::DeviceAddressBase& argument : arguments_) {
      std::array<int64_t, 1> dimensions = {
          static_cast<int64_t>(argument.size())};
      builder.AddBufferArg(argument, U8, dimensions);
    }
    for (const se::DeviceAddressBase& result : results_) {
      std::array<int64_t, 1> dimensions = {static_cast<int64_t>(result.size())};
      builder.AddBufferRet(result, U8, dimensions);
    }
    builder.AddAttributes(attributes_.Build());
    return builder.Build();
  }

  TestAttributes attributes_;
  std::vector<se::DeviceAddressBase> arguments_;
  std::vector<se::DeviceAddressBase> results_;
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
  std::array<ffi::ExecutionState, 3> states_;
  ffi::InvokeContext context_;
  absl::StatusOr<ffi::HandlerRegistration> registration_;
  absl::Status status_;
};

TEST(CollectiveFfiTest, RegistersOnlyCollectiveV3WithoutTraits) {
  absl::StatusOr<ffi::HandlerRegistration> registration =
      ffi::FindHandler(kCollectiveTarget, "CUDA");
  ASSERT_THAT(registration, IsOk());
  EXPECT_EQ(registration->metadata.traits, 0);
  EXPECT_NE(registration->bundle.instantiate, nullptr);
  EXPECT_NE(registration->bundle.prepare, nullptr);
  EXPECT_NE(registration->bundle.initialize, nullptr);
  EXPECT_NE(registration->bundle.execute, nullptr);

  for (absl::string_view target :
       {"__xla_gpu_cutedsl_collective_v1", "__xla_gpu_cutedsl_collective_v2"}) {
    EXPECT_THAT(ffi::FindHandler(target, "CUDA"),
                StatusIs(absl::StatusCode::kNotFound));
  }
}

TEST(CollectiveFfiPrepareTest, ResolvesEverySupportedCollectiveGroupMode) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);
  ASSERT_THAT(scoped_runtime.status(), IsOk());

  struct TestCase {
    const char* name;
    CollectiveOpGroupMode mode;
    std::vector<int64_t> group_offsets;
    std::vector<int64_t> group_members;
    std::vector<int64_t> module_index_by_rank;
    std::vector<GlobalDeviceId> expected_clique;
  };
  std::array<TestCase, 4> test_cases = {{
      {"cross replica",
       CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA,
       {0, 2},
       {0, 1},
       {0, 0},
       {kDevice0, GlobalDeviceId(2)}},
      {"cross partition",
       CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION,
       {0, 2},
       {0, 1},
       {0, 0},
       {kDevice0, kDevice1}},
      {"cross replica and partition",
       CollectiveOpGroupMode::
           COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION,
       {0, 2},
       {0, 1},
       {0, 0, 0, 0},
       {kDevice0, kDevice1, GlobalDeviceId(2), GlobalDeviceId(3)}},
      {"flattened id",
       CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID,
       {0, 2, 4},
       {0, 3, 1, 2},
       {0, 0},
       {kDevice0, GlobalDeviceId(3)}},
  }};

  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    TestAttributes attributes;
    attributes.modules = {std::string("module for ") + test_case.name};
    attributes.group_mode = test_case.mode;
    attributes.replica_group_offsets = test_case.group_offsets;
    attributes.replica_group_members = test_case.group_members;
    attributes.module_index_by_rank = test_case.module_index_by_rank;

    CollectiveFfiInvocation invocation(std::move(attributes),
                                       /*replica_count=*/2,
                                       /*partition_count=*/2,
                                       /*current_device=*/0);
    ASSERT_THAT(invocation.status(), IsOk());
    ASSERT_THAT(invocation.Instantiate(), IsOk());
    ASSERT_THAT(invocation.Prepare(), IsOk());

    std::vector<CollectiveCliqueRequests::CliqueRequest> requests =
        invocation.clique_requests().OrderedRequestedCliques();
    ASSERT_EQ(requests.size(), 1);
    EXPECT_THAT(requests[0].key.devices(),
                ElementsAreArray(test_case.expected_clique));
    EXPECT_EQ(requests[0].key.communication_id(), CommunicationId(17));
    EXPECT_TRUE(requests[0].dev_comms.empty());
    EXPECT_FALSE(requests[0].barrier_after_module_execution_requested);
  }
}

TEST(CollectiveFfiPrepareTest, RejectsReplicaGroupOutsideRuntimeDomain) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);
  ASSERT_THAT(scoped_runtime.status(), IsOk());

  TestAttributes attributes;
  attributes.replica_group_members = {0, 2};
  CollectiveFfiInvocation invocation(std::move(attributes),
                                     /*replica_count=*/2,
                                     /*partition_count=*/1,
                                     /*current_device=*/0);

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Instantiate(), IsOk());
  EXPECT_THAT(invocation.Prepare(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("outside logical-ID domain")));
  EXPECT_EQ(runtime.create_count, 0);
  EXPECT_EQ(invocation.clique_requests().size(), 0);
  EXPECT_EQ(invocation.memory_requests().symmetric_size(), 0);
}

TEST(CollectiveFfiPrepareTest, ValidatesRankMapAgainstExpandedRuntimeClique) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);
  ASSERT_THAT(scoped_runtime.status(), IsOk());

  TestAttributes attributes;
  attributes.group_mode = CollectiveOpGroupMode::
      COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION;
  // The raw replica group has two members, but the two partitions expand the
  // runtime clique to four ranks.
  attributes.module_index_by_rank = {0, 0};
  CollectiveFfiInvocation invocation(std::move(attributes),
                                     /*replica_count=*/2,
                                     /*partition_count=*/2,
                                     /*current_device=*/0);

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Instantiate(), IsOk());
  EXPECT_THAT(invocation.Prepare(), StatusIs(absl::StatusCode::kInvalidArgument,
                                             HasSubstr("expected 4, got 2")));
  EXPECT_EQ(runtime.create_count, 0);
  EXPECT_EQ(invocation.clique_requests().size(), 0);
}

TEST(CollectiveFfiPrepareTest, RejectsLogicalBufferRangeBeforeLoadingModule) {
  Storage allocation;
  TestAttributes attributes;
  attributes.peer_regions = {
      static_cast<int64_t>(PeerRegionEndpointV3::kArgument), 0, 48, 32, 16,
      static_cast<int64_t>(PeerMemoryKindV3::kSymmetric),
  };
  CollectiveFfiInvocation invocation(
      std::move(attributes), /*replica_count=*/2, /*partition_count=*/1,
      /*current_device=*/0, /*allocations=*/{allocation.address()},
      /*arguments=*/{allocation.address().GetByteSlice(0, 64)});

  ASSERT_THAT(invocation.status(), IsOk());
  EXPECT_THAT(invocation.Instantiate(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds containing buffer size 64")));
}

TEST(CollectiveFfiPrepareTest, RejectsMisalignedRuntimeBufferAddress) {
  Storage allocation;
  TestAttributes attributes;
  attributes.peer_regions = {
      static_cast<int64_t>(PeerRegionEndpointV3::kArgument), 0, 0, 16, 16,
      static_cast<int64_t>(PeerMemoryKindV3::kSymmetric),
  };
  CollectiveFfiInvocation invocation(
      std::move(attributes), /*replica_count=*/2, /*partition_count=*/1,
      /*current_device=*/0, /*allocations=*/{allocation.address()},
      /*arguments=*/{se::DeviceAddressBase(allocation.bytes.data() + 1, 64)});

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Instantiate(), IsOk());
  EXPECT_THAT(invocation.Prepare(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not meet required alignment")));
}

TEST(CollectiveFfiPrepareTest,
     DeduplicatesContainingAllocationsAcrossPeerRegions) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);
  ASSERT_THAT(scoped_runtime.status(), IsOk());

  Storage allocation0;
  Storage allocation1;
  TestAttributes attributes;
  attributes.peer_regions = {
      static_cast<int64_t>(PeerRegionEndpointV3::kArgument),
      0,
      0,
      16,
      16,
      static_cast<int64_t>(PeerMemoryKindV3::kSymmetric),
      static_cast<int64_t>(PeerRegionEndpointV3::kArgument),
      1,
      16,
      16,
      16,
      static_cast<int64_t>(PeerMemoryKindV3::kSymmetric),
      static_cast<int64_t>(PeerRegionEndpointV3::kResult),
      0,
      0,
      16,
      16,
      static_cast<int64_t>(PeerMemoryKindV3::kSymmetric),
  };
  CollectiveFfiInvocation invocation(
      std::move(attributes), /*replica_count=*/2, /*partition_count=*/1,
      /*current_device=*/0,
      /*allocations=*/{allocation0.address(), allocation1.address()},
      /*arguments=*/
      {
          allocation0.address().GetByteSlice(16, 64),
          allocation0.address().GetByteSlice(96, 64),
      },
      /*results=*/{allocation1.address().GetByteSlice(32, 64)});

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Instantiate(), IsOk());
  ASSERT_THAT(invocation.Prepare(), IsOk());

  EXPECT_EQ(invocation.memory_requests().symmetric_size(), 1);
  EXPECT_EQ(invocation.memory_requests().multicast_size(), 0);
  EXPECT_EQ(invocation.memory_requests().peer_size(), 0);
  std::vector<CollectiveMemoryRequests::CollectiveAllocations> requests =
      invocation.memory_requests().OrderedSymmetricAllocations();
  ASSERT_EQ(requests.size(), 1);
  EXPECT_THAT(requests[0].allocations, ElementsAre(0, 1));
}

TEST(CollectiveFfiPrepareTest,
     PreloadsUniqueFunctionsAndUsesDefaultCliqueRequirements) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);
  ASSERT_THAT(scoped_runtime.status(), IsOk());

  TestAttributes attributes;
  attributes.steps = {
      static_cast<int64_t>(CollectiveStepKindV3::kBarrier), 0,
      static_cast<int64_t>(CollectiveStepKindV3::kLaunch),  2,
      static_cast<int64_t>(CollectiveStepKindV3::kLaunch),  0,
      static_cast<int64_t>(CollectiveStepKindV3::kLaunch),  2,
  };
  attributes.modules = {"module zero", "module one"};
  attributes.module_index_by_rank = {0, 1};
  CollectiveFfiInvocation invocation(std::move(attributes),
                                     /*replica_count=*/2,
                                     /*partition_count=*/1,
                                     /*current_device=*/0);

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Instantiate(), IsOk());
  ASSERT_THAT(invocation.Prepare(), IsOk());

  EXPECT_EQ(runtime.create_count, 2);
  EXPECT_EQ(runtime.get_function_count, 4);
  EXPECT_EQ(runtime.run_count, 0);
  EXPECT_THAT(runtime.function_prefixes,
              ElementsAre("cutlass_call", "cutlass_call_2", "cutlass_call",
                          "cutlass_call_2"));

  std::vector<CollectiveCliqueRequests::CliqueRequest> requests =
      invocation.clique_requests().OrderedRequestedCliques();
  ASSERT_EQ(requests.size(), 1);
  EXPECT_TRUE(requests[0].dev_comms.empty());
  EXPECT_FALSE(requests[0].barrier_after_module_execution_requested);
}

TEST(CollectiveFfiPrepareTest, ValidatesEveryModuleBeforeRequestingClique) {
  FakeRuntime runtime;
  runtime.fail_get_function_at = 2;
  ScopedFakeRuntime scoped_runtime(&runtime);
  ASSERT_THAT(scoped_runtime.status(), IsOk());

  TestAttributes attributes;
  attributes.modules = {"valid module", "module missing function"};
  attributes.module_index_by_rank = {0, 1};
  CollectiveFfiInvocation invocation(std::move(attributes),
                                     /*replica_count=*/2,
                                     /*partition_count=*/1,
                                     /*current_device=*/0);

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Instantiate(), IsOk());
  EXPECT_THAT(invocation.Prepare(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Module 1 function ordinal 0")));
  EXPECT_EQ(runtime.create_count, 2);
  EXPECT_EQ(runtime.get_function_count, 2);
  EXPECT_TRUE(invocation.clique_requests().OrderedRequestedCliques().empty());
}

TEST(CollectiveFfiPeerAddressesTest,
     ResolvesIndependentRowsWithNonzeroBufferAndRegionOffsets) {
  Storage local0;
  Storage peer0;
  Storage local1;
  Storage peer1;
  std::array<se::DeviceAddressBase, 2> allocations = {local0.address(),
                                                      local1.address()};
  BufferAllocations buffer_allocations(allocations, /*device_ordinal=*/0,
                                       /*memory_allocator=*/nullptr);

  GpuCliqueKey clique_key({kDevice0, kDevice1},
                          /*num_local_participants=*/2, CommunicationId(7));
  auto symmetric0 = std::make_shared<FakeSymmetricMemory>(
      local0.address(),
      std::vector<se::DeviceAddressBase>{local0.address(), peer0.address()});
  auto symmetric1 = std::make_shared<FakeSymmetricMemory>(
      local1.address(),
      std::vector<se::DeviceAddressBase>{local1.address(), peer1.address()});

  absl::flat_hash_map<CollectiveMemory::Key, std::shared_ptr<SymmetricMemory>>
      symmetric_memories;
  symmetric_memories.emplace(std::make_pair(clique_key, 0), symmetric0);
  symmetric_memories.emplace(std::make_pair(clique_key, 1), symmetric1);
  CollectiveMemory collective_memory(
      buffer_allocations, std::move(symmetric_memories),
      /*mcast_memories=*/{}, /*peer_memories=*/{});

  std::array<PeerRegionV3, 2> regions = {
      Region(/*offset=*/16, /*size=*/32, /*alignment=*/16),
      Region(/*offset=*/32, /*size=*/16, /*alignment=*/32),
  };
  std::array<se::DeviceAddressBase, 2> buffers = {
      local0.address().GetByteSlice(/*offset_bytes=*/32, /*size_bytes=*/96),
      local1.address().GetByteSlice(/*offset_bytes=*/64, /*size_bytes=*/128),
  };

  absl::StatusOr<std::vector<uint64_t>> addresses =
      internal::ResolvePeerAddressesV3(clique_key, RankId(0), regions, buffers,
                                       collective_memory);

  ASSERT_THAT(addresses, IsOk());
  EXPECT_THAT(*addresses, ElementsAre(AddressValue(local0.bytes.data(), 48),
                                      AddressValue(peer0.bytes.data(), 48),
                                      AddressValue(local1.bytes.data(), 96),
                                      AddressValue(peer1.bytes.data(), 96)));
}

TEST(CollectiveFfiPeerAddressesTest, RejectsMismatchedRegionAndBufferCounts) {
  Storage local;
  std::array<se::DeviceAddressBase, 1> allocations = {local.address()};
  BufferAllocations buffer_allocations(allocations, /*device_ordinal=*/0,
                                       /*memory_allocator=*/nullptr);
  CollectiveMemory collective_memory(buffer_allocations, /*sym_memories=*/{},
                                     /*mcast_memories=*/{},
                                     /*peer_memories=*/{});
  GpuCliqueKey clique_key({kDevice0}, /*num_local_participants=*/1);
  std::array<PeerRegionV3, 1> regions = {Region(0, 16)};

  EXPECT_THAT(internal::ResolvePeerAddressesV3(
                  clique_key, RankId(0), regions,
                  absl::Span<const se::DeviceAddressBase>(), collective_memory),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not match buffer count")));
}

TEST(CollectiveFfiPeerAddressesTest, RejectsRankOutsideClique) {
  Storage local;
  std::array<se::DeviceAddressBase, 1> allocations = {local.address()};
  BufferAllocations buffer_allocations(allocations, /*device_ordinal=*/0,
                                       /*memory_allocator=*/nullptr);
  CollectiveMemory collective_memory(buffer_allocations, /*sym_memories=*/{},
                                     /*mcast_memories=*/{},
                                     /*peer_memories=*/{});
  GpuCliqueKey clique_key({kDevice0}, /*num_local_participants=*/1);

  EXPECT_THAT(internal::ResolvePeerAddressesV3(
                  clique_key, RankId(1), /*peer_regions=*/{}, /*buffers=*/{},
                  collective_memory),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("outside clique size")));
}

TEST(CollectiveFfiPeerAddressesTest, RejectsRegionOutsideLogicalFfiBuffer) {
  Storage local;
  std::array<se::DeviceAddressBase, 1> allocations = {local.address()};
  BufferAllocations buffer_allocations(allocations, /*device_ordinal=*/0,
                                       /*memory_allocator=*/nullptr);
  GpuCliqueKey clique_key({kDevice0}, /*num_local_participants=*/1);
  auto symmetric = std::make_shared<FakeSymmetricMemory>(
      local.address(), std::vector<se::DeviceAddressBase>{local.address()});
  absl::flat_hash_map<CollectiveMemory::Key, std::shared_ptr<SymmetricMemory>>
      symmetric_memories;
  symmetric_memories.emplace(std::make_pair(clique_key, 0), symmetric);
  CollectiveMemory collective_memory(
      buffer_allocations, std::move(symmetric_memories),
      /*mcast_memories=*/{}, /*peer_memories=*/{});
  std::array<PeerRegionV3, 1> regions = {Region(48, 32)};
  std::array<se::DeviceAddressBase, 1> buffers = {
      local.address().GetByteSlice(/*offset_bytes=*/16, /*size_bytes=*/64)};

  EXPECT_THAT(internal::ResolvePeerAddressesV3(clique_key, RankId(0), regions,
                                               buffers, collective_memory),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds containing buffer size 64")));
}

TEST(CollectiveFfiPeerAddressesTest, RejectsMissingSymmetricMemory) {
  Storage local;
  std::array<se::DeviceAddressBase, 1> allocations = {local.address()};
  BufferAllocations buffer_allocations(allocations, /*device_ordinal=*/0,
                                       /*memory_allocator=*/nullptr);
  CollectiveMemory collective_memory(buffer_allocations, /*sym_memories=*/{},
                                     /*mcast_memories=*/{},
                                     /*peer_memories=*/{});
  GpuCliqueKey clique_key({kDevice0}, /*num_local_participants=*/1);
  std::array<PeerRegionV3, 1> regions = {Region(0, 16)};
  std::array<se::DeviceAddressBase, 1> buffers = {local.address()};

  EXPECT_THAT(internal::ResolvePeerAddressesV3(clique_key, RankId(0), regions,
                                               buffers, collective_memory),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("No symmetric memory")));
}

TEST(CollectiveFfiPeerAddressesTest, PropagatesPeerAddressFailure) {
  Storage local;
  Storage peer;
  std::array<se::DeviceAddressBase, 1> allocations = {local.address()};
  BufferAllocations buffer_allocations(allocations, /*device_ordinal=*/0,
                                       /*memory_allocator=*/nullptr);
  GpuCliqueKey clique_key({kDevice0, kDevice1},
                          /*num_local_participants=*/2);
  auto symmetric = std::make_shared<FakeSymmetricMemory>(
      local.address(),
      std::vector<se::DeviceAddressBase>{local.address(), peer.address()});
  symmetric->set_failing_rank(RankId(1));
  absl::flat_hash_map<CollectiveMemory::Key, std::shared_ptr<SymmetricMemory>>
      symmetric_memories;
  symmetric_memories.emplace(std::make_pair(clique_key, 0), symmetric);
  CollectiveMemory collective_memory(
      buffer_allocations, std::move(symmetric_memories),
      /*mcast_memories=*/{}, /*peer_memories=*/{});
  std::array<PeerRegionV3, 1> regions = {Region(0, 16)};
  std::array<se::DeviceAddressBase, 1> buffers = {local.address()};

  EXPECT_THAT(internal::ResolvePeerAddressesV3(clique_key, RankId(0), regions,
                                               buffers, collective_memory),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("injected peer-address failure")));
}

TEST(CollectiveFfiPeerAddressesTest, RejectsPeerRangeOutsideRegistration) {
  Storage local;
  Storage peer;
  std::array<se::DeviceAddressBase, 1> allocations = {local.address()};
  BufferAllocations buffer_allocations(allocations, /*device_ordinal=*/0,
                                       /*memory_allocator=*/nullptr);
  GpuCliqueKey clique_key({kDevice0, kDevice1},
                          /*num_local_participants=*/2);
  se::DeviceAddressBase short_peer(peer.bytes.data(), /*size=*/64);
  auto symmetric = std::make_shared<FakeSymmetricMemory>(
      local.address(),
      std::vector<se::DeviceAddressBase>{local.address(), short_peer});
  absl::flat_hash_map<CollectiveMemory::Key, std::shared_ptr<SymmetricMemory>>
      symmetric_memories;
  symmetric_memories.emplace(std::make_pair(clique_key, 0), symmetric);
  CollectiveMemory collective_memory(
      buffer_allocations, std::move(symmetric_memories),
      /*mcast_memories=*/{}, /*peer_memories=*/{});
  std::array<PeerRegionV3, 1> regions = {Region(32, 48)};
  std::array<se::DeviceAddressBase, 1> buffers = {
      local.address().GetByteSlice(/*offset_bytes=*/16, /*size_bytes=*/96)};

  EXPECT_THAT(internal::ResolvePeerAddressesV3(clique_key, RankId(0), regions,
                                               buffers, collective_memory),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Peer region 0 rank 1")));
}

TEST(CollectiveFfiPeerAddressesTest, RejectsMisalignedPeerAddress) {
  Storage local;
  Storage peer;
  std::array<se::DeviceAddressBase, 1> allocations = {local.address()};
  BufferAllocations buffer_allocations(allocations, /*device_ordinal=*/0,
                                       /*memory_allocator=*/nullptr);
  GpuCliqueKey clique_key({kDevice0, kDevice1},
                          /*num_local_participants=*/2);
  auto symmetric = std::make_shared<FakeSymmetricMemory>(
      local.address(),
      std::vector<se::DeviceAddressBase>{
          local.address(),
          se::DeviceAddressBase(peer.bytes.data() + 1, peer.bytes.size() - 1)});
  absl::flat_hash_map<CollectiveMemory::Key, std::shared_ptr<SymmetricMemory>>
      symmetric_memories;
  symmetric_memories.emplace(std::make_pair(clique_key, 0), symmetric);
  CollectiveMemory collective_memory(
      buffer_allocations, std::move(symmetric_memories),
      /*mcast_memories=*/{}, /*peer_memories=*/{});
  std::array<PeerRegionV3, 1> regions = {Region(0, 16, /*alignment=*/16)};
  std::array<se::DeviceAddressBase, 1> buffers = {local.address()};

  EXPECT_THAT(internal::ResolvePeerAddressesV3(clique_key, RankId(0), regions,
                                               buffers, collective_memory),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not meet required alignment")));
}

TEST(CollectiveFfiPeerAddressesTest, RejectsLocalAddressMismatch) {
  Storage local;
  Storage different_local;
  std::array<se::DeviceAddressBase, 1> allocations = {local.address()};
  BufferAllocations buffer_allocations(allocations, /*device_ordinal=*/0,
                                       /*memory_allocator=*/nullptr);
  GpuCliqueKey clique_key({kDevice0}, /*num_local_participants=*/1);
  auto symmetric = std::make_shared<FakeSymmetricMemory>(
      local.address(),
      std::vector<se::DeviceAddressBase>{different_local.address()});
  absl::flat_hash_map<CollectiveMemory::Key, std::shared_ptr<SymmetricMemory>>
      symmetric_memories;
  symmetric_memories.emplace(std::make_pair(clique_key, 0), symmetric);
  CollectiveMemory collective_memory(
      buffer_allocations, std::move(symmetric_memories),
      /*mcast_memories=*/{}, /*peer_memories=*/{});
  std::array<PeerRegionV3, 1> regions = {Region(0, 16)};
  std::array<se::DeviceAddressBase, 1> buffers = {local.address()};

  EXPECT_THAT(internal::ResolvePeerAddressesV3(clique_key, RankId(0), regions,
                                               buffers, collective_memory),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("local address mismatch")));
}

TEST(CollectiveFfiPeerAddressesTest, RejectsAddressOverflow) {
  Storage local;
  std::array<se::DeviceAddressBase, 1> allocations = {local.address()};
  BufferAllocations buffer_allocations(allocations, /*device_ordinal=*/0,
                                       /*memory_allocator=*/nullptr);
  GpuCliqueKey clique_key({kDevice0}, /*num_local_participants=*/1);
  constexpr uintptr_t kNearAddressLimit =
      std::numeric_limits<uintptr_t>::max() - 7;
  se::DeviceAddressBase overflowing_peer(
      reinterpret_cast<void*>(kNearAddressLimit), /*size=*/64);
  auto symmetric = std::make_shared<FakeSymmetricMemory>(
      local.address(), std::vector<se::DeviceAddressBase>{overflowing_peer});
  absl::flat_hash_map<CollectiveMemory::Key, std::shared_ptr<SymmetricMemory>>
      symmetric_memories;
  symmetric_memories.emplace(std::make_pair(clique_key, 0), symmetric);
  CollectiveMemory collective_memory(
      buffer_allocations, std::move(symmetric_memories),
      /*mcast_memories=*/{}, /*peer_memories=*/{});
  std::array<PeerRegionV3, 1> regions = {
      Region(/*offset=*/16, /*size=*/16, /*alignment=*/1)};
  std::array<se::DeviceAddressBase, 1> buffers = {local.address()};

  EXPECT_THAT(internal::ResolvePeerAddressesV3(clique_key, RankId(0), regions,
                                               buffers, collective_memory),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Address overflow")));
}

}  // namespace
}  // namespace xla::gpu::cutedsl
