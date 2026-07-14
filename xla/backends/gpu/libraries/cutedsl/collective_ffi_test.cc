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

#include "xla/backends/gpu/libraries/cutedsl/ffi.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
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
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives_stub.h"
#include "xla/backends/gpu/libraries/cutedsl/config.pb.h"
#include "xla/backends/gpu/libraries/cutedsl/ffi_abi.h"
#include "xla/backends/gpu/libraries/cutedsl/module.h"
#include "xla/backends/gpu/libraries/cutedsl/runtime_api.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
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
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/mock_platform.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/types.h"
#include "tsl/platform/protobuf.h"

namespace xla::gpu::cutedsl {

namespace ffi = ::xla::ffi;
namespace se = ::stream_executor;
namespace wire = ::xla::gpu::cutedsl::proto;
using ::xla::CollectiveOpGroupMode;
using ::xla::DeviceAssignment;
using ::xla::GlobalDeviceId;
using ::xla::LocalDeviceId;
using ::xla::RankId;
using ::xla::ServiceExecutableRunOptions;
using ::xla::SymmetricMemory;
using ::xla::U64;
using ::xla::gpu::CollectiveCliqueRequests;
using ::xla::gpu::CollectiveCliques;
using ::xla::gpu::CollectiveMemory;
using ::xla::gpu::CollectiveMemoryRequests;
using ::xla::gpu::CollectiveParams;
using ::xla::gpu::CommunicationId;
using ::xla::gpu::GpuCliqueKey;
using ::xla::gpu::GpuCollectivesStub;
using ::xla::gpu::GpuExecutableRunOptions;

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

struct CapturedBuffer {
  void *buffer;
  std::vector<int64_t> shape;
};

struct FakeFunctionHandle {
  std::string prefix;
};

struct FakeRuntime {
  std::string module_bytes;
  std::vector<std::string> function_prefixes;
  std::vector<std::unique_ptr<FakeFunctionHandle>> function_handles;
  std::vector<std::string> invoked_function_prefixes;
  std::vector<size_t> expected_buffer_ranks;
  size_t expected_peer_address_count = 0;
  void *stream = nullptr;
  std::vector<CapturedBuffer> buffers;
  std::vector<uint64_t> peer_addresses;
  const uint64_t *peer_addresses_pointer = nullptr;
  bool peer_addresses_pointer_is_null = false;
  int32_t rank = -1;
  int32_t clique_size = -1;
  int32_t cuda_error = 0;
  int create_count = 0;
  int get_function_count = 0;
  int run_count = 0;
  int destroy_count = 0;
  bool fail_get_function = false;
};

FakeRuntime *fake_runtime = nullptr;

CuteDSLRT_Error_t ModuleCreate(CuteDSLRT_Module_t **module,
                               const unsigned char *bytes, size_t size,
                               const char **, size_t) {
  ++fake_runtime->create_count;
  fake_runtime->module_bytes.assign(reinterpret_cast<const char *>(bytes),
                                    size);
  *module = reinterpret_cast<CuteDSLRT_Module_t *>(fake_runtime);
  return CuteDSLRT_Error_Success;
}

CuteDSLRT_Error_t ModuleGetFunction(CuteDSLRT_Function_t **function,
                                    CuteDSLRT_Module_t *module,
                                    const char *prefix) {
  EXPECT_EQ(module, reinterpret_cast<CuteDSLRT_Module_t *>(fake_runtime));
  ++fake_runtime->get_function_count;
  fake_runtime->function_prefixes.emplace_back(prefix);
  if (fake_runtime->fail_get_function) {
    return CuteDSLRT_Error_CudaError;
  }
  fake_runtime->function_handles.push_back(
      std::make_unique<FakeFunctionHandle>(FakeFunctionHandle{prefix}));
  *function = reinterpret_cast<CuteDSLRT_Function_t *>(
      fake_runtime->function_handles.back().get());
  return CuteDSLRT_Error_Success;
}

CuteDSLRT_Error_t FunctionRun(void *function, void **arguments,
                              size_t num_arguments) {
  auto *handle = reinterpret_cast<FakeFunctionHandle *>(function);
  auto loaded =
      std::find_if(fake_runtime->function_handles.begin(),
                   fake_runtime->function_handles.end(),
                   [&](const std::unique_ptr<FakeFunctionHandle> &candidate) {
                     return candidate.get() == handle;
                   });
  if (loaded == fake_runtime->function_handles.end()) {
    ADD_FAILURE() << "FunctionRun received an unknown function handle";
    return CuteDSLRT_Error_CudaError;
  }
  ++fake_runtime->run_count;
  fake_runtime->invoked_function_prefixes.push_back(handle->prefix);

  size_t expected_arguments =
      1 + fake_runtime->expected_buffer_ranks.size() + 2;
  if (num_arguments != expected_arguments) {
    ADD_FAILURE() << "Expected " << expected_arguments
                  << " packed arguments, got " << num_arguments;
    return CuteDSLRT_Error_CudaError;
  }

  size_t index = 0;
  fake_runtime->stream = *reinterpret_cast<void **>(arguments[index++]);
  void *context_address = *static_cast<void **>(arguments[index++]);
  fake_runtime->buffers.clear();
  for (size_t buffer_rank : fake_runtime->expected_buffer_ranks) {
    auto *descriptor =
        *reinterpret_cast<CuteXlaFfiBuffer **>(arguments[index++]);
    if (descriptor == nullptr ||
        (buffer_rank != 0 && descriptor->shape == nullptr)) {
      ADD_FAILURE() << "Invalid JaxArray descriptor";
      return CuteDSLRT_Error_CudaError;
    }

    CapturedBuffer buffer{descriptor->buffer, {}};
    if (buffer_rank == 0) {
      EXPECT_EQ(descriptor->shape, nullptr);
    } else {
      buffer.shape.assign(descriptor->shape, descriptor->shape + buffer_rank);
    }
    fake_runtime->buffers.push_back(std::move(buffer));
  }

  if (context_address == nullptr) {
    ADD_FAILURE() << "Invalid CollectiveContext descriptor";
    return CuteDSLRT_Error_CudaError;
  }
  CollectiveContextAbi context;
  std::memcpy(&context, context_address, sizeof(context));
  if (fake_runtime->expected_peer_address_count != 0 &&
      context.peer_addresses == nullptr) {
    ADD_FAILURE() << "CollectiveContext has no peer-address table";
    return CuteDSLRT_Error_CudaError;
  }

  fake_runtime->peer_addresses.clear();
  fake_runtime->peer_addresses_pointer = context.peer_addresses;
  fake_runtime->peer_addresses_pointer_is_null =
      context.peer_addresses == nullptr;
  if (fake_runtime->expected_peer_address_count != 0) {
    fake_runtime->peer_addresses.assign(
        context.peer_addresses,
        context.peer_addresses + fake_runtime->expected_peer_address_count);
  }
  fake_runtime->rank = context.rank;
  fake_runtime->clique_size = context.clique_size;
  *static_cast<int32_t *>(arguments[index++]) = fake_runtime->cuda_error;
  EXPECT_EQ(index, num_arguments);
  return CuteDSLRT_Error_Success;
}

CuteDSLRT_Error_t ModuleDestroy(CuteDSLRT_Module_t *module) {
  EXPECT_EQ(module, reinterpret_cast<CuteDSLRT_Module_t *>(fake_runtime));
  ++fake_runtime->destroy_count;
  return CuteDSLRT_Error_Success;
}

const char *GetErrorName(CuteDSLRT_Error_t) { return "FakeRuntimeError"; }
const char *GetErrorString(CuteDSLRT_Error_t) { return "fake failure"; }

const RuntimeApi kRuntimeApi = {
    ModuleCreate,  ModuleGetFunction, FunctionRun,
    ModuleDestroy, GetErrorName,      GetErrorString,
};

class NotifyingEvent final : public se::Event {
 public:
  NotifyingEvent(std::shared_ptr<std::promise<void>> synchronized,
                 std::shared_ptr<std::promise<void>> destroyed)
      : synchronized_(std::move(synchronized)),
        destroyed_(std::move(destroyed)) {}

  ~NotifyingEvent() override { destroyed_->set_value(); }

  absl::Status Synchronize() override {
    synchronized_->set_value();
    return absl::OkStatus();
  }

 private:
  std::shared_ptr<std::promise<void>> synchronized_;
  std::shared_ptr<std::promise<void>> destroyed_;
};

}  // namespace

#if defined(PLATFORM_GOOGLE)
extern "C" CuteDSLRT_Error_t __wrap_CuteDSLRT_Module_Create_From_Bytes(
    CuteDSLRT_Module_t **module, const unsigned char *bytes, size_t size,
    const char **shared_libraries, size_t shared_library_count) {
  return ModuleCreate(module, bytes, size, shared_libraries,
                      shared_library_count);
}

extern "C" CuteDSLRT_Error_t __wrap_CuteDSLRT_Module_Get_Function(
    CuteDSLRT_Function_t **function, CuteDSLRT_Module_t *module,
    const char *prefix) {
  return ModuleGetFunction(function, module, prefix);
}

extern "C" CuteDSLRT_Error_t __wrap_CuteDSLRT_Function_Run(
    void *function, void **arguments, size_t argument_count) {
  return FunctionRun(function, arguments, argument_count);
}

extern "C" CuteDSLRT_Error_t __wrap_CuteDSLRT_Module_Destroy(
    CuteDSLRT_Module_t *module) {
  return ModuleDestroy(module);
}

extern "C" const char *__wrap_CuteDSLRT_GetErrorName(CuteDSLRT_Error_t error) {
  return GetErrorName(error);
}

extern "C" const char *__wrap_CuteDSLRT_GetErrorString(
    CuteDSLRT_Error_t error) {
  return GetErrorString(error);
}
#endif

namespace {

class ScopedFakeRuntime {
 public:
  explicit ScopedFakeRuntime(FakeRuntime *runtime) {
#if !defined(PLATFORM_GOOGLE)
    EXPECT_TRUE(internal::RegisterRuntimeApiForTest(&kRuntimeApi).ok());
#endif
    EXPECT_EQ(fake_runtime, nullptr);
    fake_runtime = runtime;
  }

  ~ScopedFakeRuntime() { fake_runtime = nullptr; }
};

struct TestAttributes {
  ffi::AttributesMap Build() const {
    wire::CollectiveCallConfigV3 config;
    config.set_abi_clique_size(abi_clique_size);
    config.set_group_mode(static_cast<CollectiveOpGroupMode>(group_mode));
    config.set_communication_id(communication_id);
    for (size_t group_index = 0; group_index + 1 < replica_group_offsets.size();
         ++group_index) {
      xla::ReplicaGroup *group = config.add_replica_groups();
      for (int64_t member_index = replica_group_offsets[group_index];
           member_index < replica_group_offsets[group_index + 1];
           ++member_index) {
        group->add_replica_ids(replica_group_members[member_index]);
      }
    }
    for (size_t offset = 0; offset + 5 < peer_regions.size(); offset += 6) {
      wire::PeerRegionProto *region = config.add_peer_regions();
      region->set_endpoint(
          static_cast<wire::PeerRegionEndpointProto>(peer_regions[offset]));
      region->set_buffer_index(peer_regions[offset + 1]);
      region->set_byte_offset(peer_regions[offset + 2]);
      region->set_byte_size(peer_regions[offset + 3]);
      region->set_required_alignment(peer_regions[offset + 4]);
      region->set_memory_kind(
          static_cast<wire::PeerMemoryKindProto>(peer_regions[offset + 5]));
    }
    config.set_barrier_before_launch(barrier_before_launch);

    ffi::CallFrameBuilder::AttributesBuilder attributes;
    if (!omit_module) {
      if (module_as_i64) {
        attributes.Insert("module", int64_t{1});
      } else {
        attributes.Insert("module", module);
      }
    }
    if (!omit_key) {
      if (key_as_i64) {
        attributes.Insert("key", int64_t{1});
      } else if (key.has_value()) {
        attributes.Insert("key", *key);
      } else {
        absl::StatusOr<ModuleImage> image = ModuleImage::Create(module);
        EXPECT_TRUE(image.ok()) << image.status();
        if (image.ok()) attributes.Insert("key", std::string(image->sha256()));
      }
    }
    if (!omit_config) {
      if (config_as_i64) {
        attributes.Insert("config", int64_t{1});
      } else {
        tsl::protobuf::util::JsonPrintOptions options;
        options.preserve_proto_field_names = true;
        std::string json;
        absl::Status status =
            tsl::protobuf::util::MessageToJsonString(config, &json, options);
        EXPECT_TRUE(status.ok()) << status;
        attributes.Insert("config", std::move(json));
      }
    }
    if (add_unknown_attribute) {
      attributes.Insert("unrelated", int64_t{1});
    }
    return attributes.Build();
  }

  int64_t group_mode =
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA;
  int64_t abi_clique_size = 2;
  int64_t communication_id = 17;
  std::vector<int64_t> replica_group_offsets = {0, 2};
  std::vector<int64_t> replica_group_members = {0, 1};
  std::string module = "collective ffi test module";
  std::vector<int64_t> peer_regions;
  std::optional<std::string> key;
  bool barrier_before_launch = false;
  bool add_unknown_attribute = false;
  bool omit_module = false;
  bool module_as_i64 = false;
  bool omit_key = false;
  bool key_as_i64 = false;
  bool omit_config = false;
  bool config_as_i64 = false;
};

class FakeSymmetricMemory final : public SymmetricMemory {
 public:
  FakeSymmetricMemory(se::DeviceAddressBase local_address,
                      std::vector<se::DeviceAddressBase> peer_addresses,
                      std::optional<se::DeviceAddressBase> multimem_address =
                          std::nullopt)
      : local_address_(local_address),
        peer_addresses_(std::move(peer_addresses)),
        multimem_address_(multimem_address) {}

  se::DeviceAddressBase addr() const override { return local_address_; }

  absl::StatusOr<se::DeviceAddressBase> multimem_addr() const override {
    if (!multimem_address_.has_value()) {
      return absl::UnimplementedError("injected multimem unavailability");
    }
    return *multimem_address_;
  }

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
  std::optional<se::DeviceAddressBase> multimem_address_;
  std::optional<RankId> failing_rank_;
};

struct alignas(64) Storage {
  std::array<std::byte, 256> bytes;

  se::DeviceAddressBase address() {
    return se::DeviceAddressBase(bytes.data(), bytes.size());
  }
};

wire::PeerRegionProto Region(int64_t offset, int64_t size,
                             int64_t alignment = 16,
                             wire::PeerMemoryKindProto memory_kind =
                                 wire::PEER_MEMORY_KIND_PROTO_SYMMETRIC) {
  wire::PeerRegionProto region;
  region.set_endpoint(wire::PEER_REGION_ENDPOINT_PROTO_ARGUMENT);
  region.set_buffer_index(0);
  region.set_byte_offset(offset);
  region.set_byte_size(size);
  region.set_required_alignment(alignment);
  region.set_memory_kind(memory_kind);
  return region;
}

wire::CollectiveCallConfigV3 ConfigWithRegions(
    absl::Span<const wire::PeerRegionProto> regions) {
  wire::CollectiveCallConfigV3 config;
  for (const wire::PeerRegionProto &region : regions) {
    *config.add_peer_regions() = region;
  }
  return config;
}

uint64_t AddressValue(void *address, uint64_t offset = 0) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address)) + offset;
}

class TestHostMemoryAllocation final : public se::MemoryAllocation {
 public:
  explicit TestHostMemoryAllocation(uint64_t size)
      : storage_(std::make_unique<std::byte[]>(size)), size_(size) {}

  se::DeviceAddressBase address() const override {
    return se::DeviceAddressBase(storage_.get(), size_);
  }

 private:
  std::unique_ptr<std::byte[]> storage_;
  uint64_t size_;
};

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
    peer_address_table_dimensions_ = {
        static_cast<int64_t>(attributes_.peer_regions.size() / 6),
        attributes_.abi_clique_size};
    peer_address_table_.resize(peer_address_table_dimensions_[0] *
                               peer_address_table_dimensions_[1]);

    ON_CALL(stream_, parent()).WillByDefault(Return(&executor_));
    ON_CALL(stream_, platform_specific_handle())
        .WillByDefault(
            Return(se::Stream::PlatformSpecificHandle{&fake_stream_handle_}));
    ON_CALL(executor_, GetPlatform()).WillByDefault(Return(&platform_));
    ON_CALL(platform_, Name()).WillByDefault(ReturnRef(platform_name_));
    ON_CALL(executor_, HostMemoryAllocate(testing::_))
        .WillByDefault(
            [](uint64_t size)
                -> absl::StatusOr<std::unique_ptr<se::MemoryAllocation>> {
              return std::make_unique<TestHostMemoryAllocation>(size);
            });
    ON_CALL(stream_, Memcpy(testing::A<se::DeviceAddressBase *>(),
                            testing::A<const void *>(), testing::_))
        .WillByDefault([this](se::DeviceAddressBase *destination,
                              const void *source, uint64_t size) {
          ++h2d_copy_count_;
          if (destination == nullptr || destination->size() < size ||
              (size != 0 && (destination->is_null() || source == nullptr))) {
            return absl::InvalidArgumentError("invalid test H2D copy");
          }
          if (size != 0) {
            std::memcpy(destination->opaque(), source, size);
          }
          return absl::OkStatus();
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
    collective_memory_ = std::make_unique<CollectiveMemory>(
        *buffer_allocations_,
        /*sym_memories=*/
        absl::flat_hash_map<CollectiveMemory::Key,
                            std::shared_ptr<SymmetricMemory>>{},
        /*mcast_memories=*/
        absl::flat_hash_map<CollectiveMemory::Key,
                            CollectiveMemory::MulticastMemory>{},
        /*peer_memories=*/
        absl::flat_hash_map<CollectiveMemory::Key,
                            CollectiveMemory::PeerMemory>{});

    context_.state_context = {&states_[0], &states_[1], &states_[2]};
    UpdateGpuContext();
  }

  const absl::Status &status() const { return status_; }

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

  absl::Status Initialize() {
    if (!status_.ok()) return status_;
    if (!registration_.ok()) return registration_.status();
    ffi::CallFrame call_frame = BuildCallFrame();
    return ffi::Invoke(ffi::GetXlaFfiApi(), registration_->bundle.initialize,
                       call_frame, context_, XLA_FFI_ExecutionStage_INITIALIZE);
  }

  absl::Status Execute() {
    if (!status_.ok()) return status_;
    if (!registration_.ok()) return registration_.status();
    ffi::CallFrame call_frame = BuildCallFrame();
    return ffi::Invoke(ffi::GetXlaFfiApi(), registration_->bundle.execute,
                       call_frame, context_, XLA_FFI_ExecutionStage_EXECUTE);
  }

  void ResetPerExecutionState() {
    states_[1] = ffi::ExecutionState();
    states_[2] = ffi::ExecutionState();
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
    UpdateGpuContext();
  }

  const CollectiveCliqueRequests &clique_requests() const {
    return clique_requests_;
  }

  const CollectiveMemoryRequests &memory_requests() const {
    return *memory_requests_;
  }

  se::MockStream &stream() { return stream_; }
  se::MockStreamExecutor &executor() { return executor_; }
  void *platform_stream() { return &fake_stream_handle_; }
  int h2d_copy_count() const { return h2d_copy_count_; }

  void SetPeerAddressTableDimensions(std::array<int64_t, 2> dimensions) {
    peer_address_table_dimensions_ = dimensions;
  }

  const uint64_t *peer_address_table_data() const {
    return peer_address_table_.data();
  }

  void RelocatePeerAddressTable() {
    std::vector<uint64_t> relocated(peer_address_table_.size());
    peer_address_table_.swap(relocated);
  }

 private:
  void UpdateGpuContext() {
    ffi::InvokeContext::GpuContext gpu_context;
    gpu_context.stream = &stream_;
    gpu_context.collective_params = &*collective_params_;
    gpu_context.collective_clique_requests = &clique_requests_;
    gpu_context.collective_memory_requests = memory_requests_.get();
    gpu_context.collective_cliques = &collective_cliques_;
    gpu_context.collective_memory = collective_memory_.get();
    context_.backend_context = gpu_context;
  }

  ffi::CallFrame BuildCallFrame() const {
    ffi::CallFrameBuilder builder(arguments_.size(), results_.size() + 1);
    for (const se::DeviceAddressBase &argument : arguments_) {
      std::array<int64_t, 1> dimensions = {
          static_cast<int64_t>(argument.size())};
      builder.AddBufferArg(argument, U8, dimensions);
    }
    void *peer_address_table =
        peer_address_table_.empty() ? nullptr : peer_address_table_.data();
    builder.AddBufferRet(
        se::DeviceAddressBase(peer_address_table,
                              peer_address_table_.size() * sizeof(uint64_t)),
        U64, peer_address_table_dimensions_);
    for (const se::DeviceAddressBase &result : results_) {
      std::array<int64_t, 1> dimensions = {static_cast<int64_t>(result.size())};
      builder.AddBufferRet(result, U8, dimensions);
    }
    builder.AddAttributes(attributes_.Build());
    return builder.Build();
  }

  TestAttributes attributes_;
  std::vector<se::DeviceAddressBase> arguments_;
  std::vector<se::DeviceAddressBase> results_;
  mutable std::vector<uint64_t> peer_address_table_;
  std::array<int64_t, 2> peer_address_table_dimensions_ = {};
  int h2d_copy_count_ = 0;
  int fake_stream_handle_ = 0;
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

TEST(CollectiveFfiTest, RequiresProtoJsonConfigAttribute) {
  TestAttributes attributes;
  attributes.omit_config = true;
  CollectiveFfiInvocation missing(std::move(attributes), /*replica_count=*/2,
                                  /*partition_count=*/1,
                                  /*current_device=*/0);
  ASSERT_THAT(missing.status(), IsOk());
  EXPECT_THAT(missing.Instantiate(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("attribute `config`")));

  attributes = TestAttributes();
  attributes.config_as_i64 = true;
  CollectiveFfiInvocation wrong_type(std::move(attributes), /*replica_count=*/2,
                                     /*partition_count=*/1,
                                     /*current_device=*/0);
  ASSERT_THAT(wrong_type.status(), IsOk());
  EXPECT_THAT(wrong_type.Instantiate(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("attribute `config`")));
}

TEST(CollectiveFfiTest, RequiresModuleAttribute) {
  TestAttributes attributes;
  attributes.omit_module = true;
  CollectiveFfiInvocation missing(std::move(attributes), /*replica_count=*/2,
                                  /*partition_count=*/1,
                                  /*current_device=*/0);
  ASSERT_THAT(missing.status(), IsOk());
  EXPECT_THAT(missing.Instantiate(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("attribute `module`")));

  attributes = TestAttributes();
  attributes.module_as_i64 = true;
  CollectiveFfiInvocation wrong_type(std::move(attributes), /*replica_count=*/2,
                                     /*partition_count=*/1,
                                     /*current_device=*/0);
  ASSERT_THAT(wrong_type.status(), IsOk());
  EXPECT_THAT(wrong_type.Instantiate(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("attribute `module`")));
}

TEST(CollectiveFfiTest, RequiresKeyAttribute) {
  TestAttributes attributes;
  attributes.omit_key = true;
  CollectiveFfiInvocation missing(std::move(attributes), /*replica_count=*/2,
                                  /*partition_count=*/1,
                                  /*current_device=*/0);
  ASSERT_THAT(missing.status(), IsOk());
  EXPECT_THAT(missing.Instantiate(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("attribute `key`")));

  attributes = TestAttributes();
  attributes.key_as_i64 = true;
  CollectiveFfiInvocation wrong_type(std::move(attributes), /*replica_count=*/2,
                                     /*partition_count=*/1,
                                     /*current_device=*/0);
  ASSERT_THAT(wrong_type.status(), IsOk());
  EXPECT_THAT(wrong_type.Instantiate(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("attribute `key`")));
}

TEST(CollectiveFfiTest, RejectsMismatchedModuleKey) {
  TestAttributes attributes;
  attributes.key = std::string(kModuleDigestSize, '\0');
  CollectiveFfiInvocation invocation(std::move(attributes),
                                     /*replica_count=*/2,
                                     /*partition_count=*/1,
                                     /*current_device=*/0);
  ASSERT_THAT(invocation.status(), IsOk());
  EXPECT_THAT(invocation.Instantiate(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not match")));
}

TEST(CollectiveFfiTest, IgnoresUnknownOuterAttributes) {
  TestAttributes attributes;
  attributes.add_unknown_attribute = true;
  CollectiveFfiInvocation invocation(std::move(attributes),
                                     /*replica_count=*/2,
                                     /*partition_count=*/1,
                                     /*current_device=*/0);

  ASSERT_THAT(invocation.status(), IsOk());
  EXPECT_THAT(invocation.Instantiate(), IsOk());
}

TEST(CollectiveFfiTest, RejectsPeerAddressTableWithWrongShape) {
  TestAttributes attributes;
  attributes.peer_regions = {
      wire::PEER_REGION_ENDPOINT_PROTO_ARGUMENT, 0, 0, 16, 16,
      wire::PEER_MEMORY_KIND_PROTO_SYMMETRIC,
  };
  CollectiveFfiInvocation invocation(std::move(attributes), /*replica_count=*/2,
                                     /*partition_count=*/1,
                                     /*current_device=*/0);
  invocation.SetPeerAddressTableDimensions({2, 1});

  ASSERT_THAT(invocation.status(), IsOk());
  EXPECT_THAT(invocation.Instantiate(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("has shape [2, 1]; expected [1, 2]")));
}

TEST(CollectiveFfiPrepareTest, ResolvesEverySupportedCollectiveGroupMode) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);

  struct TestCase {
    const char *name;
    CollectiveOpGroupMode mode;
    std::vector<int64_t> group_offsets;
    std::vector<int64_t> group_members;
    std::vector<GlobalDeviceId> expected_clique;
  };
  std::array<TestCase, 4> test_cases = {{
      {"cross replica",
       CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA,
       {0, 2},
       {0, 1},
       {kDevice0, GlobalDeviceId(2)}},
      {"cross partition",
       CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION,
       {0, 2},
       {0, 1},
       {kDevice0, kDevice1}},
      {"cross replica and partition",
       CollectiveOpGroupMode::
           COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION,
       {0, 2},
       {0, 1},
       {kDevice0, kDevice1, GlobalDeviceId(2), GlobalDeviceId(3)}},
      {"flattened id",
       CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID,
       {0, 2, 4},
       {0, 3, 1, 2},
       {kDevice0, GlobalDeviceId(3)}},
  }};

  for (const TestCase &test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    TestAttributes attributes;
    attributes.module = std::string("module for ") + test_case.name;
    attributes.group_mode = test_case.mode;
    attributes.abi_clique_size =
        static_cast<int64_t>(test_case.expected_clique.size());
    attributes.replica_group_offsets = test_case.group_offsets;
    attributes.replica_group_members = test_case.group_members;
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

TEST(CollectiveFfiPrepareTest,
     RejectsAbiCliqueSizeMismatchBeforeLoadingModuleOrRequestingResources) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);

  TestAttributes attributes;
  attributes.group_mode = CollectiveOpGroupMode::
      COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION;
  // The replica-group row has two members, but this mode expands it over both
  // partitions, so the runtime clique has four devices.
  attributes.abi_clique_size = 2;
  CollectiveFfiInvocation invocation(std::move(attributes),
                                     /*replica_count=*/2,
                                     /*partition_count=*/2,
                                     /*current_device=*/0);

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Instantiate(), IsOk());
  EXPECT_THAT(invocation.Prepare(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("ABI clique size 2 does not match runtime "
                                 "clique size 4")));
  EXPECT_EQ(runtime.create_count, 0);
  EXPECT_EQ(invocation.clique_requests().size(), 0);
  EXPECT_EQ(invocation.memory_requests().symmetric_size(), 0);
}

TEST(CollectiveFfiPrepareTest, RejectsReplicaGroupOutsideRuntimeDomain) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);

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

TEST(CollectiveFfiPrepareTest, RejectsLogicalBufferRangeBeforeLoadingModule) {
  Storage allocation;
  TestAttributes attributes;
  attributes.peer_regions = {
      wire::PEER_REGION_ENDPOINT_PROTO_ARGUMENT, 0, 48, 32, 16,
      wire::PEER_MEMORY_KIND_PROTO_SYMMETRIC,
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
      wire::PEER_REGION_ENDPOINT_PROTO_ARGUMENT, 0, 0, 16, 16,
      wire::PEER_MEMORY_KIND_PROTO_SYMMETRIC,
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

  Storage allocation0;
  Storage allocation1;
  TestAttributes attributes;
  attributes.peer_regions = {
      wire::PEER_REGION_ENDPOINT_PROTO_ARGUMENT,
      0,
      0,
      16,
      16,
      wire::PEER_MEMORY_KIND_PROTO_SYMMETRIC,
      wire::PEER_REGION_ENDPOINT_PROTO_ARGUMENT,
      1,
      16,
      16,
      16,
      wire::PEER_MEMORY_KIND_PROTO_SYMMETRIC,
      wire::PEER_REGION_ENDPOINT_PROTO_RESULT,
      0,
      0,
      16,
      16,
      wire::PEER_MEMORY_KIND_PROTO_MULTIMEM,
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
     AcceptsOptInPrefixBarrierAndPreloadsCutlassCall) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);

  TestAttributes attributes;
  // Host-only coverage verifies that a configured barrier remains a prefix
  // through Instantiate and Prepare. Actual barrier enqueue ordering requires
  // the registered GPU barrier kernel and is covered by the Phase 4 GPU test.
  attributes.barrier_before_launch = true;
  attributes.module = "collective prefix-barrier module";
  CollectiveFfiInvocation invocation(std::move(attributes),
                                     /*replica_count=*/2,
                                     /*partition_count=*/1,
                                     /*current_device=*/0);

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Instantiate(), IsOk());
  ASSERT_THAT(invocation.Prepare(), IsOk());

  EXPECT_EQ(runtime.create_count, 1);
  EXPECT_EQ(runtime.get_function_count, 1);
  EXPECT_EQ(runtime.run_count, 0);
  EXPECT_THAT(runtime.function_prefixes, ElementsAre("cutlass_call"));
  EXPECT_EQ(runtime.function_handles.size(), 1);

  std::vector<CollectiveCliqueRequests::CliqueRequest> requests =
      invocation.clique_requests().OrderedRequestedCliques();
  ASSERT_EQ(requests.size(), 1);
  EXPECT_TRUE(requests[0].dev_comms.empty());
  EXPECT_FALSE(requests[0].barrier_after_module_execution_requested);
}

TEST(CollectiveFfiPrepareTest, RetainsModuleAcrossSequentialExecutions) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);

  TestAttributes attributes;
  attributes.module = "collective retained module";
  {
    CollectiveFfiInvocation invocation(std::move(attributes),
                                       /*replica_count=*/2,
                                       /*partition_count=*/1,
                                       /*current_device=*/0);

    ASSERT_THAT(invocation.status(), IsOk());
    ASSERT_THAT(invocation.Instantiate(), IsOk());
    ASSERT_THAT(invocation.Prepare(), IsOk());
    EXPECT_EQ(runtime.create_count, 1);
    EXPECT_EQ(runtime.get_function_count, 1);
    EXPECT_EQ(runtime.destroy_count, 0);

    invocation.ResetPerExecutionState();
    ASSERT_THAT(invocation.Prepare(), IsOk());
    EXPECT_EQ(runtime.create_count, 1);
    EXPECT_EQ(runtime.get_function_count, 1);
    EXPECT_EQ(runtime.destroy_count, 0);
  }
  EXPECT_EQ(runtime.destroy_count, 1);
}

TEST(CollectiveFfiPrepareTest, RejectsMissingFunctionBeforeRequestingClique) {
  FakeRuntime runtime;
  runtime.fail_get_function = true;
  ScopedFakeRuntime scoped_runtime(&runtime);

  TestAttributes attributes;
  attributes.module = "module missing function";
  CollectiveFfiInvocation invocation(std::move(attributes),
                                     /*replica_count=*/2,
                                     /*partition_count=*/1,
                                     /*current_device=*/0);

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Instantiate(), IsOk());
  EXPECT_THAT(invocation.Prepare(), StatusIs(absl::StatusCode::kInternal,
                                             HasSubstr("cutlass_call")));
  EXPECT_EQ(runtime.create_count, 1);
  EXPECT_EQ(runtime.get_function_count, 1);
  EXPECT_TRUE(invocation.clique_requests().OrderedRequestedCliques().empty());
}

TEST(CollectiveFfiExecuteTest, ExecutesSingleCutlassCall) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);

  TestAttributes attributes;
  attributes.module = "collective single-launch test module";
  CollectiveFfiInvocation invocation(std::move(attributes),
                                     /*replica_count=*/2,
                                     /*partition_count=*/1,
                                     /*current_device=*/0);

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Instantiate(), IsOk());
  ASSERT_THAT(invocation.Prepare(), IsOk());
  ASSERT_THAT(invocation.Initialize(), IsOk());
  EXPECT_EQ(invocation.h2d_copy_count(), 0);

  EXPECT_CALL(invocation.executor(), CreateEvent())
      .WillOnce([]() -> absl::StatusOr<std::unique_ptr<se::Event>> {
        return absl::UnavailableError("injected completion-event failure");
      });
  EXPECT_CALL(invocation.stream(), BlockHostUntilDone())
      .WillOnce(Return(absl::OkStatus()));
  ASSERT_THAT(invocation.Execute(), IsOk());
  EXPECT_EQ(invocation.h2d_copy_count(), 0);

  EXPECT_EQ(runtime.create_count, 1);
  EXPECT_THAT(runtime.function_prefixes, ElementsAre("cutlass_call"));
  EXPECT_EQ(runtime.function_handles.size(), 1);
  EXPECT_THAT(runtime.invoked_function_prefixes,
              ElementsAre("cutlass_call"));
  EXPECT_TRUE(runtime.peer_addresses_pointer_is_null);
  EXPECT_TRUE(runtime.peer_addresses.empty());
  EXPECT_EQ(runtime.rank, 0);
  EXPECT_EQ(runtime.clique_size, 2);
}

TEST(CollectiveFfiExecuteTest, RetainsResourcesUntilCompletionEvent) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);

  TestAttributes attributes;
  attributes.module = "collective completion-event test module";
  CollectiveFfiInvocation invocation(std::move(attributes),
                                     /*replica_count=*/2,
                                     /*partition_count=*/1,
                                     /*current_device=*/0);

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Instantiate(), IsOk());
  ASSERT_THAT(invocation.Prepare(), IsOk());
  ASSERT_THAT(invocation.Initialize(), IsOk());

  auto synchronized = std::make_shared<std::promise<void>>();
  std::future<void> synchronized_future = synchronized->get_future();
  auto destroyed = std::make_shared<std::promise<void>>();
  std::future<void> destroyed_future = destroyed->get_future();
  se::Event *created_event = nullptr;
  EXPECT_CALL(invocation.executor(), CreateEvent())
      .WillOnce([&]() -> absl::StatusOr<std::unique_ptr<se::Event>> {
        auto event = std::make_unique<NotifyingEvent>(synchronized, destroyed);
        created_event = event.get();
        return std::unique_ptr<se::Event>(std::move(event));
      });
  EXPECT_CALL(invocation.stream(), RecordEvent(testing::_))
      .WillOnce([&](se::Event *recorded_event) {
        EXPECT_EQ(recorded_event, created_event);
        return absl::OkStatus();
      });
  EXPECT_CALL(invocation.stream(), BlockHostUntilDone()).Times(0);
  ASSERT_THAT(invocation.Execute(), IsOk());

  EXPECT_EQ(synchronized_future.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  EXPECT_EQ(destroyed_future.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  EXPECT_THAT(runtime.invoked_function_prefixes, ElementsAre("cutlass_call"));
}

TEST(CollectiveFfiExecuteTest,
     CopiesRegionAddressesToDeviceAndPacksCanonicalLaunchFrame) {
  FakeRuntime runtime;
  runtime.expected_buffer_ranks = {1, 1};
  runtime.expected_peer_address_count = 4;
  ScopedFakeRuntime scoped_runtime(&runtime);

  Storage local0;
  Storage remote0;
  Storage local1;
  Storage remote1;
  Storage multimem1;
  se::DeviceAddressBase argument =
      local0.address().GetByteSlice(/*offset_bytes=*/32, /*size_bytes=*/96);
  se::DeviceAddressBase result =
      local1.address().GetByteSlice(/*offset_bytes=*/64, /*size_bytes=*/128);

  TestAttributes attributes;
  attributes.module = "collective canonical-frame test module";
  attributes.peer_regions = {
      wire::PEER_REGION_ENDPOINT_PROTO_ARGUMENT,
      0,
      16,
      32,
      16,
      wire::PEER_MEMORY_KIND_PROTO_SYMMETRIC,
      wire::PEER_REGION_ENDPOINT_PROTO_RESULT,
      0,
      32,
      16,
      32,
      wire::PEER_MEMORY_KIND_PROTO_MULTIMEM,
  };
  CollectiveFfiInvocation invocation(
      std::move(attributes), /*replica_count=*/2, /*partition_count=*/1,
      /*current_device=*/1,
      /*allocations=*/{local0.address(), local1.address()},
      /*arguments=*/{argument}, /*results=*/{result});

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Instantiate(), IsOk());
  ASSERT_THAT(invocation.Prepare(), IsOk());

  std::vector<CollectiveCliqueRequests::CliqueRequest> requests =
      invocation.clique_requests().OrderedRequestedCliques();
  ASSERT_EQ(requests.size(), 1);
  const GpuCliqueKey &clique_key = requests[0].key;

  auto symmetric0 = std::make_shared<FakeSymmetricMemory>(
      local0.address(),
      std::vector<se::DeviceAddressBase>{remote0.address(), local0.address()});
  auto symmetric1 = std::make_shared<FakeSymmetricMemory>(
      local1.address(),
      std::vector<se::DeviceAddressBase>{remote1.address(), local1.address()},
      multimem1.address());
  absl::flat_hash_map<CollectiveMemory::Key, std::shared_ptr<SymmetricMemory>>
      symmetric_memories;
  symmetric_memories.emplace(std::make_pair(clique_key, 0), symmetric0);
  symmetric_memories.emplace(std::make_pair(clique_key, 1), symmetric1);
  invocation.SetSymmetricMemories(std::move(symmetric_memories));

  ASSERT_THAT(invocation.Initialize(), IsOk());
  EXPECT_EQ(invocation.h2d_copy_count(), 0);
  const uint64_t *initialized_table = invocation.peer_address_table_data();
  invocation.RelocatePeerAddressTable();
  ASSERT_NE(invocation.peer_address_table_data(), initialized_table);
  EXPECT_CALL(invocation.executor(), CreateEvent())
      .WillOnce([]() -> absl::StatusOr<std::unique_ptr<se::Event>> {
        return absl::UnavailableError("injected completion-event failure");
      });
  EXPECT_CALL(invocation.stream(), BlockHostUntilDone())
      .WillOnce(Return(absl::OkStatus()));
  ASSERT_THAT(invocation.Execute(), IsOk());
  EXPECT_EQ(invocation.h2d_copy_count(), 1);

  EXPECT_EQ(runtime.run_count, 1);
  EXPECT_EQ(runtime.stream, invocation.platform_stream());
  ASSERT_EQ(runtime.buffers.size(), 2);
  EXPECT_EQ(runtime.buffers[0].buffer, argument.opaque());
  EXPECT_THAT(runtime.buffers[0].shape, ElementsAre(96));
  EXPECT_EQ(runtime.buffers[1].buffer, result.opaque());
  EXPECT_THAT(runtime.buffers[1].shape, ElementsAre(128));
  EXPECT_THAT(runtime.peer_addresses,
              ElementsAre(AddressValue(remote0.bytes.data(), 48),
                          AddressValue(local0.bytes.data(), 48),
                          AddressValue(multimem1.bytes.data(), 96),
                          AddressValue(multimem1.bytes.data(), 96)));
  EXPECT_FALSE(runtime.peer_addresses_pointer_is_null);
  EXPECT_EQ(runtime.peer_addresses_pointer,
            invocation.peer_address_table_data());
  EXPECT_EQ(runtime.rank, 1);
  EXPECT_EQ(runtime.clique_size, 2);
}

TEST(CollectiveFfiExecuteTest, PropagatesCudaErrorWrittenByGeneratedFunction) {
  FakeRuntime runtime;
  runtime.cuda_error = 719;
  ScopedFakeRuntime scoped_runtime(&runtime);

  TestAttributes attributes;
  attributes.module = "collective CUDA-error test module";
  CollectiveFfiInvocation invocation(std::move(attributes),
                                     /*replica_count=*/2,
                                     /*partition_count=*/1,
                                     /*current_device=*/0);

  ASSERT_THAT(invocation.status(), IsOk());
  ASSERT_THAT(invocation.Instantiate(), IsOk());
  ASSERT_THAT(invocation.Prepare(), IsOk());
  ASSERT_THAT(invocation.Initialize(), IsOk());

  EXPECT_CALL(invocation.executor(), CreateEvent())
      .WillOnce([]() -> absl::StatusOr<std::unique_ptr<se::Event>> {
        return absl::UnavailableError("injected completion-event failure");
      });
  EXPECT_CALL(invocation.stream(), BlockHostUntilDone())
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_THAT(invocation.Execute(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("returned CUDA error 719")));
  EXPECT_EQ(runtime.run_count, 1);
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

  std::array<wire::PeerRegionProto, 2> regions = {
      Region(/*offset=*/16, /*size=*/32, /*alignment=*/16),
      Region(/*offset=*/32, /*size=*/16, /*alignment=*/32),
  };
  std::array<se::DeviceAddressBase, 2> buffers = {
      local0.address().GetByteSlice(/*offset_bytes=*/32, /*size_bytes=*/96),
      local1.address().GetByteSlice(/*offset_bytes=*/64, /*size_bytes=*/128),
  };

  absl::StatusOr<std::vector<uint64_t>> addresses =
      internal::ResolvePeerAddresses(clique_key, RankId(0),
                                     ConfigWithRegions(regions), buffers,
                                     collective_memory);

  ASSERT_THAT(addresses, IsOk());
  EXPECT_THAT(*addresses, ElementsAre(AddressValue(local0.bytes.data(), 48),
                                      AddressValue(peer0.bytes.data(), 48),
                                      AddressValue(local1.bytes.data(), 96),
                                      AddressValue(peer1.bytes.data(), 96)));
}

TEST(CollectiveFfiPeerAddressesTest,
     ResolvesMultimemAliasWithBufferAndRegionOffsets) {
  Storage local;
  Storage peer;
  Storage multimem;
  std::array<se::DeviceAddressBase, 1> allocations = {local.address()};
  BufferAllocations buffer_allocations(allocations, /*device_ordinal=*/0,
                                       /*memory_allocator=*/nullptr);

  GpuCliqueKey clique_key({kDevice0, kDevice1},
                          /*num_local_participants=*/2, CommunicationId(7));
  auto symmetric = std::make_shared<FakeSymmetricMemory>(
      local.address(),
      std::vector<se::DeviceAddressBase>{local.address(), peer.address()},
      multimem.address());
  symmetric->set_failing_rank(RankId(1));
  absl::flat_hash_map<CollectiveMemory::Key, std::shared_ptr<SymmetricMemory>>
      symmetric_memories;
  symmetric_memories.emplace(std::make_pair(clique_key, 0), symmetric);
  CollectiveMemory collective_memory(
      buffer_allocations, std::move(symmetric_memories),
      /*mcast_memories=*/{}, /*peer_memories=*/{});

  std::array<wire::PeerRegionProto, 1> regions = {Region(
      /*offset=*/16, /*size=*/32, /*alignment=*/16,
      wire::PEER_MEMORY_KIND_PROTO_MULTIMEM)};
  std::array<se::DeviceAddressBase, 1> buffers = {
      local.address().GetByteSlice(/*offset_bytes=*/32, /*size_bytes=*/96)};

  absl::StatusOr<std::vector<uint64_t>> addresses =
      internal::ResolvePeerAddresses(clique_key, RankId(0),
                                     ConfigWithRegions(regions), buffers,
                                     collective_memory);

  ASSERT_THAT(addresses, IsOk());
  EXPECT_THAT(*addresses,
              ElementsAre(AddressValue(multimem.bytes.data(), 48),
                          AddressValue(multimem.bytes.data(), 48)));
}

TEST(CollectiveFfiPeerAddressesTest, RejectsUnavailableMultimemAlias) {
  Storage local;
  std::array<se::DeviceAddressBase, 1> allocations = {local.address()};
  BufferAllocations buffer_allocations(allocations, /*device_ordinal=*/0,
                                       /*memory_allocator=*/nullptr);
  GpuCliqueKey clique_key({kDevice0}, /*num_local_participants=*/1);
  auto symmetric = std::make_shared<FakeSymmetricMemory>(
      local.address(),
      std::vector<se::DeviceAddressBase>{local.address()});
  absl::flat_hash_map<CollectiveMemory::Key, std::shared_ptr<SymmetricMemory>>
      symmetric_memories;
  symmetric_memories.emplace(std::make_pair(clique_key, 0), symmetric);
  CollectiveMemory collective_memory(
      buffer_allocations, std::move(symmetric_memories),
      /*mcast_memories=*/{}, /*peer_memories=*/{});
  std::array<wire::PeerRegionProto, 1> regions = {Region(
      /*offset=*/0, /*size=*/16, /*alignment=*/16,
      wire::PEER_MEMORY_KIND_PROTO_MULTIMEM)};
  std::array<se::DeviceAddressBase, 1> buffers = {local.address()};

  EXPECT_THAT(internal::ResolvePeerAddresses(
                  clique_key, RankId(0), ConfigWithRegions(regions), buffers,
                  collective_memory),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("injected multimem unavailability")));
}

TEST(CollectiveFfiPeerAddressesTest, UsesFfiAddressForLocalPeerAlias) {
  Storage local;
  Storage local_peer_alias;
  std::array<se::DeviceAddressBase, 1> allocations = {local.address()};
  BufferAllocations buffer_allocations(allocations, /*device_ordinal=*/0,
                                       /*memory_allocator=*/nullptr);

  GpuCliqueKey clique_key({kDevice0}, /*num_local_participants=*/1);
  auto symmetric = std::make_shared<FakeSymmetricMemory>(
      local.address(),
      std::vector<se::DeviceAddressBase>{local_peer_alias.address()});
  absl::flat_hash_map<CollectiveMemory::Key, std::shared_ptr<SymmetricMemory>>
      symmetric_memories;
  symmetric_memories.emplace(std::make_pair(clique_key, 0), symmetric);
  CollectiveMemory collective_memory(
      buffer_allocations, std::move(symmetric_memories),
      /*mcast_memories=*/{}, /*peer_memories=*/{});

  std::array<wire::PeerRegionProto, 1> regions = {
      Region(/*offset=*/16, /*size=*/32, /*alignment=*/16)};
  std::array<se::DeviceAddressBase, 1> buffers = {
      local.address().GetByteSlice(/*offset_bytes=*/32, /*size_bytes=*/96)};

  absl::StatusOr<std::vector<uint64_t>> addresses =
      internal::ResolvePeerAddresses(clique_key, RankId(0),
                                     ConfigWithRegions(regions), buffers,
                                     collective_memory);

  ASSERT_THAT(addresses, IsOk());
  EXPECT_THAT(*addresses, ElementsAre(AddressValue(local.bytes.data(), 48)));
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
  std::array<wire::PeerRegionProto, 1> regions = {Region(0, 16)};

  EXPECT_THAT(internal::ResolvePeerAddresses(
                  clique_key, RankId(0), ConfigWithRegions(regions),
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
  wire::CollectiveCallConfigV3 config;

  EXPECT_THAT(internal::ResolvePeerAddresses(clique_key, RankId(1), config,
                                             /*buffers=*/{}, collective_memory),
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
  std::array<wire::PeerRegionProto, 1> regions = {Region(48, 32)};
  std::array<se::DeviceAddressBase, 1> buffers = {
      local.address().GetByteSlice(/*offset_bytes=*/16, /*size_bytes=*/64)};

  EXPECT_THAT(internal::ResolvePeerAddresses(clique_key, RankId(0),
                                             ConfigWithRegions(regions),
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
  std::array<wire::PeerRegionProto, 1> regions = {Region(0, 16)};
  std::array<se::DeviceAddressBase, 1> buffers = {local.address()};

  EXPECT_THAT(internal::ResolvePeerAddresses(clique_key, RankId(0),
                                             ConfigWithRegions(regions),
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
  std::array<wire::PeerRegionProto, 1> regions = {Region(0, 16)};
  std::array<se::DeviceAddressBase, 1> buffers = {local.address()};

  EXPECT_THAT(internal::ResolvePeerAddresses(clique_key, RankId(0),
                                             ConfigWithRegions(regions),
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
  std::array<wire::PeerRegionProto, 1> regions = {Region(32, 48)};
  std::array<se::DeviceAddressBase, 1> buffers = {
      local.address().GetByteSlice(/*offset_bytes=*/16, /*size_bytes=*/96)};

  EXPECT_THAT(internal::ResolvePeerAddresses(clique_key, RankId(0),
                                             ConfigWithRegions(regions),
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
  std::array<wire::PeerRegionProto, 1> regions = {
      Region(0, 16, /*alignment=*/16)};
  std::array<se::DeviceAddressBase, 1> buffers = {local.address()};

  EXPECT_THAT(internal::ResolvePeerAddresses(clique_key, RankId(0),
                                             ConfigWithRegions(regions),
                                             buffers, collective_memory),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not meet required alignment")));
}

TEST(CollectiveFfiPeerAddressesTest, RejectsSymmetricBackingAddressMismatch) {
  Storage local;
  Storage different_local;
  std::array<se::DeviceAddressBase, 1> allocations = {local.address()};
  BufferAllocations buffer_allocations(allocations, /*device_ordinal=*/0,
                                       /*memory_allocator=*/nullptr);
  GpuCliqueKey clique_key({kDevice0}, /*num_local_participants=*/1);
  auto symmetric = std::make_shared<FakeSymmetricMemory>(
      different_local.address(),
      std::vector<se::DeviceAddressBase>{different_local.address()});
  absl::flat_hash_map<CollectiveMemory::Key, std::shared_ptr<SymmetricMemory>>
      symmetric_memories;
  symmetric_memories.emplace(std::make_pair(clique_key, 0), symmetric);
  CollectiveMemory collective_memory(
      buffer_allocations, std::move(symmetric_memories),
      /*mcast_memories=*/{}, /*peer_memories=*/{});
  std::array<wire::PeerRegionProto, 1> regions = {Region(0, 16)};
  std::array<se::DeviceAddressBase, 1> buffers = {local.address()};

  EXPECT_THAT(internal::ResolvePeerAddresses(clique_key, RankId(0),
                                             ConfigWithRegions(regions),
                                             buffers, collective_memory),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("backing address")));
}

TEST(CollectiveFfiPeerAddressesTest, RejectsAddressOverflow) {
  Storage local;
  std::array<se::DeviceAddressBase, 1> allocations = {local.address()};
  BufferAllocations buffer_allocations(allocations, /*device_ordinal=*/0,
                                       /*memory_allocator=*/nullptr);
  GpuCliqueKey clique_key({kDevice0, kDevice1},
                          /*num_local_participants=*/2);
  constexpr uintptr_t kNearAddressLimit =
      std::numeric_limits<uintptr_t>::max() - 7;
  se::DeviceAddressBase overflowing_peer(
      reinterpret_cast<void *>(kNearAddressLimit), /*size=*/64);
  auto symmetric = std::make_shared<FakeSymmetricMemory>(
      local.address(),
      std::vector<se::DeviceAddressBase>{local.address(), overflowing_peer});
  absl::flat_hash_map<CollectiveMemory::Key, std::shared_ptr<SymmetricMemory>>
      symmetric_memories;
  symmetric_memories.emplace(std::make_pair(clique_key, 0), symmetric);
  CollectiveMemory collective_memory(
      buffer_allocations, std::move(symmetric_memories),
      /*mcast_memories=*/{}, /*peer_memories=*/{});
  std::array<wire::PeerRegionProto, 1> regions = {
      Region(/*offset=*/16, /*size=*/16, /*alignment=*/1)};
  std::array<se::DeviceAddressBase, 1> buffers = {local.address()};

  EXPECT_THAT(internal::ResolvePeerAddresses(clique_key, RankId(0),
                                             ConfigWithRegions(regions),
                                             buffers, collective_memory),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Address overflow")));
}

}  // namespace
}  // namespace xla::gpu::cutedsl
