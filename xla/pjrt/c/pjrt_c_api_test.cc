/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/pjrt/c/pjrt_c_api_test.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_parser.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace pjrt {
namespace {

using ::testing::Contains;

// Serialized `ModuleOp` that does add 1.
constexpr absl::string_view module_add_one =
    R"(module {
func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "mhlo.copy"(%arg0) : (tensor<f32>) -> tensor<f32>
  %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
  %2 = mhlo.add %0, %1 : tensor<f32>
  return %2 : tensor<f32>
}})";

// HLO sample code from go/hlo-text
constexpr absl::string_view hlo_string =
    R"(
HloModule TupleCreate_module:
ENTRY %TupleCreate.v4 (v1: f32[], v2: f32[3], v3: f32[2,3]) -> (f32[], f32[3], f32[2,3]) {
  %v1 = f32[] parameter(0)
  %v2 = f32[3]{0} parameter(1)
  %v3 = f32[2,3]{1,0} parameter(2)
  ROOT %tuple = (f32[], f32[3]{0}, f32[2,3]{1,0}) tuple(f32[] %v1, f32[3]{0} %v2, f32[2,3]{1,0} %v3)
}
)";

PJRT_Client_BufferFromHostBuffer_Args CreateBufferFromHostBufferArgs(
    const std::vector<float>& data, const Shape& shape,
    const xla::PjRtClient::HostBufferSemantics host_buffer_semantics,
    PJRT_Client* client, PJRT_Device* device = nullptr) {
  PJRT_Client_BufferFromHostBuffer_Args args;
  args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  args.priv = nullptr;

  args.data = data.data();
  args.type = ::pjrt::ConvertToPjRtBufferType(shape.element_type());
  args.dims = shape.dimensions().data();
  args.num_dims = shape.dimensions().size();
  args.byte_strides = nullptr;
  args.num_byte_strides = 0;
  args.host_buffer_semantics =
      ::pjrt::ConvertToPjRtHostBufferSemantics(host_buffer_semantics);
  args.client = client;
  if (device == nullptr) {
    device = client->addressable_devices[0];
  }
  args.device = device;
  return args;
}

// Caller of this function must delete the object returned when done with it.
// Else it could lead to memory leak.
std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> create_buffer(
    PJRT_Client* client, const PJRT_Api* api, PJRT_Device* device = nullptr) {
  Shape shape = ShapeUtil::MakeShapeWithType<float>({4});
  std::vector<float> float_data(4);
  std::iota(float_data.begin(), float_data.end(), 41.0f);

  PJRT_Client_BufferFromHostBuffer_Args args = CreateBufferFromHostBufferArgs(
      float_data, shape,
      xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, client,
      device);

  PJRT_Error* error = api->PJRT_Client_BufferFromHostBuffer(&args);
  CHECK_EQ(error, nullptr);

  std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> buffer(
      args.buffer, ::pjrt::MakeBufferDeleter(api));

  std::unique_ptr<PJRT_Event, ::pjrt::PJRT_EventDeleter> event(
      args.done_with_host_buffer, ::pjrt::MakeEventDeleter(api));

  return buffer;
}

class TestCApiFactory {
 public:
  void Register(std::function<const PJRT_Api*()> factory) {
    absl::MutexLock lock(&mu_);
    CHECK(!factory_);
    factory_ = std::move(factory);
  }

  std::function<const PJRT_Api*()> Get() const {
    absl::MutexLock lock(&mu_);
    return factory_;
  }

 private:
  mutable absl::Mutex mu_;
  std::function<const PJRT_Api*()> factory_ ABSL_GUARDED_BY(mu_);
};

TestCApiFactory& GetGlobalTestCApiFactory() {
  static auto* const factory = new TestCApiFactory;
  return *factory;
}

const PJRT_Api* GetCApi() { return GetGlobalTestCApiFactory().Get()(); }

}  // namespace

void RegisterTestCApiFactory(std::function<const PJRT_Api*()> factory) {
  GetGlobalTestCApiFactory().Register(std::move(factory));
}

namespace {
class PjrtCApiTest : public ::testing::Test {
 protected:
  const PJRT_Api* api_;
  PJRT_Client* client_;

  void SetUp() override {
    api_ = GetCApi();
    client_ = make_client();
  }

  void TearDown() override { destroy_client(client_); }

  void destroy_client(PJRT_Client* client) {
    PJRT_Client_Destroy_Args destroy_args = PJRT_Client_Destroy_Args{
        .struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE,
        .priv = nullptr,
        .client = client,
    };
    PJRT_Error* error = api_->PJRT_Client_Destroy(&destroy_args);
    CHECK_EQ(error, nullptr);
  }

  PJRT_Client* make_client() {
    PJRT_Client_Create_Args create_args = PJRT_Client_Create_Args{
        .struct_size = PJRT_Client_Create_Args_STRUCT_SIZE,
        .priv = nullptr,
        .client = nullptr,
    };
    PJRT_Error* error = api_->PJRT_Client_Create(&create_args);
    CHECK_EQ(error, nullptr);
    CHECK_NE(create_args.client, nullptr);
    return create_args.client;
  }

  int GetDeviceId(PJRT_Device* device) const {
    PJRT_Device_Id_Args args = PJRT_Device_Id_Args{
        .struct_size = PJRT_Device_Id_Args_STRUCT_SIZE,
        .priv = nullptr,
        .device = device,
        .id = -1,
    };
    PJRT_Error* error = api_->PJRT_Device_Id(&args);
    CHECK_EQ(error, nullptr);
    return args.id;
  }

  bool IsValidDeviceId(PJRT_Device* device) const {
    return GetDeviceId(device) >= 0;
  }

  int GetLocalHardwareId(PJRT_Device* device) const {
    PJRT_Device_LocalHardwareId_Args args = PJRT_Device_LocalHardwareId_Args{
        .struct_size = PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE,
        .priv = nullptr,
        .device = device,
        .local_hardware_id = -1,
    };
    PJRT_Error* error = api_->PJRT_Device_LocalHardwareId(&args);
    CHECK_EQ(error, nullptr);
    return args.local_hardware_id;
  }

  absl::Span<PJRT_Device*> GetClientDevices(PJRT_Client* client) const {
    PJRT_Client_Devices_Args dev_args;
    dev_args.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
    dev_args.priv = nullptr;
    dev_args.client = client_;
    PJRT_Error* error = api_->PJRT_Client_Devices(&dev_args);
    CHECK(error == nullptr);
    return absl::MakeSpan(dev_args.devices, dev_args.num_devices);
  }

  int GetNumDevices(PJRT_Client* client) const {
    return GetClientDevices(client).size();
  }

  absl::Span<PJRT_Device*> GetClientAddressableDevices(
      PJRT_Client* client) const {
    PJRT_Client_AddressableDevices_Args addr_args;
    addr_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    addr_args.priv = nullptr;
    addr_args.client = client_;
    PJRT_Error* error = api_->PJRT_Client_AddressableDevices(&addr_args);
    CHECK(error == nullptr);
    return absl::MakeSpan(addr_args.addressable_devices,
                          addr_args.num_addressable_devices);
  }

  std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter> ToUniquePtr(
      PJRT_Error* error) {
    return std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter>{
        error, ::pjrt::MakeErrorDeleter(api_)};
  }

  std::string BuildSingleDeviceCompileOptionStr() {
    ExecutableBuildOptions build_options;
    build_options.set_device_ordinal(0);
    DeviceAssignment device_assignment(1, 1);
    device_assignment(0, 0) = 0;
    build_options.set_device_assignment(device_assignment);
    CompileOptions options;
    options.executable_build_options = build_options;
    StatusOr<CompileOptionsProto> options_proto = options.ToProto();
    CHECK(options_proto.ok());
    return options_proto->SerializeAsString();
  }

  // Returns a scalar result of execution.
  // supply as e.g. `src_buffer = args.output_lists[0][0];`
  // after calling `api_->PJRT_LoadedExecutable_Execute(&args);`
  StatusOr<float> GetProgramResult(PJRT_Buffer* src_buffer) {
    CHECK(src_buffer != nullptr);
    PJRT_Buffer_ToHostBuffer_Args args{
        .struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE,
        .priv = nullptr,
        .src = src_buffer,
        .dst = nullptr,
        .dst_size = 0,
        .event = nullptr,
    };
    PJRT_Error* error = api_->PJRT_Buffer_ToHostBuffer(&args);
    if (error != nullptr) {
      return ::pjrt::PjrtErrorToStatus(error, api_);
    }
    CHECK_EQ(args.dst_size, sizeof(float))
        << "GetProgramResult assumes scalar float result";

    PJRT_Buffer_OnDeviceTrimmedShape_Args shape_args{
        .struct_size = PJRT_Buffer_OnDeviceTrimmedShape_Args_STRUCT_SIZE,
        .priv = nullptr,
        .buffer = src_buffer,
        .element_type = -1,
        .dimensions = {},
        .dynamic_dimensions = {},
        .has_layout = false,
        .layout = {},
    };
    error = api_->PJRT_Buffer_OnDeviceTrimmedShape(&shape_args);
    if (error != nullptr) {
      return ::pjrt::PjrtErrorToStatus(error, api_);
    }
    CHECK_EQ(shape_args.dimensions.size, 0) << "Buffer is not a scalar!";
    CHECK_EQ(shape_args.element_type, xla::PrimitiveType::F32);

    float value;
    args.dst = &value;
    error = api_->PJRT_Buffer_ToHostBuffer(&args);
    if (error != nullptr) {
      return ::pjrt::PjrtErrorToStatus(error, api_);
    }

    PjRtFuture<Status> transfer_to_host =
        ::pjrt::ConvertCEventToCppFuture(args.event, api_);
    TF_RETURN_IF_ERROR(transfer_to_host.Await());
    return value;
  }

  // Runs the default executable created in PjrtCApiExecutableTest:SetUp and
  // returns its output
  StatusOr<float> RunScalarExecutableAndGetResult(
      PJRT_LoadedExecutable* executable) {
    PJRT_LoadedExecutable_Execute_Args args;
    args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.executable = executable;
    PJRT_ExecuteOptions c_options;
    args.options = &c_options;
    args.options->struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    args.options->launch_id = 0;
    args.num_devices = 1;
    args.num_args = 1;
    auto buffer = create_buffer(client_, api_);
    std::vector<PJRT_Buffer*> argument_list = {buffer.get()};
    std::vector<PJRT_Buffer**> argument_lists{argument_list.data()};
    args.argument_lists = argument_lists.data();
    args.device_complete_events = nullptr;
    args.execute_device = nullptr;

    // Allocates memory for output.
    int num_outputs_per_device = 1;
    std::vector<PJRT_Buffer*> output_list(num_outputs_per_device);
    std::vector<PJRT_Buffer**> output_lists{output_list.data()};
    args.output_lists = output_lists.data();

    PJRT_Error* error = api_->PJRT_LoadedExecutable_Execute(&args);
    if (error != nullptr) {
      return ::pjrt::PjrtErrorToStatus(error, api_);
    }

    PJRT_Buffer* result_buffer = args.output_lists[0][0];
    TF_ASSIGN_OR_RETURN(float result, GetProgramResult(result_buffer));

    // Clean up.
    for (int i = 0; i < args.num_devices; ++i) {
      for (int j = 0; j < num_outputs_per_device; ++j) {
        delete args.output_lists[i][j];
      }
    }
    return result;
  }
};

// ---------------------------------- Client -----------------------------------

TEST_F(PjrtCApiTest, ClientProcessIndex) {
  PJRT_Client_ProcessIndex_Args process_index_args =
      PJRT_Client_ProcessIndex_Args{
          .struct_size = PJRT_Client_ProcessIndex_Args_STRUCT_SIZE,
          .priv = nullptr,
          .client = client_,
          .process_index = -1,
      };
  PJRT_Error* error = api_->PJRT_Client_ProcessIndex(&process_index_args);
  CHECK_EQ(error, nullptr);

  // Single-process test should return 0
  CHECK_EQ(process_index_args.process_index, 0);
}

TEST_F(PjrtCApiTest, ClientDevices) {
  absl::Span<PJRT_Device*> devices = GetClientDevices(client_);

  ASSERT_FALSE(devices.empty());
  for (auto& device : devices) {
    ASSERT_TRUE(this->IsValidDeviceId(device));
  }
}

TEST_F(PjrtCApiTest, ClientAddressableDevices) {
  absl::Span<PJRT_Device*> addressable_devices =
      GetClientAddressableDevices(client_);

  ASSERT_FALSE(addressable_devices.empty());
  for (auto& device : addressable_devices) {
    ASSERT_TRUE(this->IsValidDeviceId(device));
  }

  absl::Span<PJRT_Device*> client_devices = GetClientDevices(client_);
  for (auto& addressable_device : addressable_devices) {
    ASSERT_THAT(client_devices, Contains(addressable_device));
  }
}

TEST_F(PjrtCApiTest, LookupDevice) {
  PJRT_Client_LookupDevice_Args lookup_device_args =
      PJRT_Client_LookupDevice_Args{
          .struct_size = PJRT_Client_LookupDevice_Args_STRUCT_SIZE,
          .priv = nullptr,
          .client = client_,
          .id = 0,
          .device = nullptr,
      };

  PJRT_Error* lookup_device_error =
      api_->PJRT_Client_LookupDevice(&lookup_device_args);

  ASSERT_EQ(lookup_device_error, nullptr);
  int id = GetDeviceId(lookup_device_args.device);
  ASSERT_EQ(id, 0);
}

TEST_F(PjrtCApiTest, LookupAddressableDevice) {
  PJRT_Client_LookupAddressableDevice_Args lookup_addressable_device_args =
      PJRT_Client_LookupAddressableDevice_Args{
          .struct_size = PJRT_Client_LookupAddressableDevice_Args_STRUCT_SIZE,
          .priv = nullptr,
          .client = client_,
          .local_hardware_id = 0,
          .addressable_device = nullptr,
      };

  PJRT_Error* lookup_addressable_device_error =
      api_->PJRT_Client_LookupAddressableDevice(
          &lookup_addressable_device_args);

  ASSERT_EQ(lookup_addressable_device_error, nullptr);
  int local_hardware_id =
      GetLocalHardwareId(lookup_addressable_device_args.addressable_device);
  ASSERT_EQ(local_hardware_id, 0);
}

TEST_F(PjrtCApiTest, GetDefaultDeviceAssignmentNominal) {
  constexpr int kNumReplicas = 2;
  constexpr int kNumPartitions = 1;
  std::vector<int> assignment_buffer(kNumReplicas * kNumPartitions);
  PJRT_Client_DefaultDeviceAssignment_Args args{
      .struct_size = PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE,
      .priv = nullptr,
      .client = client_,
      .num_replicas = kNumReplicas,
      .num_partitions = kNumPartitions,
      .default_assignment_size = assignment_buffer.size(),
      .default_assignment = assignment_buffer.data(),  // in-out
  };
  auto error = ToUniquePtr(api_->PJRT_Client_DefaultDeviceAssignment(&args));
  EXPECT_EQ(error, nullptr) << ::pjrt::GetPjrtErrorMessage(error.get(), api_);
}

TEST_F(PjrtCApiTest, GetDefaultDeviceAssignmentBufferTooSmall) {
  constexpr int kNumReplicas = 4;
  constexpr int kNumPartitions = 2;
  constexpr size_t kBufferSize = 7;
  std::vector<int> assignment_buffer(kBufferSize);
  PJRT_Client_DefaultDeviceAssignment_Args args{
      .struct_size = PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE,
      .priv = nullptr,
      .client = client_,
      .num_replicas = kNumReplicas,
      .num_partitions = kNumPartitions,
      .default_assignment_size = assignment_buffer.size(),
      .default_assignment = assignment_buffer.data(),  // in-out
  };
  auto error = ToUniquePtr(api_->PJRT_Client_DefaultDeviceAssignment(&args));
  ASSERT_NE(error, nullptr);
  xla::Status status = ::pjrt::PjrtErrorToStatus(error.get(), api_);
  EXPECT_EQ(status.code(), tsl::error::FAILED_PRECONDITION);
  EXPECT_EQ(status.error_message(),
            "PJRT_Client_DefaultDeviceAssignment: `default_assignment_size` 7"
            " < `num_replicas * num_partitions`, 4 * 2 = 8");
}

TEST_F(PjrtCApiTest, LookupDeviceNegativeId) {
  PJRT_Client_LookupDevice_Args args = PJRT_Client_LookupDevice_Args{
      .struct_size = PJRT_Client_LookupDevice_Args_STRUCT_SIZE,
      .priv = nullptr,
      .client = client_,
      .id = -1,
      .device = nullptr,
  };
  xla::Status expected =
      xla::Status(tsl::error::INVALID_ARGUMENT,
                  "No matching device found for device_id -1");

  PJRT_Error* error = api_->PJRT_Client_LookupDevice(&args);

  ASSERT_NE(error, nullptr);
  // TODO(b/236710439): check error using C API instead of punching through.
  ASSERT_EQ(error->status, expected);
  delete error;
}

TEST_F(PjrtCApiTest, LookupDeviceOutOfRangeId) {
  int out_of_range_id = GetNumDevices(client_);
  PJRT_Client_LookupDevice_Args args = PJRT_Client_LookupDevice_Args{
      .struct_size = PJRT_Client_LookupDevice_Args_STRUCT_SIZE,
      .priv = nullptr,
      .client = client_,
      .id = out_of_range_id,
      .device = nullptr,
  };
  xla::Status expected = xla::Status(
      tsl::error::INVALID_ARGUMENT,
      absl::StrCat("No matching device found for device_id ", out_of_range_id));

  PJRT_Error* error = api_->PJRT_Client_LookupDevice(&args);

  ASSERT_NE(error, nullptr);
  // TODO(b/236710439): check error using C API instead of punching through.
  ASSERT_EQ(error->status, expected);
  delete error;
}

static constexpr std::string_view kExecutableName = "operation";

std::unique_ptr<PJRT_LoadedExecutable, ::pjrt::PJRT_LoadedExecutableDeleter>
create_executable(const PJRT_Api* c_api, PJRT_Client* client,
                  const XlaComputation& computation) {
  CompileOptions compile_options;
  compile_options.executable_build_options.set_num_replicas(1);
  auto compile_result = client->client->Compile(computation, compile_options);
  CHECK(compile_result.ok());
  CHECK(*compile_result);
  return {new PJRT_LoadedExecutable{std::move(*compile_result), client},
          ::pjrt::MakeLoadedExecutableDeleter(c_api)};
}

XlaComputation CreateAddOneComputation() {
  XlaBuilder builder(std::string{kExecutableName});
  Shape s = ShapeUtil::MakeShape(F32, {});
  auto inp = Parameter(&builder, 0, s, "input");
  auto one = ConstantR0<float>(&builder, 1.0f);
  auto incremented = Add(inp, one);
  return builder.Build(incremented).value();
}

std::unique_ptr<PJRT_LoadedExecutable, ::pjrt::PJRT_LoadedExecutableDeleter>
create_executable(const PJRT_Api* c_api, PJRT_Client* client) {
  return create_executable(c_api, client, CreateAddOneComputation());
}

std::unique_ptr<PJRT_DeviceTopology, ::pjrt::PJRT_DeviceTopologyDeleter>
CreateTopology(const PJRT_Api* c_api) {
  PJRT_DeviceTopology_Create_Args init_args;
  init_args.struct_size = PJRT_DeviceTopology_Create_Args_STRUCT_SIZE;
  init_args.priv = nullptr;
  ::pjrt::LogFatalIfPjrtError(c_api->PJRT_DeviceTopology_Create(&init_args),
                              c_api);
  PJRT_DeviceTopology* c_topology = init_args.topology;
  return std::unique_ptr<PJRT_DeviceTopology,
                         ::pjrt::PJRT_DeviceTopologyDeleter>(
      c_topology, ::pjrt::MakeDeviceTopologyDeleter(c_api));
}

std::unique_ptr<PJRT_Executable, ::pjrt::PJRT_ExecutableDeleter>
create_executable(const PJRT_Api* c_api, PJRT_DeviceTopology* topology,
                  const XlaComputation& computation) {
  PJRT_Compile_Args args;
  args.struct_size = PJRT_Compile_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.client = nullptr;
  args.topology = topology;
  std::string module_str = computation.proto().SerializeAsString();
  std::string format(::pjrt::kHloFormat);

  CompileOptions compile_options;
  compile_options.executable_build_options.set_num_replicas(1);

  std::string options_str = compile_options.ToProto()->SerializeAsString();
  args.compile_options = options_str.c_str();
  args.compile_options_size = options_str.size();

  PJRT_Program program = PJRT_Program{
      .struct_size = PJRT_Program_STRUCT_SIZE,
      .priv = nullptr,
      .code = module_str.data(),
      .code_size = module_str.size(),
      .format = format.c_str(),
      .format_size = format.size(),
  };
  args.program = &program;
  ::pjrt::LogFatalIfPjrtError(c_api->PJRT_Compile(&args), c_api);
  return {args.executable, ::pjrt::MakeExecutableDeleter(c_api)};
}

void destroy_executable(PJRT_LoadedExecutable* executable,
                        const PJRT_Api* api) {
  PJRT_LoadedExecutable_Destroy_Args args{
      .struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE,
      .priv = nullptr,
      .executable = executable,
  };
  PJRT_Error* error = api->PJRT_LoadedExecutable_Destroy(&args);
  CHECK_EQ(error, nullptr);
}

TEST_F(PjrtCApiTest, BufferTransferImmutableUntilTransferCompletes) {
  Shape shape = ShapeUtil::MakeShapeWithType<float>({4});
  std::vector<float> float_data(4);
  std::iota(float_data.begin(), float_data.end(), 41.0f);

  PJRT_Client_BufferFromHostBuffer_Args args = CreateBufferFromHostBufferArgs(
      float_data, shape,
      xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
      client_);

  PJRT_Error* error = api_->PJRT_Client_BufferFromHostBuffer(&args);
  CHECK_EQ(error, nullptr);

  std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> buffer(
      args.buffer, ::pjrt::MakeBufferDeleter(api_));

  std::unique_ptr<PJRT_Event, ::pjrt::PJRT_EventDeleter> event(
      args.done_with_host_buffer, ::pjrt::MakeEventDeleter(api_));

  PJRT_Event_Await_Args await_args;
  await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  await_args.priv = nullptr;
  await_args.event = event.get();
  PJRT_Error* event_error = api_->PJRT_Event_Await(&await_args);
  ASSERT_EQ(event_error, nullptr);
}

TEST_F(PjrtCApiTest, Compile) {
  PJRT_Client_Compile_Args args = PJRT_Client_Compile_Args{
      .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
      .priv = nullptr,
      .client = client_,
  };
  std::string options_str = BuildSingleDeviceCompileOptionStr();
  args.compile_options = options_str.c_str();
  args.compile_options_size = options_str.size();

  std::string format(::pjrt::kMlirFormat);
  std::string program_code{module_add_one};
  PJRT_Program program = PJRT_Program{
      .struct_size = PJRT_Program_STRUCT_SIZE,
      .priv = nullptr,
      .code = program_code.data(),
      .code_size = program_code.length(),
      .format = format.c_str(),
      .format_size = format.size(),
  };
  args.program = &program;

  PJRT_Error* error = api_->PJRT_Client_Compile(&args);
  ::pjrt::LogFatalIfPjrtError(error, api_);

  ASSERT_EQ(error, nullptr);
  destroy_executable(args.executable, api_);
}

TEST_F(PjrtCApiTest, CompileXlaComputation) {
  PJRT_Client_Compile_Args args = PJRT_Client_Compile_Args{
      .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
      .priv = nullptr,
      .client = client_,
  };
  DeviceAssignment device_assignment(1, 1);
  device_assignment(0, 0) = 0;
  DeviceAssignmentProto proto;
  ASSERT_TRUE(device_assignment.Serialize(&proto).ok());
  std::string device_assignment_str = proto.SerializeAsString();
  std::string options_str = BuildSingleDeviceCompileOptionStr();
  args.compile_options = options_str.c_str();
  args.compile_options_size = options_str.size();

  StatusOr<std::unique_ptr<HloModule>> hlo_module =
      xla::ParseAndReturnUnverifiedModule(hlo_string);
  ASSERT_EQ(hlo_module.ok(), true);
  std::string module_str = hlo_module->get()->ToProto().SerializeAsString();

  std::string format(::pjrt::kHloFormat);
  PJRT_Program program = PJRT_Program{
      .struct_size = PJRT_Program_STRUCT_SIZE,
      .priv = nullptr,
      .code = module_str.data(),
      .code_size = module_str.size(),
      .format = format.c_str(),
      .format_size = format.size(),
  };
  args.program = &program;

  PJRT_Error* error = api_->PJRT_Client_Compile(&args);
  ::pjrt::LogFatalIfPjrtError(error, api_);

  ASSERT_EQ(error, nullptr);
  destroy_executable(args.executable, api_);
}

TEST_F(PjrtCApiTest, CompileInvalidOption) {
  PJRT_Client_Compile_Args args = PJRT_Client_Compile_Args{
      .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
      .priv = nullptr,
      .client = client_,
  };
  std::string options_str = "invalid compile options";
  args.compile_options = options_str.c_str();
  args.compile_options_size = options_str.size();

  std::string format(::pjrt::kMlirFormat);
  std::string program_code{module_add_one};
  PJRT_Program program = PJRT_Program{
      .struct_size = PJRT_Program_STRUCT_SIZE,
      .priv = nullptr,
      .code = program_code.data(),
      .code_size = program_code.length(),
      .format = format.c_str(),
      .format_size = format.size(),
  };
  args.program = &program;

  PJRT_Error* error = api_->PJRT_Client_Compile(&args);

  xla::Status status = ::pjrt::PjrtErrorToStatus(error, api_);
  EXPECT_EQ(status.code(), tsl::error::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(),
            "PJRT_Client_Compile: failed to deserialize CompileOptionsProto");
  destroy_executable(args.executable, api_);
  ::pjrt::MakeErrorDeleter(api_)(error);
}

TEST_F(PjrtCApiTest, CompileInvalidProgramFormat) {
  PJRT_Client_Compile_Args args = PJRT_Client_Compile_Args{
      .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
      .priv = nullptr,
      .client = client_,
  };
  DeviceAssignment device_assignment(1, 1);
  device_assignment(0, 0) = 0;
  DeviceAssignmentProto proto;
  ASSERT_TRUE(device_assignment.Serialize(&proto).ok());
  std::string device_assignment_str = proto.SerializeAsString();
  std::string options_str = BuildSingleDeviceCompileOptionStr();
  args.compile_options = options_str.c_str();
  args.compile_options_size = options_str.size();

  std::string format("invalid");
  PJRT_Program program = PJRT_Program{
      .struct_size = PJRT_Program_STRUCT_SIZE,
      .priv = nullptr,
      .code = nullptr,
      .code_size = 0,
      .format = format.c_str(),
      .format_size = format.size(),
  };
  args.program = &program;

  PJRT_Error* error = api_->PJRT_Client_Compile(&args);
  xla::Status status = ::pjrt::PjrtErrorToStatus(error, api_);
  EXPECT_EQ(status.code(), tsl::error::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(), "Unknown program format 'invalid'.");
  destroy_executable(args.executable, api_);
  ::pjrt::MakeErrorDeleter(api_)(error);
}

// --------------------------------- Devices -----------------------------------

TEST_F(PjrtCApiTest, DeviceId) {
  auto* device = GetClientDevices(client_)[0];

  int id = GetDeviceId(device);

  CHECK_EQ(id, 0);
}

TEST_F(PjrtCApiTest, DeviceProcessIndex) {
  PJRT_Device_ProcessIndex_Args args = PJRT_Device_ProcessIndex_Args{
      .struct_size = PJRT_Device_ProcessIndex_Args_STRUCT_SIZE,
      .priv = nullptr,
      .device = GetClientDevices(client_)[0],
      .process_index = -1,
  };
  PJRT_Error* error = api_->PJRT_Device_ProcessIndex(&args);
  ASSERT_EQ(error, nullptr);
  // For single process, it should match client process index
  CHECK_EQ(args.process_index, 0);
}

TEST_F(PjrtCApiTest, DeviceIsAddressable) {
  PJRT_Device_IsAddressable_Args args = PJRT_Device_IsAddressable_Args{
      .struct_size = PJRT_Device_IsAddressable_Args_STRUCT_SIZE,
      .priv = nullptr,
      .device = GetClientDevices(client_)[0],
      .is_addressable = false,
  };
  PJRT_Error* error = api_->PJRT_Device_IsAddressable(&args);
  ASSERT_EQ(error, nullptr);
  // All devices are addressable in single-process test
  CHECK_EQ(args.is_addressable, true);
}

TEST_F(PjrtCApiTest, DeviceAttributes) {
  auto devices = GetClientDevices(client_);
  for (const auto& device : devices) {
    auto attributes = device->device->Attributes();

    PJRT_Device_Attributes_Args args = PJRT_Device_Attributes_Args{
        .struct_size = PJRT_Device_Attributes_Args_STRUCT_SIZE,
        .priv = nullptr,
        .device = device,
    };

    PJRT_Error* error = api_->PJRT_Device_Attributes(&args);
    ASSERT_EQ(error, nullptr);
    ASSERT_EQ(args.num_attributes, attributes.size());

    for (int i = 0; i < args.num_attributes; ++i) {
      const auto& attribute = args.attributes[i];
      ASSERT_EQ(attribute.struct_size, PJRT_NamedValue_STRUCT_SIZE);
      ASSERT_EQ(attribute.priv, nullptr);
      std::string attribute_name(attribute.name, attribute.name_size);
      ASSERT_TRUE(attributes.contains(attribute_name));
      switch (attribute.type) {
        case PJRT_NamedValue::PJRT_NamedValue_kString: {
          std::string string_value(attribute.string_value);
          ASSERT_EQ(std::get<std::string>(attributes[attribute_name]),
                    string_value);
          break;
        }
        case PJRT_NamedValue::PJRT_NamedValue_kInt64: {
          ASSERT_EQ(std::get<int64_t>(attributes[attribute_name]),
                    attribute.int64_value);
          break;
        }
        case PJRT_NamedValue::PJRT_NamedValue_kInt64List: {
          const int64_t* value_ptr = attribute.int64_array_value;
          std::vector<int64_t> array_value(value_ptr,
                                           value_ptr + attribute.value_size);
          ASSERT_EQ(std::get<std::vector<int64_t>>(attributes[attribute_name]),
                    array_value);
          break;
        }
          // Do not allow other types (such as
          // PJRT_NamedValue::PJRT_NamedValue_kFloat) since device attributes
          // currently should not return other types.
        default: {
          // should never get here.
          FAIL() << "attribute value type " << attribute.type
                 << " invalid; should have been string, int64, or int64_list. "
                    "This should never occur.";
        }
      }
    }
    ASSERT_EQ(error, nullptr);
  }
}

TEST_F(PjrtCApiTest, DeviceLocalHardwareId) {
  PJRT_Device_LocalHardwareId_Args args = PJRT_Device_LocalHardwareId_Args{
      .struct_size = PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE,
      .priv = nullptr,
      .device = GetClientDevices(client_)[0],
      .local_hardware_id = -1,
  };
  PJRT_Error* error = api_->PJRT_Device_LocalHardwareId(&args);
  ASSERT_EQ(error, nullptr);
  CHECK_EQ(args.local_hardware_id, 0);
}

// ------------------------------- Executables ---------------------------------

std::unique_ptr<PJRT_Executable, ::pjrt::PJRT_ExecutableDeleter> GetExecutable(
    PJRT_LoadedExecutable* loaded_executable, const PJRT_Api* api) {
  PJRT_LoadedExecutable_GetExecutable_Args args;
  args.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.loaded_executable = loaded_executable;
  args.executable = nullptr;
  ::pjrt::LogFatalIfPjrtError(api->PJRT_LoadedExecutable_GetExecutable(&args),
                              api);
  return {args.executable, ::pjrt::MakeExecutableDeleter(api)};
}

class PjrtCApiExecutableTest : public PjrtCApiTest {
 protected:
  std::unique_ptr<PJRT_LoadedExecutable, ::pjrt::PJRT_LoadedExecutableDeleter>
      executable_;

  void SetUp() override {
    PjrtCApiTest::SetUp();
    executable_ = create_executable(api_, client_);
  }

  void TearDown() override {
    executable_.reset();
    PjrtCApiTest::TearDown();
  }

  absl::flat_hash_map<std::string, PjRtValueType>
  CreateMapFromGetCostAnalysisOutput(
      const PJRT_LoadedExecutable_GetCostAnalysis_Args& args) {
    absl::flat_hash_map<std::string, PjRtValueType> output_map;

    for (size_t i = 0; i < args.num_properties; ++i) {
      LOG(INFO) << "Cost property '" << args.properties[i].name
                << "' type: " << args.properties[i].type
                << " elements: " << args.properties[i].value_size;
      switch (args.properties[i].type) {
        case PJRT_NamedValue::PJRT_NamedValue_kFloat:
          output_map[args.properties[i].name] = args.properties[i].float_value;
          LOG(INFO) << "Cost property '" << args.properties[i].name
                    << "' value: " << args.properties[i].float_value;
          break;
        case PJRT_NamedValue::PJRT_NamedValue_kInt64:
          output_map[args.properties[i].name] = args.properties[i].int64_value;
          LOG(INFO) << "Cost property '" << args.properties[i].name
                    << "' value: " << args.properties[i].int64_value;
          break;
        case PJRT_NamedValue::PJRT_NamedValue_kInt64List: {
          PjRtValueType& output_value = output_map[args.properties[i].name];
          std::vector<int64_t>& output_int64_list =
              std::get<std::vector<int64_t>>(output_value);
          for (size_t j = 0; j < args.properties[i].value_size; ++j) {
            output_int64_list.push_back(
                args.properties[i].int64_array_value[j]);
            LOG(INFO) << "Cost property '" << args.properties[i].name
                      << "' value[" << j
                      << "]: " << args.properties[i].int64_array_value[j];
          }
          break;
        }
        case PJRT_NamedValue::PJRT_NamedValue_kString:
          output_map[args.properties[i].name] = args.properties[i].string_value;
          LOG(INFO) << "Cost property '" << args.properties[i].name
                    << "' value: '" << args.properties[i].string_value << "'";
          break;
      }
    }

    return output_map;
  }

  void VerifyReturnedFuture(PJRT_Event* device_complete_event) {
    PJRT_Event_Await_Args await_args;
    await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    await_args.priv = nullptr;
    await_args.event = device_complete_event;
    ::pjrt::LogFatalIfPjrtError(api_->PJRT_Event_Await(&await_args), api_);

    PJRT_Event_IsReady_Args ready_args;
    ready_args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
    ready_args.priv = nullptr;
    ready_args.event = device_complete_event;
    ::pjrt::LogFatalIfPjrtError(api_->PJRT_Event_IsReady(&ready_args), api_);
    EXPECT_TRUE(ready_args.is_ready);

    PJRT_Event_Destroy_Args destroy_args;
    destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    destroy_args.priv = nullptr;
    destroy_args.event = device_complete_event;
    ::pjrt::LogFatalIfPjrtError(api_->PJRT_Event_Destroy(&destroy_args), api_);
  }

  void TestExecuteSharded(bool create_device_completion_event) {
    PJRT_Client_Compile_Args args_compile = PJRT_Client_Compile_Args{
        .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
        .priv = nullptr,
        .client = client_,
    };
    // creating a multi-device executable
    DeviceAssignment device_assignment(2, 1);
    device_assignment(0, 0) = 0;
    device_assignment(1, 0) = 1;
    ExecutableBuildOptions build_options;
    build_options.set_device_ordinal(0);
    build_options.set_num_replicas(2);
    build_options.set_device_assignment(device_assignment);
    CompileOptions options;
    options.executable_build_options = build_options;
    CompileOptionsProto options_proto = options.ToProto().value();
    std::string options_str = options_proto.SerializeAsString();
    args_compile.compile_options = options_str.c_str();
    args_compile.compile_options_size = options_str.size();

    std::string format(::pjrt::kMlirFormat);
    std::string program_code{module_add_one};
    PJRT_Program program = PJRT_Program{
        .struct_size = PJRT_Program_STRUCT_SIZE,
        .priv = nullptr,
        .code = program_code.data(),
        .code_size = program_code.length(),
        .format = format.c_str(),
        .format_size = format.size(),
    };
    args_compile.program = &program;

    PJRT_Error* error_compile = api_->PJRT_Client_Compile(&args_compile);
    ::pjrt::LogFatalIfPjrtError(error_compile, api_);
    PJRT_LoadedExecutable_Execute_Args args_execute;
    args_execute.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    args_execute.priv = nullptr;
    args_execute.executable = args_compile.executable;
    PJRT_ExecuteOptions c_options;
    args_execute.options = &c_options;
    args_execute.options->struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    args_execute.options->launch_id = 0;
    args_execute.num_devices = 1;
    args_execute.num_args = 1;
    PJRT_Event* returned_future;
    if (create_device_completion_event) {
      args_execute.device_complete_events = &returned_future;
    } else {
      args_execute.device_complete_events = nullptr;
    }

    // Allocates memory for output.
    int num_outputs_per_device = 1;
    std::vector<PJRT_Buffer*> output_list(num_outputs_per_device);
    std::vector<PJRT_Buffer**> output_lists{output_list.data()};
    args_execute.output_lists = output_lists.data();

    auto devices_to_execute = GetClientAddressableDevices(client_);

    // executing on both the client addressable devices
    for (int device_id = 0; device_id <= 1; ++device_id) {
      auto buffer =
          create_buffer(client_, api_, client_->addressable_devices[device_id]);
      std::vector<PJRT_Buffer*> argument_list{buffer.get()};
      std::vector<PJRT_Buffer**> argument_lists{argument_list.data()};
      args_execute.argument_lists = argument_lists.data();
      args_execute.execute_device = devices_to_execute[device_id];

      PJRT_Error* error_execute =
          api_->PJRT_LoadedExecutable_Execute(&args_execute);
      ASSERT_EQ(error_execute, nullptr);

      PJRT_Buffer* result_buffer = args_execute.output_lists[0][0];
      TF_ASSERT_OK_AND_ASSIGN(float result, GetProgramResult(result_buffer));
      // What the executable does is to add one to the input. The input buffer
      // is 41, therefore the expected output is 42.
      EXPECT_EQ(result, 42);
      if (create_device_completion_event) {
        VerifyReturnedFuture(args_execute.device_complete_events[0]);
      }

      // Clean up.
      for (int i = 0; i < args_execute.num_devices; ++i) {
        for (int j = 0; j < num_outputs_per_device; ++j) {
          delete args_execute.output_lists[i][j];
        }
      }
    }

    destroy_executable(args_compile.executable, api_);
  }

  void TestExecutePortable(bool create_device_completion_event) {
    PJRT_Client_Compile_Args args_compile = PJRT_Client_Compile_Args{
        .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
        .priv = nullptr,
        .client = client_,
    };
    ExecutableBuildOptions build_options;
    build_options.set_device_ordinal(0);
    build_options.set_num_replicas(1);
    CompileOptions options;
    options.executable_build_options = build_options;
    options.compile_portable_executable = true;
    CompileOptionsProto options_proto = options.ToProto().value();
    std::string options_str = options_proto.SerializeAsString();
    args_compile.compile_options = options_str.c_str();
    args_compile.compile_options_size = options_str.size();

    std::string format(::pjrt::kMlirFormat);
    std::string program_code{module_add_one};
    PJRT_Program program = PJRT_Program{
        .struct_size = PJRT_Program_STRUCT_SIZE,
        .priv = nullptr,
        .code = program_code.data(),
        .code_size = program_code.length(),
        .format = format.c_str(),
        .format_size = format.size(),
    };
    args_compile.program = &program;

    PJRT_Error* error_compile = api_->PJRT_Client_Compile(&args_compile);
    ::pjrt::LogFatalIfPjrtError(error_compile, api_);
    PJRT_LoadedExecutable_Execute_Args args_execute;
    args_execute.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    args_execute.priv = nullptr;
    args_execute.executable = args_compile.executable;
    PJRT_ExecuteOptions c_options;
    args_execute.options = &c_options;
    args_execute.options->struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    args_execute.options->launch_id = 0;
    args_execute.num_devices = 1;
    args_execute.num_args = 1;
    PJRT_Event* returned_future;
    if (create_device_completion_event) {
      args_execute.device_complete_events = &returned_future;
    } else {
      args_execute.device_complete_events = nullptr;
    }

    // Allocates memory for output.
    int num_outputs_per_device = 1;
    std::vector<PJRT_Buffer*> output_list(num_outputs_per_device);
    std::vector<PJRT_Buffer**> output_lists{output_list.data()};
    args_execute.output_lists = output_lists.data();

    auto devices_to_execute = GetClientAddressableDevices(client_);
    int device_index;
    // Use a non-default device to test ExecutePortable if there are more than
    // one devices.
    if (devices_to_execute.size() > 1) {
      device_index = 1;
    } else {
      device_index = 0;
    }
    auto buffer = create_buffer(client_, api_,
                                client_->addressable_devices[device_index]);
    std::vector<PJRT_Buffer*> argument_list{buffer.get()};
    std::vector<PJRT_Buffer**> argument_lists{argument_list.data()};
    args_execute.argument_lists = argument_lists.data();
    args_execute.execute_device = devices_to_execute[device_index];

    PJRT_Error* error_execute =
        api_->PJRT_LoadedExecutable_Execute(&args_execute);
    ASSERT_EQ(error_execute, nullptr);

    PJRT_Buffer* result_buffer = args_execute.output_lists[0][0];
    TF_ASSERT_OK_AND_ASSIGN(float result, GetProgramResult(result_buffer));
    // What the executable does is to add one to the input. The input buffer is
    // 41, therefore the expected output is 42.
    EXPECT_EQ(result, 42);
    if (create_device_completion_event) {
      VerifyReturnedFuture(args_execute.device_complete_events[0]);
    }

    // Clean up.
    for (int i = 0; i < args_execute.num_devices; ++i) {
      for (int j = 0; j < num_outputs_per_device; ++j) {
        delete args_execute.output_lists[i][j];
      }
    }
    destroy_executable(args_compile.executable, api_);
  }
};

TEST_F(PjrtCApiExecutableTest, ExecutableName) {
  PJRT_Executable_Name_Args args;
  args.struct_size = PJRT_Executable_Name_Args_STRUCT_SIZE;
  args.priv = nullptr;
  auto executable = GetExecutable(executable_.get(), api_);
  args.executable = executable.get();
  PJRT_Error* error = api_->PJRT_Executable_Name(&args);
  ASSERT_EQ(error, nullptr);
  absl::string_view executable_name(args.executable_name,
                                    args.executable_name_size);

  using ::testing::StartsWith;
  ASSERT_THAT(executable_name, StartsWith(kExecutableName));
}

TEST_F(PjrtCApiExecutableTest, ExecutableNumReplicas) {
  PJRT_Executable_NumReplicas_Args args;
  args.struct_size = PJRT_Executable_NumReplicas_Args_STRUCT_SIZE;
  args.priv = nullptr;
  auto executable = GetExecutable(executable_.get(), api_);
  args.executable = executable.get();
  PJRT_Error* error = api_->PJRT_Executable_NumReplicas(&args);

  ASSERT_EQ(error, nullptr);
  ASSERT_EQ(args.num_replicas, 1);
}

TEST_F(PjrtCApiExecutableTest, ExecutableNumPartitions) {
  PJRT_Executable_NumPartitions_Args args;
  args.struct_size = PJRT_Executable_NumPartitions_Args_STRUCT_SIZE;
  args.priv = nullptr;
  auto executable = GetExecutable(executable_.get(), api_);
  args.executable = executable.get();
  PJRT_Error* error = api_->PJRT_Executable_NumPartitions(&args);

  ASSERT_EQ(error, nullptr);
  ASSERT_EQ(args.num_partitions, 1);
}

TEST_F(PjrtCApiExecutableTest, AddressableDevices) {
  PJRT_LoadedExecutable_AddressableDevices_Args args;
  args.struct_size = PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.executable = executable_.get();
  PJRT_Error* error = api_->PJRT_LoadedExecutable_AddressableDevices(&args);
  EXPECT_EQ(error, nullptr)
      << "Expected no error, but got status: " << error->status;
  absl::Span<PJRT_Device*> addressable_devices =
      absl::MakeSpan(args.addressable_devices, args.num_addressable_devices);

  EXPECT_EQ(addressable_devices.size(),
            executable_->get()->addressable_devices().size());
  ASSERT_FALSE(addressable_devices.empty());
  for (auto& device : addressable_devices) {
    ASSERT_TRUE(this->IsValidDeviceId(device));
  }

  absl::Span<PJRT_Device*> client_devices =
      GetClientAddressableDevices(client_);
  for (auto& addressable_device : addressable_devices) {
    ASSERT_THAT(client_devices, Contains(addressable_device));
  }
}

TEST_F(PjrtCApiExecutableTest, OptimizedProgram) {
  PJRT_Executable_OptimizedProgram_Args args;
  args.struct_size = PJRT_Executable_OptimizedProgram_Args_STRUCT_SIZE;
  args.priv = nullptr;
  auto executable = GetExecutable(executable_.get(), api_);
  args.executable = executable.get();
  auto program = std::make_unique<PJRT_Program>();
  program->struct_size = PJRT_Program_STRUCT_SIZE;
  program->priv = nullptr;
  program->code = nullptr;
  absl::string_view program_format = "hlo_with_config";
  program->format = program_format.data();
  program->format_size = program_format.length();
  args.program = program.get();

  // The first call to `PJRT_Executable_OptimizedProgram` populates
  // `program->code_size`, telling us how large a string to allocate
  auto error = ToUniquePtr(api_->PJRT_Executable_OptimizedProgram(&args));
  EXPECT_EQ(error, nullptr);
  constexpr size_t TWO_GIBIBYTES = 2ull * 1024 * 1024 * 1024;
  ASSERT_LT(program->code_size, TWO_GIBIBYTES);
  std::string code(args.program->code_size, 'a');
  program->code = code.data();

  // The second call to `PJRT_Executable_OptimizedProgram` assigns the
  // serialized program to `program->code` (and thus `code`).
  error = ToUniquePtr(api_->PJRT_Executable_OptimizedProgram(&args));
  EXPECT_EQ(error, nullptr) << ::pjrt::GetPjrtErrorMessage(error.get(), api_);

  // Use the PJRT C++ API to create an expected proto
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::shared_ptr<HloModule>> cpp_modules,
                          executable_->executable->GetHloModules());
  TF_ASSERT_OK_AND_ASSIGN(HloModuleProtoWithConfig expected_proto,
                          cpp_modules[0]->ToProtoWithConfig());

  HloModuleProtoWithConfig deserialized_proto;
  ASSERT_TRUE(deserialized_proto.ParseFromString(code));

  // Make sure the protos output by the C++ and C APIs are equivalent
  google::protobuf::util::MessageDifferencer diff;
  diff.set_message_field_comparison(
      google::protobuf::util::MessageDifferencer::EQUIVALENT);
  EXPECT_TRUE(diff.Equals(expected_proto, deserialized_proto));
}

TEST_F(PjrtCApiExecutableTest, ExecutableDeletion) {
  PJRT_LoadedExecutable_IsDeleted_Args is_deleted_args;
  is_deleted_args.struct_size =
      PJRT_LoadedExecutable_IsDeleted_Args_STRUCT_SIZE;
  is_deleted_args.priv = nullptr;
  is_deleted_args.executable = executable_.get();
  PJRT_Error* is_deleted_error =
      api_->PJRT_LoadedExecutable_IsDeleted(&is_deleted_args);
  ASSERT_EQ(is_deleted_error, nullptr);
  ASSERT_FALSE(is_deleted_args.is_deleted);

  PJRT_LoadedExecutable_Delete_Args delete_args;
  delete_args.struct_size = PJRT_LoadedExecutable_Delete_Args_STRUCT_SIZE;
  delete_args.priv = nullptr;
  delete_args.executable = executable_.get();
  PJRT_Error* delete_error = api_->PJRT_LoadedExecutable_Delete(&delete_args);
  ASSERT_EQ(delete_error, nullptr);

  is_deleted_error = api_->PJRT_LoadedExecutable_IsDeleted(&is_deleted_args);
  ASSERT_EQ(is_deleted_error, nullptr);
  ASSERT_TRUE(is_deleted_args.is_deleted);
}

TEST_F(PjrtCApiExecutableTest, ExecuteInputArgumentSizeExceedNumDevice) {
  PJRT_LoadedExecutable_Execute_Args args;
  args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.executable = executable_.get();
  std::unique_ptr<PJRT_ExecuteOptions> c_options(new PJRT_ExecuteOptions);
  args.options = c_options.get();
  args.options->struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
  args.options->launch_id = 0;
  // The number of addressable devices of executable_ is 1.
  args.num_devices = 2;
  args.num_args = 1;
  args.output_lists = nullptr;
  args.device_complete_events = nullptr;
  auto buffer_1 = create_buffer(client_, api_);
  auto buffer_2 = create_buffer(client_, api_);
  std::vector<PJRT_Buffer*> argument_list_1{buffer_1.get()};
  std::vector<PJRT_Buffer*> argument_list_2{buffer_2.get()};
  std::vector<PJRT_Buffer**> argument_lists{argument_list_1.data(),
                                            argument_list_2.data()};
  args.argument_lists = argument_lists.data();
  args.execute_device = nullptr;

  PJRT_Error* error = api_->PJRT_LoadedExecutable_Execute(&args);

  xla::Status status = ::pjrt::PjrtErrorToStatus(error, api_);
  EXPECT_EQ(status.code(), tsl::error::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(),
            "Attempted to execute with 2 argument lists when local device "
            "count is 1 (total replica count: 1, partition count: 1)");

  // Clean up.
  ::pjrt::MakeErrorDeleter(api_)(error);
}

TEST_F(PjrtCApiExecutableTest, NumOutputsSingle) {
  PJRT_Executable_NumOutputs_Args args;
  args.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
  args.priv = nullptr;
  auto executable = GetExecutable(executable_.get(), api_);
  args.executable = executable.get();
  PJRT_Error* error = api_->PJRT_Executable_NumOutputs(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.num_outputs, 1);
}

TEST_F(PjrtCApiExecutableTest, NumOutputsTuple) {
  XlaBuilder builder(std::string{kExecutableName});
  Shape s = ShapeUtil::MakeShape(F32, {});
  auto inp = Parameter(&builder, 0, s, "input");
  auto one = ConstantR0<float>(&builder, 1.0f);
  auto incremented = Add(inp, one);
  std::vector v = {incremented, inp};
  auto tuple = Tuple(&builder, v);
  auto computation = builder.Build(tuple).value();
  auto pjrt_executable = create_executable(api_, client_, computation);
  PJRT_Executable_NumOutputs_Args args;
  args.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
  args.priv = nullptr;
  auto base_executable = GetExecutable(pjrt_executable.get(), api_);
  args.executable = base_executable.get();
  PJRT_Error* error = api_->PJRT_Executable_NumOutputs(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.num_outputs, 2);
}

TEST_F(PjrtCApiExecutableTest, SizeOfGeneratedCodeInBytes) {
  // Call the function directly to get a reference size, check that it's not
  // zero.
  int64_t direct_call_size =
      executable_->executable->SizeOfGeneratedCodeInBytes();
  ASSERT_NE(direct_call_size, 0);

  // Call the function through PJRT C API interface, check that it's not zero.
  auto executable = GetExecutable(executable_.get(), api_);
  PJRT_Executable_SizeOfGeneratedCodeInBytes_Args args{
      .struct_size =
          PJRT_Executable_SizeOfGeneratedCodeInBytes_Args_STRUCT_SIZE,
      .priv = nullptr,
      .executable = executable.get(),
  };
  PJRT_Error* error = api_->PJRT_Executable_SizeOfGeneratedCodeInBytes(&args);
  ::pjrt::LogFatalIfPjrtError(error, api_);
  ASSERT_EQ(error, nullptr);
  ASSERT_NE(args.size_in_bytes, 0);

  // Confirm that size in bytes returned from both calls are the same.
  ASSERT_EQ(direct_call_size, args.size_in_bytes);
}

TEST_F(PjrtCApiExecutableTest, GetCostAnalysis) {
  // Call GetCostAnalysis directly
  auto program_cost_properties = executable_->get()->GetCostAnalysis();
  ASSERT_TRUE(program_cost_properties.ok());
  ASSERT_GT(program_cost_properties.value().size(), 0);

  // Call PJRT C API
  PJRT_LoadedExecutable_GetCostAnalysis_Args args{
      .struct_size = PJRT_LoadedExecutable_GetCostAnalysis_Args_STRUCT_SIZE,
      .priv = nullptr,
      .executable = executable_.get(),
      .num_properties = 0,
      .properties = nullptr};
  PJRT_Error* error = api_->PJRT_LoadedExecutable_GetCostAnalysis(&args);
  ::pjrt::LogFatalIfPjrtError(error, api_);
  ASSERT_EQ(error, nullptr);
  LOG(INFO) << "PJRT_LoadedExecutable_GetCostAnalysis returned "
            << args.num_properties << " properties.";
  ASSERT_GT(args.num_properties, 0);

  // Verify results from local call and C API are the same
  auto output_map = CreateMapFromGetCostAnalysisOutput(args);
  ASSERT_EQ(program_cost_properties.value(), output_map);

  // Call PJRT C API again (which returns cached value)
  // to confirm results are the same
  PJRT_LoadedExecutable_GetCostAnalysis_Args second_call_args{
      .struct_size = PJRT_LoadedExecutable_GetCostAnalysis_Args_STRUCT_SIZE,
      .priv = nullptr,
      .executable = executable_.get(),
      .num_properties = 0,
      .properties = nullptr};
  error = api_->PJRT_LoadedExecutable_GetCostAnalysis(&second_call_args);
  ::pjrt::LogFatalIfPjrtError(error, api_);
  ASSERT_EQ(error, nullptr);
  LOG(INFO) << "Second PJRT_LoadedExecutable_GetCostAnalysis call returned "
            << args.num_properties << " properties.";
  ASSERT_GT(args.num_properties, 0);

  auto second_call_output_map = CreateMapFromGetCostAnalysisOutput(args);
  ASSERT_EQ(program_cost_properties.value(), second_call_output_map);
}

// ---------------------------------- Buffers ----------------------------------

class PjrtCApiBufferTest : public PjrtCApiTest {
 protected:
  void SetUp() override {
    PjrtCApiTest::SetUp();
    buffer_ = create_buffer(client_, api_);
  }

  void TearDown() override {
    // buffer_ must be destroyed before the client is destroyed or else the
    // unique_ptr for buffer_ will go out of scope causing heap-use-after-free
    // error.
    buffer_.reset(nullptr);
    PjrtCApiTest::TearDown();
  }

  std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> buffer_;
};

TEST_F(PjrtCApiBufferTest, IsDeleted) {
  PJRT_Buffer_IsDeleted_Args is_deleted_args;
  is_deleted_args.struct_size = PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE;
  is_deleted_args.priv = nullptr;
  is_deleted_args.buffer = buffer_.get();
  PJRT_Error* is_deleted_error = api_->PJRT_Buffer_IsDeleted(&is_deleted_args);
  ASSERT_EQ(is_deleted_error, nullptr);
  ASSERT_FALSE(is_deleted_args.is_deleted);

  PJRT_Buffer_Delete_Args delete_args;
  delete_args.struct_size = PJRT_Buffer_Delete_Args_STRUCT_SIZE;
  delete_args.priv = nullptr;
  delete_args.buffer = buffer_.get();
  PJRT_Error* delete_error = api_->PJRT_Buffer_Delete(&delete_args);
  ASSERT_EQ(delete_error, nullptr);

  is_deleted_error = api_->PJRT_Buffer_IsDeleted(&is_deleted_args);
  ASSERT_EQ(is_deleted_error, nullptr);
  ASSERT_TRUE(is_deleted_args.is_deleted);
}

TEST_F(PjrtCApiBufferTest, GetOnDeviceSizeInBytes) {
  PJRT_Buffer_OnDeviceSizeInBytes_Args args;
  args.struct_size = PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.buffer = buffer_.get();
  PJRT_Error* on_device_size_bytes_error =
      api_->PJRT_Buffer_OnDeviceSizeInBytes(&args);

  ASSERT_EQ(on_device_size_bytes_error, nullptr);
  ASSERT_GT(args.on_device_size_in_bytes, 0);
}

TEST_F(PjrtCApiBufferTest, IsOnCpu) {
  PJRT_Buffer_IsOnCpu_Args args;
  args.struct_size = PJRT_Buffer_IsOnCpu_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.buffer = buffer_.get();
  PJRT_Error* error = api_->PJRT_Buffer_IsOnCpu(&args);
  EXPECT_EQ(error, nullptr);
  EXPECT_FALSE(args.is_on_cpu);
}

TEST_F(PjrtCApiBufferTest, Device) {
  PJRT_Buffer_Device_Args args;
  args.struct_size = PJRT_Buffer_Device_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.buffer = buffer_.get();

  // The returned device is addressable.
  std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter> error(
      api_->PJRT_Buffer_Device(&args), ::pjrt::MakeErrorDeleter(api_));
  EXPECT_EQ(error, nullptr);
  EXPECT_EQ(args.device->device,
            GetClientAddressableDevices(client_)[0]->device);
}

TEST_F(PjrtCApiBufferTest, ReadyEvent) {
  PJRT_Buffer_ReadyEvent_Args get_event_args;
  get_event_args.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
  get_event_args.priv = nullptr;
  get_event_args.buffer = buffer_.get();
  auto error = ToUniquePtr(api_->PJRT_Buffer_ReadyEvent(&get_event_args));
  ASSERT_EQ(error, nullptr);

  PJRT_Event* event = get_event_args.event;
  ASSERT_NE(event, nullptr);

  // Wait for `buffer_`'s data transfer to complete (if it hasn't already)
  PJRT_Event_Await_Args await_args;
  await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  await_args.priv = nullptr;
  await_args.event = event;
  error.reset(api_->PJRT_Event_Await(&await_args));
  ASSERT_EQ(error, nullptr);

  // Must be ready when `PJRT_Event_Await` completes
  PJRT_Event_IsReady_Args ready_args;
  ready_args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
  ready_args.priv = nullptr;
  ready_args.event = event;
  error.reset(api_->PJRT_Event_IsReady(&ready_args));
  ASSERT_EQ(error, nullptr);
  EXPECT_TRUE(ready_args.is_ready);

  // Clean up
  PJRT_Event_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  destroy_args.priv = nullptr;
  destroy_args.event = event;
  error.reset(api_->PJRT_Event_Destroy(&destroy_args));
  EXPECT_EQ(error, nullptr);
}

TEST_F(PjrtCApiBufferTest, UnsafeBufferPointer) {
  // Call the function directly to get a reference pointer value, check that the
  // result is not zero (nullptr).
  xla::PjRtBuffer* xla_pjrt_buffer = buffer_->buffer.get();
  ASSERT_NE(xla_pjrt_buffer, nullptr);
  StatusOr<std::uintptr_t> local_buffer_pointer =
      client_->client->UnsafeBufferPointer(xla_pjrt_buffer);
  ASSERT_TRUE(local_buffer_pointer.ok());
  ASSERT_NE(local_buffer_pointer.value(), 0);

  // Call the function through PJRT C API interface, check that the
  // result is not zero (nullptr).
  PJRT_Buffer_UnsafePointer_Args args{
      .struct_size = PJRT_Buffer_UnsafePointer_Args_STRUCT_SIZE,
      .priv = nullptr,
      .buffer = buffer_.get(),
  };

  PJRT_Error* error = api_->PJRT_Buffer_UnsafePointer(&args);
  ::pjrt::LogFatalIfPjrtError(error, api_);
  ASSERT_EQ(error, nullptr);
  ASSERT_NE(args.buffer_pointer, 0);

  // Confirm pointer values for direct and PJRT C API calls are the same.
  ASSERT_EQ(args.buffer_pointer, local_buffer_pointer.value());
}

// ---------------------------------- Events -----------------------------------

static PJRT_Event* EventFromPromise(PjRtFuture<Status>::Promise promise) {
  return new PJRT_Event{PjRtFuture<Status>{promise}};
}

class PjrtCApiEventsTest : public PjrtCApiTest {
 protected:
  void SetUp() override {
    PjrtCApiTest::SetUp();
    test_promise_ =  // to be set inside test cases
        std::make_unique<PjRtFuture<Status>::Promise>(
            PjRtFuture<Status>::CreatePromise());
    event_ = std::unique_ptr<PJRT_Event, ::pjrt::PJRT_EventDeleter>{
        EventFromPromise(*test_promise_), ::pjrt::MakeEventDeleter(api_)};
  }

  void TearDown() override {
    event_.reset();  // does not replace the deleter
    test_promise_.reset();
    PjrtCApiTest::TearDown();
  }

  void SetEventFromStatus(xla::Status status) { test_promise_->Set(status); }

  bool IsReady() {
    PJRT_Event_IsReady_Args ready_args;
    ready_args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
    ready_args.priv = nullptr;
    ready_args.event = event_.get();
    ::pjrt::LogFatalIfPjrtError(api_->PJRT_Event_IsReady(&ready_args), api_);
    return ready_args.is_ready;
  }

  void SetOnReady(PJRT_Event_OnReadyCallback callback, void* arguments) {
    PJRT_Event_OnReady_Args args{
        .struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE,
        .priv = nullptr,
        .event = event_.get(),
        .callback = callback,
        .user_arg = arguments,
    };

    auto error = ToUniquePtr(api_->PJRT_Event_OnReady(&args));
    CHECK(error == nullptr);
  }

  PJRT_Error* Await() {
    PJRT_Event_Await_Args args;
    args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.event = event_.get();
    return api_->PJRT_Event_Await(&args);
  }

  PJRT_Error* GetError() {
    PJRT_Event_Error_Args args;
    args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.event = event_.get();
    return api_->PJRT_Event_Error(&args);
  }

  std::unique_ptr<PjRtFuture<Status>::Promise> test_promise_;
  // tracks test_promise_
  std::unique_ptr<PJRT_Event, ::pjrt::PJRT_EventDeleter> event_;
};

constexpr static std::string_view kTestErrorMessage = "Test error message";

TEST_F(PjrtCApiEventsTest, IsReady) {
  PJRT_Event_IsReady_Args args;
  args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.event = event_.get();
  PJRT_Error* error = nullptr;

  // Not ready when initiated from a blank/default promise
  error = api_->PJRT_Event_IsReady(&args);
  ASSERT_EQ(error, nullptr);
  ASSERT_FALSE(args.is_ready);

  test_promise_->Set(xla::OkStatus());

  // Ready as soon as the promise is fulfilled
  error = api_->PJRT_Event_IsReady(&args);
  EXPECT_EQ(error, nullptr);
  EXPECT_TRUE(args.is_ready);
}

TEST_F(PjrtCApiEventsTest, IsReadyWhenPromisePreFilled) {
  SetEventFromStatus(xla::OkStatus());

  PJRT_Event_IsReady_Args args;
  args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.event = event_.get();

  PJRT_Error* error = api_->PJRT_Event_IsReady(&args);
  EXPECT_EQ(error, nullptr);
  EXPECT_TRUE(args.is_ready);
}

TEST_F(PjrtCApiEventsTest, IsReadyOnError) {
  auto test_err_code = tsl::error::Code::INTERNAL;
  const xla::Status test_status{test_err_code, kTestErrorMessage};
  SetEventFromStatus(test_status);

  PJRT_Event_IsReady_Args args;
  args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.event = event_.get();

  PJRT_Error* error = api_->PJRT_Event_IsReady(&args);
  EXPECT_EQ(error, nullptr);
  EXPECT_TRUE(args.is_ready);
}

TEST_F(PjrtCApiEventsTest, AwaitYieldsCorrectErrorWhenSet) {
  ASSERT_FALSE(IsReady());

  auto test_err_code = tsl::error::Code::INTERNAL;
  const xla::Status test_status{test_err_code, kTestErrorMessage};
  test_promise_->Set(test_status);
  ASSERT_TRUE(IsReady())
      << "Error: `event_` is not ready after `test_promise_` was Set";

  auto error = ToUniquePtr(Await());
  ASSERT_NE(error, nullptr);

  xla::Status status = ::pjrt::PjrtErrorToStatus(error.get(), api_);
  ASSERT_EQ(status.code(), test_err_code);
  absl::string_view error_message =
      ::pjrt::GetPjrtErrorMessage(error.get(), api_);
  ASSERT_EQ(error_message, kTestErrorMessage);
}

// Once PJRT_Event_Await is called, calls to PJRT_Event_Error should return
// pointers to an equivalent object
TEST_F(PjrtCApiEventsTest, GetErrorNoError) {
  test_promise_->Set(xla::OkStatus());
  ASSERT_TRUE(IsReady())
      << "Error: `event_` is not ready after `test_promise_` was Set";

  auto await_error = ToUniquePtr(Await());
  ASSERT_EQ(await_error, nullptr)
      << "`api_->PJRT_Event_Await(event_)` was not null in `" << __func__
      << "()` for `xla::OkStatus()`.";

  auto status_error = ToUniquePtr(GetError());
  ASSERT_EQ(status_error, nullptr)
      << "`api_->PJRT_Event_Error(event_)` was not null in `" << __func__
      << "()` after correctly null `api_->PJRT_Event_Await(event_)` for "
         "`xla::OkStatus()`.";
}

TEST_F(PjrtCApiEventsTest, GetErrorYesError) {
  auto test_err_code = tsl::error::Code::INTERNAL;
  const xla::Status test_status{test_err_code, kTestErrorMessage};
  test_promise_->Set(test_status);
  ASSERT_TRUE(IsReady())
      << "Error: `event_` is not ready after `test_promise_` was Set";

  auto await_error = ToUniquePtr(Await());
  ASSERT_NE(await_error, nullptr)
      << "`api_->PJRT_Event_Await(event_)` was null in `" << __func__ << "()`.";
  ASSERT_EQ(await_error->status.code(), test_err_code);
  ASSERT_EQ(await_error->status.error_message(), kTestErrorMessage);
  await_error.reset();

  auto status_error = ToUniquePtr(GetError());
  ASSERT_NE(status_error, nullptr)
      << "`api_->PJRT_Event_Error(event_)` was null in `" << __func__
      << "()` after non-null `api_->PJRT_Event_Await(event_)`.";

  xla::Status status = ::pjrt::PjrtErrorToStatus(status_error.get(), api_);
  EXPECT_EQ(status.code(), test_err_code);
  absl::string_view error_message =
      ::pjrt::GetPjrtErrorMessage(status_error.get(), api_);
  ASSERT_EQ(error_message, kTestErrorMessage);
}

TEST_F(PjrtCApiEventsTest, GetErrorThenAwait) {
  auto test_err_code = tsl::error::Code::INTERNAL;
  const xla::Status test_status{test_err_code, kTestErrorMessage};
  test_promise_->Set(test_status);
  ASSERT_TRUE(IsReady())
      << "Error: `event_` is not ready after `test_promise_` was Set";

  auto status_error = ToUniquePtr(GetError());
  ASSERT_NE(status_error, nullptr)
      << "`api_->PJRT_Event_Error(event_)` was null in `" << __func__
      << "()` after non-null `api_->PJRT_Event_Await(event_)`.";
  xla::Status status = ::pjrt::PjrtErrorToStatus(status_error.get(), api_);
  EXPECT_EQ(status.code(), test_err_code);
  absl::string_view error_message =
      ::pjrt::GetPjrtErrorMessage(status_error.get(), api_);
  EXPECT_EQ(error_message, kTestErrorMessage);
  status_error.reset();

  auto await_error = ToUniquePtr(Await());
  ASSERT_NE(await_error, nullptr)
      << "`api_->PJRT_Event_Await(event_)` was null in `" << __func__ << "()`.";
  ASSERT_EQ(await_error->status.code(), test_err_code);
  ASSERT_EQ(await_error->status.error_message(), kTestErrorMessage);
}

struct StringAndApi {
  std::string* str;
  const PJRT_Api* api;
  bool is_set = false;
  bool error_was_null = false;
};

static void StringWriteCallback(PJRT_Error* error, void* void_arg_pointer) {
  auto string_and_api = reinterpret_cast<StringAndApi*>(void_arg_pointer);
  CHECK(string_and_api != nullptr);
  if (error == nullptr) {
    *string_and_api->str = "";
    string_and_api->is_set = true;
    string_and_api->error_was_null = true;
    return;
  }
  *string_and_api->str =
      ::pjrt::GetPjrtErrorMessage(error, string_and_api->api);
  string_and_api->is_set = true;
  string_and_api->error_was_null = false;
  PJRT_Error_Destroy_Args destroy;
  destroy.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy.priv = nullptr;
  destroy.error = error;
  string_and_api->api->PJRT_Error_Destroy(&destroy);
}

constexpr static std::string_view kInitialMessage =
    "Should never end up with this string";

TEST_F(PjrtCApiEventsTest, OnReadyNoError) {
  ASSERT_FALSE(IsReady());
  std::string side_effect{kInitialMessage};
  StringAndApi wrapper{&side_effect, api_};
  SetOnReady(StringWriteCallback, &wrapper);

  test_promise_->Set(xla::OkStatus());
  ASSERT_TRUE(IsReady());
  // We must wait for the callback to complete
  absl::Mutex mu;
  mu.LockWhen(absl::Condition{+[](bool* b) { return *b; }, &wrapper.is_set});
  // The callback should be complete by now
  mu.Unlock();
  // No error -> xla::Status::error_message() returns empty string
  EXPECT_EQ(side_effect, "");
  EXPECT_TRUE(wrapper.error_was_null);
}

TEST_F(PjrtCApiEventsTest, OnReadyWithError) {
  ASSERT_FALSE(IsReady());
  std::string side_effect{kInitialMessage};
  StringAndApi wrapper{&side_effect, api_};
  SetOnReady(StringWriteCallback, &wrapper);

  auto test_err_code = tsl::error::Code::INTERNAL;
  const xla::Status test_status{test_err_code, kTestErrorMessage};
  test_promise_->Set(test_status);

  ASSERT_TRUE(IsReady());
  // We must wait for the callback to complete
  absl::Mutex mu;
  mu.LockWhen(absl::Condition{+[](bool* b) { return *b; }, &wrapper.is_set});
  // The callback should be complete by now
  mu.Unlock();
  EXPECT_EQ(side_effect, kTestErrorMessage);
  EXPECT_FALSE(wrapper.error_was_null);
}

// --------------------------------- Helpers -----------------------------------

class PjrtCApiHelpersTest : public PjrtCApiTest {};

TEST_F(PjrtCApiHelpersTest, PjrtErrorToStatus) {
  // Return success if nullptr
  EXPECT_TRUE(::pjrt::PjrtErrorToStatus(nullptr, api_).ok());

  // Return UNKNOWN status with the original message if not nullptr
  auto error = std::make_unique<PJRT_Error>();
  error->status = tsl::errors::InvalidArgument("Should be UNKNOWN");
  xla::Status status = ::pjrt::PjrtErrorToStatus(error.get(), api_);
  EXPECT_EQ(status.code(), tsl::error::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(), "Should be UNKNOWN");
}

// For these Deleter tests, the heap leak checker will check if heap memory
// allocated with `new` is properly freed at the exit.
TEST_F(PjrtCApiHelpersTest, MakeErrorDeleter) {
  PJRT_Error* error = new PJRT_Error();
  ::pjrt::MakeErrorDeleter(api_)(error);
}

TEST_F(PjrtCApiHelpersTest, MakeEventDeleter) {
  PJRT_Event* event = new PJRT_Event();
  ::pjrt::MakeEventDeleter(api_)(event);
}

}  // namespace
}  // namespace pjrt
}  // namespace xla
