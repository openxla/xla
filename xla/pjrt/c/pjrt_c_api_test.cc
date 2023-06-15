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
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// #include "platforms/xla/megascale/jax/megascale_pjrt_compiler.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/client/xla_computation.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_parser.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace pjrt {
namespace {
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
constexpr absl::string_view kHloString =
    R"(
HloModule TupleCreate_module:
ENTRY %TupleCreate.v4 (v1: f32[], v2: f32[3], v3: f32[2,3]) -> (f32[], f32[3], f32[2,3]) {
  %v1 = f32[] parameter(0)
  %v2 = f32[3]{0} parameter(1)
  %v3 = f32[2,3]{1,0} parameter(2)
  ROOT %tuple = (f32[], f32[3]{0}, f32[2,3]{1,0}) tuple(f32[] %v1, f32[3]{0} %v2, f32[2,3]{1,0} %v3)
}
)";

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
  // We directly access the internal C++ client to test if the C API has the
  // same behavior as the C++ API.
  xla::PjRtClient* cc_client_;
  XlaComputation xla_computation_;

  void SetUp() override {
    api_ = GetCApi();
    client_ = make_client();
  }

  void TearDown() override {  // destroy_client(client_);
  }

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

  int GetDeviceId(PJRT_DeviceDescription* device_desc) const {
    PJRT_DeviceDescription_Id_Args args = PJRT_DeviceDescription_Id_Args{
        .struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE,
        .priv = nullptr,
        .device_description = device_desc,
        .id = -1,
    };
    PJRT_Error* error = api_->PJRT_DeviceDescription_Id(&args);
    CHECK_EQ(error, nullptr);
    return args.id;
  }

  int GetDeviceId(PJRT_Device* device) const {
    return GetDeviceId(::pjrt::GetDeviceDescription(api_, device));
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

  absl::Span<PJRT_Device*> GetClientDevices() const {
    PJRT_Client_Devices_Args dev_args;
    dev_args.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
    dev_args.priv = nullptr;
    dev_args.client = client_;
    PJRT_Error* error = api_->PJRT_Client_Devices(&dev_args);
    TF_CHECK_OK(::pjrt::PjrtErrorToStatus(error, api_));
    return absl::MakeSpan(dev_args.devices, dev_args.num_devices);
  }

  int GetNumDevices() const { return GetClientDevices().size(); }

  absl::Span<PJRT_Device*> GetClientAddressableDevices() const {
    PJRT_Client_AddressableDevices_Args addr_args;
    addr_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    addr_args.priv = nullptr;
    addr_args.client = client_;
    PJRT_Error* error = api_->PJRT_Client_AddressableDevices(&addr_args);
    TF_CHECK_OK(::pjrt::PjrtErrorToStatus(error, api_));
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
    absl::StatusOr<CompileOptionsProto> options_proto = options.ToProto();
    TF_CHECK_OK(options_proto.status());
    return options_proto->SerializeAsString();
  }

  PJRT_Client_BufferFromHostBuffer_Args CreateBufferFromHostBufferArgs(
      const std::vector<float>& data, const Shape& shape,
      const xla::PjRtClient::HostBufferSemantics host_buffer_semantics,
      PJRT_Device* device = nullptr) {
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
    args.client = client_;
    if (device == nullptr) {
      device = GetClientAddressableDevices()[0];
    }
    args.device = device;
    return args;
  }

  std::pair<std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter>,
            PjRtFuture<absl::Status>>
  create_buffer(PJRT_Device* device = nullptr) {
    Shape shape = ShapeUtil::MakeShapeWithType<float>({4});
    std::vector<float> float_data(4);
    std::iota(float_data.begin(), float_data.end(), 41.0f);

    PJRT_Client_BufferFromHostBuffer_Args args =
    CreateBufferFromHostBufferArgs(
        float_data, shape,
        xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
        device);

    auto transfer_error =
        ToUniquePtr(api_->PJRT_Client_BufferFromHostBuffer(&args));
    EXPECT_EQ(transfer_error, nullptr);

    std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> buffer(
        args.buffer, ::pjrt::MakeBufferDeleter(api_));

    std::unique_ptr<PJRT_Event, ::pjrt::PJRT_EventDeleter>
        done_with_host_buffer_event(args.done_with_host_buffer,
                                    ::pjrt::MakeEventDeleter(api_));

    PJRT_Buffer_ReadyEvent_Args get_event_args;
    get_event_args.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    get_event_args.priv = nullptr;
    get_event_args.buffer = buffer.get();
    auto ready_event_error =
        ToUniquePtr(api_->PJRT_Buffer_ReadyEvent(&get_event_args));
    EXPECT_EQ(ready_event_error, nullptr);
    PjRtFuture<absl::Status> buffer_ready_event =
        ::pjrt::ConvertCEventToCppFuture(get_event_args.event, api_);

    return std::make_pair(std::move(buffer), buffer_ready_event);
  }

  // Returns a scalar result of execution.
  // supply as e.g. `src_buffer = args.output_lists[0][0];`
  // after calling `api_->PJRT_LoadedExecutable_Execute(&args);`
  absl::StatusOr<float> GetProgramResult(PJRT_Buffer* src_buffer) {
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

    PjRtFuture<absl::Status> transfer_to_host =
        ::pjrt::ConvertCEventToCppFuture(args.event, api_);
    TF_RETURN_IF_ERROR(transfer_to_host.Await());
    return value;
  }

  // Runs the default executable created in PjrtCApiTpuExecutableTest:SetUp and
  // returns its output
  absl::StatusOr<float> RunScalarExecutableAndGetResult(
      PJRT_LoadedExecutable* executable) {
    PJRT_LoadedExecutable_Execute_Args args;
    args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.executable = executable;
    PJRT_ExecuteOptions c_options;
    c_options.num_send_ops = 0;
    c_options.num_recv_ops = 0;
    args.options = &c_options;
    args.options->struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    args.options->launch_id = 0;
    args.num_devices = 1;
    args.num_args = 1;
    auto buffer = create_buffer().first;
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
    auto buffer_deleter = ::pjrt::MakeBufferDeleter(api_);
    for (int i = 0; i < args.num_devices; ++i) {
      for (int j = 0; j < num_outputs_per_device; ++j) {
        buffer_deleter(args.output_lists[i][j]);
      }
    }
    return result;
  }
};

// TEST_F(PjrtCApiTest, CreateClientWithOption) {
//   std::vector<PJRT_NamedValue> c_options;
//   std::string option_name = "max_inflight_computations";
//   c_options.push_back(PJRT_NamedValue{
//       .name = option_name.c_str(),
//       .name_size = option_name.size(),
//       .type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64,
//       .int64_value = 4,
//       .value_size = 1,
//   });
//   PJRT_Client_Create_Args create_args = PJRT_Client_Create_Args{
//       .struct_size = PJRT_Client_Create_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .create_options = c_options.data(),
//       .num_options = 1,
//       .client = nullptr,
//   };

//   PJRT_Error* error = api_->PJRT_Client_Create(&create_args);

//   CHECK_EQ(error, nullptr);
//   CHECK_NE(create_args.client, nullptr);

//   destroy_client(create_args.client);
// }

// ---------------------------------- Client -----------------------------------

// TEST_F(PjrtCApiTest, ClientProcessIndex) {
//   PJRT_Client_ProcessIndex_Args process_index_args =
//       PJRT_Client_ProcessIndex_Args{
//           .struct_size = PJRT_Client_ProcessIndex_Args_STRUCT_SIZE,
//           .priv = nullptr,
//           .client = client_,
//           .process_index = -1,
//       };
//   PJRT_Error* error = api_->PJRT_Client_ProcessIndex(&process_index_args);
//   CHECK_EQ(error, nullptr);

//   // Single-process test should return 0
//   CHECK_EQ(process_index_args.process_index, 0);
// }

// TEST_F(PjrtCApiTest, ClientDevices) {
//   absl::Span<PJRT_Device*> devices = GetClientDevices();

//   ASSERT_FALSE(devices.empty());
//   for (auto& device : devices) {
//     ASSERT_TRUE(this->IsValidDeviceId(device));
//   }
// }

// TEST_F(PjrtCApiTest, ClientAddressableDevices) {
//   absl::Span<PJRT_Device*> addressable_devices =
//   GetClientAddressableDevices();

//   ASSERT_FALSE(addressable_devices.empty());
//   for (auto& device : addressable_devices) {
//     ASSERT_TRUE(this->IsValidDeviceId(device));
//   }

//   absl::Span<PJRT_Device*> client_devices = GetClientDevices();
//   for (auto& addressable_device : addressable_devices) {
//     ASSERT_THAT(client_devices, ::testing::Contains(addressable_device));
//   }
// }

// TEST_F(PjrtCApiTest, LookupDevice) {
//   PJRT_Client_LookupDevice_Args lookup_device_args =
//       PJRT_Client_LookupDevice_Args{
//           .struct_size = PJRT_Client_LookupDevice_Args_STRUCT_SIZE,
//           .priv = nullptr,
//           .client = client_,
//           .id = 0,
//           .device = nullptr,
//       };

//   PJRT_Error* lookup_device_error =
//       api_->PJRT_Client_LookupDevice(&lookup_device_args);

//   ASSERT_EQ(lookup_device_error, nullptr);
//   int id = GetDeviceId(lookup_device_args.device);
//   ASSERT_EQ(id, 0);
// }

// TEST_F(PjrtCApiTest, LookupAddressableDevice) {
//   PJRT_Client_LookupAddressableDevice_Args lookup_addressable_device_args =
//       PJRT_Client_LookupAddressableDevice_Args{
//           .struct_size =
//           PJRT_Client_LookupAddressableDevice_Args_STRUCT_SIZE, .priv =
//           nullptr, .client = client_, .local_hardware_id = 0,
//           .addressable_device = nullptr,
//       };

//   PJRT_Error* lookup_addressable_device_error =
//       api_->PJRT_Client_LookupAddressableDevice(
//           &lookup_addressable_device_args);

//   ASSERT_EQ(lookup_addressable_device_error, nullptr);
//   int local_hardware_id =
//       GetLocalHardwareId(lookup_addressable_device_args.addressable_device);
//   ASSERT_EQ(local_hardware_id, 0);
// }

// TEST_F(PjrtCApiTest, GetDefaultDeviceAssignmentNominal) {
//   constexpr int kNumReplicas = 2;
//   constexpr int kNumPartitions = 1;
//   std::vector<int> assignment_buffer(kNumReplicas * kNumPartitions);
//   PJRT_Client_DefaultDeviceAssignment_Args args{
//       .struct_size = PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .client = client_,
//       .num_replicas = kNumReplicas,
//       .num_partitions = kNumPartitions,
//       .default_assignment_size = assignment_buffer.size(),
//       .default_assignment = assignment_buffer.data(),  // in-out
//   };
//   auto error = ToUniquePtr(api_->PJRT_Client_DefaultDeviceAssignment(&args));
//   EXPECT_EQ(error, nullptr) << ::pjrt::GetPjrtErrorMessage(error.get(),
//   api_);
// }

// TEST_F(PjrtCApiTest, GetDefaultDeviceAssignmentBufferTooSmall) {
//   constexpr int kNumReplicas = 4;
//   constexpr int kNumPartitions = 2;
//   constexpr size_t kBufferSize = 7;
//   std::vector<int> assignment_buffer(kBufferSize);
//   PJRT_Client_DefaultDeviceAssignment_Args args{
//       .struct_size = PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .client = client_,
//       .num_replicas = kNumReplicas,
//       .num_partitions = kNumPartitions,
//       .default_assignment_size = assignment_buffer.size(),
//       .default_assignment = assignment_buffer.data(),  // in-out
//   };
//   auto error = ToUniquePtr(api_->PJRT_Client_DefaultDeviceAssignment(&args));
//   ASSERT_NE(error, nullptr);
//   absl::Status status = ::pjrt::PjrtErrorToStatus(error.get(), api_);
//   EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
//   EXPECT_EQ(status.message(),
//             "PJRT_Client_DefaultDeviceAssignment: `default_assignment_size`
//             7" " < `num_replicas * num_partitions`, 4 * 2 = 8");
// }

// TEST_F(PjrtCApiTest, LookupDeviceNegativeId) {
//   PJRT_Client_LookupDevice_Args args = PJRT_Client_LookupDevice_Args{
//       .struct_size = PJRT_Client_LookupDevice_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .client = client_,
//       .id = -1,
//       .device = nullptr,
//   };
//   absl::Status expected =
//       absl::Status(absl::StatusCode::kInvalidArgument,
//                    "No matching device found for device_id -1");

//   auto error = ToUniquePtr(api_->PJRT_Client_LookupDevice(&args));

//   ASSERT_NE(error, nullptr);
//   absl::Status status = ::pjrt::PjrtErrorToStatus(error.get(), api_);
//   ASSERT_EQ(status, expected);
// }

// TEST_F(PjrtCApiTest, LookupDeviceOutOfRangeId) {
//   int out_of_range_id = GetNumDevices();
//   PJRT_Client_LookupDevice_Args args = PJRT_Client_LookupDevice_Args{
//       .struct_size = PJRT_Client_LookupDevice_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .client = client_,
//       .id = out_of_range_id,
//       .device = nullptr,
//   };
//   absl::Status expected = absl::Status(
//       absl::StatusCode::kInvalidArgument,
//       absl::StrCat("No matching device found for device_id ",
//       out_of_range_id));

//   auto error = ToUniquePtr(api_->PJRT_Client_LookupDevice(&args));

//   ASSERT_NE(error, nullptr);
//   absl::Status status = ::pjrt::PjrtErrorToStatus(error.get(), api_);
//   ASSERT_EQ(status, expected);
// }

// static constexpr std::string_view kExecutableName = "operation";

// std::unique_ptr<PJRT_TopologyDescription,
//                 ::pjrt::PJRT_TopologyDescriptionDeleter>
// CreateTopology(const PJRT_Api* c_api) {
//   PJRT_TopologyDescription_Create_Args init_args;
//   init_args.struct_size = PJRT_TopologyDescription_Create_Args_STRUCT_SIZE;
//   init_args.priv = nullptr;
//   ::pjrt::LogFatalIfPjrtError(
//       c_api->PJRT_TopologyDescription_Create(&init_args), c_api);
//   PJRT_TopologyDescription* c_topology = init_args.topology;
//   return std::unique_ptr<PJRT_TopologyDescription,
//                          ::pjrt::PJRT_TopologyDescriptionDeleter>(
//       c_topology, ::pjrt::MakeTopologyDescriptionDeleter(c_api));
// }

// void destroy_executable(PJRT_LoadedExecutable* executable,
//                         const PJRT_Api* api) {
//   PJRT_LoadedExecutable_Destroy_Args args{
//       .struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .executable = executable,
//   };
//   PJRT_Error* error = api->PJRT_LoadedExecutable_Destroy(&args);
//   CHECK_EQ(error, nullptr);
// }

// TEST_F(PjrtCApiTest, BufferTransferImmutableUntilTransferCompletes) {
//   Shape shape = ShapeUtil::MakeShapeWithType<float>({4});
//   std::vector<float> float_data(4);
//   std::iota(float_data.begin(), float_data.end(), 41.0f);

//   PJRT_Client_BufferFromHostBuffer_Args args =
//   CreateBufferFromHostBufferArgs(
//       float_data, shape,
//       xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes);

//   PJRT_Error* error = api_->PJRT_Client_BufferFromHostBuffer(&args);
//   CHECK_EQ(error, nullptr);

// std::unique_ptr<PJRT_TopologyDescription,
//                 ::pjrt::PJRT_TopologyDescriptionDeleter>
// CreateTopologyWithOptions(
//     const PJRT_Api* c_api, const std::string& topology_name,
//     const absl::flat_hash_map<std::string, PjRtValueType>& create_options) {
//   PJRT_TopologyDescription_Create_Args init_args;
//   init_args.struct_size = PJRT_TopologyDescription_Create_Args_STRUCT_SIZE;
//   init_args.priv = nullptr;
//   init_args.topology_name = topology_name.c_str();
//   init_args.topology_name_size = topology_name.size();
//   auto c_options =
//   ::pjrt::ConvertToPjRtNamedValueList(create_options).value();
//   init_args.create_options = c_options.data();
//   init_args.num_options = c_options.size();
//   ::pjrt::LogFatalIfPjrtError(
//       c_api->PJRT_TopologyDescription_Create(&init_args), c_api);
//   PJRT_TopologyDescription* c_topology = init_args.topology;
//   return std::unique_ptr<PJRT_TopologyDescription,
//                          ::pjrt::PJRT_TopologyDescriptionDeleter>(
//       c_topology, ::pjrt::MakeTopologyDescriptionDeleter(c_api));
// }

// std::unique_ptr<PJRT_TopologyDescription,
//                 ::pjrt::PJRT_TopologyDescriptionDeleter>
// CreateTopology(const PJRT_Api* c_api) {
//   return CreateTopologyWithOptions(c_api, "", {});
// }
//   std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> buffer(
//       args.buffer, ::pjrt::MakeBufferDeleter(api_));

//   std::unique_ptr<PJRT_Event, ::pjrt::PJRT_EventDeleter> event(
//       args.done_with_host_buffer, ::pjrt::MakeEventDeleter(api_));

//   PJRT_Event_Await_Args await_args;
//   await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
//   await_args.priv = nullptr;
//   await_args.event = event.get();
//   PJRT_Error* event_error = api_->PJRT_Event_Await(&await_args);
//   ASSERT_EQ(event_error, nullptr);
// }

// TEST_F(PjrtCApiTest, Compile) {
//   PJRT_Client_Compile_Args args = PJRT_Client_Compile_Args{
//       .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .client = client_,
//   };
//   std::string options_str = BuildSingleDeviceCompileOptionStr();
//   args.compile_options = options_str.c_str();
//   args.compile_options_size = options_str.size();

//   std::string format(::pjrt::kMlirFormat);
//   std::string program_code{module_add_one};
//   PJRT_Program program = PJRT_Program{
//       .struct_size = PJRT_Program_STRUCT_SIZE,
//       .priv = nullptr,
//       .code = program_code.data(),
//       .code_size = program_code.length(),
//       .format = format.c_str(),
//       .format_size = format.size(),
//   };
//   args.program = &program;

//   PJRT_Error* error = api_->PJRT_Client_Compile(&args);
//   ::pjrt::LogFatalIfPjrtError(error, api_);

//   ASSERT_EQ(error, nullptr);
//   destroy_executable(args.executable, api_);
// }

// TEST_F(PjrtCApiTest, CompileXlaComputation) {
//   PJRT_Client_Compile_Args args = PJRT_Client_Compile_Args{
//       .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .client = client_,
//   };
//   DeviceAssignment device_assignment(1, 1);
//   device_assignment(0, 0) = 0;
//   DeviceAssignmentProto proto;
//   ASSERT_TRUE(device_assignment.Serialize(&proto).ok());
//   std::string device_assignment_str = proto.SerializeAsString();
//   std::string options_str = BuildSingleDeviceCompileOptionStr();
//   args.compile_options = options_str.c_str();
//   args.compile_options_size = options_str.size();

//   absl::StatusOr<std::unique_ptr<HloModule>> hlo_module =
//       xla::ParseAndReturnUnverifiedModule(kHloString);
//   ASSERT_EQ(hlo_module.ok(), true);
//   std::string module_str = hlo_module->get()->ToProto().SerializeAsString();

//   std::string format(::pjrt::kHloFormat);
//   PJRT_Program program = PJRT_Program{
//       .struct_size = PJRT_Program_STRUCT_SIZE,
//       .priv = nullptr,
//       .code = module_str.data(),
//       .code_size = module_str.size(),
//       .format = format.c_str(),
//       .format_size = format.size(),
//   };
//   args.program = &program;

//   PJRT_Error* error = api_->PJRT_Client_Compile(&args);
//   ::pjrt::LogFatalIfPjrtError(error, api_);

//   ASSERT_EQ(error, nullptr);
//   destroy_executable(args.executable, api_);
// }

// TEST_F(PjrtCApiTest, CompileInvalidOption) {
//   PJRT_Client_Compile_Args args = PJRT_Client_Compile_Args{
//       .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .client = client_,
//   };
//   std::string options_str = "invalid compile options";
//   args.compile_options = options_str.c_str();
//   args.compile_options_size = options_str.size();

//   std::string format(::pjrt::kMlirFormat);
//   std::string program_code{module_add_one};
//   PJRT_Program program = PJRT_Program{
//       .struct_size = PJRT_Program_STRUCT_SIZE,
//       .priv = nullptr,
//       .code = program_code.data(),
//       .code_size = program_code.length(),
//       .format = format.c_str(),
//       .format_size = format.size(),
//   };
//   args.program = &program;

//   PJRT_Error* error = api_->PJRT_Client_Compile(&args);

//   absl::Status status = ::pjrt::PjrtErrorToStatus(error, api_);
//   EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
//   EXPECT_EQ(status.message(),
//             "PJRT_Client_Compile: failed to deserialize
//             CompileOptionsProto");
//   destroy_executable(args.executable, api_);
//   ::pjrt::MakeErrorDeleter(api_)(error);
// }

// TEST_F(PjrtCApiTest, CompileInvalidProgramFormat) {
//   PJRT_Client_Compile_Args args = PJRT_Client_Compile_Args{
//       .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .client = client_,
//   };
//   DeviceAssignment device_assignment(1, 1);
//   device_assignment(0, 0) = 0;
//   DeviceAssignmentProto proto;
//   ASSERT_TRUE(device_assignment.Serialize(&proto).ok());
//   std::string device_assignment_str = proto.SerializeAsString();
//   std::string options_str = BuildSingleDeviceCompileOptionStr();
//   args.compile_options = options_str.c_str();
//   args.compile_options_size = options_str.size();

//   std::string format("invalid");
//   PJRT_Program program = PJRT_Program{
//       .struct_size = PJRT_Program_STRUCT_SIZE,
//       .priv = nullptr,
//       .code = nullptr,
//       .code_size = 0,
//       .format = format.c_str(),
//       .format_size = format.size(),
//   };
//   args.program = &program;

//   PJRT_Error* error = api_->PJRT_Client_Compile(&args);
//   absl::Status status = ::pjrt::PjrtErrorToStatus(error, api_);
//   EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
//   EXPECT_EQ(status.message(), "Unknown program format 'invalid'.");
//   destroy_executable(args.executable, api_);
//   ::pjrt::MakeErrorDeleter(api_)(error);
// }

//   PJRT_Error* error = api_->PJRT_Client_Compile(&args);
//   ::pjrt::LogFatalIfPjrtError(error, api_);

//   ASSERT_EQ(error, nullptr);
//   destroy_executable(args.executable, api_);
// }

// TEST_F(PjrtCApiTpuTest, CompileInvalidOption) {
//   PJRT_Client_Compile_Args args = PJRT_Client_Compile_Args{
//       .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .client = client_,
//   };
//   std::string options_str = "invalid compile options";
//   args.compile_options = options_str.c_str();
//   args.compile_options_size = options_str.size();

//   std::string format(::pjrt::kMlirFormat);
//   std::string program_code{module_add_one};
//   PJRT_Program program = PJRT_Program{
//       .struct_size = PJRT_Program_STRUCT_SIZE,
//       .priv = nullptr,
//       .code = program_code.data(),
//       .code_size = program_code.length(),
//       .format = format.c_str(),
//       .format_size = format.size(),
//   };
//   args.program = &program;

//   PJRT_Error* error = api_->PJRT_Client_Compile(&args);

//   absl::Status status = ::pjrt::PjrtErrorToStatus(error, api_);
//   EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
//   EXPECT_EQ(status.error_message(),
//             "PJRT_Client_Compile: failed to deserialize
//             CompileOptionsProto");
//   destroy_executable(args.executable, api_);
//   ::pjrt::MakeErrorDeleter(api_)(error);
// }

// TEST_F(PjrtCApiTpuTest, CompileInvalidProgramFormat) {
//   PJRT_Client_Compile_Args args = PJRT_Client_Compile_Args{
//       .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .client = client_,
//   };
//   DeviceAssignment device_assignment(1, 1);
//   device_assignment(0, 0) = 0;
//   DeviceAssignmentProto proto;
//   ASSERT_TRUE(device_assignment.Serialize(&proto).ok());
//   std::string device_assignment_str = proto.SerializeAsString();
//   std::string options_str = BuildSingleDeviceCompileOptionStr();
//   args.compile_options = options_str.c_str();
//   args.compile_options_size = options_str.size();

//   std::string format("invalid");
//   PJRT_Program program = PJRT_Program{
//       .struct_size = PJRT_Program_STRUCT_SIZE,
//       .priv = nullptr,
//       .code = nullptr,
//       .code_size = 0,
//       .format = format.c_str(),
//       .format_size = format.size(),
//   };
//   args.program = &program;

//   PJRT_Error* error = api_->PJRT_Client_Compile(&args);
//   absl::Status status = ::pjrt::PjrtErrorToStatus(error, api_);
//   EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
//   EXPECT_EQ(status.error_message(), "Unknown program format 'invalid'.");
//   destroy_executable(args.executable, api_);
//   ::pjrt::MakeErrorDeleter(api_)(error);
// }

// // --------------------------------- Devices
// -----------------------------------

// TEST_F(PjrtCApiTpuTest, DeviceId) {
//   auto* device = GetClientDevices()[0];

//   int id = GetDeviceId(device);

//   CHECK_EQ(id, 0);
// }

// TEST_F(PjrtCApiTpuTest, DeviceProcessIndex) {
//   PJRT_DeviceDescription_ProcessIndex_Args args =
//       PJRT_DeviceDescription_ProcessIndex_Args{
//           .struct_size =
//           PJRT_DeviceDescription_ProcessIndex_Args_STRUCT_SIZE, .priv =
//           nullptr, .device_description =
//               ::pjrt::GetDeviceDescription(api_, GetClientDevices()[0]),
//           .process_index = -1,
//       };
//   PJRT_Error* error = api_->PJRT_DeviceDescription_ProcessIndex(&args);
//   ASSERT_EQ(error, nullptr);
//   // For single process, it should match client process index
//   CHECK_EQ(args.process_index, 0);
// }

// TEST_F(PjrtCApiTpuTest, DeviceIsAddressable) {
//   PJRT_Device_IsAddressable_Args args = PJRT_Device_IsAddressable_Args{
//       .struct_size = PJRT_Device_IsAddressable_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .device = GetClientDevices()[0],
//       .is_addressable = false,
//   };
//   PJRT_Error* error = api_->PJRT_Device_IsAddressable(&args);
//   ASSERT_EQ(error, nullptr);
//   // All devices are addressable in single-process test
//   CHECK_EQ(args.is_addressable, true);
// }

// TEST_F(PjrtCApiTpuTest, DeviceAttributes) {
//   auto devices = GetClientDevices();
//   for (const auto& device : devices) {
//     auto attributes = device->device->Attributes();

//     PJRT_DeviceDescription_Attributes_Args args =
//         PJRT_DeviceDescription_Attributes_Args{
//             .struct_size =
//             PJRT_DeviceDescription_Attributes_Args_STRUCT_SIZE, .priv =
//             nullptr, .device_description = ::pjrt::GetDeviceDescription(api_,
//             device),
//         };

//     PJRT_Error* error = api_->PJRT_DeviceDescription_Attributes(&args);
//     ASSERT_EQ(error, nullptr);
//     ASSERT_EQ(args.num_attributes, attributes.size());

//     for (int i = 0; i < args.num_attributes; ++i) {
//       const auto& attribute = args.attributes[i];
//       ASSERT_EQ(attribute.struct_size, PJRT_NamedValue_STRUCT_SIZE);
//       ASSERT_EQ(attribute.priv, nullptr);
//       std::string attribute_name(attribute.name, attribute.name_size);
//       ASSERT_TRUE(attributes.contains(attribute_name));
//       switch (attribute.type) {
//         case PJRT_NamedValue_Type::PJRT_NamedValue_kString: {
//           std::string string_value(attribute.string_value);
//           ASSERT_EQ(std::get<std::string>(attributes[attribute_name]),
//                     string_value);
//           break;
//         }
//         case PJRT_NamedValue_Type::PJRT_NamedValue_kInt64: {
//           ASSERT_EQ(std::get<int64_t>(attributes[attribute_name]),
//                     attribute.int64_value);
//           break;
//         }
//         case PJRT_NamedValue_Type::PJRT_NamedValue_kInt64List: {
//           const int64_t* value_ptr = attribute.int64_array_value;
//           std::vector<int64_t> array_value(value_ptr,
//                                            value_ptr + attribute.value_size);
//           ASSERT_EQ(std::get<std::vector<int64_t>>(attributes[attribute_name]),
//                     array_value);
//           break;
//         }
//           // Do not allow other types (such as
//           // PJRT_NamedValue::PJRT_NamedValue_kFloat) since device attributes
//           // currently should not return other types.
//         default: {
//           // should never get here.
//           FAIL() << "attribute value type " << attribute.type
//                  << " invalid; should have been string, int64, or int64_list.
//                  "
//                     "This should never occur.";
//         }
//       }
//     }
//     ASSERT_EQ(error, nullptr);
//   }
// }

// TEST_F(PjrtCApiTpuTest, DeviceKind) {
//   PJRT_DeviceDescription_Kind_Args args = PJRT_DeviceDescription_Kind_Args{
//       .struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .device_description =
//           ::pjrt::GetDeviceDescription(api_, GetClientDevices()[0]),
//   };
//   PJRT_Error* error = api_->PJRT_DeviceDescription_Kind(&args);
//   ASSERT_EQ(error, nullptr);
//   CHECK_STREQ(args.device_kind,
//   cc_client_->devices()[0]->device_kind().data());
// }

// TEST_F(PjrtCApiTpuTest, DeviceLocalHardwareId) {
//   PJRT_Device_LocalHardwareId_Args args = PJRT_Device_LocalHardwareId_Args{
//       .struct_size = PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .device = GetClientDevices()[0],
//       .local_hardware_id = -1,
//   };
//   PJRT_Error* error = api_->PJRT_Device_LocalHardwareId(&args);
//   ASSERT_EQ(error, nullptr);
//   CHECK_EQ(args.local_hardware_id, 0);
// }

// TEST_F(PjrtCApiTpuTest, DeviceDebugString) {
//   PJRT_DeviceDescription_DebugString_Args args;
//   args.device_description =
//       ::pjrt::GetDeviceDescription(api_, GetClientDevices()[0]);
//   args.struct_size = PJRT_DeviceDescription_DebugString_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   args.debug_string = nullptr;
//   PJRT_Error* error = api_->PJRT_DeviceDescription_DebugString(&args);
//   ASSERT_EQ(error, nullptr);
//   // The debug string for device 0, process 0
//   std::string expected_string = "TPU_0(process=0,(0,0,0,0))";
//   CHECK_EQ(expected_string, args.debug_string);
// }

// TEST_F(PjrtCApiTpuTest, DeviceToString) {
//   PJRT_DeviceDescription_ToString_Args args;
//   args.device_description =
//       ::pjrt::GetDeviceDescription(api_, GetClientDevices()[0]);
//   args.struct_size = PJRT_DeviceDescription_ToString_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   args.to_string = nullptr;
//   PJRT_Error* error = api_->PJRT_DeviceDescription_ToString(&args);
//   ASSERT_EQ(error, nullptr);
//   std::string expected_string =
//       "TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)";
//   CHECK_EQ(expected_string, args.to_string);
// }

// TEST_F(PjrtCApiTpuTest, DeviceMemoryStats) {
//   PJRT_Device_MemoryStats_Args args{
//       .struct_size = PJRT_Device_MemoryStats_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .device = GetClientDevices()[0]};
//   // Set bytes_in_use to -1 to make sure it gets set to something else
//   args.bytes_in_use = -1;
//   PJRT_Error* error = api_->PJRT_Device_MemoryStats(&args);
//   ASSERT_EQ(error, nullptr);
//   CHECK_GE(args.bytes_in_use, 0);
// }

// // ------------------------------- Executables
// ---------------------------------

// std::unique_ptr<PJRT_Executable, ::pjrt::PJRT_ExecutableDeleter>
// GetExecutable(
//     PJRT_LoadedExecutable* loaded_executable, const PJRT_Api* api) {
//   PJRT_LoadedExecutable_GetExecutable_Args args;
//   args.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   args.loaded_executable = loaded_executable;
//   args.executable = nullptr;
//   ::pjrt::LogFatalIfPjrtError(api->PJRT_LoadedExecutable_GetExecutable(&args),
//                               api);
//   return {args.executable, ::pjrt::MakeExecutableDeleter(api)};
// }

// class PjrtCApiTpuExecutableTest : public PjrtCApiTpuTest {
//  protected:
//   std::unique_ptr<PJRT_LoadedExecutable,
//   ::pjrt::PJRT_LoadedExecutableDeleter>
//       executable_;

//   void SetUp() override {
//     PjrtCApiTpuTest::SetUp();
//     executable_ = create_executable(api_, client_);
//   }

//   void TearDown() override {
//     executable_.reset();
//     PjrtCApiTpuTest::TearDown();
//   }

//   absl::flat_hash_map<std::string, PjRtValueType>
//   CreateMapFromGetCostAnalysisOutput(
//       const PJRT_LoadedExecutable_GetCostAnalysis_Args& args) {
//     absl::flat_hash_map<std::string, PjRtValueType> output_map;
//     return ::pjrt::ConvertFromPjRtNamedValueList(args.properties,
//                                                  args.num_properties);
//   }

//   PJRT_LoadedExecutable* BuildSingleDeviceProgramExecutable(
//       absl::string_view hlo_string) {
//     PJRT_Client_Compile_Args compile_args = PJRT_Client_Compile_Args{
//         .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
//         .priv = nullptr,
//         .client = client_,
//     };
//     std::string options_str = BuildSingleDeviceCompileOptionStr();
//     compile_args.compile_options = options_str.c_str();
//     compile_args.compile_options_size = options_str.size();

//     absl::StatusOr<std::unique_ptr<HloModule>> hlo_module =
//         xla::ParseAndReturnUnverifiedModule(hlo_string);
//     TF_CHECK_OK(hlo_module.status());
//     std::string module_str =
//     hlo_module->get()->ToProto().SerializeAsString();

//     std::string format(::pjrt::kHloFormat);
//     PJRT_Program program = PJRT_Program{
//         .struct_size = PJRT_Program_STRUCT_SIZE,
//         .priv = nullptr,
//         .code = module_str.data(),
//         .code_size = module_str.size(),
//         .format = format.c_str(),
//         .format_size = format.size(),
//     };
//     compile_args.program = std::move(&program);
//     PJRT_Error* error = api_->PJRT_Client_Compile(&compile_args);
//     ::pjrt::LogFatalIfPjrtError(error, api_);
//     return compile_args.executable;
//   }

//   void VerifyReturnedFuture(PJRT_Event* device_complete_event) {
//     PJRT_Event_Await_Args await_args;
//     await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
//     await_args.priv = nullptr;
//     await_args.event = device_complete_event;
//     ::pjrt::LogFatalIfPjrtError(api_->PJRT_Event_Await(&await_args), api_);

//     PJRT_Event_IsReady_Args ready_args;
//     ready_args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
//     ready_args.priv = nullptr;
//     ready_args.event = device_complete_event;
//     ::pjrt::LogFatalIfPjrtError(api_->PJRT_Event_IsReady(&ready_args), api_);
//     EXPECT_TRUE(ready_args.is_ready);

//     PJRT_Event_Destroy_Args destroy_args;
//     destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
//     destroy_args.priv = nullptr;
//     destroy_args.event = device_complete_event;
//     ::pjrt::LogFatalIfPjrtError(api_->PJRT_Event_Destroy(&destroy_args),
//     api_);
//   }

//   void TestExecuteSharded(bool create_device_completion_event,
//                           const DeviceAssignment& device_assignment) {
//     PJRT_Client_Compile_Args args_compile = PJRT_Client_Compile_Args{
//         .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
//         .priv = nullptr,
//         .client = client_,
//     };
//     ExecutableBuildOptions build_options;
//     build_options.set_device_ordinal(0);
//     build_options.set_num_replicas(device_assignment.replica_count());
//     build_options.set_device_assignment(device_assignment);
//     CompileOptions options;
//     options.executable_build_options = build_options;
//     CompileOptionsProto options_proto = options.ToProto().value();
//     std::string options_str = options_proto.SerializeAsString();
//     args_compile.compile_options = options_str.c_str();
//     args_compile.compile_options_size = options_str.size();

//     std::string format(::pjrt::kMlirFormat);
//     std::string program_code{module_add_one};
//     PJRT_Program program = PJRT_Program{
//         .struct_size = PJRT_Program_STRUCT_SIZE,
//         .priv = nullptr,
//         .code = program_code.data(),
//         .code_size = program_code.length(),
//         .format = format.c_str(),
//         .format_size = format.size(),
//     };
//     args_compile.program = &program;

//     PJRT_Error* error_compile = api_->PJRT_Client_Compile(&args_compile);
//     ::pjrt::LogFatalIfPjrtError(error_compile, api_);
//     PJRT_LoadedExecutable_Execute_Args args_execute;
//     args_execute.struct_size =
//     PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE; args_execute.priv =
//     nullptr; args_execute.executable = args_compile.executable;
//     PJRT_ExecuteOptions c_options;
//     c_options.num_send_ops = 0;
//     c_options.num_recv_ops = 0;
//     args_execute.options = &c_options;
//     args_execute.options->struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
//     args_execute.options->launch_id = 0;
//     args_execute.num_devices = 1;
//     args_execute.num_args = 1;
//     PJRT_Event* returned_future;
//     if (create_device_completion_event) {
//       args_execute.device_complete_events = &returned_future;
//     } else {
//       args_execute.device_complete_events = nullptr;
//     }

//     // Allocates memory for output.
//     int num_outputs_per_device = 1;
//     std::vector<PJRT_Buffer*> output_list(num_outputs_per_device);
//     std::vector<PJRT_Buffer**> output_lists{output_list.data()};
//     args_execute.output_lists = output_lists.data();

//     auto devices_to_execute = GetClientAddressableDevices();

//     // executing on all devices in device_assignment.
//     for (int device_id = 0; device_id < device_assignment.replica_count();
//          ++device_id) {
//       auto buffer = create_buffer(devices_to_execute[device_id]).first;
//       std::vector<PJRT_Buffer*> argument_list{buffer.get()};
//       std::vector<PJRT_Buffer**> argument_lists{argument_list.data()};
//       args_execute.argument_lists = argument_lists.data();
//       args_execute.execute_device = devices_to_execute[device_id];

//       PJRT_Error* error_execute =
//           api_->PJRT_LoadedExecutable_Execute(&args_execute);
//       ASSERT_EQ(error_execute, nullptr);

//       PJRT_Buffer* result_buffer = args_execute.output_lists[0][0];
//       ASSERT_OK_AND_ASSIGN(float result, GetProgramResult(result_buffer));
//       // What the executable does is to add one to the input. The input
//       buffer
//       // is 41, therefore the expected output is 42.
//       EXPECT_EQ(result, 42);
//       if (create_device_completion_event) {
//         VerifyReturnedFuture(args_execute.device_complete_events[0]);
//       }

//       // Clean up.
//       auto buffer_deleter = ::pjrt::MakeBufferDeleter(api_);
//       for (int i = 0; i < args_execute.num_devices; ++i) {
//         for (int j = 0; j < num_outputs_per_device; ++j) {
//           buffer_deleter(args_execute.output_lists[i][j]);
//         }
//       }
//     }

//     destroy_executable(args_compile.executable, api_);
//   }

//   void TestExecutePortable(bool create_device_completion_event) {
//     PJRT_Client_Compile_Args args_compile = PJRT_Client_Compile_Args{
//         .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
//         .priv = nullptr,
//         .client = client_,
//     };
//     ExecutableBuildOptions build_options;
//     build_options.set_device_ordinal(0);
//     build_options.set_num_replicas(1);
//     CompileOptions options;
//     options.executable_build_options = build_options;
//     options.compile_portable_executable = true;
//     CompileOptionsProto options_proto = options.ToProto().value();
//     std::string options_str = options_proto.SerializeAsString();
//     args_compile.compile_options = options_str.c_str();
//     args_compile.compile_options_size = options_str.size();

//     std::string format(::pjrt::kMlirFormat);
//     std::string program_code{module_add_one};
//     PJRT_Program program = PJRT_Program{
//         .struct_size = PJRT_Program_STRUCT_SIZE,
//         .priv = nullptr,
//         .code = program_code.data(),
//         .code_size = program_code.length(),
//         .format = format.c_str(),
//         .format_size = format.size(),
//     };
//     args_compile.program = &program;

//     PJRT_Error* error_compile = api_->PJRT_Client_Compile(&args_compile);
//     ::pjrt::LogFatalIfPjrtError(error_compile, api_);
//     PJRT_LoadedExecutable_Execute_Args args_execute;
//     args_execute.struct_size =
//     PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE; args_execute.priv =
//     nullptr; args_execute.executable = args_compile.executable;
//     PJRT_ExecuteOptions c_options;
//     c_options.num_send_ops = 0;
//     c_options.num_recv_ops = 0;
//     args_execute.options = &c_options;
//     args_execute.options->struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
//     args_execute.options->launch_id = 0;
//     args_execute.num_devices = 1;
//     args_execute.num_args = 1;
//     PJRT_Event* returned_future;
//     if (create_device_completion_event) {
//       args_execute.device_complete_events = &returned_future;
//     } else {
//       args_execute.device_complete_events = nullptr;
//     }

//     // Allocates memory for output.
//     int num_outputs_per_device = 1;
//     std::vector<PJRT_Buffer*> output_list(num_outputs_per_device);
//     std::vector<PJRT_Buffer**> output_lists{output_list.data()};
//     args_execute.output_lists = output_lists.data();

//     auto devices_to_execute = GetClientAddressableDevices();
//     int device_index;
//     // Use a non-default device to test ExecutePortable if there are more
//     than
//     // one devices.
//     if (devices_to_execute.size() > 1) {
//       device_index = 1;
//     } else {
//       device_index = 0;
//     }
//     auto buffer = create_buffer(devices_to_execute[device_index]).first;
//     std::vector<PJRT_Buffer*> argument_list{buffer.get()};
//     std::vector<PJRT_Buffer**> argument_lists{argument_list.data()};
//     args_execute.argument_lists = argument_lists.data();
//     args_execute.execute_device = devices_to_execute[device_index];

//     PJRT_Error* error_execute =
//         api_->PJRT_LoadedExecutable_Execute(&args_execute);
//     ASSERT_EQ(error_execute, nullptr);

//     PJRT_Buffer* result_buffer = args_execute.output_lists[0][0];
//     ASSERT_OK_AND_ASSIGN(float result, GetProgramResult(result_buffer));
//     // What the executable does is to add one to the input. The input buffer
//     is
//     // 41, therefore the expected output is 42.
//     EXPECT_EQ(result, 42);
//     if (create_device_completion_event) {
//       VerifyReturnedFuture(args_execute.device_complete_events[0]);
//     }

//     // Clean up.
//     auto buffer_deleter = ::pjrt::MakeBufferDeleter(api_);
//     for (int i = 0; i < args_execute.num_devices; ++i) {
//       for (int j = 0; j < num_outputs_per_device; ++j) {
//         buffer_deleter(args_execute.output_lists[i][j]);
//       }
//     }
//     destroy_executable(args_compile.executable, api_);
//   }
// };

// TEST_F(PjrtCApiTpuExecutableTest, ExecutableName) {
//   PJRT_Executable_Name_Args args;
//   args.struct_size = PJRT_Executable_Name_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   auto executable = GetExecutable(executable_.get(), api_);
//   args.executable = executable.get();
//   PJRT_Error* error = api_->PJRT_Executable_Name(&args);
//   ASSERT_EQ(error, nullptr);
//   absl::string_view executable_name(args.executable_name,
//                                     args.executable_name_size);

//   using ::testing::StartsWith;
//   ASSERT_THAT(executable_name, StartsWith(kExecutableName));
// }

// TEST_F(PjrtCApiTpuExecutableTest, ExecutableNumReplicas) {
//   PJRT_Executable_NumReplicas_Args args;
//   args.struct_size = PJRT_Executable_NumReplicas_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   auto executable = GetExecutable(executable_.get(), api_);
//   args.executable = executable.get();
//   PJRT_Error* error = api_->PJRT_Executable_NumReplicas(&args);

//   ASSERT_EQ(error, nullptr);
//   ASSERT_EQ(args.num_replicas, 1);
// }

// TEST_F(PjrtCApiTpuExecutableTest, ExecutableNumPartitions) {
//   PJRT_Executable_NumPartitions_Args args;
//   args.struct_size = PJRT_Executable_NumPartitions_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   auto executable = GetExecutable(executable_.get(), api_);
//   args.executable = executable.get();
//   PJRT_Error* error = api_->PJRT_Executable_NumPartitions(&args);

//   ASSERT_EQ(error, nullptr);
//   ASSERT_EQ(args.num_partitions, 1);
// }

// TEST_F(PjrtCApiTpuExecutableTest, AddressableDevices) {
//   PJRT_LoadedExecutable_AddressableDevices_Args args;
//   args.struct_size =
//   PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE; args.priv =
//   nullptr; args.executable = executable_.get(); PJRT_Error* error =
//   api_->PJRT_LoadedExecutable_AddressableDevices(&args); EXPECT_EQ(error,
//   nullptr); absl::Span<PJRT_Device*> addressable_devices =
//       absl::MakeSpan(args.addressable_devices, args.num_addressable_devices);

//   EXPECT_EQ(addressable_devices.size(),
//             executable_->get()->addressable_devices().size());
//   ASSERT_FALSE(addressable_devices.empty());
//   for (auto& device : addressable_devices) {
//     ASSERT_TRUE(this->IsValidDeviceId(device));
//   }

//   absl::Span<PJRT_Device*> client_devices = GetClientAddressableDevices();
//   for (auto& addressable_device : addressable_devices) {
//     ASSERT_THAT(client_devices, Contains(addressable_device));
//   }
// }

// TEST_F(PjrtCApiTpuExecutableTest, OptimizedProgram) {
//   PJRT_Executable_OptimizedProgram_Args args;
//   args.struct_size = PJRT_Executable_OptimizedProgram_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   auto executable = GetExecutable(executable_.get(), api_);
//   args.executable = executable.get();
//   auto program = std::make_unique<PJRT_Program>();
//   program->struct_size = PJRT_Program_STRUCT_SIZE;
//   program->priv = nullptr;
//   program->code = nullptr;
//   absl::string_view program_format = "hlo_with_config";
//   program->format = program_format.data();
//   program->format_size = program_format.length();
//   args.program = program.get();

//   // The first call to `PJRT_Executable_OptimizedProgram` populates
//   // `program->code_size`, telling us how large a string to allocate
//   auto error = ToUniquePtr(api_->PJRT_Executable_OptimizedProgram(&args));
//   EXPECT_EQ(error, nullptr);
//   constexpr size_t TWO_GIBIBYTES = 2ull * 1024 * 1024 * 1024;
//   ASSERT_LT(program->code_size, TWO_GIBIBYTES);
//   std::string code(args.program->code_size, 'a');
//   program->code = code.data();

//   // The second call to `PJRT_Executable_OptimizedProgram` assigns the
//   // serialized program to `program->code` (and thus `code`).
//   error = ToUniquePtr(api_->PJRT_Executable_OptimizedProgram(&args));
//   EXPECT_EQ(error, nullptr) << ::pjrt::GetPjrtErrorMessage(error.get(),
//   api_);

//   // Use the PJRT C++ API to create an expected proto
//   ASSERT_OK_AND_ASSIGN(std::vector<std::shared_ptr<HloModule>> cpp_modules,
//                        executable_->executable->GetHloModules());
//   ASSERT_OK_AND_ASSIGN(HloModuleProtoWithConfig expected_proto,
//                        cpp_modules[0]->ToProtoWithConfig());

//   HloModuleProtoWithConfig deserialized_proto;
//   ASSERT_TRUE(deserialized_proto.ParseFromString(code));

//   // Make sure the protos output by the C++ and C APIs are equivalent
//   google::protobuf::util::MessageDifferencer diff;
//   diff.set_message_field_comparison(
//       google::protobuf::util::MessageDifferencer::EQUIVALENT);
//   EXPECT_TRUE(diff.Equals(expected_proto, deserialized_proto));
// }

// TEST_F(PjrtCApiTpuExecutableTest, ExecutableDeletion) {
//   PJRT_LoadedExecutable_IsDeleted_Args is_deleted_args;
//   is_deleted_args.struct_size =
//       PJRT_LoadedExecutable_IsDeleted_Args_STRUCT_SIZE;
//   is_deleted_args.priv = nullptr;
//   is_deleted_args.executable = executable_.get();
//   PJRT_Error* is_deleted_error =
//       api_->PJRT_LoadedExecutable_IsDeleted(&is_deleted_args);
//   ASSERT_EQ(is_deleted_error, nullptr);
//   ASSERT_FALSE(is_deleted_args.is_deleted);

//   PJRT_LoadedExecutable_Delete_Args delete_args;
//   delete_args.struct_size = PJRT_LoadedExecutable_Delete_Args_STRUCT_SIZE;
//   delete_args.priv = nullptr;
//   delete_args.executable = executable_.get();
//   PJRT_Error* delete_error =
//   api_->PJRT_LoadedExecutable_Delete(&delete_args); ASSERT_EQ(delete_error,
//   nullptr);

//   is_deleted_error = api_->PJRT_LoadedExecutable_IsDeleted(&is_deleted_args);
//   ASSERT_EQ(is_deleted_error, nullptr);
//   ASSERT_TRUE(is_deleted_args.is_deleted);
// }

// TEST_F(PjrtCApiTpuExecutableTest, Execute) {
//   // Run the default executable created in PjrtCApiTpuExecutableTest:SetUp
//   and
//   // assign its output
//   ASSERT_OK_AND_ASSIGN(float result,
//                        RunScalarExecutableAndGetResult(executable_.get()));
//   // What this executable does is to add one to the input. The input buffer
//   is
//   // 41, therefore the expected output is 42.
//   EXPECT_EQ(result, 42);
// }

// TEST_F(PjrtCApiTpuExecutableTest, ExecuteSharded) {
//   DeviceAssignment device_assignment(2, 1);
//   device_assignment(0, 0) = 0;
//   device_assignment(1, 0) = 1;
//   TestExecuteSharded(/*create_device_completion_event=*/false,
//                      device_assignment);
// }

// TEST_F(PjrtCApiTpuExecutableTest, ExecuteShardedWithReturnedFuture) {
//   DeviceAssignment device_assignment(2, 1);
//   device_assignment(0, 0) = 0;
//   device_assignment(1, 0) = 1;
//   TestExecuteSharded(/*create_device_completion_event=*/true,
//                      device_assignment);
// }

// TEST_F(PjrtCApiTpuExecutableTest, ExecuteShardedOneReplica) {
//   DeviceAssignment device_assignment(1, 1);
//   device_assignment(0, 0) = 0;
//   TestExecuteSharded(/*create_device_completion_event=*/false,
//                      device_assignment);
// }

// TEST_F(PjrtCApiTpuExecutableTest, ExecutePortable) {
//   TestExecutePortable(/*create_device_completion_event=*/false);
// }

// TEST_F(PjrtCApiTpuExecutableTest, ExecutePortableWithReturnedFuture) {
//   TestExecutePortable(/*create_device_completion_event=*/true);
// }

// TEST_F(PjrtCApiTpuExecutableTest, ExecuteWithReturnedFutures) {
//   PJRT_LoadedExecutable_Execute_Args args;
//   args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   args.executable = executable_.get();
//   PJRT_ExecuteOptions c_options;
//   c_options.num_send_ops = 0;
//   c_options.num_recv_ops = 0;
//   args.options = &c_options;
//   args.options->struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
//   args.options->launch_id = 0;
//   args.num_devices = 1;
//   args.num_args = 1;
//   std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> buffer =
//       create_buffer().first;
//   std::vector<PJRT_Buffer*> argument_list{buffer.get()};
//   std::vector<PJRT_Buffer**> argument_lists{argument_list.data()};
//   args.argument_lists = argument_lists.data();
//   std::vector<PJRT_Event*> returned_futures(args.num_devices);
//   args.device_complete_events = returned_futures.data();
//   args.execute_device = nullptr;
//   // Allocates memory for output.
//   int num_outputs_per_device = 1;
//   std::vector<PJRT_Buffer*> output_list(num_outputs_per_device);
//   std::vector<PJRT_Buffer**> output_lists{output_list.data()};
//   args.output_lists = output_lists.data();

//   PJRT_Error* error = api_->PJRT_LoadedExecutable_Execute(&args);
//   ASSERT_EQ(error, nullptr);

//   // Each future should be ready after device execution is complete.
//   for (int i = 0; i < args.num_devices; ++i) {
//     VerifyReturnedFuture(args.device_complete_events[i]);
//   }

//   // Clean up.
//   auto buffer_deleter = ::pjrt::MakeBufferDeleter(api_);
//   for (int i = 0; i < args.num_devices; ++i) {
//     for (int j = 0; j < num_outputs_per_device; ++j) {
//       buffer_deleter(args.output_lists[i][j]);
//     }
//   }
// }

// TEST_F(PjrtCApiTpuExecutableTest, ExecuteInputArgumentSizeExceedNumDevice) {
//   PJRT_LoadedExecutable_Execute_Args args;
//   args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   args.executable = executable_.get();
//   std::unique_ptr<PJRT_ExecuteOptions> c_options(new PJRT_ExecuteOptions);
//   c_options->num_send_ops = 0;
//   c_options->num_recv_ops = 0;
//   args.options = c_options.get();
//   args.options->struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
//   args.options->launch_id = 0;
//   // The number of addressable devices of executable_ is 1.
//   args.num_devices = 2;
//   args.num_args = 1;
//   args.output_lists = nullptr;
//   args.device_complete_events = nullptr;
//   auto buffer_and_event_1 = create_buffer();
//   auto buffer_and_event_2 = create_buffer();
//   std::vector<PJRT_Buffer*> argument_list_1{buffer_and_event_1.first.get()};
//   std::vector<PJRT_Buffer*> argument_list_2{buffer_and_event_2.first.get()};
//   std::vector<PJRT_Buffer**> argument_lists{argument_list_1.data(),
//                                             argument_list_2.data()};
//   args.argument_lists = argument_lists.data();
//   args.execute_device = nullptr;

//   PJRT_Error* error = api_->PJRT_LoadedExecutable_Execute(&args);

//   absl::Status status = ::pjrt::PjrtErrorToStatus(error, api_);
//   EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
//   EXPECT_EQ(status.error_message(),
//             "Attempted to execute with 2 argument lists when local device "
//             "count is 1 (total replica count: 1, partition count: 1)");

//   // Clean up.
//   EXPECT_OK(buffer_and_event_1.second.Await());
//   EXPECT_OK(buffer_and_event_2.second.Await());
//   ::pjrt::MakeErrorDeleter(api_)(error);
// }

// TEST_F(PjrtCApiTpuExecutableTest, NumOutputsSingle) {
//   PJRT_Executable_NumOutputs_Args args;
//   args.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   auto executable = GetExecutable(executable_.get(), api_);
//   args.executable = executable.get();
//   PJRT_Error* error = api_->PJRT_Executable_NumOutputs(&args);
//   ASSERT_EQ(error, nullptr);
//   EXPECT_EQ(args.num_outputs, 1);
// }

// TEST_F(PjrtCApiTpuExecutableTest, NumOutputsTuple) {
//   XlaBuilder builder(std::string{kExecutableName});
//   Shape s = ShapeUtil::MakeShape(F32, {});
//   auto inp = Parameter(&builder, 0, s, "input");
//   auto one = ConstantR0<float>(&builder, 1.0f);
//   auto incremented = Add(inp, one);
//   std::vector v = {incremented, inp};
//   auto tuple = Tuple(&builder, v);
//   auto computation = builder.Build(tuple).value();
//   auto pjrt_executable = create_executable(api_, client_, computation);
//   PJRT_Executable_NumOutputs_Args args;
//   args.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   auto base_executable = GetExecutable(pjrt_executable.get(), api_);
//   args.executable = base_executable.get();
//   PJRT_Error* error = api_->PJRT_Executable_NumOutputs(&args);
//   ASSERT_EQ(error, nullptr);
//   EXPECT_EQ(args.num_outputs, 2);
// }

// TEST_F(PjrtCApiTpuExecutableTest, SizeOfGeneratedCodeInBytes) {
//   // Call the function directly to get a reference size, check that it's not
//   // zero.
//   int64_t direct_call_size =
//       executable_->executable->SizeOfGeneratedCodeInBytes();
//   ASSERT_NE(direct_call_size, 0);

//   // Call the function through PJRT C API interface, check that it's not
//   zero. auto executable = GetExecutable(executable_.get(), api_);
//   PJRT_Executable_SizeOfGeneratedCodeInBytes_Args args{
//       .struct_size =
//           PJRT_Executable_SizeOfGeneratedCodeInBytes_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .executable = executable.get(),
//   };
//   PJRT_Error* error =
//   api_->PJRT_Executable_SizeOfGeneratedCodeInBytes(&args);
//   ::pjrt::LogFatalIfPjrtError(error, api_);
//   ASSERT_EQ(error, nullptr);
//   ASSERT_NE(args.size_in_bytes, 0);

//   // Confirm that size in bytes returned from both calls are the same.
//   ASSERT_EQ(direct_call_size, args.size_in_bytes);
// }

// TEST_F(PjrtCApiTpuExecutableTest, GetCostAnalysis) {
//   // Call GetCostAnalysis directly
//   auto program_cost_properties = executable_->get()->GetCostAnalysis();
//   ASSERT_TRUE(program_cost_properties.ok());
//   ASSERT_GT(program_cost_properties.value().size(), 0);

//   // Call PJRT C API
//   PJRT_LoadedExecutable_GetCostAnalysis_Args args{
//       .struct_size = PJRT_LoadedExecutable_GetCostAnalysis_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .executable = executable_.get(),
//       .num_properties = 0,
//       .properties = nullptr};
//   PJRT_Error* error = api_->PJRT_LoadedExecutable_GetCostAnalysis(&args);
//   ::pjrt::LogFatalIfPjrtError(error, api_);
//   ASSERT_EQ(error, nullptr);
//   LOG(INFO) << "PJRT_LoadedExecutable_GetCostAnalysis returned "
//             << args.num_properties << " properties.";
//   ASSERT_GT(args.num_properties, 0);

//   // Verify results from local call and C API are the same
//   auto output_map = CreateMapFromGetCostAnalysisOutput(args);
//   ASSERT_EQ(program_cost_properties.value(), output_map);

//   // Call PJRT C API again (which returns cached value)
//   // to confirm results are the same
//   PJRT_LoadedExecutable_GetCostAnalysis_Args second_call_args{
//       .struct_size = PJRT_LoadedExecutable_GetCostAnalysis_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .executable = executable_.get(),
//       .num_properties = 0,
//       .properties = nullptr};
//   error = api_->PJRT_LoadedExecutable_GetCostAnalysis(&second_call_args);
//   ::pjrt::LogFatalIfPjrtError(error, api_);
//   ASSERT_EQ(error, nullptr);
//   LOG(INFO) << "Second PJRT_LoadedExecutable_GetCostAnalysis call returned "
//             << args.num_properties << " properties.";
//   ASSERT_GT(args.num_properties, 0);

//   auto second_call_output_map = CreateMapFromGetCostAnalysisOutput(args);
//   ASSERT_EQ(program_cost_properties.value(), second_call_output_map);
// }

// std::unique_ptr<PJRT_LoadedExecutable, ::pjrt::PJRT_LoadedExecutableDeleter>
// SerializeAndLoad(const PJRT_Api* c_api, PJRT_Client* client,
//                  PJRT_Executable* executable) {
//   auto serialization_args = PJRT_Executable_Serialize_Args{
//       .struct_size = PJRT_Executable_Serialize_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .executable = executable,
//       .serialized_executable = nullptr,
//   };

//   ::pjrt::LogFatalIfPjrtError(
//       c_api->PJRT_Executable_Serialize(&serialization_args), c_api);
//   PJRT_SerializedExecutable* c_serialized_exec =
//       serialization_args.serialized_executable;

//   auto data_args = PJRT_SerializedExecutable_Data_Args{
//       .struct_size = PJRT_SerializedExecutable_Data_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .serialized_executable = c_serialized_exec,
//       .data = nullptr,
//       .data_size = 0,
//   };

//   ::pjrt::LogFatalIfPjrtError(c_api->PJRT_SerializedExecutable_Data(&data_args),
//                               c_api);

//   CHECK_NE(data_args.data, nullptr);
//   CHECK_GT(data_args.data_size, 0);
//   absl::string_view serialized_exec(data_args.data, data_args.data_size);

//   // Create a second executable with C API
//   `PJRT_Executable_DeserializeAndLoad` auto deserialize_args =
//   PJRT_Executable_DeserializeAndLoad_Args{
//       .struct_size = PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .client = client,
//       .serialized_executable = serialized_exec.data(),
//       .serialized_executable_size = serialized_exec.length(),
//       .loaded_executable = nullptr,
//   };

//   ::pjrt::LogFatalIfPjrtError(
//       c_api->PJRT_Executable_DeserializeAndLoad(&deserialize_args), c_api);
//   CHECK_NE(deserialize_args.loaded_executable, nullptr);

//   auto destroy_args = PJRT_SerializedExecutable_Destroy_Args{
//       .struct_size = PJRT_SerializedExecutable_Destroy_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .serialized_executable = c_serialized_exec,
//   };
//   ::pjrt::LogFatalIfPjrtError(
//       c_api->PJRT_SerializedExecutable_Destroy(&destroy_args), c_api);

//   return {deserialize_args.loaded_executable,
//           ::pjrt::MakeLoadedExecutableDeleter(c_api)};
// }

// TEST_F(PjrtCApiTpuExecutableTest,
//        SerializedAndDeserializedIsEquivalentToOriginal) {
//   auto executable = GetExecutable(executable_.get(), api_);

//   auto reconstructed_exec = SerializeAndLoad(api_, client_,
//   executable.get());

//   // Run the default executable created in PjrtCApiTpuExecutableTest:SetUp
//   and
//   // assign its output
//   ASSERT_OK_AND_ASSIGN(float result_expected,
//                        RunScalarExecutableAndGetResult(executable_.get()));
//   // Run our reconstructed executable and assign its output
//   ASSERT_OK_AND_ASSIGN(
//       float result_of_reserialization,
//       RunScalarExecutableAndGetResult(reconstructed_exec.get()));

//   EXPECT_EQ(result_expected, result_of_reserialization);
// }

// // std::function version of PJRT_SendCallback
// using SendCallbackFunction =
//     std::function<PJRT_Error*(PJRT_Chunk*, PJRT_CallbackError*, size_t,
//     bool)>;
// // std::function version of PJRT_RecvCallback
// using RecvCallbackFunction = std::function<void(PJRT_CopyToDeviceStream*)>;

// // Wraps original `xla::SendCallback` inside `PJRT_Callback` using
// // 1) void* `user_arg` to capture `cpp_send_callback.callback`
// (std::function)
// // 2) `PJRT_SendCallback` function pointer, which reinterprets and calls
// // `user_arg` to call `cpp_send_callback.callback` function.
// //
// // TODO(yeounoh) move this to pjrt_c_api_helpers after implementing C API for
// // the opaque types `PJRT_Chunk` and `PJRT_CopyToDeviceStream`.
// PJRT_SendCallbackInfo CppSendCallbackToCSendCallback(
//     xla::SendCallback cpp_send_callback,
//     SendCallbackFunction* send_callback_function) {
//   return PJRT_SendCallbackInfo{
//       .channel_id = cpp_send_callback.channel_id,
//       // this is the void* user_arg to capture `cpp_send_callback.callback`
//       .user_arg = send_callback_function,
//       // this is the function pointer, PJRT_SendCallback
//       .send_callback = [](PJRT_Chunk* chunk, PJRT_CallbackError*
//       callback_error,
//                           size_t total_size_in_bytes, bool done,
//                           void* user_arg) -> PJRT_Error* {
//         // PJRT_SendCallback, `send_callback` is internal C interface
//         callback
//         // representation that cpatures the client C++ callback in void*
//         // `user_arg` and reinterprets in the lower-level runtime for
//         execution.
//         // `user_arg` captures `send_callback_function` which is
//         // SendCallbackFunction*.
//         SendCallbackFunction* send_callback =
//             reinterpret_cast<SendCallbackFunction*>(user_arg);
//         return (*send_callback)(chunk, callback_error, total_size_in_bytes,
//                                 done);
//       }};
// }

// // Wraps original `xla::RecvCallback` inside `PJRT_Callback` using
// // 1) void* `user_arg` to capture `cpp_send_callback.callback`
// (std::function)
// // 2) `PJRT_RecvCallback` function pointer, which reinterprets and calls
// // `user_arg` to call `cpp_recv_callback.callback` function.
// //
// // TODO(yeounoh) move this to pjrt_c_api_helpers after implementing C API for
// // the opaque types `PJRT_Chunk` and `PJRT_CopyToDeviceStream`.
// PJRT_RecvCallbackInfo CppRecvCallbackToCRecvCallback(
//     xla::RecvCallback cpp_recv_callback,
//     RecvCallbackFunction* recv_callback_function) {
//   return PJRT_RecvCallbackInfo{
//       .channel_id = cpp_recv_callback.channel_id,
//       // this is the void* user_arg to capture `cpp_recv_callback.callback`
//       .user_arg = recv_callback_function,
//       // this is the function pointer, PJRT_RecvCallback
//       .recv_callback = [](PJRT_CopyToDeviceStream* stream, void* user_arg) {
//         // PJRT_RecvCallback, `recv_callback` is internal C interface
//         callback
//         // representation that cpatures the client C++ callback in void*
//         // `user_arg` and reinterprets in the lower-level runtime for
//         execution.
//         // `user_arg` captures `recv_callback_function` which is
//         // RecvCallbackFunction*.
//         auto* recv_callback =
//             reinterpret_cast<std::function<void(PJRT_CopyToDeviceStream*)>*>(
//                 user_arg);
//         (*recv_callback)(stream);
//       }};
// }

// TEST_F(PjrtCApiTpuExecutableTest, HostCallbackSendOnly) {
//   PJRT_LoadedExecutable_Execute_Args execute_args;
//   execute_args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
//   execute_args.priv = nullptr;
//   execute_args.executable =
//   BuildSingleDeviceProgramExecutable(kSendHloString); PJRT_ExecuteOptions
//   c_options; c_options.num_send_ops = 0; c_options.num_recv_ops = 0;
//   execute_args.options = &c_options;
//   execute_args.options->struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
//   execute_args.options->launch_id = 0;
//   execute_args.num_devices = 1;
//   execute_args.num_args = 0;
//   execute_args.argument_lists = nullptr;

//   // Allocates memory for output
//   int num_outputs_per_device = 1;
//   std::vector<PJRT_Buffer*> output_list(num_outputs_per_device);
//   std::vector<PJRT_Buffer**> output_lists{output_list.data()};
//   execute_args.output_lists = output_lists.data();

//   // Populates send callbacks
//   absl::Notification notification;
//   xla::SendCallback cpp_send_callback = xla::SendCallback{
//       .channel_id = 1,
//       .callback = [&notification](const PjRtTransferMetadata& metadata,
//                                   PjRtChunk input, size_t
//                                   total_size_in_bytes, bool done) ->
//                                   absl::Status {
//         CHECK(done);
//         notification.Notify();
//         return xla::OkStatus();
//       }};
//   SendCallbackFunction send_callback_function(
//       [&send_callback = cpp_send_callback.callback](
//           PJRT_Chunk* chunk, PJRT_CallbackError* callback_error,
//           size_t total_size_in_bytes, bool done) -> PJRT_Error* {
//         // PJRT C API doesn't support
//         // use_major_to_minor_data_layout_for_callbacks = false
//         xla::Shape dummy_shape;
//         absl::Status status =
//             send_callback(xla::PjRtTransferMetadata{dummy_shape},
//                           xla::PjRtChunk(chunk->data, chunk->size,
//                                          [deleter_arg = chunk->deleter_arg,
//                                           deleter = chunk->deleter](void*
//                                           ptr) {
//                                            deleter(ptr, deleter_arg);
//                                          }),
//                           total_size_in_bytes, done);
//         if (!status.ok()) {
//           return (*callback_error)(
//               ::pjrt::StatusCodeToPjrtErrorCode(status.code()),
//               status.error_message().data(), status.error_message().size());
//         }
//         return nullptr;
//       });

//   std::vector<PJRT_SendCallbackInfo> c_send_callbacks{
//       CppSendCallbackToCSendCallback(cpp_send_callback,
//                                      &send_callback_function)};
//   std::vector<PJRT_SendCallbackInfo*> c_send_callbacks_per_device;
//   c_send_callbacks_per_device.reserve(1);
//   c_send_callbacks_per_device.push_back(c_send_callbacks.data());
//   execute_args.options->send_callbacks = c_send_callbacks_per_device.data();

//   // Allocates future events to be notifed after device execution
//   std::vector<PJRT_Event*> returned_futures(execute_args.num_devices);
//   execute_args.device_complete_events = returned_futures.data();
//   execute_args.execute_device = nullptr;

//   // Should notify after the device execution
//   execute_args.options->num_send_ops = 1;
//   execute_args.options->num_recv_ops = 0;
//   ::pjrt::LogFatalIfPjrtError(
//       api_->PJRT_LoadedExecutable_Execute(&execute_args), api_);

//   PJRT_Event_Await_Args await_args;
//   await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
//   await_args.priv = nullptr;
//   await_args.event = execute_args.device_complete_events[0];
//   ::pjrt::LogFatalIfPjrtError(api_->PJRT_Event_Await(&await_args), api_);
//   ASSERT_TRUE(notification.HasBeenNotified());

//   // Clean up.
//   ::pjrt::MakeEventDeleter(api_)(execute_args.device_complete_events[0]);
//   auto buffer_deleter = ::pjrt::MakeBufferDeleter(api_);
//   destroy_executable(execute_args.executable, api_);
//   for (int i = 0; i < execute_args.num_devices; ++i) {
//     for (int j = 0; j < num_outputs_per_device; ++j) {
//       buffer_deleter(execute_args.output_lists[i][j]);
//     }
//   }
// }

// TEST_F(PjrtCApiTpuExecutableTest, HostCallbackBasic) {
//   PJRT_LoadedExecutable_Execute_Args execute_args;
//   execute_args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
//   execute_args.priv = nullptr;
//   execute_args.executable =
//       BuildSingleDeviceProgramExecutable(kSendRecvHloString);
//   PJRT_ExecuteOptions c_options;
//   c_options.num_send_ops = 0;
//   c_options.num_recv_ops = 0;
//   execute_args.options = &c_options;
//   execute_args.options->struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
//   execute_args.options->launch_id = 0;
//   execute_args.num_devices = 1;
//   execute_args.num_args = 0;
//   execute_args.argument_lists = nullptr;
//   execute_args.device_complete_events = nullptr;
//   execute_args.execute_device = nullptr;

//   // Populate send/recv callbacks
//   thread::Channel<PjRtChunk> channel(1);
//   xla::SendCallback cpp_send_callback = xla::SendCallback{
//       .channel_id = 1,
//       .callback = [&channel](const PjRtTransferMetadata& metadata,
//                              PjRtChunk input, size_t total_size_in_bytes,
//                              bool done) -> absl::Status {
//         CHECK(done);
//         channel.writer()->Write(std::move(input));
//         return xla::OkStatus();
//       }};
//   SendCallbackFunction send_callback_function(
//       [&send_callback = cpp_send_callback.callback](
//           PJRT_Chunk* chunk, PJRT_CallbackError* callback_error,
//           size_t total_size_in_bytes, bool done) -> PJRT_Error* {
//         // PJRT C API doesn't support
//         // use_major_to_minor_data_layout_for_callbacks = false
//         xla::Shape dummy_shape;
//         absl::Status status =
//             send_callback(xla::PjRtTransferMetadata{dummy_shape},
//                           xla::PjRtChunk(chunk->data, chunk->size,
//                                          [deleter_arg = chunk->deleter_arg,
//                                           deleter = chunk->deleter](void*
//                                           ptr) {
//                                            deleter(ptr, deleter_arg);
//                                          }),
//                           total_size_in_bytes, done);
//         if (!status.ok()) {
//           return (*callback_error)(
//               ::pjrt::StatusCodeToPjrtErrorCode(status.code()),
//               status.error_message().data(), status.error_message().size());
//         }
//         return nullptr;
//       });

//   std::vector<PJRT_SendCallbackInfo> c_send_callbacks{
//       CppSendCallbackToCSendCallback(cpp_send_callback,
//                                      &send_callback_function)};
//   std::vector<PJRT_SendCallbackInfo*> c_send_callbacks_per_device;
//   c_send_callbacks_per_device.reserve(1);
//   c_send_callbacks_per_device.push_back(c_send_callbacks.data());
//   execute_args.options->send_callbacks = c_send_callbacks_per_device.data();
//   execute_args.options->num_send_ops = 1;

//   xla::RecvCallback cpp_recv_callback = xla::RecvCallback{
//       .channel_id = 2,
//       .callback = [&channel](
//                       const PjRtTransferMetadata& metadata,
//                       std::unique_ptr<CopyToDeviceStream> stream) -> void {
//         PjRtChunk output;
//         CHECK(channel.reader()->Read(&output));
//         stream->AddChunk(std::move(output)).OnReady([](absl::Status s) {
//           TF_CHECK_OK(s);
//         });
//       }};
//   RecvCallbackFunction recv_callback_function(
//       [&recv_callback = cpp_recv_callback.callback,
//        api = api_](PJRT_CopyToDeviceStream* stream) {
//         xla::Shape dummy_shape;
//         recv_callback(xla::PjRtTransferMetadata{dummy_shape},
//                       std::make_unique<CApiCopyToDeviceStream>(stream, api));
//       });

//   std::vector<PJRT_RecvCallbackInfo> c_recv_callbacks{
//       CppRecvCallbackToCRecvCallback(cpp_recv_callback,
//                                      &recv_callback_function)};
//   std::vector<PJRT_RecvCallbackInfo*> c_recv_callbacks_per_device;
//   c_recv_callbacks_per_device.push_back(c_recv_callbacks.data());
//   execute_args.options->recv_callbacks = c_recv_callbacks_per_device.data();
//   execute_args.options->num_recv_ops = 1;

//   // Allocates memory for output.
//   int num_outputs_per_device = 1;
//   std::vector<PJRT_Buffer*> output_list(num_outputs_per_device);
//   std::vector<PJRT_Buffer**> output_lists{output_list.data()};
//   execute_args.output_lists = output_lists.data();

//   ::pjrt::LogFatalIfPjrtError(
//       api_->PJRT_LoadedExecutable_Execute(&execute_args), api_);

//   // Check if the returned output is equal to {{1, 2}, {3, 4}} after the
//   // execution is complete.
//   auto output_buffer = execute_args.output_lists[0][0]->buffer.get();
//   // TODO(b/263882762): use `PJRT_BufferToHostBuffer` instead.
//   std::shared_ptr<xla::Literal> result = *output_buffer->ToLiteralSync();
//   EXPECT_TRUE(xla::LiteralTestUtil::Equal(
//       *result, xla::LiteralUtil::CreateR2({{1.0f, 2.0f}, {3.0f, 4.0f}})));

//   // Clean up.
//   destroy_executable(execute_args.executable, api_);
//   auto buffer_deleter = ::pjrt::MakeBufferDeleter(api_);
//   for (int i = 0; i < execute_args.num_devices; ++i) {
//     for (int j = 0; j < num_outputs_per_device; ++j) {
//       buffer_deleter(execute_args.output_lists[i][j]);
//     }
//   }
// }

// // ---------------------------------- Buffers
// ----------------------------------

// class PjrtCApiTpuBufferTest : public PjrtCApiTpuTest {
//  protected:
//   void SetUp() override {
//     PjrtCApiTpuTest::SetUp();
//     auto buffer_and_event = create_buffer();
//     buffer_ = std::move(buffer_and_event.first);
//     event_ = buffer_and_event.second;
//   }

//   void TearDown() override {
//     // event_ need to complete before the client is destroyed; otherwis there
//     is
//     // a data race between destroying the client and trying to access the
//     host
//     // context in the client for the callback afte host to device transfer is
//     // compeleted.
//     EXPECT_OK(event_.Await());
//     // buffer_ must be destroyed before the client is destroyed or else the
//     // unique_ptr for buffer_ will go out of scope causing
//     heap-use-after-free
//     // error.
//     buffer_.reset(nullptr);
//     PjrtCApiTpuTest::TearDown();
//   }

//   std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> buffer_;
//   PjRtFuture<absl::Status> event_;
// };

// TEST_F(PjrtCApiTpuBufferTest, IsDeleted) {
//   PJRT_Buffer_IsDeleted_Args is_deleted_args;
//   is_deleted_args.struct_size = PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE;
//   is_deleted_args.priv = nullptr;
//   is_deleted_args.buffer = buffer_.get();
//   PJRT_Error* is_deleted_error =
//   api_->PJRT_Buffer_IsDeleted(&is_deleted_args); ASSERT_EQ(is_deleted_error,
//   nullptr); ASSERT_FALSE(is_deleted_args.is_deleted);

//   PJRT_Buffer_Delete_Args delete_args;
//   delete_args.struct_size = PJRT_Buffer_Delete_Args_STRUCT_SIZE;
//   delete_args.priv = nullptr;
//   delete_args.buffer = buffer_.get();
//   PJRT_Error* delete_error = api_->PJRT_Buffer_Delete(&delete_args);
//   ASSERT_EQ(delete_error, nullptr);

//   is_deleted_error = api_->PJRT_Buffer_IsDeleted(&is_deleted_args);
//   ASSERT_EQ(is_deleted_error, nullptr);
//   ASSERT_TRUE(is_deleted_args.is_deleted);
// }

// TEST_F(PjrtCApiTpuBufferTest, GetOnDeviceSizeInBytes) {
//   PJRT_Buffer_OnDeviceSizeInBytes_Args args;
//   args.struct_size = PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   args.buffer = buffer_.get();
//   PJRT_Error* on_device_size_bytes_error =
//       api_->PJRT_Buffer_OnDeviceSizeInBytes(&args);

//   ASSERT_EQ(on_device_size_bytes_error, nullptr);
//   ASSERT_GT(args.on_device_size_in_bytes, 0);
// }

// TEST_F(PjrtCApiTpuBufferTest, CopyToDevice) {
//   absl::Span<PJRT_Device*> addressable_devices =
//   GetClientAddressableDevices(); if (addressable_devices.size() <= 1) {
//     GTEST_SKIP() << "CopyToDevice test requires more than "
//                     "one addressable devices to run.";
//   }

//   PJRT_Buffer_CopyToDevice_Args args;
//   args.struct_size = PJRT_Buffer_CopyToDevice_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   args.buffer = buffer_.get();

//   // Copy to the buffer's current device is not allowed.
//   ASSERT_EQ(buffer_.get()->buffer->device(), addressable_devices[0]->device);
//   args.dst_device = addressable_devices[0];
//   std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter> error(
//       api_->PJRT_Buffer_CopyToDevice(&args), ::pjrt::MakeErrorDeleter(api_));
//   EXPECT_NE(error, nullptr);
//   EXPECT_EQ(
//       ::pjrt::GetPjrtErrorMessage(error.get(), api_),
//       "CopyToDevice cannot accept the same source and destination devices");

//   // Copy to `dst_device` returns `dst_buffer` associated with `dst_device`.
//   args.dst_device = addressable_devices[1];
//   error.reset(api_->PJRT_Buffer_CopyToDevice(&args));
//   EXPECT_EQ(error, nullptr);
//   std::unique_ptr<PJRT_Buffer> dst_buffer(args.dst_buffer);
//   EXPECT_EQ(dst_buffer->buffer->device(), addressable_devices[1]->device);

//   // TODO(b/240779809) This is to prevent `use-of-uninitialized-value` error.
//   // Client and its resources were destroyed after the test completion, while
//   // still waiting for the data transfer to complete from TFRT non-blocking
//   // queue. `CopyRawToHost` and awaiting on the call resolved the issue. Use
//   C
//   // API ToLiteral when it is ready.
//   PjRtBuffer* output_buffer = dst_buffer->buffer.get();
//   void* dst =
//       aligned_malloc(output_buffer->GetOnDeviceSizeInBytes().value(), 0);
//   PjRtFuture<absl::Status> to_host = output_buffer->CopyRawToHost(
//       dst, 0, output_buffer->GetOnDeviceSizeInBytes().value());
//   absl::Status status = to_host.Await();
//   EXPECT_TRUE(status.ok());

//   // The input buffer is 41, and so should the output be.
//   EXPECT_EQ(*(static_cast<float*>(dst)), 41);
//   aligned_free(dst);
// }

// TEST_F(PjrtCApiTpuBufferTest, IsOnCpu) {
//   PJRT_Buffer_IsOnCpu_Args args;
//   args.struct_size = PJRT_Buffer_IsOnCpu_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   args.buffer = buffer_.get();
//   PJRT_Error* error = api_->PJRT_Buffer_IsOnCpu(&args);
//   EXPECT_EQ(error, nullptr);
//   EXPECT_FALSE(args.is_on_cpu);
// }

// TEST_F(PjrtCApiTpuBufferTest, Device) {
//   PJRT_Buffer_Device_Args args;
//   args.struct_size = PJRT_Buffer_Device_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   args.buffer = buffer_.get();

//   // The returned device is addressable.
//   std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter> error(
//       api_->PJRT_Buffer_Device(&args), ::pjrt::MakeErrorDeleter(api_));
//   EXPECT_EQ(error, nullptr);
//   EXPECT_EQ(args.device->device, GetClientAddressableDevices()[0]->device);
// }

// TEST_F(PjrtCApiTpuBufferTest, ReadyEvent) {
//   PJRT_Buffer_ReadyEvent_Args get_event_args;
//   get_event_args.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
//   get_event_args.priv = nullptr;
//   get_event_args.buffer = buffer_.get();
//   auto error = ToUniquePtr(api_->PJRT_Buffer_ReadyEvent(&get_event_args));
//   ASSERT_EQ(error, nullptr);

//   PJRT_Event* event = get_event_args.event;
//   ASSERT_NE(event, nullptr);

//   // Wait for `buffer_`'s data transfer to complete (if it hasn't already)
//   PJRT_Event_Await_Args await_args;
//   await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
//   await_args.priv = nullptr;
//   await_args.event = event;
//   error.reset(api_->PJRT_Event_Await(&await_args));
//   ASSERT_EQ(error, nullptr);

//   // Must be ready when `PJRT_Event_Await` completes
//   PJRT_Event_IsReady_Args ready_args;
//   ready_args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
//   ready_args.priv = nullptr;
//   ready_args.event = event;
//   error.reset(api_->PJRT_Event_IsReady(&ready_args));
//   ASSERT_EQ(error, nullptr);
//   EXPECT_TRUE(ready_args.is_ready);

//   // Clean up
//   PJRT_Event_Destroy_Args destroy_args;
//   destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
//   destroy_args.priv = nullptr;
//   destroy_args.event = event;
//   error.reset(api_->PJRT_Event_Destroy(&destroy_args));
//   EXPECT_EQ(error, nullptr);
// }

// TEST_F(PjrtCApiTpuBufferTest, UnsafeBufferPointer) {
//   // Call the function directly to get a reference pointer value, check that
//   the
//   // result is not zero (nullptr).
//   xla::PjRtBuffer* xla_pjrt_buffer = buffer_->buffer.get();
//   ASSERT_NE(xla_pjrt_buffer, nullptr);
//   absl::StatusOr<std::uintptr_t> local_buffer_pointer =
//       client_->client->UnsafeBufferPointer(xla_pjrt_buffer);
//   ASSERT_TRUE(local_buffer_pointer.ok());
//   ASSERT_NE(local_buffer_pointer.value(), 0);

//   // Call the function through PJRT C API interface, check that the
//   // result is not zero (nullptr).
//   PJRT_Buffer_UnsafePointer_Args args{
//       .struct_size = PJRT_Buffer_UnsafePointer_Args_STRUCT_SIZE,
//       .priv = nullptr,
//       .buffer = buffer_.get(),
//   };

//   PJRT_Error* error = api_->PJRT_Buffer_UnsafePointer(&args);
//   ::pjrt::LogFatalIfPjrtError(error, api_);
//   ASSERT_EQ(error, nullptr);
//   ASSERT_NE(args.buffer_pointer, 0);

//   // Confirm pointer values for direct and PJRT C API calls are the same.
//   ASSERT_EQ(args.buffer_pointer, local_buffer_pointer.value());
// }

// // ---------------------------------- Events
// -----------------------------------

// static PJRT_Event* EventFromPromise(PjRtFuture<absl::Status>::Promise
// promise) {
//   return new PJRT_Event{PjRtFuture<absl::Status>{promise}};
// }

// class PjrtCApiEventsTest : public PjrtCApiTpuTest {
//  protected:
//   void SetUp() override {
//     PjrtCApiTpuTest::SetUp();
//     test_promise_ =  // to be set inside test cases
//         std::make_unique<PjRtFuture<absl::Status>::Promise>(
//             PjRtFuture<absl::Status>::CreatePromise());
//     event_ = std::unique_ptr<PJRT_Event, ::pjrt::PJRT_EventDeleter>{
//         EventFromPromise(*test_promise_), ::pjrt::MakeEventDeleter(api_)};
//   }

//   void TearDown() override {
//     event_.reset();  // does not replace the deleter
//     test_promise_.reset();
//     PjrtCApiTpuTest::TearDown();
//   }

//   void SetEventFromStatus(absl::Status status) { test_promise_->Set(status);
//   }

//   bool IsReady() {
//     PJRT_Event_IsReady_Args ready_args;
//     ready_args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
//     ready_args.priv = nullptr;
//     ready_args.event = event_.get();
//     ::pjrt::LogFatalIfPjrtError(api_->PJRT_Event_IsReady(&ready_args), api_);
//     return ready_args.is_ready;
//   }

//   void SetOnReady(PJRT_Event_OnReadyCallback callback, void* arguments) {
//     PJRT_Event_OnReady_Args args{
//         .struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE,
//         .priv = nullptr,
//         .event = event_.get(),
//         .callback = callback,
//         .user_arg = arguments,
//     };

//     auto error = ToUniquePtr(api_->PJRT_Event_OnReady(&args));
//     CHECK(error == nullptr);
//   }

//   PJRT_Error* Await() {
//     PJRT_Event_Await_Args args;
//     args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
//     args.priv = nullptr;
//     args.event = event_.get();
//     return api_->PJRT_Event_Await(&args);
//   }

//   PJRT_Error* GetError() {
//     PJRT_Event_Error_Args args;
//     args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
//     args.priv = nullptr;
//     args.event = event_.get();
//     return api_->PJRT_Event_Error(&args);
//   }

//   std::unique_ptr<PjRtFuture<absl::Status>::Promise> test_promise_;
//   // tracks test_promise_
//   std::unique_ptr<PJRT_Event, ::pjrt::PJRT_EventDeleter> event_;
// };

// constexpr static std::string_view kTestErrorMessage = "Test error message";

// TEST_F(PjrtCApiEventsTest, IsReady) {
//   PJRT_Event_IsReady_Args args;
//   args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   args.event = event_.get();
//   PJRT_Error* error = nullptr;

//   // Not ready when initiated from a blank/default promise
//   error = api_->PJRT_Event_IsReady(&args);
//   ASSERT_EQ(error, nullptr);
//   ASSERT_FALSE(args.is_ready);

//   test_promise_->Set(xla::OkStatus());

//   // Ready as soon as the promise is fulfilled
//   error = api_->PJRT_Event_IsReady(&args);
//   EXPECT_EQ(error, nullptr);
//   EXPECT_TRUE(args.is_ready);
// }

// TEST_F(PjrtCApiEventsTest, IsReadyWhenPromisePreFilled) {
//   SetEventFromStatus(xla::OkStatus());

//   PJRT_Event_IsReady_Args args;
//   args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   args.event = event_.get();

//   PJRT_Error* error = api_->PJRT_Event_IsReady(&args);
//   EXPECT_EQ(error, nullptr);
//   EXPECT_TRUE(args.is_ready);
// }

// TEST_F(PjrtCApiEventsTest, IsReadyOnError) {
//   auto test_err_code = absl::StatusCode::kInternal;
//   const absl::Status test_status{test_err_code, kTestErrorMessage};
//   SetEventFromStatus(test_status);

//   PJRT_Event_IsReady_Args args;
//   args.struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   args.event = event_.get();

//   PJRT_Error* error = api_->PJRT_Event_IsReady(&args);
//   EXPECT_EQ(error, nullptr);
//   EXPECT_TRUE(args.is_ready);
// }

// TEST_F(PjrtCApiEventsTest, AwaitYieldsCorrectErrorWhenSet) {
//   ASSERT_FALSE(IsReady());

//   auto test_err_code = absl::StatusCode::kInternal;
//   const absl::Status test_status{test_err_code, kTestErrorMessage};
//   test_promise_->Set(test_status);
//   ASSERT_TRUE(IsReady())
//       << "Error: `event_` is not ready after `test_promise_` was Set";

//   auto error = ToUniquePtr(Await());
//   ASSERT_NE(error, nullptr);

//   absl::Status status = ::pjrt::PjrtErrorToStatus(error.get(), api_);
//   ASSERT_EQ(status.code(), test_err_code);
//   absl::string_view error_message =
//       ::pjrt::GetPjrtErrorMessage(error.get(), api_);
//   ASSERT_EQ(error_message, kTestErrorMessage);
// }

// TEST_F(PjrtCApiEventsTest, AwaitBlocksCallingThread) {
//   auto error = ToUniquePtr(nullptr);
//   ASSERT_FALSE(IsReady());
//   thread::Fiber action_fiber(
//       [api = api_, event = event_.get(), error = &error]() {
//         PJRT_Event_Await_Args args;
//         args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
//         args.priv = nullptr;
//         args.event = event;
//         error->reset(api->PJRT_Event_Await(&args));
//       });
//   ASSERT_FALSE(IsReady())
//       << "Error: `event_` is ready before `test_promise_` was Set";

//   auto test_err_code = absl::StatusCode::kInternal;
//   const absl::Status test_status{test_err_code, kTestErrorMessage};
//   test_promise_->Set(test_status);
//   ASSERT_TRUE(IsReady())
//       << "Error: `event_` is not ready after `test_promise_` was Set";

//   action_fiber.Join();
//   ASSERT_NE(error, nullptr);
//   absl::Status status = ::pjrt::PjrtErrorToStatus(error.get(), api_);
//   ASSERT_EQ(status.code(), test_err_code);
//   absl::string_view error_message =
//       ::pjrt::GetPjrtErrorMessage(error.get(), api_);
//   ASSERT_EQ(error_message, kTestErrorMessage);
// }

// // Once PJRT_Event_Await is called, calls to PJRT_Event_Error should return
// // pointers to an equivalent object
// TEST_F(PjrtCApiEventsTest, GetErrorNoError) {
//   test_promise_->Set(xla::OkStatus());
//   ASSERT_TRUE(IsReady())
//       << "Error: `event_` is not ready after `test_promise_` was Set";

//   auto await_error = ToUniquePtr(Await());
//   ASSERT_EQ(await_error, nullptr)
//       << "`api_->PJRT_Event_Await(event_)` was not null in `" << __func__
//       << "()` for `xla::OkStatus()`.";

//   auto status_error = ToUniquePtr(GetError());
//   ASSERT_EQ(status_error, nullptr)
//       << "`api_->PJRT_Event_Error(event_)` was not null in `" << __func__
//       << "()` after correctly null `api_->PJRT_Event_Await(event_)` for "
//          "`xla::OkStatus()`.";
// }

// TEST_F(PjrtCApiEventsTest, GetErrorYesError) {
//   auto test_err_code = absl::StatusCode::kInternal;
//   const absl::Status test_status{test_err_code, kTestErrorMessage};
//   test_promise_->Set(test_status);
//   ASSERT_TRUE(IsReady())
//       << "Error: `event_` is not ready after `test_promise_` was Set";

//   auto await_error = ToUniquePtr(Await());
//   ASSERT_NE(await_error, nullptr)
//       << "`api_->PJRT_Event_Await(event_)` was null in `" << __func__ <<
//       "()`.";
//   absl::Status await_status =
//       ::pjrt::PjrtErrorToStatus(await_error.get(), api_);
//   EXPECT_EQ(await_status, test_status);
//   await_error.reset();

//   auto status_error = ToUniquePtr(GetError());
//   ASSERT_NE(status_error, nullptr)
//       << "`api_->PJRT_Event_Error(event_)` was null in `" << __func__
//       << "()` after non-null `api_->PJRT_Event_Await(event_)`.";

//   absl::Status status = ::pjrt::PjrtErrorToStatus(status_error.get(), api_);
//   EXPECT_EQ(status, test_status);
// }

// TEST_F(PjrtCApiEventsTest, GetErrorThenAwait) {
//   auto test_err_code = absl::StatusCode::kInternal;
//   const absl::Status test_status{test_err_code, kTestErrorMessage};
//   test_promise_->Set(test_status);
//   ASSERT_TRUE(IsReady())
//       << "Error: `event_` is not ready after `test_promise_` was Set";

//   auto status_error = ToUniquePtr(GetError());
//   ASSERT_NE(status_error, nullptr)
//       << "`api_->PJRT_Event_Error(event_)` was null in `" << __func__
//       << "()` after non-null `api_->PJRT_Event_Await(event_)`.";
//   absl::Status status = ::pjrt::PjrtErrorToStatus(status_error.get(), api_);
//   EXPECT_EQ(status, test_status);
//   status_error.reset();

//   auto await_error = ToUniquePtr(Await());
//   ASSERT_NE(await_error, nullptr)
//       << "`api_->PJRT_Event_Await(event_)` was null in `" << __func__ <<
//       "()`.";
//   absl::Status await_status =
//       ::pjrt::PjrtErrorToStatus(await_error.get(), api_);
//   EXPECT_EQ(await_status, test_status);
// }

// struct StringAndApi {
//   std::string* str;
//   const PJRT_Api* api;
//   bool is_set = false;
//   bool error_was_null = false;
// };

// static void StringWriteCallback(PJRT_Error* error, void* void_arg_pointer) {
//   auto string_and_api = reinterpret_cast<StringAndApi*>(void_arg_pointer);
//   CHECK(string_and_api != nullptr);
//   if (error == nullptr) {
//     *string_and_api->str = "";
//     string_and_api->is_set = true;
//     string_and_api->error_was_null = true;
//     return;
//   }
//   *string_and_api->str =
//       ::pjrt::GetPjrtErrorMessage(error, string_and_api->api);
//   string_and_api->is_set = true;
//   string_and_api->error_was_null = false;
//   PJRT_Error_Destroy_Args destroy;
//   destroy.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
//   destroy.priv = nullptr;
//   destroy.error = error;
//   string_and_api->api->PJRT_Error_Destroy(&destroy);
// }

// constexpr static std::string_view kInitialMessage =
//     "Should never end up with this string";

// TEST_F(PjrtCApiEventsTest, OnReadyNoError) {
//   ASSERT_FALSE(IsReady());
//   std::string side_effect{kInitialMessage};
//   StringAndApi wrapper{&side_effect, api_};
//   SetOnReady(StringWriteCallback, &wrapper);

//   test_promise_->Set(xla::OkStatus());
//   ASSERT_TRUE(IsReady());
//   // We must wait for the callback to complete
//   absl::Mutex mu;
//   mu.LockWhen(absl::Condition{+[](bool* b) { return *b; }, &wrapper.is_set});
//   // The callback should be complete by now
//   mu.Unlock();
//   // No error -> absl::Status::error_message() returns empty string
//   EXPECT_EQ(side_effect, "");
//   EXPECT_TRUE(wrapper.error_was_null);
// }

// TEST_F(PjrtCApiEventsTest, OnReadyWithError) {
//   ASSERT_FALSE(IsReady());
//   std::string side_effect{kInitialMessage};
//   StringAndApi wrapper{&side_effect, api_};
//   SetOnReady(StringWriteCallback, &wrapper);

//   auto test_err_code = absl::StatusCode::kInternal;
//   const absl::Status test_status{test_err_code, kTestErrorMessage};
//   test_promise_->Set(test_status);

//   ASSERT_TRUE(IsReady());
//   // We must wait for the callback to complete
//   absl::Mutex mu;
//   mu.LockWhen(absl::Condition{+[](bool* b) { return *b; }, &wrapper.is_set});
//   // The callback should be complete by now
//   mu.Unlock();
//   EXPECT_EQ(side_effect, kTestErrorMessage);
//   EXPECT_FALSE(wrapper.error_was_null);
// }

// // ------------------------- Device Topology
// -----------------------------------

// class PjrtCApiTopologyDescriptionTest : public PjrtCApiTpuTest {};

// TEST_F(PjrtCApiTopologyDescriptionTest, Compile) {
//   auto topology = CreateTopology(api_);
//   auto executable =
//       create_executable(api_, topology.get(), CreateAddOneComputation());
//   auto pjrt_executable =
//       create_executable(api_, client_, CreateAddOneComputation());
//   auto reconstructed_exec = SerializeAndLoad(api_, client_,
//   executable.get());

//   // Run the default executable created in PjrtCApiTpuExecutableTest:SetUp
//   and
//   // assign its output
//   ASSERT_OK_AND_ASSIGN(float result_expected,
//                        RunScalarExecutableAndGetResult(pjrt_executable.get()));
//   // Run our reconstructed executable and assign its output
//   ASSERT_OK_AND_ASSIGN(
//       float result_of_reserialization,
//       RunScalarExecutableAndGetResult(reconstructed_exec.get()));

//   EXPECT_EQ(result_expected, result_of_reserialization);
// }

// TEST_F(PjrtCApiTopologyDescriptionTest, CompileWithOptions) {
//   absl::flat_hash_map<std::string, PjRtValueType> create_options = {
//       {"num_slices", PjRtValueType(static_cast<int64_t>(2))}};
//   auto topology = CreateTopologyWithOptions(api_, "v3:2x4x2",
//   create_options); auto* ms_topology =
//       static_cast<MegaScalePjRtTopologyDescription*>(topology->topology.get());
//   EXPECT_EQ(ms_topology->platform_name(), xla::TpuName());
//   EXPECT_TRUE(absl::StartsWith(ms_topology->platform_version(),
//                                "megascale+TFRT TPU v3"));
//   EXPECT_EQ(ms_topology->num_slices(), 2);
//   EXPECT_EQ(ms_topology->tfrt_topology()->ProcessCount().value(), 4);
//   EXPECT_EQ(ms_topology->tfrt_topology()->CoreCountOfDefaultType().value(),
//   32);
// }

// TEST_F(PjrtCApiTopologyDescriptionTest, GetDeviceDescriptions) {
//   auto topology = CreateTopology(api_);

//   PJRT_TopologyDescription_GetDeviceDescriptions_Args args;
//   args.struct_size =
//       PJRT_TopologyDescription_GetDeviceDescriptions_Args_STRUCT_SIZE;
//   args.priv = nullptr;
//   args.topology = topology.get();
//   ::pjrt::LogFatalIfPjrtError(
//       api_->PJRT_TopologyDescription_GetDeviceDescriptions(&args), api_);

//   auto other = topology->topology->DeviceDescriptions();
//   ASSERT_EQ(args.num_descriptions, other.size());
//   for (size_t i = 0; i < args.num_descriptions; ++i) {
//     ASSERT_EQ(other[i]->id(), GetDeviceId(args.descriptions[i]));
//   }
// }

// // --------------------------------- Helpers
// -----------------------------------

// class PjrtCApiHelpersTest : public PjrtCApiTpuTest {};

// TEST_F(PjrtCApiHelpersTest, PjrtErrorToStatus) {
//   // Return success if nullptr
//   EXPECT_TRUE(::pjrt::PjrtErrorToStatus(nullptr, api_).ok());

//   // Return UNKNOWN status with the original message if not nullptr
//   auto error = std::make_unique<PJRT_Error>();
//   error->status = tsl::errors::InvalidArgument("Should be UNKNOWN");
//   absl::Status status = ::pjrt::PjrtErrorToStatus(error.get(), api_);
//   EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
//   EXPECT_EQ(status.error_message(), "Should be UNKNOWN");
// }

// // For these Deleter tests, the heap leak checker will check if heap memory
// // allocated with `new` is properly freed at the exit.
// TEST_F(PjrtCApiHelpersTest, MakeErrorDeleter) {
//   PJRT_Error* error = new PJRT_Error();
//   ::pjrt::MakeErrorDeleter(api_)(error);
// }

// TEST_F(PjrtCApiHelpersTest, MakeEventDeleter) {
//   PJRT_Event* event = new PJRT_Event();
//   ::pjrt::MakeEventDeleter(api_)(event);
// }

}  // namespace
}  // namespace pjrt
}  // namespace xla
