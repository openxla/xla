#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/shape_util.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace pjrt {
PJRT_Error* PJRT_Device_ClearMemoryStats(
    PJRT_Device_ClearMemoryStats_Args* args);
}

namespace xla {
namespace {

absl::StatusOr<std::unique_ptr<PjRtBuffer>> AllocateBytes(PjRtClient* client,
                                                          PjRtDevice* device,
                                                          int64_t bytes) {
  int64_t num_elements = bytes / sizeof(float);
  std::vector<float> data(num_elements, 1.0f);
  Shape shape = ShapeUtil::MakeShape(F32, {num_elements});

  // The new PJRT API requires passing the memory space directly instead of the
  // device
  return client->BufferFromHostBuffer(
      data.data(), shape.element_type(), shape.dimensions(),
      /*byte_strides=*/std::nullopt,
      PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
      []() {},  // OnDone callback
      device->default_memory_space().value(),
      /*device_layout=*/nullptr);
}

TEST(PjRtMemoryStatsTest, TrackAndClearPeakMemory) {
  // Initialize the GPU PJRT Client
  GpuClientOptions options;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(options));

  PjRtDevice* device = client->addressable_devices()[0];
  auto* se_device = static_cast<PjRtStreamExecutorDevice*>(device);

  // Helper to extract allocator stats
  auto get_stats = [&]() -> tsl::AllocatorStats {
    absl::StatusOr<tsl::AllocatorStats> stats = device->GetAllocatorStats();
    EXPECT_TRUE(stats.ok());
    return *stats;
  };

  // Initial Baseline
  auto s0 = get_stats();
  int64_t initial_bytes = s0.bytes_in_use;

  // Create Tensor A (1024 Bytes)
  TF_ASSERT_OK_AND_ASSIGN(auto tensor_a,
                          AllocateBytes(client.get(), device, 1024));
  TF_ASSERT_OK(tensor_a->GetReadyFuture().Await());

  auto s1 = get_stats();
  EXPECT_GT(s1.bytes_in_use, initial_bytes);
  EXPECT_EQ(s1.peak_bytes_in_use, s1.bytes_in_use);

  // Create Tensor B (1024 Bytes)
  TF_ASSERT_OK_AND_ASSIGN(auto tensor_b,
                          AllocateBytes(client.get(), device, 1024));
  TF_ASSERT_OK(tensor_b->GetReadyFuture().Await());

  auto s2 = get_stats();
  EXPECT_GT(s2.bytes_in_use, s1.bytes_in_use);
  EXPECT_EQ(s2.peak_bytes_in_use, s2.bytes_in_use);

  tensor_a.reset();

  auto s3 = get_stats();
  EXPECT_LT(s3.bytes_in_use, s2.bytes_in_use);
  EXPECT_EQ(s3.peak_bytes_in_use, s2.peak_bytes_in_use);

  // Call ClearMemoryStats
  TF_ASSERT_OK(se_device->ClearMemoryStats());

  auto s4 = get_stats();
  EXPECT_EQ(s4.bytes_in_use, s3.bytes_in_use);
  EXPECT_EQ(s4.peak_bytes_in_use, s4.bytes_in_use);

  // Create Tensor C (512 Bytes)
  TF_ASSERT_OK_AND_ASSIGN(auto tensor_c,
                          AllocateBytes(client.get(), device, 512));
  TF_ASSERT_OK(tensor_c->GetReadyFuture().Await());

  auto s5 = get_stats();
  EXPECT_GT(s5.bytes_in_use, s4.bytes_in_use);
  EXPECT_GT(s5.peak_bytes_in_use, s4.peak_bytes_in_use);
  EXPECT_LT(s5.peak_bytes_in_use, s2.peak_bytes_in_use);

  tensor_c.reset();

  auto s6 = get_stats();
  EXPECT_LT(s6.bytes_in_use, s5.bytes_in_use);
  EXPECT_EQ(s6.peak_bytes_in_use, s5.peak_bytes_in_use);
}

TEST(PjRtMemoryStatsTest, UnimplementedOnCpu) {
  // Initialize a CPU Client
  CpuClientOptions options;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetPjRtCpuClient(options));

  PjRtDevice* device = client->addressable_devices()[0];

  // Try to call the implementation directly
  absl::Status status = device->ClearMemoryStats();

  // Assert it returns Unimplemented
  EXPECT_EQ(status.code(), absl::StatusCode::kUnimplemented);
  EXPECT_TRUE(absl::StrContains(status.message(), "not supported"));
}

TEST(PjRtMemoryStatsTest, CApiReturnsUnimplementedForCpu) {
  // Initialize a CPU Client
  CpuClientOptions options;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetPjRtCpuClient(options));

  PjRtDevice* device = client->addressable_devices()[0];

  // Initialize a minimal PJRT_Api table
  PJRT_Api api;
  memset(&api, 0, sizeof(api));
  api.struct_size = PJRT_Api_STRUCT_SIZE;
  api.PJRT_Device_ClearMemoryStats = pjrt::PJRT_Device_ClearMemoryStats;
  api.PJRT_Error_Destroy = pjrt::PJRT_Error_Destroy;
  api.PJRT_Error_GetCode = pjrt::PJRT_Error_GetCode;

  // Wrap the PjRtDevice in the internal PJRT_Device struct
  PJRT_Device c_device;
  c_device.device = device;

  PJRT_Device_ClearMemoryStats_Args args;
  args.struct_size = PJRT_Device_ClearMemoryStats_Args_STRUCT_SIZE;
  args.device = &c_device;

  // Call through the wrapper function
  PJRT_Error* error = api.PJRT_Device_ClearMemoryStats(&args);

  ASSERT_NE(error, nullptr);
  EXPECT_EQ(pjrt::GetErrorCode(error, &api), PJRT_Error_Code_UNIMPLEMENTED);

  // Free the error using the wrapper implementation
  PJRT_Error_Destroy_Args destroy_args;
  destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy_args.error = error;
  api.PJRT_Error_Destroy(&destroy_args);
}

}  // namespace
}  // namespace xla
