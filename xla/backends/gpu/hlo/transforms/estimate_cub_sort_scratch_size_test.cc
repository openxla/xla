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

#include "xla/backends/gpu/hlo/transforms/estimate_cub_sort_scratch_size.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_planner.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

class EstimateCubSortScratchSizeTest : public HloHardwareIndependentTestBase {
 public:
  void RunAndCheck(absl::string_view hlo, absl::string_view expected) {
    RunAndFilecheckHloRewrite(
        hlo,
        EstimateCubSortScratchSize(
            "CUDA", stream_executor::GpuComputeCapability(
                        stream_executor::CudaComputeCapability(9, 0))),
        expected);
  }
};

TEST_F(EstimateCubSortScratchSizeTest, U32_F32) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = u32[1000] parameter(0)
      %values = f32[1000] parameter(1)
      %custom-call = (u32[1000]{0}, f32[1000]{0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":false}
      ROOT %t = u32[1000]{0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: custom-call
    // CHECK-SAME: custom_call_target="xla.gpu.ext.cub_sort_pairs"
    // CHECK-SAME: api_version=API_VERSION_TYPED_FFI
  )");
}

TEST_F(EstimateCubSortScratchSizeTest, F32) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = f32[1000] parameter(0)
      %custom-call = (f32[1000]{0}, u8[1]{0})
        custom-call(%keys),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":false}
      ROOT %t = f32[1000]{0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: custom-call
    // CHECK-SAME: custom_call_target="xla.gpu.ext.cub_sort_keys"
    // CHECK-SAME: api_version=API_VERSION_TYPED_FFI
  )");
}

TEST_F(EstimateCubSortScratchSizeTest, S32_S32) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = s32[1000] parameter(0)
      %values = s32[1000] parameter(1)
      %custom-call = (s32[1000]{0}, s32[1000]{0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":false}
      ROOT %t = s32[1000]{0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: custom-call
    // CHECK-SAME: custom_call_target="xla.gpu.ext.cub_sort_pairs"
    // CHECK-SAME: api_version=API_VERSION_TYPED_FFI
  )");
}

TEST_F(EstimateCubSortScratchSizeTest, F32_Descending) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = f32[1000] parameter(0)
      %custom-call = (f32[1000]{0}, u8[1]{0})
        custom-call(%keys),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":true}
      ROOT %t = f32[1000]{0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: custom-call
    // CHECK-SAME: custom_call_target="xla.gpu.ext.cub_sort_keys"
    // CHECK-SAME: backend_config={descending = true, batch_size = 1 : i64}
  )");
}

TEST_F(EstimateCubSortScratchSizeTest, F32_Rank3) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = f32[10,10,10] parameter(0)
      %custom-call = (f32[10,10,10]{2,1,0}, u8[1]{0})
        custom-call(%keys),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":false}
      ROOT %t = f32[10,10,10]{2,1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: custom-call
    // CHECK-SAME: custom_call_target="xla.gpu.ext.cub_sort_keys"
    // CHECK-SAME: backend_config={descending = false, batch_size = 100 : i64}
  )");
}

TEST_F(EstimateCubSortScratchSizeTest, F32_Rank2) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = f32[10,100] parameter(0)
      %custom-call = (f32[10,100]{1,0}, u8[1]{0})
        custom-call(%keys),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":false}
      ROOT %t = f32[10,100]{1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: custom-call
    // CHECK-SAME: custom_call_target="xla.gpu.ext.cub_sort_keys"
    // CHECK-SAME: backend_config={descending = false, batch_size = 10 : i64}
  )");
}

TEST_F(EstimateCubSortScratchSizeTest, U16_F16_Descending) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = u16[16,128] parameter(0)
      %values = f16[16,128] parameter(1)
      %custom-call = (u16[16,128]{1,0}, f16[16,128]{1,0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":true}
      ROOT %t = u16[16,128]{1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: custom-call
    // CHECK-SAME: custom_call_target="xla.gpu.ext.cub_sort_pairs"
    // CHECK-SAME: backend_config={descending = true, batch_size = 16 : i64}
  )");
}

TEST_F(EstimateCubSortScratchSizeTest, U32_F32_Rank2) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = u32[16,128] parameter(0)
      %values = f32[16,128] parameter(1)
      %custom-call = (u32[16,128]{1,0}, f32[16,128]{1,0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":false}
      ROOT %t = u32[16,128]{1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: custom-call
    // CHECK-SAME: custom_call_target="xla.gpu.ext.cub_sort_pairs"
    // CHECK-SAME: backend_config={descending = false, batch_size = 16 : i64}
  )");
}

TEST_F(EstimateCubSortScratchSizeTest, U64_F64_Descending) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = u64[16,128] parameter(0)
      %values = f64[16,128] parameter(1)
      %custom-call = (u64[16,128]{1,0}, f64[16,128]{1,0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":true}
      ROOT %t = u64[16,128]{1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: custom-call
    // CHECK-SAME: custom_call_target="xla.gpu.ext.cub_sort_pairs"
    // CHECK-SAME: backend_config={descending = true, batch_size = 16 : i64}
  )");
}

TEST_F(EstimateCubSortScratchSizeTest, U16_BF16) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = u16[16,128] parameter(0)
      %values = bf16[16,128] parameter(1)
      %custom-call = (u16[16,128]{1,0}, bf16[16,128]{1,0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":false}
      ROOT %t = u16[16,128]{1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: custom-call
    // CHECK-SAME: custom_call_target="xla.gpu.ext.cub_sort_pairs"
    // CHECK-SAME: backend_config={descending = false, batch_size = 16 : i64}
  )");
}

TEST_F(EstimateCubSortScratchSizeTest, U16_BF16_Descending) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = u16[16,128] parameter(0)
      %values = bf16[16,128] parameter(1)
      %custom-call = (u16[16,128]{1,0}, bf16[16,128]{1,0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":true}
      ROOT %t = u16[16,128]{1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: custom-call
    // CHECK-SAME: custom_call_target="xla.gpu.ext.cub_sort_pairs"
    // CHECK-SAME: backend_config={descending = true, batch_size = 16 : i64}
  )");
}

TEST_F(EstimateCubSortScratchSizeTest, U16_F16) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = u16[16,128] parameter(0)
      %values = f16[16,128] parameter(1)
      %custom-call = (u16[16,128]{1,0}, f16[16,128]{1,0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":false}
      ROOT %t = u16[16,128]{1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: custom-call
    // CHECK-SAME: custom_call_target="xla.gpu.ext.cub_sort_pairs"
    // CHECK-SAME: backend_config={descending = false, batch_size = 16 : i64}
  )");
}

TEST_F(EstimateCubSortScratchSizeTest, U32_F32_Rank2_Descending) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = u32[16,128] parameter(0)
      %values = f32[16,128] parameter(1)
      %custom-call = (u32[16,128]{1,0}, f32[16,128]{1,0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":true}
      ROOT %t = u32[16,128]{1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: custom-call
    // CHECK-SAME: custom_call_target="xla.gpu.ext.cub_sort_pairs"
    // CHECK-SAME: backend_config={descending = true, batch_size = 16 : i64}
  )");
}

TEST_F(EstimateCubSortScratchSizeTest, U64_F64) {
  const char hlo[] = R"(
    HloModule m
    ENTRY main {
      %keys = u64[16,128] parameter(0)
      %values = f64[16,128] parameter(1)
      %custom-call = (u64[16,128]{1,0}, f64[16,128]{1,0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":false}
      ROOT %t = u64[16,128]{1,0} get-tuple-element(%custom-call), index=0
  })";
  RunAndCheck(hlo, R"(
    // CHECK: custom-call
    // CHECK-SAME: custom_call_target="xla.gpu.ext.cub_sort_pairs"
    // CHECK-SAME: backend_config={descending = false, batch_size = 16 : i64}
  )");
}

TEST_F(EstimateCubSortScratchSizeTest, SingleTileClampsZeroScratchToOneByte) {
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY main {
      %keys = f32[1000] parameter(0)
      %custom-call = (f32[1000]{0}, u8[1]{0})
        custom-call(%keys),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":false}
      ROOT %t = f32[1000]{0} get-tuple-element(%custom-call), index=0
  })";
  const stream_executor::CudaComputeCapability cc(9, 0);
  ASSERT_OK_AND_ASSIGN(
      int64_t planned_scratch,
      ComputeCudaCubSortScratchSize(cc, F32, std::nullopt, 1000, 1));
  EXPECT_EQ(planned_scratch, 0);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_THAT(EstimateCubSortScratchSize(
                  "CUDA", stream_executor::GpuComputeCapability(cc))
                  .Run(module.get()),
              IsOkAndHolds(true));
  const HloInstruction* call =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_EQ(call->shape().tuple_shapes(1).element_type(), U8);
  EXPECT_EQ(call->shape().tuple_shapes(1).dimensions(0), 1);
}

TEST_F(EstimateCubSortScratchSizeTest, OnesweepWritesExactNonTrivialScratch) {
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY main {
      %keys = u32[65536] parameter(0)
      %values = f32[65536] parameter(1)
      %custom-call = (u32[65536]{0}, f32[65536]{0}, u8[1]{0})
        custom-call(%keys, %values),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":false}
      ROOT %t = u32[65536]{0} get-tuple-element(%custom-call), index=0
  })";
  const stream_executor::CudaComputeCapability cc(9, 0);
  ASSERT_OK_AND_ASSIGN(int64_t expected_scratch,
                       ComputeCudaCubSortScratchSize(cc, U32, F32, 65536, 1));
  ASSERT_GT(expected_scratch, 1);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_THAT(EstimateCubSortScratchSize(
                  "CUDA", stream_executor::GpuComputeCapability(cc))
                  .Run(module.get()),
              IsOkAndHolds(true));
  const HloInstruction* call =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_EQ(call->shape().tuple_shapes(2).element_type(), U8);
  EXPECT_EQ(call->shape().tuple_shapes(2).dimensions(0), expected_scratch);
}

TEST_F(EstimateCubSortScratchSizeTest, SegmentedWritesExactNonTrivialScratch) {
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY main {
      %keys = f32[16,10000] parameter(0)
      %custom-call = (f32[16,10000]{1,0}, u8[1]{0})
        custom-call(%keys),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":true}
      ROOT %t = f32[16,10000]{1,0} get-tuple-element(%custom-call), index=0
  })";
  const stream_executor::CudaComputeCapability cc(9, 0);
  ASSERT_OK_AND_ASSIGN(
      int64_t expected_scratch,
      ComputeCudaCubSortScratchSize(cc, F32, std::nullopt, 160000, 16));
  ASSERT_GT(expected_scratch, 1);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_THAT(EstimateCubSortScratchSize(
                  "CUDA", stream_executor::GpuComputeCapability(cc))
                  .Run(module.get()),
              IsOkAndHolds(true));
  const HloInstruction* call =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_EQ(call->shape().tuple_shapes(1).element_type(), U8);
  EXPECT_EQ(call->shape().tuple_shapes(1).dimensions(0), expected_scratch);
}

TEST_F(EstimateCubSortScratchSizeTest, RejectsEmptyAndSingleElementSorts) {
  const stream_executor::CudaComputeCapability cc(9, 0);
  for (absl::string_view hlo : {
           absl::string_view(R"(
             HloModule m
             ENTRY main {
               %keys = f32[] parameter(0)
               %custom-call = (f32[], u8[1]{0})
                 custom-call(%keys),
                 custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
                 backend_config={"descending":false}
               ROOT %t = f32[] get-tuple-element(%custom-call), index=0
             })"),
           absl::string_view(R"(
             HloModule m
             ENTRY main {
               %keys = f32[0,4096] parameter(0)
               %custom-call = (f32[0,4096]{1,0}, u8[1]{0})
                 custom-call(%keys),
                 custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
                 backend_config={"descending":false}
               ROOT %t = f32[0,4096]{1,0} get-tuple-element(%custom-call), index=0
             })"),
           absl::string_view(R"(
             HloModule m
             ENTRY main {
               %keys = f32[10,0] parameter(0)
               %custom-call = (f32[10,0]{1,0}, u8[1]{0})
                 custom-call(%keys),
                 custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
                 backend_config={"descending":false}
               ROOT %t = f32[10,0]{1,0} get-tuple-element(%custom-call), index=0
             })"),
           absl::string_view(R"(
             HloModule m
             ENTRY main {
               %keys = f32[16,1] parameter(0)
               %custom-call = (f32[16,1]{1,0}, u8[1]{0})
                 custom-call(%keys),
                 custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
                 backend_config={"descending":false}
               ROOT %t = f32[16,1]{1,0} get-tuple-element(%custom-call), index=0
             })"),
       }) {
    SCOPED_TRACE(hlo);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(hlo));
    EXPECT_THAT(EstimateCubSortScratchSize(
                    "CUDA", stream_executor::GpuComputeCapability(cc))
                    .Run(module.get()),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

absl::StatusOr<std::unique_ptr<int64_t>> FakeCubSortKeysInstantiate(
    ffi::AnyBuffer, ffi::Result<ffi::AnyBuffer>, ffi::Result<ffi::BufferR1<U8>>,
    bool /*descending*/, int64_t batch_size) {
  return std::make_unique<int64_t>(batch_size * 1024);
}

absl::Status FakeCubSortDummyExecute() { return absl::OkStatus(); }

XLA_FFI_DEFINE_HANDLER(kFakeCubSortKeysInstantiate, FakeCubSortKeysInstantiate,
                       ffi::Ffi::BindInstantiate()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ret<ffi::BufferR1<U8>>()
                           .Attr<bool>("descending")
                           .Attr<int64_t>("batch_size"));

XLA_FFI_DEFINE_HANDLER(kFakeCubSortDummyExecute, FakeCubSortDummyExecute,
                       ffi::Ffi::Bind());

XLA_FFI_REGISTER_HANDLER(
    ffi::GetXlaFfiApi(),
    kCubDeviceRadixSortKeysUnassignedScratchSizeTarget.data(), "TEST_ROCM",
    {/*instantiate=*/kFakeCubSortKeysInstantiate, /*prepare=*/nullptr,
     /*initialize=*/nullptr, /*execute=*/kFakeCubSortDummyExecute});

TEST_F(EstimateCubSortScratchSizeTest, NonCudaInvokesFfiInstantiateHandler) {
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY main {
      %keys = f32[4,1000] parameter(0)
      %custom-call = (f32[4,1000]{1,0}, u8[1]{0})
        custom-call(%keys),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":true}
      ROOT %t = f32[4,1000]{1,0} get-tuple-element(%custom-call), index=0
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const stream_executor::GpuComputeCapability rocm_cc(
      stream_executor::RocmComputeCapability("gfx942"));
  ASSERT_THAT(
      EstimateCubSortScratchSize("TEST_ROCM", rocm_cc).Run(module.get()),
      IsOkAndHolds(true));
  const HloInstruction* call =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_EQ(call->custom_call_target(), kCubDeviceRadixSortKeysTarget);
  EXPECT_EQ(call->shape().tuple_shapes(1).element_type(), U8);
  EXPECT_EQ(call->shape().tuple_shapes(1).dimensions(0), 4096);
}

TEST_F(EstimateCubSortScratchSizeTest,
       NonCudaMissingFfiHandlerReturnsNotFound) {
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY main {
      %keys = f32[1000] parameter(0)
      %custom-call = (f32[1000]{0}, u8[1]{0})
        custom-call(%keys),
        custom_call_target="xla.gpu.ext.cub_sort_unassigned_scratch_size",
        backend_config={"descending":false}
      ROOT %t = f32[1000]{0} get-tuple-element(%custom-call), index=0
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const stream_executor::GpuComputeCapability rocm_cc(
      stream_executor::RocmComputeCapability("gfx942"));
  EXPECT_THAT(EstimateCubSortScratchSize("UNREGISTERED_PLATFORM", rocm_cc)
                  .Run(module.get()),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace xla::gpu
