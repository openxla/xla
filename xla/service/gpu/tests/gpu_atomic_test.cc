/* Copyright 2018 The OpenXLA Authors.

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


#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class GpuAtomicTest : public GpuCodegenTest {};

TEST_F(GpuAtomicTest, TestStore) {
  const char* hlo_string = R"(
    HloModule TensorFlowScatterV1

    update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
      lhs = s32[] parameter(0)
      ROOT rhs = s32[] parameter(1)
    }

    ENTRY main {
      operand = s32[3,3] parameter(0)
      indices = s32[2] parameter(1)
      updates = s32[2,3] parameter(2)
      ROOT scatter = s32[3,3] scatter(operand, indices, updates),
          to_apply=update_s32,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    }
)";

  CompileAndVerifyIr(hlo_string, R"(
CHECK: store atomic{{.*}}unordered, align 4
)");
}

TEST_F(GpuAtomicTest, TestStoreNoAtomic) {
  const char* hlo_string = R"(
    HloModule TensorFlowScatterV1

    update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
      lhs = s32[] parameter(0)
      ROOT rhs = s32[] parameter(1)
    }

    ENTRY main {
      operand = s32[3,3] parameter(0)
      indices = s32[2] parameter(1)
      updates = s32[2,3] parameter(2)
      ROOT scatter = s32[3,3] scatter(operand, indices, updates),
          to_apply=update_s32,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1, unique_indices=true
    }
)";

  CompileAndVerifyIr(hlo_string, R"(
CHECK-NOT: store atomic{{.*}}unordered, align 4
)");
}

TEST_F(GpuAtomicTest, TestAddAtomicF32) {
  if (is_built_with_rocm_ &&
      !backend()
           .default_stream_executor()
           ->GetDeviceDescription()
           .rocm_compute_capability()
           .gfx9_mi100_or_later()) {
    return;
  }
  const char* hlo_string = R"(
    HloModule TensorFlowScatterV1

    update_f32 (lhs: f32[], rhs: f32[]) -> f32[] {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY main {
      operand = f32[3,3] parameter(0)
      indices = s32[2] parameter(1)
      updates = f32[2,3] parameter(2)
      ROOT scatter = f32[3,3] scatter(operand, indices, updates),
          to_apply=update_f32,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1, unique_indices=false
    }
)";

  HloModuleConfig config;
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_gpu_ftz(true);
  config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  CompileAndVerifyIr(std::move(module), is_built_with_rocm_ ? R"(
CHECK: call float @llvm.amdgcn.global.atomic.fadd.f32.p1.f32(ptr addrspace(1) %[[ADDR:.*]], float %[[VALUE:.*]])
)"
                                                     : R"(
CHECK: atomicrmw fadd ptr %[[ADDR:.*]], float %[[VALUE:.*]] seq_cst
)", false);
}

TEST_F(GpuAtomicTest, TestAddAtomicF64) {
  // Atomic add required sm_60 or above.
  if (!is_built_with_rocm_ &&
      !backend()
           .default_stream_executor()
           ->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeast(6)) {
    return;
  }

  if (is_built_with_rocm_ &&
      !backend()
           .default_stream_executor()
           ->GetDeviceDescription()
           .rocm_compute_capability()
           .gfx9_mi200_or_later()) {
    return;
  }

  const char* hlo_string = R"(
    HloModule TensorFlowScatterV1

    update_f64 (lhs: f64[], rhs: f64[]) -> f64[] {
      lhs = f64[] parameter(0)
      rhs = f64[] parameter(1)
      ROOT add = f64[] add(lhs, rhs)
    }

    ENTRY main {
      operand = f64[3,3] parameter(0)
      indices = s32[2] parameter(1)
      updates = f64[2,3] parameter(2)
      ROOT scatter = f64[3,3] scatter(operand, indices, updates),
          to_apply=update_f64,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1, unique_indices=false
    }
)";

  CompileAndVerifyIr(hlo_string, is_built_with_rocm_ ? R"(
CHECK: call double @llvm.amdgcn.global.atomic.fadd.f64.p1.f64(ptr addrspace(1) %[[ADDR:.*]], double %[[VALUE:.*]])
)"
                                                     : R"(
CHECK: atomicrmw fadd ptr %[[ADDR:.*]], double %[[VALUE:.*]] seq_cst
)");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
