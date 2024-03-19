/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/service/gpu/multi_streaming_scheduling.h"

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/status.h"

namespace xla::gpu {
namespace {

class MultiStreamingSchedulingTest : public HloTestBase {};

TEST_F(MultiStreamingSchedulingTest, MultiStreamingScheduling) {
  const char* hlo = R"(
    HloModule test_module, is_scheduled=true, entry_computation_layout={(f32[1024,1024]{1,0}, f32[1024,1024]{1,0}, f32[1024,1024]{1,0}, f32[1024,1024]{1,0})->(f32[1024,1024]{1,0}, f32[1024,1024]{1,0}, f32[1024,1024]{1,0})}

    %triton_gemm_dot.5_computation (parameter_0: f32[1024,1024], parameter_1: f32[1024,1024]) -> f32[1024,1024] {
      %parameter_0 = f32[1024,1024]{1,0} parameter(0)
      %parameter_1 = f32[1024,1024]{1,0} parameter(1)
      ROOT %dot.0 = f32[1024,1024]{1,0} dot(f32[1024,1024]{1,0} %parameter_0, f32[1024,1024]{1,0} %parameter_1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    %triton_gemm_dot.7_computation (parameter_0.1: f32[1024,1024], parameter_1.1: f32[1024,1024]) -> f32[1024,1024] {
      %parameter_0.1 = f32[1024,1024]{1,0} parameter(0)
      %parameter_1.1 = f32[1024,1024]{1,0} parameter(1)
      ROOT %dot.1 = f32[1024,1024]{1,0} dot(f32[1024,1024]{1,0} %parameter_0.1, f32[1024,1024]{1,0} %parameter_1.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    %wrapped_tanh_computation (param_0: f32[1024,1024]) -> f32[1024,1024] {
      %param_0 = f32[1024,1024]{1,0} parameter(0)
      ROOT %tanh.2.1 = f32[1024,1024]{1,0} tanh(f32[1024,1024]{1,0} %param_0)
    }

    ENTRY %main.9 (Arg_0.1.0: f32[1024,1024], Arg_1.2.0: f32[1024,1024], Arg_2.3.0: f32[1024,1024], Arg_3.4.0: f32[1024,1024]) -> (f32[1024,1024], f32[1024,1024], f32[1024,1024]) {
      %Arg_3.4.0 = f32[1024,1024]{1,0} parameter(3), sharding={replicated}
      %Arg_2.3.0 = f32[1024,1024]{1,0} parameter(2), sharding={replicated}
      %Arg_1.2.0 = f32[1024,1024]{1,0} parameter(1), sharding={replicated}
      %Arg_0.1.0 = f32[1024,1024]{1,0} parameter(0), sharding={replicated}
      %triton_gemm_dot.5.0 = f32[1024,1024]{1,0} fusion(f32[1024,1024]{1,0} %Arg_0.1.0, f32[1024,1024]{1,0} %Arg_1.2.0), kind=kCustom, calls=%triton_gemm_dot.5_computation, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_gemm","triton_gemm_config":{"block_m":"128","block_n":"128","block_k":"32","split_k":"1","num_stages":"4","num_warps":"4","num_ctas":"1"}}}
      %wrapped_tanh = f32[1024,1024]{1,0} fusion(f32[1024,1024]{1,0} %triton_gemm_dot.5.0), kind=kLoop, calls=%wrapped_tanh_computation
      %triton_gemm_dot.7.0 = f32[1024,1024]{1,0} fusion(f32[1024,1024]{1,0} %Arg_2.3.0, f32[1024,1024]{1,0} %Arg_3.4.0), kind=kCustom, calls=%triton_gemm_dot.7_computation, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_gemm","triton_gemm_config":{"block_m":"128","block_n":"128","block_k":"32","split_k":"1","num_stages":"4","num_warps":"4","num_ctas":"1"}}}
      ROOT %tuple.8.0 = (f32[1024,1024]{1,0}, f32[1024,1024]{1,0}, f32[1024,1024]{1,0}) tuple(f32[1024,1024]{1,0} %triton_gemm_dot.5.0, f32[1024,1024]{1,0} %wrapped_tanh, f32[1024,1024]{1,0} %triton_gemm_dot.7.0)
    })";

  const char* expected = R"(
// CHECK:  %[[P3:.+]] = f32[1024,1024]{1,0} parameter(3), sharding={replicated}
// CHECK:  %[[P2:.+]] = f32[1024,1024]{1,0} parameter(2), sharding={replicated}
// CHECK:  %[[P1:.+]] = f32[1024,1024]{1,0} parameter(1), sharding={replicated}
// CHECK:  %[[P0:.+]] = f32[1024,1024]{1,0} parameter(0), sharding={replicated}
// CHECK: %[[START:.+]] = ((f32[1024,1024]{1,0}, f32[1024,1024]{1,0}), f32[1024,1024]{1,0}) fusion-start(%[[P0]], %[[P1]])
// CHECK-SAME: "operation_queue_id":"1"
// CHECK: f32[1024,1024]{1,0} fusion(%[[P2]], %[[P3]]), kind=kCustom, calls=%triton_gemm_dot.7_computation
// CHECK: fusion-done(%[[START]])
)";

  RunAndFilecheckHloRewrite(hlo, MultiStreamingScheduling(), expected,
                            [](HloModule* module) {
                              EXPECT_TRUE(module->has_schedule());
                              TF_CHECK_OK(module->schedule().Verify());
                            });
}

TEST_F(MultiStreamingSchedulingTest, MoveInstructionsUsedBySecondInstr) {
  const char* hlo = R"(
    HloModule test_module, is_scheduled=true, entry_computation_layout={(f32[1024,1024]{1,0}, f32[1024,1024]{1,0}, (f32[1024,1024]{1,0}, f32[1024,1024]{1,0}))->(f32[1024,1024]{1,0}, f32[1024,1024]{1,0}, f32[1024,1024]{1,0})}

    %triton_gemm_dot.5_computation (parameter_0: f32[1024,1024], parameter_1: f32[1024,1024]) -> f32[1024,1024] {
      %parameter_0 = f32[1024,1024]{1,0} parameter(0)
      %parameter_1 = f32[1024,1024]{1,0} parameter(1)
      ROOT %dot.0 = f32[1024,1024]{1,0} dot(f32[1024,1024]{1,0} %parameter_0, f32[1024,1024]{1,0} %parameter_1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    %triton_gemm_dot.7_computation (parameter_0.1: f32[1024,1024], parameter_1.1: f32[1024,1024]) -> f32[1024,1024] {
      %parameter_0.1 = f32[1024,1024]{1,0} parameter(0)
      %parameter_1.1 = f32[1024,1024]{1,0} parameter(1)
      ROOT %dot.1 = f32[1024,1024]{1,0} dot(f32[1024,1024]{1,0} %parameter_0.1, f32[1024,1024]{1,0} %parameter_1.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    %wrapped_tanh_computation (param_0: f32[1024,1024]) -> f32[1024,1024] {
      %param_0 = f32[1024,1024]{1,0} parameter(0)
      ROOT %tanh.2.1 = f32[1024,1024]{1,0} tanh(f32[1024,1024]{1,0} %param_0)
    }

    ENTRY %main.9 (Arg_0.1.0: f32[1024,1024], Arg_1.2.0: f32[1024,1024], Arg_tuple: (f32[1024,1024], f32[1024,1024])) -> (f32[1024,1024], f32[1024,1024], f32[1024,1024]) {
      %Arg_tuple = (f32[1024,1024]{1,0}, f32[1024,1024]{1,0}) parameter(2), sharding={replicated}
      %Arg_1.2.0 = f32[1024,1024]{1,0} parameter(1), sharding={replicated}
      %Arg_0.1.0 = f32[1024,1024]{1,0} parameter(0), sharding={replicated}
      %triton_gemm_dot.5.0 = f32[1024,1024]{1,0} fusion(f32[1024,1024]{1,0} %Arg_0.1.0, f32[1024,1024]{1,0} %Arg_1.2.0), kind=kCustom, calls=%triton_gemm_dot.5_computation, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_gemm","triton_gemm_config":{"block_m":"128","block_n":"128","block_k":"32","split_k":"1","num_stages":"4","num_warps":"4","num_ctas":"1"}}}
      %wrapped_tanh = f32[1024,1024]{1,0} fusion(f32[1024,1024]{1,0} %triton_gemm_dot.5.0), kind=kLoop, calls=%wrapped_tanh_computation
      %get-tuple-element.0 = f32[1024,1024]{1,0} get-tuple-element((f32[1024,1024]{1,0}, f32[1024,1024]{1,0}) Arg_tuple), index=0
      %get-tuple-element.1 = f32[1024,1024]{1,0} get-tuple-element((f32[1024,1024]{1,0}, f32[1024,1024]{1,0}) Arg_tuple), index=1
      %triton_gemm_dot.7.0 = f32[1024,1024]{1,0} fusion(f32[1024,1024]{1,0} %get-tuple-element.0, f32[1024,1024]{1,0} %get-tuple-element.1), kind=kCustom, calls=%triton_gemm_dot.7_computation, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_gemm","triton_gemm_config":{"block_m":"128","block_n":"128","block_k":"32","split_k":"1","num_stages":"4","num_warps":"4","num_ctas":"1"}}}
      ROOT %tuple.8.0 = (f32[1024,1024]{1,0}, f32[1024,1024]{1,0}, f32[1024,1024]{1,0}) tuple(f32[1024,1024]{1,0} %triton_gemm_dot.5.0, f32[1024,1024]{1,0} %wrapped_tanh, f32[1024,1024]{1,0} %triton_gemm_dot.7.0)
    })";

  const char* expected = R"(
// CHECK: %[[P2:.+]] = (f32[1024,1024]{1,0}, f32[1024,1024]{1,0}) parameter(2)
// CHECK: %[[P1:.+]] = f32[1024,1024]{1,0} parameter(1)
// CHECK: %[[P0:.+]] = f32[1024,1024]{1,0} parameter(0)
// CHECK: %[[T0:.+]] = f32[1024,1024]{1,0} get-tuple-element(%[[P2]]), index=0
// CHECK: %[[T1:.+]] = f32[1024,1024]{1,0} get-tuple-element(%[[P2]]), index=1
// CHECK: %[[START:.+]] = ((f32[1024,1024]{1,0}, f32[1024,1024]{1,0}), f32[1024,1024]{1,0}) fusion-start(%[[P0]], %[[P1]])
// CHECK-SAME: "operation_queue_id":"1"
// CHECK: f32[1024,1024]{1,0} fusion(%[[T0]], %[[T1]]), kind=kCustom, calls=%triton_gemm_dot.7_computation
// CHECK: fusion-done(%[[START]])
)";

  RunAndFilecheckHloRewrite(hlo, MultiStreamingScheduling(), expected,
                            [](HloModule* module) {
                              EXPECT_TRUE(module->has_schedule());
                              TF_CHECK_OK(module->schedule().Verify());
                            });
}

}  // namespace

}  // namespace xla::gpu
