/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/memory_space_propagation.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using MemorySpacePropagationTest = HloHardwareIndependentTestBase;

TEST_F(MemorySpacePropagationTest, PropagateHostMemorySpace) {
  constexpr absl::string_view kHlo = R"(
HloModule jit_loss_fn, entry_computation_layout={(f32[2,2]{1,0})->(f32[], f32[2,2]{1,0})}, allow_spmd_sharding_propagation_to_parameters={true}, allow_spmd_sharding_propagation_to_output={true,true}

%region_3.88.clone.clone (arg_tuple.6: (s32[], f32[2,2], s32[], f32[2,2], f32[2,2,2])) -> (s32[], f32[2,2], s32[], f32[2,2], f32[2,2,2]) {
  %arg_tuple.6 = (s32[], f32[2,2]{1,0}, s32[], f32[2,2]{1,0}, f32[2,2,2]{2,1,0}) parameter(0)
  %get-tuple-element.88 = s32[] get-tuple-element(%arg_tuple.6), index=0
  %constant.35 = s32[] constant(1)
  %add.36 = s32[] add(%get-tuple-element.88, %constant.35)
  %get-tuple-element.96 = f32[2,2]{1,0} get-tuple-element(%arg_tuple.6), index=3
  %get-tuple-element.89 = f32[2,2]{1,0} get-tuple-element(%arg_tuple.6), index=1
  %tuple.17 = (f32[2,2]{1,0}, f32[2,2]{1,0}) tuple(%get-tuple-element.96, %get-tuple-element.89)
  %opt-barrier.2 = (f32[2,2]{1,0}, f32[2,2]{1,0}) opt-barrier(%tuple.17)
  %get-tuple-element.101 = f32[2,2]{1,0} get-tuple-element(%opt-barrier.2), index=1
  %get-tuple-element.102 = f32[2,2]{1,0} get-tuple-element(%opt-barrier.2), index=0
  %cosine.2 = f32[2,2]{1,0} cosine(%get-tuple-element.102)
  %multiply.8 = f32[2,2]{1,0} multiply(%get-tuple-element.101, %cosine.2)
  %get-tuple-element.111 = s32[] get-tuple-element(%arg_tuple.6), index=2
  %get-tuple-element.120 = f32[2,2,2]{2,1,0} get-tuple-element(%arg_tuple.6), index=4
  %constant.39 = s32[] constant(-1)
  %multiply.9 = s32[] multiply(%get-tuple-element.88, %constant.39)
  %add.40 = s32[] add(%get-tuple-element.111, %multiply.9)
  %add.37 = s32[] add(%add.40, %constant.39)
  %constant.37 = s32[] constant(0)
  %compare.8 = pred[] compare(%add.37, %constant.37), direction=LT
  %add.39 = s32[] add(%add.40, %constant.35)
  %select.6 = s32[] select(%compare.8, %add.39, %add.37)
  %dynamic-slice.2 = f32[1,2,2]{2,1,0} dynamic-slice(%get-tuple-element.120, %select.6, %constant.37, %constant.37), dynamic_slice_sizes={1,2,2}
  %bitcast = f32[2,2]{1,0} bitcast(%dynamic-slice.2)
  ROOT %tuple.16 = (s32[], f32[2,2]{1,0}, s32[], f32[2,2]{1,0}, f32[2,2,2]{2,1,0}) tuple(%add.36, %multiply.8, %get-tuple-element.111, %bitcast, %get-tuple-element.120)
}

%region_4.102.clone (arg_tuple.5: (s32[], f32[2,2], s32[], f32[2,2], f32[2,2,2])) -> pred[] {
  %arg_tuple.5 = (s32[], f32[2,2]{1,0}, s32[], f32[2,2]{1,0}, f32[2,2,2]{2,1,0}) parameter(0)
  %get-tuple-element.81 = s32[] get-tuple-element(%arg_tuple.5), index=0
  %constant.34 = s32[] constant(2)
  ROOT %compare.7 = pred[] compare(%get-tuple-element.81, %constant.34), direction=LT
}

%region_0.29.clone.clone.sunk.clone (arg_tuple.8: (s32[], f32[2,2], f32[2,2,2], f32[2,2])) -> (s32[], f32[2,2], f32[2,2,2], f32[2,2]) {
  %arg_tuple.8 = (s32[], f32[2,2]{1,0}, f32[2,2,2]{2,1,0}, f32[2,2]{1,0}) parameter(0)
  %get-tuple-element.139 = s32[] get-tuple-element(%arg_tuple.8), index=0
  %constant.41 = s32[] constant(1)
  %add.43 = s32[] add(%get-tuple-element.139, %constant.41)
  %get-tuple-element.138 = f32[2,2]{1,0} get-tuple-element(%arg_tuple.8), index=1
  %sine.4 = f32[2,2]{1,0} sine(%get-tuple-element.138)
  %get-tuple-element.137 = f32[2,2,2]{2,1,0} get-tuple-element(%arg_tuple.8), index=2
  %get-tuple-element.136 = f32[2,2]{1,0} get-tuple-element(%arg_tuple.8), index=3
  %bitcast.1 = f32[1,2,2]{2,1,0} bitcast(%get-tuple-element.136)
  %constant.42 = s32[] constant(0)
  %compare.10 = pred[] compare(%get-tuple-element.139, %constant.42), direction=LT
  %constant.48 = s32[] constant(2)
  %add.47 = s32[] add(%get-tuple-element.139, %constant.48)
  %select.7 = s32[] select(%compare.10, %add.47, %get-tuple-element.139)
  %dynamic-update-slice.4 = f32[2,2,2]{2,1,0:S(5)} dynamic-update-slice(%get-tuple-element.137, %bitcast.1, %select.7, %constant.42, %constant.42)
  ROOT %tuple.20 = (s32[], f32[2,2]{1,0}, f32[2,2,2]{2,1,0}, f32[2,2]{1,0}) tuple(%add.43, %sine.4, %dynamic-update-slice.4, %get-tuple-element.138)
}

%region_1.44.clone.clone (arg_tuple.7: (s32[], f32[2,2], f32[2,2,2], f32[2,2])) -> pred[] {
  %arg_tuple.7 = (s32[], f32[2,2]{1,0}, f32[2,2,2]{2,1,0}, f32[2,2]{1,0}) parameter(0)
  %get-tuple-element.135 = s32[] get-tuple-element(%arg_tuple.7), index=0
  %constant.40 = s32[] constant(2)
  ROOT %compare.9 = pred[] compare(%get-tuple-element.135, %constant.40), direction=LT
}

%region_2.59 (Arg_0.60: f32[], Arg_1.61: f32[]) -> f32[] {
  %Arg_0.60 = f32[] parameter(0)
  %Arg_1.61 = f32[] parameter(1)
  ROOT %add.62 = f32[] add(%Arg_0.60, %Arg_1.61)
}

ENTRY %main.123 (Arg_0.1: f32[2,2]) -> (f32[], f32[2,2]) {
  %constant.5 = s32[] constant(0)
  %Arg_0.1 = f32[2,2]{1,0} parameter(0)
  %sine.7 = f32[2,2]{1,0} sine(%Arg_0.1)
  %custom-call.8 = f32[2,2,2]{2,1,0:S(5)} custom-call(), custom_call_target="AllocateBuffer", api_version=API_VERSION_TYPED_FFI
  %tuple.19 = (s32[], f32[2,2]{1,0}, f32[2,2,2]{2,1,0}, f32[2,2]{1,0}) tuple(%constant.5, %sine.7, %custom-call.8, %Arg_0.1)
  %while.2 = (s32[], f32[2,2]{1,0}, f32[2,2,2]{2,1,0}, f32[2,2]{1,0}) while(%tuple.19), condition=%region_1.44.clone.clone, body=%region_0.29.clone.clone.sunk.clone, backend_config={"known_trip_count":{"n":"2"},"known_init_step":{"init":"0","step":"1"},"known_induction_variable":{"tuple_index":"0"}}
  %get-tuple-element.55 = f32[2,2]{1,0} get-tuple-element(%while.2), index=1
  %bitcast.2 = f32[4]{0} bitcast(%get-tuple-element.55)
  %constant.4 = f32[] constant(0)
  %reduce.63 = f32[] reduce(%bitcast.2, %constant.4), dimensions={0}, to_apply=%region_2.59
  %constant.2 = f32[] constant(1)
  %broadcast.3 = f32[2,2]{1,0} broadcast(%constant.2), dimensions={}
  %get-tuple-element.45 = s32[] get-tuple-element(%while.2), index=0
  %get-tuple-element.58 = f32[2,2]{1,0} get-tuple-element(%while.2), index=3
  %get-tuple-element.57 = f32[2,2,2]{2,1,0} get-tuple-element(%while.2), index=2
  %tuple.65 = (s32[], f32[2,2]{1,0}, s32[], f32[2,2]{1,0}, f32[2,2,2]{2,1,0}) tuple(%constant.5, %broadcast.3, %get-tuple-element.45, %get-tuple-element.58, %get-tuple-element.57)
  %while.1 = (s32[], f32[2,2]{1,0}, s32[], f32[2,2]{1,0}, f32[2,2,2]{2,1,0}) while(%tuple.65), condition=%region_4.102.clone, body=%region_3.88.clone.clone, backend_config={"known_trip_count":{"n":"2"},"known_init_step":{"init":"0","step":"1"},"known_induction_variable":{"tuple_index":"0"}}
  %get-tuple-element.115 = f32[2,2]{1,0} get-tuple-element(%while.1), index=3
  %get-tuple-element.113 = f32[2,2]{1,0} get-tuple-element(%while.1), index=1
  %tuple.116 = (f32[2,2]{1,0}, f32[2,2]{1,0}) tuple(%get-tuple-element.115, %get-tuple-element.113)
  %opt-barrier.117 = (f32[2,2]{1,0}, f32[2,2]{1,0}) opt-barrier(%tuple.116)
  %get-tuple-element.119 = f32[2,2]{1,0} get-tuple-element(%opt-barrier.117), index=1
  %get-tuple-element.118 = f32[2,2]{1,0} get-tuple-element(%opt-barrier.117), index=0
  %cosine.120 = f32[2,2]{1,0} cosine(%get-tuple-element.118)
  %multiply.121 = f32[2,2]{1,0} multiply(%get-tuple-element.119, %cosine.120)
  ROOT %tuple.122 = (f32[], f32[2,2]{1,0}) tuple(%reduce.63, %multiply.121)
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloPass(MemorySpacePropagation(), module.get()));
  EXPECT_TRUE(changed);
  TF_EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                           /*allow_mixed_precision=*/false)
                   .Run(module.get())
                   .status());
}

}  // namespace
}  // namespace xla::gpu
