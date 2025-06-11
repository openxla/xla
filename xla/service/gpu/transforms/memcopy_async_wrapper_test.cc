/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/service/gpu/transforms/memcopy_async_wrapper.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using MemcopyAsyncWrapperTest = HloHardwareIndependentTestBase;

TEST_F(MemcopyAsyncWrapperTest, SimpleSliceIsWrapped) {
  const absl::string_view hlo_string = R"(
    dynamic_slice {
      p0 = f32[200] parameter(0)
      cn1 = s32[] constant(-1)
      ROOT slice = f32[100] dynamic-slice(p0, cn1), dynamic_slice_sizes={100}
    }

    ENTRY main {
      p0 = f32[200] parameter(0)
      ROOT fusion = f32[100] fusion(p0), kind=kLoop, calls=dynamic_slice
    }
  )";

  auto debug_options = HloHardwareIndependentTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_stream_annotation(true);
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  module->mutable_config().set_debug_options(debug_options);
  MemcopyAsyncWrapper wrapper_pass;

  TF_ASSERT_OK_AND_ASSIGN(bool mutated, wrapper_pass.Run(module.get()));
  absl::StatusOr<bool> filecheck_result = RunFileCheck(module->ToString({}), R"(
  // CHECK: ENTRY %main {{.*}}
  // CHECK: %[[PARAM:.*]] = f32[200]{0} parameter(0)
  // CHECK: %fusion-start = {{.*}} fusion-start(%[[PARAM]]), kind=kLoop, calls=%dynamic_slice.clone
  // CHECK: ROOT %fusion-done = f32[100]{0} fusion-done(%fusion-start)
  )");
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(*filecheck_result);

  ASSERT_TRUE(mutated);
}

}  // namespace
}  // namespace xla::gpu
