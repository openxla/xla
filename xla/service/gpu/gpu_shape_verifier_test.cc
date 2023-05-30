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

#include "xla/service/gpu/gpu_shape_verifier.h"

#include "xla/service/hlo_parser.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/test.h"

namespace xla::gpu {
namespace {

using ::testing::HasSubstr;

TEST(GpuShapeVerifierTest, MultiOutputFusionShapes) {
  const char* const hlo_string = R"(
  HloModule Module

  fusion_computation {
    ROOT root = tuple(s32[] constant(0), s32[2] constant({0, 1}))
  }
  ENTRY entry {
    ROOT fusion = (s32[], s32[2]) fusion(), calls=fusion_computation, kind=kLoop
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  HloVerifierOpts opts;
  GpuShapeVerifier verifier(opts);
  auto status = module->entry_computation()->Accept(&verifier);
  EXPECT_THAT(status.message(), HasSubstr("outputs must have the same shape"));
}

TEST(GpuShapeVerifierTest, MultiOutputFusionShapesLayoutSensitive) {
  const char* const hlo_string = R"(
  HloModule Module

  fusion_computation {
    ROOT root = tuple(s32[2,2]{1,0} constant({{0,1},{2,3}}),
                      s32[2,2]{0,1} constant({{0,1},{2,3}}))
  }
  ENTRY entry {
    ROOT fusion = (s32[2,2]{1,0}, s32[2,2]{0,1}) fusion(), calls=fusion_computation, kind=kLoop
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  {
    HloVerifierOpts opts;
    GpuShapeVerifier verifier(opts);
    auto status = module->entry_computation()->Accept(&verifier);
    TF_EXPECT_OK(status);
  }

  {
    HloVerifierOpts opts;
    opts.MakeLayoutSensitive();
    GpuShapeVerifier verifier(opts);
    auto status = module->entry_computation()->Accept(&verifier);
    EXPECT_THAT(status.message(),
                HasSubstr("outputs must have the same shape"));
  }
}

}  // namespace
}  // namespace xla::gpu
