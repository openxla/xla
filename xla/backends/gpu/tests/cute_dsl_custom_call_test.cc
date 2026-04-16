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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/ffi.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

XLA_FFI_DECLARE_HANDLER_SYMBOL(CuteDSLRT_NvJaxCutlassCallExecute);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "CuteDSLRT_NvJaxCutlassCall",
                          "CUDA", CuteDSLRT_NvJaxCutlassCallExecute);

namespace xla::gpu {
namespace {

class CuteDslCustomCallTest : public HloPjRtGpuTestBase {};

TEST_F(CuteDslCustomCallTest, RunVectorAdd) {
  std::string hlo_path =
      tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "backends", "gpu", "tests",
                        "cute_dsl_vector_add.hlo");
  std::string hlo_text;
  TF_ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), hlo_path, &hlo_text));

  EXPECT_TRUE(Run(hlo_text, /*run_hlo_passes=*/true));
}

}  // namespace
}  // namespace xla::gpu
