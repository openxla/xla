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

#include "xla/pjrt/c/pjrt_c_api_test_base.h"

#include <gtest/gtest.h>
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_gpu.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"

namespace xla {
namespace pjrt {
class GpuTest : public PjrtCApiTestBase {
 public:
  GpuTest() : PjrtCApiTestBase() {
    api_ = GetPjrtApi();
    initialize();
  }
};

TEST_F(GpuTest, PlatformName) {
  PJRT_Client_PlatformName_Args args;
  args.client = client_;
  args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  args.priv = nullptr;
  PJRT_Error* error = api_->PJRT_Client_PlatformName(&args);
  ASSERT_EQ(error, nullptr);
  absl::string_view platform_name(args.platform_name, args.platform_name_size);
  ASSERT_EQ("gpu", platform_name);
}
}  // namespace pjrt
}  // namespace xla
