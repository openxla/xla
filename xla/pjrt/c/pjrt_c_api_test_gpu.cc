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

#include "xla/pjrt/c/pjrt_c_api_gpu.h"
#include "xla/pjrt/c/pjrt_c_api_test.h"

namespace xla {
namespace pjrt {
namespace {

// Register GPU plugin as the C API for tests in pjrt_c_api_test.cc.
const bool kUnused =
    (RegisterTestCApiFactory([]() { return GetPjrtApi(); }), true);

}  // namespace
}  // namespace pjrt
}  // namespace xla
