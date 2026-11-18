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

// Tests `LoadFunctionLibrary` in a binary that doesn't link
// `:cpu_aot_object_file_loader` (like AOT runtimes).

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/service/cpu/cpu_aot_loader.h"
#include "xla/service/cpu/executable.pb.h"

namespace xla::cpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(CpuAotLoaderWithoutObjectFileLoaderTest,
     ReturnsFailedPreconditionWithoutRegisteredLoader) {
  EXPECT_THAT(LoadFunctionLibrary(/*compiled_symbols=*/{}, /*obj_files=*/{},
                                  /*data_layout_str=*/""),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("cpu_aot_object_file_loader")));
}

TEST(CpuAotLoaderWithoutObjectFileLoaderTest, UsesRegisteredLoader) {
  RegisterObjectFileLoader(
      [](const std::vector<FunctionLibrary::Symbol>&,
         absl::Span<const ObjFileProto>, absl::string_view)
          -> absl::StatusOr<std::unique_ptr<FunctionLibrary>> {
        return absl::UnavailableError("registered loader");
      });
  EXPECT_THAT(
      LoadFunctionLibrary(/*compiled_symbols=*/{}, /*obj_files=*/{},
                          /*data_layout_str=*/""),
      StatusIs(absl::StatusCode::kUnavailable, HasSubstr("registered loader")));
  RegisterObjectFileLoader(nullptr);
}

}  // namespace
}  // namespace xla::cpu
