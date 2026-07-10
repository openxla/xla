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

#include "xla/backends/gpu/libraries/cutedsl/runtime_api.h"

#include <cstddef>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xla::gpu::cutedsl {
namespace {

using ::testing::HasSubstr;

CuteDSLRT_Error_t ModuleCreate(CuteDSLRT_Module_t**, const unsigned char*,
                               std::size_t, const char**, std::size_t) {
  return kCuteDslRtSuccess;
}

CuteDSLRT_Error_t ModuleGetFunction(CuteDSLRT_Function_t**, CuteDSLRT_Module_t*,
                                    const char*) {
  return kCuteDslRtSuccess;
}

CuteDSLRT_Error_t FunctionRun(void*, void**, std::size_t) {
  return kCuteDslRtSuccess;
}

CuteDSLRT_Error_t ModuleDestroy(CuteDSLRT_Module_t*) {
  return kCuteDslRtSuccess;
}

const char* GetErrorName(CuteDSLRT_Error_t) { return "Success"; }

const char* GetErrorString(CuteDSLRT_Error_t) { return "success"; }

RuntimeFunctions MakeValidFunctions() {
  return RuntimeFunctions{
      /*module_create_from_bytes=*/ModuleCreate,
      /*module_get_function=*/ModuleGetFunction,
      /*function_run=*/FunctionRun,
      /*module_destroy=*/ModuleDestroy,
      /*get_error_name=*/GetErrorName,
      /*get_error_string=*/GetErrorString,
  };
}

class ScopedRuntimeFunctions {
 public:
  explicit ScopedRuntimeFunctions(const RuntimeFunctions* functions)
      : status_(SetRuntimeFunctionsForTesting(functions)) {}

  ~ScopedRuntimeFunctions() { ResetRuntimeFunctionsForTesting(); }

  const absl::Status& status() const { return status_; }

 private:
  absl::Status status_;
};

void ExpectSameProvider(const absl::StatusOr<RuntimeFunctions>& lhs,
                        const absl::StatusOr<RuntimeFunctions>& rhs) {
  ASSERT_EQ(lhs.ok(), rhs.ok());
  if (!lhs.ok()) {
    EXPECT_EQ(lhs.status(), rhs.status());
    return;
  }
  EXPECT_EQ(lhs->module_create_from_bytes, rhs->module_create_from_bytes);
  EXPECT_EQ(lhs->module_get_function, rhs->module_get_function);
  EXPECT_EQ(lhs->function_run, rhs->function_run);
  EXPECT_EQ(lhs->module_destroy, rhs->module_destroy);
  EXPECT_EQ(lhs->get_error_name, rhs->get_error_name);
  EXPECT_EQ(lhs->get_error_string, rhs->get_error_string);
}

TEST(RuntimeApiTest, AcceptsCompleteFunctions) {
  RuntimeFunctions functions = MakeValidFunctions();
  EXPECT_TRUE(ValidateRuntimeFunctions(&functions).ok());
}

TEST(RuntimeApiTest, RejectsNullAndMissingFunctions) {
  absl::Status status = ValidateRuntimeFunctions(nullptr);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(), HasSubstr("must not be null"));

  RuntimeFunctions functions = MakeValidFunctions();
  functions.module_create_from_bytes = nullptr;
  EXPECT_THAT(ValidateRuntimeFunctions(&functions).message(),
              HasSubstr(kModuleCreateFromBytesSymbol));

  functions = MakeValidFunctions();
  functions.module_get_function = nullptr;
  EXPECT_THAT(ValidateRuntimeFunctions(&functions).message(),
              HasSubstr(kModuleGetFunctionSymbol));

  functions = MakeValidFunctions();
  functions.function_run = nullptr;
  EXPECT_THAT(ValidateRuntimeFunctions(&functions).message(),
              HasSubstr(kFunctionRunSymbol));

  functions = MakeValidFunctions();
  functions.module_destroy = nullptr;
  EXPECT_THAT(ValidateRuntimeFunctions(&functions).message(),
              HasSubstr(kModuleDestroySymbol));

  functions = MakeValidFunctions();
  functions.get_error_name = nullptr;
  EXPECT_THAT(ValidateRuntimeFunctions(&functions).message(),
              HasSubstr(kGetErrorNameSymbol));

  functions = MakeValidFunctions();
  functions.get_error_string = nullptr;
  EXPECT_THAT(ValidateRuntimeFunctions(&functions).message(),
              HasSubstr(kGetErrorStringSymbol));
}

TEST(RuntimeApiTest, BazelProviderIsValidOrReportsHowToConfigureIt) {
  absl::StatusOr<RuntimeFunctions> functions = GetRuntimeFunctions();
  if (functions.ok()) {
    EXPECT_TRUE(ValidateRuntimeFunctions(&*functions).ok());
    return;
  }

  EXPECT_EQ(functions.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(functions.status().message(), HasSubstr("cutedsl_runtime"));
  EXPECT_THAT(functions.status().message(), HasSubstr("libcute_dsl_runtime"));
}

TEST(RuntimeApiTest, TestOverrideCanBeResetToBazelProvider) {
  absl::StatusOr<RuntimeFunctions> original = GetRuntimeFunctions();
  RuntimeFunctions functions = MakeValidFunctions();

  {
    ScopedRuntimeFunctions override(&functions);
    ASSERT_TRUE(override.status().ok()) << override.status();
    absl::StatusOr<RuntimeFunctions> overridden = GetRuntimeFunctions();
    ASSERT_TRUE(overridden.ok()) << overridden.status();
    EXPECT_EQ(overridden->function_run, &FunctionRun);
  }

  ExpectSameProvider(GetRuntimeFunctions(), original);
}

TEST(RuntimeApiTest, RejectsInvalidTestOverride) {
  absl::StatusOr<RuntimeFunctions> original = GetRuntimeFunctions();
  RuntimeFunctions functions = MakeValidFunctions();
  functions.function_run = nullptr;

  EXPECT_FALSE(SetRuntimeFunctionsForTesting(&functions).ok());
  ExpectSameProvider(GetRuntimeFunctions(), original);
}

}  // namespace
}  // namespace xla::gpu::cutedsl
