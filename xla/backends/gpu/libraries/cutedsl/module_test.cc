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

#include "xla/backends/gpu/libraries/cutedsl/module.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"
#include "xla/backends/gpu/libraries/cutedsl/runtime_api.h"

namespace xla::gpu::cutedsl {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

struct FakeRuntime {
  int create_count = 0;
  int get_function_count = 0;
  int destroy_count = 0;
  CuteDSLRT_Error_t create_error = kCuteDslRtSuccess;
  CuteDSLRT_Error_t get_function_error = kCuteDslRtSuccess;
  bool create_module_on_error = false;
  bool return_null_function = false;
  std::vector<std::string> function_prefixes;
};

FakeRuntime* fake_runtime = nullptr;

CuteDSLRT_Error_t ModuleCreate(CuteDSLRT_Module_t** module,
                               const unsigned char*, size_t, const char**,
                               size_t) {
  ++fake_runtime->create_count;
  if (fake_runtime->create_error != kCuteDslRtSuccess) {
    if (fake_runtime->create_module_on_error) {
      *module = reinterpret_cast<CuteDSLRT_Module_t*>(fake_runtime);
    }
    return fake_runtime->create_error;
  }
  *module = reinterpret_cast<CuteDSLRT_Module_t*>(fake_runtime);
  return kCuteDslRtSuccess;
}

CuteDSLRT_Error_t ModuleGetFunction(CuteDSLRT_Function_t** function,
                                    CuteDSLRT_Module_t* module,
                                    const char* prefix) {
  EXPECT_EQ(module, reinterpret_cast<CuteDSLRT_Module_t*>(fake_runtime));
  ++fake_runtime->get_function_count;
  fake_runtime->function_prefixes.emplace_back(prefix);
  if (fake_runtime->get_function_error != kCuteDslRtSuccess) {
    return fake_runtime->get_function_error;
  }
  *function = fake_runtime->return_null_function
                  ? nullptr
                  : reinterpret_cast<CuteDSLRT_Function_t*>(fake_runtime);
  return kCuteDslRtSuccess;
}

CuteDSLRT_Error_t FunctionRun(void*, void**, size_t) {
  return kCuteDslRtSuccess;
}

CuteDSLRT_Error_t ModuleDestroy(CuteDSLRT_Module_t* module) {
  EXPECT_EQ(module, reinterpret_cast<CuteDSLRT_Module_t*>(fake_runtime));
  ++fake_runtime->destroy_count;
  return kCuteDslRtSuccess;
}

const char* GetErrorName(CuteDSLRT_Error_t) { return "FakeRuntimeError"; }
const char* GetErrorString(CuteDSLRT_Error_t) { return "fake failure"; }

const RuntimeFunctions kFakeFunctions = {
    /*module_create_from_bytes=*/ModuleCreate,
    /*module_get_function=*/ModuleGetFunction,
    /*function_run=*/FunctionRun,
    /*module_destroy=*/ModuleDestroy,
    /*get_error_name=*/GetErrorName,
    /*get_error_string=*/GetErrorString,
};

class ScopedFakeRuntime {
 public:
  explicit ScopedFakeRuntime(FakeRuntime* runtime) {
    EXPECT_EQ(fake_runtime, nullptr);
    fake_runtime = runtime;
    status_ = SetRuntimeFunctionsForTesting(&kFakeFunctions);
  }

  ~ScopedFakeRuntime() {
    ResetRuntimeFunctionsForTesting();
    fake_runtime = nullptr;
  }

  const absl::Status& status() const { return status_; }

 private:
  absl::Status status_;
};

std::string Sha256(absl::string_view value) {
  llvm::SHA256 hasher;
  hasher.update(llvm::StringRef(value.data(), value.size()));
  std::array<uint8_t, kModuleCacheKeySize> digest = hasher.final();
  return std::string(reinterpret_cast<const char*>(digest.data()),
                     digest.size());
}

TEST(CuteDslModuleTest, CachesModulesByDigestAndScope) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);
  ASSERT_TRUE(scoped_runtime.status().ok()) << scoped_runtime.status();

  const std::string bytes = "module cache scope test";
  const std::string key = Sha256(bytes);
  int scope = 0;

  absl::StatusOr<std::shared_ptr<LoadedModule>> first =
      GetOrLoadModule(bytes, key);
  ASSERT_TRUE(first.ok()) << first.status();
  absl::StatusOr<std::shared_ptr<LoadedModule>> same =
      GetOrLoadModule(bytes, key);
  ASSERT_TRUE(same.ok()) << same.status();
  absl::StatusOr<std::shared_ptr<LoadedModule>> scoped =
      GetOrLoadModule(bytes, key, &scope);
  ASSERT_TRUE(scoped.ok()) << scoped.status();

  EXPECT_EQ(first->get(), same->get());
  EXPECT_NE(first->get(), scoped->get());
  EXPECT_EQ(runtime.create_count, 2);
}

TEST(CuteDslModuleTest, WeakCacheReloadsAfterLastOwnerReleases) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);
  ASSERT_TRUE(scoped_runtime.status().ok()) << scoped_runtime.status();

  const std::string bytes = "module weak cache test";
  const std::string key = Sha256(bytes);
  {
    absl::StatusOr<std::shared_ptr<LoadedModule>> loaded =
        GetOrLoadModule(bytes, key);
    ASSERT_TRUE(loaded.ok()) << loaded.status();
    EXPECT_EQ(runtime.create_count, 1);
  }
  EXPECT_EQ(runtime.destroy_count, 1);

  absl::StatusOr<std::shared_ptr<LoadedModule>> reloaded =
      GetOrLoadModule(bytes, key);
  ASSERT_TRUE(reloaded.ok()) << reloaded.status();
  EXPECT_EQ(runtime.create_count, 2);
}

TEST(CuteDslModuleTest, CachesMultipleFunctionPrefixesPerModule) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);
  ASSERT_TRUE(scoped_runtime.status().ok()) << scoped_runtime.status();

  const std::string bytes = "module function cache test";
  absl::StatusOr<std::shared_ptr<LoadedModule>> loaded =
      GetOrLoadModule(bytes, Sha256(bytes));
  ASSERT_TRUE(loaded.ok()) << loaded.status();

  absl::StatusOr<CuteDSLRT_Function_t*> first =
      (*loaded)->GetFunction("cutlass_call");
  ASSERT_TRUE(first.ok()) << first.status();
  absl::StatusOr<CuteDSLRT_Function_t*> repeated =
      (*loaded)->GetFunction("cutlass_call");
  ASSERT_TRUE(repeated.ok()) << repeated.status();
  absl::StatusOr<CuteDSLRT_Function_t*> second =
      (*loaded)->GetFunction("cutlass_call_1");
  ASSERT_TRUE(second.ok()) << second.status();

  EXPECT_EQ(*first, *repeated);
  EXPECT_EQ(runtime.create_count, 1);
  EXPECT_EQ(runtime.get_function_count, 2);
  EXPECT_THAT(runtime.function_prefixes,
              ElementsAre("cutlass_call", "cutlass_call_1"));
}

TEST(CuteDslModuleTest, RejectsInvalidDigestBeforeLoadingRuntimeModule) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);
  ASSERT_TRUE(scoped_runtime.status().ok()) << scoped_runtime.status();

  absl::StatusOr<std::shared_ptr<LoadedModule>> loaded = GetOrLoadModule(
      "module with bad key", std::string(kModuleCacheKeySize, 'x'));
  EXPECT_EQ(loaded.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(loaded.status().message(), HasSubstr("does not match"));
  EXPECT_EQ(runtime.create_count, 0);
}

TEST(CuteDslModuleTest, RejectsInvalidFunctionPrefixesWithoutRuntimeLookup) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);
  ASSERT_TRUE(scoped_runtime.status().ok()) << scoped_runtime.status();

  const std::string bytes = "module function prefix validation test";
  absl::StatusOr<std::shared_ptr<LoadedModule>> loaded =
      GetOrLoadModule(bytes, Sha256(bytes));
  ASSERT_TRUE(loaded.ok()) << loaded.status();

  EXPECT_EQ((*loaded)->GetFunction("").status().code(),
            absl::StatusCode::kInvalidArgument);
  const std::string embedded_null("cutlass\0call", 12);
  EXPECT_EQ((*loaded)->GetFunction(embedded_null).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(runtime.get_function_count, 0);
}

TEST(CuteDslModuleTest, ReportsNullFunctionAndRetainsModuleOwnership) {
  FakeRuntime runtime;
  runtime.return_null_function = true;
  ScopedFakeRuntime scoped_runtime(&runtime);
  ASSERT_TRUE(scoped_runtime.status().ok()) << scoped_runtime.status();

  const std::string bytes = "module null function test";
  absl::StatusOr<std::shared_ptr<LoadedModule>> loaded =
      GetOrLoadModule(bytes, Sha256(bytes));
  ASSERT_TRUE(loaded.ok()) << loaded.status();

  absl::StatusOr<CuteDSLRT_Function_t*> function =
      (*loaded)->GetFunction("cutlass_call_2");
  EXPECT_EQ(function.status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(function.status().message(), HasSubstr("null cutlass_call_2"));
  EXPECT_EQ(runtime.destroy_count, 0);
}

}  // namespace
}  // namespace xla::gpu::cutedsl
