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
#include "CuteDSLRuntime.h"

namespace xla::gpu::cutedsl {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

struct FakeRuntime {
  int create_count = 0;
  int get_function_count = 0;
  int destroy_count = 0;
  CuteDSLRT_Error_t create_error = CuteDSLRT_Error_Success;
  CuteDSLRT_Error_t get_function_error = CuteDSLRT_Error_Success;
  bool create_module_on_error = false;
  bool return_null_function = false;
  std::vector<std::string> function_prefixes;
};

FakeRuntime* fake_runtime = nullptr;

CuteDSLRT_Error_t ModuleCreate(CuteDSLRT_Module_t** module,
                               const unsigned char*, size_t, const char**,
                               size_t) {
  ++fake_runtime->create_count;
  if (fake_runtime->create_error != CuteDSLRT_Error_Success) {
    if (fake_runtime->create_module_on_error) {
      *module = reinterpret_cast<CuteDSLRT_Module_t*>(fake_runtime);
    }
    return fake_runtime->create_error;
  }
  *module = reinterpret_cast<CuteDSLRT_Module_t*>(fake_runtime);
  return CuteDSLRT_Error_Success;
}

CuteDSLRT_Error_t ModuleGetFunction(CuteDSLRT_Function_t** function,
                                    CuteDSLRT_Module_t* module,
                                    const char* prefix) {
  EXPECT_EQ(module, reinterpret_cast<CuteDSLRT_Module_t*>(fake_runtime));
  ++fake_runtime->get_function_count;
  fake_runtime->function_prefixes.emplace_back(prefix);
  if (fake_runtime->get_function_error != CuteDSLRT_Error_Success) {
    return fake_runtime->get_function_error;
  }
  *function = fake_runtime->return_null_function
                  ? nullptr
                  : reinterpret_cast<CuteDSLRT_Function_t*>(fake_runtime);
  return CuteDSLRT_Error_Success;
}

CuteDSLRT_Error_t FunctionRun(void*, void**, size_t) {
  return CuteDSLRT_Error_Success;
}

CuteDSLRT_Error_t ModuleDestroy(CuteDSLRT_Module_t* module) {
  EXPECT_EQ(module, reinterpret_cast<CuteDSLRT_Module_t*>(fake_runtime));
  ++fake_runtime->destroy_count;
  return CuteDSLRT_Error_Success;
}

const char* GetErrorName(CuteDSLRT_Error_t) { return "FakeRuntimeError"; }
const char* GetErrorString(CuteDSLRT_Error_t) { return "fake failure"; }

}  // namespace

extern "C" CuteDSLRT_Error_t __wrap_CuteDSLRT_Module_Create_From_Bytes(
    CuteDSLRT_Module_t** module, const unsigned char* bytes, size_t size,
    const char** shared_libraries, size_t shared_library_count) {
  return ModuleCreate(module, bytes, size, shared_libraries,
                      shared_library_count);
}

extern "C" CuteDSLRT_Error_t __wrap_CuteDSLRT_Module_Get_Function(
    CuteDSLRT_Function_t** function, CuteDSLRT_Module_t* module,
    const char* prefix) {
  return ModuleGetFunction(function, module, prefix);
}

extern "C" CuteDSLRT_Error_t __wrap_CuteDSLRT_Function_Run(
    void* function, void** arguments, size_t argument_count) {
  return FunctionRun(function, arguments, argument_count);
}

extern "C" CuteDSLRT_Error_t __wrap_CuteDSLRT_Module_Destroy(
    CuteDSLRT_Module_t* module) {
  return ModuleDestroy(module);
}

extern "C" const char* __wrap_CuteDSLRT_GetErrorName(CuteDSLRT_Error_t error) {
  return GetErrorName(error);
}

extern "C" const char* __wrap_CuteDSLRT_GetErrorString(
    CuteDSLRT_Error_t error) {
  return GetErrorString(error);
}

namespace {

class ScopedFakeRuntime {
 public:
  explicit ScopedFakeRuntime(FakeRuntime* runtime) {
    EXPECT_EQ(fake_runtime, nullptr);
    fake_runtime = runtime;
  }

  ~ScopedFakeRuntime() { fake_runtime = nullptr; }
};

std::string Sha256(absl::string_view value) {
  llvm::SHA256 hasher;
  hasher.update(llvm::StringRef(value.data(), value.size()));
  std::array<uint8_t, kModuleDigestSize> digest = hasher.final();
  return std::string(reinterpret_cast<const char*>(digest.data()),
                     digest.size());
}

absl::StatusOr<std::shared_ptr<LoadedModule>> LoadModuleForTest(
    ModuleLoader& loader, absl::string_view bytes) {
  absl::StatusOr<ModuleImage> image = ModuleImage::Create(bytes, Sha256(bytes));
  if (!image.ok()) return image.status();
  return loader.GetOrLoad(*image);
}

TEST(CuteDslModuleTest, ModuleImageOwnsValidatedBytesAndDigest) {
  std::string bytes("module\0image", 12);
  const std::string expected_bytes = bytes;
  std::string sha256 = Sha256(bytes);
  const std::string expected_sha256 = sha256;

  absl::StatusOr<ModuleImage> image = ModuleImage::Create(bytes, sha256);
  ASSERT_TRUE(image.ok()) << image.status();
  bytes[0] = 'x';
  sha256[0] ^= 1;

  EXPECT_EQ(image->bytes(), expected_bytes);
  EXPECT_EQ(image->sha256(), expected_sha256);

  absl::StatusOr<ModuleImage> derived = ModuleImage::Create(expected_bytes);
  ASSERT_TRUE(derived.ok()) << derived.status();
  EXPECT_EQ(derived->bytes(), expected_bytes);
  EXPECT_EQ(derived->sha256(), expected_sha256);
}

TEST(CuteDslModuleTest, RejectsInvalidModuleImages) {
  EXPECT_EQ(ModuleImage::Create("", Sha256("")).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ModuleImage::Create("module", "short").status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(ModuleImage::Create("module", std::string(kModuleDigestSize, 'x'))
                  .status()
                  .message(),
              HasSubstr("does not match"));
}

TEST(CuteDslModuleTest, CachesModulesByDigest) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);

  const std::string bytes = "module cache test";
  ModuleLoader loader;

  absl::StatusOr<std::shared_ptr<LoadedModule>> first =
      LoadModuleForTest(loader, bytes);
  ASSERT_TRUE(first.ok()) << first.status();
  absl::StatusOr<std::shared_ptr<LoadedModule>> same =
      LoadModuleForTest(loader, bytes);
  ASSERT_TRUE(same.ok()) << same.status();
  absl::StatusOr<std::shared_ptr<LoadedModule>> different =
      LoadModuleForTest(loader, "different module cache test");
  ASSERT_TRUE(different.ok()) << different.status();

  EXPECT_EQ(first->get(), same->get());
  EXPECT_NE(first->get(), different->get());
  EXPECT_EQ(runtime.create_count, 2);
}

TEST(CuteDslModuleTest, ModuleLoaderRetainsModules) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);

  absl::StatusOr<ModuleImage> image = ModuleImage::Create(
      "strongly cached module", Sha256("strongly cached module"));
  ASSERT_TRUE(image.ok()) << image.status();
  {
    ModuleLoader loader;
    {
      absl::StatusOr<std::shared_ptr<LoadedModule>> loaded =
          loader.GetOrLoad(*image);
      ASSERT_TRUE(loaded.ok()) << loaded.status();
    }
    EXPECT_EQ(runtime.destroy_count, 0);

    absl::StatusOr<std::shared_ptr<LoadedModule>> loaded =
        loader.GetOrLoad(*image);
    ASSERT_TRUE(loaded.ok()) << loaded.status();
    EXPECT_EQ(runtime.create_count, 1);
  }
  EXPECT_EQ(runtime.destroy_count, 1);
}

TEST(CuteDslModuleTest, ModuleLoadersIsolateFfiState) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);

  absl::StatusOr<ModuleImage> image =
      ModuleImage::Create("state-owned module", Sha256("state-owned module"));
  ASSERT_TRUE(image.ok()) << image.status();
  ModuleLoader first_loader;
  ModuleLoader second_loader;

  absl::StatusOr<std::shared_ptr<LoadedModule>> first =
      first_loader.GetOrLoad(*image);
  ASSERT_TRUE(first.ok()) << first.status();
  absl::StatusOr<std::shared_ptr<LoadedModule>> second =
      second_loader.GetOrLoad(*image);
  ASSERT_TRUE(second.ok()) << second.status();

  EXPECT_NE(first->get(), second->get());
  EXPECT_EQ(runtime.create_count, 2);
}

TEST(CuteDslModuleTest, CachesMultipleFunctionPrefixesPerModule) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);

  const std::string bytes = "module function cache test";
  ModuleLoader loader;
  absl::StatusOr<std::shared_ptr<LoadedModule>> loaded =
      LoadModuleForTest(loader, bytes);
  ASSERT_TRUE(loaded.ok()) << loaded.status();

  absl::StatusOr<LoadedModule::FunctionHandle> first =
      (*loaded)->GetFunction("cutlass_call");
  ASSERT_TRUE(first.ok()) << first.status();
  absl::StatusOr<LoadedModule::FunctionHandle> repeated =
      (*loaded)->GetFunction("cutlass_call");
  ASSERT_TRUE(repeated.ok()) << repeated.status();
  absl::StatusOr<LoadedModule::FunctionHandle> second =
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

  absl::StatusOr<ModuleImage> image = ModuleImage::Create(
      "module with bad key", std::string(kModuleDigestSize, 'x'));
  EXPECT_EQ(image.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(image.status().message(), HasSubstr("does not match"));
  EXPECT_EQ(runtime.create_count, 0);
}

TEST(CuteDslModuleTest, RejectsInvalidFunctionPrefixesWithoutRuntimeLookup) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);

  const std::string bytes = "module function prefix validation test";
  ModuleLoader loader;
  absl::StatusOr<std::shared_ptr<LoadedModule>> loaded =
      LoadModuleForTest(loader, bytes);
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

  const std::string bytes = "module null function test";
  ModuleLoader loader;
  absl::StatusOr<std::shared_ptr<LoadedModule>> loaded =
      LoadModuleForTest(loader, bytes);
  ASSERT_TRUE(loaded.ok()) << loaded.status();

  absl::StatusOr<LoadedModule::FunctionHandle> function =
      (*loaded)->GetFunction("cutlass_call_2");
  EXPECT_EQ(function.status().code(), absl::StatusCode::kInternal);
  EXPECT_THAT(function.status().message(), HasSubstr("null cutlass_call_2"));
  EXPECT_EQ(runtime.destroy_count, 0);
}

}  // namespace
}  // namespace xla::gpu::cutedsl
