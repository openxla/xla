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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"
#include "xla/backends/gpu/libraries/cutedsl/runtime.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/ffi/invoke.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/types.h"

namespace xla::gpu::cutedsl {

namespace ffi = ::xla::ffi;
using ::xla::PrimitiveType;

namespace {

using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;

constexpr char kTarget[] = "__xla_gpu_cutedsl_call_v3";
constexpr char kNoCudaGraphTarget[] = "__xla_gpu_cutedsl_call_no_cuda_graph_v3";

struct TestBufferDescriptor {
  void* buffer;
  const int64_t* shape;
};

struct FakeRuntime {
  std::string module_bytes;
  std::string function_prefix;
  std::vector<std::string> shared_libraries;
  void* stream = nullptr;
  std::vector<TestBufferDescriptor> buffers;
  int create_count = 0;
  int get_function_count = 0;
  int run_count = 0;
  int destroy_count = 0;
  CuteDSLRT_Error_t create_error = CuteDSLRT_Error_Success;
  CuteDSLRT_Error_t get_function_error = CuteDSLRT_Error_Success;
  CuteDSLRT_Error_t run_error = CuteDSLRT_Error_Success;
  int32_t cuda_error = 0;
  bool create_null_module = false;
  bool create_module_on_error = false;
  bool get_null_function = false;
  bool shared_libraries_was_null = false;
};

FakeRuntime* fake_runtime = nullptr;

CuteDSLRT_Error_t ModuleCreate(CuteDSLRT_Module_t** module,
                               const unsigned char* bytes, size_t size,
                               const char** shared_libs,
                               size_t shared_libs_size) {
  EXPECT_NE(fake_runtime, nullptr);
  ++fake_runtime->create_count;
  fake_runtime->module_bytes.assign(reinterpret_cast<const char*>(bytes), size);
  fake_runtime->shared_libraries.clear();
  fake_runtime->shared_libraries_was_null = shared_libs == nullptr;
  if (shared_libs == nullptr) {
    EXPECT_EQ(shared_libs_size, 0);
  } else {
    for (size_t i = 0; i < shared_libs_size; ++i) {
      EXPECT_NE(shared_libs[i], nullptr);
      if (shared_libs[i] != nullptr) {
        fake_runtime->shared_libraries.emplace_back(shared_libs[i]);
      }
    }
  }
  if (fake_runtime->create_error != CuteDSLRT_Error_Success) {
    if (fake_runtime->create_module_on_error) {
      *module = reinterpret_cast<CuteDSLRT_Module_t*>(fake_runtime);
    }
    return fake_runtime->create_error;
  }
  *module = fake_runtime->create_null_module
                ? nullptr
                : reinterpret_cast<CuteDSLRT_Module_t*>(fake_runtime);
  return fake_runtime->create_error;
}

CuteDSLRT_Error_t ModuleGetFunction(CuteDSLRT_Function_t** function,
                                    CuteDSLRT_Module_t* module,
                                    const char* prefix) {
  EXPECT_EQ(module, reinterpret_cast<CuteDSLRT_Module_t*>(fake_runtime));
  ++fake_runtime->get_function_count;
  fake_runtime->function_prefix = prefix;
  if (fake_runtime->get_function_error != CuteDSLRT_Error_Success) {
    return fake_runtime->get_function_error;
  }
  *function = fake_runtime->get_null_function
                  ? nullptr
                  : reinterpret_cast<CuteDSLRT_Function_t*>(fake_runtime);
  return fake_runtime->get_function_error;
}

CuteDSLRT_Error_t FunctionRun(void* function, void** arguments,
                              size_t num_arguments) {
  EXPECT_EQ(function, fake_runtime);
  EXPECT_EQ(num_arguments, 4);
  ++fake_runtime->run_count;

  fake_runtime->stream = *reinterpret_cast<void**>(arguments[0]);
  fake_runtime->buffers.clear();
  for (size_t i = 1; i < num_arguments - 1; ++i) {
    auto* descriptor = *reinterpret_cast<TestBufferDescriptor**>(arguments[i]);
    fake_runtime->buffers.push_back(*descriptor);
  }
  *reinterpret_cast<int32_t*>(arguments[num_arguments - 1]) =
      fake_runtime->cuda_error;
  return fake_runtime->run_error;
}

CuteDSLRT_Error_t ModuleDestroy(CuteDSLRT_Module_t* module) {
  EXPECT_EQ(module, reinterpret_cast<CuteDSLRT_Module_t*>(fake_runtime));
  ++fake_runtime->destroy_count;
  return CuteDSLRT_Error_Success;
}

const char* GetErrorName(CuteDSLRT_Error_t) { return "FakeRuntimeError"; }
const char* GetErrorString(CuteDSLRT_Error_t) { return "fake runtime failure"; }

const RuntimeApi kRuntimeApi = {
    ModuleCreate,  ModuleGetFunction, FunctionRun,
    ModuleDestroy, GetErrorName,      GetErrorString,
};

}  // namespace

#if defined(PLATFORM_GOOGLE)
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
#endif

namespace {

class ScopedFakeRuntime {
 public:
  explicit ScopedFakeRuntime(FakeRuntime* runtime) {
#if !defined(PLATFORM_GOOGLE)
    EXPECT_TRUE(internal::RegisterRuntimeApiForTest(&kRuntimeApi).ok());
#endif
    EXPECT_EQ(fake_runtime, nullptr);
    fake_runtime = runtime;
  }

  ~ScopedFakeRuntime() { fake_runtime = nullptr; }
};

std::string Sha256(absl::string_view value) {
  llvm::SHA256 hasher;
  hasher.update(llvm::StringRef(value.data(), value.size()));
  std::array<uint8_t, 32> digest = hasher.final();
  return std::string(reinterpret_cast<const char*>(digest.data()),
                     digest.size());
}

ffi::AttributesMap Attributes(absl::string_view module, absl::string_view key) {
  ffi::CallFrameBuilder::AttributesBuilder attributes;
  attributes.Insert("key", std::string(key));
  attributes.Insert("module", std::string(module));
  return attributes.Build();
}

class FfiTestInvocation {
 public:
  explicit FfiTestInvocation(std::string module)
      : module_(std::move(module)),
        key_(Sha256(module_)),
        registration_(ffi::FindHandler(kTarget, "CUDA")) {
    context_.state_context.instantiate = &state_;
    ON_CALL(stream_, platform_specific_handle())
        .WillByDefault(Return(stream_executor::Stream::PlatformSpecificHandle{
            &fake_stream_handle_}));
    ffi::InvokeContext::GpuContext gpu_context;
    gpu_context.stream = &stream_;
    context_.backend_context = gpu_context;
  }

  absl::Status Instantiate() {
    if (!registration_.ok()) return registration_.status();
    ffi::CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
    builder.AddAttributes(Attributes(module_, key_));
    ffi::CallFrame frame = builder.Build();
    return ffi::Invoke(ffi::GetXlaFfiApi(), registration_->bundle.instantiate,
                       frame, context_, XLA_FFI_ExecutionStage_INSTANTIATE);
  }

  absl::Status Prepare() {
    if (!registration_.ok()) return registration_.status();
    ffi::CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
    builder.AddAttributes(Attributes(module_, key_));
    ffi::CallFrame frame = builder.Build();
    return ffi::Invoke(ffi::GetXlaFfiApi(), registration_->bundle.prepare,
                       frame, context_, XLA_FFI_ExecutionStage_PREPARE);
  }

  absl::Status Initialize() {
    if (!registration_.ok()) return registration_.status();
    ffi::CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
    builder.AddAttributes(Attributes(module_, key_));
    ffi::CallFrame frame = builder.Build();
    return ffi::Invoke(ffi::GetXlaFfiApi(), registration_->bundle.initialize,
                       frame, context_, XLA_FFI_ExecutionStage_INITIALIZE);
  }

  absl::Status Execute() {
    if (!registration_.ok()) return registration_.status();
    std::array<float, 1> input = {1.0f};
    std::array<float, 1> output = {};
    std::array<int64_t, 1> dimensions = {1};
    ffi::CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/1);
    builder.AddBufferArg(
        stream_executor::DeviceAddressBase(input.data(), sizeof(input)),
        PrimitiveType::F32, dimensions);
    builder.AddBufferRet(
        stream_executor::DeviceAddressBase(output.data(), sizeof(output)),
        PrimitiveType::F32, dimensions);
    builder.AddAttributes(Attributes(module_, key_));
    ffi::CallFrame frame = builder.Build();
    return ffi::Invoke(ffi::GetXlaFfiApi(), registration_->bundle.execute,
                       frame, context_, XLA_FFI_ExecutionStage_EXECUTE);
  }

 private:
  std::string module_;
  std::string key_;
  absl::StatusOr<ffi::HandlerRegistration> registration_;
  ffi::ExecutionState state_;
  ffi::InvokeContext context_;
  int fake_stream_handle_ = 0;
  NiceMock<stream_executor::MockStream> stream_;
};

TEST(CuteDslFfiTest, RegistersBothTargetsWithExpectedTraits) {
  absl::StatusOr<ffi::HandlerRegistration> normal =
      ffi::FindHandler(kTarget, "CUDA");
  ASSERT_TRUE(normal.ok()) << normal.status();
  EXPECT_EQ(normal->metadata.traits,
            XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE);
  EXPECT_NE(normal->bundle.instantiate, nullptr);
  EXPECT_NE(normal->bundle.prepare, nullptr);
  EXPECT_NE(normal->bundle.initialize, nullptr);
  EXPECT_NE(normal->bundle.execute, nullptr);

  absl::StatusOr<ffi::HandlerRegistration> no_cuda_graph =
      ffi::FindHandler(kNoCudaGraphTarget, "CUDA");
  ASSERT_TRUE(no_cuda_graph.ok()) << no_cuda_graph.status();
  EXPECT_EQ(no_cuda_graph->metadata.traits, 0);
  EXPECT_EQ(no_cuda_graph->bundle.instantiate, normal->bundle.instantiate);
  EXPECT_EQ(no_cuda_graph->bundle.prepare, normal->bundle.prepare);
  EXPECT_EQ(no_cuda_graph->bundle.initialize, normal->bundle.initialize);
  EXPECT_NE(no_cuda_graph->bundle.execute, normal->bundle.execute);
}

TEST(CuteDslFfiTest, DoesNotRegisterEarlierTargetVersions) {
  constexpr std::array<absl::string_view, 4> kEarlierTargets = {
      "__xla_gpu_cutedsl_call_v1",
      "__xla_gpu_cutedsl_call_no_cuda_graph_v1",
      "__xla_gpu_cutedsl_call_v2",
      "__xla_gpu_cutedsl_call_no_cuda_graph_v2",
  };

  for (absl::string_view target : kEarlierTargets) {
    absl::StatusOr<ffi::HandlerRegistration> registration =
        ffi::FindHandler(target, "CUDA");
    EXPECT_EQ(registration.status().code(), absl::StatusCode::kNotFound)
        << target;
  }
}

TEST(CuteDslFfiTest, PassesNoSharedLibrariesToModuleCreation) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);

  {
    FfiTestInvocation invocation("module with a Bazel-linked runtime");
    ASSERT_TRUE(invocation.Instantiate().ok());
    ASSERT_TRUE(invocation.Prepare().ok());

    EXPECT_TRUE(runtime.shared_libraries_was_null);
    EXPECT_TRUE(runtime.shared_libraries.empty());
  }
  EXPECT_EQ(runtime.destroy_count, 1);
}

TEST(CuteDslFfiTest, RunsCompleteLifecycleThroughRuntime) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);

  const std::string module = "fake CuTeDSL object for lifecycle test";
  const std::string key = Sha256(module);
  absl::StatusOr<ffi::HandlerRegistration> registration =
      ffi::FindHandler(kTarget, "CUDA");
  ASSERT_TRUE(registration.ok()) << registration.status();

  std::array<float, 6> input = {0, 1, 2, 3, 4, 5};
  std::array<float, 6> output = {};
  std::array<int64_t, 2> dimensions = {2, 3};
  int fake_stream_handle = 0;
  NiceMock<stream_executor::MockStream> stream;
  ON_CALL(stream, platform_specific_handle())
      .WillByDefault(Return(stream_executor::Stream::PlatformSpecificHandle{
          &fake_stream_handle}));

  {
    ffi::ExecutionState state;
    ffi::InvokeContext context;
    context.state_context.instantiate = &state;

    ffi::CallFrameBuilder instantiate_builder(/*num_args=*/0,
                                              /*num_rets=*/0);
    instantiate_builder.AddAttributes(Attributes(module, key));
    ffi::CallFrame instantiate_frame = instantiate_builder.Build();
    EXPECT_TRUE(ffi::Invoke(ffi::GetXlaFfiApi(),
                            registration->bundle.instantiate, instantiate_frame,
                            context, XLA_FFI_ExecutionStage_INSTANTIATE)
                    .ok());

    ffi::CallFrameBuilder call_builder(/*num_args=*/1, /*num_rets=*/1);
    call_builder.AddBufferArg(
        stream_executor::DeviceAddressBase(input.data(), sizeof(input)),
        PrimitiveType::F32, dimensions);
    call_builder.AddBufferRet(
        stream_executor::DeviceAddressBase(output.data(), sizeof(output)),
        PrimitiveType::F32, dimensions);
    call_builder.AddAttributes(Attributes(module, key));
    ffi::CallFrame call_frame = call_builder.Build();

    EXPECT_TRUE(ffi::Invoke(ffi::GetXlaFfiApi(), registration->bundle.prepare,
                            call_frame, context, XLA_FFI_ExecutionStage_PREPARE)
                    .ok());
    EXPECT_EQ(runtime.create_count, 1);
    EXPECT_EQ(runtime.get_function_count, 1);
    EXPECT_EQ(runtime.module_bytes, module);
    EXPECT_EQ(runtime.function_prefix, "cutlass_call");

    EXPECT_TRUE(ffi::Invoke(ffi::GetXlaFfiApi(),
                            registration->bundle.initialize, call_frame,
                            context, XLA_FFI_ExecutionStage_INITIALIZE)
                    .ok());
    EXPECT_EQ(runtime.create_count, 1);
    EXPECT_EQ(runtime.get_function_count, 1);
    EXPECT_EQ(runtime.run_count, 0);

    ffi::InvokeContext::GpuContext gpu_context;
    gpu_context.stream = &stream;
    context.backend_context = gpu_context;
    EXPECT_TRUE(ffi::Invoke(ffi::GetXlaFfiApi(), registration->bundle.execute,
                            call_frame, context, XLA_FFI_ExecutionStage_EXECUTE)
                    .ok());

    EXPECT_EQ(runtime.run_count, 1);
    EXPECT_EQ(runtime.stream, &fake_stream_handle);
    ASSERT_EQ(runtime.buffers.size(), 2);
    EXPECT_EQ(runtime.buffers[0].buffer, input.data());
    EXPECT_EQ(runtime.buffers[1].buffer, output.data());
    ASSERT_NE(runtime.buffers[0].shape, nullptr);
    ASSERT_NE(runtime.buffers[1].shape, nullptr);
    EXPECT_EQ(runtime.buffers[0].shape[0], 2);
    EXPECT_EQ(runtime.buffers[0].shape[1], 3);
    EXPECT_EQ(runtime.buffers[1].shape[0], 2);
    EXPECT_EQ(runtime.buffers[1].shape[1], 3);
  }

  EXPECT_EQ(runtime.destroy_count, 1);
}

TEST(CuteDslFfiTest, RejectsKeyThatDoesNotMatchModule) {
  FakeRuntime runtime;
  ScopedFakeRuntime scoped_runtime(&runtime);

  const std::string module = "fake CuTeDSL object with invalid key";
  const std::string key(32, 'x');
  absl::StatusOr<ffi::HandlerRegistration> registration =
      ffi::FindHandler(kTarget, "CUDA");
  ASSERT_TRUE(registration.ok()) << registration.status();

  {
    ffi::ExecutionState state;
    ffi::InvokeContext context;
    context.state_context.instantiate = &state;

    ffi::CallFrameBuilder instantiate_builder(/*num_args=*/0,
                                              /*num_rets=*/0);
    instantiate_builder.AddAttributes(Attributes(module, key));
    ffi::CallFrame instantiate_frame = instantiate_builder.Build();
    absl::Status status = ffi::Invoke(
        ffi::GetXlaFfiApi(), registration->bundle.instantiate,
        instantiate_frame, context, XLA_FFI_ExecutionStage_INSTANTIATE);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(status.message(), HasSubstr("does not match"));
    EXPECT_EQ(runtime.create_count, 0);
  }
}

TEST(CuteDslFfiTest, ReportsModuleCreateFailure) {
  FakeRuntime runtime;
  runtime.create_error = CuteDSLRT_Error_InvalidArguments;
  ScopedFakeRuntime scoped_runtime(&runtime);

  FfiTestInvocation invocation("module whose creation fails");
  ASSERT_TRUE(invocation.Instantiate().ok());
  absl::Status status = invocation.Prepare();

  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), HasSubstr("Failed to create"));
  EXPECT_THAT(status.message(), HasSubstr("FakeRuntimeError (error 7)"));
  EXPECT_EQ(runtime.create_count, 1);
  EXPECT_EQ(runtime.get_function_count, 0);
  EXPECT_EQ(runtime.destroy_count, 0);
}

TEST(CuteDslFfiTest, DestroysModuleReturnedWithCreateFailure) {
  FakeRuntime runtime;
  runtime.create_error = CuteDSLRT_Error_InvalidArguments;
  runtime.create_module_on_error = true;
  ScopedFakeRuntime scoped_runtime(&runtime);

  FfiTestInvocation invocation("failed creation that returns a module");
  ASSERT_TRUE(invocation.Instantiate().ok());
  absl::Status status = invocation.Prepare();

  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), HasSubstr("Failed to create"));
  EXPECT_EQ(runtime.create_count, 1);
  EXPECT_EQ(runtime.get_function_count, 0);
  EXPECT_EQ(runtime.destroy_count, 1);
}

TEST(CuteDslFfiTest, RejectsNullModuleFromRuntime) {
  FakeRuntime runtime;
  runtime.create_null_module = true;
  ScopedFakeRuntime scoped_runtime(&runtime);

  FfiTestInvocation invocation("module whose creation returns null");
  ASSERT_TRUE(invocation.Instantiate().ok());
  absl::Status status = invocation.Prepare();

  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), HasSubstr("created a null module"));
  EXPECT_EQ(runtime.create_count, 1);
  EXPECT_EQ(runtime.get_function_count, 0);
  EXPECT_EQ(runtime.destroy_count, 0);
}

TEST(CuteDslFfiTest, RetainsModuleAfterFunctionLookupFailure) {
  FakeRuntime runtime;
  runtime.get_function_error = CuteDSLRT_Error_KernelNotFound;
  ScopedFakeRuntime scoped_runtime(&runtime);

  {
    FfiTestInvocation invocation("module whose function lookup fails");
    ASSERT_TRUE(invocation.Instantiate().ok());
    absl::Status status = invocation.Prepare();

    EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
    EXPECT_THAT(status.message(), HasSubstr("Failed to load CuTeDSL"));
    EXPECT_THAT(status.message(), HasSubstr("FakeRuntimeError (error 6)"));
    EXPECT_EQ(runtime.create_count, 1);
    EXPECT_EQ(runtime.get_function_count, 1);
    EXPECT_EQ(runtime.destroy_count, 0);
  }
  EXPECT_EQ(runtime.destroy_count, 1);
}

TEST(CuteDslFfiTest, RetainsModuleAfterNullFunctionLookup) {
  FakeRuntime runtime;
  runtime.get_null_function = true;
  ScopedFakeRuntime scoped_runtime(&runtime);

  {
    FfiTestInvocation invocation("module whose function lookup returns null");
    ASSERT_TRUE(invocation.Instantiate().ok());
    absl::Status status = invocation.Prepare();

    EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
    EXPECT_THAT(status.message(), HasSubstr("returned a null cutlass_call"));
    EXPECT_EQ(runtime.create_count, 1);
    EXPECT_EQ(runtime.get_function_count, 1);
    EXPECT_EQ(runtime.destroy_count, 0);
  }
  EXPECT_EQ(runtime.destroy_count, 1);
}

TEST(CuteDslFfiTest, ReportsRuntimeFunctionRunFailure) {
  FakeRuntime runtime;
  runtime.run_error = CuteDSLRT_Error_CudaError;
  ScopedFakeRuntime scoped_runtime(&runtime);

  {
    FfiTestInvocation invocation("module whose function run fails");
    ASSERT_TRUE(invocation.Instantiate().ok());
    ASSERT_TRUE(invocation.Prepare().ok());
    ASSERT_TRUE(invocation.Initialize().ok());
    absl::Status status = invocation.Execute();

    EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
    EXPECT_THAT(status.message(), HasSubstr("Failed to execute CuTeDSL"));
    EXPECT_THAT(status.message(), HasSubstr("FakeRuntimeError (error 1)"));
    EXPECT_EQ(runtime.run_count, 1);
  }
  EXPECT_EQ(runtime.destroy_count, 1);
}

TEST(CuteDslFfiTest, ReportsPackedCudaStatus) {
  FakeRuntime runtime;
  runtime.cuda_error = 700;
  ScopedFakeRuntime scoped_runtime(&runtime);

  {
    FfiTestInvocation invocation("module whose packed CUDA status fails");
    ASSERT_TRUE(invocation.Instantiate().ok());
    ASSERT_TRUE(invocation.Prepare().ok());
    ASSERT_TRUE(invocation.Initialize().ok());
    absl::Status status = invocation.Execute();

    EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
    EXPECT_THAT(status.message(), HasSubstr("CUDA error 700"));
    EXPECT_EQ(runtime.run_count, 1);
  }
  EXPECT_EQ(runtime.destroy_count, 1);
}

}  // namespace
}  // namespace xla::gpu::cutedsl
