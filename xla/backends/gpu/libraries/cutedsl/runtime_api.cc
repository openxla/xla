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

#include <optional>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace xla::gpu::cutedsl {
namespace {

absl::Status MissingFunction(absl::string_view symbol_name) {
  return absl::FailedPreconditionError(absl::StrCat(
      "CuTeDSL runtime is missing required function ", symbol_name));
}

struct RuntimeState {
  absl::Mutex mutex;
  std::optional<RuntimeFunctions> functions_for_testing ABSL_GUARDED_BY(mutex);
};

RuntimeState& GlobalRuntimeState() {
  static auto* state = new RuntimeState;
  return *state;
}

}  // namespace

absl::Status ValidateRuntimeFunctions(const RuntimeFunctions* functions) {
  if (functions == nullptr) {
    return absl::FailedPreconditionError(
        "CuTeDSL runtime functions must not be null");
  }

  if (functions->module_create_from_bytes == nullptr) {
    return MissingFunction(kModuleCreateFromBytesSymbol);
  }
  if (functions->module_get_function == nullptr) {
    return MissingFunction(kModuleGetFunctionSymbol);
  }
  if (functions->function_run == nullptr) {
    return MissingFunction(kFunctionRunSymbol);
  }
  if (functions->module_destroy == nullptr) {
    return MissingFunction(kModuleDestroySymbol);
  }
  if (functions->get_error_name == nullptr) {
    return MissingFunction(kGetErrorNameSymbol);
  }
  if (functions->get_error_string == nullptr) {
    return MissingFunction(kGetErrorStringSymbol);
  }

  return absl::OkStatus();
}

absl::StatusOr<RuntimeFunctions> GetRuntimeFunctions() {
  RuntimeState& state = GlobalRuntimeState();
  absl::MutexLock lock(&state.mutex);
  if (state.functions_for_testing.has_value()) {
    return *state.functions_for_testing;
  }

#ifdef XLA_CUTEDSL_RUNTIME_UNAVAILABLE
  return absl::FailedPreconditionError(
      "CuTeDSL FFI was built without a runtime; set "
      "--//xla/backends/gpu/libraries/cutedsl:cutedsl_runtime to a Bazel "
      "C++ target providing libcute_dsl_runtime.so or the standalone combined "
      "libcute_dsl_runtime.a archive");
#else
  return RuntimeFunctions{
      /*module_create_from_bytes=*/CuteDSLRT_Module_Create_From_Bytes,
      /*module_get_function=*/CuteDSLRT_Module_Get_Function,
      /*function_run=*/CuteDSLRT_Function_Run,
      /*module_destroy=*/CuteDSLRT_Module_Destroy,
      /*get_error_name=*/CuteDSLRT_GetErrorName,
      /*get_error_string=*/CuteDSLRT_GetErrorString,
  };
#endif
}

absl::Status SetRuntimeFunctionsForTesting(const RuntimeFunctions* functions) {
  absl::Status status = ValidateRuntimeFunctions(functions);
  if (!status.ok()) return status;

  RuntimeState& state = GlobalRuntimeState();
  absl::MutexLock lock(&state.mutex);
  state.functions_for_testing = *functions;
  return absl::OkStatus();
}

void ResetRuntimeFunctionsForTesting() {
  RuntimeState& state = GlobalRuntimeState();
  absl::MutexLock lock(&state.mutex);
  state.functions_for_testing.reset();
}

}  // namespace xla::gpu::cutedsl
