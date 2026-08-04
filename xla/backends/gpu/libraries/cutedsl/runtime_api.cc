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

#include <optional>
#include <type_traits>

#include "xla/backends/gpu/libraries/cutedsl/runtime.h"

#if !defined(XLA_CUTEDSL_RUNTIME_STATIC) && \
    (defined(__linux__) || defined(__APPLE__))
#include <dlfcn.h>
#endif

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"

namespace xla::gpu::cutedsl {
namespace {

#if !defined(XLA_CUTEDSL_RUNTIME_STATIC)
absl::Status ValidateRuntimeApi(const RuntimeApi& api) {
  if (api.module_create_from_bytes == nullptr ||
      api.module_get_function == nullptr || api.function_run == nullptr ||
      api.module_destroy == nullptr || api.get_error_name == nullptr ||
      api.get_error_string == nullptr) {
    return absl::FailedPreconditionError(
        "CuTeDSL runtime is missing a required function");
  }
  return absl::OkStatus();
}

constexpr char kRuntimeLibrary[] = "libcute_dsl_runtime.so";

struct RuntimeState {
  absl::Mutex mutex;
  std::optional<absl::StatusOr<RuntimeApi>> result ABSL_GUARDED_BY(mutex);
  void* library_handle ABSL_GUARDED_BY(mutex) = nullptr;
};

RuntimeState& State() {
  static absl::NoDestructor<RuntimeState> state;
  return *state;
}

#if defined(__linux__) || defined(__APPLE__)
template <typename Function>
absl::Status ResolveSymbol(void* handle, const char* name, Function* function) {
  static_assert(std::is_pointer_v<Function>);
  dlerror();
  void* symbol = dlsym(handle, name);
  const char* error = dlerror();
  if (error != nullptr || symbol == nullptr) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "CuTeDSL runtime %s does not export %s: %s", kRuntimeLibrary, name,
        error == nullptr ? "symbol not found" : error));
  }
  *function = reinterpret_cast<Function>(symbol);
  return absl::OkStatus();
}

absl::StatusOr<RuntimeApi> LoadRuntimeApi(void** retained_handle) {
  dlerror();
  void* handle = dlopen(kRuntimeLibrary, RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char* error = dlerror();
    return absl::FailedPreconditionError(absl::StrFormat(
        "Failed to load CuTeDSL runtime %s. Ensure it is installed and "
        "discoverable through LD_LIBRARY_PATH or the platform dynamic-library "
        "search path: %s",
        kRuntimeLibrary, error == nullptr ? "unknown error" : error));
  }

  RuntimeApi api = {};
  absl::Status status =
      ResolveSymbol(handle, "CuteDSLRT_Module_Create_From_Bytes",
                    &api.module_create_from_bytes);
  if (status.ok()) {
    status = ResolveSymbol(handle, "CuteDSLRT_Module_Get_Function",
                           &api.module_get_function);
  }
  if (status.ok()) {
    status = ResolveSymbol(handle, "CuteDSLRT_Function_Run", &api.function_run);
  }
  if (status.ok()) {
    status =
        ResolveSymbol(handle, "CuteDSLRT_Module_Destroy", &api.module_destroy);
  }
  if (status.ok()) {
    status =
        ResolveSymbol(handle, "CuteDSLRT_GetErrorName", &api.get_error_name);
  }
  if (status.ok()) {
    status = ResolveSymbol(handle, "CuteDSLRT_GetErrorString",
                           &api.get_error_string);
  }
  if (!status.ok()) {
    dlclose(handle);
    return status;
  }

  *retained_handle = handle;
  return api;
}
#else
absl::StatusOr<RuntimeApi> LoadRuntimeApi(void**) {
  return absl::FailedPreconditionError(
      "Dynamic CuTeDSL runtime loading is unsupported on this platform");
}
#endif

bool SameRuntimeApi(const RuntimeApi& lhs, const RuntimeApi& rhs) {
  return lhs.module_create_from_bytes == rhs.module_create_from_bytes &&
         lhs.module_get_function == rhs.module_get_function &&
         lhs.function_run == rhs.function_run &&
         lhs.module_destroy == rhs.module_destroy &&
         lhs.get_error_name == rhs.get_error_name &&
         lhs.get_error_string == rhs.get_error_string;
}
#endif

}  // namespace

namespace internal {

absl::StatusOr<const RuntimeApi*> GetRuntimeApi() {
#if defined(XLA_CUTEDSL_RUNTIME_STATIC)
  static constexpr RuntimeApi api = {
      &CuteDSLRT_Module_Create_From_Bytes,
      &CuteDSLRT_Module_Get_Function,
      &CuteDSLRT_Function_Run,
      &CuteDSLRT_Module_Destroy,
      &CuteDSLRT_GetErrorName,
      &CuteDSLRT_GetErrorString,
  };
  return &api;
#else
  RuntimeState& state = State();
  absl::MutexLock lock(&state.mutex);
  if (!state.result.has_value()) {
    state.result = LoadRuntimeApi(&state.library_handle);
  }
  if (!state.result->ok()) return state.result->status();
  return &state.result->value();
#endif
}

absl::Status RegisterRuntimeApiForTest(const RuntimeApi* api) {
#if defined(XLA_CUTEDSL_RUNTIME_STATIC)
  return absl::UnimplementedError(
      "A statically linked CuTeDSL runtime cannot be replaced");
#else
  if (api == nullptr) {
    return absl::InvalidArgumentError("CuTeDSL runtime API must not be null");
  }
  absl::Status status = ValidateRuntimeApi(*api);
  if (!status.ok()) return status;

  RuntimeState& state = State();
  absl::MutexLock lock(&state.mutex);
  if (!state.result.has_value()) {
    state.result = *api;
    return absl::OkStatus();
  }
  if (state.result->ok() && SameRuntimeApi(state.result->value(), *api)) {
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError(
      "A different CuTeDSL runtime is already registered");
#endif
}

}  // namespace internal
}  // namespace xla::gpu::cutedsl
