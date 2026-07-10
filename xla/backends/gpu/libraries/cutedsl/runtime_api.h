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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_RUNTIME_API_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_RUNTIME_API_H_

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xla::gpu::cutedsl {

// This is a private mirror of the C ABI exported by the CuTeDSL runtime.
// OpenXLA intentionally does not depend on the CuTeDSL headers.
struct CuteDSLRT_Module;
struct CuteDSLRT_Function;

using CuteDSLRT_Module_t = CuteDSLRT_Module;
using CuteDSLRT_Function_t = CuteDSLRT_Function;
using CuteDSLRT_Error_t = std::int32_t;

inline constexpr CuteDSLRT_Error_t kCuteDslRtSuccess = 0;

// Functions exported by both the shared CuTeDSL runtime and its standalone
// combined static archive. Bazel selects exactly one implementation.
extern "C" {
CuteDSLRT_Error_t CuteDSLRT_Module_Create_From_Bytes(
    CuteDSLRT_Module_t** module_obj, const unsigned char* binary_bytes,
    std::size_t binary_bytes_size, const char** shared_libs,
    std::size_t shared_libs_size);
CuteDSLRT_Error_t CuteDSLRT_Module_Get_Function(CuteDSLRT_Function_t** function,
                                                CuteDSLRT_Module_t* module_obj,
                                                const char* function_prefix);
CuteDSLRT_Error_t CuteDSLRT_Function_Run(void* function, void** args,
                                         std::size_t num_args);
CuteDSLRT_Error_t CuteDSLRT_Module_Destroy(CuteDSLRT_Module_t* module_obj);
const char* CuteDSLRT_GetErrorName(CuteDSLRT_Error_t error);
const char* CuteDSLRT_GetErrorString(CuteDSLRT_Error_t error);
}

using CuteDSLRT_Module_Create_From_BytesFn = CuteDSLRT_Error_t (*)(
    CuteDSLRT_Module_t** module_obj, const unsigned char* binary_bytes,
    std::size_t binary_bytes_size, const char** shared_libs,
    std::size_t shared_libs_size);
using CuteDSLRT_Module_Get_FunctionFn = CuteDSLRT_Error_t (*)(
    CuteDSLRT_Function_t** function, CuteDSLRT_Module_t* module_obj,
    const char* function_prefix);
using CuteDSLRT_Function_RunFn = CuteDSLRT_Error_t (*)(void* function,
                                                       void** args,
                                                       std::size_t num_args);
using CuteDSLRT_Module_DestroyFn =
    CuteDSLRT_Error_t (*)(CuteDSLRT_Module_t* module_obj);
using CuteDSLRT_GetErrorNameFn = const char* (*)(CuteDSLRT_Error_t error);
using CuteDSLRT_GetErrorStringFn = const char* (*)(CuteDSLRT_Error_t error);

// OpenXLA-owned bundle of the Bazel-linked CuTeDSL runtime functions. This is
// not part of the CuTeDSL ABI.
struct RuntimeFunctions {
  CuteDSLRT_Module_Create_From_BytesFn module_create_from_bytes = nullptr;
  CuteDSLRT_Module_Get_FunctionFn module_get_function = nullptr;
  CuteDSLRT_Function_RunFn function_run = nullptr;
  CuteDSLRT_Module_DestroyFn module_destroy = nullptr;
  CuteDSLRT_GetErrorNameFn get_error_name = nullptr;
  CuteDSLRT_GetErrorStringFn get_error_string = nullptr;
};

inline constexpr char kModuleCreateFromBytesSymbol[] =
    "CuteDSLRT_Module_Create_From_Bytes";
inline constexpr char kModuleGetFunctionSymbol[] =
    "CuteDSLRT_Module_Get_Function";
inline constexpr char kFunctionRunSymbol[] = "CuteDSLRT_Function_Run";
inline constexpr char kModuleDestroySymbol[] = "CuteDSLRT_Module_Destroy";
inline constexpr char kGetErrorNameSymbol[] = "CuteDSLRT_GetErrorName";
inline constexpr char kGetErrorStringSymbol[] = "CuteDSLRT_GetErrorString";

// Validates every function required by OpenXLA.
absl::Status ValidateRuntimeFunctions(const RuntimeFunctions* functions);

// Returns the process-wide CuTeDSL runtime functions selected by Bazel. An OSS
// build using the default unavailable provider returns FailedPrecondition.
absl::StatusOr<RuntimeFunctions> GetRuntimeFunctions();

// Process-wide test hooks used by FFI handler tests.
absl::Status SetRuntimeFunctionsForTesting(const RuntimeFunctions* functions);
void ResetRuntimeFunctionsForTesting();

}  // namespace xla::gpu::cutedsl

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_RUNTIME_API_H_
