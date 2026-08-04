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

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define XLA_CUTEDSL_RUNTIME_EXPORT __attribute__((visibility("default")))
#else
#define XLA_CUTEDSL_RUNTIME_EXPORT
#endif

typedef struct BinaryModule CuteDSLRT_Module_t;
typedef struct BinaryFunction CuteDSLRT_Function_t;

typedef enum {
  CuteDSLRT_Success = 0,
  CuteDSLRT_Error_Success = 0,
  CuteDSLRT_Error_CudaError = 1,
  CuteDSLRT_Error_InvalidBinary = 2,
  CuteDSLRT_Error_InvalidMetadata = 3,
  CuteDSLRT_Error_InvalidVersion = 4,
  CuteDSLRT_Error_LibraryAlreadyLoaded = 5,
  CuteDSLRT_Error_KernelNotFound = 6,
  CuteDSLRT_Error_InvalidArguments = 7,
  CuteDSLRT_Error_NoBinaryLoaded = 8,
  CuteDSLRT_Error_BinaryAlreadyLoaded = 9,
} CuteDSLRT_Error_t;

XLA_CUTEDSL_RUNTIME_EXPORT CuteDSLRT_Error_t CuteDSLRT_Module_Create_From_Bytes(
    CuteDSLRT_Module_t** module_obj, const unsigned char* binary_bytes,
    size_t binary_bytes_size, const char** shared_libs,
    size_t shared_libs_size);
XLA_CUTEDSL_RUNTIME_EXPORT CuteDSLRT_Error_t CuteDSLRT_Module_Get_Function(
    CuteDSLRT_Function_t** func, CuteDSLRT_Module_t* module_obj,
    const char* function_prefix);
XLA_CUTEDSL_RUNTIME_EXPORT CuteDSLRT_Error_t
CuteDSLRT_Function_Run(void* func, void** args, size_t num_args);
XLA_CUTEDSL_RUNTIME_EXPORT CuteDSLRT_Error_t
CuteDSLRT_Module_Destroy(CuteDSLRT_Module_t* module_obj);
XLA_CUTEDSL_RUNTIME_EXPORT const char* CuteDSLRT_GetErrorName(
    CuteDSLRT_Error_t error);
XLA_CUTEDSL_RUNTIME_EXPORT const char* CuteDSLRT_GetErrorString(
    CuteDSLRT_Error_t error);

#ifdef __cplusplus
}

namespace xla::gpu::cutedsl {

// XLA's private dispatch table for the CuTeDSL runtime ABI.
struct RuntimeApi {
  decltype(&CuteDSLRT_Module_Create_From_Bytes) module_create_from_bytes;
  decltype(&CuteDSLRT_Module_Get_Function) module_get_function;
  decltype(&CuteDSLRT_Function_Run) function_run;
  decltype(&CuteDSLRT_Module_Destroy) module_destroy;
  decltype(&CuteDSLRT_GetErrorName) get_error_name;
  decltype(&CuteDSLRT_GetErrorString) get_error_string;
};

}  // namespace xla::gpu::cutedsl
#endif

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_RUNTIME_API_H_
