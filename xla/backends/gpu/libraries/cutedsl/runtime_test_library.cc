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

#include <cstddef>
#include <cstdint>

#include "xla/backends/gpu/libraries/cutedsl/runtime_api.h"

namespace {

CuteDSLRT_Error_t ModuleCreate(CuteDSLRT_Module_t**, const unsigned char*,
                               size_t, const char**, size_t) {
  return CuteDSLRT_Error_Success;
}

CuteDSLRT_Error_t ModuleGetFunction(CuteDSLRT_Function_t**, CuteDSLRT_Module_t*,
                                    const char*) {
  return CuteDSLRT_Error_Success;
}

CuteDSLRT_Error_t FunctionRun(void*, void**, size_t) {
  return CuteDSLRT_Error_Success;
}

CuteDSLRT_Error_t ModuleDestroy(CuteDSLRT_Module_t*) {
  return CuteDSLRT_Error_Success;
}

const char* GetErrorName(CuteDSLRT_Error_t) { return "TestError"; }
const char* GetErrorString(CuteDSLRT_Error_t) { return "test error"; }

}  // namespace

extern "C" CuteDSLRT_Error_t CuteDSLRT_Module_Create_From_Bytes(
    CuteDSLRT_Module_t** module, const unsigned char* bytes, size_t size,
    const char** shared_libraries, size_t shared_library_count) {
  return ModuleCreate(module, bytes, size, shared_libraries,
                      shared_library_count);
}

extern "C" CuteDSLRT_Error_t CuteDSLRT_Module_Get_Function(
    CuteDSLRT_Function_t** function, CuteDSLRT_Module_t* module,
    const char* prefix) {
  return ModuleGetFunction(function, module, prefix);
}

extern "C" CuteDSLRT_Error_t CuteDSLRT_Function_Run(void* function,
                                                    void** arguments,
                                                    size_t argument_count) {
  return FunctionRun(function, arguments, argument_count);
}

extern "C" CuteDSLRT_Error_t CuteDSLRT_Module_Destroy(
    CuteDSLRT_Module_t* module) {
  return ModuleDestroy(module);
}

extern "C" const char* CuteDSLRT_GetErrorName(CuteDSLRT_Error_t error) {
  return GetErrorName(error);
}

extern "C" const char* CuteDSLRT_GetErrorString(CuteDSLRT_Error_t error) {
  return GetErrorString(error);
}
