/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_PJRT_C_PJRT_C_API_PROFILER_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_PROFILER_EXTENSION_H_

#include <cstddef>
#include <cstdint>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PJRT_API_PROFILER_EXTENSION_VERSION 0

typedef struct PJRT_Profiler PJRT_Profiler;

struct PJRT_Profiler_Create_Args {
  size_t struct_size;
  const char* options;
  size_t options_size;
  PJRT_Profiler* profiler;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Profiler_Create_Args, profiler);

typedef PJRT_Error* PJRT_Profiler_Create(PJRT_Profiler_Create_Args* args);

struct PJRT_Profiler_Destroy_Args {
  size_t struct_size;
  PJRT_Profiler* profiler;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Profiler_Destroy_Args, profiler);

typedef PJRT_Error* PJRT_Profiler_Destroy(PJRT_Profiler_Destroy_Args* args);

struct PJRT_Profiler_Start_Args {
  size_t struct_size;
  PJRT_Profiler* profiler;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Profiler_Start_Args, profiler);

typedef PJRT_Error* PJRT_Profiler_Start(PJRT_Profiler_Start_Args* args);

struct PJRT_Profiler_Stop_Args {
  size_t struct_size;
  PJRT_Profiler* profiler;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Profiler_Stop_Args, profiler);

typedef PJRT_Error* PJRT_Profiler_Stop(PJRT_Profiler_Stop_Args* args);

struct PJRT_Profiler_CollectData_Args {
  size_t struct_size;
  PJRT_Profiler* profiler;
  uint8_t* buffer;              // in/out
  size_t buffer_size_in_bytes;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Profiler_CollectData_Args, buffer_size_in_bytes);

typedef PJRT_Error* PJRT_Profiler_CollectData(
    PJRT_Profiler_CollectData_Args* args);

typedef struct PJRT_Profiler_Extension {
  PJRT_Structure_Type type;
  const void* next;
  PJRT_Profiler_Create* create;
  PJRT_Profiler_Destroy* destroy;
  PJRT_Profiler_Start* start;
  PJRT_Profiler_Stop* stop;
  PJRT_Profiler_CollectData* collect_data;
} PJRT_Profiler_Extension;

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_PROFILER_EXTENSION_H_
