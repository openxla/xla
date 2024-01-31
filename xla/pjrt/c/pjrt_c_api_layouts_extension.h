/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PJRT_C_PJRT_C_API_LAYOUTS_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_LAYOUTS_EXTENSION_H_

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PJRT_API_LAYOUTS_EXTENSION_VERSION 1

typedef struct PJRT_Layouts_MemoryLayout PJRT_Layouts_MemoryLayout;

// ---------------------------------- Methods ----------------------------------

struct PJRT_Layouts_PJRT_Buffer_GetMemoryLayout_Args {
  size_t struct_size;
  const PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Layouts_MemoryLayout* layout;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Layouts_PJRT_Buffer_GetMemoryLayout_Args,
                          layout);

// Returns the memory layout of the data in this buffer.
typedef PJRT_Error* PJRT_Layouts_PJRT_Buffer_GetMemoryLayout(
    PJRT_Layouts_PJRT_Buffer_GetMemoryLayout_Args* args);

// --------------------------- Extension entrypoint ----------------------------

typedef struct PJRT_Layouts_Extension {
  size_t struct_size;
  PJRT_Extension_Type type;
  const PJRT_Extension_Base* next;

  PJRT_Layouts_PJRT_Buffer_GetMemoryLayout*
      PJRT_Layouts_PJRT_Buffer_GetMemoryLayout;
} PJRT_Layouts_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Layouts_Extension,
                          PJRT_Layouts_PJRT_Buffer_GetMemoryLayout);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_LAYOUTS_EXTENSION_H_
