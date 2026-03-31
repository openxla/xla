/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_C_PJRT_C_API_RAW_BUFFER_EXTERNAL_H_
#define XLA_PJRT_C_PJRT_C_API_RAW_BUFFER_EXTERNAL_H_

#include "absl/status/statusor.h"
#include "xla/future.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"

namespace pjrt {

void PjRtCApiRawBuffer_Destroy(const PJRT_Api* c_api,
                               const PJRT_RawBuffer_Extension* extension,
                               PJRT_RawBuffer* buffer);

PJRT_Memory* PjRtCApiRawBuffer_GetMemorySpace(
    const PJRT_Api* c_api, const PJRT_RawBuffer_Extension* extension,
    PJRT_RawBuffer* buffer);

void* PjRtCApiRawBuffer_GetHostPointer(
    const PJRT_Api* c_api, const PJRT_RawBuffer_Extension* extension,
    PJRT_RawBuffer* buffer);

size_t PjRtCApiRawBuffer_GetOnDeviceSizeInBytes(
    const PJRT_Api* c_api, const PJRT_RawBuffer_Extension* extension,
    PJRT_RawBuffer* buffer);

xla::Future<> PjRtCApiRawBuffer_CopyRawHostToDevice(
    const PJRT_Api* c_api, const PJRT_RawBuffer_Extension* extension,
    PJRT_RawBuffer* buffer, const void* src, int64_t offset,
    int64_t transfer_size);

xla::Future<> PjRtCApiRawBuffer_CopyRawDeviceToHost(
    const PJRT_Api* c_api, const PJRT_RawBuffer_Extension* extension,
    PJRT_RawBuffer* buffer, void* dst, int64_t offset, int64_t transfer_size);

absl::StatusOr<PJRT_RawBuffer*> PjRtCApiBuffer_CreateRawAliasOfBuffer(
    const PJRT_Api* c_api, const PJRT_RawBuffer_Extension* extension,
    PJRT_Buffer* buffer);

}  // namespace pjrt

#endif  // XLA_PJRT_C_PJRT_C_API_RAW_BUFFER_EXTERNAL_H_
