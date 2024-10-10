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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_GPU_RUNTIME_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_GPU_RUNTIME_H_

#include <string>
#include <vector>

#include "absl/strings/ascii.h"

#include <sycl/sycl.hpp>

enum SYCLError_t {
  SYCL_SUCCESS,
  SYCL_ERROR_NO_DEVICE,
  SYCL_ERROR_NOT_READY,
  SYCL_ERROR_INVALID_DEVICE,
  SYCL_ERROR_INVALID_POINTER,
  SYCL_ERROR_INVALID_STREAM,
  SYCL_ERROR_DESTROY_DEFAULT_STREAM,
  SYCL_ERROR_ZE_ERROR,
};

const char* ToString(SYCLError_t error);

SYCLError_t SYCLGetContext(sycl::context** context);

SYCLError_t SYCLGetDeviceCount(int* count);

SYCLError_t SYCLGetDevice(sycl::device** device, int device_ordinal);

SYCLError_t SYCLGetFrequency(sycl::device* device_handle, uint64_t* freq,
                             uint64_t* mask);

SYCLError_t SYCLCreateStream(sycl::device* device_handle, sycl::queue** stream);

SYCLError_t SYCLDestroyStream(sycl::device* device_handle, sycl::queue* stream);

SYCLError_t SYCLCtxSynchronize(sycl::device* device_handle);

SYCLError_t SYCLMemcpyDtoH(void* dstHost, const void* srcDevice,
                           size_t ByteCount, sycl::device* device);

SYCLError_t SYCLMemcpyHtoD(void* dstDevice, const void* srcHost,
                           size_t ByteCount, sycl::device* device);

SYCLError_t SYCLMemcpyDtoD(void* dstDevice, const void* srcDevice,
                           size_t ByteCount, sycl::device* device);

SYCLError_t SYCLMemcpyDtoHAsync(void* dstHost, const void* srcDevice,
                                size_t ByteCount, sycl::queue* stream);

SYCLError_t SYCLMemcpyHtoDAsync(void* dstDevice, const void* srcHost,
                                size_t ByteCount, sycl::queue* stream);

SYCLError_t SYCLMemcpyDtoDAsync(void* dstDevice, const void* srcDevice,
                                size_t ByteCount, sycl::queue* stream);

SYCLError_t SYCLMemsetD8(void* dstDevice, unsigned char uc, size_t N,
                         sycl::device* device);

SYCLError_t SYCLMemsetD8Async(void* dstDevice, unsigned char uc, size_t N,
                              sycl::queue* stream);

SYCLError_t SYCLMemsetD32(void* dstDevice, unsigned int ui, size_t N,
                          sycl::device* device);

SYCLError_t SYCLMemsetD32Async(void* dstDevice, unsigned int ui, size_t N,
                               sycl::queue* stream);

SYCLError_t SYCLMemcpyAsync(void* dst, const void* src, size_t ByteCount,
                            SYCLError_t (*func)(void*, const void*, size_t,
                                                sycl::queue*),
                            sycl::queue* stream);

SYCLError_t SYCLStreamSynchronize(sycl::queue* stream);

void* SYCLMalloc(sycl::device* device, size_t ByteCount);

void* SYCLMallocHost(sycl::device* device, size_t ByteCount);

void* SYCLMallocShared(sycl::device* device, size_t ByteCount);

void SYCLFree(sycl::device* device, void* ptr);

sycl::event SYCLGetEventFromStream(sycl::queue* stream);

void SYCLStreamDependOnEvents(sycl::queue* stream,
                              const std::vector<sycl::event>& events);

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_GPU_RUNTIME_H_
