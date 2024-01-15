/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"

#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/synchronization/mutex.h"
#include "tsl/util/env_var.h"

namespace {
class DevicePool {
 public:
  static sycl::context& getDeviceContext() {
    static sycl::context context(GetDevicesPool());
    return context;
  }

  static SYCLError_t getDeviceCount(int* count) {
    *count = GetDevicesPool().size();
    return SYCL_SUCCESS;
  }

  static SYCLError_t getDevice(sycl::device** device, int device_ordinal) {
    if (device_ordinal >= GetDevicesPool().size()) {
      return SYCL_ERROR_INVALID_DEVICE;
    } else {
      *device = &GetDevicesPool()[device_ordinal];
      return SYCL_SUCCESS;
    }
  }

 private:
  static std::vector<sycl::device>& GetDevicesPool() {
    static absl::once_flag init_device_flag;
    static std::vector<sycl::device> devices;

    absl::call_once(init_device_flag, []() {
      for (const auto& platform : sycl::platform::get_platforms()) {
        auto platform_name = platform.get_info<sycl::info::platform::name>();
        bool is_level_zero =
            platform_name.find("Level-Zero") != std::string::npos;
        if (is_level_zero) {
          LOG(INFO) << "Selected platform: " << platform_name;
          for (const auto& device : platform.get_devices()) {
            if (device.is_gpu()) {
              devices.push_back(device);
            }
          }
        }
      }
      if (devices.size() <= 0) {
        LOG(ERROR) << "Can not found any devices.";
      }
    });

    return devices;
  }
};

}  // namespace

bool IsMultipleStreamEnabled() {
  static absl::once_flag init_flag;
  static bool is_multiple_stream_enabled = false;

  absl::call_once(init_flag, [&]() {
    const char* env = std::getenv("XLA_SYCL_ENABLE_MULTIPLE_STREAM");

    if (env != nullptr) {
      std::string str_value = absl::AsciiStrToLower(env);
      if (str_value == "1" || str_value == "true") {
        is_multiple_stream_enabled = true;
      }
    }
  });

  return is_multiple_stream_enabled;
}

/******************* SYCL context management**************************/
static sycl::async_handler SYCLAsyncHandler = [](sycl::exception_list eL) {
  for (auto& e : eL) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception& e) {
      LOG(ERROR) << "SYCL Exception: " << e.what() << ", file = " << __FILE__
                 << ", line = " << __LINE__ << ".";
    }
  }
};

class SYCLStreamPool {
 public:
  static SYCLError_t getDefaultStream(sycl::device* device_handle,
                                      sycl::queue** stream_p) {
    *stream_p = SYCLStreamPool::GetStreamsPool(device_handle)[0].get();
    return SYCL_SUCCESS;
  }

  static SYCLError_t createStream(sycl::device* device_handle,
                                  sycl::queue** stream_p) {
    if (IsMultipleStreamEnabled()) {
      sycl::property_list propList{sycl::property::queue::in_order()};
      SYCLStreamPool::GetStreamsPool(device_handle)
          .push_back(std::make_shared<sycl::queue>(
              DevicePool::getDeviceContext(), *device_handle, SYCLAsyncHandler,
              propList));
    }
    *stream_p = SYCLStreamPool::GetStreamsPool(device_handle).back().get();
    return SYCL_SUCCESS;
  }

  static SYCLError_t syncContext(sycl::device* device_handle) {
    for (auto stream : SYCLStreamPool::GetStreamsPool(device_handle)) {
      stream->wait();
    }
    return SYCL_SUCCESS;
  }

  static SYCLError_t destroyStream(sycl::device* device_handle,
                                   sycl::queue* stream_handle) {
    if (stream_handle == nullptr) return SYCL_ERROR_INVALID_STREAM;
    auto stream_pool = SYCLStreamPool::GetStreamsPool(device_handle);
    for (int i = 0; i < stream_pool.size(); i++) {
      if (stream_pool[i].get() == stream_handle) {
        stream_pool.erase(stream_pool.begin() + i);
        return SYCL_SUCCESS;
      }
    }
    return SYCL_ERROR_INVALID_STREAM;
  }

 private:
  static std::vector<std::shared_ptr<sycl::queue>>& GetStreamsPool(
      sycl::device* device_handle) {
    static std::unordered_map<sycl::device*,
                              std::vector<std::shared_ptr<sycl::queue>>>
        stream_pool_map;
    auto iter = stream_pool_map.find(device_handle);
    if (iter != stream_pool_map.end()) return iter->second;
    sycl::property_list propList{sycl::property::queue::in_order()};
    std::vector<std::shared_ptr<sycl::queue>> stream_pool = {
        std::make_shared<sycl::queue>(DevicePool::getDeviceContext(),
                                      *device_handle, SYCLAsyncHandler,
                                      propList)};
    stream_pool_map.insert(std::make_pair(device_handle, stream_pool));
    return stream_pool_map[device_handle];
  }
};

SYCLError_t SYCLGetContext(sycl::context** context) {
  *context = &DevicePool::getDeviceContext();
  return SYCL_SUCCESS;
}

SYCLError_t SYCLGetDeviceCount(int* count) {
  return DevicePool::getDeviceCount(count);
}

SYCLError_t SYCLGetDevice(sycl::device** device, int device_ordinal) {
  return DevicePool::getDevice(device, device_ordinal);
}

SYCLError_t SYCLCreateStream(sycl::device* device_handle,
                             sycl::queue** stream_p) {
  return SYCLStreamPool::createStream(device_handle, stream_p);
}

SYCLError_t SYCLDestroyStream(sycl::device* device_handle,
                              sycl::queue* stream_handle) {
  return SYCLStreamPool::destroyStream(device_handle, stream_handle);
}

SYCLError_t SYCLCtxSynchronize(sycl::device* device_handle) {
  return SYCLStreamPool::syncContext(device_handle);
}

/************************* SYCL memory management
 * ***************************/

static void memcpyHostToDevice(void* dstDevice, const void* srcHost,
                               size_t ByteCount, bool async,
                               sycl::queue* stream) {
  if (ByteCount == 0) return;

  auto event = stream->memcpy(dstDevice, srcHost, ByteCount);
  if (!async) {
    event.wait();
  }
}

static void memcpyDeviceToHost(void* dstHost, const void* srcDevice,
                               size_t ByteCount, bool async,
                               sycl::queue* stream) {
  if (ByteCount == 0) return;

  auto event = stream->memcpy(dstHost, srcDevice, ByteCount);

  if (!async) {
    event.wait();
  }
}

static void memcpyDeviceToDevice(void* dstDevice, const void* srcDevice,
                                 size_t ByteCount, bool async,
                                 sycl::queue* stream) {
  if (ByteCount == 0) return;

  auto event = stream->memcpy(dstDevice, srcDevice, ByteCount);

  if (!async) {
    event.wait();
  }
}

static void memsetDeviceD8(void* dstDevice, unsigned char value, size_t n,
                           bool async, sycl::queue* stream) {
  if (n == 0) return;

  auto event = stream->memset(dstDevice, value, n * sizeof(uint8_t));
  if (!async) {
    event.wait();
  }
}

static void memsetDeviceD32(void* dstDevice, int value, size_t n, bool async,
                            sycl::queue* stream) {
  if (n == 0) return;

  auto event = stream->fill(dstDevice, value, n);

  if (!async) {
    event.wait();
  }
}

SYCLError_t SYCLMemcpyDtoH(void* dstHost, const void* srcDevice,
                           size_t ByteCount, sycl::device* device) {
  sycl::queue* stream;
  auto res = SYCLStreamPool::getDefaultStream(device, &stream);
  memcpyDeviceToHost(dstHost, srcDevice, ByteCount, false, stream);
  return res;
}

SYCLError_t SYCLMemcpyHtoD(void* dstDevice, const void* srcHost,
                           size_t ByteCount, sycl::device* device) {
  sycl::queue* stream;
  auto res = SYCLStreamPool::getDefaultStream(device, &stream);
  memcpyHostToDevice(dstDevice, srcHost, ByteCount, false, stream);
  return res;
}

SYCLError_t SYCLMemcpyDtoD(void* dstDevice, const void* srcDevice,
                           size_t ByteCount, sycl::device* device) {
  sycl::queue* stream;
  auto res = SYCLStreamPool::getDefaultStream(device, &stream);
  memcpyDeviceToDevice(dstDevice, srcDevice, ByteCount, false, stream);
  return res;
}

SYCLError_t SYCLMemcpyDtoHAsync(void* dstHost, const void* srcDevice,
                                size_t ByteCount, sycl::queue* stream) {
  sycl::usm::alloc DstAllocType =
      get_pointer_type(dstHost, stream->get_context());
  memcpyDeviceToHost(dstHost, srcDevice, ByteCount,
                     DstAllocType == sycl::usm::alloc::host, stream);
  return SYCL_SUCCESS;
}

SYCLError_t SYCLMemcpyHtoDAsync(void* dstDevice, const void* srcHost,
                                size_t ByteCount, sycl::queue* stream) {
  sycl::usm::alloc SrcAllocType =
      get_pointer_type(srcHost, stream->get_context());
  memcpyHostToDevice(dstDevice, srcHost, ByteCount,
                     SrcAllocType == sycl::usm::alloc::host, stream);
  return SYCL_SUCCESS;
}

SYCLError_t SYCLMemcpyDtoDAsync(void* dstDevice, const void* srcDevice,
                                size_t ByteCount, sycl::queue* stream) {
  memcpyDeviceToDevice(dstDevice, srcDevice, ByteCount, true, stream);
  return SYCL_SUCCESS;
}

SYCLError_t SYCLMemsetD8(void* dstDevice, unsigned char uc, size_t N,
                         sycl::device* device) {
  sycl::queue* stream;
  auto res = SYCLStreamPool::getDefaultStream(device, &stream);
  memsetDeviceD8(dstDevice, uc, N, false, stream);
  return res;
}

SYCLError_t SYCLMemsetD8Async(void* dstDevice, unsigned char uc, size_t N,
                              sycl::queue* stream) {
  memsetDeviceD8(dstDevice, uc, N, true, stream);
  return SYCL_SUCCESS;
}

SYCLError_t SYCLMemsetD32(void* dstDevice, unsigned int ui, size_t N,
                          sycl::device* device) {
  sycl::queue* stream;
  auto res = SYCLStreamPool::getDefaultStream(device, &stream);
  memsetDeviceD32(dstDevice, ui, N, false, stream);
  return res;
}

SYCLError_t SYCLMemsetD32Async(void* dstDevice, unsigned int ui, size_t N,
                               sycl::queue* stream) {
  memsetDeviceD32(dstDevice, ui, N, true, stream);
  return SYCL_SUCCESS;
}

void* SYCLMalloc(sycl::device* device, size_t ByteCount) {
  sycl::queue* stream;
  SYCLStreamPool::getDefaultStream(device, &stream);

  // Always use default 0 stream to allocate mem
  auto ptr = aligned_alloc_device(/*alignment=*/64, ByteCount, *stream);
  return static_cast<void*>(ptr);
}

void* SYCLMallocHost(sycl::device* device, size_t ByteCount) {
  sycl::queue* stream;
  SYCLStreamPool::getDefaultStream(device, &stream);

  // Always use default 0 stream to allocate mem
  auto ptr = aligned_alloc_host(/*alignment=*/64, ByteCount, *stream);
  return static_cast<void*>(ptr);
}

void* SYCLMallocShared(sycl::device* device, size_t ByteCount) {
  sycl::queue* stream;
  SYCLStreamPool::getDefaultStream(device, &stream);

  // Always use default 0 stream to allocate mem
  auto ptr = aligned_alloc_shared(/*alignment=*/64, ByteCount, *stream);
  return static_cast<void*>(ptr);
}

void SYCLFree(sycl::device* device, void* ptr) {
  sycl::queue* stream;
  SYCLStreamPool::getDefaultStream(device, &stream);

  // Always use default 0 stream to free mem
  sycl::free(ptr, *stream);
}

const char* ToString(SYCLError_t error) {
  switch (error) {
    case SYCL_SUCCESS:
      return "SYCL succeed.";
    case SYCL_ERROR_NO_DEVICE:
      return "SYCL did not find the device.";
    case SYCL_ERROR_INVALID_DEVICE:
      return "SYCL got invalid device id.";
    case SYCL_ERROR_INVALID_POINTER:
      return "SYCL got invalid pointer.";
    case SYCL_ERROR_INVALID_STREAM:
      return "SYCL got invalid stream.";
    case SYCL_ERROR_DESTROY_DEFAULT_STREAM:
      return "SYCL cannot destroy default stream.";
    default:
      return "SYCL got invalid error code.";
  }
}
