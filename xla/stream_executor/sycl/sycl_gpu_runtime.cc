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

#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"

#include <cassert>
#include <iostream>
#include <unordered_map>

#include "absl/base/call_once.h"
#include "absl/synchronization/mutex.h"

namespace stream_executor::sycl {

namespace {

absl::Status IsValidDeviceOrdinal(int device_ordinal,
                                  const absl::string_view& function_name) {
  TF_ASSIGN_OR_RETURN(int device_count, SyclDevicePool::GetDeviceCount());
  if (device_ordinal >= 0 && device_ordinal < device_count) {
    return absl::OkStatus();
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        function_name, ": Invalid device ordinal: ", device_ordinal));
  }
}

absl::Status MemsetDevice(::sycl::queue* stream_handle, void* dst_device,
                          unsigned char value, size_t count,
                          bool async = false) {
  try {
    ::sycl::event event =
        stream_handle->memset(dst_device, value, count * sizeof(uint8_t));
    if (!async) {
      event.wait();
    }
  } catch (const ::sycl::exception& e) {
    return absl::InternalError("MemsetDevice failed: " + std::string(e.what()) +
                               ", file = " + __FILE__ +
                               ", line = " + std::to_string(__LINE__) + ".");
  }
  return absl::OkStatus();
}

absl::Status MemfillDevice(::sycl::queue* stream_handle, void* dst_device,
                           uint32_t value, size_t count, bool async = false) {
  try {
    ::sycl::event event = stream_handle->fill(dst_device, value, count);
    if (!async) {
      event.wait();
    }
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(
        "MemfillDevice failed: " + std::string(e.what()) +
        ", file = " + __FILE__ + ", line = " + std::to_string(__LINE__) + ".");
  }
  return absl::OkStatus();
}

}  // namespace

DevicePool SyclDevicePool::device_pool_;

absl::Status SyclDevicePool::InitDevicePool() {
  static absl::once_flag device_init_flag;
  static absl::Status init_status = absl::OkStatus();
  absl::call_once(device_init_flag, []() {
    DevicePool devices;
    std::vector<::sycl::platform> platform_list =
        ::sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
      std::string platform_name =
          platform.get_info<::sycl::info::platform::name>();
      // Add all Level-Zero backend GPUs to the device pool so that it can be
      // used by the SYCL runtime.
      if (platform_name.find("Level-Zero") != std::string::npos) {
        LOG(INFO) << "Selected platform: " << platform_name;
        std::vector<::sycl::device> device_list = platform.get_devices();
        for (const auto& device : device_list) {
          if (device.is_gpu()) {
            devices.push_back(device);
          }
        }
      }
    }
    if (devices.empty()) {
      init_status = absl::InternalError(
          "SyclDevicePool::InitDevicePool: No SYCL devices found with "
          "Level-Zero "
          "backend. Check oneAPI installation and environment variables.");
      return;
    }
    device_pool_ = std::move(devices);
  });
  return init_status;
}

absl::StatusOr<::sycl::context> SyclDevicePool::GetDeviceContext() {
  TF_RETURN_IF_ERROR(SyclDevicePool::InitDevicePool());
  static ::sycl::context device_context(device_pool_);
  return device_context;
}

absl::StatusOr<int> SyclDevicePool::GetDeviceCount() {
  TF_RETURN_IF_ERROR(SyclDevicePool::InitDevicePool());
  // Cast to int since device_ordinal is usually an int.
  return static_cast<int>(device_pool_.size());
}

absl::StatusOr<int> SyclDevicePool::GetDeviceOrdinal(
    const ::sycl::device& device) {
  TF_RETURN_IF_ERROR(SyclDevicePool::InitDevicePool());
  auto it = std::find(device_pool_.begin(), device_pool_.end(), device);
  if (it != device_pool_.end()) {
    return static_cast<int>(it - device_pool_.begin());
  } else {
    return absl::InternalError(
        "SyclDevicePool::GetDeviceOrdinal failed, got invalid device");
  }
}

absl::StatusOr<::sycl::device> SyclDevicePool::GetDevice(int device_ordinal) {
  TF_RETURN_IF_ERROR(SyclDevicePool::InitDevicePool());
  TF_RETURN_IF_ERROR(
      IsValidDeviceOrdinal(device_ordinal, "SyclDevicePool::GetDevice"));
  return device_pool_[device_ordinal];
}

StreamPoolMap SyclStreamPool::stream_pool_map_;
absl::Mutex SyclStreamPool::stream_pool_mu_(absl::kConstInit);

static const ::sycl::async_handler SyclAsyncHandler =
    [](::sycl::exception_list ex_list) {
      for (auto& e : ex_list) {
        try {
          std::rethrow_exception(e);
        } catch (::sycl::exception& e) {
          LOG(ERROR) << "SYCL exception: " << e.what()
                     << ", file = " << __FILE__ << ", line = " << __LINE__
                     << ".";
        }
      }
    };

absl::StatusOr<StreamPool*> SyclStreamPool::InitStreamPool(int device_ordinal) {
  {
    absl::ReaderMutexLock read_lock(&stream_pool_mu_);
    auto it = stream_pool_map_.find(device_ordinal);
    // Returns the existing non-empty stream pool for this device, if available.
    // The pool may be empty if DestroyStream was called on the last stream.
    if (it != stream_pool_map_.end() && !it->second.empty()) {
      VLOG(2) << "Check 1: Returning existing stream pool for device ordinal "
              << device_ordinal << " whose size is " << it->second.size();
      return &(it->second);
    }
  }
  // Creates a new stream pool for this device using the device and context.
  ::sycl::property_list prop_list{::sycl::property::queue::enable_profiling(),
                                  ::sycl::property::queue::in_order()};
  TF_ASSIGN_OR_RETURN(::sycl::device sycl_device,
                      SyclDevicePool::GetDevice(device_ordinal));
  TF_ASSIGN_OR_RETURN(::sycl::context sycl_context,
                      SyclDevicePool::GetDeviceContext());

  VLOG(2) << "Creating new stream pool for device ordinal " << device_ordinal;
  absl::MutexLock write_lock(&stream_pool_mu_);
  auto it = stream_pool_map_.find(device_ordinal);
  // Double-checks that another thread has not already created the pool.
  if (it != stream_pool_map_.end() && !it->second.empty()) {
    VLOG(2) << "Check 2: Returning existing stream pool for device ordinal "
            << device_ordinal << " whose size is " << it->second.size();
    return &(it->second);
  }

  StreamPool stream_pool = {std::make_shared<::sycl::queue>(
      sycl_context, sycl_device, SyclAsyncHandler, prop_list)};

  // Use assignment (not insert) to update the stream pool if it was
  // previously destroyed.
  stream_pool_map_[device_ordinal] = std::move(stream_pool);

  return &(stream_pool_map_[device_ordinal]);
}

absl::StatusOr<StreamPtr> SyclStreamPool::GetDefaultStream(int device_ordinal) {
  TF_RETURN_IF_ERROR(
      IsValidDeviceOrdinal(device_ordinal, "SyclStreamPool::GetDefaultStream"));
  TF_ASSIGN_OR_RETURN(StreamPool * stream_pool,
                      SyclStreamPool::InitStreamPool(device_ordinal));
  // InitStreamPool always returns a valid pointer, so no null check is needed.
  absl::ReaderMutexLock read_lock(&stream_pool_mu_);
  if (stream_pool->empty()) {
    return absl::InternalError(
        absl::StrCat("SyclStreamPool::GetDefaultStream: Stream pool is empty "
                     "for device ordinal ",
                     device_ordinal,
                     ". The pool may have been destroyed by another thread."));
  }
  return stream_pool->front();
}

absl::StatusOr<StreamPtr> SyclStreamPool::GetOrCreateStream(
    int device_ordinal, bool enable_multiple_streams) {
  VLOG(2) << "SyclStreamPool::GetOrCreateStream called for device ordinal "
          << device_ordinal
          << ", enable_multiple_streams: " << enable_multiple_streams;
  if (!enable_multiple_streams) {
    return SyclStreamPool::GetDefaultStream(device_ordinal);
  } else {
    TF_RETURN_IF_ERROR(IsValidDeviceOrdinal(
        device_ordinal, "SyclStreamPool::GetOrCreateStream"));
    TF_ASSIGN_OR_RETURN(StreamPool * stream_pool,
                        SyclStreamPool::InitStreamPool(device_ordinal));
    // If multiple streams are enabled, create a new stream and add it
    // to the pool, unless the pool has reached kMaxStreamsPerDevice.
    absl::MutexLock write_lock(&stream_pool_mu_);
    if (stream_pool->size() >= kMaxStreamsPerDevice) {
      VLOG(2) << "Stream pool size for device ordinal " << device_ordinal
              << " exceeds the maximum limit of " << kMaxStreamsPerDevice;
      return absl::ResourceExhaustedError(
          absl::StrCat("SyclStreamPool::GetOrCreateStream: Maximum number of "
                       "streams reached for device ordinal ",
                       device_ordinal, "."));
    }
    VLOG(2) << "Stream pool size for device ordinal " << device_ordinal << ": "
            << stream_pool->size();
    ::sycl::property_list prop_list{::sycl::property::queue::enable_profiling(),
                                    ::sycl::property::queue::in_order()};
    TF_ASSIGN_OR_RETURN(::sycl::device sycl_device,
                        SyclDevicePool::GetDevice(device_ordinal));
    TF_ASSIGN_OR_RETURN(::sycl::context sycl_context,
                        SyclDevicePool::GetDeviceContext());
    stream_pool->push_back(std::make_shared<::sycl::queue>(
        sycl_context, sycl_device, SyclAsyncHandler, prop_list));
    return stream_pool->back();
  }
}

absl::Status SyclStreamPool::SynchronizeStreamPool(int device_ordinal) {
  TF_RETURN_IF_ERROR(IsValidDeviceOrdinal(
      device_ordinal, "SyclStreamPool::SynchronizeStreamPool"));
  TF_ASSIGN_OR_RETURN(StreamPool * stream_pool,
                      SyclStreamPool::InitStreamPool(device_ordinal));
  absl::ReaderMutexLock read_lock(&stream_pool_mu_);
  if (stream_pool->empty()) {
    return absl::InternalError(
        absl::StrCat("SyclStreamPool::SynchronizeStreamPool: Stream pool is "
                     "empty for device ordinal ",
                     device_ordinal,
                     ". The pool may have been destroyed by another thread."));
  }
  for (auto& stream : *stream_pool) {
    stream->wait();
  }
  return absl::OkStatus();
}

absl::Status SyclStreamPool::DestroyStream(int device_ordinal,
                                           StreamPtr& stream_handle) {
  if (stream_handle == nullptr) {
    return absl::InvalidArgumentError(
        "SyclStreamPool::DestroyStream: Attempting to destroy a null stream "
        "handle.");
  }
  TF_RETURN_IF_ERROR(
      IsValidDeviceOrdinal(device_ordinal, "SyclStreamPool::DestroyStream"));
  TF_ASSIGN_OR_RETURN(StreamPool * stream_pool,
                      SyclStreamPool::InitStreamPool(device_ordinal));
  absl::MutexLock write_lock(&stream_pool_mu_);
  if (stream_pool->empty()) {
    return absl::InternalError(
        absl::StrCat("SyclStreamPool::DestroyStream: Stream pool is empty for "
                     "device ordinal ",
                     device_ordinal,
                     ". The pool may have been destroyed by another thread."));
  }
  auto it = std::find(stream_pool->begin(), stream_pool->end(), stream_handle);
  if (it == stream_pool->end()) {
    return absl::NotFoundError(absl::StrCat(
        "SyclStreamPool::DestroyStream: Stream handle for device ordinal ",
        device_ordinal, " not found in the pool."));
  } else {
    // Remove the stream from the pool and reset the handle.
    // The stream pool remains, but may become empty.
    stream_pool->erase(it);
    stream_handle.reset();
    VLOG(2) << "Successfully destroyed stream for device ordinal "
            << device_ordinal << ", stream pool size is "
            << stream_pool->size();
    return absl::OkStatus();
  }
}

absl::Status SyclMemsetDevice(int device_ordinal, void* dst_device,
                              unsigned char value, size_t count) {
  if (dst_device == nullptr) {
    return absl::InvalidArgumentError(
        "SyclMemsetDevice: Null pointer provided for destination.");
  }
  if (count == 0) {
    LOG(WARNING) << "SyclMemsetDevice: Attempting to set zero bytes, "
                    "skipping operation.";
    return absl::OkStatus();
  }
  TF_RETURN_IF_ERROR(IsValidDeviceOrdinal(device_ordinal, "SyclMemsetDevice"));
  TF_ASSIGN_OR_RETURN(StreamPtr stream_handle,
                      SyclStreamPool::GetDefaultStream(device_ordinal));
  return MemsetDevice(stream_handle.get(), dst_device, value, count);
}

absl::Status SyclMemsetDeviceAsync(::sycl::queue* stream_handle,
                                   void* dst_device, unsigned char value,
                                   size_t count) {
  if (dst_device == nullptr || stream_handle == nullptr) {
    return absl::InvalidArgumentError(
        "SyclMemsetDeviceAsync: Null pointer provided for destination or "
        "stream handle.");
  }
  if (count == 0) {
    LOG(WARNING) << "SyclMemsetDeviceAsync: Attempting to set zero bytes, "
                    "skipping operation.";
    return absl::OkStatus();
  }
  return MemsetDevice(stream_handle, dst_device, value, count, /*async=*/true);
}

absl::Status SyclMemfillDevice(int device_ordinal, void* dst_device,
                               uint32_t value, size_t count) {
  if (dst_device == nullptr) {
    return absl::InvalidArgumentError(
        "SyclMemfillDevice: Null pointer provided for destination.");
  }
  if (count == 0) {
    LOG(WARNING) << "SyclMemfillDevice: Attempting to fill zero bytes, "
                    "skipping operation.";
    return absl::OkStatus();
  }
  TF_RETURN_IF_ERROR(IsValidDeviceOrdinal(device_ordinal, "SyclMemfillDevice"));
  TF_ASSIGN_OR_RETURN(StreamPtr stream_handle,
                      SyclStreamPool::GetDefaultStream(device_ordinal));
  return MemfillDevice(stream_handle.get(), dst_device, value, count);
}

absl::Status SyclMemfillDeviceAsync(::sycl::queue* stream_handle,
                                    void* dst_device, uint32_t value,
                                    size_t count) {
  if (dst_device == nullptr || stream_handle == nullptr) {
    return absl::InvalidArgumentError(
        "SyclMemfillDeviceAsync: Null pointer provided for destination or "
        "stream handle.");
  }
  if (count == 0) {
    LOG(WARNING) << "SyclMemfillDeviceAsync: Attempting to fill zero bytes, "
                    "skipping operation.";
    return absl::OkStatus();
  }
  return MemfillDevice(stream_handle, dst_device, value, count, /*async=*/true);
}

// TODO(intel-tf): Need OOM checks for all SYCL memory allocation functions.
absl::StatusOr<void*> SyclMallocDevice(int device_ordinal, size_t byte_count) {
  if (byte_count == 0) {
    LOG(WARNING) << "SyclMallocDevice: Attempting to allocate zero bytes, "
                    "returning nullptr.";
    return nullptr;
  }
  TF_RETURN_IF_ERROR(IsValidDeviceOrdinal(device_ordinal, "SyclMallocDevice"));
  TF_ASSIGN_OR_RETURN(StreamPtr stream_handle,
                      SyclStreamPool::GetDefaultStream(device_ordinal));
  try {
    // Use the default stream to allocate memory
    void* ptr =
        aligned_alloc_device(/*alignment=*/64, byte_count, *stream_handle);
    return ptr;
  } catch (const std::exception& e) {
    return absl::InternalError(absl::StrCat(
        "SyclMallocDevice: Failed to allocate device memory: ", e.what(),
        ", file = ", __FILE__, ", line = ", __LINE__));
  }
}

absl::StatusOr<void*> SyclMallocHost(int device_ordinal, size_t byte_count) {
  if (byte_count == 0) {
    LOG(WARNING) << "SyclMallocHost: Attempting to allocate zero bytes, "
                    "returning nullptr.";
    return nullptr;
  }
  TF_RETURN_IF_ERROR(IsValidDeviceOrdinal(device_ordinal, "SyclMallocHost"));
  TF_ASSIGN_OR_RETURN(StreamPtr stream_handle,
                      SyclStreamPool::GetDefaultStream(device_ordinal));
  try {
    // Use the default stream to allocate memory
    void* ptr =
        aligned_alloc_host(/*alignment=*/64, byte_count, *stream_handle);
    return ptr;
  } catch (const std::exception& e) {
    return absl::InternalError(absl::StrCat(
        "SyclMallocHost: Failed to allocate host memory: ", e.what(),
        ", file = ", __FILE__, ", line = ", __LINE__));
  }
}

absl::StatusOr<void*> SyclMallocShared(int device_ordinal, size_t byte_count) {
  if (byte_count == 0) {
    LOG(WARNING) << "SyclMallocShared: Attempting to allocate zero bytes, "
                    "returning nullptr.";
    return nullptr;
  }
  TF_RETURN_IF_ERROR(IsValidDeviceOrdinal(device_ordinal, "SyclMallocShared"));
  TF_ASSIGN_OR_RETURN(StreamPtr stream_handle,
                      SyclStreamPool::GetDefaultStream(device_ordinal));
  try {
    // Use the default stream to allocate memory
    void* ptr =
        aligned_alloc_shared(/*alignment=*/64, byte_count, *stream_handle);
    return ptr;
  } catch (const std::exception& e) {
    return absl::InternalError(absl::StrCat(
        "SyclMallocShared: Failed to allocate shared memory: ", e.what(),
        ", file = ", __FILE__, ", line = ", __LINE__));
  }
}

absl::Status SyclFree(int device_ordinal, void*& ptr) {
  if (ptr == nullptr) {
    return absl::InvalidArgumentError(
        "SyclFree: Attempting to free a null pointer.");
  }
  TF_RETURN_IF_ERROR(IsValidDeviceOrdinal(device_ordinal, "SyclFree"));
  TF_ASSIGN_OR_RETURN(StreamPtr stream_handle,
                      SyclStreamPool::GetDefaultStream(device_ordinal));
  try {
    // Use the default stream to free memory
    ::sycl::free(ptr, *stream_handle);
    ptr = nullptr;
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(
        absl::StrCat("SyclFree: Failed to free memory: ", e.what(),
                     ", file = ", __FILE__, ", line = ", __LINE__));
  }
  return absl::OkStatus();
}

}  // namespace stream_executor::sycl
