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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_GPU_RUNTIME_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_GPU_RUNTIME_H_

// clang-format off
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
// clang-format on

#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "xla/stream_executor/sycl/sycl_status.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::sycl {

constexpr int kDefaultDeviceOrdinal = 0;

// The SYCL context constructor expects a std::vector<::sycl::device> as input.
// Therefore, we define DevicePool as a vector of ::sycl::device.
// <https://github.khronos.org/SYCL_Reference/iface/context.html#constructors>
using DevicePool = std::vector<::sycl::device>;

// SyclDevicePool is a utility class for managing SYCL devices.
// It provides methods to access devices, their contexts, and device counts.
// This class cannot be instantiated and is intended to be used as a static
// utility.
class SyclDevicePool {
 public:
  // Returns a static thread-safe SYCL context associated with the device pool.
  // The context is initialized on the first call and remains valid for the
  // lifetime of the process, ensuring all callers share the same context.
  //
  // This function assumes that the device pool is not modified after
  // initialization. If this assumption is violated, the context may become
  // invalid.
  static absl::StatusOr<::sycl::context> GetDeviceContext();

  // Returns the number of devices in the pool.
  static absl::StatusOr<int> GetDeviceCount();

  // Returns the device ordinal for a given device.
  static absl::StatusOr<int> GetDeviceOrdinal(const ::sycl::device& device);

  // Returns the SYCL device for a given device ordinal.
  static absl::StatusOr<::sycl::device> GetDevice(int device_ordinal);

 private:
  // The underlying device pool.
  static DevicePool device_pool_;

  // Thread-safe initialization of device_pool_ with all Level-Zero backend GPUs
  // using absl::call_once.
  static absl::Status InitDevicePool();

  // Prevent instantiation: this class is intended to be a static utility only.
  SyclDevicePool() = delete;
};

using StreamPtr = std::shared_ptr<::sycl::queue>;
using StreamPool = std::vector<StreamPtr>;
using StreamPoolMap = std::unordered_map<int, StreamPool>;

// TODO(intel-tf): kMaxStreamsPerDevice is the maximum number of streams that
// can be created per device via GetOrCreateStream when multiple streams are
// enabled.
//
// For now, we set it to 8 so that there is no unbounded growth. However, it can
// be adjusted based on the device capabilities and workload requirements.
//
// This feature will be enabled by default in the future once the performance
// implications are better understood.
constexpr int kMaxStreamsPerDevice = 8;

// Manages pools of SYCL streams (queues) per device. All methods are static and
// thread-safe via a global mutex. For high concurrency workloads, consider
// refactoring to use per-device mutexes.
// This class cannot be instantiated and is intended to be used as a
// static utility.
class SyclStreamPool {
 public:
  // Returns the default (first in the pool) SYCL stream for the given device
  // ordinal. Returns an error if the device ordinal is invalid or the stream
  // pool is empty.
  static absl::StatusOr<StreamPtr> GetDefaultStream(int device_ordinal);

  // Returns a SYCL stream for the given device ordinal.
  //
  // If multiple streams are not enabled, returns the default (first in the
  // pool) SYCL stream. If the stream pool is empty, returns an error.
  //
  // If multiple streams are enabled (via enable_multiple_streams), creates
  // a new stream up to the maximum limit (kMaxStreamsPerDevice). Returns an
  // error if the limit is reached.
  static absl::StatusOr<StreamPtr> GetOrCreateStream(
      int device_ordinal, bool enable_multiple_streams);

  // Synchronizes all streams associated with the given device ordinal.
  static absl::Status SynchronizeStreamPool(int device_ordinal);

  // Destroys a previously created SYCL stream for the given device ordinal.
  static absl::Status DestroyStream(int device_ordinal,
                                    StreamPtr& stream_handle);

 private:
  // Global mutex protecting the stream pool.
  // TODO(intel-tf): We should consider using a more fine-grained locking
  // mechanism (ex. per-device mutex) in the future to avoid performance issues.
  static absl::Mutex stream_pool_mu_;

  // The underlying stream pool for each device. The device ordinal
  // is used as the key.
  static StreamPoolMap stream_pool_map_ ABSL_GUARDED_BY(stream_pool_mu_);

  // Initializes and returns a pointer to the stream pool for the given device
  // ordinal.
  static absl::StatusOr<StreamPool*> InitStreamPool(int device_ordinal);

  // Prevent instantiation: this class is intended to be a static utility only
  SyclStreamPool() = delete;
};

}  // namespace stream_executor::sycl
#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_GPU_RUNTIME_H_
