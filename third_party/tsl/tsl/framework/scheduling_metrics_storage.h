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
#ifndef TENSORFLOW_TSL_FRAMEWORK_SCHEDULING_METRICS_STORAGE_H_
#define TENSORFLOW_TSL_FRAMEWORK_SCHEDULING_METRICS_STORAGE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "tsl/framework/real_time_in_memory_metric.h"

namespace tsl {

// Storage class that holds all the exported in memory metrics exported by TPU
// runtime.
//
// NOTE: This class is for internal use only.
class SchedulingMetricsStorage {
 public:
  static SchedulingMetricsStorage& GetGlobalStorage();

  // Gets the metrics for estimated total TPU load.
  RealTimeInMemoryMetric<int64_t>& TotalDeviceLoadNs() {
    return total_device_load_ns_;
  }

  const RealTimeInMemoryMetric<int64_t>& TotalDeviceLoadNs() const {
    return total_device_load_ns_;
  }

 private:
  RealTimeInMemoryMetric<int64_t> total_device_load_ns_;
};

}  // namespace tsl

#endif  // TENSORFLOW_TSL_FRAMEWORK_SCHEDULING_METRICS_STORAGE_H_
