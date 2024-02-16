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
#ifndef TENSORFLOW_TSL_FRAMEWORK_SCHEDULING_METRICS_TEST_UTIL_H_
#define TENSORFLOW_TSL_FRAMEWORK_SCHEDULING_METRICS_TEST_UTIL_H_

// Util methods for scheduling metrics testing purpose.
#include <cstdint>
namespace tsl {
namespace test_util {

// Sets TPU load in SchedulingMetricsStorage.
void SetDeviceLoad(const int64_t device_load_ns);

}  // namespace test_util
}  // namespace tsl

#endif  // TENSORFLOW_TSL_FRAMEWORK_SCHEDULING_METRICS_TEST_UTIL_H_
