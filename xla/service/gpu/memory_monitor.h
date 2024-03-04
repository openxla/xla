/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MEMORY_MONITOR_H_
#define XLA_SERVICE_GPU_MEMORY_MONITOR_H_

#include <vector>
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"

namespace xla::gpu {

void StartMemoryMonitor(const std::vector<se::StreamExecutor*> executors);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MEMORY_MONITOR_H_
