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

#include "xla/service/gpu/memory_monitor.h"
#include "xla/debug_options_flags.h"

#include "absl/base/call_once.h"
#include "tsl/platform/env.h"
#include "tsl/util/env_var.h"

namespace xla::gpu {

static int UseMemoryMonitoring() {
  return xla::GetDebugOptionsFromFlags().xla_gpu_enable_memory_monitoring();
}

static int64_t GetPollingFrequency() {
  int64_t polling_frequency;
  TF_CHECK_OK(
      tsl::ReadInt64FromEnvVar("TF_GPU_MEM_MONITOR_POLL_FREQ_SECS", 10, &polling_frequency));
  return polling_frequency;
}

void MemoryMonitorThread(const std::vector<se::StreamExecutor*> executors) {
  if (!UseMemoryMonitoring()) return;
  absl::once_flag log_failure_once;
  std::vector<int64_t> peak_memory(executors.size(), 0);
  int64_t polling_frequency = GetPollingFrequency();

  while (true) {
    absl::SleepFor(absl::Seconds(polling_frequency));
    for (int i = 0; i < executors.size(); ++i) {
      int device_ordinal = executors[i]->device_ordinal();
      int64_t free_memory, total_memory;
      if (!executors[i]->DeviceMemoryUsage(&free_memory, &total_memory)) {
        absl::call_once(log_failure_once, [&] {
          LOG(ERROR) << "Failed to query available memory for GPU "
                     << device_ordinal;
        });
        continue;
      }
      int64_t used_memory = total_memory - free_memory;
      if (used_memory > peak_memory[i]) {
        peak_memory[i] = used_memory;
        LOG(INFO) << "GPU Device " << device_ordinal << ": ("
                  << (used_memory >> 20) << " MiB in-use / "
                  << (total_memory >> 20) << " MiB total)";
      }
    }
  }
}

void StartMemoryMonitor(const std::vector<se::StreamExecutor*> executors) {
  static auto* monitor_thread = tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "gpu_memory_monitoring",
      [executors]() { MemoryMonitorThread(executors); });
  (void)monitor_thread;  // suppress unused variable warning
}

}  // namespace xla::gpu
