/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_EARLY_CLIENT_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_EARLY_CLIENT_H_

#include "rocm/include/rocprofiler-sdk/fwd.h"

namespace xla {
namespace profiler {

// What the load-time constructor in rocm_tracer_early_client.cc observed.
struct EarlyClientResult {
  // Set when the constructor finishes. The other defaults would pass the test
  // even if the constructor never ran.
  bool constructor_ran = false;
  // rocprofiler_is_initialized()'s return value and the state it reported,
  // before the constructor forced configuration.
  rocprofiler_status_t is_initialized_status = ROCPROFILER_STATUS_SUCCESS;
  int init_status_before = 0;
  rocprofiler_status_t force_configure_status = ROCPROFILER_STATUS_SUCCESS;
};

const EarlyClientResult& GetEarlyClientResult();

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_EARLY_CLIENT_H_
