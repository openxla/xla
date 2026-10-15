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

// Stands in for another rocprofiler-sdk tool loaded into the same process.
// Its load-time constructor forces rocprofiler-sdk configuration, so
// rocprofiler-sdk initializes and its client discovery dlopens the ROCm tracer
// library before that library's own constructor has run. This is how the
// tracer's constructor ends up running inside rocprofiler-sdk initialization
// when, for example, librocfft registers its HIP kernels first.

#include "xla/backends/profiler/gpu/rocm_tracer_early_client.h"

#include <cstdint>

#include "absl/base/attributes.h"
#include "rocm/include/rocprofiler-sdk/fwd.h"
#include "rocm/include/rocprofiler-sdk/registration.h"

namespace xla {
namespace profiler {
namespace {

// Constant-initialized, so no dynamic initializer can overwrite what the
// constructor below records.
ABSL_CONST_INIT EarlyClientResult early_client_result;

rocprofiler_tool_configure_result_t* ConfigureEarlyClient(
    uint32_t version, const char* runtime_version, uint32_t priority,
    rocprofiler_client_id_t* client_id) {
  // This client only starts rocprofiler-sdk initialization; it traces nothing.
  // rocprofiler-sdk treats nullptr as the client declining, not as an error
  // (see the example in rocprofiler-sdk/registration.h), which is why the test
  // can expect rocprofiler_force_configure() to succeed.
  return nullptr;
}

void __attribute__((constructor)) ForceConfigureEarly() {
  early_client_result.is_initialized_status =
      rocprofiler_is_initialized(&early_client_result.init_status_before);
  early_client_result.force_configure_status =
      rocprofiler_force_configure(&ConfigureEarlyClient);
  early_client_result.constructor_ran = true;
}

}  // namespace

const EarlyClientResult& GetEarlyClientResult() { return early_client_result; }

}  // namespace profiler
}  // namespace xla
