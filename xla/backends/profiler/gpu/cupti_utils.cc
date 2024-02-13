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
#include <string_view>

#include "absl/base/call_once.h"
#include "absl/memory/memory.h"
#include "xla/backends/profiler/gpu/cupti_error_manager.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"
#include "xla/backends/profiler/gpu/cupti_wrapper.h"
#include "tsl/util/env_var.h"

namespace xla {
namespace profiler {

bool IsCuptiDisabled() {
  static constexpr std::string_view kEnvStringDisableCupti =
      "TF_GPU_PROFILER_DISABLE_CUPTI";
  static constexpr bool kDisableCupti = false;
  static absl::once_flag once;  // NOLINT(clang-diagnostic-unreachable-code)
  static bool cupti_disabled = kDisableCupti;
  absl::call_once(once, [&] {
    auto status = tsl::ReadBoolFromEnvVar(kEnvStringDisableCupti, kDisableCupti,
                                          &cupti_disabled);
    if (!status.ok()) {
      LOG(WARNING) << "TF_XLA_PROFILER_GPU_DISABLE_CUPTI is not set to "
                   << "either '0', 'false', '1', or 'true'. Using the "
                   << "default setting: " << kDisableCupti;
    }
    if (cupti_disabled) {
      LOG(INFO) << kEnvStringDisableCupti << " is set to true, "
                << "XLA Profiler disabled GPU CUPTI interface to work around "
                << "potential serious bug in CUPTI lib. Such control may be "
                << "removed/disabled in future if the known issue is resolved!";
    }
  });
  return cupti_disabled;
}

CuptiInterface* GetCuptiInterface() {
  static CuptiInterface* cupti_interface =
      IsCuptiDisabled()
          ? new CuptiErrorManager(std::make_unique<CuptiWrapperStub>())
          : new CuptiErrorManager(std::make_unique<CuptiWrapper>());
  return cupti_interface;
}

}  // namespace profiler
}  // namespace xla
