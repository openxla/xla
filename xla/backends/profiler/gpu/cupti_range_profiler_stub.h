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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_RANGE_PROFILER_STUB_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_RANGE_PROFILER_STUB_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/backends/profiler/gpu/cupti_range_profiler.h"

namespace xla {
namespace profiler {

class CuptiRangeProfilerStub : public CuptiRangeProfiler {
 public:
  int NumPasses() const override { return 1; }
  absl::Status BeginPass() override { return absl::OkStatus(); }
  absl::Status EndPass() override { return absl::OkStatus(); }
  absl::Status PushRange(absl::string_view name) override {
    return absl::OkStatus();
  }
  absl::Status PopRange() override { return absl::OkStatus(); }
  absl::Status FlushAndDecode() override { return absl::OkStatus(); }
  absl::Status Deinitialize() override { return absl::OkStatus(); }
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_RANGE_PROFILER_STUB_H_
