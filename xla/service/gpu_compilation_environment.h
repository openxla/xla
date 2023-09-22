/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_COMPILATION_ENVIRONMENT_H_
#define XLA_SERVICE_GPU_COMPILATION_ENVIRONMENT_H_
#include <memory>
#include <string>

#include "absl/types/span.h"
#include "xla/statusor.h"
#include "xla/xla.pb.h"

namespace xla {

struct Flag {
  std::string name;
  std::string value;
};

// Default env is currently empty as DebugOptions is already set by default for
// uninitialized flags. This will be modified when all gpu flags are migrated
// from Debugoptions to GpuCompilationEnvironment
std::unique_ptr<GpuCompilationEnvironment> CreateDefaultGpuCompEnv();

StatusOr<std::unique_ptr<GpuCompilationEnvironment>>
CreateGpuCompEnvFromStringPairs(absl::Span<const Flag>, bool strict);

}  // namespace xla
#endif  // XLA_SERVICE_GPU_COMPILATION_ENVIRONMENT_H_
