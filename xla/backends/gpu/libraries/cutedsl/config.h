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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_CONFIG_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_CONFIG_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/libraries/cutedsl/config.pb.h"

namespace xla::gpu::cutedsl {

// Parses and validates a ProtoJSON v3 configuration. Unknown JSON fields are
// ignored, and the returned protobuf owns all parsed data. Callers must not
// mutate the validated configuration.
absl::StatusOr<proto::CollectiveCallConfigV3>
ParseAndValidateCollectiveCallConfig(absl::string_view json_config);

}  // namespace xla::gpu::cutedsl

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUTEDSL_CONFIG_H_
