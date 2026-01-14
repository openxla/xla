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

#ifndef XLA_HLO_IR_NAMED_SHARDING_VERIFIER_H_
#define XLA_HLO_IR_NAMED_SHARDING_VERIFIER_H_

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/named_sharding.h"

namespace xla {

// Verifies that the `NamedSharding` is valid.
// Checks:
// - All axes indices are within mesh bounds.
// - No duplicate axes usage across dim_shardings, replicated_axes, and
//   unreduced_axes (unless allowed by sub-axes logic, i.e. non-overlapping).
// - Sub-axes validity.
absl::Status VerifyNamedSharding(const NamedSharding& named_sharding);

// Verifies the components of a `NamedSharding`.
absl::Status VerifyNamedSharding(
    const Mesh& mesh,
    absl::Span<const NamedSharding::DimensionSharding> dim_shardings,
    absl::Span<const AxisRef> replicated_axes,
    absl::Span<const AxisRef> unreduced_axes);

}  // namespace xla

#endif  // XLA_HLO_IR_NAMED_SHARDING_VERIFIER_H_
