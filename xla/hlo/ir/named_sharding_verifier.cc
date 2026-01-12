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

#include "xla/hlo/ir/named_sharding_verifier.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/named_sharding.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

absl::Status VerifyNamedSharding(const NamedSharding& named_sharding) {
  return VerifyNamedSharding(
      named_sharding.mesh(), named_sharding.dim_shardings(),
      named_sharding.replicated_axes(), named_sharding.unreduced_axes());
}

absl::Status VerifyNamedSharding(
    const Mesh& mesh,
    absl::Span<const NamedSharding::DimensionSharding> dim_shardings,
    absl::Span<const AxisRef> replicated_axes,
    absl::Span<const AxisRef> unreduced_axes) {
  std::vector<AxisRef> all_axes;
  int64_t est_size = replicated_axes.size() + unreduced_axes.size();
  for (const auto& ds : dim_shardings) {
    est_size += ds.axes().size();
  }
  all_axes.reserve(est_size);

  for (const auto& ds : dim_shardings) {
    all_axes.insert(all_axes.end(), ds.axes().begin(), ds.axes().end());
  }
  all_axes.insert(all_axes.end(), replicated_axes.begin(),
                  replicated_axes.end());
  all_axes.insert(all_axes.end(), unreduced_axes.begin(), unreduced_axes.end());

  // `ValidateSpanOfAxes` checks:
  // - Each axis is valid for the mesh.
  // - Axes can coexist without overlap.
  TF_RETURN_IF_ERROR(ValidateSpanOfAxes(all_axes, mesh));

  return absl::OkStatus();
}

}  // namespace xla
