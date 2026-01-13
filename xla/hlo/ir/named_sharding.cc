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

#include "xla/hlo/ir/named_sharding.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_op_metadata.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/tile_assignment.h"

namespace xla {

namespace {

std::vector<xla::AxisRef> GetOrderedAxisRefs(
    const xla::NamedSharding& sharding) {
  std::map<int64_t, std::vector<int64_t>> axis_index_to_pre_sizes;
  const Mesh& mesh = sharding.mesh();
  for (int i = 0; i < mesh.axis_sizes().size(); ++i) {
    axis_index_to_pre_sizes[i].push_back(1);
    axis_index_to_pre_sizes[i].push_back(mesh.axis_sizes()[i]);
  }

  auto collect_axis_ref = [&](const AxisRef& axis_ref) {
    if (axis_ref.sub_axis_info()) {
      axis_index_to_pre_sizes[axis_ref.mesh_axis_index()].push_back(
          axis_ref.sub_axis_info()->pre_size);
      axis_index_to_pre_sizes[axis_ref.mesh_axis_index()].push_back(
          axis_ref.sub_axis_info()->pre_size * axis_ref.sub_axis_info()->size);
    }
  };

  for (const auto& dim_sharding : sharding.dim_shardings()) {
    for (const AxisRef& axis_ref : dim_sharding.axes()) {
      collect_axis_ref(axis_ref);
    }
  }
  for (const AxisRef& axis_ref : sharding.replicated_axes()) {
    collect_axis_ref(axis_ref);
  }
  for (const AxisRef& axis_ref : sharding.unreduced_axes()) {
    collect_axis_ref(axis_ref);
  }
  // TODO: Handle manual axes.

  std::vector<xla::AxisRef> axis_refs;
  for (int i = 0; i < mesh.axis_sizes().size(); ++i) {
    auto& pre_sizes = axis_index_to_pre_sizes[i];
    std::sort(pre_sizes.begin(), pre_sizes.end());
    pre_sizes.erase(std::unique(pre_sizes.begin(), pre_sizes.end()),
                    pre_sizes.end());
    if (pre_sizes.size() == 2) {
      axis_refs.push_back(xla::AxisRef(i));
      continue;
    }
    for (int j = 0; j < pre_sizes.size() - 1; ++j) {
      int64_t pre_size = pre_sizes[j];
      int64_t size = pre_sizes[j + 1] / pre_size;
      axis_refs.push_back(xla::AxisRef(i, {pre_size, size}));
    }
  }
  return axis_refs;
}

}  // namespace

void NamedSharding::DimensionSharding::Append(
    const NamedSharding::DimensionSharding& other, const Mesh& mesh) {
  if (other.axes_.empty()) {
    return;
  }
  if (axes_.empty()) {
    axes_ = other.axes_;
    return;
  }

  // Merge last element of `axes_` with first element of `other.axes_`
  if (!axes_.back().Merge(other.axes_.front(), mesh)) {
    axes_.push_back(other.axes_.front());
  }

  axes_.insert(axes_.end(), other.axes_.begin() + 1, other.axes_.end());
}

std::optional<NamedSharding::DimensionSharding>
NamedSharding::DimensionSharding::Slice(const Mesh& mesh, int64_t slice_size) {
  if (slice_size == 1) {
    return DimensionSharding({}, is_closed_);
  }
  if (getShardedSize(mesh) % slice_size != 0) {
    return std::nullopt;
  }

  int64_t axis_index = 0;
  std::vector<AxisRef> sliced_axes, remaining_axes;

  for (; axis_index < axes().size(); ++axis_index) {
    const AxisRef& curr_axis = axes()[axis_index];
    int64_t curr_axis_size = curr_axis.size(mesh);

    if (slice_size == curr_axis_size) {
      sliced_axes =
          std::vector<AxisRef>(axes().begin(), axes().begin() + axis_index + 1);
      slice_size = 1;
      break;
    }
    if (slice_size % curr_axis_size == 0) {
      slice_size /= curr_axis_size;
    } else if (curr_axis_size % slice_size == 0) {
      sliced_axes =
          std::vector<AxisRef>(axes().begin(), axes().begin() + axis_index);
      int64_t sliced_axis_pre_size =
          curr_axis.sub_axis_info() ? curr_axis.sub_axis_info()->pre_size : 1;
      sliced_axes.push_back(AxisRef(curr_axis.mesh_axis_index(),
                                    {sliced_axis_pre_size, slice_size}));
      remaining_axes.push_back(AxisRef(
          curr_axis.mesh_axis_index(),
          {sliced_axis_pre_size * slice_size, curr_axis_size / slice_size}));
      slice_size = 1;
      break;
    } else {
      return std::nullopt;
    }
  }

  if (slice_size != 1) {
    return std::nullopt;
  }

  remaining_axes.insert(remaining_axes.end(), axes().begin() + axis_index + 1,
                        axes().end());
  axes_ = std::move(remaining_axes);
  return NamedSharding::DimensionSharding(sliced_axes, is_closed_);
}

int64_t NamedSharding::DimensionSharding::getShardedSize(
    const Mesh& mesh) const {
  return std::accumulate(axes_.begin(), axes_.end(), 1,
                         [&mesh](int64_t cur, const AxisRef& axis) {
                           return cur * axis.size(mesh);
                         });
}

std::string NamedSharding::DimensionSharding::ToString(const Mesh* mesh) const {
  std::string result = "{";
  absl::StrAppend(
      &result,
      absl::StrJoin(axes_, ", ", [mesh](std::string* out, const AxisRef& axis) {
        absl::StrAppend(out, axis.ToString(mesh));
      }));

  if (!is_closed_) {
    if (axes_.empty()) {
      absl::StrAppend(&result, "?");
    } else {
      absl::StrAppend(&result, ", ?");
    }
  }

  absl::StrAppend(&result, "}");
  return result;
}

std::string NamedSharding::ToString(bool include_metadata) const {
  std::string result = "{";

  std::string metadata_str;
  if (include_metadata && !metadata_.empty()) {
    metadata_str = ", metadata={";
    absl::StrAppend(
        &metadata_str,
        absl::StrJoin(
            metadata_, ", ", [&](std::string* out, const auto& metadata) {
              absl::StrAppend(out, "{", OpMetadataToString(metadata), "}");
            }));
    absl::StrAppend(&metadata_str, "}");
  }

  // Special cases.
  if (IsReplicated() && replicated_axes_.empty()) {
    absl::StrAppend(&result, "replicated");
    absl::StrAppend(&result, metadata_str);
    absl::StrAppend(&result, "}");
    return result;
  }

  if (IsMaximal()) {
    absl::StrAppend(&result, "maximal device=");
    absl::StrAppend(&result, *mesh_.device_assignment().array().begin());
    absl::StrAppend(&result, metadata_str);
    absl::StrAppend(&result, "}");
    return result;
  }

  absl::StrAppend(&result, mesh_.ToString());

  // Dimension sharding.
  absl::StrAppend(&result, ", [");
  absl::StrAppend(
      &result,
      absl::StrJoin(dim_shardings_, ", ",
                    [&](std::string* out, const DimensionSharding& ds) {
                      absl::StrAppend(out, ds.ToString(&mesh_));
                    }));
  absl::StrAppend(&result, "]");

  if (!replicated_axes_.empty()) {
    absl::StrAppend(&result, ", replicated={");
    absl::StrAppend(&result,
                    absl::StrJoin(replicated_axes_, ", ",
                                  [&](std::string* out, const AxisRef& axis) {
                                    absl::StrAppend(out, axis.ToString(&mesh_));
                                  }));
    absl::StrAppend(&result, "}");
  }

  if (!unreduced_axes_.empty()) {
    absl::StrAppend(&result, ", unreduced={");
    absl::StrAppend(&result,
                    absl::StrJoin(unreduced_axes_, ", ",
                                  [&](std::string* out, const AxisRef& axis) {
                                    absl::StrAppend(out, axis.ToString(&mesh_));
                                  }));
    absl::StrAppend(&result, "}");
  }

  absl::StrAppend(&result, metadata_str);
  absl::StrAppend(&result, "}");

  return result;
}

TileAssignment NamedSharding::ToTileAssignment() const {
  if (IsReplicated()) {
    return TileAssignment();
  }
  if (IsMaximal()) {
    return mesh_.device_assignment();
  }
  // TODO: Add support for manual axes.

  std::vector<int64_t> tile_assignment_dims;
  tile_assignment_dims.reserve(dim_shardings_.size());
  std::map<AxisRef, int64_t> axis_ref_to_sharded_pos;
  int64_t sharded_pos = 0;
  for (const DimensionSharding& dim_sharding : dim_shardings_) {
    tile_assignment_dims.push_back(dim_sharding.getShardedSize(mesh_));
    for (const AxisRef& axis_ref : dim_sharding.axes()) {
      axis_ref_to_sharded_pos[axis_ref] = sharded_pos++;
    }
  }

  if (!unreduced_axes_.empty()) {
    int64_t& unreduced_dim = tile_assignment_dims.emplace_back(1);
    for (const AxisRef& axis_ref : unreduced_axes_) {
      unreduced_dim *= axis_ref.size(mesh_);
      axis_ref_to_sharded_pos[axis_ref] = sharded_pos++;
    }
  }

  // TODO: Add support for manual axes.

  std::vector<AxisRef> mesh_axis_refs = GetOrderedAxisRefs(*this);
  std::vector<int64_t> reshape_dims;
  reshape_dims.reserve(mesh_axis_refs.size());
  std::vector<int> transpose_perm(mesh_axis_refs.size());

  int64_t total_replicated_size = 1;
  int64_t replicated_pos = sharded_pos;
  for (int i = 0; i < mesh_axis_refs.size(); ++i) {
    const AxisRef& axis_ref = mesh_axis_refs[i];
    reshape_dims.push_back(axis_ref.size(mesh_));

    auto sharded_pos_it = axis_ref_to_sharded_pos.find(axis_ref);
    if (sharded_pos_it == axis_ref_to_sharded_pos.end()) {
      transpose_perm[replicated_pos++] = i;
      total_replicated_size *= axis_ref.size(mesh_);
    } else {
      transpose_perm[sharded_pos_it->second] = i;
    }
  }

  if (total_replicated_size > 1) {
    tile_assignment_dims.push_back(total_replicated_size);
  }

  // Simple iota case
  if (mesh_.device_assignment().iota().has_value() &&
      mesh_.device_assignment().iota()->reshape_dims().size() == 1) {
    return TileAssignment(tile_assignment_dims, reshape_dims, transpose_perm);
  }

  return mesh_.device_assignment()
      .Reshape(reshape_dims)
      .Transpose(transpose_perm)
      .Reshape(tile_assignment_dims);
}

std::ostream& operator<<(std::ostream& out,
                         const NamedSharding::DimensionSharding& sharding) {
  return out << sharding.ToString();
}

std::ostream& operator<<(std::ostream& out, const NamedSharding& sharding) {
  return out << sharding.ToString();
}

namespace test_utils {
// Construct sharding with given mesh. 'dim_shardings', 'replicated_axes',
// 'unreduced_axes' refer to axis names in the mesh.
// This is a test only helper function.
NamedSharding FromAxisNames(
    Mesh mesh, absl::Span<const std::vector<std::string>> dim_shardings,
    absl::Span<const std::string> replicated_axes,
    absl::Span<const std::string> unreduced_axes,
    absl::Span<const OpMetadata> metadata) {
  std::map<std::string, int64_t> mesh_axis_to_index;
  for (int64_t i = 0; i < mesh.axis_names().size(); ++i) {
    mesh_axis_to_index[mesh.axis_names()[i]] = i;
  }

  std::vector<NamedSharding::DimensionSharding> dim_shardings_;
  dim_shardings_.reserve(dim_shardings.size());
  for (const auto& axes_for_dim : dim_shardings) {
    std::vector<AxisRef> axis_refs;
    axis_refs.reserve(axes_for_dim.size());
    for (const std::string& axis_name : axes_for_dim) {
      auto it = mesh_axis_to_index.find(axis_name);
      CHECK(it != mesh_axis_to_index.end())
          << "Axis " << axis_name << " not found in mesh " << mesh.ToString();
      axis_refs.push_back(AxisRef(it->second));
    }
    dim_shardings_.push_back(NamedSharding::DimensionSharding(
        std::move(axis_refs), /*is_closed=*/true));
  }

  std::vector<AxisRef> replicated_axes_;
  replicated_axes_.reserve(replicated_axes.size());
  for (const std::string& axis_name : replicated_axes) {
    auto it = mesh_axis_to_index.find(axis_name);
    CHECK(it != mesh_axis_to_index.end())
        << "Axis " << axis_name << " not found in mesh " << mesh.ToString();
    replicated_axes_.push_back(AxisRef(it->second));
  }

  std::vector<AxisRef> unreduced_axes_;
  unreduced_axes_.reserve(unreduced_axes.size());
  for (const std::string& axis_name : unreduced_axes) {
    auto it = mesh_axis_to_index.find(axis_name);
    CHECK(it != mesh_axis_to_index.end())
        << "Axis " << axis_name << " not found in mesh " << mesh.ToString();
    unreduced_axes_.push_back(AxisRef(it->second));
  }

  return NamedSharding(mesh, dim_shardings_, replicated_axes_, unreduced_axes_,
                       metadata);
}
}  // namespace test_utils
}  // namespace xla
