/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/layout_util.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/overflow_util.h"
#include "xla/primitive_util.h"
#include "xla/printer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

// Internal helper for GetDefaultLayoutForShape and SetToDefaultLayout. Sets
// minor_to_major to the value that represents the default layout.
template <typename T>
void SetDefaultLayoutToContainer(T* minor_to_major) {
  // The default XLA layout is major-to-minor (dim 0 is major).
  // For more information on XLA layouts, see:
  // https://www.tensorflow.org/performance/xla/shapes
  const int64_t size = minor_to_major->size();
  for (int64_t i = 0; i < size; ++i) {
    (*minor_to_major)[i] = size - 1 - i;
  }
}

// Internal helper that creates a default layout for an array of the given
// number of dimensions.
Layout CreateDefaultLayoutForRank(int64_t num_dims) {
  Layout layout;
  DimensionVector* minor_to_major = layout.mutable_minor_to_major();
  minor_to_major->resize(num_dims, 0);
  SetDefaultLayoutToContainer(minor_to_major);
  return layout;
}

}  // namespace

/* static */ Layout LayoutUtil::MakeLayout(
    absl::Span<const int64_t> minor_to_major, absl::Span<const Tile> tiles,
    int64_t tail_padding_alignment_in_elements,
    PrimitiveType index_primitive_type, PrimitiveType pointer_primitive_type,
    int64_t element_size_in_bits, int64_t memory_space,
    absl::Span<const SplitConfig> split_configs,
    std::optional<Shape> physical_shape,
    int64_t dynamic_shape_metadata_prefix_bytes,
    absl::Span<const BitPlane> bit_planes) {
  Layout layout;
  layout.mutable_minor_to_major()->assign(minor_to_major.begin(),
                                          minor_to_major.end());
  for (const Tile& tile : tiles) {
    for (int64_t dim : tile.dimensions()) {
      if (dim < 0 && dim != Tile::kCombineDimension &&
          dim != Tile::kMatchDimension) {
        LOG(FATAL) << "Tile dimension size needs to be kCombineDimension or "
                      "kMatchDimension if it's negative. Value is "
                   << dim;
      }
    }
    *layout.add_tiles() = tile;
  }
  layout.set_tail_padding_alignment_in_elements(
      tail_padding_alignment_in_elements);
  layout.set_index_primitive_type(index_primitive_type);
  layout.set_pointer_primitive_type(pointer_primitive_type);
  layout.set_element_size_in_bits(element_size_in_bits);
  layout.set_memory_space(memory_space);
  for (const SplitConfig& split_config : split_configs) {
    layout.add_split_configs(split_config);
  }
  if (physical_shape.has_value()) {
    *layout.mutable_physical_shape() = *std::move(physical_shape);
  }
  layout.set_dynamic_shape_metadata_prefix_bytes(
      dynamic_shape_metadata_prefix_bytes);
  for (const BitPlane& bit_plane : bit_planes) {
    layout.add_bit_planes(bit_plane);
  }
  return layout;
}

/* static */ Layout LayoutUtil::MakeDescendingLayout(int64_t num_dims) {
  return CreateDefaultLayoutForRank(num_dims);
}

/* static */ bool LayoutUtil::HasDescendingLayout(const Layout& layout) {
  return absl::c_is_sorted(layout.minor_to_major(), std::greater<int64_t>());
}

/* static */ Layout LayoutUtil::MakeAscendingLayout(int64_t num_dims) {
  Layout layout;
  DimensionVector* minor_to_major = layout.mutable_minor_to_major();
  minor_to_major->resize(num_dims);
  absl::c_iota(*minor_to_major, 0);
  return layout;
}

/* static */ bool LayoutUtil::HasAscendingLayout(const Layout& layout) {
  return absl::c_is_sorted(layout.minor_to_major(), std::less<int64_t>());
}

/* static */ Layout LayoutUtil::MakeLayoutFromMajorToMinor(
    absl::Span<const int64_t> major_to_minor) {
  Layout layout;
  for (int i = major_to_minor.size() - 1; i >= 0; i--) {
    layout.add_minor_to_major(major_to_minor[i]);
  }
  return layout;
}

/* static */ Layout LayoutUtil::GetDefaultLayoutForShape(const Shape& shape) {
  if (shape.IsOpaque() || shape.IsToken()) {
    // Opaque and token types have empty layouts.
    return Layout();
  }

  // A Layout proto corresponds to a single array, not a tuple.
  CHECK(shape.IsArray());
  return CreateDefaultLayoutForRank(shape.dimensions().size());
}

/* static */ Layout LayoutUtil::GetDefaultLayoutForRank(int64_t num_dims) {
  return CreateDefaultLayoutForRank(num_dims);
}

/* static */ Layout LayoutUtil::GetDefaultLayoutForR2() {
  return CreateDefaultLayoutForRank(2);
}

/* static */ Layout LayoutUtil::GetDefaultLayoutForR3() {
  return CreateDefaultLayoutForRank(3);
}

/* static */ Layout LayoutUtil::GetDefaultLayoutForR4() {
  return CreateDefaultLayoutForRank(4);
}

/* static */ void LayoutUtil::SetToDefaultLayout(Shape* shape) {
  if (shape->IsTuple()) {
    // Tuple shape.
    for (auto& element_shape : *shape->mutable_tuple_shapes()) {
      SetToDefaultLayout(&element_shape);
    }
  } else if (shape->IsArray()) {
    auto* minor_to_major = shape->mutable_layout()->mutable_minor_to_major();
    minor_to_major->resize(shape->dimensions().size(), 0);
    SetDefaultLayoutToContainer(minor_to_major);
  } else {
    // Opaque, token types etc. have no layout.
  }
}

/* static */ Shape LayoutUtil::GetWithDefaultLayout(const Shape& shape) {
  Shape copy(shape);
  LayoutUtil::SetToDefaultLayout(&copy);
  return copy;
}

/* static */ void LayoutUtil::SetToDefaultLayout(ProgramShape* program_shape) {
  for (int i = 0; i < program_shape->parameters_size(); ++i) {
    LayoutUtil::SetToDefaultLayout(program_shape->mutable_parameters(i));
  }
  LayoutUtil::SetToDefaultLayout(program_shape->mutable_result());
}

/* static */ absl::Status LayoutUtil::ValidateLayoutInShape(
    const Shape& shape, bool allow_missing_layouts) {
  if (shape.IsTuple()) {
    for (auto& element_shape : shape.tuple_shapes()) {
      ABSL_RETURN_IF_ERROR(
          ValidateLayoutInShape(element_shape, allow_missing_layouts));
    }
    return absl::OkStatus();
  }
  if (shape.IsArray()) {
    if (!shape.has_layout()) {
      if (allow_missing_layouts) {
        return absl::OkStatus();
      }
      return InvalidArgument("shape %s does not have a layout",
                             ShapeUtil::HumanString(shape));
    }
    return ValidateLayoutForShape(shape.layout(), shape);
  }
  // Token, opaque, etc. shape.
  return absl::OkStatus();
}

namespace {

int64_t CheckedAdd(int64_t a, int64_t b) {
  std::optional<int64_t> sum = OverflowSafeAdd(a, b);
  CHECK(sum.has_value());
  return *sum;
}

int64_t CheckedMultiply(int64_t a, int64_t b) {
  auto [prod, overflow] = OverflowSafeMultiply(a, b);
  CHECK(!overflow);
  return prod;
}

std::optional<int64_t> OverflowSafeRoundUpTo(int64_t value, int64_t divisor) {
  if (value < 0 || divisor <= 0) {
    return std::nullopt;
  }
  const int64_t quotient = value / divisor + (value % divisor != 0 ? 1 : 0);
  auto [prod, overflow] = OverflowSafeMultiply(quotient, divisor);
  if (overflow) {
    return std::nullopt;
  }
  return prod;
}

std::optional<int64_t> OverflowSafeProduct(absl::Span<const int64_t> dims) {
  int64_t prod = 1;
  for (int64_t d : dims) {
    auto [next, overflow] = OverflowSafeMultiply(prod, d);
    if (overflow) {
      return std::nullopt;
    }
    prod = next;
  }
  return prod;
}

absl::Status ValidateAndApplyTopTileForBitPlanes(
    std::vector<int64_t>& current_shape, absl::Span<const Tile> tiles,
    bool forbid_match_dim, const Shape& shape, int64_t* out_inner_rank) {
  const Tile& tile = tiles[0];
  absl::Span<const int64_t> sub_tile_dims =
      tiles.size() > 1 ? tiles[1].dimensions() : absl::Span<const int64_t>();
  absl::StatusOr<LayoutUtil::ResolvedTileStep> step =
      LayoutUtil::ResolveTileStep(current_shape, tile, sub_tile_dims);
  if (!step.ok()) {
    return InvalidArgument("%s: %s", step.status().message(),
                           shape.ToString(true));
  }
  if (forbid_match_dim) {
    for (int64_t dim : tile.dimensions()) {
      if (dim == Tile::kMatchDimension) {
        return InvalidArgument(
            "kMatchDimension ('d') is not allowed in the innermost outer tile "
            "when bit_planes is set: %s",
            shape.ToString(true));
      }
    }
  }
  *out_inner_rank = static_cast<int64_t>(step->effective_tile_dims.size());
  current_shape = std::move(step->next_shape);
  return absl::OkStatus();
}

absl::Status ValidateAndApplyExactSubtile(std::vector<int64_t>& current_shape,
                                          const Tile& subtile,
                                          const Shape& shape) {
  if (subtile.dimensions().empty()) {
    return InvalidArgument("subtile dimensions must not be empty: %s",
                           shape.ToString(true));
  }
  if (subtile.dimensions().size() > current_shape.size()) {
    return InvalidArgument(
        "subtile rank (%d) exceeds expanded shape rank (%d): %s",
        subtile.dimensions().size(), current_shape.size(),
        shape.ToString(true));
  }
  const int64_t tile_rank = subtile.dimensions().size();
  const int64_t suffix_start =
      static_cast<int64_t>(current_shape.size()) - tile_rank;
  for (int64_t i = 0; i < tile_rank; ++i) {
    const int64_t t = subtile.dimension(i);
    if (t <= 0) {
      return InvalidArgument(
          "subtile dimension (%d) must be strictly positive (no '*' or 'd'): "
          "%s",
          t, shape.ToString(true));
    }
    const int64_t d = current_shape[suffix_start + i];
    if (d != Shape::kUnboundedSize && d % t != 0) {
      return InvalidArgument(
          "subtile dimension %d does not evenly divide parent dimension %d: %s",
          t, d, shape.ToString(true));
    }
  }

  std::vector<int64_t> next_shape;
  next_shape.reserve(current_shape.size() + tile_rank);
  for (int64_t i = 0; i < suffix_start; ++i) {
    next_shape.push_back(current_shape[i]);
  }
  for (int64_t i = 0; i < tile_rank; ++i) {
    const int64_t d = current_shape[suffix_start + i];
    const int64_t t = subtile.dimension(i);
    next_shape.push_back(d == Shape::kUnboundedSize ? Shape::kUnboundedSize
                                                    : d / t);
  }
  for (int64_t i = 0; i < tile_rank; ++i) {
    next_shape.push_back(subtile.dimension(i));
  }
  current_shape = std::move(next_shape);
  return absl::OkStatus();
}

absl::Status ValidateBitPlanesForShape(const Layout& layout,
                                       const Shape& shape) {
  if (layout.element_size_in_bits() != 0) {
    return InvalidArgument(
        "layout cannot specify both element_size_in_bits (%d) and bit_planes: "
        "%s",
        layout.element_size_in_bits(), shape.ToString(true));
  }
  if (!layout.split_configs().empty()) {
    return InvalidArgument(
        "layout cannot specify both split_configs and bit_planes: %s",
        shape.ToString(true));
  }
  if (layout.bit_planes().size() < 2) {
    return InvalidArgument(
        "layout bit_planes must contain at least 2 planes (got %d): %s",
        layout.bit_planes().size(), shape.ToString(true));
  }
  if (!primitive_util::IsArrayType(shape.element_type())) {
    return InvalidArgument(
        "layout bit_planes requires an array element type: "
        "%s",
        shape.ToString(true));
  }

  const int64_t elem_bits =
      primitive_util::LogicalBitWidth(shape.element_type());
  std::vector<std::pair<int64_t, int64_t>> intervals;
  intervals.reserve(layout.bit_planes().size());
  int64_t total_bits = 0;
  for (const BitPlane& plane : layout.bit_planes()) {
    if (plane.bit_offset() < 0) {
      return InvalidArgument("bit_plane bit_offset must be >= 0 (got %d): %s",
                             plane.bit_offset(), shape.ToString(true));
    }
    if (plane.bit_width() <= 0) {
      return InvalidArgument("bit_plane bit_width must be > 0 (got %d): %s",
                             plane.bit_width(), shape.ToString(true));
    }
    if (plane.bit_offset() >
        std::numeric_limits<int64_t>::max() - plane.bit_width()) {
      return InvalidArgument("bit_plane bit range overflows: %s",
                             shape.ToString(true));
    }
    std::optional<int64_t> next_total_bits =
        OverflowSafeAdd<int64_t>(total_bits, plane.bit_width());
    if (!next_total_bits.has_value()) {
      return InvalidArgument("bit_planes total bit_width overflows: %s",
                             shape.ToString(true));
    }
    total_bits = *next_total_bits;
    intervals.emplace_back(plane.bit_offset(),
                           plane.bit_offset() + plane.bit_width());
  }
  absl::c_sort(intervals);
  for (size_t i = 1; i < intervals.size(); ++i) {
    if (intervals[i - 1].second > intervals[i].first) {
      return InvalidArgument(
          "layout bit_planes have overlapping bit ranges [%d, %d) and [%d, "
          "%d): %s",
          intervals[i - 1].first, intervals[i - 1].second, intervals[i].first,
          intervals[i].second, shape.ToString(true));
    }
  }
  if (intervals[0].first != 0) {
    return InvalidArgument(
        "layout bit_planes must cover [0, %d), starting at bit 0 (got %d): %s",
        elem_bits, intervals[0].first, shape.ToString(true));
  }
  for (size_t i = 1; i < intervals.size(); ++i) {
    if (intervals[i].first != intervals[i - 1].second) {
      return InvalidArgument(
          "layout bit_planes must be contiguous from bit 0 (gap between [%d, "
          "%d) and [%d, %d)): %s",
          intervals[i - 1].first, intervals[i - 1].second, intervals[i].first,
          intervals[i].second, shape.ToString(true));
    }
  }
  if (intervals.back().second < elem_bits) {
    return InvalidArgument(
        "layout bit_planes must cover all %d bits of element type (covered up "
        "to %d): %s",
        elem_bits, intervals.back().second, shape.ToString(true));
  }

  Shape shape_with_layout = shape;
  *shape_with_layout.mutable_layout() = layout;
  const std::vector<int64_t> phys_shape =
      LayoutUtil::PhysicalDimensions(shape_with_layout);

  const bool is_static_non_negative =
      absl::c_all_of(phys_shape, [](int64_t d) { return d >= 0; });

  if (!layout.tiles().empty()) {
    std::vector<int64_t> expanded_shape = phys_shape;
    int64_t innermost_rank = 0;
    ABSL_RETURN_IF_ERROR(ValidateAndApplyTopTileForBitPlanes(
        expanded_shape, layout.tiles(),
        /*forbid_match_dim=*/layout.tiles().size() == 1, shape,
        &innermost_rank));
    if (is_static_non_negative) {
      std::optional<int64_t> padded_elems = OverflowSafeProduct(expanded_shape);
      if (padded_elems.has_value() &&
          layout.tail_padding_alignment_in_elements() != 1) {
        padded_elems = OverflowSafeRoundUpTo(
            *padded_elems, layout.tail_padding_alignment_in_elements());
      }
      if (!padded_elems.has_value()) {
        return InvalidArgument("bit_planes padded element count overflows: %s",
                               shape.ToString(true));
      }
      auto [buf_bits, bits_overflow] =
          OverflowSafeMultiply(*padded_elems, total_bits);
      if (bits_overflow) {
        return InvalidArgument("bit_planes total bit count overflows: %s",
                               shape.ToString(true));
      }
    }
    for (size_t s = 1; s < layout.tiles().size(); ++s) {
      ABSL_RETURN_IF_ERROR(
          ValidateAndApplyExactSubtile(expanded_shape, layout.tiles(s), shape));
      innermost_rank = layout.tiles(s).dimensions().size();
    }
    std::vector<int64_t> strip_dims(expanded_shape.end() - innermost_rank,
                                    expanded_shape.end());
    int64_t strip_elements = 1;
    for (int64_t d : strip_dims) {
      auto [prod, overflow] = OverflowSafeMultiply(strip_elements, d);
      if (overflow || prod <= 0) {
        return InvalidArgument("strip element count overflows: %s",
                               shape.ToString(true));
      }
      strip_elements = prod;
    }
    if (layout.tail_padding_alignment_in_elements() != 1 &&
        layout.tail_padding_alignment_in_elements() % strip_elements != 0) {
      return InvalidArgument(
          "layout tail_padding_alignment_in_elements (%d) must be a multiple "
          "of the innermost tile element count (%d) when bit_planes is set: %s",
          layout.tail_padding_alignment_in_elements(), strip_elements,
          shape.ToString(true));
    }
    for (const BitPlane& plane : layout.bit_planes()) {
      std::vector<int64_t> plane_shape = strip_dims;
      for (const Tile& ptile : plane.tiles()) {
        ABSL_RETURN_IF_ERROR(
            ValidateAndApplyExactSubtile(plane_shape, ptile, shape));
      }
    }
  } else {
    int64_t total_buf_bits = 0;
    for (const BitPlane& plane : layout.bit_planes()) {
      std::vector<int64_t> plane_shape = phys_shape;
      if (!plane.tiles().empty()) {
        int64_t unused_rank = 0;
        ABSL_RETURN_IF_ERROR(ValidateAndApplyTopTileForBitPlanes(
            plane_shape, plane.tiles(), /*forbid_match_dim=*/false, shape,
            &unused_rank));
        for (size_t s = 1; s < plane.tiles().size(); ++s) {
          ABSL_RETURN_IF_ERROR(
              ValidateAndApplyExactSubtile(plane_shape, plane.tiles(s), shape));
        }
      }
      if (is_static_non_negative) {
        std::optional<int64_t> plane_elems = OverflowSafeProduct(plane_shape);
        if (plane_elems.has_value() &&
            layout.tail_padding_alignment_in_elements() != 1) {
          plane_elems = OverflowSafeRoundUpTo(
              *plane_elems, layout.tail_padding_alignment_in_elements());
        }
        if (!plane_elems.has_value()) {
          return InvalidArgument(
              "bit_planes padded element count overflows: %s",
              shape.ToString(true));
        }
        auto [plane_bits, plane_overflow] =
            OverflowSafeMultiply(*plane_elems, plane.bit_width());
        if (plane_overflow) {
          return InvalidArgument("bit_planes total bit count overflows: %s",
                                 shape.ToString(true));
        }
        std::optional<int64_t> next_buf_bits =
            OverflowSafeAdd(total_buf_bits, plane_bits);
        if (!next_buf_bits.has_value()) {
          return InvalidArgument("bit_planes total bit count overflows: %s",
                                 shape.ToString(true));
        }
        total_buf_bits = *next_buf_bits;
      }
    }
  }
  return absl::OkStatus();
}

void ApplyTileSequenceToCoordinates(absl::Span<const Tile> tiles,
                                    std::vector<int64_t>& current_shape,
                                    std::vector<int64_t>& current_indices,
                                    int64_t* out_last_tile_rank) {
  int64_t last_rank = 0;
  for (size_t step_idx = 0; step_idx < tiles.size(); ++step_idx) {
    const Tile& tile = tiles[step_idx];
    absl::Span<const int64_t> sub_tile_dims =
        (step_idx == 0 && tiles.size() > 1) ? tiles[1].dimensions()
                                            : absl::Span<const int64_t>();
    absl::StatusOr<LayoutUtil::ResolvedTileStep> step =
        LayoutUtil::ResolveTileStep(current_shape, tile, sub_tile_dims);
    CHECK_OK(step.status());
    if (step->leading_pad_dims > 0) {
      current_indices.insert(current_indices.begin(), step->leading_pad_dims,
                             0);
    }

    std::vector<int64_t> suffix_dims = step->padded_suffix;
    std::vector<int64_t> suffix_idx(
        current_indices.begin() + step->suffix_start, current_indices.end());
    std::vector<int64_t> combined_idx;
    combined_idx.reserve(step->effective_tile_dims.size());
    for (size_t i = 0; i < tile.dimensions().size(); ++i) {
      if (tile.dimension(i) == Tile::kCombineDimension) {
        CHECK_LT(i + 1, tile.dimensions().size());
        suffix_idx[i + 1] =
            CheckedAdd(CheckedMultiply(suffix_idx[i], suffix_dims[i + 1]),
                       suffix_idx[i + 1]);
        suffix_dims[i + 1] =
            CheckedMultiply(suffix_dims[i], suffix_dims[i + 1]);
      } else {
        combined_idx.push_back(suffix_idx[i]);
      }
    }

    std::vector<int64_t> next_indices(
        current_indices.begin(), current_indices.begin() + step->suffix_start);
    next_indices.reserve(step->next_shape.size());
    for (size_t j = 0; j < step->effective_tile_dims.size(); ++j) {
      next_indices.push_back(combined_idx[j] / step->effective_tile_dims[j]);
    }
    for (size_t j = 0; j < step->effective_tile_dims.size(); ++j) {
      next_indices.push_back(combined_idx[j] % step->effective_tile_dims[j]);
    }
    last_rank = static_cast<int64_t>(step->effective_tile_dims.size());
    current_shape = std::move(step->next_shape);
    current_indices = std::move(next_indices);
  }
  if (out_last_tile_rank != nullptr) {
    *out_last_tile_rank = last_rank;
  }
}

int64_t RowMajorLinearIndex(absl::Span<const int64_t> shape_dims,
                            absl::Span<const int64_t> indices) {
  CHECK_EQ(shape_dims.size(), indices.size());
  int64_t linear_index = 0;
  int64_t multiplier = 1;
  for (int64_t i = static_cast<int64_t>(shape_dims.size()) - 1; i >= 0; --i) {
    if (indices[i] != 0) {
      linear_index =
          CheckedAdd(linear_index, CheckedMultiply(indices[i], multiplier));
    }
    if (i > 0) {
      multiplier = CheckedMultiply(multiplier, shape_dims[i]);
    }
  }
  return linear_index;
}

}  // namespace

/* static */ absl::Status LayoutUtil::ValidateLayoutForShape(
    const Layout& layout, const Shape& shape) {
  if (shape.IsTuple()) {
    return InvalidArgument("a single Layout is not valid for tuple shapes");
  }

  if (!shape.IsArray()) {
    return absl::OkStatus();
  }

  if (layout.minor_to_major().empty() && layout.memory_space() != 0 &&
      layout.bit_planes().empty()) {
    // AUTO layout with non-default memory space is allowed.
    return absl::OkStatus();
  }

  if (layout.minor_to_major().size() != shape.dimensions().size()) {
    return InvalidArgument(
        "layout minor_to_major field contains %d elements, "
        "but shape has %d dimensions: {%s}; shape: %s",
        layout.minor_to_major().size(), shape.dimensions().size(),
        absl::StrJoin(layout.minor_to_major(), ", "), shape.ToString());
  }

  absl::InlinedVector<bool, InlineRank()> dimensions_in_layout(
      shape.dimensions().size(), false);
  for (int64_t i = 0; i < shape.dimensions().size(); ++i) {
    int64_t dim = layout.minor_to_major(i);
    if (dim < 0 || dim >= shape.dimensions().size()) {
      return InvalidArgument(
          "layout minor_to_major field has out-of-bounds value: {%s}; shape: "
          "%s",
          absl::StrJoin(layout.minor_to_major(), ", "), shape.ToString());
    }
    if (dimensions_in_layout[dim]) {
      return InvalidArgument(
          "layout minor_to_major field has duplicate values: {%s}; shape: %s",
          absl::StrJoin(layout.minor_to_major(), ", "), shape.ToString());
    }
    dimensions_in_layout[dim] = true;
  }

  if (layout.tail_padding_alignment_in_elements() <= 0) {
    return InvalidArgument(
        "layout tail_padding_alignment_in_elements field is <= 0: {%d}",
        layout.tail_padding_alignment_in_elements());
  }

  if (layout.index_primitive_type() != PRIMITIVE_TYPE_INVALID) {
    return InvalidArgument(
        "layout has a index_primitive_type, but is not a sparse array: %s",
        shape.ToString());
  }
  if (layout.pointer_primitive_type() != PRIMITIVE_TYPE_INVALID) {
    return InvalidArgument(
        "layout has a pointer_primitive_type, but is not a sparse array: %s",
        shape.ToString());
  }
  if (layout.has_physical_shape()) {
    return InvalidArgument(
        "layout has a physical_shape, but is not a sparse array: %s",
        shape.ToString());
  }
  for (const auto& tile : layout.tiles()) {
    if (tile.dimensions().empty()) {
      return InvalidArgument("layout has invalid tiles: %s", shape.ToString());
    }
    for (int64_t dim : tile.dimensions()) {
      if (dim <= 0 && dim != Tile::kCombineDimension &&
          dim != Tile::kMatchDimension) {
        return InvalidArgument(
            "layout has invalid tiles: tile dimension %d must be positive, "
            "kCombineDimension, or kMatchDimension: %s",
            dim, shape.ToString());
      }
    }
  }

  if (layout.element_size_in_bits() < 0) {
    return InvalidArgument("layout element_size_in_bits field is negative: %d",
                           layout.element_size_in_bits());
  }

  // Validate split configs.
  absl::InlinedVector<bool, InlineRank()> physical_dims_in_split_config(
      shape.dimensions().size(), false);
  for (const SplitConfig& config : layout.split_configs()) {
    int64_t physical_dim = config.dimension();
    if (physical_dim < 0 || physical_dim >= shape.dimensions().size()) {
      return InvalidArgument(
          "split config has out-of-bounds physical dimension: %d; shape: %s",
          physical_dim, shape.ToString());
    }
    if (physical_dims_in_split_config[physical_dim]) {
      return InvalidArgument(
          "split config has duplicate physical dimension: %d; shape: %s",
          physical_dim, shape.ToString());
    }
    physical_dims_in_split_config[physical_dim] = true;

    int64_t logical_dim = Major(layout, physical_dim);
    int64_t dim_size = shape.dimensions(logical_dim);

    int64_t last_split_index = 0;
    for (int64_t split_index : config.split_indices()) {
      if (split_index <= last_split_index) {
        return InvalidArgument(
            "split config split indices must be strictly increasing and "
            "positive: {%s}; shape: %s",
            absl::StrJoin(config.split_indices(), ", "), shape.ToString());
      }
      if (split_index >= dim_size) {
        return InvalidArgument(
            "split config split index is out of bounds: %d (dim size %d); "
            "shape: %s",
            split_index, dim_size, shape.ToString());
      }
      last_split_index = split_index;
    }
  }

  if (!layout.bit_planes().empty()) {
    ABSL_RETURN_IF_ERROR(ValidateBitPlanesForShape(layout, shape));
  }

  return absl::OkStatus();
}

/* static */ void LayoutUtil::ClearLayout(Shape* shape) {
  if (shape->IsArray()) {
    shape->clear_layout();
  } else if (shape->IsTuple()) {
    for (auto& element_shape : *shape->mutable_tuple_shapes()) {
      ClearLayout(&element_shape);
    }
  }
}

/* static */ void LayoutUtil::ClearLayout(ProgramShape* program_shape) {
  for (int i = 0; i < program_shape->parameters_size(); ++i) {
    LayoutUtil::ClearLayout(program_shape->mutable_parameters(i));
  }
  LayoutUtil::ClearLayout(program_shape->mutable_result());
}

/* static */ void LayoutUtil::ClearTiles(Shape* shape) {
  ShapeUtil::ForEachMutableSubshape(
      shape, [](Shape* subshape, const ShapeIndex&) {
        if (subshape->has_layout()) {
          subshape->mutable_layout()->clear_tiles();
          subshape->mutable_layout()->clear_bit_planes();
        }
      });
}

/* static */ bool LayoutUtil::IsMonotonicWithDim0Minor(const Layout& layout) {
  return std::is_sorted(layout.minor_to_major().begin(),
                        layout.minor_to_major().end());
}

/* static */ bool LayoutUtil::IsMonotonicWithDim0Major(const Layout& layout) {
  return std::is_sorted(layout.minor_to_major().begin(),
                        layout.minor_to_major().end(), std::greater<int64_t>());
}

/* static */ bool LayoutUtil::HasLayout(const Shape& shape) {
  if (shape.IsTuple()) {
    // Tuple shape: all subshapes must have a layout.
    return absl::c_all_of(shape.tuple_shapes(),
                          [](const Shape& s) { return HasLayout(s); });
  }
  if (!shape.IsArray()) {
    // Opaque, token types etc. ignore layout.
    return true;
  }
  return shape.has_layout();
}

/* static */ bool LayoutUtil::HasAnyLayout(const Shape& shape) {
  if (shape.IsTuple()) {
    // Tuple shape: all subshapes must have a layout.
    return absl::c_any_of(shape.tuple_shapes(),
                          [](const Shape& s) { return HasAnyLayout(s); });
  }
  if (!shape.IsArray()) {
    // Opaque, token types etc. ignore layout.
    return true;
  }
  return shape.has_layout();
}

/* static */ bool LayoutUtil::HasMinorToMajorSetInLayout(const Shape& shape) {
  if (shape.IsTuple()) {
    return absl::c_all_of(shape.tuple_shapes(), [](const Shape& s) {
      return HasMinorToMajorSetInLayout(s);
    });
  }
  if (!shape.IsArray()) {
    return true;
  }
  return shape.has_layout() && (shape.dimensions().empty() ||
                                !shape.layout().minor_to_major().empty());
}

/* static */ bool LayoutUtil::HasAnyMinorToMajorSetInLayout(
    const Shape& shape) {
  if (shape.IsTuple()) {
    return absl::c_any_of(shape.tuple_shapes(), [](const Shape& s) {
      return HasAnyMinorToMajorSetInLayout(s);
    });
  }
  if (!shape.IsArray()) {
    return true;
  }
  return shape.has_layout() && (shape.dimensions().empty() ||
                                !shape.layout().minor_to_major().empty());
}

/* static */ bool LayoutUtil::HasLayout(const ProgramShape& program_shape) {
  for (auto& parameter_shape : program_shape.parameters()) {
    if (!LayoutUtil::HasLayout(parameter_shape)) {
      return false;
    }
  }
  return LayoutUtil::HasLayout(program_shape.result());
}

/* static */ bool LayoutUtil::HasCustomElementSizeInBits(const Shape& shape) {
  if (shape.IsTuple()) {
    return absl::c_any_of(shape.tuple_shapes(),
                          LayoutUtil::HasCustomElementSizeInBits);
  }
  if (!shape.IsArray()) {
    // Opaque or token types have no custom element size in bits.
    return false;
  }
  return shape.has_layout() && shape.layout().element_size_in_bits() != 0;
}

/* static */ bool LayoutUtil::Equal(const Layout& lhs, const Layout& rhs) {
  return lhs == rhs;
}

/* static */ std::vector<int64_t> LayoutUtil::MakeLogicalToPhysical(
    const Layout& layout) {
  std::vector<int64_t> logical_to_physical(layout.minor_to_major().size());
  for (int64_t physical = 0, end = logical_to_physical.size(); physical < end;
       ++physical) {
    const int64_t logical = Major(layout, physical);
    logical_to_physical[logical] = physical;
  }
  return logical_to_physical;
}

/* static */ void LayoutUtil::PrintHumanString(Printer* printer,
                                               const Layout& layout) {
  layout.Print(printer);
}

/* static */ std::string LayoutUtil::HumanString(const Layout& layout) {
  return layout.ToString();
}

namespace {

// Internal helper for recursively copying layouts.
absl::Status CopyLayoutInternal(const Shape& src, Shape* dst) {
  if (src.IsTuple() != dst->IsTuple()) {
    return InvalidArgument(
        "cannot copy layout from shape: shape structure differs");
  }
  if (src.IsTuple()) {
    if (ShapeUtil::TupleElementCount(src) !=
        ShapeUtil::TupleElementCount(*dst)) {
      return InvalidArgument(
          "cannot copy layout from shape: tuple element count differs");
    }
    for (int64_t i = 0; i < ShapeUtil::TupleElementCount(src); ++i) {
      ABSL_RETURN_IF_ERROR(CopyLayoutInternal(src.tuple_shapes(i),
                                              dst->mutable_tuple_shapes(i)));
    }
  } else if (src.IsArray()) {
    if (src.has_layout()) {
      if (src.dimensions().size() != dst->dimensions().size()) {
        return InvalidArgument("cannot copy layout from shape: ranks differs");
      }
      ABSL_RETURN_IF_ERROR(
          LayoutUtil::ValidateLayoutForShape(src.layout(), *dst));
      *dst->mutable_layout() = src.layout();
    } else {
      dst->clear_layout();
    }
  }
  return absl::OkStatus();
}

}  // namespace

/* static */
absl::Status LayoutUtil::CopyLayoutBetweenShapes(const Shape& src, Shape* dst) {
  return CopyLayoutInternal(src, dst);
}

/* static */ bool LayoutUtil::LayoutsInShapesEqual(
    const Shape& lhs, const Shape& rhs, std::optional<Layout::Equal> equal) {
  if (lhs.IsTuple()) {
    if (!rhs.IsTuple() || ShapeUtil::TupleElementCount(lhs) !=
                              ShapeUtil::TupleElementCount(rhs)) {
      return false;
    }
    for (int i = 0; i < ShapeUtil::TupleElementCount(lhs); ++i) {
      if (!LayoutsInShapesEqual(lhs.tuple_shapes(i), rhs.tuple_shapes(i))) {
        return false;
      }
    }
    return true;
  }
  if (lhs.IsArray()) {
    if (lhs.dimensions().size() != rhs.dimensions().size()) {
      return false;
    }
    if (!lhs.has_layout() && !rhs.has_layout()) {
      return true;
    }
    if (!lhs.has_layout() || !rhs.has_layout()) {
      return false;
    }

    if (equal.has_value()) {
      return equal.value()(lhs.layout(), rhs.layout());
    }

    return LayoutUtil::Equal(lhs.layout(), rhs.layout());
  }
  // Layouts of non-array and non-tuple shapes is ignored.
  return true;
}

/* static */ bool LayoutUtil::AreDimensionsConsecutive(
    const Layout& layout, absl::Span<const int64_t> dims) {
  absl::InlinedVector<int64_t, 8> positions_in_layout;
  for (int64_t dim : dims) {
    positions_in_layout.push_back(
        PositionInContainer(layout.minor_to_major(), dim));
  }
  absl::c_sort(positions_in_layout);
  for (size_t i = 1; i < positions_in_layout.size(); ++i) {
    if (1 != positions_in_layout[i] - positions_in_layout[i - 1]) {
      return false;
    }
  }
  return true;
}

/*static*/ Layout LayoutUtil::MoveDimToMajor(const Layout& layout,
                                             int64_t dim) {
  if (dim == MinorToMajor(layout).back()) {
    return layout;
  }
  Layout ret = layout;
  ret.clear_minor_to_major();
  for (auto d : MinorToMajor(layout)) {
    if (d != dim) {
      ret.add_minor_to_major(d);
    }
  }
  ret.add_minor_to_major(dim);
  return ret;
}

Layout LayoutUtil::MoveDimToMinor(const Layout& layout, const int64_t dim) {
  if (dim == MinorToMajor(layout).front()) {
    return layout;
  }
  Layout result = layout;
  result.clear_minor_to_major();
  result.add_minor_to_major(dim);
  for (int64_t current_dim : MinorToMajor(layout)) {
    if (current_dim != dim) {
      result.add_minor_to_major(current_dim);
    }
  }
  return result;
}

/*static*/ int64_t LayoutUtil::LinearIndex(const Shape& shape,
                                           absl::Span<const int64_t> indices) {
  CHECK(shape.IsArray());
  CHECK(shape.has_layout());
  CHECK(shape.layout().bit_planes().empty())
      << "Split bit-plane layouts do not have a single element linear index; "
         "use BitOffsetForBitPlane instead.";
  const int num_dims = shape.dimensions().size();
  CHECK_EQ(num_dims, indices.size());

  if (num_dims == 0) {
    return 0;
  }
  if (num_dims == 1) {
    return indices[0];
  }

  Tile tile = {};
  if (!shape.layout().tiles().empty()) {
    tile = shape.layout().tiles()[0];
  }

  auto resolve_match_dim = [&](int64_t i, int64_t dim_size) {
    absl::Span<const int64_t> tile_dims = tile.dimensions();
    absl::Span<const int64_t> sub_tile_dims =
        shape.layout().tiles().size() > 1 ? shape.layout().tiles(1).dimensions()
                                          : absl::Span<const int64_t>();
    int64_t sub_tile_offset = static_cast<int64_t>(sub_tile_dims.size()) -
                              static_cast<int64_t>(tile_dims.size());
    int64_t packing = 1;
    if (i + sub_tile_offset >= 0 &&
        static_cast<int64_t>(sub_tile_dims.size()) > i + sub_tile_offset) {
      packing = std::max<int64_t>(1, sub_tile_dims[i + sub_tile_offset]);
    } else if (i == static_cast<int64_t>(tile_dims.size()) - 2 &&
               !sub_tile_dims.empty()) {
      packing = std::max<int64_t>(1, sub_tile_dims[0]);
    }
    return RoundUpTo(std::max<int64_t>(dim_size, 1), packing);
  };

  int64_t linear_index = 0;
  int64_t tile_multiplier = 1;
  // Initialize to number of elements in a tile.
  for (int64_t d_idx = 0; d_idx < tile.dimensions().size(); ++d_idx) {
    int64_t tile_dim_size = tile.dimensions()[d_idx];
    if (tile_dim_size == Tile::kMatchDimension) {
      int64_t minor = tile.dimensions().size() - 1 - d_idx;
      if (minor >= shape.layout().minor_to_major().size()) {
        continue;
      }
      int64_t logical_dim = Minor(shape.layout(), minor);
      tile_dim_size = resolve_match_dim(d_idx, shape.dimensions(logical_dim));
    }
    tile_multiplier *= tile_dim_size;
  }
  int64_t within_tile_multiplier = 1;

  // We only look at the top-level tile.
  for (int64_t minor = 0; minor < num_dims; minor++) {
    int64_t logical_dim = Minor(shape.layout(), minor);
    int64_t shape_dim_size = shape.dimensions(logical_dim);
    int64_t index = indices[logical_dim];

    if (minor < tile.dimensions().size()) {
      int64_t d_idx = tile.dimensions().size() - 1 - minor;
      int64_t tile_dim_size = tile.dimensions()[d_idx];
      if (tile_dim_size == Tile::kMatchDimension) {
        tile_dim_size = resolve_match_dim(d_idx, shape_dim_size);
      }
      linear_index += tile_multiplier * (index / tile_dim_size) +
                      within_tile_multiplier * (index % tile_dim_size);
      tile_multiplier *= CeilOfRatio(shape_dim_size, tile_dim_size);
      within_tile_multiplier *= tile_dim_size;
    } else {
      linear_index += index * tile_multiplier;
      tile_multiplier *= shape_dim_size;
    }
  }
  return linear_index;
}

// See https://openxla.org/xla/tiled_layout for details on the layout and
// tiling.
/*static*/ int64_t LayoutUtil::LinearIndexForNestedTiling(
    const Shape& shape, absl::Span<const int64_t> indices) {
  CHECK(shape.IsArray());
  CHECK(shape.has_layout());
  CHECK(shape.layout().bit_planes().empty())
      << "Split bit-plane layouts do not have a single element linear index; "
         "use BitOffsetForBitPlane instead.";
  const int num_dims = shape.dimensions().size();
  CHECK_EQ(num_dims, indices.size());

  if (num_dims == 0) {
    return 0;
  }

  // 1. Initialize physical dimensions and indices (major-to-minor).
  // The layout.minor_to_major(0) is the most minor physical dimension.
  // We construct vectors in major-to-minor order to simplify tiling and
  // linearization.
  std::vector<int64_t> current_shape;
  std::vector<int64_t> current_indices;
  current_shape.reserve(num_dims);
  current_indices.reserve(num_dims);
  for (int64_t i = num_dims - 1; i >= 0; --i) {
    int64_t logical_dim = shape.layout().minor_to_major(i);
    current_shape.push_back(shape.dimensions(logical_dim));
    current_indices.push_back(indices[logical_dim]);
  }

  // 2. Iteratively apply each tile level.
  for (const Tile& tile : shape.layout().tiles()) {
    const int64_t tile_rank = tile.dimensions().size();
    if (tile_rank > current_shape.size()) {
      int64_t pad_size = tile_rank - current_shape.size();
      current_shape.insert(current_shape.begin(), pad_size, 1);
      current_indices.insert(current_indices.begin(), pad_size, 0);
    }
    // Tiling applies to a suffix of the current physical dimensions.
    CHECK_LE(tile_rank, current_shape.size());

    const int64_t suffix_start = current_shape.size() - tile_rank;
    std::vector<int64_t> next_shape;
    std::vector<int64_t> next_indices;
    next_shape.reserve(current_shape.size() + tile_rank);
    next_indices.reserve(current_indices.size() + tile_rank);

    // Prefix dimensions remain unchanged.
    for (int i = 0; i < suffix_start; ++i) {
      next_shape.push_back(current_shape[i]);
      next_indices.push_back(current_indices[i]);
    }

    // Outer tiles dimensions: ceil(d/t)
    // Outer tiles indices: floor(e/t).
    for (int i = 0; i < tile_rank; ++i) {
      int64_t d = current_shape[suffix_start + i];
      int64_t e = current_indices[suffix_start + i];
      int64_t t = tile.dimension(i);
      if (t == Tile::kMatchDimension) {
        t = d == 0 ? 1 : d;
      }
      next_shape.push_back(CeilOfRatio(d, t));
      next_indices.push_back(e / t);
    }

    // Inner tile dimensions: t.
    // Inner tile indices: e mod t.
    for (int i = 0; i < tile_rank; ++i) {
      int64_t d = current_shape[suffix_start + i];
      int64_t e = current_indices[suffix_start + i];
      int64_t t = tile.dimension(i);
      if (t == Tile::kMatchDimension) {
        t = d == 0 ? 1 : d;
      }
      next_shape.push_back(t);
      next_indices.push_back(e % t);
    }

    current_shape = std::move(next_shape);
    current_indices = std::move(next_indices);
  }

  // 3. Final linearization in the expanded row-major (major-to-minor) space.
  int64_t linear_index = 0;
  int64_t multiplier = 1;
  for (int64_t i = current_shape.size() - 1; i >= 0; --i) {
    linear_index += current_indices[i] * multiplier;
    multiplier *= current_shape[i];
  }
  return linear_index;
}

// See https://openxla.org/xla/tiled_layout for details on the layout and
// tiling.
/*static*/ std::vector<int64_t> LayoutUtil::DelinearizeIndexForNestedTiling(
    const Shape& shape, int64_t linear_index) {
  CHECK(shape.IsArray());
  CHECK(shape.has_layout());
  CHECK(shape.layout().bit_planes().empty())
      << "Split bit-plane layouts do not have a single element linear index.";
  const int num_dims = shape.dimensions().size();

  if (num_dims == 0) {
    return {};
  }
  if (ShapeUtil::IsZeroElementArray(shape)) {
    return std::vector<int64_t>(num_dims, 0);
  }

  // 1. Determine the final expanded physical shape (major-to-minor).
  // We simulate the tiling process to find the high-dimensional shape
  // that linear_index maps into.
  std::vector<int64_t> current_shape;
  current_shape.reserve(num_dims);
  for (int64_t i = num_dims - 1; i >= 0; --i) {
    int64_t logical_dim = shape.layout().minor_to_major(i);
    current_shape.push_back(shape.dimensions(logical_dim));
  }

  // Structure to store the shape state before each tiling level.
  struct TilingStep {
    std::vector<int64_t> shape_before;
    Tile tile;
  };
  std::vector<TilingStep> steps;

  for (const Tile& tile : shape.layout().tiles()) {
    const int64_t tile_rank = tile.dimensions().size();
    if (tile_rank > current_shape.size()) {
      int64_t pad_size = tile_rank - current_shape.size();
      current_shape.insert(current_shape.begin(), pad_size, 1);
    }
    steps.push_back({current_shape, tile});

    const int64_t suffix_start = current_shape.size() - tile_rank;

    std::vector<int64_t> next_shape;
    next_shape.reserve(current_shape.size() + tile_rank);
    // Prefix dimensions remain unchanged.
    for (int i = 0; i < suffix_start; ++i) {
      next_shape.push_back(current_shape[i]);
    }
    // Outer tile dimensions: ceil(d/t).
    for (int i = 0; i < tile_rank; ++i) {
      int64_t d = current_shape[suffix_start + i];
      int64_t t = tile.dimension(i);
      if (t == Tile::kMatchDimension) {
        t = d == 0 ? 1 : d;
      }
      next_shape.push_back(CeilOfRatio(d, t));
    }
    // Inner tile dimensions: t.
    for (int i = 0; i < tile_rank; ++i) {
      int64_t d = current_shape[suffix_start + i];
      int64_t t = tile.dimension(i);
      if (t == Tile::kMatchDimension) {
        t = d == 0 ? 1 : d;
      }
      next_shape.push_back(t);
    }
    current_shape = std::move(next_shape);
  }

  // 2. Delinearize linear_index into indices for the final expanded shape.
  // We use standard row-major delinearization.
  std::vector<int64_t> current_indices(current_shape.size());
  int64_t remaining_index = linear_index;
  for (int i = current_shape.size() - 1; i >= 0; --i) {
    current_indices[i] = remaining_index % current_shape[i];
    remaining_index /= current_shape[i];
  }

  // 3. Reverse the tiling steps to collapse indices.
  // For dimension size d and tile size t,
  // Each tiling step splits d into ceil(d/t) (outer) and t (inner).
  // The original index is e = outer_index * t + inner_index.
  for (int s = steps.size() - 1; s >= 0; --s) {
    const auto& step = steps[s];
    const int64_t tile_rank = step.tile.dimensions().size();
    const int64_t suffix_start = step.shape_before.size() - tile_rank;

    std::vector<int64_t> prev_indices;
    prev_indices.reserve(step.shape_before.size());
    // Prefix remains unchanged.
    for (int i = 0; i < suffix_start; ++i) {
      prev_indices.push_back(current_indices[i]);
    }
    // Combine outer and inner indices.
    // The current_indices layout is [prefix, outer_0...k-1, inner_0...k-1].
    for (int i = 0; i < tile_rank; ++i) {
      int64_t outer_idx = current_indices[suffix_start + i];
      int64_t inner_idx = current_indices[suffix_start + tile_rank + i];
      int64_t tile_dim = step.tile.dimension(i);
      if (tile_dim == Tile::kMatchDimension) {
        int64_t d = step.shape_before[suffix_start + i];
        tile_dim = d == 0 ? 1 : d;
      }
      prev_indices.push_back(outer_idx * tile_dim + inner_idx);
    }
    current_indices = std::move(prev_indices);
  }

  // 4. Map the physical major-to-minor indices back to logical dimensions.
  std::vector<int64_t> logical_indices(num_dims);
  int64_t pad_offset = current_indices.size() - num_dims;
  for (int i = 0; i < num_dims; ++i) {
    // The physical order was minor_to_major(num_dims-1) down to 0.
    int64_t logical_dim = shape.layout().minor_to_major(num_dims - 1 - i);
    logical_indices[logical_dim] = current_indices[pad_offset + i];
  }

  return logical_indices;
}

/*static*/ int64_t LayoutUtil::BitOffsetForBitPlane(
    const Shape& shape, absl::Span<const int64_t> indices,
    int64_t plane_index) {
  CHECK(shape.IsArray());
  CHECK(shape.has_layout());
  const Layout& layout = shape.layout();
  CHECK(!layout.bit_planes().empty());
  CHECK(layout.split_configs().empty());
  CHECK_GE(plane_index, 0);
  CHECK_LT(plane_index, static_cast<int64_t>(layout.bit_planes().size()));
  const int num_dims = shape.dimensions().size();
  CHECK_EQ(num_dims, indices.size());
  for (int d = 0; d < num_dims; ++d) {
    CHECK_GE(indices[d], 0);
    CHECK_LT(indices[d], shape.dimensions(d));
  }

  const std::vector<int64_t> phys_shape = PhysicalDimensions(shape);
  std::vector<int64_t> phys_indices;
  phys_indices.reserve(num_dims);
  for (int64_t i = num_dims - 1; i >= 0; --i) {
    phys_indices.push_back(indices[layout.minor_to_major(i)]);
  }

  if (!layout.tiles().empty()) {
    std::vector<int64_t> expanded_shape = phys_shape;
    std::vector<int64_t> expanded_indices = phys_indices;
    int64_t innermost_rank = 0;
    ApplyTileSequenceToCoordinates(layout.tiles(), expanded_shape,
                                   expanded_indices, &innermost_rank);
    const size_t prefix_len = expanded_shape.size() - innermost_rank;
    const int64_t strip_index = RowMajorLinearIndex(
        absl::MakeConstSpan(expanded_shape).first(prefix_len),
        absl::MakeConstSpan(expanded_indices).first(prefix_len));
    std::vector<int64_t> strip_shape(expanded_shape.begin() + prefix_len,
                                     expanded_shape.end());
    std::vector<int64_t> strip_indices(expanded_indices.begin() + prefix_len,
                                       expanded_indices.end());
    int64_t strip_elements = 1;
    for (int64_t d : strip_shape) {
      strip_elements = CheckedMultiply(strip_elements, d);
    }
    const int64_t strip_bits =
        CheckedMultiply(strip_elements, ShapeUtil::ElementSizeInBits(shape));
    int64_t prior_plane_bits = 0;
    for (int64_t q = 0; q < plane_index; ++q) {
      prior_plane_bits = CheckedAdd(
          prior_plane_bits,
          CheckedMultiply(strip_elements, layout.bit_planes(q).bit_width()));
    }
    ApplyTileSequenceToCoordinates(layout.bit_planes(plane_index).tiles(),
                                   strip_shape, strip_indices,
                                   /*out_last_tile_rank=*/nullptr);
    const int64_t within_plane_index =
        RowMajorLinearIndex(strip_shape, strip_indices);
    return CheckedAdd(
        CheckedAdd(CheckedMultiply(strip_index, strip_bits), prior_plane_bits),
        CheckedMultiply(within_plane_index,
                        layout.bit_planes(plane_index).bit_width()));
  }

  int64_t prior_plane_bits = 0;
  for (int64_t q = 0; q < plane_index; ++q) {
    std::vector<int64_t> q_shape = phys_shape;
    std::vector<int64_t> q_zero_idx(phys_shape.size(), 0);
    if (!layout.bit_planes(q).tiles().empty()) {
      ApplyTileSequenceToCoordinates(layout.bit_planes(q).tiles(), q_shape,
                                     q_zero_idx,
                                     /*out_last_tile_rank=*/nullptr);
    }
    int64_t plane_elements = 1;
    for (int64_t d : q_shape) {
      plane_elements = CheckedMultiply(plane_elements, d);
    }
    if (layout.tail_padding_alignment_in_elements() != 1) {
      std::optional<int64_t> aligned = OverflowSafeRoundUpTo(
          plane_elements, layout.tail_padding_alignment_in_elements());
      CHECK(aligned.has_value());
      plane_elements = *aligned;
    }
    prior_plane_bits = CheckedAdd(
        prior_plane_bits,
        CheckedMultiply(plane_elements, layout.bit_planes(q).bit_width()));
  }

  std::vector<int64_t> plane_shape = phys_shape;
  std::vector<int64_t> plane_indices = phys_indices;
  if (!layout.bit_planes(plane_index).tiles().empty()) {
    ApplyTileSequenceToCoordinates(layout.bit_planes(plane_index).tiles(),
                                   plane_shape, plane_indices,
                                   /*out_last_tile_rank=*/nullptr);
  }
  const int64_t within_plane_index =
      RowMajorLinearIndex(plane_shape, plane_indices);
  return CheckedAdd(
      prior_plane_bits,
      CheckedMultiply(within_plane_index,
                      layout.bit_planes(plane_index).bit_width()));
}

/*static*/ int64_t LayoutUtil::MemorySpace(const Shape& shape) {
  return shape.has_layout() ? shape.layout().memory_space()
                            : Layout::kDefaultMemorySpace;
}

/*static*/ bool LayoutUtil::ByteStridesIsMajorToMinor(
    absl::Span<const int64_t> byte_strides, absl::Span<const int64_t> dims,
    PrimitiveType element_type) {
  CHECK_EQ(dims.size(), byte_strides.size());
  int64_t stride = ShapeUtil::ByteSizeOfPrimitiveType(element_type);
  for (int i = dims.size() - 1; i >= 0; --i) {
    if (byte_strides[i] != stride) {
      return false;
    }
    stride *= dims[i];
  }
  return true;
}

/*static*/ int64_t LayoutUtil::MaxSplitSize(const Shape& shape, int64_t dim) {
  CHECK(shape.IsArray()) << ShapeUtil::HumanString(shape);
  if (!shape.has_layout()) {
    return shape.dimensions(dim);
  }
  const SplitConfig* split_config = nullptr;
  for (const SplitConfig& config : shape.layout().split_configs()) {
    if (Major(shape.layout(), config.dimension()) == dim) {
      split_config = &config;
      break;
    }
  }
  if (split_config != nullptr) {
    int64_t max_split_size = 0;
    int64_t last_split_index = 0;
    for (int split_index : split_config->split_indices()) {
      int64_t split_size = split_index - last_split_index;
      max_split_size = std::max(split_size, max_split_size);
      last_split_index = split_index;
    }
    max_split_size =
        std::max(max_split_size, shape.dimensions(dim) - last_split_index);
    return max_split_size;
  }
  return shape.dimensions(dim);
}

/*static*/ int64_t LayoutUtil::MaxElementsInPerSplit(const Shape& shape) {
  CHECK(shape.IsArray()) << ShapeUtil::HumanString(shape);
  int64_t max_elements_in = 1;
  for (int dim = 0; dim < shape.dimensions().size(); ++dim) {
    max_elements_in *= MaxSplitSize(shape, dim);
  }
  return max_elements_in;
}

/*static*/ std::optional<SplitConfig> LayoutUtil::GetSplitConfig(
    const Shape& shape) {
  const absl::Span<const SplitConfig>& configs = shape.layout().split_configs();
  CHECK_LE(configs.size(), 1);

  if (configs.empty()) {
    return std::nullopt;
  }
  return configs[0];
}

/*static*/ std::vector<int64_t> LayoutUtil::PhysicalDimensions(
    const Shape& shape) {
  CHECK(shape.IsArray()) << ShapeUtil::HumanString(shape);
  const Layout& layout = shape.layout();
  const int64_t rank = shape.dimensions().size();
  std::vector<int64_t> phys_dims;
  phys_dims.reserve(rank);
  for (int64_t i = rank - 1; i >= 0; --i) {
    const int64_t logical_dim = layout.minor_to_major(i);
    if (shape.dimensions(logical_dim) == Shape::kUnboundedSize) {
      phys_dims.push_back(Shape::kUnboundedSize);
    } else {
      phys_dims.push_back(MaxSplitSize(shape, logical_dim));
    }
  }
  return phys_dims;
}

/*static*/ absl::StatusOr<LayoutUtil::ResolvedTileStep>
LayoutUtil::ResolveTileStep(absl::Span<const int64_t> current_shape,
                            const Tile& tile,
                            absl::Span<const int64_t> sub_tile_dims) {
  if (tile.dimensions().empty()) {
    return InvalidArgument("layout has empty tile dimensions");
  }
  if (tile.dimensions().back() == Tile::kCombineDimension) {
    return InvalidArgument("tile cannot end with kCombineDimension ('*')");
  }
  for (int64_t dim : tile.dimensions()) {
    if (dim <= 0 && dim != Tile::kCombineDimension &&
        dim != Tile::kMatchDimension) {
      return InvalidArgument(
          "tile dimension %d must be positive, kCombineDimension, or "
          "kMatchDimension",
          dim);
    }
  }

  ResolvedTileStep result;
  std::vector<int64_t> padded_shape(current_shape.begin(), current_shape.end());
  if (tile.dimensions().size() > padded_shape.size()) {
    result.leading_pad_dims = tile.dimensions().size() - padded_shape.size();
    padded_shape.insert(padded_shape.begin(), result.leading_pad_dims, 1);
  }
  result.suffix_start = padded_shape.size() - tile.dimensions().size();
  result.padded_suffix.assign(padded_shape.begin() + result.suffix_start,
                              padded_shape.end());

  std::vector<int64_t> suffix = result.padded_suffix;
  std::vector<int64_t> combined_dims;
  combined_dims.reserve(tile.dimensions().size());
  result.effective_tile_dims.reserve(tile.dimensions().size());
  for (size_t i = 0; i < tile.dimensions().size(); ++i) {
    if (tile.dimension(i) == Tile::kCombineDimension) {
      if (suffix[i] == Shape::kUnboundedSize ||
          suffix[i + 1] == Shape::kUnboundedSize) {
        suffix[i + 1] = Shape::kUnboundedSize;
      } else {
        auto [prod, overflow] = OverflowSafeMultiply(suffix[i], suffix[i + 1]);
        if (overflow) {
          return InvalidArgument("combined tile dimension overflows");
        }
        suffix[i + 1] = prod;
      }
    } else {
      combined_dims.push_back(suffix[i]);
      result.effective_tile_dims.push_back(tile.dimension(i));
    }
  }

  for (size_t j = 0; j < result.effective_tile_dims.size(); ++j) {
    if (result.effective_tile_dims[j] == Tile::kMatchDimension) {
      const int64_t minor =
          static_cast<int64_t>(result.effective_tile_dims.size()) - 1 -
          static_cast<int64_t>(j);
      int64_t packing = 1;
      if (minor < static_cast<int64_t>(sub_tile_dims.size())) {
        packing = std::max<int64_t>(
            1, sub_tile_dims[sub_tile_dims.size() - 1 - minor]);
      } else if (minor == 1 && !sub_tile_dims.empty()) {
        packing = std::max<int64_t>(1, sub_tile_dims[0]);
      }
      if (combined_dims[j] == Shape::kUnboundedSize) {
        result.effective_tile_dims[j] = packing;
      } else {
        std::optional<int64_t> rounded = OverflowSafeRoundUpTo(
            std::max<int64_t>(combined_dims[j], 1), packing);
        if (!rounded.has_value()) {
          return InvalidArgument("kMatchDimension tile size overflows");
        }
        result.effective_tile_dims[j] = *rounded;
      }
    }
  }

  result.next_shape.reserve(result.suffix_start +
                            2 * result.effective_tile_dims.size());
  for (size_t i = 0; i < result.suffix_start; ++i) {
    result.next_shape.push_back(padded_shape[i]);
  }
  for (size_t j = 0; j < result.effective_tile_dims.size(); ++j) {
    if (combined_dims[j] == Shape::kUnboundedSize) {
      result.next_shape.push_back(Shape::kUnboundedSize);
    } else {
      const int64_t t = result.effective_tile_dims[j];
      result.next_shape.push_back(combined_dims[j] / t +
                                  (combined_dims[j] % t != 0 ? 1 : 0));
    }
  }
  for (int64_t t : result.effective_tile_dims) {
    result.next_shape.push_back(t);
  }
  return result;
}

/*static*/ bool LayoutUtil::IsUntiledLayout(absl::Span<const Tile> tiles,
                                            absl::Span<const int64_t> shape) {
  // Tiles are applied recursively to expand current_shape
  // Example: (t0, t1) tile applied to (..., n, m) expands it to
  // (..., ceildiv(n, t0), ceildiv(m, t1), t0, t1)
  std::vector<int64_t> current_shape(shape.begin(), shape.end());
  for (const Tile& tile : tiles) {
    const int64_t tile_ndims = tile.dimensions().size();
    CHECK_LE(tile_ndims, current_shape.size());
    const absl::Span<const int64_t> tiled_shape =
        absl::Span<const int64_t>(current_shape).last(tile_ndims);
    // new_tiled_shape will hold the tiled shape after the tile is applied.
    std::vector<int64_t> new_tiled_shape(2 * tile_ndims);
    bool allow_multiple_tiles = true;
    for (int64_t i = 0; i < tile_ndims; ++i) {
      if (tiled_shape[i] % tile.dimension(i) != 0) {
        return false;
      }
      CHECK_GT(tile.dimension(i), 0);
      new_tiled_shape[i] = tiled_shape[i] / tile.dimension(i);
      new_tiled_shape[tile_ndims + i] = tile.dimension(i);
      if (!allow_multiple_tiles && new_tiled_shape[i] != 1) {
        return false;
      }
      if (tile.dimension(i) != 1) {
        allow_multiple_tiles = false;
      }
    }
    current_shape.erase(current_shape.end() - tile_ndims, current_shape.end());
    current_shape.insert(current_shape.end(), new_tiled_shape.begin(),
                         new_tiled_shape.end());
  }
  return true;
}

}  // namespace xla
