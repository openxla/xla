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

#include "xla/backends/cpu/runtime/convolution_dims.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>

#include "absl/container/inlined_vector.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/overflow_util.h"
#include "xla/primitive_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

absl::InlinedVector<BufferUse, 4> ConvolutionBufferUses(
    const ConvolutionSlices& slices) {
  return {BufferUse::Read(slices.input_buffer, slices.input_shape),
          BufferUse::Read(slices.kernel_buffer, slices.kernel_shape),
          BufferUse::Write(slices.output_buffer, slices.output_shape)};
}

ConvolutionCanonicalDims::Dims::Dims(absl::Span<const int64_t> dims)
    : rank(dims.size()), x(dims[0]), y(dims[1]), z(rank == 3 ? dims[2] : 0) {}

static int64_t GetConvolutionRank(const Shape& input_shape) {
  // Convolution rank is the number of spatial dimensions. Besides spatial
  // dimensions, input shape contains two other dimensions (batch size and the
  // number of channels).
  return static_cast<int64_t>(input_shape.dimensions().size()) - 2;
}

static absl::Status ValidateDenseArrayShape(const Shape& shape,
                                            absl::string_view name) {
  ABSL_RETURN_IF_ERROR(ShapeUtil::ValidateShape(shape));
  if (!shape.IsArray() || !shape.is_static() || !shape.has_layout()) {
    return InvalidArgument("ConvolutionThunk: Non-static or non-array %s: %s",
                           name, shape.ToString(true));
  }
  const Layout& l = shape.layout();
  if (!l.tiles().empty() || !l.split_configs().empty() ||
      l.tail_padding_alignment_in_elements() != 1 || l.has_physical_shape() ||
      l.dynamic_shape_metadata_prefix_bytes() != 0 ||
      (l.element_size_in_bits() != 0 &&
       l.element_size_in_bits() !=
           primitive_util::BitWidth(shape.element_type()))) {
    return InvalidArgument("ConvolutionThunk: Unsupported %s layout: %s", name,
                           shape.ToString(true));
  }
  return absl::OkStatus();
}

static absl::Status ValidateCanonicalLayout(
    const Shape& shape, int64_t outer_or_in, absl::Span<const int64_t> spatial,
    int64_t out_feature, bool batch_first, absl::string_view name) {
  absl::InlinedVector<int64_t, 5> order;
  if (batch_first) {
    order.push_back(outer_or_in);
  }
  order.insert(order.end(), spatial.begin(), spatial.end());
  if (!batch_first) {
    order.push_back(outer_or_in);
  }
  order.push_back(out_feature);
  const size_t rank = shape.dimensions().size();
  const absl::Span<const int64_t> m2m = shape.layout().minor_to_major();
  if (order.size() != rank || m2m.size() != rank) {
    return InvalidArgument("ConvolutionThunk: Invalid %s dimension numbers",
                           name);
  }
  uint32_t seen = 0;
  for (size_t i = 0; i < rank; ++i) {
    const int64_t d = order[i];
    if (d < 0 || d >= static_cast<int64_t>(rank) ||
        (seen & (uint32_t{1} << d)) != 0 || m2m[rank - 1 - i] != d) {
      return InvalidArgument(
          "ConvolutionThunk: Invalid %s dimensions or layout for %s", name,
          shape.ToString(true));
    }
    seen |= uint32_t{1} << d;
  }
  return absl::OkStatus();
}

static absl::Status ValidateConvolutionShapes(
    const Shape& input_shape, const Shape& kernel_shape,
    const Shape& output_shape, const ConvolutionDimensionNumbers& dnums) {
  ABSL_RETURN_IF_ERROR(ValidateDenseArrayShape(input_shape, "input"));
  ABSL_RETURN_IF_ERROR(ValidateDenseArrayShape(kernel_shape, "kernel"));
  ABSL_RETURN_IF_ERROR(ValidateDenseArrayShape(output_shape, "output"));

  // Convolution rank.
  int64_t convolution_rank = GetConvolutionRank(input_shape);
  if (convolution_rank > 3 || convolution_rank < 1) {
    return InvalidArgument("ConvolutionThunk: Incorrect convolution rank (%d)",
                           convolution_rank);
  }

  // Rank of input, kernel and output buffers.
  if (input_shape.dimensions().size() != kernel_shape.dimensions().size() ||
      input_shape.dimensions().size() != output_shape.dimensions().size()) {
    return InvalidArgument(
        "ConvolutionThunk: Buffer ranks mismatch. Input rank (%d) vs kernel "
        "rank (%d) vs output rank (%d)",
        input_shape.dimensions().size(), kernel_shape.dimensions().size(),
        output_shape.dimensions().size());
  }

  ABSL_RETURN_IF_ERROR(
      ValidateCanonicalLayout(input_shape, dnums.input_batch_dimension(),
                              dnums.input_spatial_dimensions(),
                              dnums.input_feature_dimension(), true, "input"));
  ABSL_RETURN_IF_ERROR(ValidateCanonicalLayout(
      kernel_shape, dnums.kernel_input_feature_dimension(),
      dnums.kernel_spatial_dimensions(),
      dnums.kernel_output_feature_dimension(), false, "kernel"));
  ABSL_RETURN_IF_ERROR(ValidateCanonicalLayout(
      output_shape, dnums.output_batch_dimension(),
      dnums.output_spatial_dimensions(), dnums.output_feature_dimension(), true,
      "output"));

  // Batch size.
  auto input_batch = input_shape.dimensions(dnums.input_batch_dimension());
  auto output_batch = output_shape.dimensions(dnums.output_batch_dimension());
  if (input_batch != output_batch) {
    return InvalidArgument(
        "ConvolutionThunk: Batch sizes mismatch. Input batch (%d) vs output "
        "batch (%d)",
        input_batch, output_batch);
  }

  // Output channels / kernel filters.
  auto kernel_filters =
      kernel_shape.dimensions(dnums.kernel_output_feature_dimension());
  auto output_channels =
      output_shape.dimensions(dnums.output_feature_dimension());
  if (kernel_filters != output_channels) {
    return InvalidArgument(
        "ConvolutionThunk: Output channels mismatch. Kernel filters count (%d) "
        "should be the same as output channels count (%d)",
        kernel_filters, output_channels);
  }

  return absl::OkStatus();
}

inline bool IsConvSupportedType(PrimitiveType primitive_type) {
  return primitive_type == PrimitiveType::F16 ||
         primitive_type == PrimitiveType::F32;
}

absl::StatusOr<ConvolutionCanonicalDims> GetConvolutionCanonicalDims(
    const ConvolutionSlices& slices, const ConvolutionDimensionNumbers& dnums,
    const Window& window, int64_t feature_group_count) {
  ABSL_RETURN_IF_ERROR(ValidateConvolutionShapes(
      slices.input_shape, slices.kernel_shape, slices.output_shape, dnums));

  auto primitive_type = slices.input_shape.element_type();
  if (!IsConvSupportedType(primitive_type)) {
    return InvalidArgument("ConvolutionThunk: Unsupported element type (%s)",
                           PrimitiveType_Name(primitive_type));
  }
  if (slices.kernel_shape.element_type() != primitive_type ||
      slices.output_shape.element_type() != primitive_type) {
    return InvalidArgument(
        "ConvolutionThunk: Element types mismatch (%s vs %s vs %s)",
        PrimitiveType_Name(primitive_type),
        PrimitiveType_Name(slices.kernel_shape.element_type()),
        PrimitiveType_Name(slices.output_shape.element_type()));
  }

  absl::InlinedVector<int64_t, 3> input_dims, kernel_dims, output_dims;
  absl::InlinedVector<int64_t, 3> strides, padding_before, padding_after;
  absl::InlinedVector<int64_t, 3> base_dilation, window_dilation;

  // Lower 1D convolutions into 2D Eigen calls with a leading unit dimension.
  int64_t convolution_rank = GetConvolutionRank(slices.input_shape);
  if (convolution_rank == 1) {
    input_dims.push_back(1);
    kernel_dims.push_back(1);
    output_dims.push_back(1);
    strides.push_back(1);
    padding_before.push_back(0);
    padding_after.push_back(0);
    base_dilation.push_back(1);
    window_dilation.push_back(1);
  }

  int64_t input_batch =
      slices.input_shape.dimensions(dnums.input_batch_dimension());
  int64_t input_channels =
      slices.input_shape.dimensions(dnums.input_feature_dimension());
  int64_t kernel_channels =
      slices.kernel_shape.dimensions(dnums.kernel_input_feature_dimension());
  int64_t kernel_filters =
      slices.kernel_shape.dimensions(dnums.kernel_output_feature_dimension());

  if (feature_group_count <= 0 || input_channels % feature_group_count != 0 ||
      input_channels / feature_group_count != kernel_channels ||
      kernel_filters % feature_group_count != 0) {
    return InvalidArgument("ConvolutionThunk: Invalid feature_group_count (%d)",
                           feature_group_count);
  }
  if (window.dimensions_size() != convolution_rank) {
    return InvalidArgument("ConvolutionThunk: Window rank mismatch (%d vs %d)",
                           window.dimensions_size(), convolution_rank);
  }

  constexpr absl::int128 kMaxI64 = std::numeric_limits<int64_t>::max();
  for (int64_t i = 0; i < convolution_rank; ++i) {
    const WindowDimension& d = window.dimensions(i);
    const int64_t in_d =
        slices.input_shape.dimensions(dnums.input_spatial_dimensions(i));
    const int64_t ker_d =
        slices.kernel_shape.dimensions(dnums.kernel_spatial_dimensions(i));
    const int64_t out_d =
        slices.output_shape.dimensions(dnums.output_spatial_dimensions(i));
    if (ker_d <= 0 || (d.size() != 0 && d.size() != ker_d) || d.stride() <= 0 ||
        d.base_dilation() <= 0 || d.window_dilation() <= 0 ||
        d.window_reversal()) {
      return InvalidArgument("ConvolutionThunk: Invalid window at dim %d", i);
    }
    const absl::int128 dil_base =
        in_d == 0 ? 0 : absl::int128(in_d - 1) * d.base_dilation() + 1;
    const absl::int128 dil_win =
        absl::int128(ker_d - 1) * d.window_dilation() + 1;
    const absl::int128 padded = dil_base + d.padding_low() + d.padding_high();
    const absl::int128 expected =
        padded < dil_win ? 0 : (padded - dil_win) / d.stride() + 1;
    if (dil_base > kMaxI64 || dil_win > kMaxI64 || expected != out_d) {
      return InvalidArgument(
          "ConvolutionThunk: Output spatial dim %d mismatch (%d)", i, out_d);
    }
    input_dims.push_back(in_d);
    kernel_dims.push_back(ker_d);
    output_dims.push_back(out_d);
    strides.push_back(d.stride());
    padding_before.push_back(d.padding_low());
    padding_after.push_back(d.padding_high());
    base_dilation.push_back(d.base_dilation());
    window_dilation.push_back(d.window_dilation());
  }

  auto fits_product = [](int64_t acc, absl::Span<const int64_t> dims) {
    for (int64_t d : dims) {
      bool overflow = false;
      std::tie(acc, overflow) = OverflowSafeMultiply(acc, d);
      if (overflow) {
        return false;
      }
    }
    return acc <= std::numeric_limits<ptrdiff_t>::max();
  };
  if (!fits_product(input_batch, output_dims) ||
      !fits_product(kernel_channels, kernel_dims)) {
    return InvalidArgument("ConvolutionThunk: Patch dimensions overflow");
  }

  TF_RET_CHECK(input_dims.size() >= 2 && input_dims.size() <= 3);

  using Dims = ConvolutionCanonicalDims::Dims;
  return ConvolutionCanonicalDims{
      input_batch,         Dims(input_dims),    input_channels,
      Dims(kernel_dims),   kernel_channels,     kernel_filters,
      Dims(output_dims),   Dims(strides),       Dims(padding_before),
      Dims(padding_after), Dims(base_dilation), Dims(window_dilation),
      feature_group_count};
}

}  // namespace xla::cpu
