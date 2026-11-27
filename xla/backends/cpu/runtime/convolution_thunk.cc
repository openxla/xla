/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/convolution_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/runtime/convolution_dims.h"
#include "xla/backends/cpu/runtime/convolution_lib.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

#define EIGEN_USE_THREADS
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

absl::StatusOr<int64_t> ValidateSliceBuffer(
    const BufferAllocation::Slice& slice, const Shape& shape,
    se::DeviceAddressBase data, size_t alignment, absl::string_view name) {
  const int64_t bytes = ShapeUtil::ByteSizeOf(shape);
  if (slice.size() < bytes || data.size() < static_cast<uint64_t>(bytes)) {
    return InvalidArgument("ConvolutionThunk: %s buffer too small (%d vs %d)",
                           name, slice.size(), bytes);
  }
  if (bytes > 0 && (data.opaque() == nullptr ||
                    reinterpret_cast<uintptr_t>(data.opaque()) % alignment)) {
    return InvalidArgument("ConvolutionThunk: %s buffer null or misaligned",
                           name);
  }
  return bytes;
}

bool RangesOverlap(const void* a, int64_t a_n, const void* b, int64_t b_n) {
  if (a_n <= 0 || b_n <= 0) {
    return false;
  }
  const auto pa = reinterpret_cast<uintptr_t>(a);
  const auto pb = reinterpret_cast<uintptr_t>(b);
  return pa <= pb ? pb - pa < static_cast<uintptr_t>(a_n)
                  : pa - pb < static_cast<uintptr_t>(b_n);
}

}  // namespace

absl::StatusOr<std::unique_ptr<ConvolutionThunk>> ConvolutionThunk::Create(
    Info info, Options options, BufferAllocation::Slice input_buffer,
    const Shape& input_shape, BufferAllocation::Slice kernel_buffer,
    const Shape& kernel_shape, BufferAllocation::Slice output_buffer,
    const Shape& output_shape, const ConvolutionDimensionNumbers& dnums,
    const Window& window, int64_t feature_group_count) {
  ConvolutionSlices slices = {input_buffer, input_shape,   kernel_buffer,
                              kernel_shape, output_buffer, output_shape};

  ABSL_ASSIGN_OR_RETURN(
      ConvolutionCanonicalDims canonical_dims,
      GetConvolutionCanonicalDims(slices, dnums, window, feature_group_count));

  return absl::WrapUnique(new ConvolutionThunk(std::move(info), options, slices,
                                               canonical_dims, dnums, window));
}

ConvolutionThunk::ConvolutionThunk(
    Info info, Options options, ConvolutionSlices convolution_slices,
    ConvolutionCanonicalDims convolution_canonical_dims,
    ConvolutionDimensionNumbers dnums, Window window)
    : Thunk(Kind::kConvolution, std::move(info)),
      options_(options),
      convolution_slices_(convolution_slices),
      convolution_canonical_dims_(convolution_canonical_dims),
      dnums_(std::move(dnums)),
      window_(std::move(window)) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> ConvolutionThunk::Execute(
    const ExecuteParams& params) {
  if (params.buffer_allocations == nullptr) {
    return InvalidArgument("ConvolutionThunk: buffer_allocations is null");
  }
  ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase input_data,
                        params.buffer_allocations->GetDeviceAddress(
                            convolution_slices_.input_buffer));
  ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase kernel_data,
                        params.buffer_allocations->GetDeviceAddress(
                            convolution_slices_.kernel_buffer));
  ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase output_data,
                        params.buffer_allocations->GetDeviceAddress(
                            convolution_slices_.output_buffer));

  const size_t align =
      convolution_slices_.input_shape.element_type() == PrimitiveType::F16
          ? alignof(Eigen::half)
          : alignof(float);
  const auto& s = convolution_slices_;
  ABSL_ASSIGN_OR_RETURN(int64_t in_n,
                        ValidateSliceBuffer(s.input_buffer, s.input_shape,
                                            input_data, align, "input"));
  ABSL_ASSIGN_OR_RETURN(int64_t ker_n,
                        ValidateSliceBuffer(s.kernel_buffer, s.kernel_shape,
                                            kernel_data, align, "kernel"));
  ABSL_ASSIGN_OR_RETURN(int64_t out_n,
                        ValidateSliceBuffer(s.output_buffer, s.output_shape,
                                            output_data, align, "output"));
  if (RangesOverlap(output_data.opaque(), out_n, input_data.opaque(), in_n) ||
      RangesOverlap(output_data.opaque(), out_n, kernel_data.opaque(), ker_n)) {
    return InvalidArgument("ConvolutionThunk: output overlaps operand buffer");
  }

  VLOG(3) << absl::StreamFormat("Convolution: %v", convolution_canonical_dims_);

  VLOG(3) << absl::StreamFormat("  input: %s in slice %s (%p)",
                                convolution_slices_.input_shape.ToString(true),
                                convolution_slices_.input_buffer.ToString(),
                                input_data.opaque());
  VLOG(3) << absl::StreamFormat("  kernel: %s in slice %s (%p)",
                                convolution_slices_.kernel_shape.ToString(true),
                                convolution_slices_.kernel_buffer.ToString(),
                                kernel_data.opaque());
  VLOG(3) << absl::StreamFormat("  output: %s in slice %s (%p)",
                                convolution_slices_.output_shape.ToString(true),
                                convolution_slices_.output_buffer.ToString(),
                                output_data.opaque());

  // Eigen convolution
  if (convolution_canonical_dims_.convolution_rank() == 2) {
    return HandleEigen2DConvolution(params, input_data, kernel_data,
                                    output_data);
  } else {
    return HandleEigen3DConvolution(params, input_data, kernel_data,
                                    output_data);
  }
}

tsl::AsyncValueRef<Thunk::ExecuteEvent>
ConvolutionThunk::HandleEigen2DConvolution(const ExecuteParams& params,
                                           se::DeviceAddressBase input,
                                           se::DeviceAddressBase kernel,
                                           se::DeviceAddressBase output) {
  auto dispatch = [&](auto type_tag) {
    using ScalarType = decltype(type_tag);
    auto execute_event = tsl::MakeConstructedAsyncValueRef<ExecuteEvent>();
    const auto& d = convolution_canonical_dims_;
    internal::EigenConv2D(
        options_.multi_threaded ? params.intra_op_threadpool : nullptr,
        static_cast<ScalarType*>(output.opaque()),
        static_cast<const ScalarType*>(input.opaque()),
        static_cast<const ScalarType*>(kernel.opaque()), d.input_batch,
        d.input_dims.x, d.input_dims.y, d.input_channels, d.kernel_dims.x,
        d.kernel_dims.y, d.kernel_channels, d.kernel_filters, d.output_dims.x,
        d.output_dims.y, d.strides.x, d.strides.y, d.padding_before.x,
        d.padding_after.x, d.padding_before.y, d.padding_after.y,
        d.base_dilation.x, d.base_dilation.y, d.window_dilation.x,
        d.window_dilation.y, d.feature_group_count,
        [execute_event] { execute_event.SetStateConcrete(); });
    return execute_event;
  };

  PrimitiveType input_type = convolution_slices_.input_shape.element_type();
  switch (input_type) {
    case PrimitiveType::F16:
      return dispatch(Eigen::half{});
    case PrimitiveType::F32:
      return dispatch(float{});
    default:
      return Internal("Unsupported Conv2D input type: %s",
                      primitive_util::LowercasePrimitiveTypeName(input_type));
  }
}

tsl::AsyncValueRef<Thunk::ExecuteEvent>
ConvolutionThunk::HandleEigen3DConvolution(const ExecuteParams& params,
                                           se::DeviceAddressBase input,
                                           se::DeviceAddressBase kernel,
                                           se::DeviceAddressBase output) {
  auto dispatch = [&](auto type_tag) {
    using ScalarType = decltype(type_tag);
    auto execute_event = tsl::MakeConstructedAsyncValueRef<ExecuteEvent>();
    const auto& d = convolution_canonical_dims_;
    internal::EigenConv3D(
        options_.multi_threaded ? params.intra_op_threadpool : nullptr,
        static_cast<ScalarType*>(output.opaque()),
        static_cast<const ScalarType*>(input.opaque()),
        static_cast<const ScalarType*>(kernel.opaque()), d.input_batch,
        d.input_dims.x, d.input_dims.y, d.input_dims.z, d.input_channels,
        d.kernel_dims.x, d.kernel_dims.y, d.kernel_dims.z, d.kernel_channels,
        d.kernel_filters, d.output_dims.x, d.output_dims.y, d.output_dims.z,
        d.strides.x, d.strides.y, d.strides.z, d.padding_before.x,
        d.padding_after.x, d.padding_before.y, d.padding_after.y,
        d.padding_before.z, d.padding_after.z, d.base_dilation.x,
        d.base_dilation.y, d.base_dilation.z, d.window_dilation.x,
        d.window_dilation.y, d.window_dilation.z, d.feature_group_count,
        [execute_event] { execute_event.SetStateConcrete(); });
    return execute_event;
  };

  PrimitiveType input_type = convolution_slices_.input_shape.element_type();
  switch (input_type) {
    case PrimitiveType::F16:
      return dispatch(Eigen::half{});
    case PrimitiveType::F32:
      return dispatch(float{});
    default:
      return Internal("Unsupported Conv3D input type: %s",
                      primitive_util::LowercasePrimitiveTypeName(input_type));
  }
}

}  // namespace xla::cpu
