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

#include "xla/backends/gpu/runtime/convolution_reorder_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/convolution_filter_thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

static se::dnn::FilterDescriptor CreateFilterDescriptor(
    const ConvolutionFilterDimensions& filter_dimensions) {
  se::dnn::FilterDescriptor filter_desc(/*ndims=*/2);
  filter_desc.set_layout(se::dnn::FilterLayout::kOutputInputYX32);
  filter_desc.set_output_feature_map_count(
      filter_dimensions.output_feature_map_count());
  filter_desc.set_input_feature_map_count(
      filter_dimensions.input_feature_map_count());
  filter_desc.set_input_filter_height(filter_dimensions.input_filter_height());
  filter_desc.set_input_filter_width(filter_dimensions.input_filter_width());
  return filter_desc;
}

ConvolutionReorderThunk::ConvolutionReorderThunk(
    ThunkInfo thunk_info, ConvolutionFilterDimensions filter_dimensions,
    BufferAllocation::Slice filter_input, BufferAllocation::Slice filter_output,
    std::optional<BiasBuffers> biases)
    : Thunk(Kind::kConvolutionReorder, thunk_info),
      filter_dimensions_(std::move(filter_dimensions)),
      filter_descriptor_(CreateFilterDescriptor(filter_dimensions_)),
      filter_input_(filter_input),
      filter_output_(filter_output),
      biases_(biases) {}

absl::Status ConvolutionReorderThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  const auto& buffer_allocations = *params.buffer_allocations;

  auto filter_input = se::DeviceMemory<int8_t>(
      buffer_allocations.GetDeviceAddress(filter_input_));
  auto filter_output = se::DeviceMemory<int8_t>(
      buffer_allocations.GetDeviceAddress(filter_output_));

  std::optional<se::DeviceMemory<float>> bias_input;
  std::optional<se::DeviceMemory<float>> bias_output;
  if (biases_.has_value()) {
    bias_input = se::DeviceMemory<float>(
        buffer_allocations.GetDeviceAddress(biases_->bias_input));
    bias_output = se::DeviceMemory<float>(
        buffer_allocations.GetDeviceAddress(biases_->bias_output));
  }

  auto dnn = params.stream->parent()->AsDnn();
  if (dnn == nullptr) {
    return absl::InternalError("No DNN for stream.");
  }
  return dnn->CudnnReorderConvolutionFilterAndBias(
      params.stream, filter_descriptor_, filter_input, &filter_output,
      std::move(bias_input), std::move(bias_output));
}

absl::StatusOr<std::unique_ptr<ConvolutionReorderThunk>>
ConvolutionReorderThunk::FromProto(
    ThunkInfo thunk_info, const ConvolutionReorderThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice filter_input,
                      BufferAllocation::Slice::FromProto(proto.filter_input(),
                                                         buffer_allocations));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice filter_output,
                      BufferAllocation::Slice::FromProto(proto.filter_output(),
                                                         buffer_allocations));

  std::optional<BiasBuffers> biases;
  if (proto.has_biases()) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice bias_input,
                        BufferAllocation::Slice::FromProto(
                            proto.biases().bias_input(), buffer_allocations));
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice bias_output,
                        BufferAllocation::Slice::FromProto(
                            proto.biases().bias_output(), buffer_allocations));
    biases = {{bias_input, bias_output}};
  }

  return std::make_unique<ConvolutionReorderThunk>(
      std::move(thunk_info), proto.filter_dimensions(), filter_input,
      filter_output, biases);
}

absl::StatusOr<ThunkProto> ConvolutionReorderThunk::ToProto() const {
  ThunkProto thunk_proto;
  *thunk_proto.mutable_thunk_info() = thunk_info().ToProto();

  ConvolutionReorderThunkProto* reorder_proto =
      thunk_proto.mutable_convolution_reorder_thunk();
  *reorder_proto->mutable_filter_dimensions() = filter_dimensions_;

  TF_ASSIGN_OR_RETURN(*reorder_proto->mutable_filter_input(),
                      filter_input_.ToProto());
  TF_ASSIGN_OR_RETURN(*reorder_proto->mutable_filter_output(),
                      filter_output_.ToProto());

  if (biases_.has_value()) {
    ConvolutionReorderBiasBuffers* biases_proto =
        reorder_proto->mutable_biases();
    TF_ASSIGN_OR_RETURN(*biases_proto->mutable_bias_input(),
                        biases_->bias_input.ToProto());
    TF_ASSIGN_OR_RETURN(*biases_proto->mutable_bias_output(),
                        biases_->bias_output.ToProto());
  }

  return thunk_proto;
}

}  // namespace gpu
}  // namespace xla
