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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_CONV_UTILS_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_CONV_UTILS_H_

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/sycl/onednn_util.h"

namespace stream_executor {

namespace sycl {

struct ConvFwd {
  dnnl::convolution_forward primitive;
  dnnl::memory src;
  dnnl::memory filter;
  dnnl::memory dst;
  dnnl::memory bias;
  dnnl::memory side_input;
  dnnl::memory internal_filter;
  dnnl::memory scratchpad;
  std::unordered_map<int, dnnl::memory> args;
};

struct ConvBwdData {
  dnnl::convolution_backward_data primitive;
  dnnl::memory src;
  dnnl::memory filter;
  dnnl::memory dst;
  dnnl::memory internal_filter;
  dnnl::memory scratchpad;
  std::unordered_map<int, dnnl::memory> args;
};

struct ConvBwdWeights {
  dnnl::convolution_backward_weights primitive;
  dnnl::memory src;
  dnnl::memory filter;
  dnnl::memory dst;
  dnnl::memory internal_filter;
  dnnl::memory scratchpad;
  std::unordered_map<int, dnnl::memory> args;
};

struct ReorderOp {
  dnnl::reorder primitive;
  std::unordered_map<int, dnnl::memory> args;
};

struct OneDnnConvPrimitive {
  dnnl::engine engine;
  dnnl::stream stream;
  std::variant<ConvFwd, ConvBwdData, ConvBwdWeights> op;
  std::optional<ReorderOp> filter_reorder;
  std::optional<ReorderOp> side_input_reorder;
};

// Byte layout of the workspace buffer that XLA allocates for a convolution.
// The pre-packed filter is present only when the primitive prefers a filter
// layout other than XLA's. The pad rounds the scratchpad offset up to XLA's
// buffer alignment. A region with zero size is absent.
//
//   0                                  scratchpad_offset         total_size
//   v                                  v                         v
//   +----------------------------+-----+-------------------------+
//   |     pre-packed filter      | pad |    oneDNN scratchpad    |
//   +----------------------------+-----+-------------------------+
//   |<-- internal_filter_size -->|     |<--- scratchpad_size --->|
struct OneDnnConvWorkspaceLayout {
  size_t internal_filter_size = 0;
  size_t scratchpad_offset = 0;
  size_t scratchpad_size = 0;
  size_t total_size = 0;
};

// Everything the oneDNN convolution execution path needs that is derivable
// from an OneDnnConvConfig alone (no device buffers).
struct OneDnnConvPrimitiveDesc {
  dnn::ConvolutionKind kind;
  dnnl::engine engine;
  dnnl::memory::desc src_md;
  dnnl::memory::desc filter_md;
  dnnl::memory::desc dst_md;
  // Bias for the fused forward path; absent otherwise. `arg_key` is the
  // execution-args key to bind the bias buffer under:
  //   - alpha == 1: bias is part of the conv primitive; key is DNNL_ARG_BIAS.
  //   - alpha != 1: bias is a binary_add post-op; key is
  //     DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1.
  struct Bias {
    dnnl::memory::desc md;
    int arg_key;
  };
  std::optional<Bias> bias;
  // True when a `sum` post-op is present; the caller must copy side_input
  // into dst before executing the primitive.
  bool has_side_input_sum = false;
  std::variant<dnnl::convolution_forward::primitive_desc,
               dnnl::convolution_backward_data::primitive_desc,
               dnnl::convolution_backward_weights::primitive_desc>
      conv_pd;
  OneDnnConvWorkspaceLayout workspace;
};

// Static convolution properties used by the oneDNN backend.
struct OneDnnConvConfig {
  struct Fusion {
    dnn::ActivationMode mode;
    double side_input_scale;
    double leakyrelu_alpha;
  };

  dnn::ConvolutionKind kind;
  dnn::DataType input_type;
  dnn::DataType output_type;
  dnn::BatchDescriptor input_descriptor;
  dnn::FilterDescriptor filter_descriptor;
  dnn::BatchDescriptor output_descriptor;
  dnn::ConvolutionDescriptor conv_desc;
  double conv_result_scale = 1.0;
  std::optional<Fusion> fusion;
};

struct OnednnConvBufferPointers {
  void* input_data;
  void* filter_data;
  void* result_data;
  void* bias_data = nullptr;
  void* side_input_data = nullptr;
};

// Binds the caller's buffers to the primitive. The scratchpad and the
// pre-packed filter live inside workspace, which must be at least
// pd.workspace.total_size bytes and stay valid until the primitive finishes
// executing on the stream.
absl::StatusOr<OneDnnConvPrimitive> CreateOneDnnConvPrimitive(
    const OneDnnConvPrimitiveDesc& pd, Stream* stream,
    const OnednnConvBufferPointers& buffer_pointers,
    DeviceAddressBase workspace);

absl::StatusOr<OneDnnConvPrimitiveDesc> CreateOneDnnConvPrimitiveDesc(
    const OneDnnConvConfig& config, const dnnl::engine& engine);

absl::StatusOr<OneDnnConvPrimitiveDesc> CreateOneDnnConvPrimitiveDesc(
    const OneDnnConvConfig& config, Stream* stream);

// Returns the workspace size, in bytes, that a convolution with this config
// needs at run time on the given device. Used by the compiler, which has no
// stream, so the primitive descriptor is built on that device's engine.
absl::StatusOr<size_t> ComputeOneDnnConvWorkspaceSize(
    const OneDnnConvConfig& config, int device_ordinal);

absl::Status DoOneDnnConv(const OneDnnConvPrimitive& onednn_primitive);

}  // namespace sycl
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_CONV_UTILS_H_
