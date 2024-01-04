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

#include "xla/python/ifrt/array_to_tf.h"

#include <memory>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "third_party/tensorflow/compiler/tf2xla/type_util.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/framework/tensor_shape.h"
#include "tsl/concurrency/ref_count.h"

namespace xla::ifrt {

absl::StatusOr<std::pair<tensorflow::Tensor, xla::PjRtFuture<absl::Status>>>
ConvertArrayToTFTensors(tsl::RCReference<xla::ifrt::Array> array,
                        xla::ifrt::ArrayCopySemantics semantics) {
  if (array->sharding().devices().size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Only single-shard is implemented, but got %d",
                        array->sharding().devices().size()));
  }
  TF_ASSIGN_OR_RETURN(xla::PrimitiveType primitive_type,
                      xla::ifrt::ToPrimitiveType(array->dtype()));
  TF_ASSIGN_OR_RETURN(
      auto tf_dtype, tensorflow::EncodePrimitiveTypeAsDataType(primitive_type));
  tensorflow::TensorShape tf_shape;
  TF_RETURN_IF_ERROR(tensorflow::TensorShape::BuildTensorShape(
      array->shape().dims(), &tf_shape));
  tensorflow::Tensor result(tf_dtype, tf_shape);
  auto future = array->CopyToHostBuffer(
      result.data(), /*byte_strides=*/std::nullopt, semantics);
  return std::make_pair(std::move(result), std::move(future));
}

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> ConvertTFTensorToArray(
    xla::ifrt::Client* client, const tensorflow::Tensor& tensor,
    std::shared_ptr<const Sharding> sharding,
    Client::HostBufferSemantics semantics) {
  xla::PrimitiveType xla_dtype;
  TF_RETURN_IF_ERROR(
      tensorflow::DataTypeToPrimitiveType(tensor.dtype(), &xla_dtype));
  TF_ASSIGN_OR_RETURN(auto dtype, ToDType(xla_dtype));
  xla::ifrt::Shape shape(tensor.shape().dim_sizes());
  return client->MakeArrayFromHostBuffer(tensor.data(), dtype, shape,
                                         std::nullopt, sharding, semantics,
                                         [tensor]() {});
}

}  // namespace xla::ifrt
