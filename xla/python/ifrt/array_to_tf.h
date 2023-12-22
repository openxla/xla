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

#ifndef XLA_PYTHON_IFRT_ARRAY_TO_TF_H_
#define XLA_PYTHON_IFRT_ARRAY_TO_TF_H_

#include "absl/status/statusor.h"
#include "xla/python/ifrt/client.h"
#include "third_party/tensorflow/core/framework/tensor.h"

namespace xla::ifrt {

// Copys a single-device ifrt array to a tensorflow::Tensor.
absl::StatusOr<std::pair<tensorflow::Tensor, xla::PjRtFuture<absl::Status>>>
ConvertArrayToTFTensors(tsl::RCReference<xla::ifrt::Array> array,
                        xla::ifrt::ArrayCopySemantics semantics);

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> ConvertTFTensorToArray(
    xla::ifrt::Client* client, const tensorflow::Tensor& tensor,
    std::shared_ptr<const Sharding> sharding,
    Client::HostBufferSemantics semantics);

}  // namespace xla::ifrt

#endif  // XLA_PYTHON_IFRT_ARRAY_TO_TF_H_
