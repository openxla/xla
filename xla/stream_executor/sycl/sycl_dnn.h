/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

// oneDNN library support, implementing the general DnnSupport interface.

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_DNN_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_DNN_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/plugin_registry.h"

namespace stream_executor {
namespace gpu {

class GpuExecutor;

// onednn-library based DNN support. For details on overridden interface
// functions, see dnn.h.
class OnednnSupport : public dnn::DnnSupport {
 public:
  explicit OnednnSupport(GpuExecutor* parent);

  absl::Status Init() override;
  absl::StatusOr<stream_executor::dnn::VersionInfo> GetVersion() override;

  absl::Status DoConvolve(
      dnn::ConvolutionKind kind, dnn::DataType element_type,
      dnn::DataType output_type, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemoryBase filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemoryBase output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      dnn::AlgorithmDesc algorithm_desc, DeviceMemory<uint8_t> scratch_memory,
      dnn::ProfileResult* output_profile_result) override {
    return absl::UnimplementedError(
        "DnnSupport::DoConvolve not implemented on this platform.");
  }

  absl::Status DoPoolForward(dnn::DataType element_type, Stream* stream,
                             const dnn::PoolingDescriptor& pooling_dimensions,
                             const dnn::BatchDescriptor& input_dimensions,
                             DeviceMemoryBase input_data,
                             const dnn::BatchDescriptor& output_dimensions,
                             DeviceMemoryBase output_data,
                             ScratchAllocator* workspace_allocator) override {
    return absl::UnimplementedError(
        "DnnSupport::DoPoolForward not implemented on this platform.");
  }

  absl::Status DoPoolBackward(dnn::DataType element_type, Stream* stream,
                              const dnn::PoolingDescriptor& pooling_dimensions,
                              const dnn::BatchDescriptor& input_dimensions,
                              DeviceMemoryBase input_data,
                              const dnn::BatchDescriptor& output_dimensions,
                              DeviceMemoryBase output_data,
                              DeviceMemoryBase input_diff_data,
                              DeviceMemoryBase output_diff_data,
                              ScratchAllocator* workspace_allocator) override {
    return absl::UnimplementedError(
        "DnnSupport::DoPoolBackward not implemented on this platform.");
  }

 private:
  GpuExecutor* parent_;  // Parent executor object. Not owned.

  OnednnSupport(const OnednnSupport&) = delete;
  void operator=(const OnednnSupport&) = delete;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_DNN_H_
