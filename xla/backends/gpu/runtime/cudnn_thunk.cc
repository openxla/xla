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

#include "xla/backends/gpu/runtime/cudnn_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

CuDnnThunk::CuDnnThunk(std::string fingerprint, ThunkInfo thunk_info,
                       std::vector<BufferAllocation::Slice> args,
                       std::optional<int64_t> sdpa_dropout_seed)
    : Thunk(Kind::kCuDnn, std::move(thunk_info)),
      fingerprint_(std::move(fingerprint)),
      graph_(std::make_shared<se::dnn::LazyDnnGraph>(nullptr)),
      args_(std::move(args)),
      sdpa_dropout_seed_(sdpa_dropout_seed) {}

absl::Status CuDnnThunk::Initialize(const InitializeParams& params) {
  absl::Status ret = absl::OkStatus();
  absl::call_once(once_flag_, [&] {
    auto result = params.stream->parent()->AsDnn()->DeserializeGraph(
        *params.stream, params.src.dnn_compiled_graphs.at(fingerprint_));
    std::string().swap(fingerprint_);
    if (result.ok()) {
      graph_->swap(*result);
      if (sdpa_dropout_seed_.has_value()) {
        graph_->get()->InitDropoutState(params.local_device_count,
                                        *sdpa_dropout_seed_, 16);
      }
    }
    ret = result.status();
  });
  return ret;
}

absl::Status CuDnnThunk::ExecuteOnStream(const ExecuteParams& params) {
  InitializeParams initialize_params;
  initialize_params.stream = params.stream;
  TF_RETURN_IF_ERROR(Initialize(initialize_params));
  std::vector<se::DeviceMemoryBase> buffer_args;
  buffer_args.reserve(args_.size());
  for (const BufferAllocation::Slice& arg : args_) {
    buffer_args.push_back(params.buffer_allocations->GetDeviceAddress(arg));
  }
  return graph_->get()->Execute(*params.stream,
                                absl::Span<se::DeviceMemoryBase>(buffer_args),
                                params.collective_params->local_device_ordinal);
}

absl::StatusOr<ThunkProto> CuDnnThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();
  proto.mutable_cudnn_thunk()->set_fingerprint(fingerprint_);

  for (const BufferAllocation::Slice& arg : args_) {
    TF_ASSIGN_OR_RETURN(*proto.mutable_cudnn_thunk()->add_args(),
                        arg.ToProto());
  }
  if (sdpa_dropout_seed_.has_value()) {
    proto.mutable_cudnn_thunk()->set_sdpa_dropout_seed(
        static_cast<int64_t>(*sdpa_dropout_seed_));
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<CuDnnThunk>> CuDnnThunk::FromProto(
    ThunkInfo thunk_info, const CudnnThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  std::vector<BufferAllocation::Slice> args;
  args.reserve(proto.args_size());
  for (const buffer_assignment::BufferAllocationSliceProto& arg :
       proto.args()) {
    TF_ASSIGN_OR_RETURN(args.emplace_back(), BufferAllocation::Slice::FromProto(
                                                 arg, buffer_allocations));
  }
  std::optional<uint64_t> sdpa_dropout_seed;
  if (proto.has_sdpa_dropout_seed()) {
    sdpa_dropout_seed = static_cast<uint64_t>(proto.sdpa_dropout_seed());
  }
  return std::make_unique<CuDnnThunk>(proto.fingerprint(),
                                      std::move(thunk_info), std::move(args),
                                      sdpa_dropout_seed);
}

}  // namespace gpu
}  // namespace xla
