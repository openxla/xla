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

#include "xla/service/gpu/runtime/cudnn_thunk.h"
#include <utility>

#include "absl/status/status.h"

namespace xla {
namespace gpu {

CuDnnThunk::CuDnnThunk(std::unique_ptr<se::dnn::DnnGraph> graph,
                       int64_t plan_id, std::string fingerprint,
                       ThunkInfo thunk_info,
                       absl::Span<const KernelArgument> kernel_arguments)
    : graph_(std::move(graph)),
      plan_id_(plan_id),
      fingerprint_(std::move(fingerprint)),
      Thunk(Kind::kCuDnn, std::move(thunk_info)) {
  args_.reserve(kernel_arguments.size());
  for (const KernelArgument& kernel_argument : kernel_arguments) {
    args_.push_back(kernel_argument.slice());
  }
}

absl::Status CuDnnThunk::InitializeImpl(const InitializeParams& params) {
  std::vector<uint8_t>& cache_entry =
      params.dnn_graph_compilation_cache[fingerprint_];
  std::string().swap(fingerprint_);

  if (!cache_entry.empty()) {
    VLOG(8) << "Compilation cache hit.";
    TF_ASSIGN_OR_RETURN(
        graph_,
        params.stream->parent()->AsDnn()->DeserializeGraph(absl::string_view(
            reinterpret_cast<char*>(cache_entry.data()), cache_entry.size())));
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(bool supported, graph_->Prepare());
  if (!supported) {
    return absl::InternalError("cuDNN graph is not supported.");
  } else {
    VLOG(4) << "Plan ID: " << plan_id_;
    if (plan_id_ >= 0) {
      // Build single plan with given ID.
      if (plan_id_ >= graph_->ExecutionPlanCount()) {
        return absl::InternalError("cuDNN graph plan does not exist.");
      }
      TF_RETURN_IF_ERROR(graph_->Build(plan_id_));
    } else {
      // Build plans one by one till first successful when no plan_id was
      // provided.
      for (plan_id_ = 0; plan_id_ < graph_->ExecutionPlanCount(); ++plan_id_) {
        VLOG(7) << "Trying plan ID " << plan_id_;
        if (graph_->Build(plan_id_).ok()) {
          VLOG(7) << "Successfully built plan ID " << plan_id_;
          break;
        }
      }
      if (plan_id_ == graph_->ExecutionPlanCount()) {
        return absl::InternalError("No cuDNN plans can be built.");
      }
    }

    if (graph_->WorkspaceSize() != 0) {
      return absl::UnimplementedError(
          "Support of workspace allocation is not added yet.");
    }

    TF_ASSIGN_OR_RETURN(cache_entry, graph_->Serialize());

    return absl::OkStatus();
  }
}

absl::Status CuDnnThunk::Initialize(const InitializeParams& params) {
  absl::Status ret = absl::OkStatus();
  absl::call_once(once_flag_, [&] { ret = InitializeImpl(params); });
  return ret;
}

absl::Status CuDnnThunk::ExecuteOnStream(const ExecuteParams& params) {
  std::vector<se::DeviceMemoryBase> buffer_args;
  buffer_args.reserve(args_.size());
  for (const BufferAllocation::Slice& arg : args_) {
    buffer_args.push_back(params.buffer_allocations->GetDeviceAddress(arg));
  }
  return graph_->Execute(*params.stream,
                         absl::Span<se::DeviceMemoryBase>(buffer_args));
}

}  // namespace gpu
}  // namespace xla
