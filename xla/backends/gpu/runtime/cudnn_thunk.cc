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
#include <variant>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/traced_command.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/nvtx_utils.h"

namespace xla {
namespace gpu {
namespace {

// Zero fill commands recorded ahead of the DNN graph command. Keyed by the DNN
// graph command they precede, because the same thunk can be recorded into one
// command buffer several times (e.g. for an unrolled loop).
struct MemzeroCommands : public CommandState {
  absl::flat_hash_map<const se::CommandBuffer::Command*,
                      std::vector<const se::CommandBuffer::Command*>>
      commands;
};

}  // namespace

CuDnnThunk::CuDnnThunk(std::string fingerprint, ThunkInfo thunk_info,
                       std::vector<ShapedSlice> args,
                       std::vector<bool> output_args, bool should_memzero,
                       std::optional<int64_t> sdpa_dropout_seed)
    : TracedCommand(Kind::kCuDnn, std::move(thunk_info)),
      fingerprint_(std::move(fingerprint)),
      graph_(std::make_shared<se::dnn::LazyDnnGraph>(nullptr)),
      args_(std::move(args)),
      output_args_(std::move(output_args)),
      should_memzero_(should_memzero),
      sdpa_dropout_seed_(sdpa_dropout_seed) {}

absl::Status CuDnnThunk::Initialize(const InitializeParams& params) {
  absl::Status ret = absl::OkStatus();
  // Calling AsDnn outside call_once ensures that cuDNN handles get created for
  // all GPUs in programs using cuDNN during the executable initialization
  // phase. It's sufficient to deserialize the graph once using just one of
  // them.
  se::dnn::DnnSupport* dnn = params.stream->parent()->AsDnn();
  if (dnn == nullptr) {
    return absl::InternalError(
        "Failed to initialize DNN support for CuDnnThunk");
  }
  absl::call_once(once_flag_, [&] {
    // If the graph was externally populated (e.g. by tests that bypass the
    // fingerprint deserialization path), skip deserialization. Checking
    // inside call_once keeps the read synchronized with concurrent
    // Initialize() calls from other streams/devices.
    if (graph_->get() != nullptr) {
      return;
    }
    auto result = dnn->DeserializeGraph(
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
  RETURN_IF_ERROR(Initialize(initialize_params));
  std::vector<se::DeviceAddressBase> buffer_args;
  buffer_args.reserve(args_.size());
  for (const ShapedSlice& arg : args_) {
    auto addr = params.buffer_allocations->GetDeviceAddress(arg.slice);
    if (output_args_[buffer_args.size()]) {
      if (should_memzero_) {
        RETURN_IF_ERROR(params.stream->MemZero(&addr, addr.size()));
      }
      tsl::profiler::MarkMemoryInitialized(
          addr.opaque(), addr.size(),
          static_cast<tsl::profiler::StreamHandle>(
              params.stream->platform_specific_handle().stream));
    }
    buffer_args.push_back(addr);
  }
  return graph_->get()->Execute(
      *params.stream, absl::Span<se::DeviceAddressBase>(buffer_args),
      params.collective_params->local_device_id.value());
}

absl::StatusOr<const se::CommandBuffer::Command*> CuDnnThunk::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  CHECK(graph_ != nullptr);
  std::vector<se::DeviceAddressBase> operands;
  operands.reserve(args_.size());
  for (const ShapedSlice& arg : args_) {
    se::DeviceAddressBase buf =
        execute_params.buffer_allocations->GetDeviceAddress(arg.slice);
    VLOG(5) << "  Arg: " << arg << ": " << buf.opaque();
    operands.push_back(buf);
  }

  ASSIGN_OR_RETURN(const bool supports_explicit,
                   graph_->get()->SupportsExplicitCommandBufferConstruction());
  if (supports_explicit) {
    // The DNN graph command only covers the graph launch, so the output zero
    // fill that ExecuteOnStream applies has to be recorded separately.
    MemzeroCommands* memzero =
        should_memzero_ ? record_params.state.GetOrCreate<MemzeroCommands>(
                              this, command_buffer)
                        : nullptr;
    if (auto* create = std::get_if<RecordCreate>(&record_action)) {
      std::vector<const se::CommandBuffer::Command*> memsets;
      std::vector<const se::CommandBuffer::Command*> dependencies(
          create->dependencies.begin(), create->dependencies.end());
      if (memzero != nullptr) {
        for (int i = 0; i < operands.size(); ++i) {
          if (!output_args_[i] || operands[i].size() == 0) {
            continue;
          }
          ASSIGN_OR_RETURN(
              const se::CommandBuffer::Command* zero_fill,
              command_buffer->CreateMemset(&operands[i], uint8_t{0},
                                           /*num_elements=*/operands[i].size(),
                                           create->dependencies));
          memsets.push_back(zero_fill);
          dependencies.push_back(zero_fill);
        }
      }
      ASSIGN_OR_RETURN(
          const se::CommandBuffer::Command* command,
          command_buffer->CreateDnnGraphCommand(
              *graph_->get(), *execute_params.stream,
              absl::Span<se::DeviceAddressBase>(operands), dependencies));
      if (memzero != nullptr) {
        memzero->commands[command] = std::move(memsets);
      }
      return command;
    }
    if (auto* update = std::get_if<RecordUpdate>(&record_action)) {
      if (memzero != nullptr) {
        auto it = memzero->commands.find(update->command);
        if (it == memzero->commands.end()) {
          return Internal("Missing cuDNN output zero fill commands");
        }
        int memset_index = 0;
        for (int i = 0; i < operands.size(); ++i) {
          if (!output_args_[i] || operands[i].size() == 0) {
            continue;
          }
          RETURN_IF_ERROR(command_buffer->UpdateMemset(
              it->second[memset_index++], &operands[i], uint8_t{0},
              /*num_elements=*/operands[i].size()));
        }
      }
      RETURN_IF_ERROR(command_buffer->UpdateDnnGraphCommand(
          update->command, *graph_->get(), *execute_params.stream,
          absl::Span<se::DeviceAddressBase>(operands)));
      return update->command;
    }
    return Internal("Invalid record action");
  }
  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        if (should_memzero_) {
          for (int i = 0; i < operands.size(); ++i) {
            if (output_args_[i]) {
              RETURN_IF_ERROR(
                  stream->MemZero(&operands[i], operands[i].size()));
            }
          }
        }
        return graph_->get()->Execute(
            *stream, absl::Span<se::DeviceAddressBase>(operands),
            execute_params.collective_params->local_device_id.value());
      });
}

absl::StatusOr<ThunkProto> CuDnnThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();
  proto.mutable_cudnn_thunk()->set_fingerprint(fingerprint_);

  for (const ShapedSlice& arg : args_) {
    ASSIGN_OR_RETURN(*proto.mutable_cudnn_thunk()->add_args(), arg.ToProto());
  }
  for (const bool is_output : output_args_) {
    proto.mutable_cudnn_thunk()->add_output_args(is_output);
  }
  proto.mutable_cudnn_thunk()->set_should_memzero(should_memzero_);
  if (sdpa_dropout_seed_.has_value()) {
    proto.mutable_cudnn_thunk()->set_sdpa_dropout_seed(
        static_cast<int64_t>(*sdpa_dropout_seed_));
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<CuDnnThunk>> CuDnnThunk::FromProto(
    ThunkInfo thunk_info, const CudnnThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  std::vector<ShapedSlice> args;
  args.reserve(proto.args_size());
  for (const ShapedSliceProto& arg : proto.args()) {
    ASSIGN_OR_RETURN(args.emplace_back(),
                     ShapedSlice::FromProto(arg, buffer_allocations));
  }
  std::vector<bool> output_args;
  output_args.reserve(proto.output_args_size());
  for (const bool output_arg : proto.output_args()) {
    output_args.push_back(output_arg);
  }
  std::optional<uint64_t> sdpa_dropout_seed;
  if (proto.has_sdpa_dropout_seed()) {
    sdpa_dropout_seed = static_cast<uint64_t>(proto.sdpa_dropout_seed());
  }
  return std::make_unique<CuDnnThunk>(
      proto.fingerprint(), std::move(thunk_info), std::move(args),
      std::move(output_args), proto.should_memzero(), sdpa_dropout_seed);
}

}  // namespace gpu
}  // namespace xla
