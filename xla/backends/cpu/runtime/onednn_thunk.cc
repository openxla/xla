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

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)
#include "xla/backends/cpu/runtime/onednn_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "tsl/platform/statusor.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/onednn_convolution.h"
#include "xla/service/cpu/onednn_layer_norm.h"
#include "xla/service/cpu/onednn_matmul.h"
#include "xla/service/cpu/onednn_memory_util.h"
#include "xla/service/cpu/onednn_softmax.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

static Thunk::Kind DetermineThunkKind(const std::string& target) {
  if (target == "__onednn$convolution") {
    return Thunk::Kind::kOneDnnConvolution;
  } else if (target == "__onednn$layernorm") {
    return Thunk::Kind::kOneDnnLayerNorm;
  } else if (target == "__onednn$matmul") {
    return Thunk::Kind::kOneDnnMatMul;
  } else if (target == "__onednn$matmul_reorder") {
    return Thunk::Kind::kOneDnnMatMulReorder;
  } else if (target == "__onednn$softmax") {
    return Thunk::Kind::kOneDnnSoftmax;
  } else {
    LOG(FATAL) << "Unsupported OneDNN target: " << target;
  }
}

OneDnnThunk::OneDnnThunk(const std::string& custom_call_target, Info info,
                         OpBuffers buffers, absl::string_view config)
    : target_(custom_call_target),
      Thunk(DetermineThunkKind(custom_call_target), std::move(info)),
      op_buffers_(buffers),
      config_(config) {}

absl::StatusOr<std::unique_ptr<OneDnnThunk>> OneDnnThunk::Create(
    const std::string& custom_call_target, Info info, OpBuffers buffers,
    absl::string_view config) {
  return absl::WrapUnique(new OneDnnThunk(custom_call_target, std::move(info),
                                          std::move(buffers), config));
}

tsl::AsyncValueRef<Thunk::ExecuteEvent> OneDnnThunk::Execute(
    const ExecuteParams& params) {
  std::vector<void*> args;
  // Prepare args
  // 1: nargs
  // 2: ExecutableRunOptions
  // 3: Config
  // 4...: Operands

  // nargs
  int64_t nargs_offset = 3;
  int64_t num_operands = op_buffers_.arguments_shapes.size();
  int64_t num_args = nargs_offset + num_operands;
  args.push_back(&num_args);

  // ExecutableRunOptions
  ExecutableRunOptions run_options;
  run_options.set_intra_op_thread_pool(params.intra_op_threadpool);
  args.push_back(&run_options);

  // Config
  args.push_back(config_.data());

  // Operands
  std::vector<MemrefInfoHandler> memref_ptrs(num_operands);
  for (int i = 0; i < num_operands; ++i) {
    const auto& shape = op_buffers_.arguments_shapes[i];
    TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase arg,
                        params.buffer_allocations->GetDeviceAddress(
                            op_buffers_.arguments_buffers[i]));
    memref_ptrs[i] = CreateMemrefFromShape(shape, arg.opaque());
  }
  for (auto& memref_ptr : memref_ptrs) {
    args.push_back(static_cast<void*>(memref_ptr.get()));
  }

  // Prepare result
  const auto& shape = op_buffers_.results_shapes[0];
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase res,
                      params.buffer_allocations->GetDeviceAddress(
                          op_buffers_.results_buffers[0]));
  auto res_memref_ptr = CreateMemrefFromShape(shape, res.opaque());

  // Invoke the oneDNN function based on the target.
  if (target_ == "__onednn$convolution") {
    __xla_cpu_runtime_OneDnnConvolution(res_memref_ptr.get(), args.data());
  } else if (target_ == "__onednn$matmul") {
    if (op_buffers_.is_tuple_result) {
      // Prepare scratch
      const auto& shape = op_buffers_.results_shapes[1];
      TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase scratch,
                          params.buffer_allocations->GetDeviceAddress(
                              op_buffers_.results_buffers[1]));
      auto scratch_memref_ptr = CreateMemrefFromShape(shape, scratch.opaque());
      __xla_cpu_runtime_OneDnnMatMul(res_memref_ptr.get(),
                                     scratch_memref_ptr.get(), args.data());
    } else {
      __xla_cpu_runtime_OneDnnMatMul(res_memref_ptr.get(), nullptr,
                                     args.data());
    }
  } else if (target_ == "__onednn$matmul_reorder") {
    __xla_cpu_runtime_OneDnnMatMulReorder(res_memref_ptr.get(), args.data());
  } else if (target_ == "__onednn$layernorm") {
    __xla_cpu_runtime_OneDnnLayerNorm(res_memref_ptr.get(), args.data());
  } else if (target_ == "__onednn$softmax") {
    __xla_cpu_runtime_OneDnnSoftmax(&run_options, args[nargs_offset],
                                    res_memref_ptr.get(), config_.data());
  }
  return OkExecuteEvent();
}

Thunk::BufferUses OneDnnThunk::buffer_uses() const {
  BufferUses buffer_uses;
  for (const auto& argument : op_buffers_.arguments_buffers) {
    buffer_uses.emplace_back(argument, BufferUse::kRead);
  }
  for (const auto& result : op_buffers_.results_buffers) {
    buffer_uses.emplace_back(result, BufferUse::kWrite);
  }
  return buffer_uses;
}

}  // namespace xla::cpu

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
