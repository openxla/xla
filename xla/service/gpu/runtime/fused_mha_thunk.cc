/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/fused_mha_thunk.h"

#include <memory>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_fused_mha_runner.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/lazy_op_runner.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

FusedMHAThunk::FusedMHAThunk(
    ThunkInfo thunk_info, GpufMHAConfig config,
    BufferAllocation::Slice lhs_bmm1, BufferAllocation::Slice rhs_bmm1,
    BufferAllocation::Slice rhs_bmm2, BufferAllocation::Slice output,
    BufferAllocation::Slice scratch, BufferAllocation::Slice mask,
    BufferAllocation::Slice bias, BufferAllocation::Slice activation,
    BufferAllocation::Slice seqlen_q, BufferAllocation::Slice seqlen_k)
    : Thunk(Kind::kFusedMHA, thunk_info),
      lhs_bmm1_buffer_(lhs_bmm1),
      rhs_bmm1_buffer_(rhs_bmm1),
      rhs_bmm2_buffer_(rhs_bmm2),
      output_buffer_(output),
      scratch_buffer_(scratch),
      bias_buffer_(bias),
      activation_buffer_(activation),
      seqlen_q_buffer_(seqlen_q),
      seqlen_k_buffer_(seqlen_k),
      config_(std::move(config)) {}

FusedMultiHeadedAttentionRunner& FusedMHAThunk::GetOrCreateRunner(
    const stream_executor::Stream* stream) {
  absl::MutexLock lock(&mu_);
  auto it = runner_cache_.find(stream);
  if (it == runner_cache_.end()) {
    it = runner_cache_
             .insert({stream, std::make_unique<FusedMultiHeadedAttentionRunner>(
                                  config_)})
             .first;
  }
  return *it->second;
}

FusedMHAThunkF8::FusedMHAThunkF8(
    ThunkInfo thunk_info, GpufMHAConfig config,
    BufferAllocation::Slice lhs_bmm1, BufferAllocation::Slice rhs_bmm1,
    BufferAllocation::Slice rhs_bmm2, BufferAllocation::Slice descale_q,
    BufferAllocation::Slice descale_k, BufferAllocation::Slice descale_v,
    BufferAllocation::Slice descale_s, BufferAllocation::Slice scale_s,
    BufferAllocation::Slice scale_o, BufferAllocation::Slice output,
    BufferAllocation::Slice amax_s, BufferAllocation::Slice amax_o,
    BufferAllocation::Slice scratch, BufferAllocation::Slice activation)
    : Thunk(Kind::kFusedMHA, thunk_info),
      lhs_bmm1_buffer_(lhs_bmm1),
      rhs_bmm1_buffer_(rhs_bmm1),
      rhs_bmm2_buffer_(rhs_bmm2),
      descale_q_buffer_(descale_q),
      descale_k_buffer_(descale_k),
      descale_v_buffer_(descale_v),
      descale_s_buffer_(descale_s),
      scale_s_buffer_(scale_s),
      scale_o_buffer_(scale_o),
      output_buffer_(output),
      amax_s_buffer_(amax_s),
      amax_o_buffer_(amax_o),
      scratch_buffer_(scratch),
      activation_buffer_(activation),
      config_(std::move(config)) {}

FusedMultiHeadedAttentionF8Runner& FusedMHAThunkF8::GetOrCreateRunner(
    const stream_executor::Stream* stream) {
  absl::MutexLock lock(&mu_);
  auto it = runner_cache_.find(stream);
  if (it == runner_cache_.end()) {
    it = runner_cache_
             .insert(
                 {stream,
                  std::make_unique<FusedMultiHeadedAttentionF8Runner>(config_)})
             .first;
  }
  return *it->second;
}

std::optional<se::DeviceMemoryBase> AssignBufferIfNotNull(
    const BufferAllocations& buffer_allocations,
    BufferAllocation::Slice& slice) {
  return slice.allocation() != nullptr
             ? std::optional<se::DeviceMemoryBase>{buffer_allocations
                                                       .GetDeviceAddress(slice)}
             : std::nullopt;
}

absl::Status FusedMHAThunk::Initialize(const InitializeParams& params) {
  se::dnn::LazyOpRunner<se::dnn::FusedMHAOp>* lazy_runner =
      GetOrCreateRunner(params.stream).AsFusedMHARunner();
  TF_ASSIGN_OR_RETURN(auto config, config_.AsDnnFusedMHAOpConfig());
  return lazy_runner->GetOrCreateRunner(config, params.stream).status();
}

absl::Status FusedMHAThunkF8::Initialize(const InitializeParams& params) {
  se::dnn::LazyOpRunner<se::dnn::FusedMHAF8Op>* lazy_runner =
      GetOrCreateRunner(params.stream).AsFusedMHAF8Runner();
  TF_ASSIGN_OR_RETURN(auto config, config_.AsDnnFusedMHAF8OpConfig());
  return lazy_runner->GetOrCreateRunner(config, params.stream).status();
}

absl::Status FusedMHAThunkF8::ExecuteOnStream(const ExecuteParams& params) {
  const auto& buffer_allocations = *params.buffer_allocations;
  se::DeviceMemoryBase lhs_bmm1_buffer =
      buffer_allocations.GetDeviceAddress(lhs_bmm1_buffer_);
  se::DeviceMemoryBase rhs_bmm1_buffer =
      buffer_allocations.GetDeviceAddress(rhs_bmm1_buffer_);
  se::DeviceMemoryBase rhs_bmm2_buffer =
      buffer_allocations.GetDeviceAddress(rhs_bmm2_buffer_);
  se::DeviceMemoryBase output_buffer =
      buffer_allocations.GetDeviceAddress(output_buffer_);
  se::DeviceMemoryBase scratch_buffer =
      buffer_allocations.GetDeviceAddress(scratch_buffer_);

  se::DeviceMemoryBase descale_q_buffer =
      buffer_allocations.GetDeviceAddress(descale_q_buffer_);
  se::DeviceMemoryBase descale_k_buffer =
      buffer_allocations.GetDeviceAddress(descale_k_buffer_);
  se::DeviceMemoryBase descale_v_buffer =
      buffer_allocations.GetDeviceAddress(descale_v_buffer_);
  se::DeviceMemoryBase descale_s_buffer =
      buffer_allocations.GetDeviceAddress(descale_s_buffer_);
  se::DeviceMemoryBase scale_s_buffer =
      buffer_allocations.GetDeviceAddress(scale_s_buffer_);
  se::DeviceMemoryBase scale_o_buffer =
      buffer_allocations.GetDeviceAddress(scale_o_buffer_);
  se::DeviceMemoryBase amax_s_buffer =
      buffer_allocations.GetDeviceAddress(amax_s_buffer_);
  se::DeviceMemoryBase amax_o_buffer =
      buffer_allocations.GetDeviceAddress(amax_o_buffer_);
  std::optional<se::DeviceMemoryBase> activation_buffer =
      AssignBufferIfNotNull(buffer_allocations, activation_buffer_);

  RunFusedMHAF8Options opts;
  opts.runner_cache = &GetOrCreateRunner(params.stream);
  TF_RETURN_IF_ERROR(RunGpuFMHAF8(
      config_, lhs_bmm1_buffer, rhs_bmm1_buffer, rhs_bmm2_buffer,
      descale_q_buffer, descale_k_buffer, descale_v_buffer, descale_s_buffer,
      scale_s_buffer, scale_o_buffer, amax_s_buffer, amax_o_buffer,
      output_buffer, scratch_buffer, activation_buffer, params.stream, opts));

  if (!params.stream->ok()) {
    return Internal("FusedMHAThunk::ExecuteOnStream failed.");
  }
  return absl::OkStatus();
}

absl::Status FusedMHAThunk::ExecuteOnStream(const ExecuteParams& params) {
  const auto& buffer_allocations = *params.buffer_allocations;
  se::DeviceMemoryBase lhs_bmm1_buffer =
      buffer_allocations.GetDeviceAddress(lhs_bmm1_buffer_);
  se::DeviceMemoryBase rhs_bmm1_buffer =
      buffer_allocations.GetDeviceAddress(rhs_bmm1_buffer_);
  se::DeviceMemoryBase rhs_bmm2_buffer =
      buffer_allocations.GetDeviceAddress(rhs_bmm2_buffer_);
  se::DeviceMemoryBase output_buffer =
      buffer_allocations.GetDeviceAddress(output_buffer_);
  se::DeviceMemoryBase scratch_buffer =
      buffer_allocations.GetDeviceAddress(scratch_buffer_);

  std::optional<se::DeviceMemoryBase> bias_buffer =
      AssignBufferIfNotNull(buffer_allocations, bias_buffer_);
  std::optional<se::DeviceMemoryBase> activation_buffer =
      AssignBufferIfNotNull(buffer_allocations, activation_buffer_);
  std::optional<se::DeviceMemoryBase> seqlen_q_buffer =
      AssignBufferIfNotNull(buffer_allocations, seqlen_q_buffer_);
  std::optional<se::DeviceMemoryBase> seqlen_k_buffer =
      AssignBufferIfNotNull(buffer_allocations, seqlen_k_buffer_);
  RunFusedMHAOptions opts;
  opts.runner_cache = &GetOrCreateRunner(params.stream);
  TF_RETURN_IF_ERROR(RunGpuFMHA(config_, lhs_bmm1_buffer, rhs_bmm1_buffer,
                                rhs_bmm2_buffer, output_buffer, scratch_buffer,
                                bias_buffer, activation_buffer, seqlen_q_buffer,
                                seqlen_k_buffer, params.stream, opts));

  if (!params.stream->ok()) {
    return Internal("FusedMHAThunk::ExecuteOnStream failed.");
  }
  return absl::OkStatus();
}

FusedMHABackwardThunk::FusedMHABackwardThunk(
    ThunkInfo thunk_info, GpufMHABackwardConfig config,
    BufferAllocation::Slice bmm1_grad_gemm1_rhs,
    BufferAllocation::Slice bmm1_grad_gemm2_rhs,
    BufferAllocation::Slice bmm2_grad_gemm1_lhs,
    BufferAllocation::Slice bmm2_grad_gemm2_rhs,
    BufferAllocation::Slice d_output, BufferAllocation::Slice scratch,
    BufferAllocation::Slice d_bmm1_lhs, BufferAllocation::Slice d_bmm1_rhs,
    BufferAllocation::Slice d_bmm2_rhs, BufferAllocation::Slice d_s,
    BufferAllocation::Slice mask, BufferAllocation::Slice d_bias,
    BufferAllocation::Slice fwd_output, BufferAllocation::Slice bias,
    BufferAllocation::Slice seqlen_q, BufferAllocation::Slice seqlen_k)
    : Thunk(Kind::kFusedMHA, thunk_info),
      bmm1_grad_gemm1_rhs_buffer_(bmm1_grad_gemm1_rhs),
      bmm1_grad_gemm2_rhs_buffer_(bmm1_grad_gemm2_rhs),
      bmm2_grad_gemm1_lhs_buffer_(bmm2_grad_gemm1_lhs),
      bmm2_grad_gemm2_rhs_buffer_(bmm2_grad_gemm2_rhs),
      d_output_buffer_(d_output),
      scratch_buffer_(scratch),
      d_bmm1_lhs_buffer_(d_bmm1_lhs),
      d_bmm1_rhs_buffer_(d_bmm1_rhs),
      d_bmm2_rhs_buffer_(d_bmm2_rhs),
      d_s_buffer_(d_s),
      d_bias_buffer_(d_bias),
      fwd_output_buffer_(fwd_output),
      bias_buffer_(bias),
      seqlen_q_buffer_(seqlen_q),
      seqlen_k_buffer_(seqlen_k),
      config_(std::move(config)) {}

FusedMultiHeadedAttentionBackwardRunner&
FusedMHABackwardThunk::GetOrCreateRunner(
    const stream_executor::Stream* stream) {
  absl::MutexLock lock(&mu_);
  auto it = runner_cache_.find(stream);
  if (it == runner_cache_.end()) {
    it = runner_cache_
             .insert({stream,
                      std::make_unique<FusedMultiHeadedAttentionBackwardRunner>(
                          config_)})
             .first;
  }
  return *it->second;
}

absl::Status FusedMHABackwardThunk::Initialize(const InitializeParams& params) {
  se::dnn::LazyOpRunner<se::dnn::FusedMHABackwardOp>* lazy_runner =
      GetOrCreateRunner(params.stream).AsFusedMHABackwardRunner();
  TF_ASSIGN_OR_RETURN(auto config, config_.AsDnnFusedMHABackwardOpConfig());
  return lazy_runner->GetOrCreateRunner(config, params.stream).status();
}

absl::Status FusedMHABackwardThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  const auto& buffer_allocations = *params.buffer_allocations;
  se::DeviceMemoryBase bmm1_grad_gemm1_rhs_buffer =
      buffer_allocations.GetDeviceAddress(bmm1_grad_gemm1_rhs_buffer_);

  se::DeviceMemoryBase bmm1_grad_gemm2_rhs_buffer =
      buffer_allocations.GetDeviceAddress(bmm1_grad_gemm2_rhs_buffer_);

  se::DeviceMemoryBase bmm2_grad_gemm1_lhs_buffer =
      buffer_allocations.GetDeviceAddress(bmm2_grad_gemm1_lhs_buffer_);

  se::DeviceMemoryBase bmm2_grad_gemm2_rhs_buffer =
      buffer_allocations.GetDeviceAddress(bmm2_grad_gemm2_rhs_buffer_);

  se::DeviceMemoryBase d_output_buffer =
      buffer_allocations.GetDeviceAddress(d_output_buffer_);

  se::DeviceMemoryBase scratch_buffer =
      buffer_allocations.GetDeviceAddress(scratch_buffer_);

  se::DeviceMemoryBase d_bmm1_lhs_buffer =
      buffer_allocations.GetDeviceAddress(d_bmm1_lhs_buffer_);

  se::DeviceMemoryBase d_bmm1_rhs_buffer =
      buffer_allocations.GetDeviceAddress(d_bmm1_rhs_buffer_);

  se::DeviceMemoryBase d_bmm2_rhs_buffer =
      buffer_allocations.GetDeviceAddress(d_bmm2_rhs_buffer_);

  std::optional<se::DeviceMemoryBase> d_s_buffer =
      AssignBufferIfNotNull(buffer_allocations, d_s_buffer_);
  std::optional<se::DeviceMemoryBase> d_bias_buffer =
      AssignBufferIfNotNull(buffer_allocations, d_bias_buffer_);
  std::optional<se::DeviceMemoryBase> fwd_output_buffer =
      AssignBufferIfNotNull(buffer_allocations, fwd_output_buffer_);
  std::optional<se::DeviceMemoryBase> bias_buffer =
      AssignBufferIfNotNull(buffer_allocations, bias_buffer_);
  std::optional<se::DeviceMemoryBase> seqlen_q_buffer =
      AssignBufferIfNotNull(buffer_allocations, seqlen_q_buffer_);
  std::optional<se::DeviceMemoryBase> seqlen_k_buffer =
      AssignBufferIfNotNull(buffer_allocations, seqlen_k_buffer_);
  RunFusedMHABackwardOptions opts;

  opts.runner_cache = &GetOrCreateRunner(params.stream);

  TF_RETURN_IF_ERROR(RunGpuFMHABackward(
      config_, bmm1_grad_gemm1_rhs_buffer, bmm1_grad_gemm2_rhs_buffer,
      bmm2_grad_gemm1_lhs_buffer, bmm2_grad_gemm2_rhs_buffer, d_output_buffer,
      scratch_buffer, d_bmm1_lhs_buffer, d_bmm1_rhs_buffer, d_bmm2_rhs_buffer,
      d_s_buffer, d_bias_buffer, fwd_output_buffer, bias_buffer,
      seqlen_q_buffer, seqlen_k_buffer, params.stream, opts));
  if (!params.stream->ok()) {
    return Internal("FusedMHABackwardThunk::ExecuteOnStream failed.");
  }
  return absl::OkStatus();
}

FusedMHABackwardThunkF8::FusedMHABackwardThunkF8(
    ThunkInfo thunk_info, GpufMHABackwardConfig config,
    BufferAllocation::Slice bmm1_grad_gemm1_rhs,
    BufferAllocation::Slice bmm1_grad_gemm2_rhs,
    BufferAllocation::Slice bmm2_grad_gemm1_lhs,
    BufferAllocation::Slice bmm2_grad_gemm2_rhs,
    BufferAllocation::Slice fwd_output, BufferAllocation::Slice d_output,
    BufferAllocation::Slice descale_q, BufferAllocation::Slice descale_k,
    BufferAllocation::Slice descale_v, BufferAllocation::Slice descale_o,
    BufferAllocation::Slice descale_dO, BufferAllocation::Slice descale_s,
    BufferAllocation::Slice descale_dP, BufferAllocation::Slice scale_s,
    BufferAllocation::Slice scale_dQ, BufferAllocation::Slice scale_dK,
    BufferAllocation::Slice scale_dV, BufferAllocation::Slice scale_dP,

    BufferAllocation::Slice d_bmm1_lhs, BufferAllocation::Slice d_bmm1_rhs,
    BufferAllocation::Slice d_bmm2_rhs,

    BufferAllocation::Slice amax_dQ, BufferAllocation::Slice amax_dK,
    BufferAllocation::Slice amax_dV, BufferAllocation::Slice amax_dP,
    BufferAllocation::Slice scratch)
    : Thunk(Kind::kFusedMHA, thunk_info),
      bmm1_grad_gemm1_rhs_buffer_(bmm1_grad_gemm1_rhs),
      bmm1_grad_gemm2_rhs_buffer_(bmm1_grad_gemm2_rhs),
      bmm2_grad_gemm1_lhs_buffer_(bmm2_grad_gemm1_lhs),
      bmm2_grad_gemm2_rhs_buffer_(bmm2_grad_gemm2_rhs),
      d_output_buffer_(d_output),
      scratch_buffer_(scratch),
      d_bmm1_lhs_buffer_(d_bmm1_lhs),
      d_bmm1_rhs_buffer_(d_bmm1_rhs),
      d_bmm2_rhs_buffer_(d_bmm2_rhs),
      fwd_output_buffer_(fwd_output),
      descale_q_buffer_(descale_q),
      descale_k_buffer_(descale_k),
      descale_v_buffer_(descale_v),
      descale_o_buffer_(descale_o),
      descale_dO_buffer_(descale_dO),
      descale_s_buffer_(descale_s),
      descale_dP_buffer_(descale_dP),
      scale_s_buffer_(scale_s),
      scale_dQ_buffer_(scale_dQ),
      scale_dK_buffer_(scale_dK),
      scale_dV_buffer_(scale_dV),
      scale_dP_buffer_(scale_dP),
      amax_dQ_buffer_(amax_dQ),
      amax_dK_buffer_(amax_dK),
      amax_dV_buffer_(amax_dV),
      amax_dP_buffer_(amax_dP),

      config_(std::move(config)) {}

FusedMultiHeadedAttentionBackwardF8Runner&
FusedMHABackwardThunkF8::GetOrCreateRunner(
    const stream_executor::Stream* stream) {
  absl::MutexLock lock(&mu_);
  auto it = runner_cache_.find(stream);
  if (it == runner_cache_.end()) {
    it = runner_cache_
             .insert(
                 {stream,
                  std::make_unique<FusedMultiHeadedAttentionBackwardF8Runner>(
                      config_)})
             .first;
  }
  return *it->second;
}

absl::Status FusedMHABackwardThunkF8::Initialize(
    const InitializeParams& params) {
  se::dnn::LazyOpRunner<se::dnn::FusedMHABackwardF8Op>* lazy_runner =
      GetOrCreateRunner(params.stream).AsFusedMHABackwardF8Runner();
  TF_ASSIGN_OR_RETURN(auto config, config_.AsDnnFusedMHABackwardF8OpConfig());
  return lazy_runner->GetOrCreateRunner(config, params.stream).status();
}

absl::Status FusedMHABackwardThunkF8::ExecuteOnStream(
    const ExecuteParams& params) {
  const auto& buffer_allocations = *params.buffer_allocations;
  se::DeviceMemoryBase bmm1_grad_gemm1_rhs_buffer =
      buffer_allocations.GetDeviceAddress(bmm1_grad_gemm1_rhs_buffer_);

  se::DeviceMemoryBase bmm1_grad_gemm2_rhs_buffer =
      buffer_allocations.GetDeviceAddress(bmm1_grad_gemm2_rhs_buffer_);

  se::DeviceMemoryBase bmm2_grad_gemm1_lhs_buffer =
      buffer_allocations.GetDeviceAddress(bmm2_grad_gemm1_lhs_buffer_);

  se::DeviceMemoryBase bmm2_grad_gemm2_rhs_buffer =
      buffer_allocations.GetDeviceAddress(bmm2_grad_gemm2_rhs_buffer_);
  se::DeviceMemoryBase fwd_output_buffer =
      buffer_allocations.GetDeviceAddress(fwd_output_buffer_);      

  se::DeviceMemoryBase d_output_buffer =
      buffer_allocations.GetDeviceAddress(d_output_buffer_);

  se::DeviceMemoryBase scratch_buffer =
      buffer_allocations.GetDeviceAddress(scratch_buffer_);

  se::DeviceMemoryBase d_bmm1_lhs_buffer =
      buffer_allocations.GetDeviceAddress(d_bmm1_lhs_buffer_);

  se::DeviceMemoryBase d_bmm1_rhs_buffer =
      buffer_allocations.GetDeviceAddress(d_bmm1_rhs_buffer_);

  se::DeviceMemoryBase d_bmm2_rhs_buffer =
      buffer_allocations.GetDeviceAddress(d_bmm2_rhs_buffer_);

  se::DeviceMemoryBase descale_q_buffer =
      buffer_allocations.GetDeviceAddress(descale_q_buffer_);
  se::DeviceMemoryBase descale_k_buffer =
      buffer_allocations.GetDeviceAddress(descale_k_buffer_);
  se::DeviceMemoryBase descale_v_buffer =
      buffer_allocations.GetDeviceAddress(descale_v_buffer_);
  se::DeviceMemoryBase descale_o_buffer =
      buffer_allocations.GetDeviceAddress(descale_o_buffer_);
  se::DeviceMemoryBase descale_dO_buffer =
      buffer_allocations.GetDeviceAddress(descale_dO_buffer_);
  se::DeviceMemoryBase descale_s_buffer =
      buffer_allocations.GetDeviceAddress(descale_s_buffer_);
  se::DeviceMemoryBase descale_dP_buffer =
      buffer_allocations.GetDeviceAddress(descale_dP_buffer_);

  se::DeviceMemoryBase scale_s_buffer =
      buffer_allocations.GetDeviceAddress(scale_s_buffer_);
  se::DeviceMemoryBase scale_dQ_buffer =
      buffer_allocations.GetDeviceAddress(scale_dQ_buffer_);
  se::DeviceMemoryBase scale_dK_buffer =
      buffer_allocations.GetDeviceAddress(scale_dK_buffer_);
  se::DeviceMemoryBase scale_dV_buffer =
      buffer_allocations.GetDeviceAddress(scale_dV_buffer_);
  se::DeviceMemoryBase scale_dP_buffer =
      buffer_allocations.GetDeviceAddress(scale_dP_buffer_);

  se::DeviceMemoryBase amax_dQ_buffer =
      buffer_allocations.GetDeviceAddress(amax_dQ_buffer_);
  se::DeviceMemoryBase amax_dK_buffer =
      buffer_allocations.GetDeviceAddress(amax_dK_buffer_);
  se::DeviceMemoryBase amax_dV_buffer =
      buffer_allocations.GetDeviceAddress(amax_dV_buffer_);
  se::DeviceMemoryBase amax_dP_buffer =
      buffer_allocations.GetDeviceAddress(amax_dP_buffer_);

  RunFusedMHABackwardF8Options opts;

  opts.runner_cache = &GetOrCreateRunner(params.stream);

  TF_RETURN_IF_ERROR(RunGpuFMHABackwardF8(
      config_, bmm1_grad_gemm1_rhs_buffer, bmm1_grad_gemm2_rhs_buffer,
      bmm2_grad_gemm1_lhs_buffer, bmm2_grad_gemm2_rhs_buffer, fwd_output_buffer, d_output_buffer,
      descale_q_buffer, descale_k_buffer, descale_v_buffer, descale_o_buffer,
      descale_dO_buffer, descale_s_buffer, descale_dP_buffer, scale_s_buffer,
      scale_dQ_buffer, scale_dK_buffer, scale_dV_buffer, scale_dP_buffer,
      d_bmm1_lhs_buffer, d_bmm1_rhs_buffer, d_bmm2_rhs_buffer, amax_dQ_buffer,
      amax_dK_buffer, amax_dV_buffer, amax_dP_buffer, scratch_buffer,
      params.stream, opts));
  if (!params.stream->ok()) {
    return Internal("FusedMHABackwardThunkF8::ExecuteOnStream failed.");
  }
  return absl::OkStatus();
}
}  // namespace gpu
}  // namespace xla
