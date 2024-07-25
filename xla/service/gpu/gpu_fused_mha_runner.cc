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

#include "xla/service/gpu/gpu_fused_mha_runner.h"

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "Eigen/Core"  // from @eigen_archive
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/lazy_op_runner.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {
using se::DeviceMemory;
using se::DeviceMemoryBase;
using se::dnn::DataType;
using se::dnn::MatmulTensorDescriptor;
using se::dnn::TensorDescriptor;

template <typename ElementType, typename BiasType, typename OutputType>
absl::Status RunFusedMHA(GpufMHAParams params, se::Stream *stream,
                         RunFusedMHAOptions options,
                         DeviceMemory<ElementType> lhs_bmm1_buffer,
                         DeviceMemory<ElementType> rhs_bmm1_buffer,
                         DeviceMemory<ElementType> rhs_bmm2_buffer,
                         DeviceMemory<OutputType> output_buffer,
                         DeviceMemoryBase bias_buffer,
                         DeviceMemoryBase scratch_memory,
                         DeviceMemoryBase activation_output,
                         DeviceMemoryBase seqlen_q, DeviceMemoryBase seqlen_k) {
  se::dnn::LazyOpRunner<se::dnn::FusedMHAOp> *lazy_runner =
      options.runner_cache->AsFusedMHARunner();
  std::optional<se::dnn::LazyOpRunner<se::dnn::FusedMHAOp>> local_runner;
  if (!lazy_runner) {
    local_runner.emplace(params.config->algorithm);
    lazy_runner = &*local_runner;
  }
  std::optional<double> dropout_rate;
  if (params.config->dropout_rate) {
    dropout_rate = *params.config->dropout_rate;
  }

  std::optional<int64_t> seed;
  if (params.config->seed) {
    seed = *params.config->seed;
  }

  TF_ASSIGN_OR_RETURN(se::dnn::FusedMHAOp::Config config,
                      params.config->AsDnnFusedMHAOpConfig());
  TF_ASSIGN_OR_RETURN(auto *runner,
                      lazy_runner->GetOrCreateRunner(config, stream));
  return (*runner)(stream, options.profile_result, scratch_memory,
                   lhs_bmm1_buffer, rhs_bmm1_buffer, rhs_bmm2_buffer,
                   output_buffer, bias_buffer, activation_output, seqlen_q,
                   seqlen_k);
}

template <typename ElementType, typename OutputType>
absl::Status RunFusedMHAF8(
    GpufMHAF8Params params, se::Stream *stream, RunFusedMHAF8Options options,
    DeviceMemory<ElementType> lhs_bmm1_buffer,
    DeviceMemory<ElementType> rhs_bmm1_buffer,
    DeviceMemory<ElementType> rhs_bmm2_buffer,
    DeviceMemory<float> descale_q_buffer, DeviceMemory<float> descale_k_buffer,
    DeviceMemory<float> descale_v_buffer, DeviceMemory<float> descale_s_buffer,
    DeviceMemory<float> scale_s_buffer, DeviceMemory<float> scale_o_buffer,
    DeviceMemory<float> amax_s_buffer, DeviceMemory<float> amax_o_buffer,
    DeviceMemory<OutputType> output_buffer, DeviceMemoryBase scratch_memory,
    DeviceMemoryBase activation_output) {
  se::dnn::LazyOpRunner<se::dnn::FusedMHAF8Op> *lazy_runner =
      options.runner_cache->AsFusedMHAF8Runner();
  std::optional<se::dnn::LazyOpRunner<se::dnn::FusedMHAF8Op>> local_runner;
  if (!lazy_runner) {
    local_runner.emplace(params.config->algorithm);
    lazy_runner = &*local_runner;
  }

  TF_ASSIGN_OR_RETURN(se::dnn::FusedMHAF8Op::Config config,
                      params.config->AsDnnFusedMHAF8OpConfig());
  TF_ASSIGN_OR_RETURN(auto *runner,
                      lazy_runner->GetOrCreateRunner(config, stream));
  return (*runner)(
      stream, options.profile_result, scratch_memory, lhs_bmm1_buffer,
      rhs_bmm1_buffer, rhs_bmm2_buffer, descale_q_buffer, descale_k_buffer,
      descale_v_buffer, descale_s_buffer, scale_s_buffer, scale_o_buffer,
      amax_s_buffer, amax_o_buffer, output_buffer, activation_output);
}

template <typename ElementType, typename OutputType>
absl::Status RunGpuFMHAF8Impl(const GpufMHAF8Params &params, se::Stream *stream,
                              se::DeviceMemoryBase scratch_memory,
                              RunFusedMHAF8Options options) {
  auto lhs_bmm1_buffer = se::DeviceMemory<ElementType>(params.lhs_bmm1_buffer);
  auto rhs_bmm1_buffer = se::DeviceMemory<ElementType>(params.rhs_bmm1_buffer);
  auto rhs_bmm2_buffer = se::DeviceMemory<ElementType>(params.rhs_bmm2_buffer);
  auto descale_q_buffer = se::DeviceMemory<float>(params.descale_q_buffer);
  auto descale_k_buffer = se::DeviceMemory<float>(params.descale_k_buffer);
  auto descale_v_buffer = se::DeviceMemory<float>(params.descale_v_buffer);
  auto descale_s_buffer = se::DeviceMemory<float>(params.descale_s_buffer);
  auto scale_s_buffer = se::DeviceMemory<float>(params.scale_s_buffer);
  auto scale_o_buffer = se::DeviceMemory<float>(params.scale_o_buffer);
  auto amax_s_buffer = se::DeviceMemory<float>(params.amax_s_buffer);
  auto amax_o_buffer = se::DeviceMemory<float>(params.amax_o_buffer);
  auto output_buffer = se::DeviceMemory<OutputType>(params.output_buffer);

  auto activation_buffer =
      params.activation_buffer.has_value()
          ? se::DeviceMemory<OutputType>(*params.activation_buffer)
          : se::DeviceMemoryBase();

  se::dnn::AlgorithmDesc algorithm = params.config->algorithm;
  if (options.runner_cache) {
    algorithm = options.runner_cache->ToAlgorithmDesc();
  }

  absl::Status run_status = absl::OkStatus();
  switch (params.config->kind) {
    case CudnnfMHAKind::kSoftmaxf8:
      run_status = RunFusedMHAF8<ElementType, OutputType>(
          params, stream, options, lhs_bmm1_buffer, rhs_bmm1_buffer,
          rhs_bmm2_buffer, descale_q_buffer, descale_k_buffer, descale_v_buffer,
          descale_s_buffer, scale_s_buffer, scale_o_buffer, amax_s_buffer,
          amax_o_buffer, output_buffer, scratch_memory, activation_buffer);
      break;
    default:
      return Internal("Invalid cuDNN fMHA f8 kind");
  }

  if (!run_status.ok()) {
    return run_status;
  }

  if (!stream->ok()) {
    return Internal("Unable to launch FMHA F8 with type %s and algorithm %s",
                    CudnnfMHAKindToString(params.config->kind),
                    algorithm.ToString());
  }

  return absl::OkStatus();
}

template <typename ElementType, typename OutputType>
absl::Status RunFusedMHABackwardF8(
    GpufMHABackwardF8Params params, se::Stream *stream,
    RunFusedMHABackwardF8Options options,
    DeviceMemory<ElementType> bmm1_grad_gemm1_rhs_buffer,
    DeviceMemory<ElementType> bmm1_grad_gemm2_rhs_buffer,
    DeviceMemory<ElementType> bmm2_grad_gemm2_rhs_buffer,
    DeviceMemoryBase fwd_output_buffer,
    DeviceMemory<ElementType> d_output_buffer,
    DeviceMemory<ElementType> bmm2_grad_gemm1_lhs_buffer,
    DeviceMemory<float> descale_q_buffer, DeviceMemory<float> descale_k_buffer,
    DeviceMemory<float> descale_v_buffer, DeviceMemory<float> descale_o_buffer,
    DeviceMemory<float> descale_dO_buffer, DeviceMemory<float> descale_s_buffer,
    DeviceMemory<float> descale_dP_buffer, DeviceMemory<float> scale_s_buffer,
    DeviceMemory<float> scale_dQ_buffer, DeviceMemory<float> scale_dK_buffer,
    DeviceMemory<float> scale_dV_buffer, DeviceMemory<float> scale_dP_buffer,
    DeviceMemory<OutputType> d_bmm1_lhs_buffer,
    DeviceMemory<OutputType> d_bmm1_rhs_buffer,
    DeviceMemory<OutputType> d_bmm2_rhs_buffer,
    DeviceMemory<float> amax_dQ_buffer, DeviceMemory<float> amax_dK_buffer,
    DeviceMemory<float> amax_dV_buffer, DeviceMemory<float> amax_dP_buffer,
    DeviceMemoryBase scratch_memory) {
  se::dnn::LazyOpRunner<se::dnn::FusedMHABackwardF8Op> *lazy_runner =
      options.runner_cache->AsFusedMHABackwardF8Runner();
  std::optional<se::dnn::LazyOpRunner<se::dnn::FusedMHABackwardF8Op>>
      local_runner;
  if (!lazy_runner) {
    local_runner.emplace(params.config->algorithm);
    lazy_runner = &*local_runner;
  }

  TF_ASSIGN_OR_RETURN(se::dnn::FusedMHABackwardF8Op::Config config,
                      params.config->AsDnnFusedMHABackwardF8OpConfig());
  TF_ASSIGN_OR_RETURN(auto *runner,
                      lazy_runner->GetOrCreateRunner(config, stream));
  // TODO: pass in real softmax_sum, dQ_accum, fwd_output
  return (*runner)(
      stream, options.profile_result, scratch_memory,
      bmm1_grad_gemm1_rhs_buffer, bmm1_grad_gemm2_rhs_buffer,
      bmm2_grad_gemm1_lhs_buffer, fwd_output_buffer, d_output_buffer,
      bmm2_grad_gemm2_rhs_buffer, descale_q_buffer, descale_k_buffer,
      descale_v_buffer, descale_o_buffer, descale_dO_buffer, descale_s_buffer,
      descale_dP_buffer, scale_s_buffer, scale_dQ_buffer, scale_dK_buffer,
      scale_dV_buffer, scale_dP_buffer, d_bmm1_lhs_buffer, d_bmm1_rhs_buffer,
      d_bmm2_rhs_buffer, amax_dQ_buffer, amax_dK_buffer, amax_dV_buffer,
      amax_dP_buffer);
  return absl::OkStatus();
}

template <typename ElementType, typename BiasType, typename OutputType>
absl::Status RunGpuFMHAImpl(const GpufMHAParams &params, se::Stream *stream,
                            se::DeviceMemoryBase scratch_memory,
                            RunFusedMHAOptions options) {
  auto lhs_bmm1_buffer = se::DeviceMemory<ElementType>(params.lhs_bmm1_buffer);
  auto rhs_bmm1_buffer = se::DeviceMemory<ElementType>(params.rhs_bmm1_buffer);
  auto rhs_bmm2_buffer = se::DeviceMemory<ElementType>(params.rhs_bmm2_buffer);
  auto output_buffer = se::DeviceMemory<OutputType>(params.output_buffer);
  auto activation_buffer =
      params.activation_buffer.has_value()
          ? se::DeviceMemory<OutputType>(*params.activation_buffer)
          : se::DeviceMemoryBase();
  auto bias_buffer = params.bias_buffer.has_value()
                         ? se::DeviceMemory<BiasType>(*params.bias_buffer)
                         : se::DeviceMemoryBase();
  auto seqlen_q_buffer =
      params.seqlen_q_buffer.has_value()
          ? se::DeviceMemory<BiasType>(*params.seqlen_q_buffer)
          : se::DeviceMemoryBase();
  auto seqlen_k_buffer =
      params.seqlen_k_buffer.has_value()
          ? se::DeviceMemory<BiasType>(*params.seqlen_k_buffer)
          : se::DeviceMemoryBase();
  se::dnn::AlgorithmDesc algorithm = params.config->algorithm;
  if (options.runner_cache) {
    algorithm = options.runner_cache->ToAlgorithmDesc();
  }

  absl::Status run_status = absl::OkStatus();
  switch (params.config->kind) {
    case CudnnfMHAKind::kSoftmaxDropout:
    case CudnnfMHAKind::kSoftmax:
    case CudnnfMHAKind::kScaleBiasSoftmax:
    case CudnnfMHAKind::kScaleBiasSoftmaxDropout:
      run_status = RunFusedMHA<ElementType, BiasType, OutputType>(
          params, stream, options, lhs_bmm1_buffer, rhs_bmm1_buffer,
          rhs_bmm2_buffer, output_buffer, bias_buffer, scratch_memory,
          activation_buffer, seqlen_q_buffer, seqlen_k_buffer);
      break;
    default:
      return Internal("Invalid cuDNN fMHA kind");
  }

  if (!run_status.ok()) {
    return run_status;
  }

  if (!stream->ok()) {
    return Internal("Unable to launch FMHA with type %s and algorithm %s",
                    CudnnfMHAKindToString(params.config->kind),
                    algorithm.ToString());
  }

  return absl::OkStatus();
}

template <typename ElementType, typename OutputType>
absl::Status RunFusedMHABackward(
    GpufMHABackwardParams params, se::Stream *stream,
    RunFusedMHABackwardOptions options,
    DeviceMemory<ElementType> bmm1_grad_gemm1_rhs_buffer,
    DeviceMemory<ElementType> bmm1_grad_gemm2_rhs_buffer,
    DeviceMemory<ElementType> bmm2_grad_gemm1_lhs_buffer,
    DeviceMemory<ElementType> bmm2_grad_gemm2_rhs_buffer,
    DeviceMemory<ElementType> d_output_buffer,
    DeviceMemory<OutputType> d_bmm1_lhs_buffer,
    DeviceMemory<OutputType> d_bmm1_rhs_buffer,
    DeviceMemory<OutputType> d_bmm2_rhs_buffer, DeviceMemoryBase d_s_buffer,
    DeviceMemoryBase d_bias_buffer, DeviceMemoryBase fwd_output_buffer,
    DeviceMemoryBase bias_buffer, DeviceMemoryBase scratch_memory,
    DeviceMemoryBase seqlen_q, DeviceMemoryBase seqlen_k) {
  se::dnn::LazyOpRunner<se::dnn::FusedMHABackwardOp> *lazy_runner =
      options.runner_cache->AsFusedMHABackwardRunner();
  std::optional<se::dnn::LazyOpRunner<se::dnn::FusedMHABackwardOp>>
      local_runner;
  if (!lazy_runner) {
    local_runner.emplace(params.config->algorithm);
    lazy_runner = &*local_runner;
  }
  std::optional<double> dropout_rate;
  if (params.config->dropout_rate) {
    dropout_rate = *params.config->dropout_rate;
  }

  std::optional<int64_t> seed;
  if (params.config->seed) {
    seed = *params.config->seed;
  }

  TF_ASSIGN_OR_RETURN(se::dnn::FusedMHABackwardOp::Config config,
                      params.config->AsDnnFusedMHABackwardOpConfig());
  TF_ASSIGN_OR_RETURN(auto *runner,
                      lazy_runner->GetOrCreateRunner(config, stream));
  // TODO: pass in real softmax_sum, dQ_accum, fwd_output
  return (*runner)(stream, options.profile_result, scratch_memory,
                   bmm1_grad_gemm1_rhs_buffer, bmm1_grad_gemm2_rhs_buffer,
                   bmm2_grad_gemm1_lhs_buffer, bmm2_grad_gemm2_rhs_buffer,
                   d_output_buffer, d_bmm1_lhs_buffer, d_bmm1_rhs_buffer,
                   d_bmm2_rhs_buffer, d_s_buffer, d_bias_buffer,
                   fwd_output_buffer, bias_buffer, seqlen_q, seqlen_k);
  return absl::OkStatus();
}

template <typename ElementType, typename BiasType, typename OutputType>
absl::Status RunGpuFMHABackwardImpl(const GpufMHABackwardParams &params,
                                    se::Stream *stream,
                                    se::DeviceMemoryBase scratch_memory,
                                    RunFusedMHABackwardOptions options) {
  auto bmm1_grad_gemm1_rhs_buffer =
      se::DeviceMemory<ElementType>(params.bmm1_grad_gemm1_rhs_buffer);
  auto bmm1_grad_gemm2_rhs_buffer =
      se::DeviceMemory<ElementType>(params.bmm1_grad_gemm2_rhs_buffer);
  auto bmm2_grad_gemm1_lhs_buffer =
      se::DeviceMemory<ElementType>(params.bmm2_grad_gemm1_lhs_buffer);
  auto bmm2_grad_gemm2_rhs_buffer =
      se::DeviceMemory<ElementType>(params.bmm2_grad_gemm2_rhs_buffer);
  auto d_output_buffer = se::DeviceMemory<ElementType>(params.d_output_buffer);
  auto d_bmm1_lhs_buffer =
      se::DeviceMemory<OutputType>(params.d_bmm1_lhs_buffer);
  auto d_bmm1_rhs_buffer =
      se::DeviceMemory<OutputType>(params.d_bmm1_rhs_buffer);
  auto d_bmm2_rhs_buffer =
      se::DeviceMemory<OutputType>(params.d_bmm2_rhs_buffer);

  // optional buffers
  auto d_s_buffer = params.d_s_buffer.has_value()
                        ? se::DeviceMemory<OutputType>(*params.d_s_buffer)
                        : se::DeviceMemoryBase();

  auto d_bias_buffer = params.d_bias_buffer.has_value()
                           ? se::DeviceMemory<OutputType>(*params.d_bias_buffer)
                           : se::DeviceMemoryBase();

  auto fwd_output_buffer =
      params.fwd_output_buffer.has_value()
          ? se::DeviceMemory<ElementType>(*params.fwd_output_buffer)
          : se::DeviceMemoryBase();

  auto bias_buffer = params.bias_buffer.has_value()
                         ? se::DeviceMemory<BiasType>(*params.bias_buffer)
                         : se::DeviceMemoryBase();

  auto seqlen_q_buffer =
      params.seqlen_q_buffer.has_value()
          ? se::DeviceMemory<BiasType>(*params.seqlen_q_buffer)
          : se::DeviceMemoryBase();

  auto seqlen_k_buffer =
      params.seqlen_k_buffer.has_value()
          ? se::DeviceMemory<BiasType>(*params.seqlen_k_buffer)
          : se::DeviceMemoryBase();

  se::dnn::AlgorithmDesc algorithm = params.config->algorithm;
  if (options.runner_cache) {
    algorithm = options.runner_cache->ToAlgorithmDesc();
  }

  absl::Status run_status = absl::OkStatus();
  switch (params.config->kind) {
    case CudnnfMHAKind::kBackwardSoftmaxDropout:
    case CudnnfMHAKind::kBackwardSoftmax:
    case CudnnfMHAKind::kBackwardScaleBiasSoftmax:
    case CudnnfMHAKind::kBackwardScaleBiasSoftmaxDropout:
      run_status = RunFusedMHABackward<ElementType, OutputType>(
          params, stream, options, bmm1_grad_gemm1_rhs_buffer,
          bmm1_grad_gemm2_rhs_buffer, bmm2_grad_gemm1_lhs_buffer,
          bmm2_grad_gemm2_rhs_buffer, d_output_buffer, d_bmm1_lhs_buffer,
          d_bmm1_rhs_buffer, d_bmm2_rhs_buffer, d_s_buffer, d_bias_buffer,
          fwd_output_buffer, bias_buffer, scratch_memory, seqlen_q_buffer,
          seqlen_k_buffer);
      break;
    default:
      return Internal("Invalid cuDNN fMHA kind");
  }

  if (!run_status.ok()) {
    return run_status;
  }

  if (!stream->ok()) {
    return Internal("Unable to launch FMHA with type %s and algorithm %s",
                    CudnnfMHAKindToString(params.config->kind),
                    algorithm.ToString());
  }

  return run_status;
}

template <typename ElementType, typename OutputType>
absl::Status RunGpuFMHABackwardF8Impl(const GpufMHABackwardF8Params &params,
                                      se::Stream *stream,
                                      se::DeviceMemoryBase scratch_memory,
                                      RunFusedMHABackwardF8Options options) {
  auto bmm1_grad_gemm1_rhs_buffer =
      se::DeviceMemory<ElementType>(params.bmm1_grad_gemm1_rhs_buffer);
  auto bmm1_grad_gemm2_rhs_buffer =
      se::DeviceMemory<ElementType>(params.bmm1_grad_gemm2_rhs_buffer);
  auto bmm2_grad_gemm2_rhs_buffer =
      se::DeviceMemory<ElementType>(params.bmm2_grad_gemm2_rhs_buffer);
  auto bmm2_grad_gemm1_lhs_buffer =
      se::DeviceMemory<ElementType>(params.bmm2_grad_gemm1_lhs_buffer);
  auto d_output_buffer = se::DeviceMemory<ElementType>(params.d_output_buffer);
  auto d_bmm1_lhs_buffer =
      se::DeviceMemory<OutputType>(params.d_bmm1_lhs_buffer);
  auto d_bmm1_rhs_buffer =
      se::DeviceMemory<OutputType>(params.d_bmm1_rhs_buffer);
  auto d_bmm2_rhs_buffer =
      se::DeviceMemory<OutputType>(params.d_bmm2_rhs_buffer);
  auto fwd_output_buffer =
      params.fwd_output_buffer.has_value()
          ? se::DeviceMemory<ElementType>(*params.fwd_output_buffer)
          : se::DeviceMemoryBase();

  auto descale_q_buffer = se::DeviceMemory<float>(params.descale_q_buffer);
  auto descale_k_buffer = se::DeviceMemory<float>(params.descale_k_buffer);
  auto descale_v_buffer = se::DeviceMemory<float>(params.descale_v_buffer);
  auto descale_o_buffer = se::DeviceMemory<float>(params.descale_o_buffer);
  auto descale_dO_buffer = se::DeviceMemory<float>(params.descale_dO_buffer);
  auto descale_s_buffer = se::DeviceMemory<float>(params.descale_s_buffer);
  auto descale_dP_buffer = se::DeviceMemory<float>(params.descale_dP_buffer);

  auto scale_s_buffer = se::DeviceMemory<float>(params.scale_s_buffer);
  auto scale_dQ_buffer = se::DeviceMemory<float>(params.scale_dQ_buffer);
  auto scale_dK_buffer = se::DeviceMemory<float>(params.scale_dK_buffer);
  auto scale_dV_buffer = se::DeviceMemory<float>(params.scale_dV_buffer);
  auto scale_dP_buffer = se::DeviceMemory<float>(params.scale_dP_buffer);

  auto amax_dQ_buffer = se::DeviceMemory<float>(params.amax_dQ_buffer);
  auto amax_dK_buffer = se::DeviceMemory<float>(params.amax_dK_buffer);
  auto amax_dV_buffer = se::DeviceMemory<float>(params.amax_dV_buffer);
  auto amax_dP_buffer = se::DeviceMemory<float>(params.amax_dP_buffer);

  se::dnn::AlgorithmDesc algorithm = params.config->algorithm;
  if (options.runner_cache) {
    algorithm = options.runner_cache->ToAlgorithmDesc();
  }

  absl::Status run_status = absl::OkStatus();
  switch (params.config->kind) {
    case CudnnfMHAKind::kBackwardSoftmaxf8:
      run_status = RunFusedMHABackwardF8<ElementType, OutputType>(
          params, stream, options, bmm1_grad_gemm1_rhs_buffer,
          bmm1_grad_gemm2_rhs_buffer, bmm2_grad_gemm2_rhs_buffer,
          fwd_output_buffer, d_output_buffer, bmm2_grad_gemm1_lhs_buffer,
          descale_q_buffer, descale_k_buffer, descale_v_buffer,
          descale_o_buffer, descale_dO_buffer, descale_s_buffer,
          descale_dP_buffer, scale_s_buffer, scale_dQ_buffer, scale_dK_buffer,
          scale_dV_buffer, scale_dP_buffer, d_bmm1_lhs_buffer,
          d_bmm1_rhs_buffer, d_bmm2_rhs_buffer, amax_dQ_buffer, amax_dK_buffer,
          amax_dV_buffer, amax_dP_buffer, scratch_memory);
      break;
    default:
      return Internal("Invalid cuDNN fMHA kind");
  }

  if (!run_status.ok()) {
    return run_status;
  }

  if (!stream->ok()) {
    return Internal("Unable to launch FMHA with type %s and algorithm %s",
                    CudnnfMHAKindToString(params.config->kind),
                    algorithm.ToString());
  }

  return run_status;
}
}  // namespace

// F8
/*static*/ absl::StatusOr<GpufMHAConfig> GpufMHAConfig::For(
    const GpufMHAF8Descriptor &desc) {
  // Get shapes from desc.
  const Shape &lhs_bmm1_shape = desc.lhs_bmm1_shape;
  const Shape &rhs_bmm1_shape = desc.rhs_bmm1_shape;
  const Shape &rhs_bmm2_shape = desc.rhs_bmm2_shape;
  const Shape &intermediate_lhs_bmm2_shape = desc.intermediate_lhs_bmm2_shape;
  const Shape &output_shape = desc.output_shapes[0];

  // Get DNN dtype from primtive types
  TF_ASSIGN_OR_RETURN(
      DataType lhs_bmm1_type,
      GetDNNDataTypeFromPrimitiveType(lhs_bmm1_shape.element_type()));
  TF_ASSIGN_OR_RETURN(
      DataType rhs_bmm1_type,
      GetDNNDataTypeFromPrimitiveType(rhs_bmm1_shape.element_type()));

  TF_ASSIGN_OR_RETURN(
      DataType rhs_bmm2_type,
      GetDNNDataTypeFromPrimitiveType(rhs_bmm2_shape.element_type()));
  TF_ASSIGN_OR_RETURN(DataType lhs_bmm2_type,
                      GetDNNDataTypeFromPrimitiveType(
                          intermediate_lhs_bmm2_shape.element_type()));
  TF_ASSIGN_OR_RETURN(DataType output_type, GetDNNDataTypeFromPrimitiveType(
                                                output_shape.element_type()));
  GpufMHAConfig config;
  config.input_type = lhs_bmm1_shape.element_type();
  config.output_type = output_shape.element_type();

  // Get MatmulTensorDescriptors for BMM1
  config.lhs_bmm1 =
      MatmulTensorDescriptor::For(lhs_bmm1_type, lhs_bmm1_shape.dimensions(),
                                  desc.lhs_bmm1_shape.layout().minor_to_major(),
                                  desc.bmm1_dnums.lhs_batch_dimensions(),
                                  desc.bmm1_dnums.lhs_contracting_dimensions());
  config.rhs_bmm1 =
      MatmulTensorDescriptor::For(rhs_bmm1_type, rhs_bmm1_shape.dimensions(),
                                  desc.rhs_bmm1_shape.layout().minor_to_major(),
                                  desc.bmm1_dnums.rhs_batch_dimensions(),
                                  desc.bmm1_dnums.rhs_contracting_dimensions());

  // Get MatmulTensorDescriptors for BMM2
  config.rhs_bmm2 =
      MatmulTensorDescriptor::For(rhs_bmm2_type, rhs_bmm2_shape.dimensions(),
                                  desc.rhs_bmm2_shape.layout().minor_to_major(),
                                  desc.bmm2_dnums.rhs_batch_dimensions(),
                                  desc.bmm2_dnums.rhs_contracting_dimensions());

  config.intermediate_lhs_bmm2 = MatmulTensorDescriptor::For(
      lhs_bmm2_type, intermediate_lhs_bmm2_shape.dimensions(),
      desc.intermediate_lhs_bmm2_shape.layout().minor_to_major(),
      desc.bmm2_dnums.lhs_batch_dimensions(),
      desc.bmm2_dnums.lhs_contracting_dimensions());

  config.output = TensorDescriptor::For(output_type, output_shape.dimensions(),
                                        output_shape.layout().minor_to_major());

  if (desc.output_shapes.size() > 3) {
    const Shape &activation_shape = desc.output_shapes.back();
    // Generally, activation should have same type as output, but set it
    // explicityly just to be safe.
    TF_ASSIGN_OR_RETURN(
        DataType activation_type,
        GetDNNDataTypeFromPrimitiveType(activation_shape.element_type()));
    config.activation =
        TensorDescriptor::For(activation_type, activation_shape.dimensions(),
                              activation_shape.layout().minor_to_major());
  }

  config.kind = desc.kind;
  config.mask_type = desc.mask_type;
  const CudnnfMHABackendConfig &backend_config = desc.backend_config;
  config.algorithm = se::dnn::AlgorithmDesc(backend_config.algorithm());
  config.fmha_scale.emplace(backend_config.fmha_scale());
  return config;
}

/*static*/ absl::StatusOr<GpufMHAConfig> GpufMHAConfig::For(
    const GpufMHADescriptor &desc) {
  // Get shapes from desc.
  const Shape &lhs_bmm1_shape = desc.lhs_bmm1_shape;
  const Shape &rhs_bmm1_shape = desc.rhs_bmm1_shape;
  const Shape &rhs_bmm2_shape = desc.rhs_bmm2_shape;
  const Shape &intermediate_lhs_bmm2_shape = desc.intermediate_lhs_bmm2_shape;
  const Shape &output_shape = desc.output_shapes[0];

  // Get DNN dtype from primtive types
  TF_ASSIGN_OR_RETURN(
      DataType lhs_bmm1_type,
      GetDNNDataTypeFromPrimitiveType(lhs_bmm1_shape.element_type()));
  TF_ASSIGN_OR_RETURN(
      DataType rhs_bmm1_type,
      GetDNNDataTypeFromPrimitiveType(rhs_bmm1_shape.element_type()));

  TF_ASSIGN_OR_RETURN(
      DataType rhs_bmm2_type,
      GetDNNDataTypeFromPrimitiveType(rhs_bmm2_shape.element_type()));
  TF_ASSIGN_OR_RETURN(DataType lhs_bmm2_type,
                      GetDNNDataTypeFromPrimitiveType(
                          intermediate_lhs_bmm2_shape.element_type()));
  TF_ASSIGN_OR_RETURN(DataType output_type, GetDNNDataTypeFromPrimitiveType(
                                                output_shape.element_type()));
  GpufMHAConfig config;
  config.input_type = lhs_bmm1_shape.element_type();
  config.output_type = output_shape.element_type();

  // Get MatmulTensorDescriptors for BMM1
  config.lhs_bmm1 =
      MatmulTensorDescriptor::For(lhs_bmm1_type, lhs_bmm1_shape.dimensions(),
                                  desc.lhs_bmm1_shape.layout().minor_to_major(),
                                  desc.bmm1_dnums.lhs_batch_dimensions(),
                                  desc.bmm1_dnums.lhs_contracting_dimensions());
  config.rhs_bmm1 =
      MatmulTensorDescriptor::For(rhs_bmm1_type, rhs_bmm1_shape.dimensions(),
                                  desc.rhs_bmm1_shape.layout().minor_to_major(),
                                  desc.bmm1_dnums.rhs_batch_dimensions(),
                                  desc.bmm1_dnums.rhs_contracting_dimensions());

  // Get MatmulTensorDescriptors for BMM2
  config.rhs_bmm2 =
      MatmulTensorDescriptor::For(rhs_bmm2_type, rhs_bmm2_shape.dimensions(),
                                  desc.rhs_bmm2_shape.layout().minor_to_major(),
                                  desc.bmm2_dnums.rhs_batch_dimensions(),
                                  desc.bmm2_dnums.rhs_contracting_dimensions());

  config.intermediate_lhs_bmm2 = MatmulTensorDescriptor::For(
      lhs_bmm2_type, intermediate_lhs_bmm2_shape.dimensions(),
      desc.intermediate_lhs_bmm2_shape.layout().minor_to_major(),
      desc.bmm2_dnums.lhs_batch_dimensions(),
      desc.bmm2_dnums.lhs_contracting_dimensions());

  config.output = TensorDescriptor::For(output_type, output_shape.dimensions(),
                                        output_shape.layout().minor_to_major());

  if (desc.output_shapes.size() > 1) {
    const Shape &activation_shape = desc.output_shapes.back();
    // Generally, activation should have same type as output, but set it
    // explicityly just to be safe.
    TF_ASSIGN_OR_RETURN(
        DataType activation_type,
        GetDNNDataTypeFromPrimitiveType(activation_shape.element_type()));
    config.activation =
        TensorDescriptor::For(activation_type, activation_shape.dimensions(),
                              activation_shape.layout().minor_to_major());
  }

  if (desc.mask_shape) {
    const Shape &mask_shape = *desc.mask_shape;
    TF_ASSIGN_OR_RETURN(DataType mask_type, GetDNNDataTypeFromPrimitiveType(
                                                mask_shape.element_type()));
    config.mask = TensorDescriptor::For(mask_type, mask_shape.dimensions(),
                                        mask_shape.layout().minor_to_major());
  }

  if (desc.bias_shape) {
    const Shape &bias_shape = *desc.bias_shape;
    TF_ASSIGN_OR_RETURN(DataType bias_type, GetDNNDataTypeFromPrimitiveType(
                                                bias_shape.element_type()));
    config.bias = TensorDescriptor::For(bias_type, bias_shape.dimensions(),
                                        bias_shape.layout().minor_to_major());
  }
  config.kind = desc.kind;
  config.mask_type = desc.mask_type;
  const CudnnfMHABackendConfig &backend_config = desc.backend_config;
  config.algorithm = se::dnn::AlgorithmDesc(backend_config.algorithm());
  config.fmha_scale.emplace(backend_config.fmha_scale());
  config.dropout_rate.emplace(backend_config.dropout_rate());
  config.seed.emplace(backend_config.seed());
  return config;
}

/*static*/ absl::StatusOr<GpufMHABackwardConfig> GpufMHABackwardConfig::For(
    const GpufMHABackwardDescriptor &desc) {
  // Get shapes from desc.

  const Shape &bmm1_grad_gemm1_rhs_shape = desc.bmm1_grad_gemm1_rhs_shape;
  const Shape &bmm1_grad_gemm2_rhs_shape = desc.bmm1_grad_gemm2_rhs_shape;
  const Shape &bmm2_grad_gemm1_lhs_shape = desc.bmm2_grad_gemm1_lhs_shape;
  const Shape &bmm2_grad_gemm2_rhs_shape = desc.bmm2_grad_gemm2_rhs_shape;
  const Shape &d_output_shape = desc.d_output_shape;
  const Shape &d_bmm1_lhs_shape = desc.d_bmm1_lhs_shape;
  const Shape &d_bmm1_rhs_shape = desc.d_bmm1_rhs_shape;
  const Shape &d_bmm2_rhs_shape = desc.d_bmm2_rhs_shape;
  // Get DNN dtype from primtive types
  TF_ASSIGN_OR_RETURN(DataType bmm1_grad_gemm1_rhs_type,
                      GetDNNDataTypeFromPrimitiveType(
                          bmm1_grad_gemm1_rhs_shape.element_type()));

  TF_ASSIGN_OR_RETURN(DataType bmm1_grad_gemm2_rhs_type,
                      GetDNNDataTypeFromPrimitiveType(
                          bmm1_grad_gemm2_rhs_shape.element_type()));

  TF_ASSIGN_OR_RETURN(DataType bmm2_grad_gemm1_lhs_type,
                      GetDNNDataTypeFromPrimitiveType(
                          bmm2_grad_gemm1_lhs_shape.element_type()));

  TF_ASSIGN_OR_RETURN(DataType bmm2_grad_gemm2_rhs_type,
                      GetDNNDataTypeFromPrimitiveType(
                          bmm2_grad_gemm2_rhs_shape.element_type()));

  TF_ASSIGN_OR_RETURN(
      DataType d_output_type,
      GetDNNDataTypeFromPrimitiveType(d_output_shape.element_type()));

  TF_ASSIGN_OR_RETURN(
      DataType d_bmm1_lhs_type,
      GetDNNDataTypeFromPrimitiveType(d_bmm1_lhs_shape.element_type()));

  TF_ASSIGN_OR_RETURN(
      DataType d_bmm1_rhs_type,
      GetDNNDataTypeFromPrimitiveType(d_bmm1_rhs_shape.element_type()));

  TF_ASSIGN_OR_RETURN(
      DataType d_bmm2_rhs_type,
      GetDNNDataTypeFromPrimitiveType(d_bmm2_rhs_shape.element_type()));

  GpufMHABackwardConfig config;
  config.input_type = bmm1_grad_gemm1_rhs_shape.element_type();
  config.output_type = d_bmm1_lhs_shape.element_type();

  // Get MatmulTensorDescriptors for lhs of BMM1 grad GEMM 1
  config.bmm1_grad_gemm1_rhs = MatmulTensorDescriptor::For(
      bmm1_grad_gemm1_rhs_type, bmm1_grad_gemm1_rhs_shape.dimensions(),
      desc.bmm1_grad_gemm1_rhs_shape.layout().minor_to_major(),
      desc.bmm1_grad_gemm1_dnums.rhs_batch_dimensions(),
      desc.bmm1_grad_gemm1_dnums.rhs_contracting_dimensions());

  // Get MatmulTensorDescriptors for rhs of BMM1 grad GEMM 2
  config.bmm1_grad_gemm2_rhs = MatmulTensorDescriptor::For(
      bmm1_grad_gemm2_rhs_type, bmm1_grad_gemm2_rhs_shape.dimensions(),
      desc.bmm1_grad_gemm2_rhs_shape.layout().minor_to_major(),
      desc.bmm1_grad_gemm2_dnums.rhs_batch_dimensions(),
      desc.bmm1_grad_gemm2_dnums.rhs_contracting_dimensions());

  // Get MatmulTensorDescriptors for BMM2 grad GEMM 1
  config.bmm2_grad_gemm1_lhs = MatmulTensorDescriptor::For(
      bmm2_grad_gemm1_lhs_type, bmm2_grad_gemm1_lhs_shape.dimensions(),
      desc.bmm2_grad_gemm1_lhs_shape.layout().minor_to_major(),
      desc.bmm2_grad_gemm1_dnums.lhs_batch_dimensions(),
      desc.bmm2_grad_gemm1_dnums.lhs_contracting_dimensions());

  config.d_output = MatmulTensorDescriptor::For(
      d_output_type, d_output_shape.dimensions(),
      desc.d_output_shape.layout().minor_to_major(),
      desc.bmm2_grad_gemm1_dnums.rhs_batch_dimensions(),
      desc.bmm2_grad_gemm1_dnums.rhs_contracting_dimensions());

  // Get MatmulTensorDescriptors for BMM2 grad GEMM 2
  config.bmm2_grad_gemm2_rhs = MatmulTensorDescriptor::For(
      bmm2_grad_gemm2_rhs_type, bmm2_grad_gemm2_rhs_shape.dimensions(),
      desc.bmm2_grad_gemm2_rhs_shape.layout().minor_to_major(),
      desc.bmm2_grad_gemm2_dnums.rhs_batch_dimensions(),
      desc.bmm2_grad_gemm2_dnums
          .rhs_contracting_dimensions());  // FMHA TODO: transpose here?

  config.d_bmm1_lhs =
      TensorDescriptor::For(d_bmm1_lhs_type, d_bmm1_lhs_shape.dimensions(),
                            d_bmm1_lhs_shape.layout().minor_to_major());
  config.d_bmm1_rhs =
      TensorDescriptor::For(d_bmm1_rhs_type, d_bmm1_rhs_shape.dimensions(),
                            d_bmm1_rhs_shape.layout().minor_to_major());
  config.d_bmm2_rhs =
      TensorDescriptor::For(d_bmm2_rhs_type, d_bmm2_rhs_shape.dimensions(),
                            d_bmm2_rhs_shape.layout().minor_to_major());
  config.d_s = TensorDescriptor::For(
      bmm2_grad_gemm1_lhs_type, bmm2_grad_gemm1_lhs_shape.dimensions(),
      bmm2_grad_gemm1_lhs_shape.layout().minor_to_major());

  if (desc.d_bias_shape) {
    const Shape &d_bias_shape = *desc.d_bias_shape;
    // Get DNN dtype from primtive types
    TF_ASSIGN_OR_RETURN(DataType d_bias_type, GetDNNDataTypeFromPrimitiveType(
                                                  d_bias_shape.element_type()));
    config.d_bias =
        TensorDescriptor::For(d_bias_type, d_bias_shape.dimensions(),
                              d_bias_shape.layout().minor_to_major());
  }

  if (desc.mask_shape) {
    const Shape &mask_shape = *desc.mask_shape;
    TF_ASSIGN_OR_RETURN(DataType mask_type, GetDNNDataTypeFromPrimitiveType(
                                                mask_shape.element_type()));
    config.mask = TensorDescriptor::For(mask_type, mask_shape.dimensions(),
                                        mask_shape.layout().minor_to_major());
  }
  if (desc.fwd_output_shape) {
    const Shape &fwd_output_shape = *desc.fwd_output_shape;
    TF_ASSIGN_OR_RETURN(
        DataType fwd_output_type,
        GetDNNDataTypeFromPrimitiveType(fwd_output_shape.element_type()));
    config.fwd_output =
        TensorDescriptor::For(fwd_output_type, fwd_output_shape.dimensions(),
                              fwd_output_shape.layout().minor_to_major());
  }

  if (desc.bias_shape) {
    const Shape &bias_shape = *desc.bias_shape;
    TF_ASSIGN_OR_RETURN(DataType bias_type, GetDNNDataTypeFromPrimitiveType(
                                                bias_shape.element_type()));
    config.bias = TensorDescriptor::For(bias_type, bias_shape.dimensions(),
                                        bias_shape.layout().minor_to_major());
  }

  config.kind = desc.kind;
  config.mask_type = desc.mask_type;
  config.force_deterministic = desc.force_deterministic;
  const CudnnfMHABackendConfig &backend_config = desc.backend_config;
  config.algorithm = se::dnn::AlgorithmDesc(backend_config.algorithm());
  config.fmha_scale.emplace(backend_config.fmha_scale());
  config.dropout_rate.emplace(backend_config.dropout_rate());
  config.seed.emplace(backend_config.seed());
  return config;
}

/*static*/ absl::StatusOr<GpufMHABackwardConfig> GpufMHABackwardConfig::For(
    const GpufMHABackwardF8Descriptor &desc) {
  // Get shapes from desc.

  const Shape &bmm1_grad_gemm1_rhs_shape = desc.bmm1_grad_gemm1_rhs_shape;
  const Shape &bmm1_grad_gemm2_rhs_shape = desc.bmm1_grad_gemm2_rhs_shape;
  const Shape &bmm2_grad_gemm1_lhs_shape = desc.bmm2_grad_gemm1_lhs_shape;
  const Shape &bmm2_grad_gemm2_rhs_shape = desc.bmm2_grad_gemm2_rhs_shape;
  const Shape &d_output_shape = desc.d_output_shape;
  const Shape &d_bmm1_lhs_shape = desc.d_bmm1_lhs_shape;
  const Shape &d_bmm1_rhs_shape = desc.d_bmm1_rhs_shape;
  const Shape &d_bmm2_rhs_shape = desc.d_bmm2_rhs_shape;
  // Get DNN dtype from primtive types
  TF_ASSIGN_OR_RETURN(DataType bmm1_grad_gemm1_rhs_type,
                      GetDNNDataTypeFromPrimitiveType(
                          bmm1_grad_gemm1_rhs_shape.element_type()));

  TF_ASSIGN_OR_RETURN(DataType bmm1_grad_gemm2_rhs_type,
                      GetDNNDataTypeFromPrimitiveType(
                          bmm1_grad_gemm2_rhs_shape.element_type()));

  TF_ASSIGN_OR_RETURN(DataType bmm2_grad_gemm1_lhs_type,
                      GetDNNDataTypeFromPrimitiveType(
                          bmm2_grad_gemm1_lhs_shape.element_type()));

  TF_ASSIGN_OR_RETURN(DataType bmm2_grad_gemm2_rhs_type,
                      GetDNNDataTypeFromPrimitiveType(
                          bmm2_grad_gemm2_rhs_shape.element_type()));

  TF_ASSIGN_OR_RETURN(
      DataType d_output_type,
      GetDNNDataTypeFromPrimitiveType(d_output_shape.element_type()));

  TF_ASSIGN_OR_RETURN(
      DataType d_bmm1_lhs_type,
      GetDNNDataTypeFromPrimitiveType(d_bmm1_lhs_shape.element_type()));

  TF_ASSIGN_OR_RETURN(
      DataType d_bmm1_rhs_type,
      GetDNNDataTypeFromPrimitiveType(d_bmm1_rhs_shape.element_type()));

  TF_ASSIGN_OR_RETURN(
      DataType d_bmm2_rhs_type,
      GetDNNDataTypeFromPrimitiveType(d_bmm2_rhs_shape.element_type()));

  GpufMHABackwardConfig config;
  config.input_type = bmm1_grad_gemm1_rhs_shape.element_type();
  config.output_type = d_bmm1_lhs_shape.element_type();

  // Get MatmulTensorDescriptors for lhs of BMM1 grad GEMM 1
  config.bmm1_grad_gemm1_rhs = MatmulTensorDescriptor::For(
      bmm1_grad_gemm1_rhs_type, bmm1_grad_gemm1_rhs_shape.dimensions(),
      desc.bmm1_grad_gemm1_rhs_shape.layout().minor_to_major(),
      desc.bmm1_grad_gemm1_dnums.rhs_batch_dimensions(),
      desc.bmm1_grad_gemm1_dnums.rhs_contracting_dimensions());

  // Get MatmulTensorDescriptors for rhs of BMM1 grad GEMM 2
  config.bmm1_grad_gemm2_rhs = MatmulTensorDescriptor::For(
      bmm1_grad_gemm2_rhs_type, bmm1_grad_gemm2_rhs_shape.dimensions(),
      desc.bmm1_grad_gemm2_rhs_shape.layout().minor_to_major(),
      desc.bmm1_grad_gemm2_dnums.rhs_batch_dimensions(),
      desc.bmm1_grad_gemm2_dnums.rhs_contracting_dimensions());

  // Get MatmulTensorDescriptors for BMM2 grad GEMM 1
  config.bmm2_grad_gemm1_lhs = MatmulTensorDescriptor::For(
      bmm2_grad_gemm1_lhs_type, bmm2_grad_gemm1_lhs_shape.dimensions(),
      desc.bmm2_grad_gemm1_lhs_shape.layout().minor_to_major(),
      desc.bmm2_grad_gemm1_dnums.lhs_batch_dimensions(),
      desc.bmm2_grad_gemm1_dnums.lhs_contracting_dimensions());

  config.d_output = MatmulTensorDescriptor::For(
      d_output_type, d_output_shape.dimensions(),
      desc.d_output_shape.layout().minor_to_major(),
      desc.bmm2_grad_gemm1_dnums.rhs_batch_dimensions(),
      desc.bmm2_grad_gemm1_dnums.rhs_contracting_dimensions());

  // Get MatmulTensorDescriptors for BMM2 grad GEMM 2
  config.bmm2_grad_gemm2_rhs = MatmulTensorDescriptor::For(
      bmm2_grad_gemm2_rhs_type, bmm2_grad_gemm2_rhs_shape.dimensions(),
      desc.bmm2_grad_gemm2_rhs_shape.layout().minor_to_major(),
      desc.bmm2_grad_gemm2_dnums.rhs_batch_dimensions(),
      desc.bmm2_grad_gemm2_dnums
          .rhs_contracting_dimensions());  // FMHA TODO: transpose here?

  config.d_bmm1_lhs =
      TensorDescriptor::For(d_bmm1_lhs_type, d_bmm1_lhs_shape.dimensions(),
                            d_bmm1_lhs_shape.layout().minor_to_major());
  config.d_bmm1_rhs =
      TensorDescriptor::For(d_bmm1_rhs_type, d_bmm1_rhs_shape.dimensions(),
                            d_bmm1_rhs_shape.layout().minor_to_major());
  config.d_bmm2_rhs =
      TensorDescriptor::For(d_bmm2_rhs_type, d_bmm2_rhs_shape.dimensions(),
                            d_bmm2_rhs_shape.layout().minor_to_major());
  config.d_s = TensorDescriptor::For(
      bmm2_grad_gemm1_lhs_type, bmm2_grad_gemm1_lhs_shape.dimensions(),
      bmm2_grad_gemm1_lhs_shape.layout().minor_to_major());

  if (desc.fwd_output_shape) {
    const Shape &fwd_output_shape = *desc.fwd_output_shape;
    TF_ASSIGN_OR_RETURN(
        DataType fwd_output_type,
        GetDNNDataTypeFromPrimitiveType(fwd_output_shape.element_type()));
    config.fwd_output =
        TensorDescriptor::For(fwd_output_type, fwd_output_shape.dimensions(),
                              fwd_output_shape.layout().minor_to_major());
  }

  config.kind = desc.kind;
  config.mask_type = desc.mask_type;
  const CudnnfMHABackendConfig &backend_config = desc.backend_config;
  config.algorithm = se::dnn::AlgorithmDesc(backend_config.algorithm());
  config.fmha_scale.emplace(backend_config.fmha_scale());
  return config;
}

absl::StatusOr<se::dnn::FusedMHAOp::Config>
GpufMHAConfig::AsDnnFusedMHAOpConfig() const {
  double scale = 1.0;
  if (fmha_scale.has_value()) {
    scale = *fmha_scale;
  }
  TF_ASSIGN_OR_RETURN(se::dnn::FMHAMaskKind mask_type,
                      GetDNNFmhaMaskKindFromCudnnFmhaMaskKind(mask_type));

  return se::dnn::FusedMHAOp::Config{
      scale,    lhs_bmm1, rhs_bmm1,   rhs_bmm2,     intermediate_lhs_bmm2,
      output,   bias,     activation, dropout_rate, seed,
      mask_type};
}

absl::StatusOr<se::dnn::FusedMHAF8Op::Config>
GpufMHAConfig::AsDnnFusedMHAF8OpConfig() const {
  double scale = 1.0;
  if (fmha_scale.has_value()) {
    scale = *fmha_scale;
  }
  TF_ASSIGN_OR_RETURN(se::dnn::FMHAMaskKind mask_type,
                      GetDNNFmhaMaskKindFromCudnnFmhaMaskKind(mask_type));

  return se::dnn::FusedMHAF8Op::Config{
      scale,  lhs_bmm1,   rhs_bmm1, rhs_bmm2, intermediate_lhs_bmm2,
      output, activation, mask_type};
}

absl::StatusOr<se::dnn::FusedMHABackwardOp::Config>
GpufMHABackwardConfig::AsDnnFusedMHABackwardOpConfig() const {
  double scale = 1.0;
  if (fmha_scale.has_value()) {
    scale = *fmha_scale;
  }
  TF_ASSIGN_OR_RETURN(se::dnn::FMHAMaskKind mask_type,
                      GetDNNFmhaMaskKindFromCudnnFmhaMaskKind(mask_type));

  return se::dnn::FusedMHABackwardOp::Config{scale,
                                             bmm1_grad_gemm1_rhs,
                                             bmm1_grad_gemm2_rhs,
                                             bmm2_grad_gemm1_lhs,
                                             bmm2_grad_gemm2_rhs,
                                             d_output,
                                             d_bmm1_lhs,
                                             d_bmm1_rhs,
                                             d_bmm2_rhs,
                                             d_s,
                                             d_bias,
                                             fwd_output,
                                             bias,
                                             dropout_rate,
                                             seed,
                                             mask_type,
                                             force_deterministic};
}

absl::StatusOr<se::dnn::FusedMHABackwardF8Op::Config>
GpufMHABackwardConfig::AsDnnFusedMHABackwardF8OpConfig() const {
  double scale = 1.0;
  if (fmha_scale.has_value()) {
    scale = *fmha_scale;
  }
  TF_ASSIGN_OR_RETURN(se::dnn::FMHAMaskKind mask_type,
                      GetDNNFmhaMaskKindFromCudnnFmhaMaskKind(mask_type));

  return se::dnn::FusedMHABackwardF8Op::Config{scale,
                                               bmm1_grad_gemm1_rhs,
                                               bmm1_grad_gemm2_rhs,
                                               bmm2_grad_gemm1_lhs,
                                               bmm2_grad_gemm2_rhs,
                                               d_output,
                                               d_bmm1_lhs,
                                               d_bmm1_rhs,
                                               d_bmm2_rhs,
                                               fwd_output,
                                               mask_type};
}

/*static*/ absl::StatusOr<GpufMHAF8Params> GpufMHAF8Params::For(
    const GpufMHAConfig &config, se::DeviceMemoryBase lhs_bmm1_buffer,
    se::DeviceMemoryBase rhs_bmm1_buffer, se::DeviceMemoryBase rhs_bmm2_buffer,
    se::DeviceMemoryBase descale_q_buffer,
    se::DeviceMemoryBase descale_k_buffer,
    se::DeviceMemoryBase descale_v_buffer,
    se::DeviceMemoryBase descale_s_buffer, se::DeviceMemoryBase scale_s_buffer,
    se::DeviceMemoryBase scale_o_buffer, se::DeviceMemoryBase amax_s_buffer,
    se::DeviceMemoryBase amax_o_buffer, se::DeviceMemoryBase output_buffer,
    std::optional<se::DeviceMemoryBase> activation_buffer) {
  GpufMHAF8Params params;
  params.config = &config;
  params.lhs_bmm1_buffer = lhs_bmm1_buffer;
  params.rhs_bmm1_buffer = rhs_bmm1_buffer;
  params.rhs_bmm2_buffer = rhs_bmm2_buffer;
  params.output_buffer = output_buffer;
  params.activation_buffer = activation_buffer;
  params.descale_q_buffer = descale_q_buffer;
  params.descale_k_buffer = descale_k_buffer;
  params.descale_v_buffer = descale_v_buffer;
  params.descale_s_buffer = descale_s_buffer;
  params.scale_s_buffer = scale_s_buffer;
  params.scale_o_buffer = scale_o_buffer;
  params.amax_s_buffer = amax_s_buffer;
  params.amax_o_buffer = amax_o_buffer;
  return params;
}

/*static*/ absl::StatusOr<GpufMHAParams> GpufMHAParams::For(
    const GpufMHAConfig &config, se::DeviceMemoryBase lhs_bmm1_buffer,
    se::DeviceMemoryBase rhs_bmm1_buffer, se::DeviceMemoryBase rhs_bmm2_buffer,
    se::DeviceMemoryBase output_buffer,
    std::optional<se::DeviceMemoryBase> bias_buffer,
    std::optional<se::DeviceMemoryBase> activation_buffer,
    std::optional<se::DeviceMemoryBase> seqlen_q_buffer,
    std::optional<se::DeviceMemoryBase> seqlen_k_buffer) {
  GpufMHAParams params;
  params.config = &config;
  params.lhs_bmm1_buffer = lhs_bmm1_buffer;
  params.rhs_bmm1_buffer = rhs_bmm1_buffer;
  params.rhs_bmm2_buffer = rhs_bmm2_buffer;
  params.output_buffer = output_buffer;
  params.activation_buffer = activation_buffer;
  params.bias_buffer = bias_buffer;
  params.seqlen_q_buffer = seqlen_q_buffer;
  params.seqlen_k_buffer = seqlen_k_buffer;
  return params;
}

/*static*/ absl::StatusOr<GpufMHABackwardParams> GpufMHABackwardParams::For(
    const GpufMHABackwardConfig &config,
    se::DeviceMemoryBase bmm1_grad_gemm1_rhs_buffer,
    se::DeviceMemoryBase bmm1_grad_gemm2_rhs_buffer,
    se::DeviceMemoryBase bmm2_grad_gemm1_lhs_buffer,
    se::DeviceMemoryBase bmm2_grad_gemm2_rhs_buffer,
    se::DeviceMemoryBase d_output_buffer,
    se::DeviceMemoryBase d_bmm1_lhs_buffer,
    se::DeviceMemoryBase d_bmm1_rhs_buffer,
    se::DeviceMemoryBase d_bmm2_rhs_buffer,
    std::optional<se::DeviceMemoryBase> d_s_buffer,
    std::optional<se::DeviceMemoryBase> d_bias_buffer,
    std::optional<se::DeviceMemoryBase> fwd_output_buffer,
    std::optional<se::DeviceMemoryBase> bias_buffer,
    std::optional<se::DeviceMemoryBase> seqlen_q_buffer,
    std::optional<se::DeviceMemoryBase> seqlen_k_buffer) {
  GpufMHABackwardParams params;
  params.config = &config;
  params.bmm1_grad_gemm1_rhs_buffer = bmm1_grad_gemm1_rhs_buffer;
  params.bmm1_grad_gemm2_rhs_buffer = bmm1_grad_gemm2_rhs_buffer;
  params.bmm2_grad_gemm1_lhs_buffer = bmm2_grad_gemm1_lhs_buffer;
  params.bmm2_grad_gemm2_rhs_buffer = bmm2_grad_gemm2_rhs_buffer;
  params.d_output_buffer = d_output_buffer;
  params.d_bmm1_lhs_buffer = d_bmm1_lhs_buffer;
  params.d_bmm1_rhs_buffer = d_bmm1_rhs_buffer;
  params.d_bmm2_rhs_buffer = d_bmm2_rhs_buffer;
  params.d_s_buffer = d_s_buffer;
  params.d_bias_buffer = d_bias_buffer;
  params.fwd_output_buffer = fwd_output_buffer;
  params.bias_buffer = bias_buffer;
  params.seqlen_q_buffer = seqlen_q_buffer;
  params.seqlen_k_buffer = seqlen_k_buffer;
  return params;
}

/*static*/ absl::StatusOr<GpufMHABackwardF8Params> GpufMHABackwardF8Params::For(
    const GpufMHABackwardConfig &config,
    se::DeviceMemoryBase bmm1_grad_gemm1_rhs_buffer,
    se::DeviceMemoryBase bmm1_grad_gemm2_rhs_buffer,
    se::DeviceMemoryBase bmm2_grad_gemm2_rhs_buffer,
    se::DeviceMemoryBase d_output_buffer,
    se::DeviceMemoryBase bmm2_grad_gemm1_lhs_buffer,
    se::DeviceMemoryBase descale_q_buffer,
    se::DeviceMemoryBase descale_k_buffer,
    se::DeviceMemoryBase descale_v_buffer,
    se::DeviceMemoryBase descale_o_buffer,
    se::DeviceMemoryBase descale_dO_buffer,
    se::DeviceMemoryBase descale_s_buffer,
    se::DeviceMemoryBase descale_dP_buffer, se::DeviceMemoryBase scale_s_buffer,
    se::DeviceMemoryBase scale_dQ_buffer, se::DeviceMemoryBase scale_dK_buffer,
    se::DeviceMemoryBase scale_dV_buffer, se::DeviceMemoryBase scale_dP_buffer,
    se::DeviceMemoryBase d_bmm1_lhs_buffer,
    se::DeviceMemoryBase d_bmm1_rhs_buffer,
    se::DeviceMemoryBase d_bmm2_rhs_buffer, se::DeviceMemoryBase amax_dQ_buffer,
    se::DeviceMemoryBase amax_dK_buffer, se::DeviceMemoryBase amax_dV_buffer,
    se::DeviceMemoryBase amax_dP_buffer,
    std::optional<se::DeviceMemoryBase> fwd_output_buffer) {
  GpufMHABackwardF8Params params;
  params.config = &config;
  params.bmm1_grad_gemm1_rhs_buffer = bmm1_grad_gemm1_rhs_buffer;
  params.bmm1_grad_gemm2_rhs_buffer = bmm1_grad_gemm2_rhs_buffer;
  params.bmm2_grad_gemm1_lhs_buffer = bmm2_grad_gemm1_lhs_buffer;
  params.bmm2_grad_gemm2_rhs_buffer = bmm2_grad_gemm2_rhs_buffer;
  params.d_output_buffer = d_output_buffer;
  params.d_bmm1_lhs_buffer = d_bmm1_lhs_buffer;
  params.d_bmm1_rhs_buffer = d_bmm1_rhs_buffer;
  params.d_bmm2_rhs_buffer = d_bmm2_rhs_buffer;
  params.fwd_output_buffer = fwd_output_buffer;

  params.descale_q_buffer = descale_q_buffer;
  params.descale_k_buffer = descale_k_buffer;
  params.descale_v_buffer = descale_v_buffer;
  params.descale_o_buffer = descale_o_buffer;
  params.descale_dO_buffer = descale_dO_buffer;
  params.descale_s_buffer = descale_s_buffer;
  params.descale_dP_buffer = descale_dP_buffer;

  params.scale_s_buffer = scale_s_buffer;
  params.scale_dQ_buffer = scale_dQ_buffer;
  params.scale_dK_buffer = scale_dK_buffer;
  params.scale_dV_buffer = scale_dV_buffer;
  params.scale_dP_buffer = scale_dP_buffer;

  params.amax_dQ_buffer = amax_dQ_buffer;
  params.amax_dK_buffer = amax_dK_buffer;
  params.amax_dV_buffer = amax_dV_buffer;
  params.amax_dP_buffer = amax_dP_buffer;
  return params;
}

absl::Status RunGpuFMHAF8(
    const GpufMHAConfig &fmha_config, se::DeviceMemoryBase lhs_bmm1_buffer,
    se::DeviceMemoryBase rhs_bmm1_buffer, se::DeviceMemoryBase rhs_bmm2_buffer,
    se::DeviceMemoryBase descale_q_buffer,
    se::DeviceMemoryBase descale_k_buffer,
    se::DeviceMemoryBase descale_v_buffer,
    se::DeviceMemoryBase descale_s_buffer, se::DeviceMemoryBase scale_s_buffer,
    se::DeviceMemoryBase scale_o_buffer, se::DeviceMemoryBase amax_s_buffer,
    se::DeviceMemoryBase amax_o_buffer, se::DeviceMemoryBase output_buffer,
    se::DeviceMemoryBase scratch_buffer,
    std::optional<se::DeviceMemoryBase> activation_buffer, se::Stream *stream,
    RunFusedMHAF8Options options) {
  TF_ASSIGN_OR_RETURN(
      GpufMHAF8Params params,
      GpufMHAF8Params::For(fmha_config, lhs_bmm1_buffer, rhs_bmm1_buffer,
                           rhs_bmm2_buffer, descale_q_buffer, descale_k_buffer,
                           descale_v_buffer, descale_s_buffer, scale_s_buffer,
                           scale_o_buffer, amax_s_buffer, amax_o_buffer,
                           output_buffer, activation_buffer));
  PrimitiveType input_primitive_type = fmha_config.input_type;
  switch (input_primitive_type) {
    case F8E4M3FN:
      return RunGpuFMHAF8Impl<tsl::float8_e4m3fn, tsl::float8_e4m3fn>(
          params, stream, scratch_buffer, options);
    case F8E5M2:
      return RunGpuFMHAF8Impl<tsl::float8_e5m2, tsl::float8_e5m2>(
          params, stream, scratch_buffer, options);
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unimplemented fused MHA with %s", ToString(fmha_config)));
  }
  return absl::OkStatus();
}

absl::Status RunGpuFMHABackwardF8(
    const GpufMHABackwardConfig &fmha_config,
    se::DeviceMemoryBase bmm1_grad_gemm1_rhs_buffer,
    se::DeviceMemoryBase bmm1_grad_gemm2_rhs_buffer,
    se::DeviceMemoryBase bmm2_grad_gemm2_rhs_buffer,
    se::DeviceMemoryBase d_output_buffer,
    se::DeviceMemoryBase bmm2_grad_gemm1_lhs_buffer,
    se::DeviceMemoryBase descale_q_buffer,
    se::DeviceMemoryBase descale_k_buffer,
    se::DeviceMemoryBase descale_v_buffer,
    se::DeviceMemoryBase descale_o_buffer,
    se::DeviceMemoryBase descale_dO_buffer,
    se::DeviceMemoryBase descale_s_buffer,
    se::DeviceMemoryBase descale_dP_buffer, se::DeviceMemoryBase scale_s_buffer,
    se::DeviceMemoryBase scale_dQ_buffer, se::DeviceMemoryBase scale_dK_buffer,
    se::DeviceMemoryBase scale_dV_buffer, se::DeviceMemoryBase scale_dP_buffer,
    se::DeviceMemoryBase d_bmm1_lhs_buffer,
    se::DeviceMemoryBase d_bmm1_rhs_buffer,
    se::DeviceMemoryBase d_bmm2_rhs_buffer, se::DeviceMemoryBase amax_dQ_buffer,
    se::DeviceMemoryBase amax_dK_buffer, se::DeviceMemoryBase amax_dV_buffer,
    se::DeviceMemoryBase amax_dP_buffer, se::DeviceMemoryBase scratch_buffer,
    std::optional<se::DeviceMemoryBase> fwd_output_buffer, se::Stream *stream,
    RunFusedMHABackwardF8Options options) {
  TF_ASSIGN_OR_RETURN(
      GpufMHABackwardF8Params params,
      GpufMHABackwardF8Params::For(
          fmha_config, bmm1_grad_gemm1_rhs_buffer, bmm1_grad_gemm2_rhs_buffer,
          bmm2_grad_gemm2_rhs_buffer, d_output_buffer,
          bmm2_grad_gemm1_lhs_buffer, descale_q_buffer, descale_k_buffer,
          descale_v_buffer, descale_o_buffer, descale_dO_buffer,
          descale_s_buffer, descale_dP_buffer, scale_s_buffer, scale_dQ_buffer,
          scale_dK_buffer, scale_dV_buffer, scale_dP_buffer, d_bmm1_lhs_buffer,
          d_bmm1_rhs_buffer, d_bmm2_rhs_buffer, amax_dQ_buffer, amax_dK_buffer,
          amax_dV_buffer, amax_dP_buffer, fwd_output_buffer));
  PrimitiveType input_primitive_type = fmha_config.input_type;
  switch (input_primitive_type) {
    case F8E4M3FN:
      return RunGpuFMHABackwardF8Impl<tsl::float8_e4m3fn, tsl::float8_e4m3fn>(
          params, stream, scratch_buffer, options);
    case F8E5M2:
      return RunGpuFMHABackwardF8Impl<tsl::float8_e5m2, tsl::float8_e5m2>(
          params, stream, scratch_buffer, options);
    default:
      return Unimplemented("Unimplemented fused MHA backward F8");
  }
  return absl::OkStatus();
}

absl::Status RunGpuFMHA(const GpufMHAConfig &fmha_config,
                        se::DeviceMemoryBase lhs_bmm1_buffer,
                        se::DeviceMemoryBase rhs_bmm1_buffer,
                        se::DeviceMemoryBase rhs_bmm2_buffer,
                        se::DeviceMemoryBase output_buffer,
                        se::DeviceMemoryBase scratch_buffer,
                        std::optional<se::DeviceMemoryBase> bias_buffer,
                        std::optional<se::DeviceMemoryBase> activation_buffer,
                        std::optional<se::DeviceMemoryBase> seqlen_q_buffer,
                        std::optional<se::DeviceMemoryBase> seqlen_k_buffer,
                        se::Stream *stream, RunFusedMHAOptions options) {
  TF_ASSIGN_OR_RETURN(
      GpufMHAParams params,
      GpufMHAParams::For(fmha_config, lhs_bmm1_buffer, rhs_bmm1_buffer,
                         rhs_bmm2_buffer, output_buffer, bias_buffer,
                         activation_buffer, seqlen_q_buffer, seqlen_k_buffer));
  PrimitiveType input_primitive_type = fmha_config.input_type;
  switch (input_primitive_type) {
    case F16:
      return RunGpuFMHAImpl<Eigen::half, Eigen::half, Eigen::half>(
          params, stream, scratch_buffer, options);
    case BF16:
      return RunGpuFMHAImpl<Eigen::bfloat16, Eigen::bfloat16, Eigen::bfloat16>(
          params, stream, scratch_buffer, options);
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unimplemented fused MHA with %s", ToString(fmha_config)));
  }
  return absl::OkStatus();
}

absl::Status RunGpuFMHABackward(
    const GpufMHABackwardConfig &fmha_config,
    se::DeviceMemoryBase bmm1_grad_gemm1_rhs_buffer,
    se::DeviceMemoryBase bmm1_grad_gemm2_rhs_buffer,
    se::DeviceMemoryBase bmm2_grad_gemm1_lhs_buffer,
    se::DeviceMemoryBase bmm2_grad_gemm2_rhs_buffer,
    se::DeviceMemoryBase d_output_buffer, se::DeviceMemoryBase scratch_buffer,
    se::DeviceMemoryBase d_bmm1_lhs_buffer,
    se::DeviceMemoryBase d_bmm1_rhs_buffer,
    se::DeviceMemoryBase d_bmm2_rhs_buffer,
    std::optional<se::DeviceMemoryBase> d_s_buffer,
    std::optional<se::DeviceMemoryBase> d_bias_buffer,
    std::optional<se::DeviceMemoryBase> fwd_output_buffer,
    std::optional<se::DeviceMemoryBase> bias_buffer,
    std::optional<se::DeviceMemoryBase> seqlen_q_buffer,
    std::optional<se::DeviceMemoryBase> seqlen_k_buffer, se::Stream *stream,
    RunFusedMHABackwardOptions options) {
  TF_ASSIGN_OR_RETURN(
      GpufMHABackwardParams params,
      GpufMHABackwardParams::For(
          fmha_config, bmm1_grad_gemm1_rhs_buffer, bmm1_grad_gemm2_rhs_buffer,
          bmm2_grad_gemm1_lhs_buffer, bmm2_grad_gemm2_rhs_buffer,
          d_output_buffer, d_bmm1_lhs_buffer, d_bmm1_rhs_buffer,
          d_bmm2_rhs_buffer, d_s_buffer, d_bias_buffer, fwd_output_buffer,
          bias_buffer, seqlen_q_buffer, seqlen_k_buffer));
  PrimitiveType input_primitive_type = fmha_config.input_type;
  switch (input_primitive_type) {
    case F16:
      return RunGpuFMHABackwardImpl<Eigen::half, Eigen::half, Eigen::half>(
          params, stream, scratch_buffer, options);
    case BF16:
      return RunGpuFMHABackwardImpl<Eigen::bfloat16, Eigen::bfloat16,
                                    Eigen::bfloat16>(params, stream,
                                                     scratch_buffer, options);
    default:
      return Unimplemented("Unimplemented fused MHA backward");
  }
  return absl::OkStatus();
}

std::string ToString(const GpufMHAConfig &config) {
  std::string result = "GpufMHAConfig:\n";
  absl::StrAppend(&result,
                  "input_type: ", PrimitiveType_Name(config.input_type), ", ");
  absl::StrAppend(
      &result, "output_type: ", PrimitiveType_Name(config.output_type), ", ");
  absl::StrAppend(&result, "Kind: ", CudnnfMHAKindToString(config.kind), ", ");
  if (config.fmha_scale) {
    absl::StrAppend(&result, "fmha_scale: ", *config.fmha_scale, ", ");
  }
  if (config.dropout_rate) {
    absl::StrAppend(&result, "dropout_rate: ", *config.dropout_rate, ", ");
  }
  if (config.seed) {
    absl::StrAppend(&result, "seed: ", *config.seed, ", ");
  }
  absl::StrAppend(&result, "Algorithm Desc: ", config.algorithm.ToString(),
                  "\n");
  absl::StrAppend(&result, "lhs_bmm1: ", config.lhs_bmm1.ToString(), "\n");
  absl::StrAppend(&result, "rhs_bmm1: ", config.rhs_bmm1.ToString(), "\n");
  absl::StrAppend(&result, "rhs_bmm2: ", config.rhs_bmm2.ToString(), "\n");
  absl::StrAppend(&result, "intermediate_lhs_bmm2: ",
                  config.intermediate_lhs_bmm2.ToString(), "\n");
  absl::StrAppend(&result, "output: ", config.output.ToString(), "\n");

  if (config.mask) {
    absl::StrAppend(&result, "mask: ", (*config.mask).ToString(), "\n");
  }

  if (config.bias) {
    absl::StrAppend(&result, "bias: ", (*config.bias).ToString(), "\n");
  }

  return result;
}

// std::string ToString(const GpufMHAConfig &config) {
//   std::string result = "GpufMHAConfig:\n";
//   absl::StrAppend(&result,
//                   "input_type: ", PrimitiveType_Name(config.input_type), ",
//                   ");
//   absl::StrAppend(
//       &result, "output_type: ", PrimitiveType_Name(config.output_type), ",
//       ");
//   absl::StrAppend(&result, "Kind: ", CudnnfMHAKindToString(config.kind), ",
//   "); if (config.fmha_scale) {
//     absl::StrAppend(&result, "fmha_scale: ", *config.fmha_scale, ", ");
//   }
//   absl::StrAppend(&result, "Algorithm Desc: ", config.algorithm.ToString(),
//                   "\n");
//   absl::StrAppend(&result, "lhs_bmm1: ", config.lhs_bmm1.ToString(), "\n");
//   absl::StrAppend(&result, "rhs_bmm1: ", config.rhs_bmm1.ToString(), "\n");
//   absl::StrAppend(&result, "rhs_bmm2: ", config.rhs_bmm2.ToString(), "\n");
//   absl::StrAppend(&result, "intermediate_lhs_bmm2: ",
//                   config.intermediate_lhs_bmm2.ToString(), "\n");
//   absl::StrAppend(&result, "output: ", config.output.ToString(), "\n");
//   return result;
// }

}  // namespace gpu
}  // namespace xla
