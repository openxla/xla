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

#include "xla/service/cpu/onednn_util.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/log/log.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_common.hpp"
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
#include "oneapi/dnnl/dnnl_types.h"

#define EIGEN_USE_THREADS

namespace xla {
namespace cpu {

dnnl::post_ops PopulateOneDnnPostOps(
    int& fused_operand_idx, const dnnl::engine& cpu_engine,
    const std::vector<dnnl::memory::desc>& fused_mds,
    const OneDnnFusionConfig* fusion_config,
    FusedOperandsRef* fused_operands_ref, dnnl::memory::desc* bias_md) {
  dnnl::post_ops post_ops;
  int linear_scale_idx = 0;
  for (auto& fused_op : fusion_config->ops()) {
    switch (fused_op) {
      case OneDnnFusionConfig::RELU:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_relu, 0.f, 0.f);
        break;
      case OneDnnFusionConfig::TANH:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_tanh, 0.f, 0.f);
        break;
      case OneDnnFusionConfig::GELU_TANH:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0.f, 0.f);
        break;
      case OneDnnFusionConfig::GELU_ERF:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_gelu_erf, 0.f, 0.f);
        break;
      case OneDnnFusionConfig::RELU6:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_clip_v2, 0.f, 6.0f);
        break;
      case OneDnnFusionConfig::SIGMOID:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_logistic, 0.f, 0.f);
        break;
      case OneDnnFusionConfig::SWISH:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_swish, 1.0f, 0.0f);
        break;
      case OneDnnFusionConfig::SUM:
        post_ops.append_sum();
        // oneDNN does not require an input for SUM post-op.
        fused_operand_idx++;
        break;
      case OneDnnFusionConfig::BIAS: {
        *bias_md = fused_mds.at(fused_operand_idx);
        if (fused_operands_ref) {
          fused_operands_ref->postop_args.emplace_back(
              DNNL_ARG_BIAS,
              dnnl::memory(*bias_md, cpu_engine,
                           fused_operands_ref->bufs[fused_operand_idx]));
        }
        fused_operand_idx++;
      } break;
      case OneDnnFusionConfig::ELU:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_elu, 1.0f, 0.0f);
        break;
      case OneDnnFusionConfig::BINARY_ADD: {
        auto binary_md = fused_mds.at(fused_operand_idx);
        if (fused_operands_ref) {
          auto arg_idx =
              DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_ops.len()) | DNNL_ARG_SRC_1;
          fused_operands_ref->postop_args.emplace_back(
              arg_idx,
              dnnl::memory(binary_md, cpu_engine,
                           fused_operands_ref->bufs[fused_operand_idx]));
        }
        post_ops.append_binary(dnnl::algorithm::binary_add, binary_md);
        fused_operand_idx++;
      } break;
      case OneDnnFusionConfig::LINEAR: {
        float const_float = fusion_config->alpha()[linear_scale_idx];
        post_ops.append_eltwise(dnnl::algorithm::eltwise_linear, const_float,
                                0.f);
        linear_scale_idx++;
      } break;
      default:
        LOG(FATAL) << __FILE__ << ":" << __LINE__
                   << " Attempt to call OneDNN runtime library with "
                      "unsupported post op."
                   << std::endl;
    }
  }
  return post_ops;
}

void AddQuantParamArgs(
    bool is_conv, bool conv_groups, dnnl::primitive_attr& attrs,
    int& fused_operand_idx, const dnnl::engine& cpu_engine,
    const std::vector<dnnl::memory::desc>& fused_mds, const dnnl::memory::desc& input_md,
    const dnnl::memory::desc& weights_md, const dnnl::memory::desc& output_md,
    FusedOperandsRef* fused_operands_ref, QuantizationParams* qparams) {
  dnnl::memory::data_type res_dt = output_md.get_data_type();
  qparams->quant_result =
      (res_dt == dnnl::memory::data_type::s8 || res_dt == dnnl::memory::data_type::u8);
  qparams->quant_operands = weights_md.get_data_type() == dnnl::memory::data_type::s8;
  dnnl::memory::data_type inp_dt = input_md.get_data_type();
  if (qparams->quant_operands) {
    // Hybrid quantization is not currently supported.
    CHECK_NE(inp_dt == dnnl::memory::data_type::s8 ||
                          inp_dt == dnnl::memory::data_type::u8, 0);

    auto src_scale_md = fused_mds.at(fused_operand_idx);
    if (fused_operands_ref) {
      fused_operands_ref->postop_args.emplace_back(
          DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
          dnnl::memory(src_scale_md, cpu_engine,
                 fused_operands_ref->bufs[fused_operand_idx]));
    }

    fused_operand_idx++;
    auto src_zp_md = fused_mds.at(fused_operand_idx);
    if (fused_operands_ref) {
      int* src_zp_data = (int*)fused_operands_ref->bufs[fused_operand_idx];
      // We may need to negate the sign of the zp to get the original one
      // because the hlo optimizer flips the zp sign in uniform_dequantize
      // pattern.
      qparams->src_zp_vec[0] =
          qparams->negated_src_zp ? src_zp_data[0] * -1 : src_zp_data[0];
      fused_operands_ref->postop_args.emplace_back(
          DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC,
          dnnl::memory(src_zp_md, cpu_engine, qparams->src_zp_vec.data()));
    }

    fused_operand_idx++;
    auto wei_scale_md = fused_mds.at(fused_operand_idx);
    int wei_scale_size = wei_scale_md.get_dims()[0];
    if (fused_operands_ref) {
      fused_operands_ref->postop_args.emplace_back(
          DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
          dnnl::memory(wei_scale_md, cpu_engine,
                 fused_operands_ref->bufs[fused_operand_idx]));
    }
    fused_operand_idx++;

    // We skip the wei_zp arg which is used in weights decompression that we
    // don't support currently.
    fused_operand_idx++;

    // oneDNN only supports common scale/zp for src (no per-channel support).
    CHECK_EQ(src_scale_md.get_dims()[0], 1);
    CHECK_EQ(src_zp_md.get_dims()[0],
                          src_scale_md.get_dims()[0]);
    if (qparams->quant_result) {
      auto dst_scale_md = fused_mds.at(fused_operand_idx);
      if (fused_operands_ref) {
        float* scale_data = (float*)fused_operands_ref->bufs[fused_operand_idx];
        // We may need to compute the reciprocal of scale to get the original
        // one because the hlo optimizer changes it in uniform_quantize pattern.
        qparams->dst_scale_vec[0] =
            qparams->inversed_dst_scale ? 1.0 / scale_data[0] : scale_data[0];
        fused_operands_ref->postop_args.emplace_back(
            DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
            dnnl::memory(dst_scale_md, cpu_engine, qparams->dst_scale_vec.data()));
      }

      fused_operand_idx++;
      auto dst_zp_md = fused_mds.at(fused_operand_idx);
      if (fused_operands_ref) {
        auto dst_zp_buf = fused_operands_ref->bufs[fused_operand_idx];
        if (dst_zp_md.get_data_type() == dnnl::memory::data_type::f32) {
          // oneDNN expects zp to be int32 not f32.
          qparams->dst_zp_vec[0] = static_cast<int>(((float*)dst_zp_buf)[0]);
        } else {
          qparams->dst_zp_vec[0] = ((int*)dst_zp_buf)[0];
        }

        auto dst_zp_md_new =
            dnnl::memory::desc(dst_zp_md.get_dims(), dnnl::memory::data_type::s32,
                         dnnl::memory::format_tag::x);
        fused_operands_ref->postop_args.emplace_back(
            DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST,
            dnnl::memory(dst_zp_md_new, cpu_engine, qparams->dst_zp_vec.data()));
      }
      fused_operand_idx++;

      // oneDNN only supports common scale/zp for dst (no per-channel
      // support).
      CHECK_EQ(dst_scale_md.get_dims()[0], 1);
      CHECK_EQ(dst_zp_md.get_dims()[0],
                            dst_scale_md.get_dims()[0]);
    }
    // We set the mask to zero as we are using single scale and zero point
    // for the whole tensor.
    attrs.set_scales_mask(DNNL_ARG_SRC, 0);
    attrs.set_zero_points_mask(DNNL_ARG_SRC, 0);
    int wei_mask;
    if (wei_scale_size == 1) {
      wei_mask = 0;
    } else {
      wei_mask = !is_conv ? 2 : conv_groups ? 3 : 1;
    }

    attrs.set_scales_mask(DNNL_ARG_WEIGHTS, wei_mask);
    if (qparams->quant_result) {
      attrs.set_scales_mask(DNNL_ARG_DST, 0);
      attrs.set_zero_points_mask(DNNL_ARG_DST, 0);
    }
  }
}

}  // namespace cpu
}  // namespace xla
