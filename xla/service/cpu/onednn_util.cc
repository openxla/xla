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

#include "xla/service/cpu/onednn_util.h"

#define EIGEN_USE_THREADS

namespace xla {
namespace cpu {

std::unique_ptr<tsl::OneDnnThreadPool> CreateOneDnnThreadPool(
    const Eigen::ThreadPoolDevice* threadpool_device) {
#ifndef ENABLE_ONEDNN_OPENMP
  if (threadpool_device != nullptr) {
    return std::make_unique<tsl::OneDnnThreadPool>(threadpool_device->getPool(),
                                                   false);
  }
#endif  // !ENABLE_ONEDNN_OPENMP
  return nullptr;
}

dnnl::stream MakeOneDnnStream(
    const dnnl::engine& cpu_engine,
    dnnl::threadpool_interop::threadpool_iface* thread_pool) {
  return (thread_pool != nullptr)
             ? dnnl::threadpool_interop::make_stream(cpu_engine, thread_pool)
             : dnnl::stream(cpu_engine);
}

void AddQuantParamArgs(bool is_conv, bool conv_groups,
                       dnnl::primitive_attr& attrs, int& fused_operand_idx,
                       const engine& cpu_engine,
                       const std::vector<memory::desc>& fused_mds,
                       const memory::desc& input_md,
                       const memory::desc& weights_md, const memory::desc& output_md,
                       FusedOperandsRef* fused_operands_ref,
                       QuantizationParams* qparams) {
  memory::data_type res_dt = output_md.get_data_type();
  qparams->quant_result = 
      (res_dt == memory::data_type::s8 || res_dt == memory::data_type::u8);
  qparams->quant_operands = 
      weights_md.get_data_type() == memory::data_type::s8;
  memory::data_type inp_dt = input_md.get_data_type();
  if (qparams->quant_operands) {
    // Hybrid quantization is not currently supported.
    XLA_LIGHTWEIGHT_CHECK(inp_dt == memory::data_type::s8 ||
                          inp_dt == memory::data_type::u8);

    auto src_scale_md = fused_mds.at(fused_operand_idx);
    auto src_scale_buf = fused_operands_ref
                              ? fused_operands_ref->bufs[fused_operand_idx]
                              : nullptr;
    fused_operand_idx++;
    auto src_zp_md = fused_mds.at(fused_operand_idx);
    auto src_zp_buf = fused_operands_ref
                          ? fused_operands_ref->bufs[fused_operand_idx]
                          : nullptr;
    fused_operand_idx++;
    auto wei_scale_md = fused_mds.at(fused_operand_idx);
    auto wei_scale_buf = fused_operands_ref
                              ? fused_operands_ref->bufs[fused_operand_idx]
                              : nullptr;
    fused_operand_idx++;
    int wei_scale_size = wei_scale_md.get_dims()[0];
    auto wei_zp_md = fused_mds.at(fused_operand_idx);
    auto wei_zp_buf = fused_operands_ref
                          ? fused_operands_ref->bufs[fused_operand_idx]
                          : nullptr;
    fused_operand_idx++;
    // oneDNN only supports common scale/zp for src (no per-channel support).
    XLA_LIGHTWEIGHT_CHECK(src_scale_md.get_dims()[0] == 1);
    XLA_LIGHTWEIGHT_CHECK(src_zp_md.get_dims()[0] ==
                          src_scale_md.get_dims()[0]);
    if (fused_operands_ref) {
      fused_operands_ref->postop_args.emplace_back(
          DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
          memory(src_scale_md, cpu_engine, src_scale_buf));
      int* src_zp_data = (int*)src_zp_buf;

      // We may need to negate the sign of the zp to get the original one
      // because the hlo optimizer flips the zp sign in uniform_dequantize
      // pattern.
      qparams->src_zp_vec[0] =
          qparams->negated_src_zp ? src_zp_data[0] * -1 : src_zp_data[0];
      fused_operands_ref->postop_args.emplace_back(
          DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC,
          memory(src_zp_md, cpu_engine, qparams->src_zp_vec.data()));
      fused_operands_ref->postop_args.emplace_back(
          DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
          memory(wei_scale_md, cpu_engine, wei_scale_buf));

      // Weights' zero point is supported by oneDNN only in weights
      // decompression.
      auto wei_zp_mem = memory(wei_zp_md, cpu_engine, wei_zp_buf);
      if (qparams->quant_result) {
        auto dst_scale_md = fused_mds.at(fused_operand_idx);
        auto dst_scale_buf = fused_operands_ref->bufs[fused_operand_idx];
        fused_operand_idx++;
        auto dst_zp_md = fused_mds.at(fused_operand_idx);
        auto dst_zp_buf = fused_operands_ref->bufs[fused_operand_idx];
        fused_operand_idx++;

        // oneDNN only supports common scale/zp for dst (no per-channel
        // support).
        XLA_LIGHTWEIGHT_CHECK(dst_scale_md.get_dims()[0] == 1);
        XLA_LIGHTWEIGHT_CHECK(dst_zp_md.get_dims()[0] ==
                              dst_scale_md.get_dims()[0]);

        float* scale_data = (float*)dst_scale_buf;
        // We may need to compute the reciprocal of scale to get the original
        // one because the hlo optimizer changes it in uniform_quantize pattern.
        qparams->dst_scale_vec[0] =
            qparams->inversed_dst_scale ? 1.0 / scale_data[0] : scale_data[0];
        if (dst_zp_md.get_data_type() == memory::data_type::f32) {
          // oneDNN expects zp to be int32 not f32.
          qparams->dst_zp_vec[0] = static_cast<int>(((float*)dst_zp_buf)[0]);
        } else {
          qparams->dst_zp_vec[0] = ((int*)dst_zp_buf)[0];
        }
        fused_operands_ref->postop_args.emplace_back(
            DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
            memory(dst_scale_md, cpu_engine, qparams->dst_scale_vec.data()));
        auto dst_zp_md_new =
            memory::desc(dst_zp_md.get_dims(), memory::data_type::s32,
                          memory::format_tag::x);
        fused_operands_ref->postop_args.emplace_back(
            DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST,
            memory(dst_zp_md_new, cpu_engine, qparams->dst_zp_vec.data()));
      }
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

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
