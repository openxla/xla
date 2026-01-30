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

#ifndef XLA_SERVICE_CPU_ONEDNN_UTIL_H_
#define XLA_SERVICE_CPU_ONEDNN_UTIL_H_

#define EIGEN_USE_THREADS

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_common.hpp"
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_config.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/cpu_info.h"
#include "xla/service/cpu/backend_config.pb.h"

namespace xla {
namespace cpu {

struct QuantizationParams {
  QuantizationParams()
      : src_zp_vec(1),
        dst_zp_vec(1),
        dst_scale_vec(1),
        quant_operands(false),
        quant_result(false) {}
  std::vector<int> src_zp_vec;
  std::vector<int> dst_zp_vec;
  std::vector<float> dst_scale_vec;
  bool quant_operands;
  bool quant_result;
  bool negated_src_zp;
  bool inversed_dst_scale;
};

inline bool IsSupportedType(xla::PrimitiveType dtype) {
  using tsl::port::CPUFeature;
  // TODO(intel-tf): Enable more types.
  switch (dtype) {
    case F32:
      return true;
    case BF16:
      return TestCPUFeature(CPUFeature::AVX512F) ||
             TestCPUFeature(CPUFeature::AVX_NE_CONVERT) ||
             TestCPUFeature(CPUFeature::AMX_BF16);
    case F16:
      return (TestCPUFeature(CPUFeature::AVX512BW) &&
              (TestCPUFeature(CPUFeature::AVX512_FP16) ||
               TestCPUFeature(CPUFeature::AMX_FP16))) ||
             TestCPUFeature(CPUFeature::AVX_NE_CONVERT);
    case S8:
    case U8:
      return TestCPUFeature(CPUFeature::AVX_VNNI_INT8) ||
             TestCPUFeature(CPUFeature::AVX512_VNNI) ||
             TestCPUFeature(CPUFeature::AMX_INT8);
    default:
      return false;
  }
  return false;
}

inline bool HasAMXTile() {
  return TestCPUFeature(tsl::port::CPUFeature::AMX_TILE);
}

struct FusedOperandsRef {
  const std::vector<void*>& bufs;
  std::vector<std::pair<int, dnnl::memory>>& postop_args;
};

// These template functions must have explicit specialization at the definition
// site.
template <typename PrimDesc>
std::unique_ptr<PrimDesc> CreateOneDnnPrimDesc(HloInstruction*);

template <BackendConfig::BackendConfigOneofCase config,
          typename TransformationType = void>
struct PrimitiveTrait;

template <BackendConfig::BackendConfigOneofCase config>
typename PrimitiveTrait<config>::pointer_type GetKernelConfig(
    absl::StatusOr<BackendConfig>*);

dnnl::post_ops PopulateOneDnnPostOps(
    int& fused_operand_idx, const dnnl::engine& cpu_engine,
    const std::vector<dnnl::memory::desc>& fused_mds,
    const OneDnnFusionConfig* fusion_config,
    FusedOperandsRef* fused_operands_ref = nullptr,
    dnnl::memory::desc* bias_md = nullptr);

void AddQuantParamArgs(bool is_conv, bool conv_groups,
                       dnnl::primitive_attr& attrs, int& fused_operand_idx,
                       const dnnl::engine& cpu_engine,
                       const std::vector<dnnl::memory::desc>& fused_mds,
                       const dnnl::memory::desc& input_md,
                       const dnnl::memory::desc& weights_md,
                       const dnnl::memory::desc& output_md,
                       FusedOperandsRef* fused_operands_ref = nullptr,
                       QuantizationParams* qparams = nullptr);

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_ONEDNN_UTIL_H_
