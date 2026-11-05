/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/hipdnn_custom_call_compiler.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/rocm/rocm_dnn.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

using se::dnn::DataType;
using se::dnn::MatmulTensorDescriptor;
using se::dnn::TensorDescriptor;

absl::StatusOr<CudnnfMHAMaskKind> AsCudnnFmhaMaskKind(
    CudnnfMHABackendConfig_MaskType mask_type) {
  switch (mask_type) {
    case CudnnfMHABackendConfig::NO_MASK:
      return CudnnfMHAMaskKind::kNoMask;
    case CudnnfMHABackendConfig::PADDING:
      return CudnnfMHAMaskKind::kPadding;
    case CudnnfMHABackendConfig::CAUSAL:
      return CudnnfMHAMaskKind::kCausal;
    case CudnnfMHABackendConfig::PADDING_CAUSAL:
      return CudnnfMHAMaskKind::kPaddingCausal;
    case CudnnfMHABackendConfig::ALIBI:
      return CudnnfMHAMaskKind::kAlibi;
    default:
      return Internal("Unknown fmha mask kind.");
  }
}

absl::StatusOr<TensorDescriptor> TensorDescriptorFor(const Shape& shape) {
  TF_ASSIGN_OR_RETURN(const DataType type,
                      GetDNNDataTypeFromPrimitiveType(shape.element_type()));
  return TensorDescriptor::For(type, shape.dimensions(),
                               shape.layout().minor_to_major());
}

enum Side { LHS, RHS };

absl::StatusOr<MatmulTensorDescriptor> MatmulTensorDescriptorFor(
    const Shape& shape, const DotDimensionNumbers& dnums, const Side side) {
  TF_ASSIGN_OR_RETURN(const DataType type,
                      GetDNNDataTypeFromPrimitiveType(shape.element_type()));
  return MatmulTensorDescriptor::For(
      type, shape.dimensions(), shape.layout().minor_to_major(),
      (side == LHS) ? dnums.lhs_batch_dimensions()
                    : dnums.rhs_batch_dimensions(),
      (side == LHS) ? dnums.lhs_contracting_dimensions()
                    : dnums.rhs_contracting_dimensions());
}

absl::StatusOr<std::unique_ptr<se::dnn::DnnGraph>> BuildGraphForForwardFMHA(
    se::dnn::DnnSupport* dnn_support,
    const se::DeviceDescription& gpu_device_info,
    HloCustomCallInstruction* custom_call) {
  TF_ASSIGN_OR_RETURN(const CudnnfMHAKind kind, GetCudnnfMHAKind(custom_call));
  TF_ASSIGN_OR_RETURN(const GpuBackendConfig gpu_config,
                      custom_call->backend_config<GpuBackendConfig>());
  const CudnnfMHABackendConfig& config = gpu_config.cudnn_fmha_backend_config();

  TF_ASSIGN_OR_RETURN(
      MatmulTensorDescriptor q,
      MatmulTensorDescriptorFor(custom_call->operand(0)->shape(),
                                config.bmm1_dot_dimension_numbers(), LHS));
  TF_ASSIGN_OR_RETURN(
      MatmulTensorDescriptor k,
      MatmulTensorDescriptorFor(custom_call->operand(1)->shape(),
                                config.bmm1_dot_dimension_numbers(), RHS));
  TF_ASSIGN_OR_RETURN(
      MatmulTensorDescriptor v,
      MatmulTensorDescriptorFor(custom_call->operand(2)->shape(),
                                config.bmm2_dot_dimension_numbers(), RHS));
  TF_ASSIGN_OR_RETURN(
      TensorDescriptor output,
      TensorDescriptorFor(ShapeUtil::GetSubshape(custom_call->shape(), {0})));

  std::optional<TensorDescriptor> activation;
  if (ShapeUtil::TupleElementCount(custom_call->shape()) == 3) {
    TF_ASSIGN_OR_RETURN(
        activation,
        TensorDescriptorFor(ShapeUtil::GetSubshape(custom_call->shape(), {1})));
  }

  int input_index = 3;
  std::optional<TensorDescriptor> bias;
  if (kind == CudnnfMHAKind::kScaleBiasSoftmax ||
      kind == CudnnfMHAKind::kScaleBiasSoftmaxDropout) {
    TF_ASSIGN_OR_RETURN(
        bias, TensorDescriptorFor(custom_call->operand(input_index++)->shape()));
  }

  TF_ASSIGN_OR_RETURN(const CudnnfMHAMaskKind cudnn_mask_type,
                      AsCudnnFmhaMaskKind(config.mask_type()));
  TF_ASSIGN_OR_RETURN(
      const se::dnn::FMHAMaskKind dnn_mask_type,
      GetDNNFmhaMaskKindFromCudnnFmhaMaskKind(cudnn_mask_type));

  // Padding, ragged and paged attention add operands that the builder rejects
  // anyway; leaving input_index short of operand_count() here would hide a
  // real mismatch, so check it only for the variants that get that far.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::dnn::DnnGraph> graph,
      stream_executor::gpu::GetHipdnnFlashAttentionOperationGraph(
          dnn_support, gpu_device_info, q, k, v, output, bias, activation,
          config.fmha_scale(), config.dropout_rate() > 0.0, dnn_mask_type,
          config.sliding_window_length(), config.max_seg_per_batch(),
          config.is_paged_attention(),
          /*has_score_mod=*/!custom_call->called_computations().empty()));
  TF_RET_CHECK(input_index == custom_call->operand_count());
  return graph;
}

class HipdnnCustomCallVisitor : public DfsHloRewriteVisitor {
 public:
  HipdnnCustomCallVisitor(se::StreamExecutor* stream_exec,
                          const se::DeviceDescription& gpu_device_info,
                          BinaryMap& compilation_results)
      : stream_exec_(stream_exec),
        gpu_device_info_(gpu_device_info),
        compilation_results_(compilation_results) {}

  absl::Status HandleCustomCall(HloInstruction* hlo) override {
    if (!IsCustomCallTofMHA(*hlo)) {
      if (IsCustomCallTofMHAF8(*hlo) || IsCustomCallToBlockScaledDot(*hlo)) {
        return absl::UnimplementedError(
            absl::StrCat("hipDNN has no engine for ", hlo->custom_call_target(),
                         "; it is a cuDNN-only Custom Call."));
      }
      return absl::OkStatus();
    }
    if (IsBwdCustomCallTofMHA(*hlo)) {
      return absl::UnimplementedError(
          "hipDNN has no backward attention engine, so attention cannot be "
          "differentiated on this backend.");
    }

    TF_ASSIGN_OR_RETURN(const std::string fingerprint_without_workspace,
                        FingerprintWithBackendConfig<GpuBackendConfig>(*hlo));
    auto it = workspace_sizes_.find(fingerprint_without_workspace);
    if (it != workspace_sizes_.cend()) {
      VLOG(4) << "Cache hit.";
      AddWorkspace(*hlo, it->second);
      MarkAsChanged();
      return absl::OkStatus();
    }

    if (stream_exec_ == nullptr) {
      return absl::UnimplementedError(
          "hipDNN attention Custom Calls cannot be compiled without a device.");
    }
    se::dnn::DnnSupport* dnn_support = stream_exec_->AsDnn();
    if (dnn_support == nullptr) {
      return absl::InternalError("No DNN support for the ROCm executor.");
    }

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<se::dnn::DnnGraph> graph,
        BuildGraphForForwardFMHA(dnn_support, gpu_device_info_,
                                 DynCast<HloCustomCallInstruction>(hlo)));
    TF_ASSIGN_OR_RETURN(const int64_t workspace_size,
                        stream_executor::gpu::HipdnnGraphWorkspaceSize(*graph));
    workspace_sizes_.insert(it, {fingerprint_without_workspace, workspace_size});
    AddWorkspace(*hlo, workspace_size);

    TF_ASSIGN_OR_RETURN(std::string serialized,
                        stream_executor::gpu::SerializeHipdnnGraph(*graph));
    // Recompute the fingerprint now that the workspace is part of the shape,
    // so it matches the one the thunk emitter will look up.
    TF_ASSIGN_OR_RETURN(const std::string fingerprint_with_workspace,
                        FingerprintWithBackendConfig<GpuBackendConfig>(*hlo));
    compilation_results_[fingerprint_with_workspace] = std::move(serialized);

    MarkAsChanged();
    return absl::OkStatus();
  }

 private:
  void AddWorkspace(HloInstruction& hlo, int64_t workspace_size) {
    if (workspace_size == 0) {
      return;
    }
    VLOG(4) << "Applying workspace size " << workspace_size << " to "
            << hlo.ToString();
    hlo.mutable_shape()->mutable_tuple_shapes()->back().set_dimensions(
        0, workspace_size);
  }

  se::StreamExecutor* stream_exec_;
  const se::DeviceDescription& gpu_device_info_;
  BinaryMap& compilation_results_;
  absl::flat_hash_map<std::string, int64_t> workspace_sizes_;
};

}  // namespace

absl::StatusOr<bool> HipdnnCustomCallCompiler::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_SCOPED_LOGGING_TIMER_LEVEL("hipDNN custom call compiler", 8);
  return HipdnnCustomCallVisitor(stream_exec_, gpu_device_info_,
                                 compilation_results_)
      .RunOnModule(module, execution_threads);
}

}  // namespace gpu
}  // namespace xla
