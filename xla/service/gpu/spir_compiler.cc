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

#include "xla/service/gpu/spir_compiler.h"

#include <stdlib.h>

#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tsl/platform/path.h"
#include "tsl/platform/status.h"
#include "tsl/util/env_var.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/call_inliner.h"
#include "xla/service/convert_mover.h"
#include "xla/service/dot_dimension_merger.h"
#include "xla/service/dump.h"
#include "xla/service/float_normalization.h"
#include "xla/service/float_support.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/buffer_sharing.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/cudnn_fused_conv_rewriter.h"
#include "xla/service/gpu/gpu_conv_padding_legalization.h"
#include "xla/service/gpu/gpu_conv_rewriter.h"
#include "xla/service/gpu/gpu_layout_assignment.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/service/gpu/triangular_solve_rewriter.h"
#include "xla/service/hlo_constant_folding.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/layout_normalization.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/reshape_decomposer.h"
#include "xla/service/reshape_mover.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/types.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

class ConvBfloat16Support : public FloatSupport {
 public:
  explicit ConvBfloat16Support()
      : FloatSupport(BF16), is_conv_bf16_supported_(true) {}

  bool SupportsLowPrecisionOperand(const HloInstruction& hlo,
                                   int64_t operand_index) const override {
    return (hlo.opcode() != HloOpcode::kConvolution) || is_conv_bf16_supported_;
  }

  bool SupportsLowPrecisionOutput(const HloInstruction& hlo) const override {
    return (hlo.opcode() != HloOpcode::kConvolution) || is_conv_bf16_supported_;
  }

  bool SupportsMixedPrecisions(const HloInstruction& hlo) const override {
    // Skip all HLOs other than convolutions.
    return (hlo.opcode() != HloOpcode::kConvolution);
  }

 private:
  bool is_conv_bf16_supported_;
};

absl::Status SPIRCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, se::GpuComputeCapability gpu_version,
    se::dnn::VersionInfo dnn_version,
    se::DeviceMemoryAllocator* device_allocator) {
  auto cuda_compute_capability =
      std::get<se::CudaComputeCapability>(gpu_version);
  // Convert convolutions into CustomCalls to onednn, then canonicalize them
  // (GpuConvPaddingLegalization).
  HloPassPipeline pipeline("conv_canonicalization");
  pipeline.AddInvariantCheckerDebug<HloVerifier>(
      /*layout_sensitive=*/false,
      /*allow_mixed_precision=*/false);

  // Convert upsupported bf16 convolutions to f32.
  ConvBfloat16Support conv_bf16_support;
  pipeline.AddPass<FloatNormalization>(&conv_bf16_support);

  pipeline.AddPass<GpuConvRewriter>();
  pipeline.AddPass<CudnnFusedConvRewriter>(cuda_compute_capability);
  pipeline.AddPass<GpuConvPaddingLegalization>();

  // The conv padding/vectorization passes which we need to get rid of.  They
  // also leave behind unnecessary tuple/get-tuple-element pairs that
  // TupleSimplifier fixes.
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<TupleSimplifier>();

  AlgebraicSimplifierOptions algsimp_options;
  algsimp_options.set_enable_conv_operand_swap(false);
  algsimp_options.set_enable_unconditional_reduce_of_concat_replacement(false);
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(algsimp_options);

  // tf2xla bridge, GpuConvRewriter, and introduce reshapes and transposes. Run
  // ReshapeMover to a fixed point.  Include algsimp because ReshapeMover relies
  // on it.
  [&, &pipeline = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
          "reshape_mover_after_conv_canonicalization")] {
    ReshapeMoverOptions reshape_mover_options;
    reshape_mover_options.reshape_of_1d_broadcast_is_cheap = true;
    pipeline.AddPass<HloPassFix<ReshapeMover>>(reshape_mover_options);
    pipeline.AddPass<AlgebraicSimplifier>(algsimp_options);
  }();

  // The reshapes and transposes can possibly be eliminated using
  // AlgebraicSimplifier. ConvertMover and ReshapeMover fight with each other.
  // ConvertMover wants to move some converts down the graph, but ReshapeMover
  // wants to move them up the graph. We run ConvertMover and algsimp to a fixed
  // point.
  [&, &pipeline = pipeline.AddPass<HloPassFix<HloPassPipeline>>(
          "simplify_after_conv_canonicalization")] {
    pipeline.AddPass<ConvertMover>();
    pipeline.AddPass<AlgebraicSimplifier>(algsimp_options);
  }();

  // GpuConvRewriter, GpuConvPaddingLegalization and may add instructions which
  // can be simplified by constant folding.
  pipeline.AddPass<HloConstantFolding>();
  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());

  return absl::OkStatus();
}

absl::Status SPIRCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options, const TargetConfig& gpu_target_config,
    tsl::thread::ThreadPool* thread_pool) {
  HloPassPipeline pre_pipeline("spir post-layout_assignment part 1");

  // This needs to run before GemmRewriter, which is part of
  // OptimizeHloPostLayoutAssignment().
  auto cuda_compute_capability = std::get<se::CudaComputeCapability>(
      gpu_target_config.device_description.gpu_compute_capability());

  pre_pipeline.AddPass<DotDimensionMerger>();

  // Padding a gemm operand that's a constant results in pad(constant).  Run
  // constant-folding to simplify this into a new constant.
  pre_pipeline.AddPass<HloConstantFolding>();
  TF_RETURN_IF_ERROR(pre_pipeline.Run(hlo_module).status());

  TF_RETURN_IF_ERROR(GpuCompiler::OptimizeHloPostLayoutAssignment(
      hlo_module, stream_exec, options, gpu_target_config, thread_pool));

  HloPassPipeline post_pipeline("spir post-layout_assignment part 2");

  // Transform TriangularSolve ops into custom-calls, so we can add temp
  // memory.
  post_pipeline.AddPass<TriangularSolveRewriter>();

  TF_RETURN_IF_ERROR(post_pipeline.Run(hlo_module).status());

  return absl::OkStatus();
}

SPIRCompiler::SPIRCompiler()
    : GpuCompiler(stream_executor::sycl::kSyclPlatformId, spir::TargetTriple(),
                  spir::DataLayout()) {}

HloDataflowAnalysis::CanShareBuffer SPIRCompiler::GetCanShareBuffer() const {
  return &CanShareBufferHint;
}

absl::StatusOr<GpuCompiler::BackendCompileResult>
SPIRCompiler::CompileTargetBinary(const HloModuleConfig& module_config,
                                  llvm::Module* llvm_module,
                                  se::GpuComputeCapability gpu_version,
                                  bool relocatable,
                                  const HloModule* debug_module,
                                  const CompileOptions& options) {
  if (relocatable) {
    return Unimplemented("relocatable target binary is not implemented");
  }

  std::vector<uint8_t> spir;
  {
    // This may print multiple lines per HLO compilation because of the
    // parallelized compilation of LLVM modules.
    XLA_SCOPED_LOGGING_TIMER_IF(
        "SPIRCompiler::CompileTargetBinary - CompileToSpir",
        !options.is_autotuning_compilation);
    TF_ASSIGN_OR_RETURN(spir,
                        spir::CompileToSpir(llvm_module, gpu_version,
                                            module_config.debug_options()));
  }

  return BackendCompileResult{"", std::move(spir)};
}

}  // namespace gpu
}  // namespace xla
