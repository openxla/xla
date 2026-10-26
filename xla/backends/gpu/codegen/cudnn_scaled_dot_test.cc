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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <ostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/uniform_int_distribution.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/gpu/autotuner/gpu_profiler.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/backends/gpu/transforms/composite_rewriter.h"
#include "xla/backends/gpu/transforms/cudnn_custom_call_compiler.h"
#include "xla/backends/gpu/transforms/cudnn_fusion_compiler.h"
#include "xla/backends/gpu/transforms/scaled_dot_rewriter.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/hlo_interpreter_reference_mixin.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

class CudnnScaledDotTestBase
    : public HloInterpreterReferenceMixin<HloPjRtGpuTestBase> {
 public:
  se::StreamExecutor* stream_executor() const {
    auto platform =
        se::PlatformManager::PlatformWithId(stream_executor_platform_id());
    CHECK_OK(platform);
    auto executor = (*platform)->ExecutorForDevice(0);
    CHECK_OK(executor);
    return *executor;
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloPjRtGpuTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_autotune_level(0);
    debug_options.set_xla_gpu_cudnn_gemm_fusion_level(2);
    debug_options.clear_xla_gpu_experimental_autotune_backends();
    debug_options.add_xla_gpu_experimental_autotune_backends(
        autotuner::Backend::CUDNN);
    debug_options.add_xla_gpu_experimental_autotune_backends(
        autotuner::Backend::CUBLASLT);
    debug_options.set_xla_gpu_experimental_scaled_dot_with_triton(false);
    return debug_options;
  }

  se::CudaComputeCapability get_cuda_cc() const {
    return device_description().cuda_compute_capability();
  }

  bool IsAtLeastBlackwellWithCuDnn9() const {
    return get_cuda_cc().IsAtLeastBlackwell() &&
           gpu_target_config()
                   .device_description.dnn_version()
                   .major_version() >= 9;
  }

  absl::Status PopulateScale(Literal* scale, std::minstd_rand0* engine) {
    switch (scale->shape().element_type()) {
      case F8E8M0FNU: {
        absl::uniform_int_distribution<int> exponent_distribution(-4, 1);
        return scale->Populate<float8_e8m0fnu>(
            [&](absl::Span<const int64_t> /*indices*/) {
              return float8_e8m0fnu(
                  std::ldexp(1.0f, exponent_distribution(*engine)));
            });
      }
      case F8E4M3FN:
        for (float8_e4m3fn& value : scale->data<float8_e4m3fn>()) {
          value = float8_e4m3fn(std::abs(static_cast<float>(value)));
        }
        return absl::OkStatus();
      case F8E5M2:
        for (float8_e5m2& value : scale->data<float8_e5m2>()) {
          value = float8_e5m2(std::abs(static_cast<float>(value)));
        }
        return absl::OkStatus();
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported scale type: ",
                         PrimitiveType_Name(scale->shape().element_type())));
    }
  }

  absl::StatusOr<std::vector<Literal>> MakeScaledDotArguments(
      const HloModule* module) {
    std::minstd_rand0 engine;
    FakeArgumentsOptions options;
    options.engine = &engine;
    ABSL_ASSIGN_OR_RETURN(std::vector<Literal> arguments,
                          MakeFakeArguments(module, options));
    if (arguments.size() < 4) {
      return absl::InternalError(absl::StrCat(
          "Expected at least 4 scaled-dot arguments, got ", arguments.size()));
    }
    ABSL_RETURN_IF_ERROR(PopulateScale(&arguments[2], &engine));
    ABSL_RETURN_IF_ERROR(PopulateScale(&arguments[3], &engine));
    return arguments;
  }
};

enum class CuDnnSupportResult {
  kSupported,  // Lowered to native cuDNN block-scaled graph, execution succeeds
  kNotSupported,  // Not supported by cuDNN block scaling, falls back to
                  // decomposed dot
  kFailed,        // Lowered to native cuDNN block-scaled graph, but execution
                  // exhibits discrepancy
};

absl::string_view CuDnnSupportResultToString(CuDnnSupportResult result) {
  switch (result) {
    case CuDnnSupportResult::kSupported:
      return "supported";
    case CuDnnSupportResult::kNotSupported:
      return "not_supported";
    case CuDnnSupportResult::kFailed:
      return "failed";
  }
}

std::ostream& operator<<(std::ostream& os, CuDnnSupportResult result) {
  return os << CuDnnSupportResultToString(result);
}

struct ScaledDotCoverageTestCase {
  PrimitiveType lhs_type;
  PrimitiveType rhs_type;
  PrimitiveType scale_type;
  int block_size;
  bool lhs_k_minor;
  bool rhs_k_minor;
  CuDnnSupportResult expected_result;
};

std::vector<ScaledDotCoverageTestCase> GetCoverageTestCases() {
  return {
      // 1. MXFP8 E4M3FN x E4M3FN (scale E8M0FNU, block 32)
      {F8E4M3FN, F8E4M3FN, F8E8M0FNU, 32, false, false,
       CuDnnSupportResult::kSupported},
      {F8E4M3FN, F8E4M3FN, F8E8M0FNU, 32, false, true,
       CuDnnSupportResult::kSupported},
      {F8E4M3FN, F8E4M3FN, F8E8M0FNU, 32, true, false,
       CuDnnSupportResult::kSupported},
      {F8E4M3FN, F8E4M3FN, F8E8M0FNU, 32, true, true,
       CuDnnSupportResult::kSupported},

      // 2. MXFP8 E4M3FN x E5M2 (scale E8M0FNU, block 32)
      {F8E4M3FN, F8E5M2, F8E8M0FNU, 32, false, false,
       CuDnnSupportResult::kSupported},
      {F8E4M3FN, F8E5M2, F8E8M0FNU, 32, false, true,
       CuDnnSupportResult::kSupported},
      {F8E4M3FN, F8E5M2, F8E8M0FNU, 32, true, false,
       CuDnnSupportResult::kSupported},
      {F8E4M3FN, F8E5M2, F8E8M0FNU, 32, true, true,
       CuDnnSupportResult::kSupported},

      // 3. E4M3FN x F4E2M1FN (scale E8M0FNU, block 32)
      {F8E4M3FN, F4E2M1FN, F8E8M0FNU, 32, false, false,
       CuDnnSupportResult::kNotSupported},
      {F8E4M3FN, F4E2M1FN, F8E8M0FNU, 32, false, true,
       CuDnnSupportResult::kNotSupported},
      {F8E4M3FN, F4E2M1FN, F8E8M0FNU, 32, true, false,
       CuDnnSupportResult::kNotSupported},
      {F8E4M3FN, F4E2M1FN, F8E8M0FNU, 32, true, true,
       CuDnnSupportResult::kNotSupported},

      // 4. MXFP8 E5M2 x E4M3FN (scale E8M0FNU, block 32)
      {F8E5M2, F8E4M3FN, F8E8M0FNU, 32, false, false,
       CuDnnSupportResult::kSupported},
      {F8E5M2, F8E4M3FN, F8E8M0FNU, 32, false, true,
       CuDnnSupportResult::kSupported},
      {F8E5M2, F8E4M3FN, F8E8M0FNU, 32, true, false,
       CuDnnSupportResult::kSupported},
      {F8E5M2, F8E4M3FN, F8E8M0FNU, 32, true, true,
       CuDnnSupportResult::kSupported},

      // 5. MXFP8 E5M2 x E5M2 (scale E8M0FNU, block 32)
      {F8E5M2, F8E5M2, F8E8M0FNU, 32, false, false,
       CuDnnSupportResult::kSupported},
      {F8E5M2, F8E5M2, F8E8M0FNU, 32, false, true,
       CuDnnSupportResult::kSupported},
      {F8E5M2, F8E5M2, F8E8M0FNU, 32, true, false,
       CuDnnSupportResult::kSupported},
      {F8E5M2, F8E5M2, F8E8M0FNU, 32, true, true,
       CuDnnSupportResult::kSupported},

      // 6. E5M2 x F4E2M1FN (scale E8M0FNU, block 32)
      {F8E5M2, F4E2M1FN, F8E8M0FNU, 32, false, false,
       CuDnnSupportResult::kNotSupported},
      {F8E5M2, F4E2M1FN, F8E8M0FNU, 32, false, true,
       CuDnnSupportResult::kNotSupported},
      {F8E5M2, F4E2M1FN, F8E8M0FNU, 32, true, false,
       CuDnnSupportResult::kNotSupported},
      {F8E5M2, F4E2M1FN, F8E8M0FNU, 32, true, true,
       CuDnnSupportResult::kNotSupported},

      // 7. F4E2M1FN x E4M3FN (scale E8M0FNU, block 32)
      {F4E2M1FN, F8E4M3FN, F8E8M0FNU, 32, false, false,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F8E4M3FN, F8E8M0FNU, 32, false, true,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F8E4M3FN, F8E8M0FNU, 32, true, false,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F8E4M3FN, F8E8M0FNU, 32, true, true,
       CuDnnSupportResult::kNotSupported},

      // 8. F4E2M1FN x E5M2 (scale E8M0FNU, block 32)
      {F4E2M1FN, F8E5M2, F8E8M0FNU, 32, false, false,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F8E5M2, F8E8M0FNU, 32, false, true,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F8E5M2, F8E8M0FNU, 32, true, false,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F8E5M2, F8E8M0FNU, 32, true, true,
       CuDnnSupportResult::kNotSupported},

      // 9. F4E2M1FN x F4E2M1FN (scale E8M0FNU, block 32)
      {F4E2M1FN, F4E2M1FN, F8E8M0FNU, 32, false, false,
       CuDnnSupportResult::kSupported},
      {F4E2M1FN, F4E2M1FN, F8E8M0FNU, 32, false, true,
       CuDnnSupportResult::kSupported},
      {F4E2M1FN, F4E2M1FN, F8E8M0FNU, 32, true, false,
       CuDnnSupportResult::kSupported},
      {F4E2M1FN, F4E2M1FN, F8E8M0FNU, 32, true, true,
       CuDnnSupportResult::kSupported},

      // 10. F4E2M1FN x F4E2M1FN (scale E8M0FNU, block 16)
      {F4E2M1FN, F4E2M1FN, F8E8M0FNU, 16, false, false,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F4E2M1FN, F8E8M0FNU, 16, false, true,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F4E2M1FN, F8E8M0FNU, 16, true, false,
       CuDnnSupportResult::kNotSupported},
      {F4E2M1FN, F4E2M1FN, F8E8M0FNU, 16, true, true,
       CuDnnSupportResult::kNotSupported},

      // 11. NVFP4 F4E2M1FN x F4E2M1FN (scale E4M3FN, block 16)
      {F4E2M1FN, F4E2M1FN, F8E4M3FN, 16, false, false,
       CuDnnSupportResult::kSupported},
      {F4E2M1FN, F4E2M1FN, F8E4M3FN, 16, false, true,
       CuDnnSupportResult::kSupported},
      {F4E2M1FN, F4E2M1FN, F8E4M3FN, 16, true, false,
       CuDnnSupportResult::kSupported},
      {F4E2M1FN, F4E2M1FN, F8E4M3FN, 16, true, true,
       CuDnnSupportResult::kSupported},
  };
}

std::string ScaledDotCoverageTestParamToString(
    const ::testing::TestParamInfo<ScaledDotCoverageTestCase>& info) {
  const auto& tc = info.param;
  return absl::StrCat(
      PrimitiveType_Name(tc.lhs_type), "_", PrimitiveType_Name(tc.rhs_type),
      "_", PrimitiveType_Name(tc.scale_type), "_Block", tc.block_size, "_Lhs",
      tc.lhs_k_minor, "_Rhs", tc.rhs_k_minor);
}

class CudnnScaledDotCoverageTest
    : public CudnnScaledDotTestBase,
      public ::testing::WithParamInterface<ScaledDotCoverageTestCase> {};

TEST_P(CudnnScaledDotCoverageTest, Executes) {
  const auto& param = GetParam();
  const bool lhs_k_minor = param.lhs_k_minor;
  const bool rhs_k_minor = param.rhs_k_minor;

  auto cc = get_cuda_cc();
  std::string device_name = "Unknown";
  if (cc.IsAtLeastBlackwell()) {
    device_name = "B200";
  } else if (cc.IsAtLeastHopper()) {
    device_name = "H100";
  } else {
    device_name = "Pre-Hopper";
  }

  std::string lhs_name =
      primitive_util::LowercasePrimitiveTypeName(param.lhs_type);
  std::string rhs_name =
      primitive_util::LowercasePrimitiveTypeName(param.rhs_type);
  std::string scale_name =
      primitive_util::LowercasePrimitiveTypeName(param.scale_type);
  std::string lhs_display =
      lhs_k_minor ? lhs_name : absl::StrCat(lhs_name, ".T");
  std::string rhs_display =
      rhs_k_minor ? absl::StrCat(rhs_name, ".T") : rhs_name;

  LOG(ERROR) << "Report Device: " << device_name;
  LOG(ERROR) << "Report LHS: " << lhs_display;
  LOG(ERROR) << "Report RHS: " << rhs_display;
  LOG(ERROR) << "Report Scale: " << scale_name;
  LOG(ERROR) << "Report BlockSize: " << param.block_size;

  constexpr int64_t m = 128;
  constexpr int64_t n = 128;
  constexpr int64_t k = 256;
  const int64_t scale_k = k / param.block_size;

  const std::string lhs_shape =
      lhs_k_minor ? absl::StrCat(m, ",", k) : absl::StrCat(k, ",", m);
  const std::string rhs_shape =
      rhs_k_minor ? absl::StrCat(n, ",", k) : absl::StrCat(k, ",", n);
  const std::string lhs_scale_shape = lhs_k_minor
                                          ? absl::StrCat(m, ",", scale_k)
                                          : absl::StrCat(scale_k, ",", m);
  const std::string rhs_scale_shape = rhs_k_minor
                                          ? absl::StrCat(n, ",", scale_k)
                                          : absl::StrCat(scale_k, ",", n);

  // 1. Stage: After Composite
  std::string after_composite = "N/A";
  constexpr absl::string_view kCompositeHloTemplate = R"hlo(
HloModule test_composite

%xla.scaled_dot.1 {
  %p0 = $lhs_type[$lhs_shape]{1,0} parameter(0)
  %p1 = $rhs_type[$rhs_shape]{1,0} parameter(1)
  %p2 = $scale_type[$lhs_scale_shape]{1,0} parameter(2)
  %p3 = $scale_type[$rhs_scale_shape]{1,0} parameter(3)
  %zero = bf16[] constant(0)
  ROOT %dummy = bf16[$output_shape]{1,0} broadcast(%zero), dimensions={}
}

ENTRY %main {
  %lhs = $lhs_type[$lhs_shape]{1,0} parameter(0)
  %rhs = $rhs_type[$rhs_shape]{1,0} parameter(1)
  %lhs_scale = $scale_type[$lhs_scale_shape]{1,0} parameter(2)
  %rhs_scale = $scale_type[$rhs_scale_shape]{1,0} parameter(3)
  ROOT %call = bf16[$output_shape]{1,0} call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      to_apply=%xla.scaled_dot.1,
      is_composite=true,
      frontend_attributes={
        composite.attributes="{dimension_numbers=[[[$lhs_contracting_dim],[$rhs_contracting_dim]],[[],[]]]}",
        composite.name="xla.scaled_dot",
        composite.version="1"
      }
}
)hlo";

  std::string composite_hlo =
      absl::StrReplaceAll(kCompositeHloTemplate,
                          {{"$lhs_type", lhs_name},
                           {"$rhs_type", rhs_name},
                           {"$scale_type", scale_name},
                           {"$lhs_shape", lhs_shape},
                           {"$rhs_shape", rhs_shape},
                           {"$lhs_scale_shape", lhs_scale_shape},
                           {"$rhs_scale_shape", rhs_scale_shape},
                           {"$output_shape", absl::StrCat(m, ",", n)},
                           {"$lhs_contracting_dim", lhs_k_minor ? "1" : "0"},
                           {"$rhs_contracting_dim", rhs_k_minor ? "1" : "0"}});

  if (auto comp_module = ParseAndReturnVerifiedModule(composite_hlo);
      comp_module.ok()) {
    CompositeRewriter rewriter;
    if (auto rewrite_status = rewriter.Run(comp_module->get());
        rewrite_status.ok()) {
      const auto* root =
          (*comp_module)->entry_computation()->root_instruction();
      after_composite = HloOpcodeString(root->opcode());
    }
  }
  LOG(ERROR) << "Report AfterComposite: " << after_composite;
  EXPECT_EQ(after_composite, "scaled-dot");

  // 2. Stage: Optimized HLO (compiled directly from composite_hlo)
  std::string optimized_hlo = "N/A";
  HloInstruction* custom_call_or_fusion = nullptr;
  auto optimized_module_or = GetOptimizedModule(composite_hlo);
  if (!optimized_module_or.ok()) {
    LOG(ERROR) << "GetOptimizedModule error: " << optimized_module_or.status();
  }
  if (optimized_module_or.ok()) {
    HloModule& optimized_module = **optimized_module_or;
    for (HloComputation* comp : optimized_module.computations()) {
      for (HloInstruction* instr : comp->instructions()) {
        if (instr->opcode() == HloOpcode::kCustomCall &&
            instr->custom_call_target() == "__cudnn$blockScaledDot") {
          optimized_hlo = "__cudnn$blockScaledDot";
          custom_call_or_fusion = instr;
          break;
        }
        if (instr->opcode() == HloOpcode::kFusion) {
          if (auto gpu_cfg = instr->backend_config<GpuBackendConfig>();
              gpu_cfg.ok() &&
              gpu_cfg->fusion_backend_config().kind() == "__cudnn$fusion") {
            const HloComputation* fused_comp =
                instr->fused_instructions_computation();
            bool has_scaled_dot = false;
            for (const HloInstruction* fused_instr :
                 fused_comp->instructions()) {
              if (fused_instr->opcode() == HloOpcode::kScaledDot) {
                has_scaled_dot = true;
                break;
              }
            }
            optimized_hlo = has_scaled_dot ? "__cudnn$fusion" : "dot";
            custom_call_or_fusion = instr;
            break;
          }
        }
        if (instr->opcode() == HloOpcode::kScaledDot) {
          optimized_hlo = "scaled-dot";
        } else if (instr->opcode() == HloOpcode::kDot &&
                   optimized_hlo == "N/A") {
          optimized_hlo = "dot";
        }
      }
      if (custom_call_or_fusion != nullptr) {
        break;
      }
    }
  }
  LOG(ERROR) << "Report OptimizedHLO: " << optimized_hlo;

  // 3. Stage: cuDNN Graph
  std::string cudnn_graph_pattern = "N/A";
  if (custom_call_or_fusion != nullptr && IsAtLeastBlackwellWithCuDnn9() &&
      optimized_hlo != "dot" && optimized_module_or.ok()) {
    HloModule& optimized_module = **optimized_module_or;
    if (custom_call_or_fusion->opcode() == HloOpcode::kFusion) {
      BinaryMap compilation_results;
      CuDnnFusionCompiler cudnn_compiler(stream_executor()->AsDnn(),
                                         device_description(),
                                         compilation_results);
      auto cloned = optimized_module.Clone();
      auto res = cudnn_compiler.Run(cloned.get());
      if (res.ok()) {
        cudnn_graph_pattern = "BLOCK_SCALE_DEQUANTIZE + MATMUL";
      } else {
        cudnn_graph_pattern = absl::StrCat("FAILED: ", res.status().message());
      }
    } else if (custom_call_or_fusion->opcode() == HloOpcode::kCustomCall) {
      BinaryMap compilation_results;
      CuDnnCustomCallCompiler custom_call_compiler(stream_executor()->AsDnn(),
                                                   device_description(),
                                                   compilation_results);
      auto cloned = optimized_module.Clone();
      auto res = custom_call_compiler.Run(cloned.get());
      if (res.ok() && !compilation_results.empty()) {
        cudnn_graph_pattern = "BLOCK_SCALE_DEQUANTIZE + MATMUL";
      } else if (!res.ok()) {
        cudnn_graph_pattern = absl::StrCat("FAILED: ", res.status().message());
      }
    }
  }

  // 4. Stage: Execution Status
  std::string status = "FAILURE";
  if (!IsAtLeastBlackwellWithCuDnn9()) {
    GTEST_SKIP()
        << "Block scaled dot cuDNN execution requires Blackwell and cuDNN 9+.";
  }
  if (optimized_module_or.ok()) {
    auto ref_module_or = ParseAndReturnVerifiedModule(composite_hlo);
    if (ref_module_or.ok()) {
      CompositeRewriter composite_pass;
      (void)composite_pass.Run(ref_module_or->get());
      ScaledDotRewriter decompose_pass;
      if (auto decomposed = decompose_pass.Run(ref_module_or->get());
          decomposed.ok()) {
        auto ref_optimized_or =
            GetOptimizedModule((*ref_module_or)->ToString());
        if (ref_optimized_or.ok()) {
          absl::StatusOr<std::vector<Literal>> args_or =
              MakeScaledDotArguments(optimized_module_or->get());
          CHECK_OK(args_or.status());
          std::vector<const Literal*> arg_ptrs;
          arg_ptrs.reserve(args_or->size());
          for (const Literal& arg : *args_or) {
            arg_ptrs.push_back(&arg);
          }
          ::testing::AssertionResult run_result = RunAndCompareTwoModules(
              std::move(*optimized_module_or), std::move(*ref_optimized_or),
              arg_ptrs, ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-5},
              /*run_hlo_passes=*/false);
          status = run_result ? "SUCCESS" : "FAILURE";
          if (!run_result) {
            LOG(ERROR) << "RunAndCompare error: " << run_result.message();
          }
        } else {
          LOG(ERROR) << "ref_optimized error: " << ref_optimized_or.status();
        }
      }
    }
  }

  CuDnnSupportResult actual = CuDnnSupportResult::kNotSupported;
  if (optimized_hlo == "__cudnn$fusion" ||
      optimized_hlo == "__cudnn$blockScaledDot") {
    actual = (cudnn_graph_pattern == "BLOCK_SCALE_DEQUANTIZE + MATMUL" &&
              status == "SUCCESS")
                 ? CuDnnSupportResult::kSupported
                 : CuDnnSupportResult::kFailed;
  }

  LOG(ERROR) << "Report CuDnnGraph: " << CuDnnSupportResultToString(actual);
  LOG(ERROR) << "Report CuDnnGraphPattern: " << cudnn_graph_pattern;
  LOG(ERROR) << "Report Status: " << status;

  CuDnnSupportResult expected = param.expected_result;
  EXPECT_EQ(actual, expected);

  switch (expected) {
    case CuDnnSupportResult::kSupported:
      EXPECT_EQ(optimized_hlo, "__cudnn$fusion");
      EXPECT_EQ(cudnn_graph_pattern, "BLOCK_SCALE_DEQUANTIZE + MATMUL");
      EXPECT_EQ(status, "SUCCESS");
      break;
    case CuDnnSupportResult::kFailed:
      EXPECT_EQ(optimized_hlo, "__cudnn$fusion");
      EXPECT_EQ(cudnn_graph_pattern, "BLOCK_SCALE_DEQUANTIZE + MATMUL");
      EXPECT_EQ(status, "FAILURE");
      break;
    case CuDnnSupportResult::kNotSupported:
      EXPECT_EQ(optimized_hlo, "dot");
      EXPECT_EQ(cudnn_graph_pattern, "N/A");
      EXPECT_EQ(status, "SUCCESS");
      break;
  }
}

INSTANTIATE_TEST_SUITE_P(CudnnScaledDotCoverageTestSuite,
                         CudnnScaledDotCoverageTest,
                         ::testing::ValuesIn(GetCoverageTestCases()),
                         ScaledDotCoverageTestParamToString);

TEST_F(CudnnScaledDotTestBase,
       AutoEntryLayoutEmitsZeroOperandTransposeAndLowersToCuDnn) {
  if (!IsAtLeastBlackwellWithCuDnn9()) {
    GTEST_SKIP()
        << "Block scaled dot cuDNN execution requires Blackwell and cuDNN 9+.";
  }
  constexpr absl::string_view kAutoLayoutHlo = R"hlo(
HloModule test_auto_layout

ENTRY %main {
  %lhs = f8e4m3fn[256,128] parameter(0)
  %rhs = f8e4m3fn[256,128] parameter(1)
  %lhs_scale = f8e8m0fnu[8,128] parameter(2)
  %rhs_scale = f8e8m0fnu[8,128] parameter(3)
  ROOT %dot = bf16[128,128] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}
}
)hlo";

  HloModuleConfig config = GetModuleConfigForTest();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(
                           kAutoLayoutHlo, config,
                           HloParserOptions().set_fill_missing_layouts(false)));
  module->mutable_entry_computation_layout()->SetToDefaultLayout();
  for (int i = 0; i < 4; ++i) {
    module->mutable_entry_computation_layout()
        ->mutable_parameter_layout(i)
        ->Clear();
  }

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> optimized_module,
      compiler()->RunHloPasses(std::move(module), stream_executor(),
                               Compiler::CompileOptions{}));

  const HloInstruction* cudnn_fusion = nullptr;
  for (const HloInstruction* instr :
       optimized_module->entry_computation()->instructions()) {
    // Entry computation must not transpose the data operands (0 and 1) when
    // entry parameter layouts are AUTO: LayoutAssignment propagates {0,1}
    // directly to the entry parameters.
    EXPECT_NE(instr->opcode(), HloOpcode::kTranspose)
        << "Unexpected transpose in entry computation: " << instr->ToString();
    if (instr->opcode() == HloOpcode::kFusion) {
      absl::StatusOr<GpuBackendConfig> gpu_cfg =
          instr->backend_config<GpuBackendConfig>();
      if (gpu_cfg.ok() &&
          gpu_cfg->fusion_backend_config().kind() == "__cudnn$fusion") {
        cudnn_fusion = instr;
      }
    }
  }
  ASSERT_NE(cudnn_fusion, nullptr) << optimized_module->ToString();
  const HloInstruction* lhs_swizzle_call = cudnn_fusion->operand(0)->operand(0);
  const HloInstruction* rhs_swizzle_call = cudnn_fusion->operand(1)->operand(0);
  EXPECT_EQ(lhs_swizzle_call->operand(0)->opcode(), HloOpcode::kBitcast);
  EXPECT_EQ(lhs_swizzle_call->operand(0)->operand(0)->opcode(),
            HloOpcode::kParameter);
  EXPECT_EQ(rhs_swizzle_call->operand(0)->opcode(), HloOpcode::kBitcast);
  EXPECT_EQ(rhs_swizzle_call->operand(0)->operand(0)->opcode(),
            HloOpcode::kParameter);

  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(optimized_module->entry_computation_layout()
                  .parameter_layout(i)
                  .shape()
                  .layout()
                  .minor_to_major(),
              LayoutUtil::MakeLayout({0, 1}).minor_to_major());
  }

  BinaryMap compilation_results;
  CuDnnFusionCompiler cudnn_compiler(stream_executor()->AsDnn(),
                                     device_description(), compilation_results);
  std::unique_ptr<HloModule> cloned = optimized_module->Clone();
  EXPECT_TRUE(cudnn_compiler.Run(cloned.get()).ok());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> ref_module,
      ParseAndReturnVerifiedModule(kAutoLayoutHlo, GetModuleConfigForTest()));
  ScaledDotRewriter decompose_pass;
  ASSERT_OK(decompose_pass.Run(ref_module.get()).status());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> ref_optimized,
                       GetOptimizedModule(ref_module->ToString()));
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                       MakeScaledDotArguments(ref_optimized.get()));
  std::vector<const Literal*> arg_ptrs;
  arg_ptrs.reserve(args.size());
  for (const Literal& arg : args) {
    arg_ptrs.push_back(&arg);
  }
  EXPECT_TRUE(RunAndCompareTwoModules(
      std::move(optimized_module), std::move(ref_optimized), arg_ptrs,
      ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}, /*run_hlo_passes=*/false));
}

// =============================================================================
// B200 Scaled Dot Benchmark Suite (cuDNN vs Triton vs Decomposed BF16 Fallback)
// =============================================================================

enum class BenchmarkBackendMode {
  kCuDnn,
  kTriton,
  kDecomposedFallback,
};

absl::string_view BenchmarkBackendModeToString(BenchmarkBackendMode mode) {
  switch (mode) {
    case BenchmarkBackendMode::kCuDnn:
      return "cuDNN";
    case BenchmarkBackendMode::kTriton:
      return "Triton";
    case BenchmarkBackendMode::kDecomposedFallback:
      return "Fallback_BF16";
  }
}

enum class SelectedBackend {
  kCuDnnFusion,
  kTritonScaledDotFusion,
  kDecomposedDotFallback,
  kUnknown,
};

absl::string_view SelectedBackendToString(SelectedBackend backend) {
  switch (backend) {
    case SelectedBackend::kCuDnnFusion:
      return "__cudnn$fusion";
    case SelectedBackend::kTritonScaledDotFusion:
      return "__triton_nested_gemm_fusion";
    case SelectedBackend::kDecomposedDotFallback:
      return "decomposed_dot_cublaslt";
    case SelectedBackend::kUnknown:
      return "unknown";
  }
}

enum class EntryLayoutMode {
  kFixed10,
  kAuto,
};

absl::string_view EntryLayoutModeToString(EntryLayoutMode mode) {
  switch (mode) {
    case EntryLayoutMode::kFixed10:
      return "Fixed_{1,0}";
    case EntryLayoutMode::kAuto:
      return "AUTO";
  }
}

struct BenchmarkShapeCase {
  absl::string_view name;
  int64_t m;
  int64_t n;
  int64_t k;
};

struct BenchmarkFormatCase {
  absl::string_view name;
  PrimitiveType lhs_type;
  PrimitiveType rhs_type;
  PrimitiveType scale_type;
  int block_size;
};

struct BenchmarkTransposeCase {
  absl::string_view name;
  bool lhs_k_minor;
  bool rhs_k_minor;
};

struct BackendVerificationInfo {
  SelectedBackend selected_backend = SelectedBackend::kUnknown;
  bool cudnn_graph_compiled = false;
  int num_data_transposes = 0;
  int num_scale_transposes = 0;
  int num_scale_swizzles = 0;
  std::string lhs_param_layout;
  std::string rhs_param_layout;
};

struct BenchmarkRowResult {
  std::string shape_name;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  std::string format_name;
  std::string transpose_name;
  std::string entry_layout_mode;
  std::string requested_backend;
  BackendVerificationInfo verification;
  double median_us = 0.0;
  double mean_us = 0.0;
  double min_us = 0.0;
  double p90_us = 0.0;
  double achieved_tflops = 0.0;
  double achieved_gbps = 0.0;
  std::string status;
};

class CudnnScaledDotBenchmarkTest : public CudnnScaledDotTestBase {
 protected:
  BackendVerificationInfo VerifySelectedBackend(
      const HloModule& optimized_module, BenchmarkBackendMode requested_mode,
      EntryLayoutMode layout_mode, bool lhs_k_minor, bool rhs_k_minor) {
    BackendVerificationInfo info;
    const ComputationLayout& entry_layout =
        optimized_module.entry_computation_layout();
    int64_t k_dim = 0;
    if (entry_layout.parameter_count() >= 2) {
      const Shape& lhs_shape = entry_layout.parameter_layout(0).shape();
      info.lhs_param_layout = lhs_shape.layout().ToString();
      info.rhs_param_layout =
          entry_layout.parameter_layout(1).shape().layout().ToString();
      if (lhs_shape.dimensions().size() == 2) {
        k_dim = lhs_shape.dimensions(lhs_k_minor ? 1 : 0);
      }
    }

    for (const HloComputation* comp : optimized_module.computations()) {
      if (absl::StrContains(comp->name(), "_swizzle")) {
        ++info.num_scale_swizzles;
        continue;
      }
      for (const HloInstruction* instr : comp->instructions()) {
        if (instr->opcode() == HloOpcode::kTranspose) {
          if (instr->shape().dimensions().size() == 2 && k_dim > 0 &&
              instr->shape().dimensions(1) == k_dim) {
            ++info.num_data_transposes;
          } else {
            ++info.num_scale_transposes;
          }
        }
        if (instr->opcode() == HloOpcode::kFusion) {
          absl::StatusOr<GpuBackendConfig> gpu_cfg =
              instr->backend_config<GpuBackendConfig>();
          if (!gpu_cfg.ok()) {
            continue;
          }
          absl::string_view kind = gpu_cfg->fusion_backend_config().kind();
          bool has_scaled_dot = false;
          for (const HloInstruction* fused_instr :
               instr->fused_instructions_computation()->instructions()) {
            if (fused_instr->opcode() == HloOpcode::kScaledDot) {
              has_scaled_dot = true;
              break;
            }
          }
          if (kind == "__cudnn$fusion" && has_scaled_dot) {
            info.selected_backend = SelectedBackend::kCuDnnFusion;
          } else if ((kind == "__triton_nested_gemm_fusion" ||
                      kind == "__triton_gemm") &&
                     has_scaled_dot) {
            info.selected_backend = SelectedBackend::kTritonScaledDotFusion;
          } else if (kind == "__cudnn$fusion" && !has_scaled_dot &&
                     info.selected_backend == SelectedBackend::kUnknown) {
            info.selected_backend = SelectedBackend::kDecomposedDotFallback;
          }
        } else if (instr->opcode() == HloOpcode::kCustomCall &&
                   absl::StrContains(instr->custom_call_target(),
                                     "__cublas$lt$matmul") &&
                   info.selected_backend == SelectedBackend::kUnknown) {
          info.selected_backend = SelectedBackend::kDecomposedDotFallback;
        } else if (instr->opcode() == HloOpcode::kDot &&
                   info.selected_backend == SelectedBackend::kUnknown) {
          info.selected_backend = SelectedBackend::kDecomposedDotFallback;
        }
      }
    }

    if (info.selected_backend == SelectedBackend::kCuDnnFusion) {
      BinaryMap compilation_results;
      CuDnnFusionCompiler cudnn_compiler(stream_executor()->AsDnn(),
                                         device_description(),
                                         compilation_results);
      std::unique_ptr<HloModule> cloned = optimized_module.Clone();
      absl::Status compile_status = cudnn_compiler.Run(cloned.get()).status();
      info.cudnn_graph_compiled = compile_status.ok();
      EXPECT_TRUE(info.cudnn_graph_compiled)
          << "CuDnnFusionCompiler failed: " << compile_status;
    }

    switch (requested_mode) {
      case BenchmarkBackendMode::kCuDnn: {
        EXPECT_EQ(info.selected_backend, SelectedBackend::kCuDnnFusion);
        EXPECT_TRUE(info.cudnn_graph_compiled);
        EXPECT_EQ(info.num_scale_swizzles, 2);
        EXPECT_EQ(info.num_scale_transposes, 0);
        int expected_data_transposes =
            (layout_mode == EntryLayoutMode::kAuto)
                ? 0
                : ((lhs_k_minor ? 0 : 1) + (rhs_k_minor ? 0 : 1));
        EXPECT_EQ(info.num_data_transposes, expected_data_transposes);
        break;
      }
      case BenchmarkBackendMode::kTriton:
        EXPECT_EQ(info.selected_backend,
                  SelectedBackend::kTritonScaledDotFusion);
        break;
      case BenchmarkBackendMode::kDecomposedFallback:
        EXPECT_EQ(info.selected_backend,
                  SelectedBackend::kDecomposedDotFallback);
        break;
    }
    return info;
  }

  std::string BuildCompositeHlo(const BenchmarkShapeCase& shape_case,
                                const BenchmarkFormatCase& format_case,
                                const BenchmarkTransposeCase& transpose_case) {
    const bool lhs_k_minor = transpose_case.lhs_k_minor;
    const bool rhs_k_minor = transpose_case.rhs_k_minor;
    const int64_t scale_k = shape_case.k / format_case.block_size;

    const std::string lhs_name =
        primitive_util::LowercasePrimitiveTypeName(format_case.lhs_type);
    const std::string rhs_name =
        primitive_util::LowercasePrimitiveTypeName(format_case.rhs_type);
    const std::string scale_name =
        primitive_util::LowercasePrimitiveTypeName(format_case.scale_type);

    const std::string lhs_shape =
        lhs_k_minor ? absl::StrCat(shape_case.m, ",", shape_case.k)
                    : absl::StrCat(shape_case.k, ",", shape_case.m);
    const std::string rhs_shape =
        rhs_k_minor ? absl::StrCat(shape_case.n, ",", shape_case.k)
                    : absl::StrCat(shape_case.k, ",", shape_case.n);
    const std::string lhs_scale_shape =
        lhs_k_minor ? absl::StrCat(shape_case.m, ",", scale_k)
                    : absl::StrCat(scale_k, ",", shape_case.m);
    const std::string rhs_scale_shape =
        rhs_k_minor ? absl::StrCat(shape_case.n, ",", scale_k)
                    : absl::StrCat(scale_k, ",", shape_case.n);

    constexpr absl::string_view kCompositeHloTemplate = R"hlo(
HloModule benchmark_composite

%xla.scaled_dot.1 {
  %p0 = $lhs_type[$lhs_shape]{1,0} parameter(0)
  %p1 = $rhs_type[$rhs_shape]{1,0} parameter(1)
  %p2 = $scale_type[$lhs_scale_shape]{1,0} parameter(2)
  %p3 = $scale_type[$rhs_scale_shape]{1,0} parameter(3)
  %zero = bf16[] constant(0)
  ROOT %dummy = bf16[$output_shape]{1,0} broadcast(%zero), dimensions={}
}

ENTRY %main {
  %lhs = $lhs_type[$lhs_shape]{1,0} parameter(0)
  %rhs = $rhs_type[$rhs_shape]{1,0} parameter(1)
  %lhs_scale = $scale_type[$lhs_scale_shape]{1,0} parameter(2)
  %rhs_scale = $scale_type[$rhs_scale_shape]{1,0} parameter(3)
  ROOT %call = bf16[$output_shape]{1,0} call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      to_apply=%xla.scaled_dot.1,
      is_composite=true,
      frontend_attributes={
        composite.attributes="{dimension_numbers=[[[$lhs_contracting_dim],[$rhs_contracting_dim]],[[],[]]]}",
        composite.name="xla.scaled_dot",
        composite.version="1"
      }
}
)hlo";

    return absl::StrReplaceAll(
        kCompositeHloTemplate,
        {{"$lhs_type", lhs_name},
         {"$rhs_type", rhs_name},
         {"$scale_type", scale_name},
         {"$lhs_shape", lhs_shape},
         {"$rhs_shape", rhs_shape},
         {"$lhs_scale_shape", lhs_scale_shape},
         {"$rhs_scale_shape", rhs_scale_shape},
         {"$output_shape", absl::StrCat(shape_case.m, ",", shape_case.n)},
         {"$lhs_contracting_dim", lhs_k_minor ? "1" : "0"},
         {"$rhs_contracting_dim", rhs_k_minor ? "1" : "0"}});
  }

  BenchmarkRowResult RunSingleBenchmark(
      const BenchmarkShapeCase& shape_case,
      const BenchmarkFormatCase& format_case,
      const BenchmarkTransposeCase& transpose_case, EntryLayoutMode layout_mode,
      BenchmarkBackendMode backend_mode, const HloModule* ref_optimized,
      absl::Span<const Literal* const> ref_arg_ptrs) {
    BenchmarkRowResult row;
    row.shape_name = std::string(shape_case.name);
    row.m = shape_case.m;
    row.n = shape_case.n;
    row.k = shape_case.k;
    row.format_name = std::string(format_case.name);
    row.transpose_name = std::string(transpose_case.name);
    row.entry_layout_mode = std::string(EntryLayoutModeToString(layout_mode));
    row.requested_backend =
        std::string(BenchmarkBackendModeToString(backend_mode));

    const bool lhs_k_minor = transpose_case.lhs_k_minor;
    const bool rhs_k_minor = transpose_case.rhs_k_minor;
    const std::string composite_hlo =
        BuildCompositeHlo(shape_case, format_case, transpose_case);

    HloModuleConfig config = GetModuleConfigForTest();
    DebugOptions& debug_options = config.mutable_debug_options();
    debug_options.clear_xla_gpu_enable_command_buffer();
    switch (backend_mode) {
      case BenchmarkBackendMode::kCuDnn:
        debug_options.set_xla_gpu_autotune_level(0);
        debug_options.set_xla_gpu_cudnn_gemm_fusion_level(2);
        debug_options.clear_xla_gpu_experimental_autotune_backends();
        debug_options.add_xla_gpu_experimental_autotune_backends(
            autotuner::Backend::CUDNN);
        debug_options.add_xla_gpu_experimental_autotune_backends(
            autotuner::Backend::CUBLASLT);
        debug_options.set_xla_gpu_experimental_scaled_dot_with_triton(false);
        break;
      case BenchmarkBackendMode::kTriton:
        debug_options.set_xla_gpu_autotune_level(0);
        debug_options.set_xla_gpu_cudnn_gemm_fusion_level(0);
        debug_options.set_xla_gpu_cublas_fallback(false);
        debug_options.set_xla_gpu_experimental_disable_binary_libraries(true);
        debug_options.clear_xla_gpu_experimental_autotune_backends();
        debug_options.add_xla_gpu_experimental_autotune_backends(
            autotuner::Backend::TRITON);
        debug_options.set_xla_gpu_experimental_scaled_dot_with_triton(true);
        break;
      case BenchmarkBackendMode::kDecomposedFallback:
        debug_options.set_xla_gpu_autotune_level(0);
        debug_options.set_xla_gpu_cudnn_gemm_fusion_level(0);
        debug_options.set_xla_gpu_enable_triton_gemm(false);
        debug_options.clear_xla_gpu_experimental_autotune_backends();
        debug_options.add_xla_gpu_experimental_autotune_backends(
            autotuner::Backend::CUBLASLT);
        debug_options.set_xla_gpu_experimental_scaled_dot_with_triton(false);
        break;
    }

    absl::StatusOr<std::unique_ptr<VerifiedHloModule>> parsed_or =
        ParseAndReturnVerifiedModule(composite_hlo, config);
    if (!parsed_or.ok()) {
      row.status = absl::StrCat("PARSE_ERROR: ", parsed_or.status().message());
      return row;
    }
    std::unique_ptr<VerifiedHloModule> module = std::move(*parsed_or);

    if (backend_mode == BenchmarkBackendMode::kDecomposedFallback) {
      CompositeRewriter composite_pass;
      CHECK_OK(composite_pass.Run(module.get()).status());
      ScaledDotRewriter decompose_pass;
      CHECK_OK(decompose_pass.Run(module.get()).status());
    }

    if (layout_mode == EntryLayoutMode::kAuto) {
      for (int i = 0; i < 4; ++i) {
        module->mutable_entry_computation_layout()
            ->mutable_parameter_layout(i)
            ->Clear();
      }
    }

    absl::StatusOr<std::unique_ptr<HloModule>> optimized_or =
        compiler()->RunHloPasses(std::move(module), stream_executor(),
                                 Compiler::CompileOptions());
    if (!optimized_or.ok()) {
      row.status =
          absl::StrCat("HLO_PASS_ERROR: ", optimized_or.status().message());
      return row;
    }
    std::unique_ptr<HloModule> optimized_module = std::move(*optimized_or);

    row.verification = VerifySelectedBackend(
        *optimized_module, backend_mode, layout_mode, lhs_k_minor, rhs_k_minor);

    // Compute total I/O bytes from the entry computation program shape.
    const ProgramShape program_shape =
        optimized_module->entry_computation_layout().ComputeProgramShape();
    int64_t total_io_bytes = ShapeUtil::ByteSizeOf(program_shape.result());
    for (const Shape& param_shape : program_shape.parameters()) {
      total_io_bytes += ShapeUtil::ByteSizeOf(param_shape);
    }

    // Avoid executing Triton block_size=16 (NVFP4) on Blackwell, which triggers
    // a sticky CUDA_ERROR_INVALID_INSTRUCTION hardware exception on the GPU
    // context (see scaled_dot_device_test.cc, b/496426227).
    if (backend_mode == BenchmarkBackendMode::kTriton &&
        format_case.block_size == 16 && get_cuda_cc().IsAtLeastBlackwell()) {
      row.status = "SKIPPED_CUDA_ERROR_INVALID_INSTRUCTION_B496426227";
      return row;
    }

    if (backend_mode != BenchmarkBackendMode::kDecomposedFallback &&
        ref_optimized != nullptr && !ref_arg_ptrs.empty()) {
      ::testing::AssertionResult cmp_result = RunAndCompareTwoModules(
          optimized_module->Clone(), ref_optimized->Clone(), ref_arg_ptrs,
          ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}, /*run_hlo_passes=*/false);
      EXPECT_TRUE(cmp_result)
          << row.shape_name << " " << row.format_name << " "
          << row.transpose_name << " " << row.entry_layout_mode << " "
          << row.requested_backend << ": " << cmp_result.message();
    }

    absl::StatusOr<std::unique_ptr<Executable>> executable_or =
        compiler()->RunBackend(std::move(optimized_module), stream_executor(),
                               Compiler::CompileOptions());
    if (!executable_or.ok()) {
      row.status =
          absl::StrCat("BACKEND_ERROR: ", executable_or.status().message());
      return row;
    }
    std::unique_ptr<Executable> executable = std::move(*executable_or);

    ProfileOptions profile_options;
    profile_options.redzone_padding_bytes = 0;
    profile_options.should_init_buffers = false;
    std::unique_ptr<GpuProfiler> profiler =
        GpuProfiler::Create(stream_executor(), profile_options);
    CHECK(profiler != nullptr);

    absl::StatusOr<std::unique_ptr<InputBuffers>> buffers_or =
        profiler->CreateInputBuffers(executable.get());
    if (!buffers_or.ok()) {
      row.status =
          absl::StrCat("BUFFER_ALLOC_ERROR: ", buffers_or.status().message());
      return row;
    }
    std::unique_ptr<InputBuffers> buffers = std::move(*buffers_or);

    constexpr int kWarmupIterations = 10;
    constexpr int kTimedIterations = 50;

    for (int i = 0; i < kWarmupIterations; ++i) {
      absl::StatusOr<ProfileResult> warmup_res =
          profiler->Profile(executable.get(), *buffers);
      if (!warmup_res.ok()) {
        row.status =
            absl::StrCat("WARMUP_EXEC_ERROR: ", warmup_res.status().message());
        return row;
      }
    }

    std::vector<double> durations_us;
    durations_us.reserve(kTimedIterations);
    for (int i = 0; i < kTimedIterations; ++i) {
      absl::StatusOr<ProfileResult> prof_res =
          profiler->Profile(executable.get(), *buffers);
      if (!prof_res.ok()) {
        row.status =
            absl::StrCat("TIMED_EXEC_ERROR: ", prof_res.status().message());
        return row;
      }
      durations_us.push_back(absl::ToDoubleMicroseconds(prof_res->duration));
    }

    absl::c_sort(durations_us);
    row.min_us = durations_us.front();
    row.median_us = durations_us[durations_us.size() / 2];
    row.p90_us = durations_us[static_cast<size_t>(durations_us.size() * 0.90)];
    row.mean_us =
        std::accumulate(durations_us.begin(), durations_us.end(), 0.0) /
        static_cast<double>(durations_us.size());

    const double flops = 2.0 * static_cast<double>(shape_case.m) *
                         static_cast<double>(shape_case.n) *
                         static_cast<double>(shape_case.k);
    if (row.median_us > 0.0) {
      row.achieved_tflops = (flops / (row.median_us * 1e-6)) / 1e12;
      row.achieved_gbps =
          (static_cast<double>(total_io_bytes) / (row.median_us * 1e-6)) / 1e9;
    }
    row.status = "OK";
    return row;
  }
};

TEST_F(CudnnScaledDotBenchmarkTest, BenchmarkB200ScaledDotBackends) {
  if (!IsAtLeastBlackwellWithCuDnn9()) {
    GTEST_SKIP() << "B200 benchmark requires Blackwell and cuDNN 9+.";
  }

  const se::DeviceDescription& desc = device_description();
  const int sm_count = desc.core_count();
  const double clock_ghz = desc.clock_rate_ghz();
  const double mem_bw_gbps = static_cast<double>(desc.memory_bandwidth()) / 1e9;
  // B200 Tensor Core theoretical FLOPs/clock/SM (dense):
  // BF16: 8192 FMA = 16384 FLOPs/clk/SM
  // FP8: 16384 FMA = 32768 FLOPs/clk/SM
  // FP4: 32768 FMA = 65536 FLOPs/clk/SM
  const double peak_bf16_tflops =
      static_cast<double>(sm_count) * clock_ghz * 16384.0 / 1e3;
  const double peak_fp8_tflops =
      static_cast<double>(sm_count) * clock_ghz * 32768.0 / 1e3;
  const double peak_fp4_tflops =
      static_cast<double>(sm_count) * clock_ghz * 65536.0 / 1e3;

  LOG(ERROR)
      << "==============================================================";
  LOG(ERROR) << "B200 HARDWARE & ROOFLINE SPECIFICATIONS";
  LOG(ERROR) << "  Device Name:          " << desc.name();
  LOG(ERROR) << "  Compute Capability:   " << get_cuda_cc().ToString();
  LOG(ERROR) << "  cuDNN Version:        " << desc.dnn_version().ToString();
  LOG(ERROR) << "  SM Count:             " << sm_count;
  LOG(ERROR) << "  Max Clock Rate (GHz): "
             << absl::StrFormat("%.3f", clock_ghz);
  LOG(ERROR) << "  HBM Bandwidth (GB/s): "
             << absl::StrFormat("%.1f", mem_bw_gbps);
  LOG(ERROR) << "  L2 Cache Size (MB):   "
             << absl::StrFormat("%.2f",
                                static_cast<double>(desc.l2_cache_size()) /
                                    (1024.0 * 1024.0));
  LOG(ERROR) << "  Theoretical Peak BF16 Tensor Core (TFLOPS): "
             << absl::StrFormat("%.1f", peak_bf16_tflops);
  LOG(ERROR) << "  Theoretical Peak FP8  Tensor Core (TFLOPS): "
             << absl::StrFormat("%.1f", peak_fp8_tflops);
  LOG(ERROR) << "  Theoretical Peak FP4  Tensor Core (TFLOPS): "
             << absl::StrFormat("%.1f", peak_fp4_tflops);
  LOG(ERROR)
      << "==============================================================";

  const std::vector<BenchmarkShapeCase> shapes = {
      {"Prefill_2048x4096x4096", 2048, 4096, 4096},
      {"Decode_128x4096x4096", 128, 4096, 4096},
      {"Decode_256x4096x4096", 256, 4096, 4096},
  };

  const std::vector<BenchmarkFormatCase> formats = {
      {"MXFP8", F8E4M3FN, F8E4M3FN, F8E8M0FNU, 32},
      {"MXFP4", F4E2M1FN, F4E2M1FN, F8E8M0FNU, 32},
      {"NVFP4", F4E2M1FN, F4E2M1FN, F8E4M3FN, 16},
  };

  const std::vector<BenchmarkTransposeCase> transposes = {
      {"NT", /*lhs_k_minor=*/true, /*rhs_k_minor=*/true},
      {"NN", /*lhs_k_minor=*/true, /*rhs_k_minor=*/false},
      {"TN", /*lhs_k_minor=*/false, /*rhs_k_minor=*/false},
      {"TT", /*lhs_k_minor=*/false, /*rhs_k_minor=*/true},
  };

  std::vector<BenchmarkRowResult> results;
  for (const BenchmarkShapeCase& shape : shapes) {
    for (const BenchmarkFormatCase& format : formats) {
      for (const BenchmarkTransposeCase& tr : transposes) {
        std::unique_ptr<HloModule> ref_optimized;
        std::vector<Literal> ref_args;
        std::vector<const Literal*> ref_arg_ptrs;
        if (shape.m == 128) {
          const std::string composite_hlo =
              BuildCompositeHlo(shape, format, tr);
          ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> ref_module,
                               ParseAndReturnVerifiedModule(
                                   composite_hlo, GetModuleConfigForTest()));
          CompositeRewriter composite_pass;
          ASSERT_OK(composite_pass.Run(ref_module.get()).status());
          ScaledDotRewriter decompose_pass;
          ASSERT_OK(decompose_pass.Run(ref_module.get()).status());
          ASSERT_OK_AND_ASSIGN(ref_optimized,
                               GetOptimizedModule(ref_module->ToString()));
          ASSERT_OK_AND_ASSIGN(ref_args,
                               MakeScaledDotArguments(ref_optimized.get()));
          ref_arg_ptrs.reserve(ref_args.size());
          for (const Literal& arg : ref_args) {
            ref_arg_ptrs.push_back(&arg);
          }
        }
        for (EntryLayoutMode layout_mode :
             {EntryLayoutMode::kFixed10, EntryLayoutMode::kAuto}) {
          for (BenchmarkBackendMode backend_mode :
               {BenchmarkBackendMode::kCuDnn, BenchmarkBackendMode::kTriton,
                BenchmarkBackendMode::kDecomposedFallback}) {
            if (backend_mode == BenchmarkBackendMode::kDecomposedFallback &&
                layout_mode == EntryLayoutMode::kAuto) {
              continue;
            }
            BenchmarkRowResult res =
                RunSingleBenchmark(shape, format, tr, layout_mode, backend_mode,
                                   ref_optimized.get(), ref_arg_ptrs);
            LOG(ERROR) << absl::StrFormat(
                "BENCH_ROW | %-22s | %-5s | %-2s | %-11s | %-13s | %-27s | "
                "data_tr=%d | scale_tr=%d | swiz=%d | lhs=%s rhs=%s | "
                "med=%.2f us | mean=%.2f us | min=%.2f us | %.1f TFLOPS | "
                "%.1f GB/s | %s",
                res.shape_name, res.format_name, res.transpose_name,
                res.entry_layout_mode, res.requested_backend,
                SelectedBackendToString(res.verification.selected_backend),
                res.verification.num_data_transposes,
                res.verification.num_scale_transposes,
                res.verification.num_scale_swizzles,
                res.verification.lhs_param_layout,
                res.verification.rhs_param_layout, res.median_us, res.mean_us,
                res.min_us, res.achieved_tflops, res.achieved_gbps, res.status);
            results.push_back(std::move(res));
          }
        }
      }
    }
  }

  LOG(ERROR) << "=== COMPLETED " << results.size() << " BENCHMARK CONFIGS ===";
}

}  // namespace
}  // namespace gpu
}  // namespace xla
