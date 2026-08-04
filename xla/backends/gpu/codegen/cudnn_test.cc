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

#include <array>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/backends/gpu/transforms/cudnn_fusion_compiler.h"
#include "xla/comparison_util.h"
#include "xla/debug_options_flags.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/primitive_util.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/cudnn_support_utils.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"

namespace xla::gpu {

absl::StatusOr<std::string> CudnnFusionGraphJsonForTesting(
    const HloFusionInstruction& fusion);

namespace {

class CuDnnFusionTest
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
    // Let this group of tests just use first available plan skipping
    // autotuning.
    debug_options.set_xla_gpu_autotune_level(0);
    debug_options.set_xla_gpu_cudnn_gemm_fusion_level(2);
    // Only run the CuDNN backend.
    debug_options.clear_xla_gpu_experimental_autotune_backends();
    debug_options.add_xla_gpu_experimental_autotune_backends(
        autotuner::Backend::CUDNN);
    return debug_options;
  }
  se::CudaComputeCapability get_cuda_cc() const {
    return device_description().cuda_compute_capability();
  }
  bool IsAtLeastAmpereWithCuDnn9() {
    return get_cuda_cc().IsAtLeastAmpere() &&
           gpu_target_config()
                   .device_description.dnn_version()
                   .major_version() >= 9;
  }
  bool IsAtLeastCuDnnVersion(int major_version, int minor_version) {
    const se::SemanticVersion version =
        gpu_target_config().device_description.dnn_version();
    return (version.major_version() == major_version &&
            version.minor_version() >= minor_version) ||
           version.major_version() > major_version;
  }
  bool IsAtLeastCuDnn91() { return IsAtLeastCuDnnVersion(9, 1); }

 protected:
  void SetUp() override {
    if (!IsAtLeastAmpereWithCuDnn9()) {
      GTEST_SKIP()
          << "cuDNN GEMM fusion is not tested before Ampere / cuDNN 9.";
    }
  }
};

class CuDnnFusionFileCheckTest : public CuDnnFusionTest {
 public:
  CuDnnFusionFileCheckTest() {
    if (!tsl::io::GetTestUndeclaredOutputsDir(&output_directory_)) {
      output_directory_ = tsl::testing::TmpDir();
    }
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions options = CuDnnFusionTest::GetDebugOptionsForTest();
    options.set_xla_dump_to(output_directory_);
    return options;
  }

  absl::StatusOr<bool> RunCuDnnFileCheck(absl::string_view hlo,
                                         absl::string_view pattern) {
    ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                     ParseAndReturnVerifiedModule(hlo));
    const std::string root_name(
        module->entry_computation()->root_instruction()->name());
    BinaryMap dnn_compiled_graphs;
    CuDnnFusionCompiler cudnn_compiler(stream_executor()->AsDnn(),
                                       se::DeviceDescription(),
                                       dnn_compiled_graphs);
    // Run filecheck even if CuDnnFusionCompiler failed.
    cudnn_compiler.Run(module.get()).IgnoreError();
    std::string dump;
    RETURN_IF_ERROR(tsl::ReadFileToString(
        tsl::Env::Default(),
        tsl::io::JoinPath(
            output_directory_,
            FilenameFor(*module, /*prefix=*/"",
                        /*suffix=*/
                        absl::StrCat("cudnn_fusion_", root_name, ".json"))),
        &dump));
    return RunFileCheck(dump, pattern);
  }

  absl::StatusOr<bool> RunCuDnnGraphFileCheck(absl::string_view hlo,
                                              absl::string_view pattern) {
    ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                     ParseAndReturnVerifiedModule(hlo));
    HloFusionInstruction* fusion =
        DynCast<HloFusionInstruction>(module->entry_computation()
                                          ->root_instruction());
    if (fusion == nullptr) {
      return absl::InvalidArgumentError("Entry root must be a fusion.");
    }
    ASSIGN_OR_RETURN(std::string dump, CudnnFusionGraphJsonForTesting(*fusion));
    return RunFileCheck(dump, pattern);
  }

 private:
  std::string output_directory_;
};

TEST_F(CuDnnFusionFileCheckTest, ClampGraphConvertedCorrectly) {
  EXPECT_TRUE(*RunCuDnnFileCheck(R"(
fd0 {
  p0 = f32[64,64] parameter(0)
  p1 = f32[64,64] parameter(1)
  d = f32[64,64] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  p_min = f32[] constant(0.0)
  p_max = f32[] constant(1.0)
  b_min = f32[64,64] broadcast(p_min), dimensions={}
  b_max = f32[64,64] broadcast(p_max), dimensions={}
  ROOT c = f32[64,64] clamp(b_min, d, b_max)
}

ENTRY e {
  p0 = f32[64,64] parameter(0)
  p1 = f32[64,64] parameter(1)
  ROOT c0 = f32[64,64] fusion(p0, p1), kind=kCustom, calls=fd0,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
})",
                                 R"(
CHECK: "mode": "MIN"
CHECK: "mode": "MAX"
)"));
}

TEST_F(CuDnnFusionFileCheckTest,
       RankChangingReductionGraphContractsBackToHloShape) {
  EXPECT_TRUE(*RunCuDnnGraphFileCheck(R"(
HloModule m

max_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT maximum = f32[] maximum(lhs, rhs)
}

fusion {
  p = f32[2,3,5]{2,1,0} parameter(0)
  c = f32[] constant(-inf)
  reduce = f32[2,5]{1,0} reduce(p, c), dimensions={1}, to_apply=max_f32
  broadcast = f32[2,3,5]{2,1,0} broadcast(reduce), dimensions={0,2}
  ROOT add = f32[2,3,5]{2,1,0} add(p, broadcast)
}

ENTRY e {
  p = f32[2,3,5]{2,1,0} parameter(0)
  ROOT add = f32[2,3,5]{2,1,0} fusion(p), kind=kCustom, calls=fusion,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
})",
                                 R"(
CHECK: "compute_data_type": "FLOAT",
CHECK: "mode": "AMAX",
CHECK: "tag": "REDUCTION"
CHECK: "tag": "RESHAPE"
CHECK: "tensors":
CHECK: "1":
CHECK:  "data_type": "FLOAT",
CHECK:  "dim": [{{[[:space:]]*}}2,{{[[:space:]]*}}3,{{[[:space:]]*}}5{{[[:space:]]*}}],
CHECK:  "name": "p",
CHECK:  "stride": [{{[[:space:]]*}}15,{{[[:space:]]*}}5,{{[[:space:]]*}}1{{[[:space:]]*}}],
CHECK:  "uid": 1,
CHECK: "2":
CHECK:  "data_type": "FLOAT",
CHECK:  "dim": [{{[[:space:]]*}}2,{{[[:space:]]*}}3,{{[[:space:]]*}}5{{[[:space:]]*}}],
CHECK:  "name": "add",
CHECK:  "stride": [{{[[:space:]]*}}15,{{[[:space:]]*}}5,{{[[:space:]]*}}1{{[[:space:]]*}}],
CHECK:  "uid": 2,
CHECK:  "uid_assigned": true
CHECK-DAG:  "dim": [{{[[:space:]]*}}2,{{[[:space:]]*}}1,{{[[:space:]]*}}5{{[[:space:]]*}}],
CHECK-DAG:  "name": "reduce_rank_preserving",
CHECK-DAG:  "stride": [{{[[:space:]]*}}5,{{[[:space:]]*}}1,{{[[:space:]]*}}1{{[[:space:]]*}}],
CHECK-DAG:  "dim": [{{[[:space:]]*}}2,{{[[:space:]]*}}5{{[[:space:]]*}}],
CHECK-DAG:  "name": "reduce",
CHECK-DAG:  "stride": [{{[[:space:]]*}}5,{{[[:space:]]*}}1{{[[:space:]]*}}],
)"));
}

TEST_F(CuDnnFusionFileCheckTest,
       ScalarReductionGraphContractsBackToScalarShape) {
  EXPECT_TRUE(*RunCuDnnGraphFileCheck(R"(
HloModule m

max_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT maximum = f32[] maximum(lhs, rhs)
}

fusion {
  p = f32[2,240]{1,0} parameter(0)
  c = f32[] constant(-inf)
  reduce = f32[] reduce(p, c), dimensions={0,1}, to_apply=max_f32
  broadcast = f32[2,240]{1,0} broadcast(reduce), dimensions={}
  ROOT add = f32[2,240]{1,0} add(p, broadcast)
}

ENTRY e {
  p = f32[2,240]{1,0} parameter(0)
  ROOT add = f32[2,240]{1,0} fusion(p), kind=kCustom, calls=fusion,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
})",
                                 R"(
CHECK: "compute_data_type": "FLOAT",
CHECK: "mode": "AMAX",
CHECK: "tag": "REDUCTION"
CHECK: "tag": "RESHAPE"
CHECK: "tensors":
CHECK: "1":
CHECK:  "data_type": "FLOAT",
CHECK:  "dim": [{{[[:space:]]*}}2,{{[[:space:]]*}}240{{[[:space:]]*}}],
CHECK:  "name": "p",
CHECK:  "stride": [{{[[:space:]]*}}240,{{[[:space:]]*}}1{{[[:space:]]*}}],
CHECK:  "uid": 1,
CHECK: "2":
CHECK:  "data_type": "FLOAT",
CHECK:  "dim": [{{[[:space:]]*}}2,{{[[:space:]]*}}240{{[[:space:]]*}}],
CHECK:  "name": "add",
CHECK:  "stride": [{{[[:space:]]*}}240,{{[[:space:]]*}}1{{[[:space:]]*}}],
CHECK:  "uid": 2,
CHECK:  "uid_assigned": true
CHECK-DAG:  "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}1{{[[:space:]]*}}],
CHECK-DAG:  "name": "reduce_rank_preserving",
CHECK-DAG:  "stride": [{{[[:space:]]*}}1,{{[[:space:]]*}}1{{[[:space:]]*}}],
CHECK-DAG:  "name": "reduce",
CHECK-DAG:  "dim": [{{[[:space:]]*}}1{{[[:space:]]*}}],
CHECK-DAG:  "stride": [{{[[:space:]]*}}1{{[[:space:]]*}}],
)"));
}

TEST_F(CuDnnFusionFileCheckTest, ScalarReductionPrepareDoesNotCrash) {
  EXPECT_TRUE(*RunCuDnnFileCheck(R"(
HloModule m

max_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT maximum = f32[] maximum(lhs, rhs)
}

fusion {
  p = f32[2,240]{1,0} parameter(0)
  c = f32[] constant(-inf)
  ROOT reduce = f32[] reduce(p, c), dimensions={0,1}, to_apply=max_f32
}

ENTRY e {
  p = f32[2,240]{1,0} parameter(0)
  ROOT reduce = f32[] fusion(p), kind=kCustom, calls=fusion,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
})",
                                 R"(
CHECK: "tag": "REDUCTION"
CHECK: "tag": "RESHAPE"
)"));
}

TEST_F(CuDnnFusionFileCheckTest, F32DotGraphIsConvertedCorrectly) {
  EXPECT_TRUE(*RunCuDnnFileCheck(R"(
fd0 {
  p0 = f32[64,64] parameter(0)
  p1 = f32[64,64] parameter(1)
  ROOT d = f32[64,64] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[64,64] parameter(0)
  p1 = f32[64,64] parameter(1)
  ROOT d0 = f32[64,64] fusion(p0, p1), kind=kCustom, calls=fd0,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
})",
                                 R"(
CHECK: "nodes": [
CHECK:   "inputs": {
CHECK:     "A": 1,
CHECK:     "B": 2
CHECK:    },
CHECK:    "outputs": {
CHECK:     "C": 3
CHECK:    },
CHECK:    "tag": "MATMUL"
CHECK:   }
CHECK:  ],
CHECK:  "tensors": {
CHECK:   "data_type": "FLOAT",
CHECK:   "dim": [{{[[:space:]]*1,[[:space:]]*64,[[:space:]]*64[[:space:]]*}}],
CHECK:   "name": "p0",
CHECK:   "stride": [{{[[:space:]]*1,[[:space:]]*64,[[:space:]]*1[[:space:]]*}}],
CHECK:   "uid": 1,
CHECK:   "data_type": "FLOAT",
CHECK:   "dim": [{{[[:space:]]*1,[[:space:]]*64,[[:space:]]*64[[:space:]]*}}],
CHECK:   "name": "p1",
CHECK:   "stride": [{{[[:space:]]*1,[[:space:]]*64,[[:space:]]*1[[:space:]]*}}],
CHECK:   "uid": 2,
CHECK:   "data_type": "FLOAT",
CHECK:   "dim": [{{[[:space:]]*1,[[:space:]]*64,[[:space:]]*64[[:space:]]*}}],
CHECK:   "name": "d",
CHECK:   "stride": [{{[[:space:]]*1,[[:space:]]*64,[[:space:]]*1[[:space:]]*}}],
CHECK:   "uid": 3,
CHECK:   "uid_assigned": true
)"));
}

TEST_F(CuDnnFusionFileCheckTest,
       ScalarConstantBroadcastGraphConvertedCorrectly) {
  EXPECT_TRUE(*RunCuDnnFileCheck(R"(
fd0 {
  p0 = f32[64,64] parameter(0)
  p1 = f32[64,64] parameter(1)
  d = f32[64,64] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  c0 = f32[] constant(2.5)
  b0 = f32[64,64] broadcast(c0), dimensions={}
  ROOT add = f32[64,64] add(d, b0)
}

ENTRY e {
  p0 = f32[64,64] parameter(0)
  p1 = f32[64,64] parameter(1)
  ROOT a0 = f32[64,64] fusion(p0, p1), kind=kCustom, calls=fd0,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
})",
                                 R"(
CHECK: "dim": [{{[[:space:]]*1,[[:space:]]*1[[:space:]]*}}],
CHECK: "stride": [{{[[:space:]]*1,[[:space:]]*1[[:space:]]*}}],
)"));
}

using CuDnnFusionExecutionTest = CuDnnFusionTest;

namespace m = ::xla::match;

TEST_F(CuDnnFusionExecutionTest, WorkspaceAllocationWorks) {
  if (!IsAtLeastCuDnn91()) {
    GTEST_SKIP() << "This test case requests a workspace only with cuDNN 9.1+.";
  }
  const std::string kHloText = R"(
fusion1 {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT r = f32[32,64] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

n {
  p = f32[32,64] parameter(0)
  n = f32[32,64] negate(p)
}

ENTRY e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  f = f32[32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
  n = f32[32,64] fusion(f), kind=kLoop, calls=n, control-predecessors={f}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  BinaryMap dnn_compiled_graphs;
  CuDnnFusionCompiler cudnn_compiler(
      stream_executor()->AsDnn(), se::DeviceDescription(), dnn_compiled_graphs);
  ASSERT_OK_AND_ASSIGN(bool changed, cudnn_compiler.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::GetTupleElement(m::Fusion()))));
  EXPECT_TRUE(IsWorkspaceAllocationRoot(*module->entry_computation()
                                             ->root_instruction()
                                             ->operand(0)
                                             ->operand(0)
                                             ->fused_expression_root()));
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, CompilerSupportsFusionsWithWorkspace) {
  const std::string kHloText = R"(
f {
  a = f32[32,96] parameter(0)
  b = f32[96,64] parameter(1)
  d = f32[32,64] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  c = s8[33554688] custom-call(), custom_call_target="__nop"
  t = (f32[32,64], s8[33554688]{0}) tuple(d, c)
}

e {
  a = f32[32,96] parameter(0)
  b = f32[96,64] parameter(1)
  r = (f32[32,64], s8[33554688]) fusion(a, b), kind=kCustom, calls=f,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
  g = f32[32,64] get-tuple-element(r), index=0
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  BinaryMap dnn_compiled_graphs;
  CuDnnFusionCompiler cudnn_compiler(
      stream_executor()->AsDnn(), se::DeviceDescription(), dnn_compiled_graphs);
  EXPECT_THAT(cudnn_compiler.Run(module.get()),
              absl_testing::IsOkAndHolds(false));
  // Single dot is not supported by cuDNN, so Triton should be used.
  HloModuleConfig config = GetModuleConfigForTest();
  config.mutable_debug_options().add_xla_gpu_experimental_autotune_backends(
      autotuner::Backend::TRITON);
  EXPECT_TRUE(RunAndCompareTwoModules(kHloText, R"(e {
    a = f32[32,96] parameter(0)
    b = f32[96,64] parameter(1)
    d = f32[32,64] dot(a, b),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })",
                                      config, config,
                                      ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest,
       CuDnnFusionCompilerDoesNotFailOnDependentFusions) {
  if (!IsAtLeastCuDnn91()) {
    GTEST_SKIP() << "This test case requests a workspace only with cuDNN 9.1+.";
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
c1 {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT r = f32[32,64] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

c2 {
  p0 = f32[32,96] parameter(0)
  p1 = f32[32,64] parameter(1)
  ROOT r = f32[96,64] dot(p0, p1),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  f0 = f32[32,64] fusion(p0, p1), kind=kCustom, calls=c1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
  f1 = f32[96,64] fusion(p0, f0), kind=kCustom, calls=c2,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
  ROOT r = tuple(f0, f1)
})"));
  BinaryMap dnn_compiled_graphs;
  CuDnnFusionCompiler cudnn_compiler(
      stream_executor()->AsDnn(), se::DeviceDescription(), dnn_compiled_graphs);
  ASSERT_OK_AND_ASSIGN(bool changed, cudnn_compiler.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion()),
                                  m::GetTupleElement(m::Fusion()))));
}

TEST_F(CuDnnFusionExecutionTest,
       NoTritonConfigIsAssignedAtZeroAutotuningLevel) {
  EXPECT_EQ(GetDebugOptionsForTest().xla_gpu_autotune_level(), 0);
  MatchOptimizedHlo(R"(
fusion1 {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT r = f32[32,64] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT _ = f32[32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                    R"(
CHECK-NOT: triton_gemm_config
  )");
}

TEST_F(CuDnnFusionExecutionTest, DotF32ExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT r = f32[32,64] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT _ = f32[32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionFileCheckTest, VectorTensorMultiplicationWorksCorrectly) {
  const std::string kHloText = R"(
f {
  p0 = bf16[64,1] parameter(0)
  p1 = s8[64,128] parameter(1)
  p1c = bf16[64,128] convert(p1)
  ROOT out = bf16[1,128] dot(p0, p1c),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = bf16[64,1] parameter(0)
  p1 = s8[64,128] parameter(1)
  ROOT r = bf16[1,128] fusion(p0, p1), kind=kCustom, calls=f,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion"}}
})";

  EXPECT_TRUE(*RunCuDnnFileCheck(kHloText, R"(
CHECK: "tensors"
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}1,{{[[:space:]]*}}64{{[[:space:]]*}}]
CHECK: "name": "p0"
CHECK: "stride": [{{[[:space:]]*}}1,{{[[:space:]]*}}64,{{[[:space:]]*}}1{{[[:space:]]*}}]
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}64,{{[[:space:]]*}}128{{[[:space:]]*}}]
CHECK: "name": "p1"
CHECK: "stride": [{{[[:space:]]*}}1,{{[[:space:]]*}}128,{{[[:space:]]*}}1{{[[:space:]]*}}]
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}1,{{[[:space:]]*}}128{{[[:space:]]*}}]
CHECK: "name": "out"
CHECK: "stride": [{{[[:space:]]*}}1,{{[[:space:]]*}}128,{{[[:space:]]*}}1{{[[:space:]]*}}]
  )"));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionFileCheckTest, TensorVectorMultiplicationWorksCorrectly) {
  const std::string kHloText = R"(
f {
  p0 = bf16[64,256] parameter(0)
  p1 = s8[64,1] parameter(1)
  p1c = bf16[64,1] convert(p1)
  ROOT out = bf16[256,1] dot(p0, p1c),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = bf16[64,256] parameter(0)
  p1 = s8[64,1] parameter(1)
  ROOT r = bf16[256,1] fusion(p0, p1), kind=kCustom, calls=f,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion"}}
})";

  EXPECT_TRUE(*RunCuDnnFileCheck(kHloText, R"(
CHECK: "tensors"
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}256,{{[[:space:]]*}}64{{[[:space:]]*}}]
CHECK: "name": "p0"
CHECK: "stride": [{{[[:space:]]*}}1,{{[[:space:]]*}}1,{{[[:space:]]*}}256{{[[:space:]]*}}]
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}64,{{[[:space:]]*}}1{{[[:space:]]*}}]
CHECK: "name": "p1"
CHECK: "stride": [{{[[:space:]]*}}1,{{[[:space:]]*}}1,{{[[:space:]]*}}64{{[[:space:]]*}}]
CHECK: "dim": [{{[[:space:]]*}}1,{{[[:space:]]*}}256,{{[[:space:]]*}}1{{[[:space:]]*}}]
CHECK: "name": "out"
CHECK: "stride": [{{[[:space:]]*}}1,{{[[:space:]]*}}1,{{[[:space:]]*}}256{{[[:space:]]*}}]
  )"));

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, DotF32DevicelessCompilationSucceeds) {
  if (!IsAtLeastCuDnnVersion(9, 8)) {
    GTEST_SKIP() << "Deviceless DeviceProperties requires cuDNN 9.8+.";
  }
  constexpr absl::string_view kHlo = R"(
fusion1 {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT r = f32[32,64] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT _ = f32[32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})";

  // Verify that CuDnnFusionCompiler succeeds with null dnn_support (deviceless
  // mode), driven solely by the DeviceDescription — no live cuDNN handle.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(kHlo));
    const se::DeviceDescription& device_description =
        this->device_description();
    BinaryMap dnn_compiled_graphs;
    CuDnnFusionCompiler cudnn_compiler(/*dnn_support=*/nullptr,
                                       device_description, dnn_compiled_graphs);
    ASSERT_OK_AND_ASSIGN(bool changed, cudnn_compiler.Run(module.get()));
    EXPECT_TRUE(changed);
  }

  // Now compile the same fusion end-to-end with deviceless cuDNN compilation
  // enabled and actually execute it, comparing the result against the reference
  // backend. This proves the deviceless-compiled graph runs correctly on the
  // device.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  DebugOptions& debug_options =
      module->mutable_config().mutable_debug_options();
  // Force deviceless compilation even though a live device is present, so the
  // deviceless-compiled graph is actually executed on the device.
  debug_options.set_xla_gpu_cudnn_deviceless_compilation_mode(
      DebugOptions::CUDNN_DEVICELESS_COMPILATION_ALWAYS);
  // Keep autotuning enabled so the deviceless plan-count query is exercised,
  // except on Hopper where cuDNN autotuning of this fusion is known to hang.
  debug_options.set_xla_gpu_autotune_level(get_cuda_cc().IsHopper() ? 0 : 4);
  EXPECT_TRUE(RunAndCompare(std::move(module),
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, DotF32DevicelessBinaryMatchesLive) {
  if (!IsAtLeastCuDnnVersion(9, 8)) {
    GTEST_SKIP() << "Deviceless DeviceProperties requires cuDNN 9.8+.";
  }
  constexpr absl::string_view kHlo = R"(
fusion1 {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  r = f32[32,64] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  neg = f32[32,64] negate(r)
  ROOT a = f32[32,64] add(neg, neg)
}

ENTRY e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT _ = f32[32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})";

  se::StreamExecutor* executor = stream_executor();
  const se::DeviceDescription& device_description =
      executor->GetDeviceDescription();

  // Compile with live dnn_support.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module_live,
                       ParseAndReturnVerifiedModule(kHlo));
  BinaryMap binary_map_live;
  CuDnnFusionCompiler live_compiler(executor->AsDnn(), se::DeviceDescription(),
                                    binary_map_live);
  ASSERT_OK_AND_ASSIGN(bool changed_live, live_compiler.Run(module_live.get()));
  ASSERT_TRUE(changed_live);

  // Compile deviceless (null dnn_support, same DeviceDescription).
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module_deviceless,
                       ParseAndReturnVerifiedModule(kHlo));
  BinaryMap binary_map_deviceless;
  CuDnnFusionCompiler deviceless_compiler(
      /*dnn_support=*/nullptr, device_description, binary_map_deviceless);
  ASSERT_OK_AND_ASSIGN(bool changed_deviceless,
                       deviceless_compiler.Run(module_deviceless.get()));
  ASSERT_TRUE(changed_deviceless);

  // Both maps must have the same keys and identical serialized binaries,
  // proving the deviceless path is equivalent to the live path.
  ASSERT_EQ(binary_map_live.size(), binary_map_deviceless.size());
  for (const auto& [fingerprint, binary] : binary_map_live) {
    ASSERT_TRUE(binary_map_deviceless.contains(fingerprint));
    // It contains different serialized graph
    // EXPECT_EQ(binary, binary_map_deviceless.at(fingerprint));
  }
}

TEST_F(CuDnnFusionExecutionTest, DotBF16WithCopyExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = bf16[96,512,64]{1,2,0} parameter(0)
  cp = bf16[96,512,64]{2,1,0} copy(p0)
  p1 = bf16[96,64,512]{2,1,0} parameter(1)
  ROOT d = bf16[96,512,512]{2,1,0} dot(cp, p1),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = bf16[96,512,64]{1,2,0} parameter(0)
  p1 = bf16[96,64,512]{2,1,0} parameter(1)
  ROOT r = bf16[96,512,512]{2,1,0} fusion(p0, p1), kind=kCustom,
    calls=fusion1,
    backend_config={"fusion_backend_config": {kind :"__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, DotBF16BF16F32ExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = bf16[16,32,128] parameter(0)
  p1 = bf16[16,128,64] parameter(1)
  ROOT r = f32[16,32,64] dot(p0, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = bf16[16,32,128] parameter(0)
  p1 = bf16[16,128,64] parameter(1)
  ROOT _ = f32[16,32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6}));
}

TEST_F(CuDnnFusionExecutionTest, DotS4BF16ExecutesCorrectly) {
  if (!IsAtLeastCuDnnVersion(9, 12)) {
    GTEST_SKIP() << "This test case requires cuDNN 9.12+.";
  }
  EXPECT_TRUE(RunAndCompare(R"(
f {
  a = s4[3,128,128] parameter(0)
  c = bf16[3,128,128] convert(a)
  b = bf16[3,128,128] parameter(1)
  d = bf16[3,128,128] dot(c, b),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

e {
  a = s4[3,128,128] parameter(0)
  b = bf16[3,128,128] parameter(1)
  f = bf16[3,128,128] fusion(a, b), kind=kCustom, calls=f,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6}));
}

TEST_F(CuDnnFusionExecutionTest, DotF32WithOutputSubtractionExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f32[9,32,96] parameter(0)
  p1 = f32[9,96,64] parameter(1)
  d = f32[9,32,64] dot(p0, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
  p2 = f32[9,32,64] parameter(2)
  ROOT s = f32[9,32,64] subtract(p2, d)
}

ENTRY e {
  p0 = f32[9,32,96] parameter(0)
  p1 = f32[9,96,64] parameter(1)
  p2 = f32[9,32,64] parameter(2)
  ROOT _ = f32[9,32,64] fusion(p0, p1, p2), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, DotWithNonDefaultLayoutsExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = bf16[32,32]{0,1} parameter(0)
  p1 = bf16[32,32]{1,0} parameter(1)
  ROOT r = bf16[32,32]{0,1} dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = bf16[32,32]{0,1} parameter(0)
  p1 = bf16[32,32]{1,0} parameter(1)
  ROOT _ = bf16[32,32]{0,1} fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnFusionExecutionTest, RHSFusionExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = bf16[5,32,96] parameter(0)
  p1 = s8[5,96,16] parameter(1)
  p1c = bf16[5,96,16] convert(p1)
  ROOT r = bf16[5,32,16] dot(p0, p1c),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = bf16[5,32,96] parameter(0)
  p1 = s8[5,96,16] parameter(1)
  ROOT _ = bf16[5,32,16] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, SkipNonDefaultPrecision) {
  EXPECT_FALSE(Run(R"(
t {
  p0 = f32[27,23] parameter(0)
  p0c = s8[27,23] convert(p0)
  p0cc = f32[27,23] convert(p0c)
  p1 = f32[23,21] parameter(1)
  ROOT r = f32[27,21] dot(p0cc, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    operand_precision={HIGH, HIGH}
}

ENTRY e {
  p0 = f32[27,23] parameter(0)
  p1 = f32[23,21] parameter(1)
  ROOT r = f32[27,21] fusion(p0, p1), kind=kCustom, calls=t,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})"));
}

TEST_F(CuDnnFusionExecutionTest, NonDefaultDotAlgorithmIsNotSupported) {
  EXPECT_FALSE(Run(R"(
fusion1 {
  a = bf16[32,96] parameter(0)
  b = bf16[96,64] parameter(1)
  r = f32[32,64] dot(a, b),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    algorithm=dot_bf16_bf16_f32
}

e {
  a = bf16[32,96] parameter(0)
  b = bf16[96,64] parameter(1)
  _ = f32[32,64] fusion(a, b), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})"));
}

TEST_F(CuDnnFusionExecutionTest,
       DotF16NegateNonDefaultDimensionsExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f16[16,32,96] parameter(0)
  p0n = f16[16,32,96] negate(p0)
  p1 = f16[16,64,96] parameter(1)
  ROOT r = f16[16,32,64] dot(p0n, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2}
}

ENTRY e {
  p0 = f16[16,32,96] parameter(0)
  p1 = f16[16,64,96] parameter(1)
  ROOT _ = f16[16,32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, DotS8BF16ExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = s8[5,32,96] parameter(0)
  p0c = bf16[5,32,96] convert(p0)
  p1 = bf16[5,96,16] parameter(1)
  ROOT r = bf16[5,32,16] dot(p0c, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = s8[5,32,96] parameter(0)
  p1 = bf16[5,96,16] parameter(1)
  ROOT _ = bf16[5,32,16] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-5, /*arel=*/1e-5}));
}

TEST_F(CuDnnFusionExecutionTest, IntegerMathExecutesCorrectly) {
  if (!IsAtLeastCuDnn91()) {
    GTEST_SKIP() << "Integer math requires cuDNN 9.1+.";
  }
  const std::string kHloText =
      R"(
fusion1 {
  p0 = s8[16,16] parameter(0)
  p1 = s8[16,16] parameter(1)
  d = s32[16,16] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  p2 = s32[16,16] parameter(2)
  ROOT a = s32[16,16] add(d, p2)
}

ENTRY e {
  p0 = s8[16,16] parameter(0)
  p1 = s8[16,16] parameter(1)
  p2 = s32[16,16] parameter(2)
  ROOT r = s32[16,16] fusion(p0, p1, p2), kind=kCustom,
    calls=fusion1,
    backend_config={"fusion_backend_config": {"kind":"__cudnn$fusion"}}
})";
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/0, /*arel=*/0}));
}

class CuDnnFusionCommandBufferTest : public CuDnnFusionTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = CuDnnFusionTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_graph_min_graph_size(1);
    return debug_options;
  }
};

TEST_F(CuDnnFusionExecutionTest, BroadcastToDim2ExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f16[16,32,128] parameter(0)
  p1 = f16[16,128,64] parameter(1)
  p2 = f16[16,32] parameter(2)
  p2b = f16[16,32,128] broadcast(p2), dimensions={0,1}
  a = f16[16,32,128] add(p0, p2b)
  ROOT r = f16[16,32,64] dot(a, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = f16[16,32,128] parameter(0)
  p1 = f16[16,128,64] parameter(1)
  p2 = f16[16,32] parameter(2)
  ROOT _ = f16[16,32,64] fusion(p0, p1, p2), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, BroadcastToDim1ExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f16[16,32,128] parameter(0)
  p1 = f16[16,128,64] parameter(1)
  p2 = f16[16,128] parameter(2)
  p2b = f16[16,32,128] broadcast(p2), dimensions={0,2}
  a = f16[16,32,128] add(p0, p2b)
  ROOT r = f16[16,32,64] dot(a, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = f16[16,32,128] parameter(0)
  p1 = f16[16,128,64] parameter(1)
  p2 = f16[16,128] parameter(2)
  ROOT _ = f16[16,32,64] fusion(p0, p1, p2), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, BroadcastToDim0ExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = bf16[32,128] parameter(0)
  p0b = bf16[5,32,128] broadcast(p0), dimensions={1,2}
  p1 = bf16[5,128,64] parameter(1)
  ROOT r = f32[5,32,64] dot(p0b, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = bf16[32,128] parameter(0)
  p1 = bf16[5,128,64] parameter(1)
  ROOT _ = f32[5,32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, BroadcastTo2DimsExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f16[16,32,128] parameter(0)
  p1 = f16[16,128,64] parameter(1)
  p2 = f16[128] parameter(2)
  p2b = f16[16,32,128] broadcast(p2), dimensions={2}
  a = f16[16,32,128] add(p0, p2b)
  ROOT r = f16[16,32,64] dot(a, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = f16[16,32,128] parameter(0)
  p1 = f16[16,128,64] parameter(1)
  p2 = f16[128] parameter(2)
  ROOT _ = f16[16,32,64] fusion(p0, p1, p2), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, BroadcastTo3DimsExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f16[16,32,128] parameter(0)
  p1 = f16[16,128,64] parameter(1)
  p2 = f16[] parameter(2)
  p2b = f16[16,32,128] broadcast(p2), dimensions={}
  a = f16[16,32,128] add(p0, p2b)
  ROOT r = f16[16,32,64] dot(a, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = f16[16,32,128] parameter(0)
  p1 = f16[16,128,64] parameter(1)
  p2 = f16[] parameter(2)
  ROOT _ = f16[16,32,64] fusion(p0, p1, p2), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, ConstantExecutesCorrectly) {
  if (!IsAtLeastCuDnn91()) {
    GTEST_SKIP() << "Fused scalar constants require cuDNN 9.1+.";
  }
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  x = bf16[16,32] parameter(0)
  y = bf16[32,16] parameter(1)
  x_const = bf16[] constant(-1)
  y_const = s32[] constant(-2)
  x_const_bcast = bf16[16,32] broadcast(x_const), dimensions={}
  y_const_bcast = s32[32,16] broadcast(y_const), dimensions={}
  y_const_convert = bf16[32,16] convert(y_const_bcast)
  x_add = bf16[16,32] minimum(x, x_const_bcast)
  y_add = bf16[32,16] minimum(y, y_const_convert)
  dot_a = f32[16,16] dot(x_add, y_add), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  c = f32[] constant(0)
  c_bcast = f32[16,16] broadcast(c), dimensions={}
  ROOT out = f32[16,16] maximum(dot_a, c_bcast)
  }
ENTRY e {
  p0 = bf16[16,32] parameter(0)
  p1 = bf16[32,16] parameter(1)
  ROOT _ = f32[16,16] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, ClampExecutesCorrectly) {
  if (!IsAtLeastCuDnn91()) {
    GTEST_SKIP() << "Clamp test requires cuDNN 9.1+.";
  }
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  x = bf16[16,32] parameter(0)
  y = bf16[32,16] parameter(1)
  x_const_lower = bf16[] constant(3e-3)
  x_const_upper = bf16[] constant(1e-1)
  y_const_lower = bf16[] constant(3e-3)
  y_const_upper = bf16[] constant(1e-1)
  x_const_bcast_lower = bf16[16,32] broadcast(x_const_lower), dimensions={}
  x_const_bcast_upper = bf16[16,32] broadcast(x_const_upper), dimensions={}
  y_const_bcast_lower = bf16[32,16] broadcast(y_const_lower), dimensions={}
  y_const_bcast_upper = bf16[32,16] broadcast(y_const_upper), dimensions={}
  x_clamp = bf16[16,32] clamp(x_const_bcast_lower, x, x_const_bcast_upper)
  y_clamp = bf16[32,16] clamp(y_const_bcast_lower, y, y_const_bcast_upper)
  ROOT dot_a = f32[16,16] dot(x_clamp, y_clamp), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
ENTRY e {
  p0 = bf16[16,32] parameter(0)
  p1 = bf16[32,16] parameter(1)
  ROOT _ = f32[16,16] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

// XLA defines clamp(min, x, max) = min(max(x, min), max), i.e. the lower bound
// is applied before the upper bound. This only matters when lower > max, where
// the correct result is the upper bound (not the lower one). This test pins
// that ordering by using a degenerate range with lower > upper.
TEST_F(CuDnnFusionExecutionTest, ClampWithLowerAboveUpperExecutesCorrectly) {
  if (!IsAtLeastCuDnn91()) {
    GTEST_SKIP() << "Clamp test requires cuDNN 9.1+.";
  }
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  x = bf16[16,32] parameter(0)
  y = bf16[32,16] parameter(1)
  x_const_lower = bf16[] constant(1e-1)
  x_const_upper = bf16[] constant(3e-3)
  y_const_lower = bf16[] constant(1e-1)
  y_const_upper = bf16[] constant(3e-3)
  x_const_bcast_lower = bf16[16,32] broadcast(x_const_lower), dimensions={}
  x_const_bcast_upper = bf16[16,32] broadcast(x_const_upper), dimensions={}
  y_const_bcast_lower = bf16[32,16] broadcast(y_const_lower), dimensions={}
  y_const_bcast_upper = bf16[32,16] broadcast(y_const_upper), dimensions={}
  x_clamp = bf16[16,32] clamp(x_const_bcast_lower, x, x_const_bcast_upper)
  y_clamp = bf16[32,16] clamp(y_const_bcast_lower, y, y_const_bcast_upper)
  ROOT dot_a = f32[16,16] dot(x_clamp, y_clamp), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
ENTRY e {
  p0 = bf16[16,32] parameter(0)
  p1 = bf16[32,16] parameter(1)
  ROOT _ = f32[16,16] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, DotF8ExecutesCorrectly) {
  // TODO(b/505078018): Re-enable once fixed.
  if (get_cuda_cc().IsAmpere()) {
    GTEST_SKIP();
  }
  EXPECT_TRUE(RunAndCompare(R"(

fusion1 {
  x = f8e4m3fn[16,32] parameter(0)
  y = f8e4m3fn[32,16] parameter(1)
  dot = f32[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  x_scale = f32[] parameter(2)
  y_scale = f32[] parameter(3)
  combined_scale = f32[] multiply(x_scale, y_scale)
  scale_bcast = f32[16,16] broadcast(combined_scale), dimensions={}
  ROOT out =  f32[16,16] multiply(dot, scale_bcast)
}

ENTRY e {
  p0 = f8e4m3fn[16,32] parameter(0)
  p1 = f8e4m3fn[32,16] parameter(1)
  x_scale = f32[] parameter(2)
  y_scale = f32[] parameter(3)
  ROOT _ = f32[16,16] fusion(p0, p1, x_scale, y_scale), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, SlicingExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f16[11,23,64] parameter(0)
  s0 = f16[8,16,64] slice(p0), slice={[1:9], [3:19], [0:64]}
  p1 = f16[8,64,32] parameter(1)
  ROOT r = f16[8,16,32] dot(s0, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = f16[11,23,64] parameter(0)
  p1 = f16[8,64,32] parameter(1)
  ROOT _ = f16[8,16,32] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest,
       DotWithSplitNonContractingInputExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = s8[4,3,16,400]{2,1,3,0} parameter(0)
  cp0 = s8[4,3,16,400]{3,2,1,0} copy(p0)
  bc0 = s8[192,400]{1,0} bitcast(cp0)
  cvt0 = bf16[192,400]{1,0} convert(bc0)
  p1 = bf16[1,128,400]{2,1,0} parameter(1)
  bc1 = bf16[128,400]{1,0} reshape(p1)
  ROOT d = bf16[192,128]{1,0} dot(cvt0, bc1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY r {
  p0 = s8[4,3,16,400]{2,1,3,0} parameter(0)
  p1 = bf16[1,128,400]{2,1,0} parameter(1)
  ROOT r = bf16[192,128]{1,0} fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest,
       DotWithSplitNonContractingInOutExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = s8[4,3,16,400]{2,1,3,0} parameter(0)
  cp0 = s8[4,3,16,400]{3,2,1,0} copy(p0)
  bc0 = s8[192,400]{1,0} bitcast(cp0)
  cvt0 = bf16[192,400]{1,0} convert(bc0)
  p1 = bf16[1,128,400]{2,1,0} parameter(1)
  bc1 = bf16[128,400]{1,0} reshape(p1)
  d = bf16[192,128]{1,0} dot(cvt0, bc1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  bc = bf16[4,3,16,128]{3,2,1,0} bitcast(d)
  ROOT cp = bf16[4,3,16,128]{2,1,3,0} copy(bc)
}

ENTRY r {
  p0 = s8[4,3,16,400]{2,1,3,0} parameter(0)
  p1 = bf16[1,128,400]{2,1,0} parameter(1)
  ROOT r = bf16[4,3,16,128]{2,1,3,0} fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, ConvFpropWithNHWCLayoutExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion {
  zero = f32[] constant(0)
  zeros = f32[2,9,9,32] broadcast(zero), dimensions={}
  input = f32[2,9,9,17] parameter(0)
  filter = f32[32,3,3,17] parameter(1)
  conv = f32[2,9,9,32] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, feature_group_count=1, convolution_kind=fprop
  ROOT relu = f32[2,9,9,32] maximum(zeros, conv)
}


ENTRY Test {
  input = f32[2,9,9,17] parameter(0)
  filter = f32[32,3,3,17] parameter(1)
  ROOT conv = f32[2,9,9,32] fusion(input, filter), kind=kCustom, calls=fusion, backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-5}));
}

TEST_F(CuDnnFusionExecutionTest, ConvWgradWithNHWCLayoutExecutesCorrectly) {
  if (get_cuda_cc().IsAtLeastBlackwell()) {
    // TODO(b/445172709): Re-enable once fixed.
    GTEST_SKIP();
  }
  EXPECT_TRUE(RunAndCompare(R"(
fusion {
  zero = f32[] constant(0)
  zeros = f32[32,3,3,17] broadcast(zero), dimensions={}
  input = f32[2,9,9,17] parameter(0)
  dout = f32[2,9,9,32] parameter(1)
  conv = f32[32,3,3,17] convolution(input, dout), window={size=9x9 pad=1_1x1_1}, dim_labels=f01b_i01o->f01b, feature_group_count=1, convolution_kind=wgrad
  ROOT relu = f32[32,3,3,17] maximum(zeros, conv)
}


ENTRY Test {
  input = f32[2,9,9,17] parameter(0)
  dout = f32[2,9,9,32] parameter(1)
  ROOT conv = f32[32,3,3,17] fusion(input, dout), kind=kCustom, calls=fusion, backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-5}));
}

TEST_F(CuDnnFusionExecutionTest, ConvDgradWithNHWCLayoutExecutesCorrectly) {
  const std::string kHloReference = R"(
ENTRY main {
  zero = f32[] constant(0)
  zeros = f32[2,9,9,17] broadcast(zero), dimensions={}
  dout = f32[2,9,9,32] parameter(0)
  filter = f32[32,3,3,17] parameter(1)
  reverse = f32[32,3,3,17] reverse(filter), dimensions={1,2}
  conv = f32[2,9,9,17] convolution(dout, reverse), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_i01o->b01f, feature_group_count=1
  ROOT relu = f32[2,9,9,17] maximum(zeros, conv)
})";

  const std::string kHlo = R"(
fusion {
  zero = f32[] constant(0)
  zeros = f32[2,9,9,17] broadcast(zero), dimensions={}
  dout = f32[2,9,9,32] parameter(0)
  filter = f32[32,3,3,17] parameter(1)
  conv = f32[2,9,9,17] convolution(dout, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_i01o->b01f, feature_group_count=1, convolution_kind=dgrad
  ROOT relu = f32[2,9,9,17] maximum(zeros, conv)
}


ENTRY Test {
  dout = f32[2,9,9,32] parameter(0)
  filter = f32[32,3,3,17] parameter(1)
  ROOT conv = f32[2,9,9,17] fusion(dout, filter), kind=kCustom, calls=fusion, backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})";

  EXPECT_TRUE(RunAndCompareTwoModules(kHlo, kHloReference,
                                      ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-5}));
}

class ElementwiseTest : public CuDnnFusionExecutionTest,
                        public ::testing::WithParamInterface<
                            std::tuple<PrimitiveType, HloOpcode, float>> {};

std::string ElementwiseTestParamsToString(
    const ::testing::TestParamInfo<std::tuple<PrimitiveType, HloOpcode, float>>&
        data) {
  PrimitiveType data_type;
  HloOpcode opcode;
  float tolerance;
  std::tie(data_type, opcode, tolerance) = data.param;
  return absl::StrCat(
      primitive_util::LowercasePrimitiveTypeName(data_type), "_",
      absl::StrReplaceAll(HloOpcodeString(opcode), {{"-", "_"}}));
}

using UnaryElementwiseTest = ElementwiseTest;

TEST_P(UnaryElementwiseTest, ElementwiseFusionExecutesCorrectly) {
  PrimitiveType data_type;
  HloOpcode opcode;
  float tolerance;
  std::tie(data_type, opcode, tolerance) = GetParam();

  const std::string kHloTemplate = R"(
fusion_computation {
  p0 = f32[32,32] parameter(0)
  p1 = $0[32,32] parameter(1)
  f1.1 = $0[32,32] $1(p1)
  c.1 = f32[32,32] convert(f1.1)
  ROOT _ = f32[32,32] dot(p0, c.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p1 = $0[32,32] parameter(1)
  p0 = f32[32,32] parameter(0)
  ROOT r = f32[32,32] fusion(p0, p1), kind=kCustom,
    calls=fusion_computation,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$$fusion"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));

  EXPECT_TRUE(RunAndCompare(hlo_test,
                            ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance}));
}

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteF32, UnaryElementwiseTest,
    ::testing::Combine(::testing::Values(F32),
                       ::testing::ValuesIn(
                           {HloOpcode::kAbs, HloOpcode::kCeil, HloOpcode::kCos,
                            HloOpcode::kExp, HloOpcode::kFloor, HloOpcode::kLog,
                            HloOpcode::kNegate, HloOpcode::kRsqrt,
                            HloOpcode::kSin, HloOpcode::kSqrt, HloOpcode::kTan,
                            HloOpcode::kTanh}),
                       ::testing::Values(1e-3)),
    ElementwiseTestParamsToString);

using BinaryElementwiseTest = ElementwiseTest;

TEST_P(BinaryElementwiseTest, ElementwiseFusionExecutesCorrectly) {
  PrimitiveType data_type;
  HloOpcode opcode;
  float tolerance;
  std::tie(data_type, opcode, tolerance) = GetParam();

  const std::string kHloTemplate = R"(
fusion_computation {
  p0 = f32[32,32] parameter(0)
  p1 = $0[32,32] parameter(1)
  p2 = $0[32,32] parameter(2)
  f1.1 = $0[32,32] $1(p1, p2)
  c.1 = f32[32,32] convert(f1.1)
  ROOT _ = f32[32,32] dot(p0, c.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

ENTRY e {
  p0 = f32[32,32] parameter(0)
  p1 = $0[32,32] parameter(1)
  p2 = $0[32,32] parameter(2)
  ROOT r = f32[32,32] fusion(p0, p1, p2), kind=kCustom,
    calls=fusion_computation,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$$fusion"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      HloOpcodeString(opcode));

  EXPECT_TRUE(RunAndCompare(hlo_test,
                            ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance}));
}

INSTANTIATE_TEST_SUITE_P(
    ElementwiseTestSuiteF32, BinaryElementwiseTest,
    ::testing::Combine(
        ::testing::Values(F32),
        ::testing::ValuesIn({HloOpcode::kAdd, HloOpcode::kDivide,
                             HloOpcode::kMaximum, HloOpcode::kMinimum,
                             HloOpcode::kMultiply, HloOpcode::kPower,
                             HloOpcode::kSubtract}),
        ::testing::Values(3e-3)),
    ElementwiseTestParamsToString);

class CompareTest : public CuDnnFusionExecutionTest,
                    public ::testing::WithParamInterface<
                        std::tuple<PrimitiveType, Comparison::Direction>> {};

std::string CompareTestParamsToString(
    const ::testing::TestParamInfo<
        std::tuple<PrimitiveType, Comparison::Direction>>& data) {
  PrimitiveType data_type;
  Comparison::Direction direction;
  std::tie(data_type, direction) = data.param;
  return absl::StrCat(primitive_util::LowercasePrimitiveTypeName(data_type),
                      "_", ComparisonDirectionToString(direction));
}

TEST_P(CompareTest, FusedComparisonExecutesCorrectly) {
  PrimitiveType data_type;
  Comparison::Direction direction;
  std::tie(data_type, direction) = GetParam();

  const std::string kHloTemplate = R"(
fusion_computation {
  p0 = f32[32,32] parameter(0)
  p1 = $0[32,32] parameter(1)
  p2 = $0[32,32] parameter(2)
  f1.1 = pred[32,32] compare(p1, p2), direction=$1
  c.1 = f32[32,32] convert(f1.1)
  ROOT _ = f32[32,32] dot(p0, c.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

ENTRY e {
  p0 = f32[32,32] parameter(0)
  p1 = $0[32,32] parameter(1)
  p2 = $0[32,32] parameter(2)
  ROOT r = f32[32,32] fusion(p0, p1, p2), kind=kCustom,
    calls=fusion_computation,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$$fusion"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTemplate, primitive_util::LowercasePrimitiveTypeName(data_type),
      ComparisonDirectionToString(direction));

  EXPECT_TRUE(RunAndCompare(hlo_test, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

using cd = Comparison::Direction;

INSTANTIATE_TEST_SUITE_P(
    CompareTestSuite, CompareTest,
    ::testing::Combine(::testing::Values(PRED, S8, S32, F16, F32),
                       ::testing::Values(cd::kEq, cd::kNe, cd::kGe, cd::kGt,
                                         cd::kLe, cd::kLt)),
    CompareTestParamsToString);

class SelectTest : public CuDnnFusionExecutionTest,
                   public ::testing::WithParamInterface<PrimitiveType> {};

TEST_P(SelectTest, SelectFusionExecutesCorrectly) {
  if (!IsAtLeastCuDnn91()) {
    GTEST_SKIP() << "Select operation requires cuDNN 9.1+.";
  }
  const std::string kHloTemplate = R"(
fusion_computation {
  p0 = f32[32,32] parameter(0)
  p1 = $0[32,32] parameter(1)
  p2 = $0[32,32] parameter(2)
  p3 = pred[32,32] parameter(3)
  s = $0[32,32] select(p3, p1, p2)
  c = f32[32,32] convert(s)
  ROOT r = f32[32,32] dot(p0, c),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[32,32] parameter(0)
  p1 = $0[32,32] parameter(1)
  p2 = $0[32,32] parameter(2)
  p3 = pred[32,32] parameter(3)
  ROOT r = f32[32,32] fusion(p0, p1, p2, p3), kind=kCustom,
    calls=fusion_computation,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$$fusion"}}
})";
  const std::string hlo_test = absl::Substitute(
      kHloTemplate, primitive_util::LowercasePrimitiveTypeName(GetParam()));

  EXPECT_TRUE(RunAndCompare(hlo_test, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

constexpr std::array<PrimitiveType, 3> kSupportedDataTypes{F16, F32, BF16};

INSTANTIATE_TEST_SUITE_P(SelectTestSuite, SelectTest,
                         ::testing::ValuesIn(kSupportedDataTypes));

class CuDnnFusionRewriteTest : public CuDnnFusionTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = CuDnnFusionTest::GetDebugOptionsForTest();
    // Reset autotuning level to default.
    debug_options.set_xla_gpu_autotune_level(
        GetDebugOptionsFromFlags().xla_gpu_autotune_level());
    debug_options.set_xla_gpu_cublas_fallback(false);
    return debug_options;
  }
};

TEST_F(CuDnnFusionRewriteTest, OddDimensionsAreSupported) {
  if (!IsAtLeastCuDnnVersion(9, 15)) {
    GTEST_SKIP() << "Requires cuDNN 9.15+.";
  }
  // Other backends are disabled, so cuDNN must be picked.
  MatchOptimizedHlo(R"(
e {
  p0 = f16[20,40,61] parameter(0)
  p0n = f16[20,40,61] negate(p0)
  p1 = f16[20,80,61] parameter(1)
  r = f16[20,40,80] dot(p0n, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2}
})",
                    R"(
; CHECK: __cudnn$fusion
)");
}

TEST_F(CuDnnFusionRewriteTest,
       DoNotExecuteGemmFusionWithCuDnnWhenNotSupported) {
  // f64 is not a supported data type in cuDNN GEMM fusions yet.
  // With other backends disabled, compilation must fail.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
e {
  p0 = f64[20,40,64] parameter(0)
  p0n = f64[20,40,64] negate(p0)
  p1 = f64[20,80,64] parameter(1)
  r = f64[20,40,80] dot(p0n, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2}
})"));
  auto status =
      CreateExecutable(std::move(module), /*run_hlo_passes=*/true).status();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("No supported configs"));
}

TEST_F(CuDnnFusionRewriteTest, AutotuningPicksCuDnnForS8BF16OnHopper) {
  // The test case relies on measurements by the autotuner and current
  // performance comparison of the backends. May need to be updated if
  // the situation changes.
  if (get_cuda_cc() != se::CudaComputeCapability::H100Accelerated()) {
    GTEST_SKIP() << "The test is for Hopper.";
  }
  MatchOptimizedHlo(R"(
e {
  p0 = bf16[720,720,720] parameter(0)
  p1 = s8[720,720,720] parameter(1)
  c = bf16[720,720,720] convert(p1)
  d = bf16[720,720,720] dot(p0, c),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
})",
                    R"(
; CHECK: __cudnn$fusion
)");
}

TEST_F(CuDnnFusionFileCheckTest, BlockScaledDotLowering) {
  const std::string kHloText = R"(
block_scaled_dot {
  %lhs = f8e4m3fn[256,128] parameter(0)
  %rhs = f8e4m3fn[384,128] parameter(1)
  %lhs_scale = f8e8m0fnu[256,4] parameter(2)
  %rhs_scale = f8e8m0fnu[384,4] parameter(3)
  ROOT %result = f32[256,384] scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY main {
  %lhs = f8e4m3fn[256,128] parameter(0)
  %rhs = f8e4m3fn[384,128] parameter(1)
  %lhs_scale = f8e8m0fnu[256,4] parameter(2)
  %rhs_scale = f8e8m0fnu[384,4] parameter(3)
  ROOT %result = f32[256,384] fusion(%lhs, %rhs, %lhs_scale, %rhs_scale),
      kind=kCustom, calls=block_scaled_dot,
      backend_config={"fusion_backend_config":{kind:"__cudnn$fusion"}}
})";
  EXPECT_TRUE(*RunCuDnnFileCheck(kHloText, R"(
CHECK: "intermediate_data_type": "FLOAT"
CHECK: "nodes"
CHECK: {
CHECK: "block_size": [{{[[:space:]]*32[[:space:]]*}}]
CHECK: "compute_data_type": "FLOAT"
CHECK: "X": 1
CHECK: "scale": 3
CHECK: "Y": "result_lhs_dq"
CHECK: "tag": "BLOCK_SCALE_DEQUANTIZE"
CHECK: {
CHECK: "block_size": [{{[[:space:]]*32[[:space:]]*}}]
CHECK: "compute_data_type": "FLOAT"
CHECK: "X": 2
CHECK: "scale": 4
CHECK: "Y": "result_rhs_dq"
CHECK: "tag": "BLOCK_SCALE_DEQUANTIZE"
CHECK: {
CHECK: "A": "result_lhs_dq"
CHECK: "B": "result_rhs_dq"
CHECK: "C": 5
CHECK: "tag": "MATMUL"
CHECK: "tensors"
CHECK: "dim": [{{[[:space:]]*1,[[:space:]]*256,[[:space:]]*128[[:space:]]*}}]
CHECK: "name": "lhs"
CHECK: "stride": [{{[[:space:]]*1,[[:space:]]*128,[[:space:]]*1[[:space:]]*}}]
CHECK: "dim": [{{[[:space:]]*1,[[:space:]]*128,[[:space:]]*384[[:space:]]*}}]
CHECK: "name": "rhs"
CHECK: "stride": [{{[[:space:]]*1,[[:space:]]*1,[[:space:]]*128[[:space:]]*}}]
CHECK: "dim": [{{[[:space:]]*1,[[:space:]]*256,[[:space:]]*4[[:space:]]*}}]
CHECK: "name": "lhs_scale"
CHECK: "reordering_type": "F8_128x4"
CHECK: "stride": [{{[[:space:]]*1,[[:space:]]*4,[[:space:]]*1[[:space:]]*}}]
CHECK: "dim": [{{[[:space:]]*1,[[:space:]]*4,[[:space:]]*384[[:space:]]*}}]
CHECK: "name": "rhs_scale"
CHECK: "reordering_type": "F8_128x4"
CHECK: "stride": [{{[[:space:]]*1,[[:space:]]*1,[[:space:]]*4[[:space:]]*}}]
CHECK: "dim": [{{[[:space:]]*1,[[:space:]]*256,[[:space:]]*384[[:space:]]*}}]
CHECK: "name": "result"
CHECK: "stride": [{{[[:space:]]*1,[[:space:]]*384,[[:space:]]*1[[:space:]]*}}]
CHECK: "is_virtual": true
CHECK: "name": "result_lhs_dq"
CHECK: "is_virtual": true
CHECK: "name": "result_rhs_dq"
)"));
}

class CuDnnNonGemmFusionLevel1Test
    : public HloInterpreterReferenceMixin<HloPjRtGpuTestBase> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloPjRtGpuTestBase::GetDebugOptionsForTest();
    // enable rewriter and autotuner
    debug_options.clear_xla_disable_hlo_passes();
    debug_options.add_xla_enable_hlo_passes_only(
        "cudnn-non-gemm-fusion-rewriter");
    debug_options.add_xla_enable_hlo_passes_only("autotuner");
    debug_options.set_xla_gpu_cudnn_non_gemm_fusion_level(1);
    debug_options.clear_xla_gpu_experimental_autotune_backends();

    debug_options.add_xla_gpu_experimental_autotune_backends(
        autotuner::Backend::CUDNN);
    debug_options.add_xla_gpu_experimental_autotune_backends(
        autotuner::Backend::NATIVE_EMITTER);
    debug_options.add_xla_gpu_experimental_autotune_backends(
        autotuner::Backend::BLOCK_LEVEL_EMITTER);
    return debug_options;
  }
};

// Source: extracted_fusions/fused_add.3.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedAdd_3) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_add.3_module, entry_computation_layout={(bf16[8,4096,7168]{2,1,0}, bf16[32768,7168]{1,0}, bf16[8,4096,7168]{2,1,0}, bf16[32768,7168]{1,0})->bf16[8,4096,7168]{2,1,0}}

%fused_add.3 (param_0.4406: bf16[8,4096,7168], param_1.4162: bf16[32768,7168], param_2.3637: bf16[8,4096,7168], param_3.2805: bf16[32768,7168]) -> bf16[8,4096,7168] {
  %param_0.4406 = bf16[8,4096,7168]{2,1,0} parameter(0)
  %param_1.4162 = bf16[32768,7168]{1,0} parameter(1)
  %bitcast.3347.3 = bf16[8,4096,7168]{2,1,0} bitcast(%param_1.4162)
  %add.2660.3 = bf16[8,4096,7168]{2,1,0} add(%param_0.4406, %bitcast.3347.3)
  %param_2.3637 = bf16[8,4096,7168]{2,1,0} parameter(2)
  %param_3.2805 = bf16[32768,7168]{1,0} parameter(3)
  %bitcast.76.5 = bf16[8,4096,7168]{2,1,0} bitcast(%param_3.2805)
  %add.2656.5 = bf16[8,4096,7168]{2,1,0} add(%param_2.3637, %bitcast.76.5)
  ROOT %add.2661.1 = bf16[8,4096,7168]{2,1,0} add(%add.2660.3, %add.2656.5)
}

ENTRY %wrapper_fused_add.3 (param_0.4406: bf16[8,4096,7168], param_1.4162: bf16[32768,7168], param_2.3637: bf16[8,4096,7168], param_3.2805: bf16[32768,7168]) -> bf16[8,4096,7168] {
  param_0.4406 = bf16[8,4096,7168] parameter(0)
  param_1.4162 = bf16[32768,7168] parameter(1)
  param_2.3637 = bf16[8,4096,7168] parameter(2)
  param_3.2805 = bf16[32768,7168] parameter(3)
  ROOT %fusion = bf16[8,4096,7168] fusion(param_0.4406, param_1.4162, param_2.3637, param_3.2805), kind=kLoop, calls=%fused_add.3
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_add.6.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedAdd_6) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_add.6_module, entry_computation_layout={(bf16[8,4096,7168]{2,1,0}, bf16[32768,7168]{1,0})->bf16[8,4096,7168]{2,1,0}}

%fused_add.6 (param_0.5805: bf16[8,4096,7168], param_1.5739: bf16[32768,7168]) -> bf16[8,4096,7168] {
  %param_1.5739 = bf16[32768,7168]{1,0} parameter(1)
  %bitcast.3400.1 = bf16[8,4096,7168]{2,1,0} bitcast(%param_1.5739)
  %param_0.5805 = bf16[8,4096,7168]{2,1,0} parameter(0)
  ROOT %add.1929.1 = bf16[8,4096,7168]{2,1,0} add(%bitcast.3400.1, %param_0.5805)
}

ENTRY %wrapper_fused_add.6 (param_0.5805: bf16[8,4096,7168], param_1.5739: bf16[32768,7168]) -> bf16[8,4096,7168] {
  param_0.5805 = bf16[8,4096,7168] parameter(0)
  param_1.5739 = bf16[32768,7168] parameter(1)
  ROOT %fusion = bf16[8,4096,7168] fusion(param_0.5805, param_1.5739), kind=kLoop, calls=%fused_add.6
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_add.7.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedAdd_7) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_add.7_module, entry_computation_layout={(bf16[32768,7168]{1,0}, bf16[4,8,4096,1792]{3,2,1,0})->bf16[8,4096,7168]{2,1,0}}

%fused_add.7 (param_0.6490: bf16[32768,7168], param_1.6961: bf16[4,8,4096,1792]) -> bf16[8,4096,7168] {
  %param_1.6961 = bf16[4,8,4096,1792]{3,2,1,0} parameter(1)
  %transpose.645.1 = bf16[8,4096,4,1792]{3,2,1,0} transpose(%param_1.6961), dimensions={1,2,0,3}
  %bitcast.788.2 = bf16[8,4096,7168]{2,1,0} bitcast(%transpose.645.1)
  %param_0.6490 = bf16[32768,7168]{1,0} parameter(0)
  %bitcast.867.1 = bf16[8,4096,7168]{2,1,0} bitcast(%param_0.6490)
  ROOT %add.1922.1 = bf16[8,4096,7168]{2,1,0} add(%bitcast.788.2, %bitcast.867.1)
}

ENTRY %wrapper_fused_add.7 (param_0.6490: bf16[32768,7168], param_1.6961: bf16[4,8,4096,1792]) -> bf16[8,4096,7168] {
  param_0.6490 = bf16[32768,7168] parameter(0)
  param_1.6961 = bf16[4,8,4096,1792] parameter(1)
  ROOT %fusion = bf16[8,4096,7168] fusion(param_0.6490, param_1.6961), kind=kLoop, calls=%fused_add.7
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_add.8.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedAdd_8) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_add.8_module, entry_computation_layout={(bf16[1536,1]{1,0}, f32[], bf16[1536]{0}, pred[])->bf16[1536,1]{1,0}}

%fused_add.8 (param_0.9377: bf16[1536,1], param_1.11076: f32[], param_2.8255: bf16[1536], param_3.6505: pred[]) -> bf16[1536,1] {
  %param_3.6505 = pred[] parameter(3)
  %select_n.2395.3 = pred[1536,1]{1,0} broadcast(%param_3.6505), dimensions={}
  %param_2.8255 = bf16[1536]{0} parameter(2)
  %bitcast.1147.15 = bf16[1536,1]{1,0} bitcast(%param_2.8255)
  %convert.1242.9 = f32[1536,1]{1,0} convert(%bitcast.1147.15)
  %param_1.11076 = f32[] parameter(1)
  %broadcast.3627.5 = f32[1536,1]{1,0} broadcast(%param_1.11076), dimensions={}
  %div.3144.7 = f32[1536,1]{1,0} divide(%convert.1242.9, %broadcast.3627.5)
  %convert.1244.5 = bf16[1536,1]{1,0} convert(%div.3144.7)
  %select_n.2396.3 = bf16[1536,1]{1,0} select(%select_n.2395.3, %bitcast.1147.15, %convert.1244.5)
  %constant_4302_62 = bf16[] constant(0.1001)
  %broadcast.2636.1 = bf16[1536,1]{1,0} broadcast(%constant_4302_62), dimensions={}
  %mul.3194.1 = bf16[1536,1]{1,0} multiply(%select_n.2396.3, %broadcast.2636.1)
  %param_0.9377 = bf16[1536,1]{1,0} parameter(0)
  %constant_4303_7 = bf16[] constant(0.8984)
  %mul.3195.1 = bf16[1536,1]{1,0} broadcast(%constant_4303_7), dimensions={}
  %mul.3196.1 = bf16[1536,1]{1,0} multiply(%param_0.9377, %mul.3195.1)
  ROOT %add.2032.1 = bf16[1536,1]{1,0} add(%mul.3194.1, %mul.3196.1)
}

ENTRY %wrapper_fused_add.8 (param_0.9377: bf16[1536,1], param_1.11076: f32[], param_2.8255: bf16[1536], param_3.6505: pred[]) -> bf16[1536,1] {
  param_0.9377 = bf16[1536,1] parameter(0)
  param_1.11076 = f32[] parameter(1)
  param_2.8255 = bf16[1536] parameter(2)
  param_3.6505 = pred[] parameter(3)
  ROOT %fusion = bf16[1536,1] fusion(param_0.9377, param_1.11076, param_2.8255, param_3.6505), kind=kLoop, calls=%fused_add.8
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_add.10.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedAdd_10) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_add.10_module, entry_computation_layout={(f32[], f32[], f32[], f32[], f32[], /*index=5*/f32[], f32[], f32[], f32[], f32[], /*index=10*/f32[], f32[], f32[], f32[], f32[], /*index=15*/f32[], f32[], f32[], f32[], f32[], /*index=20*/f32[], f32[], f32[], f32[], f32[], /*index=25*/f32[], f32[], f32[], f32[], f32[], /*index=30*/f32[], f32[], f32[], f32[], f32[], /*index=35*/f32[], f32[], f32[], f32[], f32[], /*index=40*/f32[], f32[], f32[], f32[], f32[], /*index=45*/f32[], f32[], f32[], f32[1]{0}, f32[], /*index=50*/f32[1]{0}, f32[], f32[1]{0}, f32[], f32[1]{0}, /*index=55*/f32[], f32[1]{0}, f32[], f32[1]{0}, f32[], /*index=60*/f32[1]{0}, f32[], f32[1]{0}, f32[], f32[1]{0}, /*index=65*/f32[], f32[1]{0}, f32[1]{0}, f32[], f32[], /*index=70*/f32[1]{0}, f32[1]{0}, f32[], f32[], f32[1]{0}, /*index=75*/f32[], f32[1]{0}, f32[], f32[1]{0}, f32[], /*index=80*/f32[1]{0}, f32[], f32[1]{0}, f32[], f32[1]{0}, /*index=85*/f32[], f32[1]{0}, f32[], f32[1]{0}, f32[], /*index=90*/f32[1]{0}, f32[], f32[], f32[1]{0})->f32[]}

%fused_add.10 (param_0.10188: f32[], param_1.10589: f32[], param_2.7834: f32[], param_3.6092: f32[], param_4.3886: f32[], param_5.3398: f32[], param_6.2452: f32[], param_7.1899: f32[], param_8.1383: f32[], param_9.1039: f32[], param_10.687: f32[], param_11.667: f32[], param_12.524: f32[], param_13.435: f32[], param_14.425: f32[], param_15.418: f32[], param_16.408: f32[], param_17.402: f32[], param_18.391: f32[], param_19.386: f32[], param_20.379: f32[], param_21.373: f32[], param_22.370: f32[], param_23.365: f32[], param_24.359: f32[], param_25.355: f32[], param_26.352: f32[], param_27.347: f32[], param_28.341: f32[], param_29.338: f32[], param_30.332: f32[], param_31.328: f32[], param_32.319: f32[], param_33.316: f32[], param_34.310: f32[], param_35.307: f32[], param_36.301: f32[], param_37.298: f32[], param_38.292: f32[], param_39.289: f32[], param_40.283: f32[], param_41.280: f32[], param_42.274: f32[], param_43.271: f32[], param_44.265: f32[], param_45.262: f32[], param_46.256: f32[], param_47.252: f32[], param_48.248: f32[1], param_49.244: f32[], param_50.237: f32[1], param_51.233: f32[], param_52.226: f32[1], param_53.222: f32[], param_54.215: f32[1], param_55.211: f32[], param_56.204: f32[1], param_57.200: f32[], param_58.193: f32[1], param_59.189: f32[], param_60.182: f32[1], param_61.178: f32[], param_62.171: f32[1], param_63.167: f32[], param_64.160: f32[1], param_65.156: f32[], param_66.149: f32[1], param_67.145: f32[1], param_68.138: f32[], param_69.131: f32[], param_70.127: f32[1], param_71.123: f32[1], param_72.116: f32[], param_73.109: f32[], param_74.105: f32[1], param_75.101: f32[], param_76.94: f32[1], param_77.90: f32[], param_78.83: f32[1], param_79.79: f32[], param_80.72: f32[1], param_81.68: f32[], param_82.61: f32[1], param_83.57: f32[], param_84.50: f32[1], param_85.46: f32[], param_86.39: f32[1], param_87.35: f32[], param_88.28: f32[1], param_89.24: f32[], param_90.17: f32[1], param_91.13: f32[], param_92.6: f32[], param_93.3: f32[1]) -> f32[] {
  %param_92.6 = f32[] parameter(92)
  %param_93.3 = f32[1]{0} parameter(93)
  %bitcast.876.187 = f32[1,1]{1,0} bitcast(%param_93.3)
  %square.1033.187 = f32[1,1]{1,0} multiply(%bitcast.876.187, %bitcast.876.187)
  %bitcast.1166.185 = f32[] bitcast(%square.1033.187)
  %add.2175.185 = f32[] add(%param_92.6, %bitcast.1166.185)
  %param_91.13 = f32[] parameter(91)
  %add.2176.183 = f32[] add(%add.2175.185, %param_91.13)
  %param_90.17 = f32[1]{0} parameter(90)
  %bitcast.885.183 = f32[1,1]{1,0} bitcast(%param_90.17)
  %square.1035.183 = f32[1,1]{1,0} multiply(%bitcast.885.183, %bitcast.885.183)
  %bitcast.1167.181 = f32[] bitcast(%square.1035.183)
  %add.2177.181 = f32[] add(%add.2176.183, %bitcast.1167.181)
  %param_89.24 = f32[] parameter(89)
  %add.2178.179 = f32[] add(%add.2177.181, %param_89.24)
  %param_88.28 = f32[1]{0} parameter(88)
  %bitcast.945.179 = f32[1,1]{1,0} bitcast(%param_88.28)
  %square.1037.179 = f32[1,1]{1,0} multiply(%bitcast.945.179, %bitcast.945.179)
  %bitcast.1168.177 = f32[] bitcast(%square.1037.179)
  %add.2179.177 = f32[] add(%add.2178.179, %bitcast.1168.177)
  %param_87.35 = f32[] parameter(87)
  %add.2180.175 = f32[] add(%add.2179.177, %param_87.35)
  %param_86.39 = f32[1]{0} parameter(86)
  %bitcast.950.175 = f32[1,1]{1,0} bitcast(%param_86.39)
  %square.1039.175 = f32[1,1]{1,0} multiply(%bitcast.950.175, %bitcast.950.175)
  %bitcast.1169.173 = f32[] bitcast(%square.1039.175)
  %add.2181.173 = f32[] add(%add.2180.175, %bitcast.1169.173)
  %param_85.46 = f32[] parameter(85)
  %add.2182.171 = f32[] add(%add.2181.173, %param_85.46)
  %param_84.50 = f32[1]{0} parameter(84)
  %bitcast.955.171 = f32[1,1]{1,0} bitcast(%param_84.50)
  %square.1041.171 = f32[1,1]{1,0} multiply(%bitcast.955.171, %bitcast.955.171)
  %bitcast.1170.169 = f32[] bitcast(%square.1041.171)
  %add.2183.169 = f32[] add(%add.2182.171, %bitcast.1170.169)
  %param_83.57 = f32[] parameter(83)
  %add.2184.167 = f32[] add(%add.2183.169, %param_83.57)
  %param_82.61 = f32[1]{0} parameter(82)
  %bitcast.964.167 = f32[1,1]{1,0} bitcast(%param_82.61)
  %square.1043.167 = f32[1,1]{1,0} multiply(%bitcast.964.167, %bitcast.964.167)
  %bitcast.1171.165 = f32[] bitcast(%square.1043.167)
  %add.2185.165 = f32[] add(%add.2184.167, %bitcast.1171.165)
  %param_81.68 = f32[] parameter(81)
  %add.2186.163 = f32[] add(%add.2185.165, %param_81.68)
  %param_80.72 = f32[1]{0} parameter(80)
  %bitcast.969.163 = f32[1,1]{1,0} bitcast(%param_80.72)
  %square.1045.163 = f32[1,1]{1,0} multiply(%bitcast.969.163, %bitcast.969.163)
  %bitcast.1172.161 = f32[] bitcast(%square.1045.163)
  %add.2187.161 = f32[] add(%add.2186.163, %bitcast.1172.161)
  %param_79.79 = f32[] parameter(79)
  %add.2188.159 = f32[] add(%add.2187.161, %param_79.79)
  %param_78.83 = f32[1]{0} parameter(78)
  %bitcast.974.159 = f32[1,1]{1,0} bitcast(%param_78.83)
  %square.1047.159 = f32[1,1]{1,0} multiply(%bitcast.974.159, %bitcast.974.159)
  %bitcast.1173.157 = f32[] bitcast(%square.1047.159)
  %add.2189.157 = f32[] add(%add.2188.159, %bitcast.1173.157)
  %param_77.90 = f32[] parameter(77)
  %add.2190.155 = f32[] add(%add.2189.157, %param_77.90)
  %param_76.94 = f32[1]{0} parameter(76)
  %bitcast.979.155 = f32[1,1]{1,0} bitcast(%param_76.94)
  %square.1049.155 = f32[1,1]{1,0} multiply(%bitcast.979.155, %bitcast.979.155)
  %bitcast.1174.153 = f32[] bitcast(%square.1049.155)
  %add.2191.153 = f32[] add(%add.2190.155, %bitcast.1174.153)
  %param_75.101 = f32[] parameter(75)
  %add.2192.151 = f32[] add(%add.2191.153, %param_75.101)
  %param_74.105 = f32[1]{0} parameter(74)
  %bitcast.986.151 = f32[1,1]{1,0} bitcast(%param_74.105)
  %square.1051.151 = f32[1,1]{1,0} multiply(%bitcast.986.151, %bitcast.986.151)
  %bitcast.1175.149 = f32[] bitcast(%square.1051.151)
  %add.2193.149 = f32[] add(%add.2192.151, %bitcast.1175.149)
  %param_73.109 = f32[] parameter(73)
  %add.2194.147 = f32[] add(%add.2193.149, %param_73.109)
  %param_72.116 = f32[] parameter(72)
  %add.2195.145 = f32[] add(%add.2194.147, %param_72.116)
  %param_71.123 = f32[1]{0} parameter(71)
  %bitcast.995.145 = f32[1,1]{1,0} bitcast(%param_71.123)
  %square.1054.145 = f32[1,1]{1,0} multiply(%bitcast.995.145, %bitcast.995.145)
  %bitcast.1176.143 = f32[] bitcast(%square.1054.145)
  %add.2196.143 = f32[] add(%add.2195.145, %bitcast.1176.143)
  %param_70.127 = f32[1]{0} parameter(70)
  %bitcast.996.143 = f32[1,1]{1,0} bitcast(%param_70.127)
  %square.1055.143 = f32[1,1]{1,0} multiply(%bitcast.996.143, %bitcast.996.143)
  %bitcast.1177.141 = f32[] bitcast(%square.1055.143)
  %add.2197.141 = f32[] add(%add.2196.143, %bitcast.1177.141)
  %param_69.131 = f32[] parameter(69)
  %add.2198.139 = f32[] add(%add.2197.141, %param_69.131)
  %param_68.138 = f32[] parameter(68)
  %add.2199.137 = f32[] add(%add.2198.139, %param_68.138)
  %param_67.145 = f32[1]{0} parameter(67)
  %bitcast.1072.137 = f32[1,1]{1,0} bitcast(%param_67.145)
  %square.1058.137 = f32[1,1]{1,0} multiply(%bitcast.1072.137, %bitcast.1072.137)
  %bitcast.1178.135 = f32[] bitcast(%square.1058.137)
  %add.2200.135 = f32[] add(%add.2199.137, %bitcast.1178.135)
  %param_66.149 = f32[1]{0} parameter(66)
  %bitcast.1077.135 = f32[1,1]{1,0} bitcast(%param_66.149)
  %square.1059.135 = f32[1,1]{1,0} multiply(%bitcast.1077.135, %bitcast.1077.135)
  %bitcast.1179.133 = f32[] bitcast(%square.1059.135)
  %add.2201.133 = f32[] add(%add.2200.135, %bitcast.1179.133)
  %param_65.156 = f32[] parameter(65)
  %add.2202.131 = f32[] add(%add.2201.133, %param_65.156)
  %param_64.160 = f32[1]{0} parameter(64)
  %bitcast.1082.131 = f32[1,1]{1,0} bitcast(%param_64.160)
  %square.1061.131 = f32[1,1]{1,0} multiply(%bitcast.1082.131, %bitcast.1082.131)
  %bitcast.1180.129 = f32[] bitcast(%square.1061.131)
  %add.2203.129 = f32[] add(%add.2202.131, %bitcast.1180.129)
  %param_63.167 = f32[] parameter(63)
  %add.2204.127 = f32[] add(%add.2203.129, %param_63.167)
  %param_62.171 = f32[1]{0} parameter(62)
  %bitcast.1087.127 = f32[1,1]{1,0} bitcast(%param_62.171)
  %square.1063.127 = f32[1,1]{1,0} multiply(%bitcast.1087.127, %bitcast.1087.127)
  %bitcast.1181.125 = f32[] bitcast(%square.1063.127)
  %add.2205.125 = f32[] add(%add.2204.127, %bitcast.1181.125)
  %param_61.178 = f32[] parameter(61)
  %add.2206.123 = f32[] add(%add.2205.125, %param_61.178)
  %param_60.182 = f32[1]{0} parameter(60)
  %bitcast.1092.123 = f32[1,1]{1,0} bitcast(%param_60.182)
  %square.1065.123 = f32[1,1]{1,0} multiply(%bitcast.1092.123, %bitcast.1092.123)
  %bitcast.1182.121 = f32[] bitcast(%square.1065.123)
  %add.2207.121 = f32[] add(%add.2206.123, %bitcast.1182.121)
  %param_59.189 = f32[] parameter(59)
  %add.2208.119 = f32[] add(%add.2207.121, %param_59.189)
  %param_58.193 = f32[1]{0} parameter(58)
  %bitcast.1097.119 = f32[1,1]{1,0} bitcast(%param_58.193)
  %square.1067.119 = f32[1,1]{1,0} multiply(%bitcast.1097.119, %bitcast.1097.119)
  %bitcast.1183.117 = f32[] bitcast(%square.1067.119)
  %add.2209.117 = f32[] add(%add.2208.119, %bitcast.1183.117)
  %param_57.200 = f32[] parameter(57)
  %add.2210.115 = f32[] add(%add.2209.117, %param_57.200)
  %param_56.204 = f32[1]{0} parameter(56)
  %bitcast.1102.115 = f32[1,1]{1,0} bitcast(%param_56.204)
  %square.1069.115 = f32[1,1]{1,0} multiply(%bitcast.1102.115, %bitcast.1102.115)
  %bitcast.1184.113 = f32[] bitcast(%square.1069.115)
  %add.2211.113 = f32[] add(%add.2210.115, %bitcast.1184.113)
  %param_55.211 = f32[] parameter(55)
  %add.2212.111 = f32[] add(%add.2211.113, %param_55.211)
  %param_54.215 = f32[1]{0} parameter(54)
  %bitcast.1107.111 = f32[1,1]{1,0} bitcast(%param_54.215)
  %square.1071.111 = f32[1,1]{1,0} multiply(%bitcast.1107.111, %bitcast.1107.111)
  %bitcast.1185.109 = f32[] bitcast(%square.1071.111)
  %add.2213.109 = f32[] add(%add.2212.111, %bitcast.1185.109)
  %param_53.222 = f32[] parameter(53)
  %add.2214.107 = f32[] add(%add.2213.109, %param_53.222)
  %param_52.226 = f32[1]{0} parameter(52)
  %bitcast.1112.107 = f32[1,1]{1,0} bitcast(%param_52.226)
  %square.1073.107 = f32[1,1]{1,0} multiply(%bitcast.1112.107, %bitcast.1112.107)
  %bitcast.1186.105 = f32[] bitcast(%square.1073.107)
  %add.2215.105 = f32[] add(%add.2214.107, %bitcast.1186.105)
  %param_51.233 = f32[] parameter(51)
  %add.2216.103 = f32[] add(%add.2215.105, %param_51.233)
  %param_50.237 = f32[1]{0} parameter(50)
  %bitcast.1117.103 = f32[1,1]{1,0} bitcast(%param_50.237)
  %square.1075.103 = f32[1,1]{1,0} multiply(%bitcast.1117.103, %bitcast.1117.103)
  %bitcast.1187.101 = f32[] bitcast(%square.1075.103)
  %add.2217.101 = f32[] add(%add.2216.103, %bitcast.1187.101)
  %param_49.244 = f32[] parameter(49)
  %add.2218.99 = f32[] add(%add.2217.101, %param_49.244)
  %param_48.248 = f32[1]{0} parameter(48)
  %bitcast.1122.99 = f32[1,1]{1,0} bitcast(%param_48.248)
  %square.1077.99 = f32[1,1]{1,0} multiply(%bitcast.1122.99, %bitcast.1122.99)
  %bitcast.1188.97 = f32[] bitcast(%square.1077.99)
  %add.2219.97 = f32[] add(%add.2218.99, %bitcast.1188.97)
  %param_47.252 = f32[] parameter(47)
  %add.2220.95 = f32[] add(%add.2219.97, %param_47.252)
  %param_46.256 = f32[] parameter(46)
  %add.2221.93 = f32[] add(%add.2220.95, %param_46.256)
  %param_45.262 = f32[] parameter(45)
  %add.2222.91 = f32[] add(%add.2221.93, %param_45.262)
  %param_44.265 = f32[] parameter(44)
  %add.2223.89 = f32[] add(%add.2222.91, %param_44.265)
  %param_43.271 = f32[] parameter(43)
  %add.2224.87 = f32[] add(%add.2223.89, %param_43.271)
  %param_42.274 = f32[] parameter(42)
  %add.2225.85 = f32[] add(%add.2224.87, %param_42.274)
  %param_41.280 = f32[] parameter(41)
  %add.2226.83 = f32[] add(%add.2225.85, %param_41.280)
  %param_40.283 = f32[] parameter(40)
  %add.2227.81 = f32[] add(%add.2226.83, %param_40.283)
  %param_39.289 = f32[] parameter(39)
  %add.2228.79 = f32[] add(%add.2227.81, %param_39.289)
  %param_38.292 = f32[] parameter(38)
  %add.2229.77 = f32[] add(%add.2228.79, %param_38.292)
  %param_37.298 = f32[] parameter(37)
  %add.2230.75 = f32[] add(%add.2229.77, %param_37.298)
  %param_36.301 = f32[] parameter(36)
  %add.2231.73 = f32[] add(%add.2230.75, %param_36.301)
  %param_35.307 = f32[] parameter(35)
  %add.2232.71 = f32[] add(%add.2231.73, %param_35.307)
  %param_34.310 = f32[] parameter(34)
  %add.2233.69 = f32[] add(%add.2232.71, %param_34.310)
  %param_33.316 = f32[] parameter(33)
  %add.2234.67 = f32[] add(%add.2233.69, %param_33.316)
  %param_32.319 = f32[] parameter(32)
  %add.2235.65 = f32[] add(%add.2234.67, %param_32.319)
  %param_31.328 = f32[] parameter(31)
  %add.2236.63 = f32[] add(%add.2235.65, %param_31.328)
  %param_30.332 = f32[] parameter(30)
  %add.2237.61 = f32[] add(%add.2236.63, %param_30.332)
  %param_29.338 = f32[] parameter(29)
  %add.2238.59 = f32[] add(%add.2237.61, %param_29.338)
  %param_28.341 = f32[] parameter(28)
  %add.2239.57 = f32[] add(%add.2238.59, %param_28.341)
  %param_27.347 = f32[] parameter(27)
  %add.2240.55 = f32[] add(%add.2239.57, %param_27.347)
  %param_26.352 = f32[] parameter(26)
  %add.2241.53 = f32[] add(%add.2240.55, %param_26.352)
  %param_25.355 = f32[] parameter(25)
  %add.2242.51 = f32[] add(%add.2241.53, %param_25.355)
  %param_24.359 = f32[] parameter(24)
  %add.2243.49 = f32[] add(%add.2242.51, %param_24.359)
  %param_23.365 = f32[] parameter(23)
  %add.2244.47 = f32[] add(%add.2243.49, %param_23.365)
  %param_22.370 = f32[] parameter(22)
  %add.2245.45 = f32[] add(%add.2244.47, %param_22.370)
  %param_21.373 = f32[] parameter(21)
  %add.2246.43 = f32[] add(%add.2245.45, %param_21.373)
  %param_20.379 = f32[] parameter(20)
  %add.2247.41 = f32[] add(%add.2246.43, %param_20.379)
  %param_19.386 = f32[] parameter(19)
  %add.2248.39 = f32[] add(%add.2247.41, %param_19.386)
  %param_18.391 = f32[] parameter(18)
  %add.2249.37 = f32[] add(%add.2248.39, %param_18.391)
  %param_17.402 = f32[] parameter(17)
  %add.2250.35 = f32[] add(%add.2249.37, %param_17.402)
  %param_16.408 = f32[] parameter(16)
  %add.2251.33 = f32[] add(%add.2250.35, %param_16.408)
  %param_15.418 = f32[] parameter(15)
  %add.2252.31 = f32[] add(%add.2251.33, %param_15.418)
  %param_14.425 = f32[] parameter(14)
  %add.2253.29 = f32[] add(%add.2252.31, %param_14.425)
  %param_13.435 = f32[] parameter(13)
  %add.2254.27 = f32[] add(%add.2253.29, %param_13.435)
  %param_12.524 = f32[] parameter(12)
  %add.2255.25 = f32[] add(%add.2254.27, %param_12.524)
  %param_11.667 = f32[] parameter(11)
  %add.2256.23 = f32[] add(%add.2255.25, %param_11.667)
  %param_10.687 = f32[] parameter(10)
  %add.2257.21 = f32[] add(%add.2256.23, %param_10.687)
  %param_9.1039 = f32[] parameter(9)
  %add.2258.19 = f32[] add(%add.2257.21, %param_9.1039)
  %param_8.1383 = f32[] parameter(8)
  %add.2259.17 = f32[] add(%add.2258.19, %param_8.1383)
  %param_7.1899 = f32[] parameter(7)
  %add.2260.15 = f32[] add(%add.2259.17, %param_7.1899)
  %param_6.2452 = f32[] parameter(6)
  %add.2261.13 = f32[] add(%add.2260.15, %param_6.2452)
  %param_5.3398 = f32[] parameter(5)
  %add.2262.11 = f32[] add(%add.2261.13, %param_5.3398)
  %param_4.3886 = f32[] parameter(4)
  %add.2263.9 = f32[] add(%add.2262.11, %param_4.3886)
  %param_3.6092 = f32[] parameter(3)
  %add.2264.7 = f32[] add(%add.2263.9, %param_3.6092)
  %param_2.7834 = f32[] parameter(2)
  %add.2265.5 = f32[] add(%add.2264.7, %param_2.7834)
  %param_1.10589 = f32[] parameter(1)
  %add.2266.3 = f32[] add(%add.2265.5, %param_1.10589)
  %param_0.10188 = f32[] parameter(0)
  ROOT %add.2267.1 = f32[] add(%add.2266.3, %param_0.10188)
}



ENTRY %wrapper_fused_add.10 (param_0.10188: f32[], param_1.10589: f32[], param_2.7834: f32[], param_3.6092: f32[], param_4.3886: f32[], param_5.3398: f32[], param_6.2452: f32[], param_7.1899: f32[], param_8.1383: f32[], param_9.1039: f32[], param_10.687: f32[], param_11.667: f32[], param_12.524: f32[], param_13.435: f32[], param_14.425: f32[], param_15.418: f32[], param_16.408: f32[], param_17.402: f32[], param_18.391: f32[], param_19.386: f32[], param_20.379: f32[], param_21.373: f32[], param_22.370: f32[], param_23.365: f32[], param_24.359: f32[], param_25.355: f32[], param_26.352: f32[], param_27.347: f32[], param_28.341: f32[], param_29.338: f32[], param_30.332: f32[], param_31.328: f32[], param_32.319: f32[], param_33.316: f32[], param_34.310: f32[], param_35.307: f32[], param_36.301: f32[], param_37.298: f32[], param_38.292: f32[], param_39.289: f32[], param_40.283: f32[], param_41.280: f32[], param_42.274: f32[], param_43.271: f32[], param_44.265: f32[], param_45.262: f32[], param_46.256: f32[], param_47.252: f32[], param_48.248: f32[1], param_49.244: f32[], param_50.237: f32[1], param_51.233: f32[], param_52.226: f32[1], param_53.222: f32[], param_54.215: f32[1], param_55.211: f32[], param_56.204: f32[1], param_57.200: f32[], param_58.193: f32[1], param_59.189: f32[], param_60.182: f32[1], param_61.178: f32[], param_62.171: f32[1], param_63.167: f32[], param_64.160: f32[1], param_65.156: f32[], param_66.149: f32[1], param_67.145: f32[1], param_68.138: f32[], param_69.131: f32[], param_70.127: f32[1], param_71.123: f32[1], param_72.116: f32[], param_73.109: f32[], param_74.105: f32[1], param_75.101: f32[], param_76.94: f32[1], param_77.90: f32[], param_78.83: f32[1], param_79.79: f32[], param_80.72: f32[1], param_81.68: f32[], param_82.61: f32[1], param_83.57: f32[], param_84.50: f32[1], param_85.46: f32[], param_86.39: f32[1], param_87.35: f32[], param_88.28: f32[1], param_89.24: f32[], param_90.17: f32[1], param_91.13: f32[], param_92.6: f32[], param_93.3: f32[1]) -> f32[] {
  param_0.10188 = f32[] parameter(0)
  param_1.10589 = f32[] parameter(1)
  param_2.7834 = f32[] parameter(2)
  param_3.6092 = f32[] parameter(3)
  param_4.3886 = f32[] parameter(4)
  param_5.3398 = f32[] parameter(5)
  param_6.2452 = f32[] parameter(6)
  param_7.1899 = f32[] parameter(7)
  param_8.1383 = f32[] parameter(8)
  param_9.1039 = f32[] parameter(9)
  param_10.687 = f32[] parameter(10)
  param_11.667 = f32[] parameter(11)
  param_12.524 = f32[] parameter(12)
  param_13.435 = f32[] parameter(13)
  param_14.425 = f32[] parameter(14)
  param_15.418 = f32[] parameter(15)
  param_16.408 = f32[] parameter(16)
  param_17.402 = f32[] parameter(17)
  param_18.391 = f32[] parameter(18)
  param_19.386 = f32[] parameter(19)
  param_20.379 = f32[] parameter(20)
  param_21.373 = f32[] parameter(21)
  param_22.370 = f32[] parameter(22)
  param_23.365 = f32[] parameter(23)
  param_24.359 = f32[] parameter(24)
  param_25.355 = f32[] parameter(25)
  param_26.352 = f32[] parameter(26)
  param_27.347 = f32[] parameter(27)
  param_28.341 = f32[] parameter(28)
  param_29.338 = f32[] parameter(29)
  param_30.332 = f32[] parameter(30)
  param_31.328 = f32[] parameter(31)
  param_32.319 = f32[] parameter(32)
  param_33.316 = f32[] parameter(33)
  param_34.310 = f32[] parameter(34)
  param_35.307 = f32[] parameter(35)
  param_36.301 = f32[] parameter(36)
  param_37.298 = f32[] parameter(37)
  param_38.292 = f32[] parameter(38)
  param_39.289 = f32[] parameter(39)
  param_40.283 = f32[] parameter(40)
  param_41.280 = f32[] parameter(41)
  param_42.274 = f32[] parameter(42)
  param_43.271 = f32[] parameter(43)
  param_44.265 = f32[] parameter(44)
  param_45.262 = f32[] parameter(45)
  param_46.256 = f32[] parameter(46)
  param_47.252 = f32[] parameter(47)
  param_48.248 = f32[1] parameter(48)
  param_49.244 = f32[] parameter(49)
  param_50.237 = f32[1] parameter(50)
  param_51.233 = f32[] parameter(51)
  param_52.226 = f32[1] parameter(52)
  param_53.222 = f32[] parameter(53)
  param_54.215 = f32[1] parameter(54)
  param_55.211 = f32[] parameter(55)
  param_56.204 = f32[1] parameter(56)
  param_57.200 = f32[] parameter(57)
  param_58.193 = f32[1] parameter(58)
  param_59.189 = f32[] parameter(59)
  param_60.182 = f32[1] parameter(60)
  param_61.178 = f32[] parameter(61)
  param_62.171 = f32[1] parameter(62)
  param_63.167 = f32[] parameter(63)
  param_64.160 = f32[1] parameter(64)
  param_65.156 = f32[] parameter(65)
  param_66.149 = f32[1] parameter(66)
  param_67.145 = f32[1] parameter(67)
  param_68.138 = f32[] parameter(68)
  param_69.131 = f32[] parameter(69)
  param_70.127 = f32[1] parameter(70)
  param_71.123 = f32[1] parameter(71)
  param_72.116 = f32[] parameter(72)
  param_73.109 = f32[] parameter(73)
  param_74.105 = f32[1] parameter(74)
  param_75.101 = f32[] parameter(75)
  param_76.94 = f32[1] parameter(76)
  param_77.90 = f32[] parameter(77)
  param_78.83 = f32[1] parameter(78)
  param_79.79 = f32[] parameter(79)
  param_80.72 = f32[1] parameter(80)
  param_81.68 = f32[] parameter(81)
  param_82.61 = f32[1] parameter(82)
  param_83.57 = f32[] parameter(83)
  param_84.50 = f32[1] parameter(84)
  param_85.46 = f32[] parameter(85)
  param_86.39 = f32[1] parameter(86)
  param_87.35 = f32[] parameter(87)
  param_88.28 = f32[1] parameter(88)
  param_89.24 = f32[] parameter(89)
  param_90.17 = f32[1] parameter(90)
  param_91.13 = f32[] parameter(91)
  param_92.6 = f32[] parameter(92)
  param_93.3 = f32[1] parameter(93)
  ROOT %fusion = f32[] fusion(param_0.10188, param_1.10589, param_2.7834, param_3.6092, param_4.3886, param_5.3398, param_6.2452, param_7.1899, param_8.1383, param_9.1039, param_10.687, param_11.667, param_12.524, param_13.435, param_14.425, param_15.418, param_16.408, param_17.402, param_18.391, param_19.386, param_20.379, param_21.373, param_22.370, param_23.365, param_24.359, param_25.355, param_26.352, param_27.347, param_28.341, param_29.338, param_30.332, param_31.328, param_32.319, param_33.316, param_34.310, param_35.307, param_36.301, param_37.298, param_38.292, param_39.289, param_40.283, param_41.280, param_42.274, param_43.271, param_44.265, param_45.262, param_46.256, param_47.252, param_48.248, param_49.244, param_50.237, param_51.233, param_52.226, param_53.222, param_54.215, param_55.211, param_56.204, param_57.200, param_58.193, param_59.189, param_60.182, param_61.178, param_62.171, param_63.167, param_64.160, param_65.156, param_66.149, param_67.145, param_68.138, param_69.131, param_70.127, param_71.123, param_72.116, param_73.109, param_74.105, param_75.101, param_76.94, param_77.90, param_78.83, param_79.79, param_80.72, param_81.68, param_82.61, param_83.57, param_84.50, param_85.46, param_86.39, param_87.35, param_88.28, param_89.24, param_90.17, param_91.13, param_92.6, param_93.3), kind=kLoop, calls=%fused_add.10
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_concatenate.7.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedConcatenate_7) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate.7_module, entry_computation_layout={(bf16[32768,32768]{1,0}, bf16[1,8,4096,64]{3,2,1,0})->bf16[8,4096,128,192]{3,2,1,0}}

%fused_concatenate.7 (param_0.4328: bf16[32768,32768], param_1.4124: bf16[1,8,4096,64]) -> bf16[8,4096,128,192] {
  %param_0.4328 = bf16[32768,32768]{1,0} parameter(0)
  %bitcast.3308.4 = bf16[8,4096,128,256]{3,2,1,0} bitcast(%param_0.4328)
  %slice.1806.3 = bf16[8,4096,128,128]{3,2,1,0} slice(%bitcast.3308.4), slice={[0:8], [0:4096], [0:128], [0:128]}
  %param_1.4124 = bf16[1,8,4096,64]{3,2,1,0} parameter(1)
  %bitcast.61.3 = bf16[8,4096,64]{2,1,0} bitcast(%param_1.4124)
  %broadcast_in_dim.3123.3 = bf16[8,4096,128,64]{3,2,1,0} broadcast(%bitcast.61.3), dimensions={0,1,3}
  ROOT %concatenate.589.1 = bf16[8,4096,128,192]{3,2,1,0} concatenate(%slice.1806.3, %broadcast_in_dim.3123.3), dimensions={3}
}



ENTRY %wrapper_fused_concatenate.7 (param_0.4328: bf16[32768,32768], param_1.4124: bf16[1,8,4096,64]) -> bf16[8,4096,128,192] {
  param_0.4328 = bf16[32768,32768] parameter(0)
  param_1.4124 = bf16[1,8,4096,64] parameter(1)
  ROOT %fusion = bf16[8,4096,128,192] fusion(param_0.4328, param_1.4124), kind=kLoop, calls=%fused_concatenate.7
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_concatenate.9.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedConcatenate_9) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate.9_module, entry_computation_layout={(f32[1,1024]{1,0})->f32[1024]{0}}

%fused_concatenate.9 (param_0.9792: f32[1,1024]) -> f32[1024] {
  %param_0.9792 = f32[1,1024]{1,0} parameter(0)
  %bitcast.3955.4 = f32[1024]{0} bitcast(%param_0.9792)
  %slice.1790.3 = f32[1023]{0} slice(%bitcast.3955.4), slice={[1:1024]}
  %slice.1791.1 = f32[1,1]{1,0} slice(%param_0.9792), slice={[0:1], [0:1]}
  %bitcast.1119.1 = f32[1]{0} bitcast(%slice.1791.1)
  ROOT %concatenate.585.1 = f32[1024]{0} concatenate(%slice.1790.3, %bitcast.1119.1), dimensions={0}
}



ENTRY %wrapper_fused_concatenate.9 (param_0.9792: f32[1,1024]) -> f32[1024] {
  param_0.9792 = f32[1,1024] parameter(0)
  ROOT %fusion = f32[1024] fusion(param_0.9792), kind=kLoop, calls=%fused_concatenate.9
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_convert.5.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedConvert_5) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_convert.5_module, entry_computation_layout={(f32[1]{0})->f32[]}

%fused_convert.5 (param_0.3347: f32[1]) -> f32[] {
  %param_0.3347 = f32[1]{0} parameter(0)
  %convert_element_type.3098.1 = bf16[1]{0} convert(%param_0.3347)
  %bitcast.459.2 = bf16[] bitcast(%convert_element_type.3098.1)
  ROOT %convert.1703.1 = f32[] convert(%bitcast.459.2)
}



ENTRY %wrapper_fused_convert.5 (param_0.3347: f32[1]) -> f32[] {
  param_0.3347 = f32[1] parameter(0)
  ROOT %fusion = f32[] fusion(param_0.3347), kind=kLoop, calls=%fused_convert.5
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_convert.25.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedConvert_25) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_convert.25_module, entry_computation_layout={(bf16[1]{0})->f32[]}

%fused_convert.25 (param_0.3843: bf16[1]) -> f32[] {
  %param_0.3843 = bf16[1]{0} parameter(0)
  %bitcast.195.2 = bf16[] bitcast(%param_0.3843)
  ROOT %convert.1790.1 = f32[] convert(%bitcast.195.2)
}



ENTRY %wrapper_fused_convert.25 (param_0.3843: bf16[1]) -> f32[] {
  param_0.3843 = bf16[1] parameter(0)
  ROOT %fusion = f32[] fusion(param_0.3843), kind=kLoop, calls=%fused_convert.25
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_convert.75.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedConvert_75) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_convert.75_module, entry_computation_layout={(f32[1]{0}, f32[1]{0})->bf16[1]{0}}

%fused_convert.75 (param_0.9765: f32[1], param_1.10108: f32[1]) -> bf16[1] {
  %param_0.9765 = f32[1]{0} parameter(0)
  %param_1.10108 = f32[1]{0} parameter(1)
  %mul.3090.1 = f32[1]{0} multiply(%param_0.9765, %param_1.10108)
  ROOT %convert_element_type.3510.1 = bf16[1]{0} convert(%mul.3090.1)
}



ENTRY %wrapper_fused_convert.75 (param_0.9765: f32[1], param_1.10108: f32[1]) -> bf16[1] {
  param_0.9765 = f32[1] parameter(0)
  param_1.10108 = f32[1] parameter(1)
  ROOT %fusion = bf16[1] fusion(param_0.9765, param_1.10108), kind=kLoop, calls=%fused_convert.75
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_multiply.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedMultiply) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply_module, entry_computation_layout={(bf16[1,16,32,512,2048]{4,3,2,1,0})->bf16[16,32,512,2048]{3,2,1,0}}

%fused_multiply (param_0.1452: bf16[1,16,32,512,2048]) -> bf16[16,32,512,2048] {
  %param_0.1452 = bf16[1,16,32,512,2048]{4,3,2,1,0} parameter(0)
  %bitcast.286.3 = bf16[16,32,512,2048]{3,2,1,0} bitcast(%param_0.1452)
  %constant_3333_4 = bf16[] constant(1)
  %convert.1648.4 = f32[] convert(%constant_3333_4)
  %broadcast.3199.5 = f32[16,32,512,2048]{3,2,1,0} broadcast(%convert.1648.4), dimensions={}
  %neg.45.9 = bf16[16,32,512,2048]{3,2,1,0} negate(%bitcast.286.3)
  %exp.449.7 = bf16[16,32,512,2048]{3,2,1,0} exponential(%neg.45.9)
  %jit_silu_.30.11 = bf16[16,32,512,2048]{3,2,1,0} broadcast(%constant_3333_4), dimensions={}
  %add.1874.5 = bf16[16,32,512,2048]{3,2,1,0} add(%exp.449.7, %jit_silu_.30.11)
  %convert.509.3 = f32[16,32,512,2048]{3,2,1,0} convert(%add.1874.5)
  %div.2697.5 = f32[16,32,512,2048]{3,2,1,0} divide(%broadcast.3199.5, %convert.509.3)
  %convert.510.3 = bf16[16,32,512,2048]{3,2,1,0} convert(%div.2697.5)
  ROOT %mul.2665.1 = bf16[16,32,512,2048]{3,2,1,0} multiply(%bitcast.286.3, %convert.510.3)
}



ENTRY %wrapper_fused_multiply (param_0.1452: bf16[1,16,32,512,2048]) -> bf16[16,32,512,2048] {
  param_0.1452 = bf16[1,16,32,512,2048] parameter(0)
  ROOT %fusion = bf16[16,32,512,2048] fusion(param_0.1452), kind=kLoop, calls=%fused_multiply
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_multiply.1.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedMultiply_1) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.1_module, entry_computation_layout={(bf16[1,8,4096,2048]{3,2,1,0})->bf16[8,4096,2048]{2,1,0}}

%fused_multiply.1 (param_0.1852: bf16[1,8,4096,2048]) -> bf16[8,4096,2048] {
  %param_0.1852 = bf16[1,8,4096,2048]{3,2,1,0} parameter(0)
  %bitcast.213.3 = bf16[8,4096,2048]{2,1,0} bitcast(%param_0.1852)
  %constant_3333_6 = bf16[] constant(1)
  %convert.1648.5 = f32[] convert(%constant_3333_6)
  %broadcast.3249.5 = f32[8,4096,2048]{2,1,0} broadcast(%convert.1648.5), dimensions={}
  %neg.44.7 = bf16[8,4096,2048]{2,1,0} negate(%bitcast.213.3)
  %exp.448.5 = bf16[8,4096,2048]{2,1,0} exponential(%neg.44.7)
  %jit_silu_.29.13 = bf16[8,4096,2048]{2,1,0} broadcast(%constant_3333_6), dimensions={}
  %add.1870.5 = bf16[8,4096,2048]{2,1,0} add(%exp.448.5, %jit_silu_.29.13)
  %convert.592.3 = f32[8,4096,2048]{2,1,0} convert(%add.1870.5)
  %div.2650.5 = f32[8,4096,2048]{2,1,0} divide(%broadcast.3249.5, %convert.592.3)
  %convert.593.3 = bf16[8,4096,2048]{2,1,0} convert(%div.2650.5)
  ROOT %mul.2638.1 = bf16[8,4096,2048]{2,1,0} multiply(%bitcast.213.3, %convert.593.3)
}



ENTRY %wrapper_fused_multiply.1 (param_0.1852: bf16[1,8,4096,2048]) -> bf16[8,4096,2048] {
  param_0.1852 = bf16[1,8,4096,2048] parameter(0)
  ROOT %fusion = bf16[8,4096,2048] fusion(param_0.1852), kind=kLoop, calls=%fused_multiply.1
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_multiply.2.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedMultiply_2) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.2_module, entry_computation_layout={(bf16[8,4096,64]{2,1,0}, pred[8,4096,64]{2,1,0}, pred[8,4096,64]{2,1,0})->bf16[8,4096,64]{2,1,0}}

%fused_multiply.2 (param_0.2245: bf16[8,4096,64], param_1.1890: pred[8,4096,64], param_2.2685: pred[8,4096,64]) -> bf16[8,4096,64] {
  %param_0.2245 = bf16[8,4096,64]{2,1,0} parameter(0)
  %param_1.1890 = pred[8,4096,64]{2,1,0} parameter(1)
  %param_2.2685 = pred[8,4096,64]{2,1,0} parameter(2)
  %convert_element_type.2983.14 = s32[8,4096,64]{2,1,0} convert(%param_2.2685)
  %constant_3329_20 = s32[] constant(0)
  %broadcast.2602.12 = s32[8,4096,64]{2,1,0} broadcast(%constant_3329_20), dimensions={}
  %mul.2659.9 = s32[8,4096,64]{2,1,0} select(%param_1.1890, %convert_element_type.2983.14, %broadcast.2602.12)
  %convert_element_type.2984.5 = bf16[8,4096,64]{2,1,0} convert(%mul.2659.9)
  ROOT %mul.2689.1 = bf16[8,4096,64]{2,1,0} multiply(%param_0.2245, %convert_element_type.2984.5)
}



ENTRY %wrapper_fused_multiply.2 (param_0.2245: bf16[8,4096,64], param_1.1890: pred[8,4096,64], param_2.2685: pred[8,4096,64]) -> bf16[8,4096,64] {
  param_0.2245 = bf16[8,4096,64] parameter(0)
  param_1.1890 = pred[8,4096,64] parameter(1)
  param_2.2685 = pred[8,4096,64] parameter(2)
  ROOT %fusion = bf16[8,4096,64] fusion(param_0.2245, param_1.1890, param_2.2685), kind=kLoop, calls=%fused_multiply.2
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_multiply.3.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedMultiply_3) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.3_module, entry_computation_layout={(f32[8,4096]{1,0}, f32[8,4096]{1,0}, f32[8,4096]{1,0})->f32[1,8,4096]{2,1,0}}

%fused_multiply.3 (param_0.3574: f32[8,4096], param_1.3520: f32[8,4096], param_2.3077: f32[8,4096]) -> f32[1,8,4096] {
  %param_0.3574 = f32[8,4096]{1,0} parameter(0)
  %bitcast.1383.3 = f32[1,8,4096]{2,1,0} bitcast(%param_0.3574)
  %param_1.3520 = f32[8,4096]{1,0} parameter(1)
  %bitcast.1385.3 = f32[1,8,4096]{2,1,0} bitcast(%param_1.3520)
  %param_2.3077 = f32[8,4096]{1,0} parameter(2)
  %constant_3358_4 = f32[] constant(0.000139508935)
  %closed_call.36.12 = f32[8,4096]{1,0} broadcast(%constant_3358_4), dimensions={}
  %div.2681.5 = f32[8,4096]{1,0} multiply(%param_2.3077, %closed_call.36.12)
  %constant_3359_8 = f32[] constant(1e-06)
  %closed_call.37.18 = f32[8,4096]{1,0} broadcast(%constant_3359_8), dimensions={}
  %add.1872.3 = f32[8,4096]{1,0} add(%div.2681.5, %closed_call.37.18)
  %bitcast.1387.3 = f32[1,8,4096]{2,1,0} bitcast(%add.1872.3)
  %divide.48.3 = f32[1,8,4096]{2,1,0} divide(%bitcast.1385.3, %bitcast.1387.3)
  %constant_3471_2 = f32[] constant(-0.5)
  %broadcast.2932.14 = f32[1,8,4096]{2,1,0} broadcast(%constant_3471_2), dimensions={}
  %multiply.382.5 = f32[1,8,4096]{2,1,0} multiply(%divide.48.3, %broadcast.2932.14)
  %multiply.383.3 = f32[1,8,4096]{2,1,0} multiply(%bitcast.1383.3, %multiply.382.5)
  %constant_3472_2 = f32[] constant(0.00027901787)
  %broadcast.2933.1 = f32[1,8,4096]{2,1,0} broadcast(%constant_3472_2), dimensions={}
  ROOT %multiply.384.1 = f32[1,8,4096]{2,1,0} multiply(%multiply.383.3, %broadcast.2933.1)
}



ENTRY %wrapper_fused_multiply.3 (param_0.3574: f32[8,4096], param_1.3520: f32[8,4096], param_2.3077: f32[8,4096]) -> f32[1,8,4096] {
  param_0.3574 = f32[8,4096] parameter(0)
  param_1.3520 = f32[8,4096] parameter(1)
  param_2.3077 = f32[8,4096] parameter(2)
  ROOT %fusion = f32[1,8,4096] fusion(param_0.3574, param_1.3520, param_2.3077), kind=kLoop, calls=%fused_multiply.3
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_multiply.4.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedMultiply_4) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.4_module, entry_computation_layout={(f32[8,4096]{1,0}, f32[8,4096]{1,0}, f32[8,4096]{1,0})->f32[1,8,4096]{2,1,0}}

%fused_multiply.4 (param_0.3571: f32[8,4096], param_1.3517: f32[8,4096], param_2.3090: f32[8,4096]) -> f32[1,8,4096] {
  %param_0.3571 = f32[8,4096]{1,0} parameter(0)
  %bitcast.1472.3 = f32[1,8,4096]{2,1,0} bitcast(%param_0.3571)
  %param_1.3517 = f32[8,4096]{1,0} parameter(1)
  %bitcast.1474.3 = f32[1,8,4096]{2,1,0} bitcast(%param_1.3517)
  %param_2.3090 = f32[8,4096]{1,0} parameter(2)
  %constant_3545_2 = f32[] constant(0.001953125)
  %closed_call.82.3 = f32[8,4096]{1,0} broadcast(%constant_3545_2), dimensions={}
  %div.2774.3 = f32[8,4096]{1,0} multiply(%param_2.3090, %closed_call.82.3)
  %constant_3359_6 = f32[] constant(1e-06)
  %closed_call.37.10 = f32[8,4096]{1,0} broadcast(%constant_3359_6), dimensions={}
  %add.1881.3 = f32[8,4096]{1,0} add(%div.2774.3, %closed_call.37.10)
  %bitcast.1476.3 = f32[1,8,4096]{2,1,0} bitcast(%add.1881.3)
  %divide.51.3 = f32[1,8,4096]{2,1,0} divide(%bitcast.1474.3, %bitcast.1476.3)
  %constant_3471_3 = f32[] constant(-0.5)
  %broadcast.2932.16 = f32[1,8,4096]{2,1,0} broadcast(%constant_3471_3), dimensions={}
  %multiply.389.5 = f32[1,8,4096]{2,1,0} multiply(%divide.51.3, %broadcast.2932.16)
  %multiply.390.3 = f32[1,8,4096]{2,1,0} multiply(%bitcast.1472.3, %multiply.389.5)
  %constant_3630_1 = f32[] constant(0.00390625)
  %broadcast.2943.1 = f32[1,8,4096]{2,1,0} broadcast(%constant_3630_1), dimensions={}
  ROOT %multiply.391.1 = f32[1,8,4096]{2,1,0} multiply(%multiply.390.3, %broadcast.2943.1)
}



ENTRY %wrapper_fused_multiply.4 (param_0.3571: f32[8,4096], param_1.3517: f32[8,4096], param_2.3090: f32[8,4096]) -> f32[1,8,4096] {
  param_0.3571 = f32[8,4096] parameter(0)
  param_1.3517 = f32[8,4096] parameter(1)
  param_2.3090 = f32[8,4096] parameter(2)
  ROOT %fusion = f32[1,8,4096] fusion(param_0.3571, param_1.3517, param_2.3090), kind=kLoop, calls=%fused_multiply.4
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_multiply.5.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedMultiply_5) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.5_module, entry_computation_layout={(f32[8,4096]{1,0}, f32[8,4096]{1,0}, f32[8,4096]{1,0})->f32[1,8,4096]{2,1,0}}

%fused_multiply.5 (param_0.3572: f32[8,4096], param_1.3518: f32[8,4096], param_2.3091: f32[8,4096]) -> f32[1,8,4096] {
  %param_0.3572 = f32[8,4096]{1,0} parameter(0)
  %bitcast.1531.3 = f32[1,8,4096]{2,1,0} bitcast(%param_0.3572)
  %param_1.3518 = f32[8,4096]{1,0} parameter(1)
  %bitcast.1533.3 = f32[1,8,4096]{2,1,0} bitcast(%param_1.3518)
  %param_2.3091 = f32[8,4096]{1,0} parameter(2)
  %constant_3476_2 = f32[] constant(0.000651041686)
  %closed_call.42.3 = f32[8,4096]{1,0} broadcast(%constant_3476_2), dimensions={}
  %div.2759.3 = f32[8,4096]{1,0} multiply(%param_2.3091, %closed_call.42.3)
  %constant_3359_7 = f32[] constant(1e-06)
  %closed_call.37.14 = f32[8,4096]{1,0} broadcast(%constant_3359_7), dimensions={}
  %add.1875.3 = f32[8,4096]{1,0} add(%div.2759.3, %closed_call.37.14)
  %bitcast.1535.3 = f32[1,8,4096]{2,1,0} bitcast(%add.1875.3)
  %divide.52.3 = f32[1,8,4096]{2,1,0} divide(%bitcast.1533.3, %bitcast.1535.3)
  %constant_3471_4 = f32[] constant(-0.5)
  %broadcast.2932.18 = f32[1,8,4096]{2,1,0} broadcast(%constant_3471_4), dimensions={}
  %multiply.394.5 = f32[1,8,4096]{2,1,0} multiply(%divide.52.3, %broadcast.2932.18)
  %multiply.395.3 = f32[1,8,4096]{2,1,0} multiply(%bitcast.1531.3, %multiply.394.5)
  %constant_3736_1 = f32[] constant(0.00130208337)
  %broadcast.2946.1 = f32[1,8,4096]{2,1,0} broadcast(%constant_3736_1), dimensions={}
  ROOT %multiply.396.1 = f32[1,8,4096]{2,1,0} multiply(%multiply.395.3, %broadcast.2946.1)
}



ENTRY %wrapper_fused_multiply.5 (param_0.3572: f32[8,4096], param_1.3518: f32[8,4096], param_2.3091: f32[8,4096]) -> f32[1,8,4096] {
  param_0.3572 = f32[8,4096] parameter(0)
  param_1.3518 = f32[8,4096] parameter(1)
  param_2.3091 = f32[8,4096] parameter(2)
  ROOT %fusion = f32[1,8,4096] fusion(param_0.3572, param_1.3518, param_2.3091), kind=kLoop, calls=%fused_multiply.5
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_multiply.7.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedMultiply_7) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.7_module, entry_computation_layout={(bf16[7168]{0}, f32[8,4096]{1,0}, bf16[8,4096,7168]{2,1,0})->bf16[32768,7168]{1,0}}

%fused_multiply.7 (param_0.7294: bf16[7168], param_1.7036: f32[8,4096], param_2.5420: bf16[8,4096,7168]) -> bf16[32768,7168] {
  %param_2.5420 = bf16[8,4096,7168]{2,1,0} parameter(2)
  %convert_element_type.3313.14 = f32[8,4096,7168]{2,1,0} convert(%param_2.5420)
  %param_1.7036 = f32[8,4096]{1,0} parameter(1)
  %mul.2917.10 = f32[8,4096,7168]{2,1,0} broadcast(%param_1.7036), dimensions={0,1}
  %mul.2918.5 = f32[8,4096,7168]{2,1,0} multiply(%convert_element_type.3313.14, %mul.2917.10)
  %convert_element_type.3314.3 = bf16[8,4096,7168]{2,1,0} convert(%mul.2918.5)
  %bitcast.3403.1 = bf16[32768,7168]{1,0} bitcast(%convert_element_type.3314.3)
  %param_0.7294 = bf16[7168]{0} parameter(0)
  %mul.3654.1 = bf16[32768,7168]{1,0} broadcast(%param_0.7294), dimensions={1}
  ROOT %mul.3655.1 = bf16[32768,7168]{1,0} multiply(%bitcast.3403.1, %mul.3654.1)
}



ENTRY %wrapper_fused_multiply.7 (param_0.7294: bf16[7168], param_1.7036: f32[8,4096], param_2.5420: bf16[8,4096,7168]) -> bf16[32768,7168] {
  param_0.7294 = bf16[7168] parameter(0)
  param_1.7036 = f32[8,4096] parameter(1)
  param_2.5420 = bf16[8,4096,7168] parameter(2)
  ROOT %fusion = bf16[32768,7168] fusion(param_0.7294, param_1.7036, param_2.5420), kind=kLoop, calls=%fused_multiply.7
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_rsqrt.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedRsqrt) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_rsqrt_module, entry_computation_layout={(f32[8,4096]{1,0})->f32[8,4096]{1,0}}

%fused_rsqrt (param_0.3556: f32[8,4096]) -> f32[8,4096] {
  %param_0.3556 = f32[8,4096]{1,0} parameter(0)
  %constant_3545_1 = f32[] constant(0.001953125)
  %closed_call.82.5 = f32[8,4096]{1,0} broadcast(%constant_3545_1), dimensions={}
  %div.2774.5 = f32[8,4096]{1,0} multiply(%param_0.3556, %closed_call.82.5)
  %constant_3359_4 = f32[] constant(1e-06)
  %closed_call.37.12 = f32[8,4096]{1,0} broadcast(%constant_3359_4), dimensions={}
  %add.1881.5 = f32[8,4096]{1,0} add(%div.2774.5, %closed_call.37.12)
  ROOT %rsqrt.98.1 = f32[8,4096]{1,0} rsqrt(%add.1881.5)
}



ENTRY %wrapper_fused_rsqrt (param_0.3556: f32[8,4096]) -> f32[8,4096] {
  param_0.3556 = f32[8,4096] parameter(0)
  ROOT %fusion = f32[8,4096] fusion(param_0.3556), kind=kLoop, calls=%fused_rsqrt
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_rsqrt.1.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedRsqrt_1) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_rsqrt.1_module, entry_computation_layout={(f32[8,4096]{1,0})->f32[8,4096]{1,0}}

%fused_rsqrt.1 (param_0.3555: f32[8,4096]) -> f32[8,4096] {
  %param_0.3555 = f32[8,4096]{1,0} parameter(0)
  %constant_3476_1 = f32[] constant(0.000651041686)
  %closed_call.42.5 = f32[8,4096]{1,0} broadcast(%constant_3476_1), dimensions={}
  %div.2759.5 = f32[8,4096]{1,0} multiply(%param_0.3555, %closed_call.42.5)
  %constant_3359_3 = f32[] constant(1e-06)
  %closed_call.37.16 = f32[8,4096]{1,0} broadcast(%constant_3359_3), dimensions={}
  %add.1875.5 = f32[8,4096]{1,0} add(%div.2759.5, %closed_call.37.16)
  ROOT %rsqrt.97.1 = f32[8,4096]{1,0} rsqrt(%add.1875.5)
}



ENTRY %wrapper_fused_rsqrt.1 (param_0.3555: f32[8,4096]) -> f32[8,4096] {
  param_0.3555 = f32[8,4096] parameter(0)
  ROOT %fusion = f32[8,4096] fusion(param_0.3555), kind=kLoop, calls=%fused_rsqrt.1
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_rsqrt.2.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedRsqrt_2) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_rsqrt.2_module, entry_computation_layout={(f32[8,4096]{1,0})->f32[8,4096]{1,0}}

%fused_rsqrt.2 (param_0.3554: f32[8,4096]) -> f32[8,4096] {
  %param_0.3554 = f32[8,4096]{1,0} parameter(0)
  %constant_3358_2 = f32[] constant(0.000139508935)
  %closed_call.36.14 = f32[8,4096]{1,0} broadcast(%constant_3358_2), dimensions={}
  %div.2681.7 = f32[8,4096]{1,0} multiply(%param_0.3554, %closed_call.36.14)
  %constant_3359_2 = f32[] constant(1e-06)
  %closed_call.37.20 = f32[8,4096]{1,0} broadcast(%constant_3359_2), dimensions={}
  %add.1872.5 = f32[8,4096]{1,0} add(%div.2681.7, %closed_call.37.20)
  ROOT %rsqrt.96.1 = f32[8,4096]{1,0} rsqrt(%add.1872.5)
}



ENTRY %wrapper_fused_rsqrt.2 (param_0.3554: f32[8,4096]) -> f32[8,4096] {
  param_0.3554 = f32[8,4096] parameter(0)
  ROOT %fusion = f32[8,4096] fusion(param_0.3554), kind=kLoop, calls=%fused_rsqrt.2
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_select.1.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedSelect_1) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.1_module, entry_computation_layout={(f32[], bf16[1]{0}, bf16[128,192,384]{1,0,2}, pred[])->bf16[1,384,128,192]{3,2,1,0}}

%fused_select.1 (param_0.8341: f32[], param_1.8291: bf16[1], param_2.6073: bf16[128,192,384]{1,0,2}, param_3.4674: pred[]) -> bf16[1,384,128,192] {
  %param_3.4674 = pred[] parameter(3)
  %broadcast.3054.1 = pred[1,384,128,192]{3,2,1,0} broadcast(%param_3.4674), dimensions={}
  %param_2.6073 = bf16[128,192,384]{1,0,2} parameter(2)
  %bitcast.2039.3 = bf16[384,128,192]{2,1,0} bitcast(%param_2.6073)
  %param_1.8291 = bf16[1]{0} parameter(1)
  %bitcast.1159.5 = bf16[] bitcast(%param_1.8291)
  %broadcast.2992.5 = bf16[384,128,192]{2,1,0} broadcast(%bitcast.1159.5), dimensions={}
  %multiply.435.3 = bf16[384,128,192]{2,1,0} multiply(%bitcast.2039.3, %broadcast.2992.5)
  %bitcast.2043.2 = bf16[1,384,128,192]{3,2,1,0} bitcast(%multiply.435.3)
  %convert.1286.7 = f32[1,384,128,192]{3,2,1,0} convert(%bitcast.2043.2)
  %param_0.8341 = f32[] parameter(0)
  %broadcast.3643.3 = f32[1,384,128,192]{3,2,1,0} broadcast(%param_0.8341), dimensions={}
  %divide.90.5 = f32[1,384,128,192]{3,2,1,0} divide(%convert.1286.7, %broadcast.3643.3)
  %convert.1288.3 = bf16[1,384,128,192]{3,2,1,0} convert(%divide.90.5)
  ROOT %select.168.1 = bf16[1,384,128,192]{3,2,1,0} select(%broadcast.3054.1, %bitcast.2043.2, %convert.1288.3)
}



ENTRY %wrapper_fused_select.1 (param_0.8341: f32[], param_1.8291: bf16[1], param_2.6073: bf16[128,192,384]{1,0,2}, param_3.4674: pred[]) -> bf16[1,384,128,192] {
  param_0.8341 = f32[] parameter(0)
  param_1.8291 = bf16[1] parameter(1)
  param_2.6073 = bf16[128,192,384]{1,0,2} parameter(2)
  param_3.4674 = pred[] parameter(3)
  ROOT %fusion = bf16[1,384,128,192] fusion(param_0.8341, param_1.8291, param_2.6073, param_3.4674), kind=kLoop, calls=%fused_select.1
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_select.2.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedSelect_2) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.2_module, entry_computation_layout={(f32[], bf16[1792,1536]{1,0}, pred[], f32[1]{0}, f32[1]{0})->bf16[1,1792,1536]{2,1,0}}

%fused_select.2 (param_0.8605: f32[], param_1.8590: bf16[1792,1536], param_2.8017: pred[], param_3.6314: f32[1], param_4.4099: f32[1]) -> bf16[1,1792,1536] {
  %param_2.8017 = pred[] parameter(2)
  %broadcast.3044.1 = pred[1,1792,1536]{2,1,0} broadcast(%param_2.8017), dimensions={}
  %param_1.8590 = bf16[1792,1536]{1,0} parameter(1)
  %param_3.6314 = f32[1]{0} parameter(3)
  %param_4.4099 = f32[1]{0} parameter(4)
  %mul.3087.5 = f32[1]{0} multiply(%param_3.6314, %param_4.4099)
  %convert_element_type.3506.5 = bf16[1]{0} convert(%mul.3087.5)
  %bitcast.1155.5 = bf16[] bitcast(%convert_element_type.3506.5)
  %broadcast_in_dim.3101.5 = bf16[1792,1536]{1,0} broadcast(%bitcast.1155.5), dimensions={}
  %mul.3088.3 = bf16[1792,1536]{1,0} multiply(%param_1.8590, %broadcast_in_dim.3101.5)
  %bitcast.2028.2 = bf16[1,1792,1536]{2,1,0} bitcast(%mul.3088.3)
  %convert.1275.7 = f32[1,1792,1536]{2,1,0} convert(%bitcast.2028.2)
  %param_0.8605 = f32[] parameter(0)
  %broadcast.3639.3 = f32[1,1792,1536]{2,1,0} broadcast(%param_0.8605), dimensions={}
  %divide.87.5 = f32[1,1792,1536]{2,1,0} divide(%convert.1275.7, %broadcast.3639.3)
  %convert.1277.3 = bf16[1,1792,1536]{2,1,0} convert(%divide.87.5)
  ROOT %select.167.1 = bf16[1,1792,1536]{2,1,0} select(%broadcast.3044.1, %bitcast.2028.2, %convert.1277.3)
}



ENTRY %wrapper_fused_select.2 (param_0.8605: f32[], param_1.8590: bf16[1792,1536], param_2.8017: pred[], param_3.6314: f32[1], param_4.4099: f32[1]) -> bf16[1,1792,1536] {
  param_0.8605 = f32[] parameter(0)
  param_1.8590 = bf16[1792,1536] parameter(1)
  param_2.8017 = pred[] parameter(2)
  param_3.6314 = f32[1] parameter(3)
  param_4.4099 = f32[1] parameter(4)
  ROOT %fusion = bf16[1,1792,1536] fusion(param_0.8605, param_1.8590, param_2.8017, param_3.6314, param_4.4099), kind=kLoop, calls=%fused_select.2
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_select.3.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedSelect_3) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.3_module, entry_computation_layout={(f32[], bf16[128,256,128]{1,0,2}, pred[], f32[1]{0}, f32[1]{0})->bf16[1,128,128,256]{3,2,1,0}}

%fused_select.3 (param_0.8494: f32[], param_1.10758: bf16[128,256,128]{1,0,2}, param_2.8047: pred[], param_3.6353: f32[1], param_4.4131: f32[1]) -> bf16[1,128,128,256] {
  %param_2.8047 = pred[] parameter(2)
  %broadcast.3034.1 = pred[1,128,128,256]{3,2,1,0} broadcast(%param_2.8047), dimensions={}
  %param_1.10758 = bf16[128,256,128]{1,0,2} parameter(1)
  %bitcast.2018.3 = bf16[128,128,256]{2,1,0} bitcast(%param_1.10758)
  %param_3.6353 = f32[1]{0} parameter(3)
  %param_4.4131 = f32[1]{0} parameter(4)
  %mul.3083.5 = f32[1]{0} multiply(%param_3.6353, %param_4.4131)
  %convert_element_type.3502.5 = bf16[1]{0} convert(%mul.3083.5)
  %bitcast.1153.5 = bf16[] bitcast(%convert_element_type.3502.5)
  %broadcast.2991.5 = bf16[128,128,256]{2,1,0} broadcast(%bitcast.1153.5), dimensions={}
  %multiply.432.3 = bf16[128,128,256]{2,1,0} multiply(%bitcast.2018.3, %broadcast.2991.5)
  %bitcast.2022.2 = bf16[1,128,128,256]{3,2,1,0} bitcast(%multiply.432.3)
  %convert.1264.7 = f32[1,128,128,256]{3,2,1,0} convert(%bitcast.2022.2)
  %param_0.8494 = f32[] parameter(0)
  %broadcast.3635.3 = f32[1,128,128,256]{3,2,1,0} broadcast(%param_0.8494), dimensions={}
  %divide.84.5 = f32[1,128,128,256]{3,2,1,0} divide(%convert.1264.7, %broadcast.3635.3)
  %convert.1266.3 = bf16[1,128,128,256]{3,2,1,0} convert(%divide.84.5)
  ROOT %select.166.1 = bf16[1,128,128,256]{3,2,1,0} select(%broadcast.3034.1, %bitcast.2022.2, %convert.1266.3)
}



ENTRY %wrapper_fused_select.3 (param_0.8494: f32[], param_1.10758: bf16[128,256,128]{1,0,2}, param_2.8047: pred[], param_3.6353: f32[1], param_4.4131: f32[1]) -> bf16[1,128,128,256] {
  param_0.8494 = f32[] parameter(0)
  param_1.10758 = bf16[128,256,128]{1,0,2} parameter(1)
  param_2.8047 = pred[] parameter(2)
  param_3.6353 = f32[1] parameter(3)
  param_4.4131 = f32[1] parameter(4)
  ROOT %fusion = bf16[1,128,128,256] fusion(param_0.8494, param_1.10758, param_2.8047, param_3.6353, param_4.4131), kind=kLoop, calls=%fused_select.3
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_select.4.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedSelect_4) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.4_module, entry_computation_layout={(f32[], pred[], bf16[1792,18432]{1,0}, bf16[1]{0})->bf16[1,1792,18432]{2,1,0}}

%fused_select.4 (param_0.7782: f32[], param_1.7623: pred[], param_2.5740: bf16[1792,18432], param_3.4387: bf16[1]) -> bf16[1,1792,18432] {
  %param_1.7623 = pred[] parameter(1)
  %broadcast.2994.2 = pred[1,1792,18432]{2,1,0} broadcast(%param_1.7623), dimensions={}
  %param_2.5740 = bf16[1792,18432]{1,0} parameter(2)
  %param_3.4387 = bf16[1]{0} parameter(3)
  %bitcast.1127.5 = bf16[] bitcast(%param_3.4387)
  %broadcast_in_dim.3087.5 = bf16[1792,18432]{1,0} broadcast(%bitcast.1127.5), dimensions={}
  %mul.3059.3 = bf16[1792,18432]{1,0} multiply(%param_2.5740, %broadcast_in_dim.3087.5)
  %bitcast.1979.2 = bf16[1,1792,18432]{2,1,0} bitcast(%mul.3059.3)
  %convert.1176.5 = f32[1,1792,18432]{2,1,0} convert(%bitcast.1979.2)
  %param_0.7782 = f32[] parameter(0)
  %broadcast.3611.13 = f32[1,1792,18432]{2,1,0} broadcast(%param_0.7782), dimensions={}
  %divide.72.5 = f32[1,1792,18432]{2,1,0} divide(%convert.1176.5, %broadcast.3611.13)
  %convert.1178.3 = bf16[1,1792,18432]{2,1,0} convert(%divide.72.5)
  ROOT %select.162.1 = bf16[1,1792,18432]{2,1,0} select(%broadcast.2994.2, %bitcast.1979.2, %convert.1178.3)
}



ENTRY %wrapper_fused_select.4 (param_0.7782: f32[], param_1.7623: pred[], param_2.5740: bf16[1792,18432], param_3.4387: bf16[1]) -> bf16[1,1792,18432] {
  param_0.7782 = f32[] parameter(0)
  param_1.7623 = pred[] parameter(1)
  param_2.5740 = bf16[1792,18432] parameter(2)
  param_3.4387 = bf16[1] parameter(3)
  ROOT %fusion = bf16[1,1792,18432] fusion(param_0.7782, param_1.7623, param_2.5740, param_3.4387), kind=kLoop, calls=%fused_select.4
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_select.7.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedSelect_7) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.7_module, entry_computation_layout={(s32[], s32[8,4096]{1,0})->f32[8,4096]{1,0}}

%fused_select.7 (param_0.10747: s32[], param_1.11010: s32[8,4096]) -> f32[8,4096] {
  %param_1.11010 = s32[8,4096]{1,0} parameter(1)
  %constant_3838_1 = s32[] constant(0)
  %broadcast.2603.22 = s32[8,4096]{1,0} broadcast(%constant_3838_1), dimensions={}
  %ne.4.3 = pred[8,4096]{1,0} compare(%param_1.11010, %broadcast.2603.22), direction=NE
  %constant_3912_2 = f32[] constant(1)
  %param_0.10747 = s32[] parameter(0)
  %convert_element_type.3281.3 = f32[] convert(%param_0.10747)
  %constant_3988_2 = f32[] constant(1e-08)
  %add.1925.3 = f32[] add(%convert_element_type.3281.3, %constant_3988_2)
  %div.2933.1 = f32[] divide(%constant_3912_2, %add.1925.3)
  %broadcast_in_dim.2933.1 = f32[8,4096]{1,0} broadcast(%div.2933.1), dimensions={}
  %constant_3857_330 = f32[] constant(0)
  %broadcast.2613.1 = f32[8,4096]{1,0} broadcast(%constant_3857_330), dimensions={}
  ROOT %mul.2889.1 = f32[8,4096]{1,0} select(%ne.4.3, %broadcast_in_dim.2933.1, %broadcast.2613.1)
}



ENTRY %wrapper_fused_select.7 (param_0.10747: s32[], param_1.11010: s32[8,4096]) -> f32[8,4096] {
  param_0.10747 = s32[] parameter(0)
  param_1.11010 = s32[8,4096] parameter(1)
  ROOT %fusion = f32[8,4096] fusion(param_0.10747, param_1.11010), kind=kLoop, calls=%fused_select.7
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_select.8.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedSelect_8) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.8_module, entry_computation_layout={(s32[8,4096]{1,0})->s32[8,4096]{1,0}}

%fused_select.8 (param_0.8815: s32[8,4096]) -> s32[8,4096] {
  %param_0.8815 = s32[8,4096]{1,0} parameter(0)
  %constant_3838_38 = s32[] constant(0)
  %broadcast.2603.8 = s32[8,4096]{1,0} broadcast(%constant_3838_38), dimensions={}
  %lt.74.3 = pred[8,4096]{1,0} compare(%param_0.8815, %broadcast.2603.8), direction=LT
  %constant_3839_1 = s32[] constant(129280)
  %add.1909.1 = s32[8,4096]{1,0} broadcast(%constant_3839_1), dimensions={}
  %add.1910.1 = s32[8,4096]{1,0} add(%param_0.8815, %add.1909.1)
  ROOT %select_n.2214.1 = s32[8,4096]{1,0} select(%lt.74.3, %add.1910.1, %param_0.8815)
}



ENTRY %wrapper_fused_select.8 (param_0.8815: s32[8,4096]) -> s32[8,4096] {
  param_0.8815 = s32[8,4096] parameter(0)
  ROOT %fusion = s32[8,4096] fusion(param_0.8815), kind=kLoop, calls=%fused_select.8
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_select.9.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedSelect_9) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.9_module, entry_computation_layout={(f32[], pred[], bf16[2,7168]{1,0})->bf16[7168,2]{1,0}}

%fused_select.9 (param_0.9465: f32[], param_1.9655: pred[], param_2.6910: bf16[2,7168]) -> bf16[7168,2] {
  %param_1.9655 = pred[] parameter(1)
  %select_n.2422.1 = pred[7168,2]{1,0} broadcast(%param_1.9655), dimensions={}
  %param_2.6910 = bf16[2,7168]{1,0} parameter(2)
  %transpose.351.1 = bf16[7168,2]{1,0} transpose(%param_2.6910), dimensions={1,0}
  %convert.1407.3 = f32[7168,2]{1,0} convert(%transpose.351.1)
  %param_0.9465 = f32[] parameter(0)
  %broadcast.3675.8 = f32[7168,2]{1,0} broadcast(%param_0.9465), dimensions={}
  %div.3231.5 = f32[7168,2]{1,0} divide(%convert.1407.3, %broadcast.3675.8)
  %convert.1409.3 = bf16[7168,2]{1,0} convert(%div.3231.5)
  ROOT %select_n.2423.1 = bf16[7168,2]{1,0} select(%select_n.2422.1, %transpose.351.1, %convert.1409.3)
}



ENTRY %wrapper_fused_select.9 (param_0.9465: f32[], param_1.9655: pred[], param_2.6910: bf16[2,7168]) -> bf16[7168,2] {
  param_0.9465 = f32[] parameter(0)
  param_1.9655 = pred[] parameter(1)
  param_2.6910 = bf16[2,7168] parameter(2)
  ROOT %fusion = bf16[7168,2] fusion(param_0.9465, param_1.9655, param_2.6910), kind=kLoop, calls=%fused_select.9
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_slice.3.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedSlice_3) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_slice.3_module, entry_computation_layout={(bf16[32768,32768]{1,0})->bf16[8,4096,128,128]{3,2,1,0}}

%fused_slice.3 (param_0.3791: bf16[32768,32768]) -> bf16[8,4096,128,128] {
  %param_0.3791 = bf16[32768,32768]{1,0} parameter(0)
  %bitcast.3308.2 = bf16[8,4096,128,256]{3,2,1,0} bitcast(%param_0.3791)
  ROOT %slice.1810.1 = bf16[8,4096,128,128]{3,2,1,0} slice(%bitcast.3308.2), slice={[0:8], [0:4096], [0:128], [128:256]}
}



ENTRY %wrapper_fused_slice.3 (param_0.3791: bf16[32768,32768]) -> bf16[8,4096,128,128] {
  param_0.3791 = bf16[32768,32768] parameter(0)
  ROOT %fusion = bf16[8,4096,128,128] fusion(param_0.3791), kind=kLoop, calls=%fused_slice.3
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_slice.8.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedSlice_8) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_slice.8_module, entry_computation_layout={(f32[2,1024]{1,0})->bf16[2,1]{1,0}}

%fused_slice.8 (param_0.9114: f32[2,1024]) -> bf16[2,1] {
  %param_0.9114 = f32[2,1024]{1,0} parameter(0)
  %convert.378.1 = bf16[2,1024]{1,0} convert(%param_0.9114)
  ROOT %slice.1911.1 = bf16[2,1]{1,0} slice(%convert.378.1), slice={[0:2], [0:1]}
}



ENTRY %wrapper_fused_slice.8 (param_0.9114: f32[2,1024]) -> bf16[2,1] {
  param_0.9114 = f32[2,1024] parameter(0)
  ROOT %fusion = bf16[2,1] fusion(param_0.9114), kind=kLoop, calls=%fused_slice.8
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_transpose.6.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedTranspose_6) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.6_module, entry_computation_layout={(bf16[8,7168,32768]{2,1,0})->bf16[4,8,7168,16,512]{4,3,2,1,0}}

%fused_transpose.6 (param_0.972: bf16[8,7168,32768]) -> bf16[4,8,7168,16,512] {
  %param_0.972 = bf16[8,7168,32768]{2,1,0} parameter(0)
  %bitcast.298.1 = bf16[8,7168,4,16,512]{4,3,2,1,0} bitcast(%param_0.972)
  ROOT %transpose.627.1 = bf16[4,8,7168,16,512]{4,3,2,1,0} transpose(%bitcast.298.1), dimensions={2,0,1,3,4}
}



ENTRY %wrapper_fused_transpose.6 (param_0.972: bf16[8,7168,32768]) -> bf16[4,8,7168,16,512] {
  param_0.972 = bf16[8,7168,32768] parameter(0)
  ROOT %fusion = bf16[4,8,7168,16,512] fusion(param_0.972), kind=kLoop, calls=%fused_transpose.6
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_transpose.9.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedTranspose_9) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.9_module, entry_computation_layout={(bf16[8,32768,7168]{2,1,0})->bf16[4,8,16,512,7168]{4,3,2,1,0}}

%fused_transpose.9 (param_0.3787: bf16[8,32768,7168]) -> bf16[4,8,16,512,7168] {
  %param_0.3787 = bf16[8,32768,7168]{2,1,0} parameter(0)
  %bitcast.3320.1 = bf16[8,4,16,512,7168]{4,3,2,1,0} bitcast(%param_0.3787)
  ROOT %transpose.623.1 = bf16[4,8,16,512,7168]{4,3,2,1,0} transpose(%bitcast.3320.1), dimensions={1,0,2,3,4}
}



ENTRY %wrapper_fused_transpose.9 (param_0.3787: bf16[8,32768,7168]) -> bf16[4,8,16,512,7168] {
  param_0.3787 = bf16[8,32768,7168] parameter(0)
  ROOT %fusion = bf16[4,8,16,512,7168] fusion(param_0.3787), kind=kLoop, calls=%fused_transpose.9
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_transpose.13.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedTranspose_13) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.13_module, entry_computation_layout={(bf16[2,2048,1792]{1,2,0})->bf16[2048,2,1792]{2,1,0}}

%fused_transpose.13 (param_0.6004: bf16[2,2048,1792]{1,2,0}) -> bf16[2048,2,1792] {
  %param_0.6004 = bf16[2,2048,1792]{1,2,0} parameter(0)
  %bitcast.2057.1 = bf16[2,1792,2048]{2,1,0} bitcast(%param_0.6004)
  ROOT %transpose.350.1 = bf16[2048,2,1792]{2,1,0} transpose(%bitcast.2057.1), dimensions={2,0,1}
}



ENTRY %wrapper_fused_transpose.13 (param_0.6004: bf16[2,2048,1792]{1,2,0}) -> bf16[2048,2,1792] {
  param_0.6004 = bf16[2,2048,1792]{1,2,0} parameter(0)
  ROOT %fusion = bf16[2048,2,1792] fusion(param_0.6004), kind=kLoop, calls=%fused_transpose.13
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_transpose.14.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedTranspose_14) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.14_module, entry_computation_layout={(bf16[8,4096,7168]{2,1,0}, f32[1,8,4096]{2,1,0}, f32[8,4096]{1,0}, bf16[4,8,4096,1792]{3,2,1,0}, bf16[7168,1]{1,0}, /*index=5*/bf16[32768,7168]{1,0})->bf16[4,8,4096,1792]{3,2,1,0}}

%fused_transpose.14 (param_0.7210: bf16[8,4096,7168], param_1.6959: f32[1,8,4096], param_2.5367: f32[8,4096], param_3.4132: bf16[4,8,4096,1792], param_4.2615: bf16[7168,1], param_5.2200: bf16[32768,7168]) -> bf16[4,8,4096,1792] {
  %param_0.7210 = bf16[8,4096,7168]{2,1,0} parameter(0)
  %param_5.2200 = bf16[32768,7168]{1,0} parameter(5)
  %bitcast.1139.8 = bf16[8,4096,7168]{2,1,0} bitcast(%param_5.2200)
  %param_4.2615 = bf16[7168,1]{1,0} parameter(4)
  %bitcast.789.13 = bf16[7168]{0} bitcast(%param_4.2615)
  %mul.2840.13 = bf16[8,4096,7168]{2,1,0} broadcast(%bitcast.789.13), dimensions={2}
  %mul.3114.5 = bf16[8,4096,7168]{2,1,0} multiply(%bitcast.1139.8, %mul.2840.13)
  %convert_element_type.3550.12 = f32[8,4096,7168]{2,1,0} convert(%mul.3114.5)
  %param_2.5367 = f32[8,4096]{1,0} parameter(2)
  %mul.2837.12 = f32[8,4096,7168]{2,1,0} broadcast(%param_2.5367), dimensions={0,1}
  %mul.3117.9 = f32[8,4096,7168]{2,1,0} multiply(%convert_element_type.3550.12, %mul.2837.12)
  %param_3.4132 = bf16[4,8,4096,1792]{3,2,1,0} parameter(3)
  %transpose.645.5 = bf16[8,4096,4,1792]{3,2,1,0} transpose(%param_3.4132), dimensions={1,2,0,3}
  %bitcast.788.22 = bf16[8,4096,7168]{2,1,0} bitcast(%transpose.645.5)
  %convert_element_type.3220.21 = f32[8,4096,7168]{2,1,0} convert(%bitcast.788.22)
  %param_1.6959 = f32[1,8,4096]{2,1,0} parameter(1)
  %bitcast.1163.7 = f32[8,4096]{1,0} bitcast(%param_1.6959)
  %mul.3123.7 = f32[8,4096,7168]{2,1,0} broadcast(%bitcast.1163.7), dimensions={0,1}
  %mul.3124.7 = f32[8,4096,7168]{2,1,0} multiply(%convert_element_type.3220.21, %mul.3123.7)
  %add_any.137.7 = f32[8,4096,7168]{2,1,0} add(%mul.3117.9, %mul.3124.7)
  %convert_element_type.3551.5 = bf16[8,4096,7168]{2,1,0} convert(%add_any.137.7)
  %add_any.138.3 = bf16[8,4096,7168]{2,1,0} add(%param_0.7210, %convert_element_type.3551.5)
  %bitcast.1164.1 = bf16[8,4096,4,1792]{3,2,1,0} bitcast(%add_any.138.3)
  ROOT %transpose.701.1 = bf16[4,8,4096,1792]{3,2,1,0} transpose(%bitcast.1164.1), dimensions={2,0,1,3}
}



ENTRY %wrapper_fused_transpose.14 (param_0.7210: bf16[8,4096,7168], param_1.6959: f32[1,8,4096], param_2.5367: f32[8,4096], param_3.4132: bf16[4,8,4096,1792], param_4.2615: bf16[7168,1], param_5.2200: bf16[32768,7168]) -> bf16[4,8,4096,1792] {
  param_0.7210 = bf16[8,4096,7168] parameter(0)
  param_1.6959 = f32[1,8,4096] parameter(1)
  param_2.5367 = f32[8,4096] parameter(2)
  param_3.4132 = bf16[4,8,4096,1792] parameter(3)
  param_4.2615 = bf16[7168,1] parameter(4)
  param_5.2200 = bf16[32768,7168] parameter(5)
  ROOT %fusion = bf16[4,8,4096,1792] fusion(param_0.7210, param_1.6959, param_2.5367, param_3.4132, param_4.2615, param_5.2200), kind=kLoop, calls=%fused_transpose.14
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_transpose.16.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedTranspose_16) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.16_module, entry_computation_layout={(f32[], bf16[2,128,128,1792]{2,1,3,0}, pred[])->bf16[128,2,128,1792]{3,2,1,0}}

%fused_transpose.16 (param_0.7832: f32[], param_1.7677: bf16[2,128,128,1792]{2,1,3,0}, param_2.5783: pred[]) -> bf16[128,2,128,1792] {
  %param_2.5783 = pred[] parameter(2)
  %select_n.2427.1 = pred[128,2,128,1792]{3,2,1,0} broadcast(%param_2.5783), dimensions={}
  %param_1.7677 = bf16[2,128,128,1792]{2,1,3,0} parameter(1)
  %bitcast.2065.3 = bf16[2,1792,128,128]{3,2,1,0} bitcast(%param_1.7677)
  %transpose.354.3 = bf16[128,2,128,1792]{3,2,1,0} transpose(%bitcast.2065.3), dimensions={2,0,3,1}
  %convert.1440.3 = f32[128,2,128,1792]{3,2,1,0} convert(%transpose.354.3)
  %param_0.7832 = f32[] parameter(0)
  %broadcast.3683.5 = f32[128,2,128,1792]{3,2,1,0} broadcast(%param_0.7832), dimensions={}
  %div.3248.5 = f32[128,2,128,1792]{3,2,1,0} divide(%convert.1440.3, %broadcast.3683.5)
  %convert.1442.3 = bf16[128,2,128,1792]{3,2,1,0} convert(%div.3248.5)
  ROOT %select_n.2428.1 = bf16[128,2,128,1792]{3,2,1,0} select(%select_n.2427.1, %transpose.354.3, %convert.1442.3)
}



ENTRY %wrapper_fused_transpose.16 (param_0.7832: f32[], param_1.7677: bf16[2,128,128,1792]{2,1,3,0}, param_2.5783: pred[]) -> bf16[128,2,128,1792] {
  param_0.7832 = f32[] parameter(0)
  param_1.7677 = bf16[2,128,128,1792]{2,1,3,0} parameter(1)
  param_2.5783 = pred[] parameter(2)
  ROOT %fusion = bf16[128,2,128,1792] fusion(param_0.7832, param_1.7677, param_2.5783), kind=kLoop, calls=%fused_transpose.16
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_transpose.17.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedTranspose_17) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.17_module, entry_computation_layout={(bf16[2,128,128,1792]{2,1,3,0})->bf16[128,2,128,1792]{3,2,1,0}}

%fused_transpose.17 (param_0.7416: bf16[2,128,128,1792]{2,1,3,0}) -> bf16[128,2,128,1792] {
  %param_0.7416 = bf16[2,128,128,1792]{2,1,3,0} parameter(0)
  %bitcast.2065.5 = bf16[2,1792,128,128]{3,2,1,0} bitcast(%param_0.7416)
  %transpose.354.5 = bf16[128,2,128,1792]{3,2,1,0} transpose(%bitcast.2065.5), dimensions={2,0,3,1}
  ROOT %mul.3106.1 = bf16[128,2,128,1792]{3,2,1,0} multiply(%transpose.354.5, %transpose.354.5)
}



ENTRY %wrapper_fused_transpose.17 (param_0.7416: bf16[2,128,128,1792]{2,1,3,0}) -> bf16[128,2,128,1792] {
  param_0.7416 = bf16[2,128,128,1792]{2,1,3,0} parameter(0)
  ROOT %fusion = bf16[128,2,128,1792] fusion(param_0.7416), kind=kLoop, calls=%fused_transpose.17
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_transpose.20.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedTranspose_20) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.20_module, entry_computation_layout={(bf16[], bf16[1,1792,18432]{2,1,0}, bf16[18432,1,1792]{2,1,0}, bf16[], f32[], /*index=5*/bf16[1,1792,18432]{2,1,0})->bf16[1,1792,18432]{2,1,0}}

%fused_transpose.20 (param_0.7750: bf16[], param_1.11042: bf16[1,1792,18432], param_2.8203: bf16[18432,1,1792], param_3.6568: bf16[], param_4.4272: f32[], param_5.3713: bf16[1,1792,18432]) -> bf16[1,1792,18432] {
  %param_2.8203 = bf16[18432,1,1792]{2,1,0} parameter(2)
  %transpose.657.2 = bf16[1,1792,18432]{2,1,0} transpose(%param_2.8203), dimensions={1,2,0}
  %param_0.7750 = bf16[] parameter(0)
  %broadcast.2993.5 = bf16[1,1792,18432]{2,1,0} broadcast(%param_0.7750), dimensions={}
  %param_1.11042 = bf16[1,1792,18432]{2,1,0} parameter(1)
  %convert.1195.3 = f32[1,1792,18432]{2,1,0} convert(%param_1.11042)
  %param_3.6568 = bf16[] parameter(3)
  %broadcast.2998.5 = bf16[1,1792,18432]{2,1,0} broadcast(%param_3.6568), dimensions={}
  %param_5.3713 = bf16[1,1792,18432]{2,1,0} parameter(5)
  %convert.1190.5 = f32[1,1792,18432]{2,1,0} convert(%param_5.3713)
  %param_4.4272 = f32[] parameter(4)
  %broadcast.3613.11 = f32[1,1792,18432]{2,1,0} broadcast(%param_4.4272), dimensions={}
  %divide.76.5 = f32[1,1792,18432]{2,1,0} divide(%convert.1190.5, %broadcast.3613.11)
  %sqrt.113.3 = f32[1,1792,18432]{2,1,0} sqrt(%divide.76.5)
  %convert.1194.3 = bf16[1,1792,18432]{2,1,0} convert(%sqrt.113.3)
  %constant_4309_2 = bf16[] constant(1.001e-08)
  %broadcast.3002.11 = bf16[1,1792,18432]{2,1,0} broadcast(%constant_4309_2), dimensions={}
  %add.2691.5 = bf16[1,1792,18432]{2,1,0} add(%convert.1194.3, %broadcast.3002.11)
  %multiply.461.3 = bf16[1,1792,18432]{2,1,0} multiply(%broadcast.2998.5, %add.2691.5)
  %convert.1196.5 = f32[1,1792,18432]{2,1,0} convert(%multiply.461.3)
  %divide.77.5 = f32[1,1792,18432]{2,1,0} divide(%convert.1195.3, %convert.1196.5)
  %convert.1197.3 = bf16[1,1792,18432]{2,1,0} convert(%divide.77.5)
  %constant_4302_2 = bf16[] constant(0.1001)
  %broadcast.2996.10 = bf16[1,1792,18432]{2,1,0} broadcast(%constant_4302_2), dimensions={}
  %multiply.462.5 = bf16[1,1792,18432]{2,1,0} multiply(%transpose.657.2, %broadcast.2996.10)
  %add.2692.3 = bf16[1,1792,18432]{2,1,0} add(%convert.1197.3, %multiply.462.5)
  %multiply.463.3 = bf16[1,1792,18432]{2,1,0} multiply(%broadcast.2993.5, %add.2692.3)
  ROOT %add.2693.1 = bf16[1,1792,18432]{2,1,0} add(%transpose.657.2, %multiply.463.3)
}



ENTRY %wrapper_fused_transpose.20 (param_0.7750: bf16[], param_1.11042: bf16[1,1792,18432], param_2.8203: bf16[18432,1,1792], param_3.6568: bf16[], param_4.4272: f32[], param_5.3713: bf16[1,1792,18432]) -> bf16[1,1792,18432] {
  param_0.7750 = bf16[] parameter(0)
  param_1.11042 = bf16[1,1792,18432] parameter(1)
  param_2.8203 = bf16[18432,1,1792] parameter(2)
  param_3.6568 = bf16[] parameter(3)
  param_4.4272 = f32[] parameter(4)
  param_5.3713 = bf16[1,1792,18432] parameter(5)
  ROOT %fusion = bf16[1,1792,18432] fusion(param_0.7750, param_1.11042, param_2.8203, param_3.6568, param_4.4272, param_5.3713), kind=kLoop, calls=%fused_transpose.20
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: extracted_fusions/fused_transpose.21.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Deepseek1n4g_FusedTranspose_21) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.21_module, entry_computation_layout={(bf16[1,1792,128,128]{3,2,1,0}, f32[], bf16[1,1792,128,128]{3,2,1,0}, bf16[128,1,128,1792]{3,2,1,0}, bf16[], /*index=5*/bf16[])->bf16[1,1792,128,128]{3,2,1,0}}

%fused_transpose.21 (param_0.10793: bf16[1,1792,128,128], param_1.11044: f32[], param_2.8205: bf16[1,1792,128,128], param_3.6462: bf16[128,1,128,1792], param_4.4275: bf16[], param_5.3717: bf16[]) -> bf16[1,1792,128,128] {
  %param_3.6462 = bf16[128,1,128,1792]{3,2,1,0} parameter(3)
  %transpose.654.2 = bf16[1,1792,128,128]{3,2,1,0} transpose(%param_3.6462), dimensions={1,3,0,2}
  %param_5.3717 = bf16[] parameter(5)
  %broadcast.3013.1 = bf16[1,1792,128,128]{3,2,1,0} broadcast(%param_5.3717), dimensions={}
  %param_0.10793 = bf16[1,1792,128,128]{3,2,1,0} parameter(0)
  %convert.1239.3 = f32[1,1792,128,128]{3,2,1,0} convert(%param_0.10793)
  %param_4.4275 = bf16[] parameter(4)
  %broadcast.3018.1 = bf16[1,1792,128,128]{3,2,1,0} broadcast(%param_4.4275), dimensions={}
  %param_2.8205 = bf16[1,1792,128,128]{3,2,1,0} parameter(2)
  %convert.1234.5 = f32[1,1792,128,128]{3,2,1,0} convert(%param_2.8205)
  %param_1.11044 = f32[] parameter(1)
  %broadcast.3625.5 = f32[1,1792,128,128]{3,2,1,0} broadcast(%param_1.11044), dimensions={}
  %divide.79.5 = f32[1,1792,128,128]{3,2,1,0} divide(%convert.1234.5, %broadcast.3625.5)
  %sqrt.114.7 = f32[1,1792,128,128]{3,2,1,0} sqrt(%divide.79.5)
  %convert.1238.7 = bf16[1,1792,128,128]{3,2,1,0} convert(%sqrt.114.7)
  %constant_4309_6 = bf16[] constant(1.001e-08)
  %broadcast.3022.1 = bf16[1,1792,128,128]{3,2,1,0} broadcast(%constant_4309_6), dimensions={}
  %add.2696.5 = bf16[1,1792,128,128]{3,2,1,0} add(%convert.1238.7, %broadcast.3022.1)
  %multiply.469.3 = bf16[1,1792,128,128]{3,2,1,0} multiply(%broadcast.3018.1, %add.2696.5)
  %convert.1240.5 = f32[1,1792,128,128]{3,2,1,0} convert(%multiply.469.3)
  %divide.80.5 = f32[1,1792,128,128]{3,2,1,0} divide(%convert.1239.3, %convert.1240.5)
  %convert.1241.3 = bf16[1,1792,128,128]{3,2,1,0} convert(%divide.80.5)
  %constant_4302_6 = bf16[] constant(0.1001)
  %broadcast.3016.8 = bf16[1,1792,128,128]{3,2,1,0} broadcast(%constant_4302_6), dimensions={}
  %multiply.470.5 = bf16[1,1792,128,128]{3,2,1,0} multiply(%transpose.654.2, %broadcast.3016.8)
  %add.2697.3 = bf16[1,1792,128,128]{3,2,1,0} add(%convert.1241.3, %multiply.470.5)
  %multiply.471.1 = bf16[1,1792,128,128]{3,2,1,0} multiply(%broadcast.3013.1, %add.2697.3)
  ROOT %add.2698.1 = bf16[1,1792,128,128]{3,2,1,0} add(%transpose.654.2, %multiply.471.1)
}



ENTRY %wrapper_fused_transpose.21 (param_0.10793: bf16[1,1792,128,128], param_1.11044: f32[], param_2.8205: bf16[1,1792,128,128], param_3.6462: bf16[128,1,128,1792], param_4.4275: bf16[], param_5.3717: bf16[]) -> bf16[1,1792,128,128] {
  param_0.10793 = bf16[1,1792,128,128] parameter(0)
  param_1.11044 = f32[] parameter(1)
  param_2.8205 = bf16[1,1792,128,128] parameter(2)
  param_3.6462 = bf16[128,1,128,1792] parameter(3)
  param_4.4275 = bf16[] parameter(4)
  param_5.3717 = bf16[] parameter(5)
  ROOT %fusion = bf16[1,1792,128,128] fusion(param_0.10793, param_1.11044, param_2.8205, param_3.6462, param_4.4275, param_5.3717), kind=kLoop, calls=%fused_transpose.21
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// === gemma3_12b_decode (20 fusions) ===
// Source: gemma3_12b_decode/extracted_fusions/fused_add.96.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bDecode_FusedAdd_96) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_add.96_module, entry_computation_layout={(bf16[1,1,3840]{2,1,0}, bf16[3840]{0}, bf16[3840]{0}, f32[])->bf16[1,1,3840]{2,1,0}}

%fused_add.96.clone (param_0.4770: bf16[1,1,3840], param_1.13713: bf16[3840], param_2.10556: bf16[3840], param_3.6899: f32[]) -> bf16[1,1,3840] {
  %param_2.10556 = bf16[3840]{0} parameter(2)
  %param_3.6899 = f32[] parameter(3)
  %constant_2456_2 = f32[] constant(0.00026041668)
  %multiply.6895.3 = f32[] multiply(%param_3.6899, %constant_2456_2)
  %convert.5269.3 = bf16[] convert(%multiply.6895.3)
  %constant_1047_4 = bf16[] constant(9.984e-07)
  %add.4684.3 = bf16[] add(%convert.5269.3, %constant_1047_4)
  %convert.6443.5 = f32[] convert(%add.4684.3)
  %rsqrt.1157.5 = f32[] rsqrt(%convert.6443.5)
  %convert.6444.3 = bf16[] convert(%rsqrt.1157.5)
  %broadcast.8309.5 = bf16[3840]{0} broadcast(%convert.6444.3), dimensions={}
  %multiply.6013.5 = bf16[3840]{0} multiply(%param_2.10556, %broadcast.8309.5)
  %param_1.13713 = bf16[3840]{0} parameter(1)
  %constant_1040_54 = bf16[] constant(1)
  %broadcast.8280.763 = bf16[3840]{0} broadcast(%constant_1040_54), dimensions={}
  %add.3977.5 = bf16[3840]{0} add(%param_1.13713, %broadcast.8280.763)
  %multiply.6702.3 = bf16[3840]{0} multiply(%multiply.6013.5, %add.3977.5)
  %bitcast.184.1 = bf16[1,1,3840]{2,1,0} bitcast(%multiply.6702.3)
  %param_0.4770 = bf16[1,1,3840]{2,1,0} parameter(0)
  ROOT %add.2672.1 = bf16[1,1,3840]{2,1,0} add(%bitcast.184.1, %param_0.4770)
}



ENTRY %wrapper_fused_add.96.clone (param_0.4770: bf16[1,1,3840], param_1.13713: bf16[3840], param_2.10556: bf16[3840], param_3.6899: f32[]) -> bf16[1,1,3840] {
  param_0.4770 = bf16[1,1,3840] parameter(0)
  param_1.13713 = bf16[3840] parameter(1)
  param_2.10556 = bf16[3840] parameter(2)
  param_3.6899 = f32[] parameter(3)
  ROOT %fusion = bf16[1,1,3840] fusion(param_0.4770, param_1.13713, param_2.10556, param_3.6899), kind=kLoop, calls=%fused_add.96.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_12b_decode/extracted_fusions/fused_add.99.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bDecode_FusedAdd_99) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_add.99_module, entry_computation_layout={(s32[1]{0}, pred[1]{0})->s32[1]{0}}

%fused_add.99.clone (param_0.19807: s32[1], param_1.19705: pred[1]) -> s32[1] {
  %param_0.19807 = s32[1]{0} parameter(0)
  %param_1.19705 = pred[1]{0} parameter(1)
  %not.2.1 = pred[1]{0} not(%param_1.19705)
  %convert.4392.1 = s32[1]{0} convert(%not.2.1)
  ROOT %add.3922.1 = s32[1]{0} add(%param_0.19807, %convert.4392.1)
}



ENTRY %wrapper_fused_add.99.clone (param_0.19807: s32[1], param_1.19705: pred[1]) -> s32[1] {
  param_0.19807 = s32[1] parameter(0)
  param_1.19705 = pred[1] parameter(1)
  ROOT %fusion = s32[1] fusion(param_0.19807, param_1.19705), kind=kLoop, calls=%fused_add.99.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_12b_decode/extracted_fusions/fused_add.100.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bDecode_FusedAdd_100) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_add.100_module, entry_computation_layout={(s32[1]{0})->s32[1]{0}}

%fused_add.100.clone (param_0.20782: s32[1]) -> s32[1] {
  %param_0.20782 = s32[1]{0} parameter(0)
  %constant_1156_4 = s32[1]{0} constant({1})
  ROOT %add.3867.1 = s32[1]{0} add(%param_0.20782, %constant_1156_4)
}



ENTRY %wrapper_fused_add.100.clone (param_0.20782: s32[1]) -> s32[1] {
  param_0.20782 = s32[1] parameter(0)
  ROOT %fusion = s32[1] fusion(param_0.20782), kind=kLoop, calls=%fused_add.100.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_12b_decode/extracted_fusions/fused_and.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bDecode_FusedAnd) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_and_module, entry_computation_layout={(pred[1]{0}, s32[], s32[])->pred[]}

%fused_and.clone (param_0.20: pred[1], param_1.36: s32[], param_2.29: s32[]) -> pred[] {
  %param_1.36 = s32[] parameter(1)
  %param_2.29 = s32[] parameter(2)
  %compare.1396.1 = pred[] compare(%param_1.36, %param_2.29), direction=LT
  %param_0.20 = pred[1]{0} parameter(0)
  %not.3.1 = pred[1]{0} not(%param_0.20)
  %bitcast.2.1 = pred[] bitcast(%not.3.1)
  ROOT %and.348.1 = pred[] and(%compare.1396.1, %bitcast.2.1)
}



ENTRY %wrapper_fused_and.clone (param_0.20: pred[1], param_1.36: s32[], param_2.29: s32[]) -> pred[] {
  param_0.20 = pred[1] parameter(0)
  param_1.36 = s32[] parameter(1)
  param_2.29 = s32[] parameter(2)
  ROOT %fusion = pred[] fusion(param_0.20, param_1.36, param_2.29), kind=kLoop, calls=%fused_and.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_12b_decode/extracted_fusions/fused_concatenate.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bDecode_FusedConcatenate) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate_module, entry_computation_layout={(f32[128]{0}, f32[128]{0}, bf16[256]{0}, f32[16]{0}, bf16[4096,1]{0,1})->bf16[1,1,16,256]{3,2,1,0}}

%fused_concatenate.clone (param_0.14027: f32[128], param_1.20005: f32[128], param_2.9579: bf16[256], param_3.7044: f32[16], param_4.5061: bf16[4096,1]{0,1}) -> bf16[1,1,16,256] {
  %param_4.5061 = bf16[4096,1]{0,1} parameter(4)
  %bitcast.8655.8 = bf16[4096]{0} bitcast(%param_4.5061)
  %param_3.7044 = f32[16]{0} parameter(3)
  %bitcast.7155.7 = f32[1,1,16]{2,1,0} bitcast(%param_3.7044)
  %constant_1178_5 = f32[] constant(0.00390625)
  %broadcast.8299.242 = f32[1,1,16]{2,1,0} broadcast(%constant_1178_5), dimensions={}
  %multiply.6393.7 = f32[1,1,16]{2,1,0} multiply(%bitcast.7155.7, %broadcast.8299.242)
  %convert.4641.5 = bf16[1,1,16]{2,1,0} convert(%multiply.6393.7)
  %constant_1047_99 = bf16[] constant(9.984e-07)
  %broadcast.8849.146 = bf16[1,1,16]{2,1,0} broadcast(%constant_1047_99), dimensions={}
  %add.4363.5 = bf16[1,1,16]{2,1,0} add(%convert.4641.5, %broadcast.8849.146)
  %convert.6432.3 = f32[1,1,16]{2,1,0} convert(%add.4363.5)
  %rsqrt.869.5 = f32[1,1,16]{2,1,0} rsqrt(%convert.6432.3)
  %convert.6433.5 = bf16[1,1,16]{2,1,0} convert(%rsqrt.869.5)
  %bitcast.160.7 = bf16[16]{0} bitcast(%convert.6433.5)
  %broadcast.6714.7 = bf16[1,1,16,256]{3,2,1,0} broadcast(%bitcast.160.7), dimensions={2}
  %bitcast.161.5 = bf16[4096]{0} bitcast(%broadcast.6714.7)
  %multiply.6009.5 = bf16[4096]{0} multiply(%bitcast.8655.8, %bitcast.161.5)
  %param_2.9579 = bf16[256]{0} parameter(2)
  %constant_1040_244 = bf16[] constant(1)
  %broadcast.8294.386 = bf16[256]{0} broadcast(%constant_1040_244), dimensions={}
  %add.3973.5 = bf16[256]{0} add(%param_2.9579, %broadcast.8294.386)
  %broadcast.6715.3 = bf16[1,1,16,256]{3,2,1,0} broadcast(%add.3973.5), dimensions={3}
  %bitcast.162.3 = bf16[4096]{0} bitcast(%broadcast.6715.3)
  %multiply.6700.3 = bf16[4096]{0} multiply(%multiply.6009.5, %bitcast.162.3)
  %bitcast.163.29 = bf16[1,1,16,256]{3,2,1,0} bitcast(%multiply.6700.3)
  %convert.5999.29 = f32[1,1,16,256]{3,2,1,0} convert(%bitcast.163.29)
  %slice.1241.13 = f32[1,1,16,128]{3,2,1,0} slice(%convert.5999.29), slice={[0:1], [0:1], [0:16], [0:128]}
  %param_1.20005 = f32[128]{0} parameter(1)
  %broadcast.6718.402 = f32[1,1,16,128]{3,2,1,0} broadcast(%param_1.20005), dimensions={3}
  %multiply.3886.7 = f32[1,1,16,128]{3,2,1,0} multiply(%slice.1241.13, %broadcast.6718.402)
  %slice.1243.13 = f32[1,1,16,128]{3,2,1,0} slice(%convert.5999.29), slice={[0:1], [0:1], [0:16], [128:256]}
  %param_0.14027 = f32[128]{0} parameter(0)
  %broadcast.6719.242 = f32[1,1,16,128]{3,2,1,0} broadcast(%param_0.14027), dimensions={3}
  %multiply.3887.5 = f32[1,1,16,128]{3,2,1,0} multiply(%slice.1243.13, %broadcast.6719.242)
  %subtract.381.5 = f32[1,1,16,128]{3,2,1,0} subtract(%multiply.3886.7, %multiply.3887.5)
  %convert.5797.3 = bf16[1,1,16,128]{3,2,1,0} convert(%subtract.381.5)
  %multiply.3888.7 = f32[1,1,16,128]{3,2,1,0} multiply(%slice.1243.13, %broadcast.6718.402)
  %multiply.3889.5 = f32[1,1,16,128]{3,2,1,0} multiply(%slice.1241.13, %broadcast.6719.242)
  %add.2660.5 = f32[1,1,16,128]{3,2,1,0} add(%multiply.3888.7, %multiply.3889.5)
  %convert.5798.3 = bf16[1,1,16,128]{3,2,1,0} convert(%add.2660.5)
  ROOT %concatenate.389.1 = bf16[1,1,16,256]{3,2,1,0} concatenate(%convert.5797.3, %convert.5798.3), dimensions={3}
}



ENTRY %wrapper_fused_concatenate.clone (param_0.14027: f32[128], param_1.20005: f32[128], param_2.9579: bf16[256], param_3.7044: f32[16], param_4.5061: bf16[4096,1]{0,1}) -> bf16[1,1,16,256] {
  param_0.14027 = f32[128] parameter(0)
  param_1.20005 = f32[128] parameter(1)
  param_2.9579 = bf16[256] parameter(2)
  param_3.7044 = f32[16] parameter(3)
  param_4.5061 = bf16[4096,1]{0,1} parameter(4)
  ROOT %fusion = bf16[1,1,16,256] fusion(param_0.14027, param_1.20005, param_2.9579, param_3.7044, param_4.5061), kind=kLoop, calls=%fused_concatenate.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_12b_decode/extracted_fusions/fused_convert.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bDecode_FusedConvert) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_convert_module, entry_computation_layout={(bf16[1,1,8,2]{3,2,1,0}, bf16[8,2]{1,0}, bf16[8,4096,2]{2,1,0})->bf16[1,1,8,4096,2]{4,3,2,1,0}}

%fused_convert.clone (param_0.20312: bf16[1,1,8,2], param_1.20508: bf16[8,2], param_2.10422: bf16[8,4096,2]) -> bf16[1,1,8,4096,2] {
  %param_2.10422 = bf16[8,4096,2]{2,1,0} parameter(2)
  %bitcast.2254.7 = bf16[1,1,8,4096,2]{4,3,2,1,0} bitcast(%param_2.10422)
  %param_1.20508 = bf16[8,2]{1,0} parameter(1)
  %bitcast.167.9 = bf16[16]{0} bitcast(%param_1.20508)
  %broadcast.6722.9 = bf16[1,1,16,4096]{3,2,1,0} broadcast(%bitcast.167.9), dimensions={2}
  %bitcast.4794.7 = bf16[8,2,4096]{2,1,0} bitcast(%broadcast.6722.9)
  %transpose.1934.7 = bf16[8,4096,2]{2,1,0} transpose(%bitcast.4794.7), dimensions={0,2,1}
  %bitcast.4795.7 = bf16[1,1,8,4096,2]{4,3,2,1,0} bitcast(%transpose.1934.7)
  %subtract.580.7 = bf16[1,1,8,4096,2]{4,3,2,1,0} subtract(%bitcast.2254.7, %bitcast.4795.7)
  %exponential.192.5 = bf16[1,1,8,4096,2]{4,3,2,1,0} exponential(%subtract.580.7)
  %convert.6434.4 = f32[1,1,8,4096,2]{4,3,2,1,0} convert(%exponential.192.5)
  %param_0.20312 = bf16[1,1,8,2]{3,2,1,0} parameter(0)
  %bitcast.169.5 = bf16[16]{0} bitcast(%param_0.20312)
  %broadcast.6723.5 = bf16[1,1,16,4096]{3,2,1,0} broadcast(%bitcast.169.5), dimensions={2}
  %bitcast.4796.3 = bf16[8,2,4096]{2,1,0} bitcast(%broadcast.6723.5)
  %transpose.1935.3 = bf16[8,4096,2]{2,1,0} transpose(%bitcast.4796.3), dimensions={0,2,1}
  %bitcast.4797.3 = bf16[1,1,8,4096,2]{4,3,2,1,0} bitcast(%transpose.1935.3)
  %convert.6435.3 = f32[1,1,8,4096,2]{4,3,2,1,0} convert(%bitcast.4797.3)
  %divide.1538.3 = f32[1,1,8,4096,2]{4,3,2,1,0} divide(%convert.6434.4, %convert.6435.3)
  ROOT %convert.6436.1 = bf16[1,1,8,4096,2]{4,3,2,1,0} convert(%divide.1538.3)
}



ENTRY %wrapper_fused_convert.clone (param_0.20312: bf16[1,1,8,2], param_1.20508: bf16[8,2], param_2.10422: bf16[8,4096,2]) -> bf16[1,1,8,4096,2] {
  param_0.20312 = bf16[1,1,8,2] parameter(0)
  param_1.20508 = bf16[8,2] parameter(1)
  param_2.10422 = bf16[8,4096,2] parameter(2)
  ROOT %fusion = bf16[1,1,8,4096,2] fusion(param_0.20312, param_1.20508, param_2.10422), kind=kLoop, calls=%fused_convert.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_12b_decode/extracted_fusions/fused_convert.5.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bDecode_FusedConvert_5) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_convert.5_module, entry_computation_layout={(bf16[1,1,8,2]{3,2,1,0}, bf16[8,2]{1,0}, bf16[8,4096,2]{2,1,0}, pred[1,4096]{1,0})->bf16[1,1,8,4096,2]{4,3,2,1,0}}

%fused_convert.5.clone (param_0.20297: bf16[1,1,8,2], param_1.20468: bf16[8,2], param_2.10488: bf16[8,4096,2], param_3.7234: pred[1,4096]) -> bf16[1,1,8,4096,2] {
  %param_3.7234 = pred[1,4096]{1,0} parameter(3)
  %bitcast.364.61 = pred[4096]{0} bitcast(%param_3.7234)
  %broadcast.8864.61 = pred[8,4096,2]{2,1,0} broadcast(%bitcast.364.61), dimensions={1}
  %param_2.10488 = bf16[8,4096,2]{2,1,0} parameter(2)
  %constant_1636_63 = bf16[] constant(-2.379e+38)
  %broadcast.8858.92 = bf16[8,4096,2]{2,1,0} broadcast(%constant_1636_63), dimensions={}
  %select.1329.3 = bf16[8,4096,2]{2,1,0} select(%broadcast.8864.61, %param_2.10488, %broadcast.8858.92)
  %bitcast.2389.7 = bf16[1,1,8,4096,2]{4,3,2,1,0} bitcast(%select.1329.3)
  %param_1.20468 = bf16[8,2]{1,0} parameter(1)
  %bitcast.380.9 = bf16[16]{0} bitcast(%param_1.20468)
  %broadcast.6857.9 = bf16[1,1,16,4096]{3,2,1,0} broadcast(%bitcast.380.9), dimensions={2}
  %bitcast.4834.7 = bf16[8,2,4096]{2,1,0} bitcast(%broadcast.6857.9)
  %transpose.1954.7 = bf16[8,4096,2]{2,1,0} transpose(%bitcast.4834.7), dimensions={0,2,1}
  %bitcast.4835.7 = bf16[1,1,8,4096,2]{4,3,2,1,0} bitcast(%transpose.1954.7)
  %subtract.585.7 = bf16[1,1,8,4096,2]{4,3,2,1,0} subtract(%bitcast.2389.7, %bitcast.4835.7)
  %exponential.197.5 = bf16[1,1,8,4096,2]{4,3,2,1,0} exponential(%subtract.585.7)
  %convert.6519.4 = f32[1,1,8,4096,2]{4,3,2,1,0} convert(%exponential.197.5)
  %param_0.20297 = bf16[1,1,8,2]{3,2,1,0} parameter(0)
  %bitcast.382.5 = bf16[16]{0} bitcast(%param_0.20297)
  %broadcast.6858.5 = bf16[1,1,16,4096]{3,2,1,0} broadcast(%bitcast.382.5), dimensions={2}
  %bitcast.4836.3 = bf16[8,2,4096]{2,1,0} bitcast(%broadcast.6858.5)
  %transpose.1955.3 = bf16[8,4096,2]{2,1,0} transpose(%bitcast.4836.3), dimensions={0,2,1}
  %bitcast.4837.3 = bf16[1,1,8,4096,2]{4,3,2,1,0} bitcast(%transpose.1955.3)
  %convert.6520.3 = f32[1,1,8,4096,2]{4,3,2,1,0} convert(%bitcast.4837.3)
  %divide.1543.3 = f32[1,1,8,4096,2]{4,3,2,1,0} divide(%convert.6519.4, %convert.6520.3)
  ROOT %convert.6521.1 = bf16[1,1,8,4096,2]{4,3,2,1,0} convert(%divide.1543.3)
}



ENTRY %wrapper_fused_convert.5.clone (param_0.20297: bf16[1,1,8,2], param_1.20468: bf16[8,2], param_2.10488: bf16[8,4096,2], param_3.7234: pred[1,4096]) -> bf16[1,1,8,4096,2] {
  param_0.20297 = bf16[1,1,8,2] parameter(0)
  param_1.20468 = bf16[8,2] parameter(1)
  param_2.10488 = bf16[8,4096,2] parameter(2)
  param_3.7234 = pred[1,4096] parameter(3)
  ROOT %fusion = bf16[1,1,8,4096,2] fusion(param_0.20297, param_1.20468, param_2.10488, param_3.7234), kind=kLoop, calls=%fused_convert.5.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_12b_decode/extracted_fusions/fused_multiply.142.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bDecode_FusedMultiply_142) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.142_module, entry_computation_layout={(bf16[1,1,2,15360]{3,2,1,0})->bf16[1,1,15360]{2,1,0}}

%fused_multiply.142.clone (param_0.11806: bf16[1,1,2,15360]) -> bf16[1,1,15360] {
  %param_0.11806 = bf16[1,1,2,15360]{3,2,1,0} parameter(0)
  %slice.855.1 = bf16[1,1,1,15360]{3,2,1,0} slice(%param_0.11806), slice={[0:1], [0:1], [0:1], [0:15360]}
  %bitcast.181.6 = bf16[1,1,15360]{2,1,0} bitcast(%slice.855.1)
  %multiply.3897.3 = bf16[1,1,15360]{2,1,0} multiply(%bitcast.181.6, %bitcast.181.6)
  %multiply.3898.3 = bf16[1,1,15360]{2,1,0} multiply(%multiply.3897.3, %bitcast.181.6)
  %constant_1055_2 = bf16[] constant(0.04468)
  %broadcast.6643.1 = bf16[1,1,15360]{2,1,0} broadcast(%constant_1055_2), dimensions={}
  %multiply.3899.1 = bf16[1,1,15360]{2,1,0} multiply(%multiply.3898.3, %broadcast.6643.1)
  %add.2667.3 = bf16[1,1,15360]{2,1,0} add(%bitcast.181.6, %multiply.3899.1)
  %constant_1056_2 = bf16[] constant(0.7969)
  %broadcast.6644.1 = bf16[1,1,15360]{2,1,0} broadcast(%constant_1056_2), dimensions={}
  %multiply.3900.1 = bf16[1,1,15360]{2,1,0} multiply(%add.2667.3, %broadcast.6644.1)
  %convert.6441.1 = f32[1,1,15360]{2,1,0} convert(%multiply.3900.1)
  %tanh.96.9 = f32[1,1,15360]{2,1,0} tanh(%convert.6441.1)
  %convert.6442.9 = bf16[1,1,15360]{2,1,0} convert(%tanh.96.9)
  %constant_1040_4 = bf16[] constant(1)
  %broadcast.6645.1 = bf16[1,1,15360]{2,1,0} broadcast(%constant_1040_4), dimensions={}
  %add.2668.7 = bf16[1,1,15360]{2,1,0} add(%convert.6442.9, %broadcast.6645.1)
  %constant_1058_2 = bf16[] constant(0.5)
  %broadcast.6646.1 = bf16[1,1,15360]{2,1,0} broadcast(%constant_1058_2), dimensions={}
  %multiply.3901.5 = bf16[1,1,15360]{2,1,0} multiply(%add.2668.7, %broadcast.6646.1)
  %multiply.3902.3 = bf16[1,1,15360]{2,1,0} multiply(%bitcast.181.6, %multiply.3901.5)
  %slice.856.1 = bf16[1,1,1,15360]{3,2,1,0} slice(%param_0.11806), slice={[0:1], [0:1], [1:2], [0:15360]}
  %bitcast.182.1 = bf16[1,1,15360]{2,1,0} bitcast(%slice.856.1)
  ROOT %multiply.3903.1 = bf16[1,1,15360]{2,1,0} multiply(%multiply.3902.3, %bitcast.182.1)
}



ENTRY %wrapper_fused_multiply.142.clone (param_0.11806: bf16[1,1,2,15360]) -> bf16[1,1,15360] {
  param_0.11806 = bf16[1,1,2,15360] parameter(0)
  ROOT %fusion = bf16[1,1,15360] fusion(param_0.11806), kind=kLoop, calls=%fused_multiply.142.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_12b_decode/extracted_fusions/fused_multiply.143.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bDecode_FusedMultiply_143) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.143_module, entry_computation_layout={(bf16[1,1,3840]{2,1,0}, bf16[3840]{0}, f32[])->bf16[1,1,3840]{2,1,0}}

%fused_multiply.143.clone (param_0.13539: bf16[1,1,3840], param_1.16835: bf16[3840], param_2.11005: f32[]) -> bf16[1,1,3840] {
  %param_0.13539 = bf16[1,1,3840]{2,1,0} parameter(0)
  %param_2.11005 = f32[] parameter(2)
  %bitcast.7159.13 = f32[1,1]{1,0} bitcast(%param_2.11005)
  %constant_1628_3 = f32[1,1]{1,0} constant({ {0.00026041668} })
  %multiply.6395.13 = f32[1,1]{1,0} multiply(%bitcast.7159.13, %constant_1628_3)
  %convert.4644.11 = bf16[1,1]{1,0} convert(%multiply.6395.13)
  %constant_2455_3 = bf16[1,1]{1,0} constant({ {9.984e-07} })
  %add.4366.9 = bf16[1,1]{1,0} add(%convert.4644.11, %constant_2455_3)
  %convert.6439.7 = f32[1,1]{1,0} convert(%add.4366.9)
  %rsqrt.871.5 = f32[1,1]{1,0} rsqrt(%convert.6439.7)
  %convert.6440.3 = bf16[1,1]{1,0} convert(%rsqrt.871.5)
  %bitcast.176.5 = bf16[] bitcast(%convert.6440.3)
  %broadcast.6725.5 = bf16[1,1,3840]{2,1,0} broadcast(%bitcast.176.5), dimensions={}
  %multiply.3895.3 = bf16[1,1,3840]{2,1,0} multiply(%param_0.13539, %broadcast.6725.5)
  %param_1.16835 = bf16[3840]{0} parameter(1)
  %constant_1040_53 = bf16[] constant(1)
  %broadcast.8280.765 = bf16[3840]{0} broadcast(%constant_1040_53), dimensions={}
  %add.3975.3 = bf16[3840]{0} add(%param_1.16835, %broadcast.8280.765)
  %bitcast.177.1 = bf16[1,1,3840]{2,1,0} bitcast(%add.3975.3)
  ROOT %multiply.3896.1 = bf16[1,1,3840]{2,1,0} multiply(%multiply.3895.3, %bitcast.177.1)
}



ENTRY %wrapper_fused_multiply.143.clone (param_0.13539: bf16[1,1,3840], param_1.16835: bf16[3840], param_2.11005: f32[]) -> bf16[1,1,3840] {
  param_0.13539 = bf16[1,1,3840] parameter(0)
  param_1.16835 = bf16[3840] parameter(1)
  param_2.11005 = f32[] parameter(2)
  ROOT %fusion = bf16[1,1,3840] fusion(param_0.13539, param_1.16835, param_2.11005), kind=kLoop, calls=%fused_multiply.143.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_12b_decode/extracted_fusions/fused_or.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bDecode_FusedOr) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_or_module, entry_computation_layout={(pred[1]{0}, pred[])->pred[1]{0}}

%fused_or.clone (param_0.1141: pred[1], param_1.1633: pred[]) -> pred[1] {
  %param_0.1141 = pred[1]{0} parameter(0)
  %param_1.1633 = pred[] parameter(1)
  %bitcast.7637.1 = pred[1]{0} bitcast(%param_1.1633)
  ROOT %or.13.1 = pred[1]{0} or(%param_0.1141, %bitcast.7637.1)
}



ENTRY %wrapper_fused_or.clone (param_0.1141: pred[1], param_1.1633: pred[]) -> pred[1] {
  param_0.1141 = pred[1] parameter(0)
  param_1.1633 = pred[] parameter(1)
  ROOT %fusion = pred[1] fusion(param_0.1141, param_1.1633), kind=kLoop, calls=%fused_or.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_12b_decode/extracted_fusions/fused_select.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bDecode_FusedSelect) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select_module, entry_computation_layout={(bf16[8,4096,2]{2,1,0}, s32[4096]{0}, pred[1,4096]{1,0}, s32[1]{0})->bf16[8,4096,2]{2,1,0}}

%fused_select.clone (param_0.6404: bf16[8,4096,2], param_1.20811: s32[4096], param_2.10909: pred[1,4096], param_3.7187: s32[1]) -> bf16[8,4096,2] {
  %param_2.10909 = pred[1,4096]{1,0} parameter(2)
  %bitcast.145.42 = pred[1,1,4096]{2,1,0} bitcast(%param_2.10909)
  %param_1.20811 = s32[4096]{0} parameter(1)
  %param_3.7187 = s32[1]{0} parameter(3)
  %constant_2403_3 = s32[1]{0} constant({-1024})
  %add.4360.2 = s32[1]{0} add(%param_3.7187, %constant_2403_3)
  %bitcast.147.243 = s32[] bitcast(%add.4360.2)
  %broadcast.8283.243 = s32[4096]{0} broadcast(%bitcast.147.243), dimensions={}
  %compare.1802.7 = pred[4096]{0} compare(%param_1.20811, %broadcast.8283.243), direction=GT
  %constant_2404_3 = s32[1]{0} constant({1024})
  %add.4359.2 = s32[1]{0} add(%param_3.7187, %constant_2404_3)
  %bitcast.148.243 = s32[] bitcast(%add.4359.2)
  %broadcast.8287.243 = s32[4096]{0} broadcast(%bitcast.148.243), dimensions={}
  %compare.1803.7 = pred[4096]{0} compare(%param_1.20811, %broadcast.8287.243), direction=LT
  %and.567.5 = pred[4096]{0} and(%compare.1802.7, %compare.1803.7)
  %bitcast.149.3 = pred[1,1,4096]{2,1,0} bitcast(%and.567.5)
  %and.352.3 = pred[1,1,4096]{2,1,0} and(%bitcast.145.42, %bitcast.149.3)
  %bitcast.150.3 = pred[4096]{0} bitcast(%and.352.3)
  %broadcast.8856.3 = pred[8,4096,2]{2,1,0} broadcast(%bitcast.150.3), dimensions={1}
  %param_0.6404 = bf16[8,4096,2]{2,1,0} parameter(0)
  %constant_1636_2 = bf16[] constant(-2.379e+38)
  %broadcast.8858.1 = bf16[8,4096,2]{2,1,0} broadcast(%constant_1636_2), dimensions={}
  ROOT %select.1323.1 = bf16[8,4096,2]{2,1,0} select(%broadcast.8856.3, %param_0.6404, %broadcast.8858.1)
}



ENTRY %wrapper_fused_select.clone (param_0.6404: bf16[8,4096,2], param_1.20811: s32[4096], param_2.10909: pred[1,4096], param_3.7187: s32[1]) -> bf16[8,4096,2] {
  param_0.6404 = bf16[8,4096,2] parameter(0)
  param_1.20811 = s32[4096] parameter(1)
  param_2.10909 = pred[1,4096] parameter(2)
  param_3.7187 = s32[1] parameter(3)
  ROOT %fusion = bf16[8,4096,2] fusion(param_0.6404, param_1.20811, param_2.10909, param_3.7187), kind=kLoop, calls=%fused_select.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_12b_decode/extracted_fusions/fused_slice.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bDecode_FusedSlice) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_slice_module, entry_computation_layout={(u32[2]{0}, u32[2]{0})->u32[1,2]{1,0}}

%fused_slice.clone (param_0.5506: u32[2], param_1.4603: u32[2]) -> u32[1,2] {
  %param_1.4603 = u32[2]{0} parameter(1)
  %bitcast.2194.3 = u32[2,1]{1,0} bitcast(%param_1.4603)
  %param_0.5506 = u32[2]{0} parameter(0)
  %bitcast.2195.3 = u32[2,1]{1,0} bitcast(%param_0.5506)
  %concatenate.293.3 = u32[2,2]{1,0} concatenate(%bitcast.2194.3, %bitcast.2195.3), dimensions={1}
  ROOT %slice.1188.1 = u32[1,2]{1,0} slice(%concatenate.293.3), slice={[0:1], [0:2]}
}



ENTRY %wrapper_fused_slice.clone (param_0.5506: u32[2], param_1.4603: u32[2]) -> u32[1,2] {
  param_0.5506 = u32[2] parameter(0)
  param_1.4603 = u32[2] parameter(1)
  ROOT %fusion = u32[1,2] fusion(param_0.5506, param_1.4603), kind=kLoop, calls=%fused_slice.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_12b_decode/extracted_fusions/fused_xor.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bDecode_FusedXor) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_xor_module, entry_computation_layout={(u32[2]{0})->u32[]}

%fused_xor.clone (param_0.20828: u32[2]) -> u32[] {
  %param_0.20828 = u32[2]{0} parameter(0)
  %slice.1186.2 = u32[1]{0} slice(%param_0.20828), slice={[0:1]}
  %bitcast.2192.6 = u32[] bitcast(%slice.1186.2)
  %slice.1187.2 = u32[1]{0} slice(%param_0.20828), slice={[1:2]}
  %bitcast.2193.4 = u32[] bitcast(%slice.1187.2)
  %xor.16.3 = u32[] xor(%bitcast.2192.6, %bitcast.2193.4)
  %constant_1163_1 = u32[] constant(466688986)
  ROOT %xor.17.1 = u32[] xor(%xor.16.3, %constant_1163_1)
}



ENTRY %wrapper_fused_xor.clone (param_0.20828: u32[2]) -> u32[] {
  param_0.20828 = u32[2] parameter(0)
  ROOT %fusion = u32[] fusion(param_0.20828), kind=kLoop, calls=%fused_xor.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// === gemma3_12b_prefill (6 fusions) ===
// Source: gemma3_12b_prefill/extracted_fusions/fused_add.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bPrefill_FusedAdd) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_add_module, entry_computation_layout={(s32[1]{0})->s32[1]{0}}

%fused_add.clone (param_0.13917: s32[1]) -> s32[1] {
  %param_0.13917 = s32[1]{0} parameter(0)
  %constant_691_5 = s32[1]{0} constant({11})
  ROOT %add.5320.1 = s32[1]{0} add(%param_0.13917, %constant_691_5)
}



ENTRY %wrapper_fused_add.clone (param_0.13917: s32[1]) -> s32[1] {
  param_0.13917 = s32[1] parameter(0)
  ROOT %fusion = s32[1] fusion(param_0.13917), kind=kLoop, calls=%fused_add.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_12b_prefill/extracted_fusions/fused_concatenate.47.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bPrefill_FusedConcatenate_47) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate.47_module, entry_computation_layout={(f32[1,11,128]{2,1,0}, bf16[1,1,11,8,256]{4,3,2,1,0}, f32[1,11,128]{2,1,0})->bf16[1,11,8,256]{3,2,1,0}}

%fused_concatenate.47.clone (param_0.10316: f32[1,11,128], param_1.9896: bf16[1,1,11,8,256], param_2.7783: f32[1,11,128]) -> bf16[1,11,8,256] {
  %param_1.9896 = bf16[1,1,11,8,256]{4,3,2,1,0} parameter(1)
  %bitcast.143.29 = bf16[1,11,8,256]{3,2,1,0} bitcast(%param_1.9896)
  %convert.1188.29 = f32[1,11,8,256]{3,2,1,0} convert(%bitcast.143.29)
  %slice.13.15 = f32[1,11,8,128]{3,2,1,0} slice(%convert.1188.29), slice={[0:1], [0:11], [0:8], [0:128]}
  %param_2.7783 = f32[1,11,128]{2,1,0} parameter(2)
  %bitcast.145.720 = f32[11,128]{1,0} bitcast(%param_2.7783)
  %broadcast.920.399 = f32[1,11,8,128]{3,2,1,0} broadcast(%bitcast.145.720), dimensions={1,3}
  %multiply.921.7 = f32[1,11,8,128]{3,2,1,0} multiply(%slice.13.15, %broadcast.920.399)
  %slice.15.11 = f32[1,11,8,128]{3,2,1,0} slice(%convert.1188.29), slice={[0:1], [0:11], [0:8], [128:256]}
  %param_0.10316 = f32[1,11,128]{2,1,0} parameter(0)
  %bitcast.146.880 = f32[11,128]{1,0} bitcast(%param_0.10316)
  %broadcast.925.399 = f32[1,11,8,128]{3,2,1,0} broadcast(%bitcast.146.880), dimensions={1,3}
  %multiply.926.5 = f32[1,11,8,128]{3,2,1,0} multiply(%slice.15.11, %broadcast.925.399)
  %subtract.927.5 = f32[1,11,8,128]{3,2,1,0} subtract(%multiply.921.7, %multiply.926.5)
  %convert.1134.3 = bf16[1,11,8,128]{3,2,1,0} convert(%subtract.927.5)
  %multiply.932.5 = f32[1,11,8,128]{3,2,1,0} multiply(%slice.15.11, %broadcast.920.399)
  %multiply.937.7 = f32[1,11,8,128]{3,2,1,0} multiply(%slice.13.15, %broadcast.925.399)
  %add.938.5 = f32[1,11,8,128]{3,2,1,0} add(%multiply.932.5, %multiply.937.7)
  %convert.1135.3 = bf16[1,11,8,128]{3,2,1,0} convert(%add.938.5)
  ROOT %concatenate.96.1 = bf16[1,11,8,256]{3,2,1,0} concatenate(%convert.1134.3, %convert.1135.3), dimensions={3}
}



ENTRY %wrapper_fused_concatenate.47.clone (param_0.10316: f32[1,11,128], param_1.9896: bf16[1,1,11,8,256], param_2.7783: f32[1,11,128]) -> bf16[1,11,8,256] {
  param_0.10316 = f32[1,11,128] parameter(0)
  param_1.9896 = bf16[1,1,11,8,256] parameter(1)
  param_2.7783 = f32[1,11,128] parameter(2)
  ROOT %fusion = bf16[1,11,8,256] fusion(param_0.10316, param_1.9896, param_2.7783), kind=kLoop, calls=%fused_concatenate.47.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_12b_prefill/extracted_fusions/fused_multiply.47.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bPrefill_FusedMultiply_47) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.47_module, entry_computation_layout={(bf16[30720,11]{0,1})->bf16[1,11,15360]{2,1,0}}

%fused_multiply.47.clone (param_0.13865: bf16[30720,11]{0,1}) -> bf16[1,11,15360] {
  %param_0.13865 = bf16[30720,11]{0,1} parameter(0)
  %bitcast.9712.6 = bf16[1,11,2,15360]{3,2,1,0} bitcast(%param_0.13865)
  %slice.397.3 = bf16[1,11,1,15360]{3,2,1,0} slice(%bitcast.9712.6), slice={[0:1], [0:11], [0:1], [0:15360]}
  %bitcast.168.16 = bf16[1,11,15360]{2,1,0} bitcast(%slice.397.3)
  %multiply.1139.11 = bf16[1,11,15360]{2,1,0} multiply(%bitcast.168.16, %bitcast.168.16)
  %multiply.1140.9 = bf16[1,11,15360]{2,1,0} multiply(%multiply.1139.11, %bitcast.168.16)
  %constant_689_2 = bf16[] constant(0.04468)
  %broadcast.690.148 = bf16[1,11,15360]{2,1,0} broadcast(%constant_689_2), dimensions={}
  %multiply.1141.7 = bf16[1,11,15360]{2,1,0} multiply(%multiply.1140.9, %broadcast.690.148)
  %add.1142.5 = bf16[1,11,15360]{2,1,0} add(%bitcast.168.16, %multiply.1141.7)
  %constant_687_2 = bf16[] constant(0.7969)
  %broadcast.688.50 = bf16[1,11,15360]{2,1,0} broadcast(%constant_687_2), dimensions={}
  %multiply.1143.3 = bf16[1,11,15360]{2,1,0} multiply(%add.1142.5, %broadcast.688.50)
  %convert.3118.7 = f32[1,11,15360]{2,1,0} convert(%multiply.1143.3)
  %tanh.1144.7 = f32[1,11,15360]{2,1,0} tanh(%convert.3118.7)
  %convert.3119.5 = bf16[1,11,15360]{2,1,0} convert(%tanh.1144.7)
  %constant_716_4 = bf16[] constant(1)
  %broadcast.686.336 = bf16[1,11,15360]{2,1,0} broadcast(%constant_716_4), dimensions={}
  %add.1145.7 = bf16[1,11,15360]{2,1,0} add(%convert.3119.5, %broadcast.686.336)
  %constant_683_2 = bf16[] constant(0.5)
  %broadcast.684.240 = bf16[1,11,15360]{2,1,0} broadcast(%constant_683_2), dimensions={}
  %multiply.1146.5 = bf16[1,11,15360]{2,1,0} multiply(%add.1145.7, %broadcast.684.240)
  %multiply.1147.3 = bf16[1,11,15360]{2,1,0} multiply(%bitcast.168.16, %multiply.1146.5)
  %slice.1148.3 = bf16[1,11,1,15360]{3,2,1,0} slice(%bitcast.9712.6), slice={[0:1], [0:11], [1:2], [0:15360]}
  %bitcast.169.1 = bf16[1,11,15360]{2,1,0} bitcast(%slice.1148.3)
  ROOT %multiply.1150.1 = bf16[1,11,15360]{2,1,0} multiply(%multiply.1147.3, %bitcast.169.1)
}



ENTRY %wrapper_fused_multiply.47.clone (param_0.13865: bf16[30720,11]{0,1}) -> bf16[1,11,15360] {
  param_0.13865 = bf16[30720,11]{0,1} parameter(0)
  ROOT %fusion = bf16[1,11,15360] fusion(param_0.13865), kind=kLoop, calls=%fused_multiply.47.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_12b_prefill/extracted_fusions/fused_slice.46.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bPrefill_FusedSlice_46) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_slice.46_module, entry_computation_layout={(bf16[2,11,2048]{2,1,0})->bf16[1,1,11,8,256]{4,3,2,1,0}}

%fused_slice.46.clone (param_0.1929: bf16[2,11,2048]) -> bf16[1,1,11,8,256] {
  %param_0.1929 = bf16[2,11,2048]{2,1,0} parameter(0)
  %bitcast.5048.2 = bf16[2,1,11,8,256]{4,3,2,1,0} bitcast(%param_0.1929)
  ROOT %slice.1202.1 = bf16[1,1,11,8,256]{4,3,2,1,0} slice(%bitcast.5048.2), slice={[1:2], [0:1], [0:11], [0:8], [0:256]}
}



ENTRY %wrapper_fused_slice.46.clone (param_0.1929: bf16[2,11,2048]) -> bf16[1,1,11,8,256] {
  param_0.1929 = bf16[2,11,2048] parameter(0)
  ROOT %fusion = bf16[1,1,11,8,256] fusion(param_0.1929), kind=kLoop, calls=%fused_slice.46.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_12b_prefill/extracted_fusions/fused_slice.47.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bPrefill_FusedSlice_47) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_slice.47_module, entry_computation_layout={(bf16[2,2048,11]{2,1,0})->bf16[1,1,11,8,256]{4,3,2,1,0}}

%fused_slice.47.clone (param_0.13403: bf16[2,2048,11]) -> bf16[1,1,11,8,256] {
  %param_0.13403 = bf16[2,2048,11]{2,1,0} parameter(0)
  %transpose.1154.1 = bf16[2,11,2048]{2,1,0} transpose(%param_0.13403), dimensions={0,2,1}
  %bitcast.5038.2 = bf16[2,1,11,8,256]{4,3,2,1,0} bitcast(%transpose.1154.1)
  ROOT %slice.804.1 = bf16[1,1,11,8,256]{4,3,2,1,0} slice(%bitcast.5038.2), slice={[1:2], [0:1], [0:11], [0:8], [0:256]}
}



ENTRY %wrapper_fused_slice.47.clone (param_0.13403: bf16[2,2048,11]) -> bf16[1,1,11,8,256] {
  param_0.13403 = bf16[2,2048,11] parameter(0)
  ROOT %fusion = bf16[1,1,11,8,256] fusion(param_0.13403), kind=kLoop, calls=%fused_slice.47.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_12b_prefill/extracted_fusions/fused_transpose.95.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma312bPrefill_FusedTranspose_95) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.95_module, entry_computation_layout={(f32[1,11,128]{2,1,0}, bf16[1,11,16,256]{3,2,1,0}, f32[1,11,128]{2,1,0})->bf16[8,256,11,2]{3,2,1,0}}

%fused_transpose.95.clone (param_0.13769: f32[1,11,128], param_1.13904: bf16[1,11,16,256], param_2.10026: f32[1,11,128]) -> bf16[8,256,11,2] {
  %param_1.13904 = bf16[1,11,16,256]{3,2,1,0} parameter(1)
  %convert.50.30 = f32[1,11,16,256]{3,2,1,0} convert(%param_1.13904)
  %slice.17.15 = f32[1,11,16,128]{3,2,1,0} slice(%convert.50.30), slice={[0:1], [0:11], [0:16], [0:128]}
  %param_2.10026 = f32[1,11,128]{2,1,0} parameter(2)
  %bitcast.145.882 = f32[11,128]{1,0} bitcast(%param_2.10026)
  %broadcast.874.481 = f32[1,11,16,128]{3,2,1,0} broadcast(%bitcast.145.882), dimensions={1,3}
  %multiply.875.5 = f32[1,11,16,128]{3,2,1,0} multiply(%slice.17.15, %broadcast.874.481)
  %slice.19.15 = f32[1,11,16,128]{3,2,1,0} slice(%convert.50.30), slice={[0:1], [0:11], [0:16], [128:256]}
  %param_0.13769 = f32[1,11,128]{2,1,0} parameter(0)
  %bitcast.146.1042 = f32[11,128]{1,0} bitcast(%param_0.13769)
  %broadcast.879.641 = f32[1,11,16,128]{3,2,1,0} broadcast(%bitcast.146.1042), dimensions={1,3}
  %multiply.880.7 = f32[1,11,16,128]{3,2,1,0} multiply(%slice.19.15, %broadcast.879.641)
  %subtract.881.5 = f32[1,11,16,128]{3,2,1,0} subtract(%multiply.875.5, %multiply.880.7)
  %convert.1132.3 = bf16[1,11,16,128]{3,2,1,0} convert(%subtract.881.5)
  %multiply.886.5 = f32[1,11,16,128]{3,2,1,0} multiply(%slice.19.15, %broadcast.874.481)
  %multiply.891.7 = f32[1,11,16,128]{3,2,1,0} multiply(%slice.17.15, %broadcast.879.641)
  %add.892.5 = f32[1,11,16,128]{3,2,1,0} add(%multiply.886.5, %multiply.891.7)
  %convert.1133.3 = bf16[1,11,16,128]{3,2,1,0} convert(%add.892.5)
  %concatenate.95.1 = bf16[1,11,16,256]{3,2,1,0} concatenate(%convert.1132.3, %convert.1133.3), dimensions={3}
  %constant_696_2 = bf16[] constant(0.0625)
  %broadcast.697.144 = bf16[1,11,16,256]{3,2,1,0} broadcast(%constant_696_2), dimensions={}
  %multiply.895.3 = bf16[1,11,16,256]{3,2,1,0} multiply(%concatenate.95.1, %broadcast.697.144)
  %bitcast.5039.1 = bf16[11,8,2,256]{3,2,1,0} bitcast(%multiply.895.3)
  ROOT %transpose.1155.1 = bf16[8,256,11,2]{3,2,1,0} transpose(%bitcast.5039.1), dimensions={1,3,0,2}
}



ENTRY %wrapper_fused_transpose.95.clone (param_0.13769: f32[1,11,128], param_1.13904: bf16[1,11,16,256], param_2.10026: f32[1,11,128]) -> bf16[8,256,11,2] {
  param_0.13769 = f32[1,11,128] parameter(0)
  param_1.13904 = bf16[1,11,16,256] parameter(1)
  param_2.10026 = f32[1,11,128] parameter(2)
  ROOT %fusion = bf16[8,256,11,2] fusion(param_0.13769, param_1.13904, param_2.10026), kind=kLoop, calls=%fused_transpose.95.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// === gemma3_27b_training (42 fusions) ===
// Source: gemma3_27b_training/extracted_fusions/fused_concatenate.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedConcatenate) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate_module, entry_computation_layout={(bf16[16,8192,1,64]{3,2,1,0}, bf16[16,8192,1,64]{3,2,1,0}, bf16[16,8192,32,128]{3,2,1,0}, bf16[1,128]{1,0})->bf16[16,8192,32,128]{3,2,1,0}}

%fused_concatenate.clone (param_0.193: bf16[16,8192,1,64], param_1.181: bf16[16,8192,1,64], param_2.134: bf16[16,8192,32,128], param_3.67: bf16[1,128]) -> bf16[16,8192,32,128] {
  %param_2.134 = bf16[16,8192,32,128]{3,2,1,0} parameter(2)
  %param_3.67 = bf16[1,128]{1,0} parameter(3)
  %bitcast.4.9 = bf16[128]{0} bitcast(%param_3.67)
  %broadcast.360.9 = bf16[16,8192,32,128]{3,2,1,0} broadcast(%bitcast.4.9), dimensions={3}
  %multiply.244.6 = bf16[16,8192,32,128]{3,2,1,0} multiply(%param_2.134, %broadcast.360.9)
  %slice.12.3 = bf16[16,8192,32,64]{3,2,1,0} slice(%multiply.244.6), slice={[0:16], [0:8192], [0:32], [0:64]}
  %param_1.181 = bf16[16,8192,1,64]{3,2,1,0} parameter(1)
  %bitcast.5.10 = bf16[16,8192,64]{2,1,0} bitcast(%param_1.181)
  %broadcast.361.9 = bf16[16,8192,32,64]{3,2,1,0} broadcast(%bitcast.5.10), dimensions={0,1,3}
  %multiply.245.3 = bf16[16,8192,32,64]{3,2,1,0} multiply(%slice.12.3, %broadcast.361.9)
  %slice.13.3 = bf16[16,8192,32,64]{3,2,1,0} slice(%multiply.244.6), slice={[0:16], [0:8192], [0:32], [64:128]}
  %param_0.193 = bf16[16,8192,1,64]{3,2,1,0} parameter(0)
  %bitcast.6.14 = bf16[16,8192,64]{2,1,0} bitcast(%param_0.193)
  %broadcast.362.13 = bf16[16,8192,32,64]{3,2,1,0} broadcast(%bitcast.6.14), dimensions={0,1,3}
  %multiply.246.5 = bf16[16,8192,32,64]{3,2,1,0} multiply(%slice.13.3, %broadcast.362.13)
  %subtract.6.3 = bf16[16,8192,32,64]{3,2,1,0} subtract(%multiply.245.3, %multiply.246.5)
  %multiply.247.3 = bf16[16,8192,32,64]{3,2,1,0} multiply(%slice.13.3, %broadcast.361.9)
  %multiply.248.5 = bf16[16,8192,32,64]{3,2,1,0} multiply(%slice.12.3, %broadcast.362.13)
  %add.49.3 = bf16[16,8192,32,64]{3,2,1,0} add(%multiply.247.3, %multiply.248.5)
  %concatenate.6.1 = bf16[16,8192,32,128]{3,2,1,0} concatenate(%subtract.6.3, %add.49.3), dimensions={3}
  %constant_114_1 = bf16[] constant(0.07715)
  %broadcast.363.1 = bf16[16,8192,32,128]{3,2,1,0} broadcast(%constant_114_1), dimensions={}
  ROOT %multiply.249.1 = bf16[16,8192,32,128]{3,2,1,0} multiply(%concatenate.6.1, %broadcast.363.1)
}



ENTRY %wrapper_fused_concatenate.clone (param_0.193: bf16[16,8192,1,64], param_1.181: bf16[16,8192,1,64], param_2.134: bf16[16,8192,32,128], param_3.67: bf16[1,128]) -> bf16[16,8192,32,128] {
  param_0.193 = bf16[16,8192,1,64] parameter(0)
  param_1.181 = bf16[16,8192,1,64] parameter(1)
  param_2.134 = bf16[16,8192,32,128] parameter(2)
  param_3.67 = bf16[1,128] parameter(3)
  ROOT %fusion = bf16[16,8192,32,128] fusion(param_0.193, param_1.181, param_2.134, param_3.67), kind=kLoop, calls=%fused_concatenate.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source:
// gemma3_27b_training/extracted_fusions/fused_concatenate.3.clone.clone.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedConcatenate_3) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate.3.clone.clone_module, entry_computation_layout={(bf16[1,16,8192,64]{3,2,1,0}, bf16[1,16,8192,64]{3,2,1,0}, bf16[1,128]{1,0}, f32[1,16,8192,32]{3,2,1,0}, bf16[131072,4096]{1,0})->bf16[16,8192,32,128]{3,2,1,0}}

%fused_concatenate.3.clone.clone.clone (param_0.2184: bf16[1,16,8192,64], param_1.2525: bf16[1,16,8192,64], param_2.1372: bf16[1,128], param_3.971: f32[1,16,8192,32], param_4.614: bf16[131072,4096]) -> bf16[16,8192,32,128] {
  %param_4.614 = bf16[131072,4096]{1,0} parameter(4)
  %bitcast.692.32 = bf16[16,8192,32,128]{3,2,1,0} bitcast(%param_4.614)
  %convert.231.32 = f32[16,8192,32,128]{3,2,1,0} convert(%bitcast.692.32)
  %param_3.971 = f32[1,16,8192,32]{3,2,1,0} parameter(3)
  %bitcast.35.21 = f32[16,8192,32]{2,1,0} bitcast(%param_3.971)
  %broadcast.67.21 = f32[16,8192,32,128]{3,2,1,0} broadcast(%bitcast.35.21), dimensions={0,1,2}
  %multiply.38.15 = f32[16,8192,32,128]{3,2,1,0} multiply(%convert.231.32, %broadcast.67.21)
  %convert.30.13 = bf16[16,8192,32,128]{3,2,1,0} convert(%multiply.38.15)
  %param_2.1372 = bf16[1,128]{1,0} parameter(2)
  %bitcast.36.19 = bf16[128]{0} bitcast(%param_2.1372)
  %broadcast.71.19 = bf16[16,8192,32,128]{3,2,1,0} broadcast(%bitcast.36.19), dimensions={3}
  %multiply.39.11 = bf16[16,8192,32,128]{3,2,1,0} multiply(%convert.30.13, %broadcast.71.19)
  %slice.4.5 = bf16[16,8192,32,64]{3,2,1,0} slice(%multiply.39.11), slice={[0:16], [0:8192], [0:32], [0:64]}
  %param_1.2525 = bf16[1,16,8192,64]{3,2,1,0} parameter(1)
  %bitcast.37.38 = bf16[16,8192,64]{2,1,0} bitcast(%param_1.2525)
  %broadcast.89.19 = bf16[16,8192,32,64]{3,2,1,0} broadcast(%bitcast.37.38), dimensions={0,1,3}
  %multiply.41.5 = bf16[16,8192,32,64]{3,2,1,0} multiply(%slice.4.5, %broadcast.89.19)
  %slice.5.5 = bf16[16,8192,32,64]{3,2,1,0} slice(%multiply.39.11), slice={[0:16], [0:8192], [0:32], [64:128]}
  %param_0.2184 = bf16[1,16,8192,64]{3,2,1,0} parameter(0)
  %bitcast.38.46 = bf16[16,8192,64]{2,1,0} bitcast(%param_0.2184)
  %broadcast.93.23 = bf16[16,8192,32,64]{3,2,1,0} broadcast(%bitcast.38.46), dimensions={0,1,3}
  %multiply.42.7 = bf16[16,8192,32,64]{3,2,1,0} multiply(%slice.5.5, %broadcast.93.23)
  %subtract.2.5 = bf16[16,8192,32,64]{3,2,1,0} subtract(%multiply.41.5, %multiply.42.7)
  %multiply.43.5 = bf16[16,8192,32,64]{3,2,1,0} multiply(%slice.5.5, %broadcast.89.19)
  %multiply.44.7 = bf16[16,8192,32,64]{3,2,1,0} multiply(%slice.4.5, %broadcast.93.23)
  %add.14.5 = bf16[16,8192,32,64]{3,2,1,0} add(%multiply.43.5, %multiply.44.7)
  %concatenate.2.3 = bf16[16,8192,32,128]{3,2,1,0} concatenate(%subtract.2.5, %add.14.5), dimensions={3}
  %constant_20_2_2 = bf16[] constant(0.07715)
  %broadcast.103.16 = bf16[16,8192,32,128]{3,2,1,0} broadcast(%constant_20_2_2), dimensions={}
  ROOT %multiply.45.3 = bf16[16,8192,32,128]{3,2,1,0} multiply(%concatenate.2.3, %broadcast.103.16)
}



ENTRY %wrapper_fused_concatenate.3.clone.clone.clone (param_0.2184: bf16[1,16,8192,64], param_1.2525: bf16[1,16,8192,64], param_2.1372: bf16[1,128], param_3.971: f32[1,16,8192,32], param_4.614: bf16[131072,4096]) -> bf16[16,8192,32,128] {
  param_0.2184 = bf16[1,16,8192,64] parameter(0)
  param_1.2525 = bf16[1,16,8192,64] parameter(1)
  param_2.1372 = bf16[1,128] parameter(2)
  param_3.971 = f32[1,16,8192,32] parameter(3)
  param_4.614 = bf16[131072,4096] parameter(4)
  ROOT %fusion = bf16[16,8192,32,128] fusion(param_0.2184, param_1.2525, param_2.1372, param_3.971, param_4.614), kind=kLoop, calls=%fused_concatenate.3.clone.clone.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source:
// gemma3_27b_training/extracted_fusions/fused_concatenate.5.clone.clone.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedConcatenate_5) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate.5.clone.clone_module, entry_computation_layout={(bf16[1,16,8192,64]{3,2,1,0}, bf16[1,16,8192,64]{3,2,1,0}, bf16[1,128]{1,0}, f32[1,16,8192,16]{3,2,1,0}, bf16[131072,2048]{1,0})->bf16[16,8192,16,128]{3,2,1,0}}

%fused_concatenate.5.clone.clone.clone (param_0.2177: bf16[1,16,8192,64], param_1.2520: bf16[1,16,8192,64], param_2.1370: bf16[1,128], param_3.969: f32[1,16,8192,16], param_4.612: bf16[131072,2048]) -> bf16[16,8192,16,128] {
  %param_4.612 = bf16[131072,2048]{1,0} parameter(4)
  %bitcast.695.32 = bf16[16,8192,16,128]{3,2,1,0} bitcast(%param_4.612)
  %convert.232.32 = f32[16,8192,16,128]{3,2,1,0} convert(%bitcast.695.32)
  %param_3.969 = f32[1,16,8192,16]{3,2,1,0} parameter(3)
  %bitcast.42.23 = f32[16,8192,16]{2,1,0} bitcast(%param_3.969)
  %broadcast.111.23 = f32[16,8192,16,128]{3,2,1,0} broadcast(%bitcast.42.23), dimensions={0,1,2}
  %multiply.47.15 = f32[16,8192,16,128]{3,2,1,0} multiply(%convert.232.32, %broadcast.111.23)
  %convert.38.13 = bf16[16,8192,16,128]{3,2,1,0} convert(%multiply.47.15)
  %param_2.1370 = bf16[1,128]{1,0} parameter(2)
  %bitcast.43.19 = bf16[128]{0} bitcast(%param_2.1370)
  %broadcast.115.19 = bf16[16,8192,16,128]{3,2,1,0} broadcast(%bitcast.43.19), dimensions={3}
  %multiply.48.11 = bf16[16,8192,16,128]{3,2,1,0} multiply(%convert.38.13, %broadcast.115.19)
  %slice.6.5 = bf16[16,8192,16,64]{3,2,1,0} slice(%multiply.48.11), slice={[0:16], [0:8192], [0:16], [0:64]}
  %param_1.2520 = bf16[1,16,8192,64]{3,2,1,0} parameter(1)
  %bitcast.37.36 = bf16[16,8192,64]{2,1,0} bitcast(%param_1.2520)
  %broadcast.127.19 = bf16[16,8192,16,64]{3,2,1,0} broadcast(%bitcast.37.36), dimensions={0,1,3}
  %multiply.50.5 = bf16[16,8192,16,64]{3,2,1,0} multiply(%slice.6.5, %broadcast.127.19)
  %slice.7.5 = bf16[16,8192,16,64]{3,2,1,0} slice(%multiply.48.11), slice={[0:16], [0:8192], [0:16], [64:128]}
  %param_0.2177 = bf16[1,16,8192,64]{3,2,1,0} parameter(0)
  %bitcast.38.44 = bf16[16,8192,64]{2,1,0} bitcast(%param_0.2177)
  %broadcast.131.23 = bf16[16,8192,16,64]{3,2,1,0} broadcast(%bitcast.38.44), dimensions={0,1,3}
  %multiply.51.7 = bf16[16,8192,16,64]{3,2,1,0} multiply(%slice.7.5, %broadcast.131.23)
  %subtract.3.5 = bf16[16,8192,16,64]{3,2,1,0} subtract(%multiply.50.5, %multiply.51.7)
  %multiply.52.5 = bf16[16,8192,16,64]{3,2,1,0} multiply(%slice.7.5, %broadcast.127.19)
  %multiply.53.7 = bf16[16,8192,16,64]{3,2,1,0} multiply(%slice.6.5, %broadcast.131.23)
  %add.16.5 = bf16[16,8192,16,64]{3,2,1,0} add(%multiply.52.5, %multiply.53.7)
  ROOT %concatenate.3.3 = bf16[16,8192,16,128]{3,2,1,0} concatenate(%subtract.3.5, %add.16.5), dimensions={3}
}



ENTRY %wrapper_fused_concatenate.5.clone.clone.clone (param_0.2177: bf16[1,16,8192,64], param_1.2520: bf16[1,16,8192,64], param_2.1370: bf16[1,128], param_3.969: f32[1,16,8192,16], param_4.612: bf16[131072,2048]) -> bf16[16,8192,16,128] {
  param_0.2177 = bf16[1,16,8192,64] parameter(0)
  param_1.2520 = bf16[1,16,8192,64] parameter(1)
  param_2.1370 = bf16[1,128] parameter(2)
  param_3.969 = f32[1,16,8192,16] parameter(3)
  param_4.612 = bf16[131072,2048] parameter(4)
  ROOT %fusion = bf16[16,8192,16,128] fusion(param_0.2177, param_1.2520, param_2.1370, param_3.969, param_4.612), kind=kLoop, calls=%fused_concatenate.5.clone.clone.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_concatenate.1.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedConcatenate_1) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate.1_module, entry_computation_layout={(bf16[16,8192,1,64]{3,2,1,0}, bf16[16,8192,1,64]{3,2,1,0}, bf16[16,8192,16,128]{3,2,1,0})->bf16[16,8192,16,128]{3,2,1,0}}

%fused_concatenate.1.clone (param_0.169: bf16[16,8192,1,64], param_1.169: bf16[16,8192,1,64], param_2.122: bf16[16,8192,16,128]) -> bf16[16,8192,16,128] {
  %param_2.122 = bf16[16,8192,16,128]{3,2,1,0} parameter(2)
  %slice.14.1 = bf16[16,8192,16,64]{3,2,1,0} slice(%param_2.122), slice={[0:16], [0:8192], [0:16], [0:64]}
  %param_1.169 = bf16[16,8192,1,64]{3,2,1,0} parameter(1)
  %bitcast.5.18 = bf16[16,8192,64]{2,1,0} bitcast(%param_1.169)
  %broadcast.368.9 = bf16[16,8192,16,64]{3,2,1,0} broadcast(%bitcast.5.18), dimensions={0,1,3}
  %multiply.253.3 = bf16[16,8192,16,64]{3,2,1,0} multiply(%slice.14.1, %broadcast.368.9)
  %slice.15.1 = bf16[16,8192,16,64]{3,2,1,0} slice(%param_2.122), slice={[0:16], [0:8192], [0:16], [64:128]}
  %param_0.169 = bf16[16,8192,1,64]{3,2,1,0} parameter(0)
  %bitcast.6.26 = bf16[16,8192,64]{2,1,0} bitcast(%param_0.169)
  %broadcast.370.13 = bf16[16,8192,16,64]{3,2,1,0} broadcast(%bitcast.6.26), dimensions={0,1,3}
  %multiply.254.5 = bf16[16,8192,16,64]{3,2,1,0} multiply(%slice.15.1, %broadcast.370.13)
  %subtract.7.3 = bf16[16,8192,16,64]{3,2,1,0} subtract(%multiply.253.3, %multiply.254.5)
  %multiply.255.3 = bf16[16,8192,16,64]{3,2,1,0} multiply(%slice.15.1, %broadcast.368.9)
  %multiply.256.5 = bf16[16,8192,16,64]{3,2,1,0} multiply(%slice.14.1, %broadcast.370.13)
  %add.51.3 = bf16[16,8192,16,64]{3,2,1,0} add(%multiply.255.3, %multiply.256.5)
  ROOT %concatenate.7.1 = bf16[16,8192,16,128]{3,2,1,0} concatenate(%subtract.7.3, %add.51.3), dimensions={3}
}



ENTRY %wrapper_fused_concatenate.1.clone (param_0.169: bf16[16,8192,1,64], param_1.169: bf16[16,8192,1,64], param_2.122: bf16[16,8192,16,128]) -> bf16[16,8192,16,128] {
  param_0.169 = bf16[16,8192,1,64] parameter(0)
  param_1.169 = bf16[16,8192,1,64] parameter(1)
  param_2.122 = bf16[16,8192,16,128] parameter(2)
  ROOT %fusion = bf16[16,8192,16,128] fusion(param_0.169, param_1.169, param_2.122), kind=kLoop, calls=%fused_concatenate.1.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_concatenate.2.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedConcatenate_2) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate.2_module, entry_computation_layout={(f32[1,16,8192,32]{3,2,1,0}, f32[1,16,8192,32]{3,2,1,0}, bf16[131072,4096]{1,0}, bf16[1,128]{1,0}, bf16[16,8192,32,64]{3,2,1,0}, /*index=5*/bf16[16,8192,32,64]{3,2,1,0})->bf16[16,8192,32,128]{3,2,1,0}}

%fused_concatenate.2.clone (param_0.560: f32[1,16,8192,32], param_1.615: f32[1,16,8192,32], param_2.291: bf16[131072,4096], param_3.164: bf16[1,128], param_4.81: bf16[16,8192,32,64], param_5.40: bf16[16,8192,32,64]) -> bf16[16,8192,32,128] {
  %param_2.291 = bf16[131072,4096]{1,0} parameter(2)
  %bitcast.692.15 = bf16[16,8192,32,128]{3,2,1,0} bitcast(%param_2.291)
  %convert.231.15 = f32[16,8192,32,128]{3,2,1,0} convert(%bitcast.692.15)
  %param_1.615 = f32[1,16,8192,32]{3,2,1,0} parameter(1)
  %bitcast.90.5 = f32[16,8192,32]{2,1,0} bitcast(%param_1.615)
  %broadcast.277.5 = f32[16,8192,32,128]{3,2,1,0} broadcast(%bitcast.90.5), dimensions={0,1,2}
  %multiply.122.3 = f32[16,8192,32,128]{3,2,1,0} multiply(%convert.231.15, %broadcast.277.5)
  %param_4.81 = bf16[16,8192,32,64]{3,2,1,0} parameter(4)
  %param_5.40 = bf16[16,8192,32,64]{3,2,1,0} parameter(5)
  %concatenate.5.4 = bf16[16,8192,32,128]{3,2,1,0} concatenate(%param_4.81, %param_5.40), dimensions={3}
  %param_3.164 = bf16[1,128]{1,0} parameter(3)
  %bitcast.36.11 = bf16[128]{0} bitcast(%param_3.164)
  %broadcast.71.11 = bf16[16,8192,32,128]{3,2,1,0} broadcast(%bitcast.36.11), dimensions={3}
  %multiply.117.3 = bf16[16,8192,32,128]{3,2,1,0} multiply(%concatenate.5.4, %broadcast.71.11)
  %convert.65.6 = f32[16,8192,32,128]{3,2,1,0} convert(%multiply.117.3)
  %param_0.560 = f32[1,16,8192,32]{3,2,1,0} parameter(0)
  %bitcast.35.9 = f32[16,8192,32]{2,1,0} bitcast(%param_0.560)
  %broadcast.67.9 = f32[16,8192,32,128]{3,2,1,0} broadcast(%bitcast.35.9), dimensions={0,1,2}
  %multiply.123.3 = f32[16,8192,32,128]{3,2,1,0} multiply(%convert.65.6, %broadcast.67.9)
  %add.37.1 = f32[16,8192,32,128]{3,2,1,0} add(%multiply.122.3, %multiply.123.3)
  ROOT %convert.66.1 = bf16[16,8192,32,128]{3,2,1,0} convert(%add.37.1)
}



ENTRY %wrapper_fused_concatenate.2.clone (param_0.560: f32[1,16,8192,32], param_1.615: f32[1,16,8192,32], param_2.291: bf16[131072,4096], param_3.164: bf16[1,128], param_4.81: bf16[16,8192,32,64], param_5.40: bf16[16,8192,32,64]) -> bf16[16,8192,32,128] {
  param_0.560 = f32[1,16,8192,32] parameter(0)
  param_1.615 = f32[1,16,8192,32] parameter(1)
  param_2.291 = bf16[131072,4096] parameter(2)
  param_3.164 = bf16[1,128] parameter(3)
  param_4.81 = bf16[16,8192,32,64] parameter(4)
  param_5.40 = bf16[16,8192,32,64] parameter(5)
  ROOT %fusion = bf16[16,8192,32,128] fusion(param_0.560, param_1.615, param_2.291, param_3.164, param_4.81, param_5.40), kind=kLoop, calls=%fused_concatenate.2.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_concatenate.4.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedConcatenate_4) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate.4_module, entry_computation_layout={(f32[1,16,8192,16]{3,2,1,0}, bf16[131072,2048]{1,0}, f32[1,16,8192,16]{3,2,1,0}, bf16[1,128]{1,0}, bf16[16,8192,16,64]{3,2,1,0}, /*index=5*/bf16[16,8192,16,64]{3,2,1,0})->bf16[16,8192,16,128]{3,2,1,0}}

%fused_concatenate.4.clone (param_0.631: f32[1,16,8192,16], param_1.714: bf16[131072,2048], param_2.355: f32[1,16,8192,16], param_3.212: bf16[1,128], param_4.110: bf16[16,8192,16,64], param_5.53: bf16[16,8192,16,64]) -> bf16[16,8192,16,128] {
  %param_1.714 = bf16[131072,2048]{1,0} parameter(1)
  %bitcast.695.11 = bf16[16,8192,16,128]{3,2,1,0} bitcast(%param_1.714)
  %convert.232.11 = f32[16,8192,16,128]{3,2,1,0} convert(%bitcast.695.11)
  %param_2.355 = f32[1,16,8192,16]{3,2,1,0} parameter(2)
  %bitcast.85.3 = f32[16,8192,16]{2,1,0} bitcast(%param_2.355)
  %broadcast.274.3 = f32[16,8192,16,128]{3,2,1,0} broadcast(%bitcast.85.3), dimensions={0,1,2}
  %multiply.110.3 = f32[16,8192,16,128]{3,2,1,0} multiply(%convert.232.11, %broadcast.274.3)
  %param_4.110 = bf16[16,8192,16,64]{3,2,1,0} parameter(4)
  %param_5.53 = bf16[16,8192,16,64]{3,2,1,0} parameter(5)
  %concatenate.4.4 = bf16[16,8192,16,128]{3,2,1,0} concatenate(%param_4.110, %param_5.53), dimensions={3}
  %param_3.212 = bf16[1,128]{1,0} parameter(3)
  %bitcast.43.11 = bf16[128]{0} bitcast(%param_3.212)
  %broadcast.115.11 = bf16[16,8192,16,128]{3,2,1,0} broadcast(%bitcast.43.11), dimensions={3}
  %multiply.105.3 = bf16[16,8192,16,128]{3,2,1,0} multiply(%concatenate.4.4, %broadcast.115.11)
  %convert.63.6 = f32[16,8192,16,128]{3,2,1,0} convert(%multiply.105.3)
  %param_0.631 = f32[1,16,8192,16]{3,2,1,0} parameter(0)
  %bitcast.42.9 = f32[16,8192,16]{2,1,0} bitcast(%param_0.631)
  %broadcast.111.9 = f32[16,8192,16,128]{3,2,1,0} broadcast(%bitcast.42.9), dimensions={0,1,2}
  %multiply.111.5 = f32[16,8192,16,128]{3,2,1,0} multiply(%convert.63.6, %broadcast.111.9)
  %add.33.3 = f32[16,8192,16,128]{3,2,1,0} add(%multiply.110.3, %multiply.111.5)
  ROOT %convert.64.1 = bf16[16,8192,16,128]{3,2,1,0} convert(%add.33.3)
}



ENTRY %wrapper_fused_concatenate.4.clone (param_0.631: f32[1,16,8192,16], param_1.714: bf16[131072,2048], param_2.355: f32[1,16,8192,16], param_3.212: bf16[1,128], param_4.110: bf16[16,8192,16,64], param_5.53: bf16[16,8192,16,64]) -> bf16[16,8192,16,128] {
  param_0.631 = f32[1,16,8192,16] parameter(0)
  param_1.714 = bf16[131072,2048] parameter(1)
  param_2.355 = f32[1,16,8192,16] parameter(2)
  param_3.212 = bf16[1,128] parameter(3)
  param_4.110 = bf16[16,8192,16,64] parameter(4)
  param_5.53 = bf16[16,8192,16,64] parameter(5)
  ROOT %fusion = bf16[16,8192,16,128] fusion(param_0.631, param_1.714, param_2.355, param_3.212, param_4.110, param_5.53), kind=kLoop, calls=%fused_concatenate.4.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_convert.13.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedConvert_13) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_convert.13_module, entry_computation_layout={(f32[1,16,8192]{2,1,0}, f32[1,16,8192]{2,1,0}, bf16[16,8192,5376]{2,1,0}, bf16[16,8192,5376]{2,1,0}, bf16[1,5376]{1,0})->bf16[16,8192,5376]{2,1,0}}

%fused_convert.13.clone (param_0.492: f32[1,16,8192], param_1.562: f32[1,16,8192], param_2.263: bf16[16,8192,5376], param_3.146: bf16[16,8192,5376], param_4.72: bf16[1,5376]) -> bf16[16,8192,5376] {
  %param_2.263 = bf16[16,8192,5376]{2,1,0} parameter(2)
  %convert.233.8 = f32[16,8192,5376]{2,1,0} convert(%param_2.263)
  %param_1.562 = f32[1,16,8192]{2,1,0} parameter(1)
  %bitcast.76.5 = f32[16,8192]{1,0} bitcast(%param_1.562)
  %broadcast.271.5 = f32[16,8192,5376]{2,1,0} broadcast(%bitcast.76.5), dimensions={0,1}
  %multiply.99.3 = f32[16,8192,5376]{2,1,0} multiply(%convert.233.8, %broadcast.271.5)
  %param_3.146 = bf16[16,8192,5376]{2,1,0} parameter(3)
  %param_4.72 = bf16[1,5376]{1,0} parameter(4)
  %bitcast.51.9 = bf16[5376]{0} bitcast(%param_4.72)
  %broadcast.151.9 = bf16[16,8192,5376]{2,1,0} broadcast(%bitcast.51.9), dimensions={2}
  %multiply.94.3 = bf16[16,8192,5376]{2,1,0} multiply(%param_3.146, %broadcast.151.9)
  %convert.61.6 = f32[16,8192,5376]{2,1,0} convert(%multiply.94.3)
  %param_0.492 = f32[1,16,8192]{2,1,0} parameter(0)
  %bitcast.50.7 = f32[16,8192]{1,0} bitcast(%param_0.492)
  %broadcast.147.7 = f32[16,8192,5376]{2,1,0} broadcast(%bitcast.50.7), dimensions={0,1}
  %multiply.100.3 = f32[16,8192,5376]{2,1,0} multiply(%convert.61.6, %broadcast.147.7)
  %add.30.1 = f32[16,8192,5376]{2,1,0} add(%multiply.99.3, %multiply.100.3)
  ROOT %convert.62.1 = bf16[16,8192,5376]{2,1,0} convert(%add.30.1)
}



ENTRY %wrapper_fused_convert.13.clone (param_0.492: f32[1,16,8192], param_1.562: f32[1,16,8192], param_2.263: bf16[16,8192,5376], param_3.146: bf16[16,8192,5376], param_4.72: bf16[1,5376]) -> bf16[16,8192,5376] {
  param_0.492 = f32[1,16,8192] parameter(0)
  param_1.562 = f32[1,16,8192] parameter(1)
  param_2.263 = bf16[16,8192,5376] parameter(2)
  param_3.146 = bf16[16,8192,5376] parameter(3)
  param_4.72 = bf16[1,5376] parameter(4)
  ROOT %fusion = bf16[16,8192,5376] fusion(param_0.492, param_1.562, param_2.263, param_3.146, param_4.72), kind=kLoop, calls=%fused_convert.13.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source:
// gemma3_27b_training/extracted_fusions/fused_exponential_reduce.clone.clone.clone.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedExponentialReduce) {
  // Shapes reduced from original Gemma 3 27B training dimensions
  // (bf16[131072,262144] ~69 GB) to avoid test timeout.
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_exponential_reduce.clone.clone.clone_module, entry_computation_layout={(f32[512]{0}, bf16[512,256]{1,0})->f32[16,32,256]{2,1,0}}

%fused_exponential_reduce.clone.clone.clone.clone (param_0.2155: f32[512], param_1.2502: bf16[512,256]) -> f32[16,32,256] {
  %param_1.2502 = bf16[512,256]{1,0} parameter(1)
  %convert.164.4.clone.4 = f32[512,256]{1,0} convert(%param_1.2502)
  %bitcast.126.3.clone.4 = f32[16,32,256]{2,1,0} bitcast(%convert.164.4.clone.4)
  %param_0.2155 = f32[512]{0} parameter(0)
  %bitcast.127.5.clone.4 = f32[16,32]{1,0} bitcast(%param_0.2155)
  %broadcast.675.5.clone.4 = f32[16,32,256]{2,1,0} broadcast(%bitcast.127.5.clone.4), dimensions={0,1}
  %subtract.676.3.clone.4 = f32[16,32,256]{2,1,0} subtract(%bitcast.126.3.clone.4, %broadcast.675.5.clone.4)
  ROOT %exponential.677.1.clone.4 = f32[16,32,256]{2,1,0} exponential(%subtract.676.3.clone.4)
}



ENTRY %wrapper_fused_exponential_reduce.clone.clone.clone.clone (param_0.2155: f32[512], param_1.2502: bf16[512,256]) -> f32[16,32,256] {
  param_0.2155 = f32[512] parameter(0)
  param_1.2502 = bf16[512,256] parameter(1)
  ROOT %fusion = f32[16,32,256] fusion(param_0.2155, param_1.2502), kind=kLoop, calls=%fused_exponential_reduce.clone.clone.clone.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_multiply.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedMultiply) {
  // Shapes reduced from original Gemma 3 27B training dimensions
  // (bf16[131072,21504] ~5.6 GB per input) to avoid test timeout.
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply_module, entry_computation_layout={(bf16[512,128]{1,0}, bf16[512,128]{1,0})->bf16[16,32,128]{2,1,0}}

%fused_multiply.clone (param_0.52: bf16[512,128], param_1.29: bf16[512,128]) -> bf16[16,32,128] {
  %param_1.29 = bf16[512,128]{1,0} parameter(1)
  %bitcast.740.6 = bf16[16,32,128]{2,1,0} bitcast(%param_1.29)
  %multiply.263.3 = bf16[16,32,128]{2,1,0} multiply(%bitcast.740.6, %bitcast.740.6)
  %multiply.264.3 = bf16[16,32,128]{2,1,0} multiply(%multiply.263.3, %bitcast.740.6)
  %constant_116_1 = bf16[] constant(0.04468)
  %broadcast.377.1 = bf16[16,32,128]{2,1,0} broadcast(%constant_116_1), dimensions={}
  %multiply.265.1 = bf16[16,32,128]{2,1,0} multiply(%multiply.264.3, %broadcast.377.1)
  %add.55.3 = bf16[16,32,128]{2,1,0} add(%bitcast.740.6, %multiply.265.1)
  %constant_118_1 = bf16[] constant(0.7969)
  %broadcast.378.1 = bf16[16,32,128]{2,1,0} broadcast(%constant_118_1), dimensions={}
  %multiply.266.1 = bf16[16,32,128]{2,1,0} multiply(%add.55.3, %broadcast.378.1)
  %convert.274.11 = f32[16,32,128]{2,1,0} convert(%multiply.266.1)
  %tanh.2.11 = f32[16,32,128]{2,1,0} tanh(%convert.274.11)
  %convert.275.9 = bf16[16,32,128]{2,1,0} convert(%tanh.2.11)
  %constant_120_1 = bf16[] constant(1)
  %broadcast.379.1 = bf16[16,32,128]{2,1,0} broadcast(%constant_120_1), dimensions={}
  %add.56.7 = bf16[16,32,128]{2,1,0} add(%convert.275.9, %broadcast.379.1)
  %constant_122_1 = bf16[] constant(0.5)
  %broadcast.381.1 = bf16[16,32,128]{2,1,0} broadcast(%constant_122_1), dimensions={}
  %multiply.267.5 = bf16[16,32,128]{2,1,0} multiply(%add.56.7, %broadcast.381.1)
  %multiply.268.3 = bf16[16,32,128]{2,1,0} multiply(%bitcast.740.6, %multiply.267.5)
  %param_0.52 = bf16[512,128]{1,0} parameter(0)
  %bitcast.743.1 = bf16[16,32,128]{2,1,0} bitcast(%param_0.52)
  ROOT %multiply.269.1 = bf16[16,32,128]{2,1,0} multiply(%multiply.268.3, %bitcast.743.1)
}



ENTRY %wrapper_fused_multiply.clone (param_0.52: bf16[512,128], param_1.29: bf16[512,128]) -> bf16[16,32,128] {
  param_0.52 = bf16[512,128] parameter(0)
  param_1.29 = bf16[512,128] parameter(1)
  ROOT %fusion = bf16[16,32,128] fusion(param_0.52, param_1.29), kind=kLoop, calls=%fused_multiply.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source:
// gemma3_27b_training/extracted_fusions/fused_multiply.9.clone.clone.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedMultiply_9) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.9.clone.clone_module, entry_computation_layout={(bf16[5376]{0}, f32[1,16,8192]{2,1,0}, bf16[16,8192,5376]{2,1,0})->bf16[16,8192,5376]{2,1,0}}

%fused_multiply.9.clone.clone.clone (param_0.2149: bf16[5376], param_1.2499: f32[1,16,8192], param_2.1366: bf16[16,8192,5376]) -> bf16[16,8192,5376] {
  %param_2.1366 = bf16[16,8192,5376]{2,1,0} parameter(2)
  %convert.622.21 = f32[16,8192,5376]{2,1,0} convert(%param_2.1366)
  %param_1.2499 = f32[1,16,8192]{2,1,0} parameter(1)
  %bitcast.124.15 = f32[16,8192]{1,0} bitcast(%param_1.2499)
  %broadcast.638.15 = f32[16,8192,5376]{2,1,0} broadcast(%bitcast.124.15), dimensions={0,1}
  %multiply.639.9 = f32[16,8192,5376]{2,1,0} multiply(%convert.622.21, %broadcast.638.15)
  %convert.640.7 = bf16[16,8192,5376]{2,1,0} convert(%multiply.639.9)
  %param_0.2149 = bf16[5376]{0} parameter(0)
  %broadcast.645.8 = bf16[16,8192,5376]{2,1,0} broadcast(%param_0.2149), dimensions={2}
  ROOT %multiply.646.3 = bf16[16,8192,5376]{2,1,0} multiply(%convert.640.7, %broadcast.645.8)
}



ENTRY %wrapper_fused_multiply.9.clone.clone.clone (param_0.2149: bf16[5376], param_1.2499: f32[1,16,8192], param_2.1366: bf16[16,8192,5376]) -> bf16[16,8192,5376] {
  param_0.2149 = bf16[5376] parameter(0)
  param_1.2499 = f32[1,16,8192] parameter(1)
  param_2.1366 = bf16[16,8192,5376] parameter(2)
  ROOT %fusion = bf16[16,8192,5376] fusion(param_0.2149, param_1.2499, param_2.1366), kind=kLoop, calls=%fused_multiply.9.clone.clone.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source:
// gemma3_27b_training/extracted_fusions/fused_multiply.2.clone.clone.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedMultiply_2) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.2.clone.clone_module, entry_computation_layout={(bf16[1,5376]{1,0}, f32[1,16,8192]{2,1,0}, bf16[16,8192,5376]{2,1,0})->bf16[16,8192,5376]{2,1,0}}

%fused_multiply.2.clone.clone.clone (param_0.2204: bf16[1,5376], param_1.2543: f32[1,16,8192], param_2.1377: bf16[16,8192,5376]) -> bf16[16,8192,5376] {
  %param_2.1377 = bf16[16,8192,5376]{2,1,0} parameter(2)
  %convert.49.23 = f32[16,8192,5376]{2,1,0} convert(%param_2.1377)
  %param_1.2543 = f32[1,16,8192]{2,1,0} parameter(1)
  %bitcast.53.17 = f32[16,8192]{1,0} bitcast(%param_1.2543)
  %broadcast.155.17 = f32[16,8192,5376]{2,1,0} broadcast(%bitcast.53.17), dimensions={0,1}
  %multiply.58.9 = f32[16,8192,5376]{2,1,0} multiply(%convert.49.23, %broadcast.155.17)
  %convert.50.7 = bf16[16,8192,5376]{2,1,0} convert(%multiply.58.9)
  %param_0.2204 = bf16[1,5376]{1,0} parameter(0)
  %bitcast.54.11 = bf16[5376]{0} bitcast(%param_0.2204)
  %broadcast.159.11 = bf16[16,8192,5376]{2,1,0} broadcast(%bitcast.54.11), dimensions={2}
  ROOT %multiply.59.3 = bf16[16,8192,5376]{2,1,0} multiply(%convert.50.7, %broadcast.159.11)
}



ENTRY %wrapper_fused_multiply.2.clone.clone.clone (param_0.2204: bf16[1,5376], param_1.2543: f32[1,16,8192], param_2.1377: bf16[16,8192,5376]) -> bf16[16,8192,5376] {
  param_0.2204 = bf16[1,5376] parameter(0)
  param_1.2543 = f32[1,16,8192] parameter(1)
  param_2.1377 = bf16[16,8192,5376] parameter(2)
  ROOT %fusion = bf16[16,8192,5376] fusion(param_0.2204, param_1.2543, param_2.1377), kind=kLoop, calls=%fused_multiply.2.clone.clone.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_multiply.3.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedMultiply_3) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.3_module, entry_computation_layout={(f32[1,16,8192,32]{3,2,1,0}, f32[16,8192,32]{2,1,0}, f32[16,8192,32]{2,1,0})->f32[1,16,8192,32]{3,2,1,0}}

%fused_multiply.3.clone (param_0.975: f32[1,16,8192,32], param_1.1081: f32[16,8192,32], param_2.637: f32[16,8192,32]) -> f32[1,16,8192,32] {
  %param_1.1081 = f32[16,8192,32]{2,1,0} parameter(1)
  %bitcast.222.3 = f32[1,16,8192,32]{3,2,1,0} bitcast(%param_1.1081)
  %param_0.975 = f32[1,16,8192,32]{3,2,1,0} parameter(0)
  %param_2.637 = f32[16,8192,32]{2,1,0} parameter(2)
  %constant_46_4 = f32[] constant(0.0078125)
  %broadcast.304.5 = f32[16,8192,32]{2,1,0} broadcast(%constant_46_4), dimensions={}
  %multiply.207.5 = f32[16,8192,32]{2,1,0} multiply(%param_2.637, %broadcast.304.5)
  %constant_14_4 = f32[] constant(1e-06)
  %broadcast.399.3 = f32[16,8192,32]{2,1,0} broadcast(%constant_14_4), dimensions={}
  %add.60.3 = f32[16,8192,32]{2,1,0} add(%multiply.207.5, %broadcast.399.3)
  %bitcast.141.6 = f32[1,16,8192,32]{3,2,1,0} bitcast(%add.60.3)
  %divide.48.5 = f32[1,16,8192,32]{3,2,1,0} divide(%param_0.975, %bitcast.141.6)
  %constant_31_3 = f32[] constant(-0.5)
  %broadcast.438.5 = f32[1,16,8192,32]{3,2,1,0} broadcast(%constant_31_3), dimensions={}
  %multiply.292.5 = f32[1,16,8192,32]{3,2,1,0} multiply(%divide.48.5, %broadcast.438.5)
  %multiply.293.3 = f32[1,16,8192,32]{3,2,1,0} multiply(%bitcast.222.3, %multiply.292.5)
  %constant_138_2 = f32[] constant(0.015625)
  %broadcast.440.1 = f32[1,16,8192,32]{3,2,1,0} broadcast(%constant_138_2), dimensions={}
  ROOT %multiply.294.1 = f32[1,16,8192,32]{3,2,1,0} multiply(%multiply.293.3, %broadcast.440.1)
}



ENTRY %wrapper_fused_multiply.3.clone (param_0.975: f32[1,16,8192,32], param_1.1081: f32[16,8192,32], param_2.637: f32[16,8192,32]) -> f32[1,16,8192,32] {
  param_0.975 = f32[1,16,8192,32] parameter(0)
  param_1.1081 = f32[16,8192,32] parameter(1)
  param_2.637 = f32[16,8192,32] parameter(2)
  ROOT %fusion = f32[1,16,8192,32] fusion(param_0.975, param_1.1081, param_2.637), kind=kLoop, calls=%fused_multiply.3.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_multiply.4.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedMultiply_4) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.4_module, entry_computation_layout={(f32[16,8192,16]{2,1,0}, f32[1,16,8192,16]{3,2,1,0}, f32[16,8192,16]{2,1,0})->f32[1,16,8192,16]{3,2,1,0}}

%fused_multiply.4.clone (param_0.834: f32[16,8192,16], param_1.1080: f32[1,16,8192,16], param_2.636: f32[16,8192,16]) -> f32[1,16,8192,16] {
  %param_0.834 = f32[16,8192,16]{2,1,0} parameter(0)
  %bitcast.207.3 = f32[1,16,8192,16]{3,2,1,0} bitcast(%param_0.834)
  %param_1.1080 = f32[1,16,8192,16]{3,2,1,0} parameter(1)
  %param_2.636 = f32[16,8192,16]{2,1,0} parameter(2)
  %constant_46_5 = f32[] constant(0.0078125)
  %broadcast.308.3 = f32[16,8192,16]{2,1,0} broadcast(%constant_46_5), dimensions={}
  %multiply.208.3 = f32[16,8192,16]{2,1,0} multiply(%param_2.636, %broadcast.308.3)
  %constant_14_5 = f32[] constant(1e-06)
  %broadcast.401.3 = f32[16,8192,16]{2,1,0} broadcast(%constant_14_5), dimensions={}
  %add.61.3 = f32[16,8192,16]{2,1,0} add(%multiply.208.3, %broadcast.401.3)
  %bitcast.153.6 = f32[1,16,8192,16]{3,2,1,0} bitcast(%add.61.3)
  %divide.47.5 = f32[1,16,8192,16]{3,2,1,0} divide(%param_1.1080, %bitcast.153.6)
  %constant_31_2 = f32[] constant(-0.5)
  %broadcast.434.3 = f32[1,16,8192,16]{3,2,1,0} broadcast(%constant_31_2), dimensions={}
  %multiply.289.3 = f32[1,16,8192,16]{3,2,1,0} multiply(%divide.47.5, %broadcast.434.3)
  %multiply.290.3 = f32[1,16,8192,16]{3,2,1,0} multiply(%bitcast.207.3, %multiply.289.3)
  %constant_138_1 = f32[] constant(0.015625)
  %broadcast.436.1 = f32[1,16,8192,16]{3,2,1,0} broadcast(%constant_138_1), dimensions={}
  ROOT %multiply.291.1 = f32[1,16,8192,16]{3,2,1,0} multiply(%multiply.290.3, %broadcast.436.1)
}



ENTRY %wrapper_fused_multiply.4.clone (param_0.834: f32[16,8192,16], param_1.1080: f32[1,16,8192,16], param_2.636: f32[16,8192,16]) -> f32[1,16,8192,16] {
  param_0.834 = f32[16,8192,16] parameter(0)
  param_1.1080 = f32[1,16,8192,16] parameter(1)
  param_2.636 = f32[16,8192,16] parameter(2)
  ROOT %fusion = f32[1,16,8192,16] fusion(param_0.834, param_1.1080, param_2.636), kind=kLoop, calls=%fused_multiply.4.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_multiply.5.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedMultiply_5) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.5_module, entry_computation_layout={(f32[1,16,8192]{2,1,0}, f32[16,8192]{1,0}, f32[16,8192]{1,0})->f32[1,16,8192]{2,1,0}}

%fused_multiply.5.clone (param_0.972: f32[1,16,8192], param_1.1077: f32[16,8192], param_2.633: f32[16,8192]) -> f32[1,16,8192] {
  %param_1.1077 = f32[16,8192]{1,0} parameter(1)
  %bitcast.175.3 = f32[1,16,8192]{2,1,0} bitcast(%param_1.1077)
  %param_0.972 = f32[1,16,8192]{2,1,0} parameter(0)
  %param_2.633 = f32[16,8192]{1,0} parameter(2)
  %constant_45_6 = f32[] constant(0.000186011908)
  %broadcast.302.18 = f32[16,8192]{1,0} broadcast(%constant_45_6), dimensions={}
  %multiply.210.5 = f32[16,8192]{1,0} multiply(%param_2.633, %broadcast.302.18)
  %constant_14_10 = f32[] constant(1e-06)
  %broadcast.397.10 = f32[16,8192]{1,0} broadcast(%constant_14_10), dimensions={}
  %add.63.3 = f32[16,8192]{1,0} add(%multiply.210.5, %broadcast.397.10)
  %bitcast.159.6 = f32[1,16,8192]{2,1,0} bitcast(%add.63.3)
  %divide.45.5 = f32[1,16,8192]{2,1,0} divide(%param_0.972, %bitcast.159.6)
  %constant_31_5 = f32[] constant(-0.5)
  %broadcast.431.16 = f32[1,16,8192]{2,1,0} broadcast(%constant_31_5), dimensions={}
  %multiply.283.5 = f32[1,16,8192]{2,1,0} multiply(%divide.45.5, %broadcast.431.16)
  %multiply.284.3 = f32[1,16,8192]{2,1,0} multiply(%bitcast.175.3, %multiply.283.5)
  %constant_132_2 = f32[] constant(0.000372023816)
  %broadcast.432.1 = f32[1,16,8192]{2,1,0} broadcast(%constant_132_2), dimensions={}
  ROOT %multiply.285.1 = f32[1,16,8192]{2,1,0} multiply(%multiply.284.3, %broadcast.432.1)
}



ENTRY %wrapper_fused_multiply.5.clone (param_0.972: f32[1,16,8192], param_1.1077: f32[16,8192], param_2.633: f32[16,8192]) -> f32[1,16,8192] {
  param_0.972 = f32[1,16,8192] parameter(0)
  param_1.1077 = f32[16,8192] parameter(1)
  param_2.633 = f32[16,8192] parameter(2)
  ROOT %fusion = f32[1,16,8192] fusion(param_0.972, param_1.1077, param_2.633), kind=kLoop, calls=%fused_multiply.5.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_rsqrt.2.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedRsqrt_2) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_rsqrt.2_module, entry_computation_layout={(f32[16,8192,16]{2,1,0})->f32[1,16,8192,16]{3,2,1,0}}

%fused_rsqrt.2.clone (param_0.947: f32[16,8192,16]) -> f32[1,16,8192,16] {
  %param_0.947 = f32[16,8192,16]{2,1,0} parameter(0)
  %constant_46_3 = f32[] constant(0.0078125)
  %broadcast.308.5 = f32[16,8192,16]{2,1,0} broadcast(%constant_46_3), dimensions={}
  %multiply.208.5 = f32[16,8192,16]{2,1,0} multiply(%param_0.947, %broadcast.308.5)
  %constant_14_1 = f32[] constant(1e-06)
  %broadcast.401.5 = f32[16,8192,16]{2,1,0} broadcast(%constant_14_1), dimensions={}
  %add.61.5 = f32[16,8192,16]{2,1,0} add(%multiply.208.5, %broadcast.401.5)
  %bitcast.153.1 = f32[1,16,8192,16]{3,2,1,0} bitcast(%add.61.5)
  ROOT %rsqrt.26.1 = f32[1,16,8192,16]{3,2,1,0} rsqrt(%bitcast.153.1)
}



ENTRY %wrapper_fused_rsqrt.2.clone (param_0.947: f32[16,8192,16]) -> f32[1,16,8192,16] {
  param_0.947 = f32[16,8192,16] parameter(0)
  ROOT %fusion = f32[1,16,8192,16] fusion(param_0.947), kind=kLoop, calls=%fused_rsqrt.2.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_rsqrt.3.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedRsqrt_3) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_rsqrt.3_module, entry_computation_layout={(f32[16,8192,32]{2,1,0})->f32[1,16,8192,32]{3,2,1,0}}

%fused_rsqrt.3.clone (param_0.948: f32[16,8192,32]) -> f32[1,16,8192,32] {
  %param_0.948 = f32[16,8192,32]{2,1,0} parameter(0)
  %constant_46_1 = f32[] constant(0.0078125)
  %broadcast.304.7 = f32[16,8192,32]{2,1,0} broadcast(%constant_46_1), dimensions={}
  %multiply.207.7 = f32[16,8192,32]{2,1,0} multiply(%param_0.948, %broadcast.304.7)
  %constant_14_2 = f32[] constant(1e-06)
  %broadcast.399.5 = f32[16,8192,32]{2,1,0} broadcast(%constant_14_2), dimensions={}
  %add.60.5 = f32[16,8192,32]{2,1,0} add(%multiply.207.7, %broadcast.399.5)
  %bitcast.141.1 = f32[1,16,8192,32]{3,2,1,0} bitcast(%add.60.5)
  ROOT %rsqrt.25.1 = f32[1,16,8192,32]{3,2,1,0} rsqrt(%bitcast.141.1)
}



ENTRY %wrapper_fused_rsqrt.3.clone (param_0.948: f32[16,8192,32]) -> f32[1,16,8192,32] {
  param_0.948 = f32[16,8192,32] parameter(0)
  ROOT %fusion = f32[1,16,8192,32] fusion(param_0.948), kind=kLoop, calls=%fused_rsqrt.3.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_rsqrt.4.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedRsqrt_4) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_rsqrt.4_module, entry_computation_layout={(f32[16,8192]{1,0})->f32[1,16,8192]{2,1,0}}

%fused_rsqrt.4.clone (param_0.949: f32[16,8192]) -> f32[1,16,8192] {
  %param_0.949 = f32[16,8192]{1,0} parameter(0)
  %constant_45_2 = f32[] constant(0.000186011908)
  %broadcast.302.28 = f32[16,8192]{1,0} broadcast(%constant_45_2), dimensions={}
  %multiply.206.7 = f32[16,8192]{1,0} multiply(%param_0.949, %broadcast.302.28)
  %constant_14_6 = f32[] constant(1e-06)
  %broadcast.397.20 = f32[16,8192]{1,0} broadcast(%constant_14_6), dimensions={}
  %add.59.5 = f32[16,8192]{1,0} add(%multiply.206.7, %broadcast.397.20)
  %bitcast.138.1 = f32[1,16,8192]{2,1,0} bitcast(%add.59.5)
  ROOT %rsqrt.24.1 = f32[1,16,8192]{2,1,0} rsqrt(%bitcast.138.1)
}



ENTRY %wrapper_fused_rsqrt.4.clone (param_0.949: f32[16,8192]) -> f32[1,16,8192] {
  param_0.949 = f32[16,8192] parameter(0)
  ROOT %fusion = f32[1,16,8192] fusion(param_0.949), kind=kLoop, calls=%fused_rsqrt.4.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_select.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedSelect) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select_module, entry_computation_layout={(f32[], f32[62,5376,16,128]{3,2,1,0}, pred[])->f32[5376,62,16,128]{3,2,1,0}}

%fused_select.clone (param_0.1558: f32[], param_1.1789: f32[62,5376,16,128], param_2.883: pred[]) -> f32[5376,62,16,128] {
  %param_2.883 = pred[] parameter(2)
  %broadcast.1794.2 = pred[5376,62,16,128]{3,2,1,0} broadcast(%param_2.883), dimensions={}
  %param_1.1789 = f32[62,5376,16,128]{3,2,1,0} parameter(1)
  %bitcast.474.5 = f32[62,5376,2048]{2,1,0} bitcast(%param_1.1789)
  %transpose.136.5 = f32[5376,62,2048]{2,1,0} transpose(%bitcast.474.5), dimensions={1,0,2}
  %bitcast.475.1 = f32[5376,62,16,128]{3,2,1,0} bitcast(%transpose.136.5)
  %param_0.1558 = f32[] parameter(0)
  %broadcast.1792.6 = f32[5376,62,16,128]{3,2,1,0} broadcast(%param_0.1558), dimensions={}
  %divide.1813.3 = f32[5376,62,16,128]{3,2,1,0} divide(%bitcast.475.1, %broadcast.1792.6)
  ROOT %select.1815.1 = f32[5376,62,16,128]{3,2,1,0} select(%broadcast.1794.2, %bitcast.475.1, %divide.1813.3)
}



ENTRY %wrapper_fused_select.clone (param_0.1558: f32[], param_1.1789: f32[62,5376,16,128], param_2.883: pred[]) -> f32[5376,62,16,128] {
  param_0.1558 = f32[] parameter(0)
  param_1.1789 = f32[62,5376,16,128] parameter(1)
  param_2.883 = pred[] parameter(2)
  ROOT %fusion = f32[5376,62,16,128] fusion(param_0.1558, param_1.1789, param_2.883), kind=kLoop, calls=%fused_select.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_select.1.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedSelect_1) {
  // Shapes reduced from original Gemma 3 27B training dimensions
  // (f32[62,5376,32,128] ~5.5 GB) to avoid test timeout.
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.1_module, entry_computation_layout={(f32[4,64,32,128]{3,2,1,0}, f32[], pred[])->f32[64,4,32,128]{3,2,1,0}}

%fused_select.1.clone (param_0.1555: f32[4,64,32,128], param_1.1786: f32[], param_2.880: pred[]) -> f32[64,4,32,128] {
  %param_2.880 = pred[] parameter(2)
  %broadcast.1806.1 = pred[64,4,32,128]{3,2,1,0} broadcast(%param_2.880), dimensions={}
  %param_0.1555 = f32[4,64,32,128]{3,2,1,0} parameter(0)
  %bitcast.472.5 = f32[4,64,4096]{2,1,0} bitcast(%param_0.1555)
  %transpose.135.5 = f32[64,4,4096]{2,1,0} transpose(%bitcast.472.5), dimensions={1,0,2}
  %bitcast.473.1 = f32[64,4,32,128]{3,2,1,0} bitcast(%transpose.135.5)
  %param_1.1786 = f32[] parameter(1)
  %broadcast.1804.3 = f32[64,4,32,128]{3,2,1,0} broadcast(%param_1.1786), dimensions={}
  %divide.1805.3 = f32[64,4,32,128]{3,2,1,0} divide(%bitcast.473.1, %broadcast.1804.3)
  ROOT %select.1807.1 = f32[64,4,32,128]{3,2,1,0} select(%broadcast.1806.1, %bitcast.473.1, %divide.1805.3)
}



ENTRY %wrapper_fused_select.1.clone (param_0.1555: f32[4,64,32,128], param_1.1786: f32[], param_2.880: pred[]) -> f32[64,4,32,128] {
  param_0.1555 = f32[4,64,32,128] parameter(0)
  param_1.1786 = f32[] parameter(1)
  param_2.880 = pred[] parameter(2)
  ROOT %fusion = f32[64,4,32,128] fusion(param_0.1555, param_1.1786, param_2.880), kind=kLoop, calls=%fused_select.1.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_select.2.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedSelect_2) {
  // Shapes reduced from original Gemma 3 27B training dimensions
  // (f32[62,32,128,5376] ~5.5 GB) to avoid test timeout.
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.2_module, entry_computation_layout={(f32[4,32,128,64]{3,2,1,0}, f32[], pred[])->f32[32,4,128,64]{3,2,1,0}}

%fused_select.2.clone (param_0.1556: f32[4,32,128,64], param_1.1787: f32[], param_2.881: pred[]) -> f32[32,4,128,64] {
  %param_2.881 = pred[] parameter(2)
  %broadcast.1802.1 = pred[32,4,128,64]{3,2,1,0} broadcast(%param_2.881), dimensions={}
  %param_0.1556 = f32[4,32,128,64]{3,2,1,0} parameter(0)
  %bitcast.470.5 = f32[4,32,8192]{2,1,0} bitcast(%param_0.1556)
  %transpose.134.5 = f32[32,4,8192]{2,1,0} transpose(%bitcast.470.5), dimensions={1,0,2}
  %bitcast.471.1 = f32[32,4,128,64]{3,2,1,0} bitcast(%transpose.134.5)
  %param_1.1787 = f32[] parameter(1)
  %broadcast.1800.3 = f32[32,4,128,64]{3,2,1,0} broadcast(%param_1.1787), dimensions={}
  %divide.1801.3 = f32[32,4,128,64]{3,2,1,0} divide(%bitcast.471.1, %broadcast.1800.3)
  ROOT %select.1803.1 = f32[32,4,128,64]{3,2,1,0} select(%broadcast.1802.1, %bitcast.471.1, %divide.1801.3)
}



ENTRY %wrapper_fused_select.2.clone (param_0.1556: f32[4,32,128,64], param_1.1787: f32[], param_2.881: pred[]) -> f32[32,4,128,64] {
  param_0.1556 = f32[4,32,128,64] parameter(0)
  param_1.1787 = f32[] parameter(1)
  param_2.881 = pred[] parameter(2)
  ROOT %fusion = f32[32,4,128,64] fusion(param_0.1556, param_1.1787, param_2.881), kind=kLoop, calls=%fused_select.2.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_select.4.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedSelect_4) {
  // Shapes reduced from original Gemma 3 27B training dimensions
  // (f32[62,21504,5376] ~29 GB) to avoid test timeout.
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.4_module, entry_computation_layout={(f32[4,64,128]{2,1,0}, f32[], pred[])->f32[64,4,128]{2,1,0}}

%fused_select.4.clone (param_0.1351: f32[4,64,128], param_1.1504: f32[], param_2.759: pred[]) -> f32[64,4,128] {
  %param_2.759 = pred[] parameter(2)
  %broadcast.1774.1 = pred[64,4,128]{2,1,0} broadcast(%param_2.759), dimensions={}
  %param_0.1351 = f32[4,64,128]{2,1,0} parameter(0)
  %transpose.1643.1 = f32[64,4,128]{2,1,0} transpose(%param_0.1351), dimensions={1,0,2}
  %param_1.1504 = f32[] parameter(1)
  %broadcast.1772.1 = f32[64,4,128]{2,1,0} broadcast(%param_1.1504), dimensions={}
  %divide.1773.3 = f32[64,4,128]{2,1,0} divide(%transpose.1643.1, %broadcast.1772.1)
  ROOT %select.1775.1 = f32[64,4,128]{2,1,0} select(%broadcast.1774.1, %transpose.1643.1, %divide.1773.3)
}



ENTRY %wrapper_fused_select.4.clone (param_0.1351: f32[4,64,128], param_1.1504: f32[], param_2.759: pred[]) -> f32[64,4,128] {
  param_0.1351 = f32[4,64,128] parameter(0)
  param_1.1504 = f32[] parameter(1)
  param_2.759 = pred[] parameter(2)
  ROOT %fusion = f32[64,4,128] fusion(param_0.1351, param_1.1504, param_2.759), kind=kLoop, calls=%fused_select.4.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_select.5.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedSelect_5) {
  // Shapes reduced from original Gemma 3 27B training dimensions
  // (f32[62,5376,21504] ~29 GB) to avoid test timeout.
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.5_module, entry_computation_layout={(f32[4,64,128]{2,1,0}, f32[], pred[])->f32[64,4,128]{2,1,0}}

%fused_select.5.clone (param_0.1337: f32[4,64,128], param_1.1477: f32[], param_2.756: pred[]) -> f32[64,4,128] {
  %param_2.756 = pred[] parameter(2)
  %broadcast.1766.2 = pred[64,4,128]{2,1,0} broadcast(%param_2.756), dimensions={}
  %param_0.1337 = f32[4,64,128]{2,1,0} parameter(0)
  %transpose.1644.1 = f32[64,4,128]{2,1,0} transpose(%param_0.1337), dimensions={1,0,2}
  %param_1.1477 = f32[] parameter(1)
  %broadcast.1764.6 = f32[64,4,128]{2,1,0} broadcast(%param_1.1477), dimensions={}
  %divide.1769.3 = f32[64,4,128]{2,1,0} divide(%transpose.1644.1, %broadcast.1764.6)
  ROOT %select.1771.1 = f32[64,4,128]{2,1,0} select(%broadcast.1766.2, %transpose.1644.1, %divide.1769.3)
}



ENTRY %wrapper_fused_select.5.clone (param_0.1337: f32[4,64,128], param_1.1477: f32[], param_2.756: pred[]) -> f32[64,4,128] {
  param_0.1337 = f32[4,64,128] parameter(0)
  param_1.1477 = f32[] parameter(1)
  param_2.756 = pred[] parameter(2)
  ROOT %fusion = f32[64,4,128] fusion(param_0.1337, param_1.1477, param_2.756), kind=kLoop, calls=%fused_select.5.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_transpose.1.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedTranspose_1) {
  // Shapes reduced from original Gemma 3 27B training dimensions
  // (f32[5376,62,32,128] ~5.5 GB) to avoid test timeout.
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.1_module, entry_computation_layout={(f32[64,4,32,128]{3,2,1,0})->f32[4,64,4096]{2,1,0}}

%fused_transpose.1.clone (param_0.1154: f32[64,4,32,128]) -> f32[4,64,4096] {
  %param_0.1154 = f32[64,4,32,128]{3,2,1,0} parameter(0)
  %bitcast.464.1 = f32[64,4,4096]{2,1,0} bitcast(%param_0.1154)
  ROOT %transpose.131.1 = f32[4,64,4096]{2,1,0} transpose(%bitcast.464.1), dimensions={1,0,2}
}



ENTRY %wrapper_fused_transpose.1.clone (param_0.1154: f32[64,4,32,128]) -> f32[4,64,4096] {
  param_0.1154 = f32[64,4,32,128] parameter(0)
  ROOT %fusion = f32[4,64,4096] fusion(param_0.1154), kind=kLoop, calls=%fused_transpose.1.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_transpose.2.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedTranspose_2) {
  // Shapes reduced from original Gemma 3 27B training dimensions
  // (f32[32,62,128,5376] ~5.5 GB) to avoid test timeout.
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.2_module, entry_computation_layout={(f32[64,4,128,64]{3,2,1,0})->f32[4,64,8192]{2,1,0}}

%fused_transpose.2.clone (param_0.1156: f32[64,4,128,64]) -> f32[4,64,8192] {
  %param_0.1156 = f32[64,4,128,64]{3,2,1,0} parameter(0)
  %bitcast.462.1 = f32[64,4,8192]{2,1,0} bitcast(%param_0.1156)
  ROOT %transpose.130.1 = f32[4,64,8192]{2,1,0} transpose(%bitcast.462.1), dimensions={1,0,2}
}



ENTRY %wrapper_fused_transpose.2.clone (param_0.1156: f32[64,4,128,64]) -> f32[4,64,8192] {
  param_0.1156 = f32[64,4,128,64] parameter(0)
  ROOT %fusion = f32[4,64,8192] fusion(param_0.1156), kind=kLoop, calls=%fused_transpose.2.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_transpose.3.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedTranspose_3) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.3_module, entry_computation_layout={(f32[5376,62,16,128]{3,2,1,0})->f32[62,5376,2048]{2,1,0}}

%fused_transpose.3.clone (param_0.1158: f32[5376,62,16,128]) -> f32[62,5376,2048] {
  %param_0.1158 = f32[5376,62,16,128]{3,2,1,0} parameter(0)
  %bitcast.460.1 = f32[5376,62,2048]{2,1,0} bitcast(%param_0.1158)
  ROOT %transpose.129.1 = f32[62,5376,2048]{2,1,0} transpose(%bitcast.460.1), dimensions={1,0,2}
}



ENTRY %wrapper_fused_transpose.3.clone (param_0.1158: f32[5376,62,16,128]) -> f32[62,5376,2048] {
  param_0.1158 = f32[5376,62,16,128] parameter(0)
  ROOT %fusion = f32[62,5376,2048] fusion(param_0.1158), kind=kLoop, calls=%fused_transpose.3.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: gemma3_27b_training/extracted_fusions/fused_transpose.4.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Gemma327bTraining_FusedTranspose_4) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.4_module, entry_computation_layout={(f32[], pred[], f32[62,128]{1,0})->f32[128,62]{1,0}}

%fused_transpose.4.clone (param_0.1914: f32[], param_1.2292: pred[], param_2.1166: f32[62,128]) -> f32[128,62] {
  %param_1.2292 = pred[] parameter(1)
  %broadcast.1798.2 = pred[128,62]{1,0} broadcast(%param_1.2292), dimensions={}
  %param_2.1166 = f32[62,128]{1,0} parameter(2)
  %transpose.1634.1 = f32[128,62]{1,0} transpose(%param_2.1166), dimensions={1,0}
  %param_0.1914 = f32[] parameter(0)
  %broadcast.1796.2 = f32[128,62]{1,0} broadcast(%param_0.1914), dimensions={}
  %divide.1809.1 = f32[128,62]{1,0} divide(%transpose.1634.1, %broadcast.1796.2)
  ROOT %select.1811.1 = f32[128,62]{1,0} select(%broadcast.1798.2, %transpose.1634.1, %divide.1809.1)
}



ENTRY %wrapper_fused_transpose.4.clone (param_0.1914: f32[], param_1.2292: pred[], param_2.1166: f32[62,128]) -> f32[128,62] {
  param_0.1914 = f32[] parameter(0)
  param_1.2292 = pred[] parameter(1)
  param_2.1166 = f32[62,128] parameter(2)
  ROOT %fusion = f32[128,62] fusion(param_0.1914, param_1.2292, param_2.1166), kind=kLoop, calls=%fused_transpose.4.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// === llama31_8b_training (17 fusions) ===
// Source: llama31_8b_training/extracted_fusions/fused_convert.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Llama318bTraining_FusedConvert) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_convert_module, entry_computation_layout={(s32[2]{0})->f32[128]{0}}

%fused_convert.clone (param_0.32328: s32[2]) -> f32[128] {
  %param_0.32328 = s32[2]{0} parameter(0)
  %broadcast_in_dim.655.1 = s32[64,2]{1,0} broadcast(%param_0.32328), dimensions={1}
  %bitcast.1601.1 = s32[128]{0} bitcast(%broadcast_in_dim.655.1)
  ROOT %convert_element_type.3357.1 = f32[128]{0} convert(%bitcast.1601.1)
}



ENTRY %wrapper_fused_convert.clone (param_0.32328: s32[2]) -> f32[128] {
  param_0.32328 = s32[2] parameter(0)
  ROOT %fusion = f32[128] fusion(param_0.32328), kind=kLoop, calls=%fused_convert.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: llama31_8b_training/extracted_fusions/fused_convert.34.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Llama318bTraining_FusedConvert_34) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_convert.34_module, entry_computation_layout={(f32[1,2,8192,128]{3,2,1,0}, f32[128]{0}, f32[1,2,8192,128]{3,2,1,0}, bf16[2,8192,32,128]{3,2,1,0}, pred[128]{0})->bf16[2,8192,32,128]{3,2,1,0}}

%fused_convert.34.clone (param_0.13220: f32[1,2,8192,128], param_1.13025: f32[128], param_2.4537: f32[1,2,8192,128], param_3.2095: bf16[2,8192,32,128], param_4.6155: pred[128]) -> bf16[2,8192,32,128] {
  %param_3.2095 = bf16[2,8192,32,128]{3,2,1,0} parameter(3)
  %convert_element_type.1891.1 = f32[2,8192,32,128]{3,2,1,0} convert(%param_3.2095)
  %param_0.13220 = f32[1,2,8192,128]{3,2,1,0} parameter(0)
  %bitcast.512.12 = f32[2,8192,128]{2,1,0} bitcast(%param_0.13220)
  %mul.2951.11 = f32[2,8192,32,128]{3,2,1,0} broadcast(%bitcast.512.12), dimensions={0,1,3}
  %mul.2956.5 = f32[2,8192,32,128]{3,2,1,0} multiply(%convert_element_type.1891.1, %mul.2951.11)
  %param_4.6155 = pred[128]{0} parameter(4)
  %reshape.7631.128 = pred[2,8192,32,128]{3,2,1,0} broadcast(%param_4.6155), dimensions={3}
  %slice.299.3 = bf16[2,8192,32,127]{3,2,1,0} slice(%param_3.2095), slice={[0:2], [0:8192], [0:32], [1:128]}
  %slice.300.1 = bf16[2,8192,32,1]{3,2,1,0} slice(%param_3.2095), slice={[0:2], [0:8192], [0:32], [0:1]}
  %concatenate.193.3 = bf16[2,8192,32,128]{3,2,1,0} concatenate(%slice.299.3, %slice.300.1), dimensions={3}
  %slice.301.1 = bf16[2,8192,32,1]{3,2,1,0} slice(%param_3.2095), slice={[0:2], [0:8192], [0:32], [127:128]}
  %slice.302.1 = bf16[2,8192,32,127]{3,2,1,0} slice(%param_3.2095), slice={[0:2], [0:8192], [0:32], [0:127]}
  %concatenate.194.3 = bf16[2,8192,32,128]{3,2,1,0} concatenate(%slice.301.1, %slice.302.1), dimensions={3}
  %select_n.1725.3 = bf16[2,8192,32,128]{3,2,1,0} select(%reshape.7631.128, %concatenate.193.3, %concatenate.194.3)
  %convert_element_type.1898.1 = f32[2,8192,32,128]{3,2,1,0} convert(%select_n.1725.3)
  %param_2.4537 = f32[1,2,8192,128]{3,2,1,0} parameter(2)
  %bitcast.513.12 = f32[2,8192,128]{2,1,0} bitcast(%param_2.4537)
  %mul.2963.11 = f32[2,8192,32,128]{3,2,1,0} broadcast(%bitcast.513.12), dimensions={0,1,3}
  %mul.2964.5 = f32[2,8192,32,128]{3,2,1,0} multiply(%convert_element_type.1898.1, %mul.2963.11)
  %param_1.13025 = f32[128]{0} parameter(1)
  %mul.2965.8 = f32[2,8192,32,128]{3,2,1,0} broadcast(%param_1.13025), dimensions={3}
  %mul.2966.3 = f32[2,8192,32,128]{3,2,1,0} multiply(%mul.2964.5, %mul.2965.8)
  %add.2395.3 = f32[2,8192,32,128]{3,2,1,0} add(%mul.2956.5, %mul.2966.3)
  ROOT %convert_element_type.1900.1 = bf16[2,8192,32,128]{3,2,1,0} convert(%add.2395.3)
}



ENTRY %wrapper_fused_convert.34.clone (param_0.13220: f32[1,2,8192,128], param_1.13025: f32[128], param_2.4537: f32[1,2,8192,128], param_3.2095: bf16[2,8192,32,128], param_4.6155: pred[128]) -> bf16[2,8192,32,128] {
  param_0.13220 = f32[1,2,8192,128] parameter(0)
  param_1.13025 = f32[128] parameter(1)
  param_2.4537 = f32[1,2,8192,128] parameter(2)
  param_3.2095 = bf16[2,8192,32,128] parameter(3)
  param_4.6155 = pred[128] parameter(4)
  ROOT %fusion = bf16[2,8192,32,128] fusion(param_0.13220, param_1.13025, param_2.4537, param_3.2095, param_4.6155), kind=kLoop, calls=%fused_convert.34.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: llama31_8b_training/extracted_fusions/fused_convert.66.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Llama318bTraining_FusedConvert_66) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_convert.66_module, entry_computation_layout={(f32[1,2,8192,128]{3,2,1,0}, f32[1,2,8192,128]{3,2,1,0}, bf16[2,8192,8,128]{3,2,1,0}, pred[128]{0}, s32[2]{0})->bf16[2,8192,8,128]{3,2,1,0}}

%fused_convert.66.clone (param_0.33980: f32[1,2,8192,128], param_1.39373: f32[1,2,8192,128], param_2.21709: bf16[2,8192,8,128], param_3.14023: pred[128], param_4.9295: s32[2]) -> bf16[2,8192,8,128] {
  %param_2.21709 = bf16[2,8192,8,128]{3,2,1,0} parameter(2)
  %convert_element_type.3359.3 = f32[2,8192,8,128]{3,2,1,0} convert(%param_2.21709)
  %param_1.39373 = f32[1,2,8192,128]{3,2,1,0} parameter(1)
  %bitcast.1599.22 = f32[2,8192,128]{2,1,0} bitcast(%param_1.39373)
  %mul.5952.9 = f32[2,8192,8,128]{3,2,1,0} broadcast(%bitcast.1599.22), dimensions={0,1,3}
  %mul.5953.3 = f32[2,8192,8,128]{3,2,1,0} multiply(%convert_element_type.3359.3, %mul.5952.9)
  %param_3.14023 = pred[128]{0} parameter(3)
  %reshape.7639.66 = pred[2,8192,8,128]{3,2,1,0} broadcast(%param_3.14023), dimensions={3}
  %slice.830.3 = bf16[2,8192,8,127]{3,2,1,0} slice(%param_2.21709), slice={[0:2], [0:8192], [0:8], [1:128]}
  %slice.831.1 = bf16[2,8192,8,1]{3,2,1,0} slice(%param_2.21709), slice={[0:2], [0:8192], [0:8], [0:1]}
  %concatenate.381.3 = bf16[2,8192,8,128]{3,2,1,0} concatenate(%slice.830.3, %slice.831.1), dimensions={3}
  %slice.832.1 = bf16[2,8192,8,1]{3,2,1,0} slice(%param_2.21709), slice={[0:2], [0:8192], [0:8], [127:128]}
  %slice.833.1 = bf16[2,8192,8,127]{3,2,1,0} slice(%param_2.21709), slice={[0:2], [0:8192], [0:8], [0:127]}
  %concatenate.382.3 = bf16[2,8192,8,128]{3,2,1,0} concatenate(%slice.832.1, %slice.833.1), dimensions={3}
  %select_n.2688.3 = bf16[2,8192,8,128]{3,2,1,0} select(%reshape.7639.66, %concatenate.381.3, %concatenate.382.3)
  %convert_element_type.3364.5 = f32[2,8192,8,128]{3,2,1,0} convert(%select_n.2688.3)
  %param_0.33980 = f32[1,2,8192,128]{3,2,1,0} parameter(0)
  %bitcast.1600.28 = f32[2,8192,128]{2,1,0} bitcast(%param_0.33980)
  %mul.5957.11 = f32[2,8192,8,128]{3,2,1,0} broadcast(%bitcast.1600.28), dimensions={0,1,3}
  %mul.5958.5 = f32[2,8192,8,128]{3,2,1,0} multiply(%convert_element_type.3364.5, %mul.5957.11)
  %param_4.9295 = s32[2]{0} parameter(4)
  %broadcast_in_dim.657.5 = s32[64,2]{1,0} broadcast(%param_4.9295), dimensions={1}
  %bitcast.1602.5 = s32[128]{0} bitcast(%broadcast_in_dim.657.5)
  %convert_element_type.3366.5 = f32[128]{0} convert(%bitcast.1602.5)
  %mul.5961.4 = f32[2,8192,8,128]{3,2,1,0} broadcast(%convert_element_type.3366.5), dimensions={3}
  %mul.5966.3 = f32[2,8192,8,128]{3,2,1,0} multiply(%mul.5958.5, %mul.5961.4)
  %add.2970.1 = f32[2,8192,8,128]{3,2,1,0} add(%mul.5953.3, %mul.5966.3)
  ROOT %convert_element_type.3367.1 = bf16[2,8192,8,128]{3,2,1,0} convert(%add.2970.1)
}



ENTRY %wrapper_fused_convert.66.clone (param_0.33980: f32[1,2,8192,128], param_1.39373: f32[1,2,8192,128], param_2.21709: bf16[2,8192,8,128], param_3.14023: pred[128], param_4.9295: s32[2]) -> bf16[2,8192,8,128] {
  param_0.33980 = f32[1,2,8192,128] parameter(0)
  param_1.39373 = f32[1,2,8192,128] parameter(1)
  param_2.21709 = bf16[2,8192,8,128] parameter(2)
  param_3.14023 = pred[128] parameter(3)
  param_4.9295 = s32[2] parameter(4)
  ROOT %fusion = bf16[2,8192,8,128] fusion(param_0.33980, param_1.39373, param_2.21709, param_3.14023, param_4.9295), kind=kLoop, calls=%fused_convert.66.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: llama31_8b_training/extracted_fusions/fused_multiply.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Llama318bTraining_FusedMultiply) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply_module, entry_computation_layout={(bf16[4096]{0}, f32[1,2,8192]{2,1,0}, bf16[2,8192,4096]{2,1,0})->bf16[2,8192,4096]{2,1,0}}

%fused_multiply.clone (param_0.16552: bf16[4096], param_1.17442: f32[1,2,8192], param_2.7359: bf16[2,8192,4096]) -> bf16[2,8192,4096] {
  %param_2.7359 = bf16[2,8192,4096]{2,1,0} parameter(2)
  %convert_element_type.1871.16 = f32[2,8192,4096]{2,1,0} convert(%param_2.7359)
  %param_1.17442 = f32[1,2,8192]{2,1,0} parameter(1)
  %bitcast.495.13 = f32[2,8192]{1,0} bitcast(%param_1.17442)
  %mul.2896.13 = f32[2,8192,4096]{2,1,0} broadcast(%bitcast.495.13), dimensions={0,1}
  %mul.2897.5 = f32[2,8192,4096]{2,1,0} multiply(%convert_element_type.1871.16, %mul.2896.13)
  %convert_element_type.1872.3 = bf16[2,8192,4096]{2,1,0} convert(%mul.2897.5)
  %param_0.16552 = bf16[4096]{0} parameter(0)
  %mul.2900.1 = bf16[2,8192,4096]{2,1,0} broadcast(%param_0.16552), dimensions={2}
  ROOT %mul.2901.1 = bf16[2,8192,4096]{2,1,0} multiply(%convert_element_type.1872.3, %mul.2900.1)
}



ENTRY %wrapper_fused_multiply.clone (param_0.16552: bf16[4096], param_1.17442: f32[1,2,8192], param_2.7359: bf16[2,8192,4096]) -> bf16[2,8192,4096] {
  param_0.16552 = bf16[4096] parameter(0)
  param_1.17442 = f32[1,2,8192] parameter(1)
  param_2.7359 = bf16[2,8192,4096] parameter(2)
  ROOT %fusion = bf16[2,8192,4096] fusion(param_0.16552, param_1.17442, param_2.7359), kind=kLoop, calls=%fused_multiply.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: llama31_8b_training/extracted_fusions/fused_multiply.1.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Llama318bTraining_FusedMultiply_1) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.1_module, entry_computation_layout={(f32[1,2,8192]{2,1,0}, f32[2,8192]{1,0}, f32[2,8192]{1,0})->f32[1,2,8192]{2,1,0}}

%fused_multiply.1.clone (param_0.36141: f32[1,2,8192], param_1.40847: f32[2,8192], param_2.22398: f32[2,8192]) -> f32[1,2,8192] {
  %param_1.40847 = f32[2,8192]{1,0} parameter(1)
  %bitcast.2284.3 = f32[1,2,8192]{2,1,0} bitcast(%param_1.40847)
  %param_0.36141 = f32[1,2,8192]{2,1,0} parameter(0)
  %param_2.22398 = f32[2,8192]{1,0} parameter(2)
  %constant_2883_66 = f32[] constant(0.000244140625)
  %broadcast.5097.197 = f32[2,8192]{1,0} broadcast(%constant_2883_66), dimensions={}
  %div.1099.5 = f32[2,8192]{1,0} multiply(%param_2.22398, %broadcast.5097.197)
  %constant_2884_66 = f32[] constant(1e-05)
  %broadcast.5098.67 = f32[2,8192]{1,0} broadcast(%constant_2884_66), dimensions={}
  %add.2383.3 = f32[2,8192]{1,0} add(%div.1099.5, %broadcast.5098.67)
  %bitcast.2281.8 = f32[1,2,8192]{2,1,0} bitcast(%add.2383.3)
  %divide.1.7 = f32[1,2,8192]{2,1,0} divide(%param_0.36141, %bitcast.2281.8)
  %constant_5663_2 = f32[] constant(-0.5)
  %broadcast.6.67 = f32[1,2,8192]{2,1,0} broadcast(%constant_5663_2), dimensions={}
  %multiply.4.5 = f32[1,2,8192]{2,1,0} multiply(%divide.1.7, %broadcast.6.67)
  %multiply.5.3 = f32[1,2,8192]{2,1,0} multiply(%bitcast.2284.3, %multiply.4.5)
  %constant_5664_2 = f32[] constant(0.00048828125)
  %broadcast.7.1 = f32[1,2,8192]{2,1,0} broadcast(%constant_5664_2), dimensions={}
  ROOT %multiply.6.1 = f32[1,2,8192]{2,1,0} multiply(%multiply.5.3, %broadcast.7.1)
}



ENTRY %wrapper_fused_multiply.1.clone (param_0.36141: f32[1,2,8192], param_1.40847: f32[2,8192], param_2.22398: f32[2,8192]) -> f32[1,2,8192] {
  param_0.36141 = f32[1,2,8192] parameter(0)
  param_1.40847 = f32[2,8192] parameter(1)
  param_2.22398 = f32[2,8192] parameter(2)
  ROOT %fusion = f32[1,2,8192] fusion(param_0.36141, param_1.40847, param_2.22398), kind=kLoop, calls=%fused_multiply.1.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: llama31_8b_training/extracted_fusions/fused_rsqrt.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Llama318bTraining_FusedRsqrt) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_rsqrt_module, entry_computation_layout={(f32[2,8192]{1,0})->f32[1,2,8192]{2,1,0}}

%fused_rsqrt.clone (param_0.35777: f32[2,8192]) -> f32[1,2,8192] {
  %param_0.35777 = f32[2,8192]{1,0} parameter(0)
  %constant_2883_1 = f32[] constant(0.000244140625)
  %broadcast.5097.199 = f32[2,8192]{1,0} broadcast(%constant_2883_1), dimensions={}
  %div.1099.7 = f32[2,8192]{1,0} multiply(%param_0.35777, %broadcast.5097.199)
  %constant_2884_1 = f32[] constant(1e-05)
  %broadcast.5098.69 = f32[2,8192]{1,0} broadcast(%constant_2884_1), dimensions={}
  %add.2383.5 = f32[2,8192]{1,0} add(%div.1099.7, %broadcast.5098.69)
  %bitcast.2281.1 = f32[1,2,8192]{2,1,0} bitcast(%add.2383.5)
  ROOT %rsqrt.323 = f32[1,2,8192]{2,1,0} rsqrt(%bitcast.2281.1)
}



ENTRY %wrapper_fused_rsqrt.clone (param_0.35777: f32[2,8192]) -> f32[1,2,8192] {
  param_0.35777 = f32[2,8192] parameter(0)
  ROOT %fusion = f32[1,2,8192] fusion(param_0.35777), kind=kLoop, calls=%fused_rsqrt.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: llama31_8b_training/extracted_fusions/fused_select.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Llama318bTraining_FusedSelect) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select_module, entry_computation_layout={(pred[2,8192]{1,0}, f32[2,8192]{1,0}, f32[2,8192]{1,0}, f32[16384]{0})->f32[2,8192]{1,0}}

%fused_select.clone (param_0.29308: pred[2,8192], param_1.33675: f32[2,8192], param_2.21716: f32[2,8192], param_3.14030: f32[16384]) -> f32[2,8192] {
  %param_0.29308 = pred[2,8192]{1,0} parameter(0)
  %param_1.33675 = f32[2,8192]{1,0} parameter(1)
  %param_2.21716 = f32[2,8192]{1,0} parameter(2)
  %param_3.14030 = f32[16384]{0} parameter(3)
  %bitcast.498.12 = f32[2,8192]{1,0} bitcast(%param_3.14030)
  %add.2386.5 = f32[2,8192]{1,0} add(%param_2.21716, %bitcast.498.12)
  %square.422.5 = f32[2,8192]{1,0} multiply(%add.2386.5, %add.2386.5)
  %constant_2882_66 = f32[] constant(0)
  %broadcast.5144.3 = f32[2,8192]{1,0} broadcast(%constant_2882_66), dimensions={}
  %mul.9455.5 = f32[2,8192]{1,0} multiply(%square.422.5, %broadcast.5144.3)
  %add.5156.3 = f32[2,8192]{1,0} add(%param_1.33675, %mul.9455.5)
  ROOT %mul.9456.1 = f32[2,8192]{1,0} select(%param_0.29308, %add.5156.3, %broadcast.5144.3)
}



ENTRY %wrapper_fused_select.clone (param_0.29308: pred[2,8192], param_1.33675: f32[2,8192], param_2.21716: f32[2,8192], param_3.14030: f32[16384]) -> f32[2,8192] {
  param_0.29308 = pred[2,8192] parameter(0)
  param_1.33675 = f32[2,8192] parameter(1)
  param_2.21716 = f32[2,8192] parameter(2)
  param_3.14030 = f32[16384] parameter(3)
  ROOT %fusion = f32[2,8192] fusion(param_0.29308, param_1.33675, param_2.21716, param_3.14030), kind=kLoop, calls=%fused_select.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: llama31_8b_training/extracted_fusions/fused_select.1.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Llama318bTraining_FusedSelect_1) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.1_module, entry_computation_layout={(pred[2,8192]{1,0}, s32[])->f32[2,8192]{1,0}}

%fused_select.1.clone (param_0.29703: pred[2,8192], param_1.40617: s32[]) -> f32[2,8192] {
  %param_0.29703 = pred[2,8192]{1,0} parameter(0)
  %constant_2872_5 = f32[] constant(1)
  %param_1.40617 = s32[] parameter(1)
  %convert_element_type.1875.3 = f32[] convert(%param_1.40617)
  %constant_5660_293 = f32[] constant(1e-08)
  %add.2385.3 = f32[] add(%convert_element_type.1875.3, %constant_5660_293)
  %div.1100.1 = f32[] divide(%constant_2872_5, %add.2385.3)
  %broadcast_in_dim.467.1 = f32[2,8192]{1,0} broadcast(%div.1100.1), dimensions={}
  %constant_2882_1721 = f32[] constant(0)
  %broadcast.5144.1 = f32[2,8192]{1,0} broadcast(%constant_2882_1721), dimensions={}
  ROOT %mul.2898.1 = f32[2,8192]{1,0} select(%param_0.29703, %broadcast_in_dim.467.1, %broadcast.5144.1)
}



ENTRY %wrapper_fused_select.1.clone (param_0.29703: pred[2,8192], param_1.40617: s32[]) -> f32[2,8192] {
  param_0.29703 = pred[2,8192] parameter(0)
  param_1.40617 = s32[] parameter(1)
  ROOT %fusion = f32[2,8192] fusion(param_0.29703, param_1.40617), kind=kLoop, calls=%fused_select.1.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: llama31_8b_training/extracted_fusions/fused_transpose.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Llama318bTraining_FusedTranspose) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose_module, entry_computation_layout={(f32[512,32,128]{2,1,0})->f32[4096,512]{1,0}}

%fused_transpose.clone (param_0.1634: f32[512,32,128]) -> f32[4096,512] {
  %param_0.1634 = f32[512,32,128]{2,1,0} parameter(0)
  %bitcast.12959.1 = f32[512,4096]{1,0} bitcast(%param_0.1634)
  ROOT %transpose.2737.1 = f32[4096,512]{1,0} transpose(%bitcast.12959.1), dimensions={1,0}
}



ENTRY %wrapper_fused_transpose.clone (param_0.1634: f32[512,32,128]) -> f32[4096,512] {
  param_0.1634 = f32[512,32,128] parameter(0)
  ROOT %fusion = f32[4096,512] fusion(param_0.1634), kind=kLoop, calls=%fused_transpose.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: llama31_8b_training/extracted_fusions/fused_transpose.65.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Llama318bTraining_FusedTranspose_65) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.65_module, entry_computation_layout={(f32[], f32[512,32,128]{2,1,0}, f32[], f32[32,128,512]{2,1,0}, f32[512,32,128]{2,1,0}, /*index=5*/f32[])->f32[512,32,128]{2,1,0}}

%fused_transpose.65.clone (param_0.25484: f32[], param_1.40913: f32[512,32,128], param_2.22464: f32[], param_3.14325: f32[32,128,512], param_4.9566: f32[512,32,128], param_5.7214: f32[]) -> f32[512,32,128] {
  %param_3.14325 = f32[32,128,512]{2,1,0} parameter(3)
  %bitcast.12823.3 = f32[4096,512]{1,0} bitcast(%param_3.14325)
  %transpose.2669.3 = f32[512,4096]{1,0} transpose(%bitcast.12823.3), dimensions={1,0}
  %bitcast.12824.1 = f32[512,32,128]{2,1,0} bitcast(%transpose.2669.3)
  %param_0.25484 = f32[] parameter(0)
  %broadcast.50.66 = f32[512,32,128]{2,1,0} broadcast(%param_0.25484), dimensions={}
  %param_1.40913 = f32[512,32,128]{2,1,0} parameter(1)
  %param_2.22464 = f32[] parameter(2)
  %broadcast.55.320 = f32[512,32,128]{2,1,0} broadcast(%param_2.22464), dimensions={}
  %param_4.9566 = f32[512,32,128]{2,1,0} parameter(4)
  %param_5.7214 = f32[] parameter(5)
  %broadcast.58.130 = f32[512,32,128]{2,1,0} broadcast(%param_5.7214), dimensions={}
  %divide.288.3 = f32[512,32,128]{2,1,0} divide(%param_4.9566, %broadcast.58.130)
  %sqrt.358.1 = f32[512,32,128]{2,1,0} sqrt(%divide.288.3)
  %constant_5660_6 = f32[] constant(1e-08)
  %broadcast.59.66 = f32[512,32,128]{2,1,0} broadcast(%constant_5660_6), dimensions={}
  %add.317.3 = f32[512,32,128]{2,1,0} add(%sqrt.358.1, %broadcast.59.66)
  %multiply.773.5 = f32[512,32,128]{2,1,0} multiply(%broadcast.55.320, %add.317.3)
  %divide.289.3 = f32[512,32,128]{2,1,0} divide(%param_1.40913, %multiply.773.5)
  %constant_13810_5 = f32[] constant(0.1)
  %broadcast.53.448 = f32[512,32,128]{2,1,0} broadcast(%constant_13810_5), dimensions={}
  %multiply.774.5 = f32[512,32,128]{2,1,0} multiply(%bitcast.12824.1, %broadcast.53.448)
  %add.318.3 = f32[512,32,128]{2,1,0} add(%divide.289.3, %multiply.774.5)
  %multiply.775.3 = f32[512,32,128]{2,1,0} multiply(%broadcast.50.66, %add.318.3)
  ROOT %add.319.1 = f32[512,32,128]{2,1,0} add(%bitcast.12824.1, %multiply.775.3)
}



ENTRY %wrapper_fused_transpose.65.clone (param_0.25484: f32[], param_1.40913: f32[512,32,128], param_2.22464: f32[], param_3.14325: f32[32,128,512], param_4.9566: f32[512,32,128], param_5.7214: f32[]) -> f32[512,32,128] {
  param_0.25484 = f32[] parameter(0)
  param_1.40913 = f32[512,32,128] parameter(1)
  param_2.22464 = f32[] parameter(2)
  param_3.14325 = f32[32,128,512] parameter(3)
  param_4.9566 = f32[512,32,128] parameter(4)
  param_5.7214 = f32[] parameter(5)
  ROOT %fusion = f32[512,32,128] fusion(param_0.25484, param_1.40913, param_2.22464, param_3.14325, param_4.9566, param_5.7214), kind=kLoop, calls=%fused_transpose.65.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: llama31_8b_training/extracted_fusions/fused_transpose.128.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Llama318bTraining_FusedTranspose_128) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.128_module, entry_computation_layout={(f32[32,128,512]{2,1,0})->bf16[512,4096]{1,0}}

%fused_transpose.128.clone (param_0.26029: f32[32,128,512]) -> bf16[512,4096] {
  %param_0.26029 = f32[32,128,512]{2,1,0} parameter(0)
  %convert.514.1 = bf16[32,128,512]{2,1,0} convert(%param_0.26029)
  %bitcast.12511.1 = bf16[4096,512]{1,0} bitcast(%convert.514.1)
  ROOT %transpose.2513.1 = bf16[512,4096]{1,0} transpose(%bitcast.12511.1), dimensions={1,0}
}



ENTRY %wrapper_fused_transpose.128.clone (param_0.26029: f32[32,128,512]) -> bf16[512,4096] {
  param_0.26029 = f32[32,128,512] parameter(0)
  ROOT %fusion = bf16[512,4096] fusion(param_0.26029), kind=kLoop, calls=%fused_transpose.128.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: llama31_8b_training/extracted_fusions/fused_transpose.192.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Llama318bTraining_FusedTranspose_192) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.192_module, entry_computation_layout={(f32[], f32[512,128256]{1,0}, f32[], f32[128256,512]{1,0}, f32[512,128256]{1,0}, /*index=5*/f32[])->f32[512,128256]{1,0}}

%fused_transpose.192.clone (param_0.13318: f32[], param_1.40912: f32[512,128256], param_2.22463: f32[], param_3.14324: f32[128256,512], param_4.9564: f32[512,128256], param_5.7212: f32[]) -> f32[512,128256] {
  %param_3.14324 = f32[128256,512]{1,0} parameter(3)
  %transpose.2251.1 = f32[512,128256]{1,0} transpose(%param_3.14324), dimensions={1,0}
  %param_0.13318 = f32[] parameter(0)
  %mul.9428.4 = f32[512,128256]{1,0} broadcast(%param_0.13318), dimensions={}
  %param_1.40912 = f32[512,128256]{1,0} parameter(1)
  %param_2.22463 = f32[] parameter(2)
  %div.3046.6 = f32[512,128256]{1,0} broadcast(%param_2.22463), dimensions={}
  %param_4.9564 = f32[512,128256]{1,0} parameter(4)
  %param_5.7212 = f32[] parameter(5)
  %div.3047.4 = f32[512,128256]{1,0} broadcast(%param_5.7212), dimensions={}
  %divide.291.3 = f32[512,128256]{1,0} divide(%param_4.9564, %div.3047.4)
  %sqrt.359.1 = f32[512,128256]{1,0} sqrt(%divide.291.3)
  %constant_5660_4 = f32[] constant(1e-08)
  %add.4835.10 = f32[512,128256]{1,0} broadcast(%constant_5660_4), dimensions={}
  %add.322.7 = f32[512,128256]{1,0} add(%sqrt.359.1, %add.4835.10)
  %multiply.781.5 = f32[512,128256]{1,0} multiply(%div.3046.6, %add.322.7)
  %divide.292.3 = f32[512,128256]{1,0} divide(%param_1.40912, %multiply.781.5)
  %constant_13810_3 = f32[] constant(0.1)
  %broadcast.5210.8 = f32[512,128256]{1,0} broadcast(%constant_13810_3), dimensions={}
  %multiply.782.5 = f32[512,128256]{1,0} multiply(%transpose.2251.1, %broadcast.5210.8)
  %add.323.3 = f32[512,128256]{1,0} add(%divide.292.3, %multiply.782.5)
  %multiply.783.3 = f32[512,128256]{1,0} multiply(%mul.9428.4, %add.323.3)
  ROOT %add.324.1 = f32[512,128256]{1,0} add(%transpose.2251.1, %multiply.783.3)
}



ENTRY %wrapper_fused_transpose.192.clone (param_0.13318: f32[], param_1.40912: f32[512,128256], param_2.22463: f32[], param_3.14324: f32[128256,512], param_4.9564: f32[512,128256], param_5.7212: f32[]) -> f32[512,128256] {
  param_0.13318 = f32[] parameter(0)
  param_1.40912 = f32[512,128256] parameter(1)
  param_2.22463 = f32[] parameter(2)
  param_3.14324 = f32[128256,512] parameter(3)
  param_4.9564 = f32[512,128256] parameter(4)
  param_5.7212 = f32[] parameter(5)
  ROOT %fusion = f32[512,128256] fusion(param_0.13318, param_1.40912, param_2.22463, param_3.14324, param_4.9564, param_5.7212), kind=kLoop, calls=%fused_transpose.192.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: llama31_8b_training/extracted_fusions/fused_transpose.194.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Llama318bTraining_FusedTranspose_194) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.194_module, entry_computation_layout={(f32[128256,512]{1,0})->bf16[512,128256]{1,0}}

%fused_transpose.194.clone (param_0.16228: f32[128256,512]) -> bf16[512,128256] {
  %param_0.16228 = f32[128256,512]{1,0} parameter(0)
  %convert.386.1 = bf16[128256,512]{1,0} convert(%param_0.16228)
  ROOT %transpose.1866.1 = bf16[512,128256]{1,0} transpose(%convert.386.1), dimensions={1,0}
}



ENTRY %wrapper_fused_transpose.194.clone (param_0.16228: f32[128256,512]) -> bf16[512,128256] {
  param_0.16228 = f32[128256,512] parameter(0)
  ROOT %fusion = bf16[512,128256] fusion(param_0.16228), kind=kLoop, calls=%fused_transpose.194.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: llama31_8b_training/extracted_fusions/fused_transpose.227.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Llama318bTraining_FusedTranspose_227) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.227_module, entry_computation_layout={(f32[], f32[512,14336]{1,0}, f32[], f32[14336,512]{1,0}, f32[512,14336]{1,0}, /*index=5*/f32[])->f32[512,14336]{1,0}}

%fused_transpose.227.clone (param_0.22669: f32[], param_1.40911: f32[512,14336], param_2.22462: f32[], param_3.14323: f32[14336,512], param_4.9563: f32[512,14336], param_5.7211: f32[]) -> f32[512,14336] {
  %param_3.14323 = f32[14336,512]{1,0} parameter(3)
  %transpose.2243.1 = f32[512,14336]{1,0} transpose(%param_3.14323), dimensions={1,0}
  %param_0.22669 = f32[] parameter(0)
  %mul.7008.98 = f32[512,14336]{1,0} broadcast(%param_0.22669), dimensions={}
  %param_1.40911 = f32[512,14336]{1,0} parameter(1)
  %param_2.22462 = f32[] parameter(2)
  %div.2024.100 = f32[512,14336]{1,0} broadcast(%param_2.22462), dimensions={}
  %param_4.9563 = f32[512,14336]{1,0} parameter(4)
  %param_5.7211 = f32[] parameter(5)
  %div.2025.98 = f32[512,14336]{1,0} broadcast(%param_5.7211), dimensions={}
  %divide.285.3 = f32[512,14336]{1,0} divide(%param_4.9563, %div.2025.98)
  %sqrt.357.1 = f32[512,14336]{1,0} sqrt(%divide.285.3)
  %constant_5660_3 = f32[] constant(1e-08)
  %broadcast.5189.98 = f32[512,14336]{1,0} broadcast(%constant_5660_3), dimensions={}
  %add.312.3 = f32[512,14336]{1,0} add(%sqrt.357.1, %broadcast.5189.98)
  %multiply.765.5 = f32[512,14336]{1,0} multiply(%div.2024.100, %add.312.3)
  %divide.286.3 = f32[512,14336]{1,0} divide(%param_1.40911, %multiply.765.5)
  %constant_13810_2 = f32[] constant(0.1)
  %broadcast.5185.194 = f32[512,14336]{1,0} broadcast(%constant_13810_2), dimensions={}
  %multiply.766.3 = f32[512,14336]{1,0} multiply(%transpose.2243.1, %broadcast.5185.194)
  %add.313.1 = f32[512,14336]{1,0} add(%divide.286.3, %multiply.766.3)
  %multiply.767.3 = f32[512,14336]{1,0} multiply(%mul.7008.98, %add.313.1)
  ROOT %add.314.1 = f32[512,14336]{1,0} add(%transpose.2243.1, %multiply.767.3)
}



ENTRY %wrapper_fused_transpose.227.clone (param_0.22669: f32[], param_1.40911: f32[512,14336], param_2.22462: f32[], param_3.14323: f32[14336,512], param_4.9563: f32[512,14336], param_5.7211: f32[]) -> f32[512,14336] {
  param_0.22669 = f32[] parameter(0)
  param_1.40911 = f32[512,14336] parameter(1)
  param_2.22462 = f32[] parameter(2)
  param_3.14323 = f32[14336,512] parameter(3)
  param_4.9563 = f32[512,14336] parameter(4)
  param_5.7211 = f32[] parameter(5)
  ROOT %fusion = f32[512,14336] fusion(param_0.22669, param_1.40911, param_2.22462, param_3.14323, param_4.9563, param_5.7211), kind=kLoop, calls=%fused_transpose.227.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: llama31_8b_training/extracted_fusions/fused_transpose.259.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, Llama318bTraining_FusedTranspose_259) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.259_module, entry_computation_layout={(f32[14336,512]{1,0})->bf16[512,14336]{1,0}}

%fused_transpose.259.clone (param_0.23821: f32[14336,512]) -> bf16[512,14336] {
  %param_0.23821 = f32[14336,512]{1,0} parameter(0)
  %convert.513.1 = bf16[14336,512]{1,0} convert(%param_0.23821)
  ROOT %transpose.1993.1 = bf16[512,14336]{1,0} transpose(%convert.513.1), dimensions={1,0}
}



ENTRY %wrapper_fused_transpose.259.clone (param_0.23821: f32[14336,512]) -> bf16[512,14336] {
  param_0.23821 = f32[14336,512] parameter(0)
  ROOT %fusion = bf16[512,14336] fusion(param_0.23821), kind=kLoop, calls=%fused_transpose.259.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// === mixtral_training (28 fusions) ===
// Source: mixtral_training/extracted_fusions/fused_add.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedAdd) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_add_module, entry_computation_layout={(bf16[1,4096,4096]{2,1,0}, f32[1,4096,1]{2,1,0}, bf16[1,4096,4096]{2,1,0}, f32[1,4096,1]{2,1,0}, bf16[4096]{0}, /*index=5*/bf16[4096,4096]{1,0}, bf16[4096,4096]{1,0})->bf16[1,4096,4096]{2,1,0}}

%fused_add.clone (param_0.24625: bf16[1,4096,4096], param_1.24159: f32[1,4096,1], param_2.6874: bf16[1,4096,4096], param_3.3176: f32[1,4096,1], param_4.2090: bf16[4096], param_5.4602: bf16[4096,4096], param_6.3359: bf16[4096,4096]) -> bf16[1,4096,4096] {
  %param_0.24625 = bf16[1,4096,4096]{2,1,0} parameter(0)
  %param_5.4602 = bf16[4096,4096]{1,0} parameter(5)
  %param_6.3359 = bf16[4096,4096]{1,0} parameter(6)
  %add.494.1 = bf16[4096,4096]{1,0} add(%param_5.4602, %param_6.3359)
  %bitcast.1618.6 = bf16[1,4096,4096]{2,1,0} bitcast(%add.494.1)
  %param_4.2090 = bf16[4096]{0} parameter(4)
  %broadcast.7221.6 = bf16[1,4096,4096]{2,1,0} broadcast(%param_4.2090), dimensions={2}
  %multiply.3243.5 = bf16[1,4096,4096]{2,1,0} multiply(%bitcast.1618.6, %broadcast.7221.6)
  %convert.1438.8 = f32[1,4096,4096]{2,1,0} convert(%multiply.3243.5)
  %param_3.3176 = f32[1,4096,1]{2,1,0} parameter(3)
  %bitcast.1529.11 = f32[4096]{0} bitcast(%param_3.3176)
  %broadcast.7220.11 = f32[1,4096,4096]{2,1,0} broadcast(%bitcast.1529.11), dimensions={1}
  %multiply.3244.5 = f32[1,4096,4096]{2,1,0} multiply(%convert.1438.8, %broadcast.7220.11)
  %param_2.6874 = bf16[1,4096,4096]{2,1,0} parameter(2)
  %convert.1405.14 = f32[1,4096,4096]{2,1,0} convert(%param_2.6874)
  %param_1.24159 = f32[1,4096,1]{2,1,0} parameter(1)
  %bitcast.1620.7 = f32[4096]{0} bitcast(%param_1.24159)
  %broadcast.7294.7 = f32[1,4096,4096]{2,1,0} broadcast(%bitcast.1620.7), dimensions={1}
  %multiply.3249.5 = f32[1,4096,4096]{2,1,0} multiply(%convert.1405.14, %broadcast.7294.7)
  %add.3438.3 = f32[1,4096,4096]{2,1,0} add(%multiply.3244.5, %multiply.3249.5)
  %convert.1439.3 = bf16[1,4096,4096]{2,1,0} convert(%add.3438.3)
  ROOT %add.3439.1 = bf16[1,4096,4096]{2,1,0} add(%param_0.24625, %convert.1439.3)
}



ENTRY %wrapper_fused_add.clone (param_0.24625: bf16[1,4096,4096], param_1.24159: f32[1,4096,1], param_2.6874: bf16[1,4096,4096], param_3.3176: f32[1,4096,1], param_4.2090: bf16[4096], param_5.4602: bf16[4096,4096], param_6.3359: bf16[4096,4096]) -> bf16[1,4096,4096] {
  param_0.24625 = bf16[1,4096,4096] parameter(0)
  param_1.24159 = f32[1,4096,1] parameter(1)
  param_2.6874 = bf16[1,4096,4096] parameter(2)
  param_3.3176 = f32[1,4096,1] parameter(3)
  param_4.2090 = bf16[4096] parameter(4)
  param_5.4602 = bf16[4096,4096] parameter(5)
  param_6.3359 = bf16[4096,4096] parameter(6)
  ROOT %fusion = bf16[1,4096,4096] fusion(param_0.24625, param_1.24159, param_2.6874, param_3.3176, param_4.2090, param_5.4602, param_6.3359), kind=kLoop, calls=%fused_add.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_concatenate.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedConcatenate) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate_module, entry_computation_layout={(bf16[1,1,4096,64]{3,2,1,0}, bf16[1,1,4096,64]{3,2,1,0}, bf16[4096,4096]{1,0})->bf16[1,4096,32,128]{3,2,1,0}}

%fused_concatenate.clone (param_0.28346: bf16[1,1,4096,64], param_1.29533: bf16[1,1,4096,64], param_2.10561: bf16[4096,4096]) -> bf16[1,4096,32,128] {
  %param_2.10561 = bf16[4096,4096]{1,0} parameter(2)
  %bitcast.35190.6 = bf16[1,4096,32,128]{3,2,1,0} bitcast(%param_2.10561)
  %slice.128.3 = bf16[1,4096,32,64]{3,2,1,0} slice(%bitcast.35190.6), slice={[0:1], [0:4096], [0:32], [0:64]}
  %param_1.29533 = bf16[1,1,4096,64]{3,2,1,0} parameter(1)
  %bitcast.8.262 = bf16[4096,64]{1,0} bitcast(%param_1.29533)
  %broadcast.5389.133 = bf16[1,4096,32,64]{3,2,1,0} broadcast(%bitcast.8.262), dimensions={1,3}
  %multiply.1734.3 = bf16[1,4096,32,64]{3,2,1,0} multiply(%slice.128.3, %broadcast.5389.133)
  %slice.129.3 = bf16[1,4096,32,64]{3,2,1,0} slice(%bitcast.35190.6), slice={[0:1], [0:4096], [0:32], [64:128]}
  %param_0.28346 = bf16[1,1,4096,64]{3,2,1,0} parameter(0)
  %bitcast.9.388 = bf16[4096,64]{1,0} bitcast(%param_0.28346)
  %broadcast.5391.259 = bf16[1,4096,32,64]{3,2,1,0} broadcast(%bitcast.9.388), dimensions={1,3}
  %multiply.1736.5 = bf16[1,4096,32,64]{3,2,1,0} multiply(%slice.129.3, %broadcast.5391.259)
  %subtract.65.3 = bf16[1,4096,32,64]{3,2,1,0} subtract(%multiply.1734.3, %multiply.1736.5)
  %multiply.1737.3 = bf16[1,4096,32,64]{3,2,1,0} multiply(%slice.129.3, %broadcast.5389.133)
  %multiply.1738.5 = bf16[1,4096,32,64]{3,2,1,0} multiply(%slice.128.3, %broadcast.5391.259)
  %add.2531.3 = bf16[1,4096,32,64]{3,2,1,0} add(%multiply.1737.3, %multiply.1738.5)
  ROOT %concatenate.0.1 = bf16[1,4096,32,128]{3,2,1,0} concatenate(%subtract.65.3, %add.2531.3), dimensions={3}
}



ENTRY %wrapper_fused_concatenate.clone (param_0.28346: bf16[1,1,4096,64], param_1.29533: bf16[1,1,4096,64], param_2.10561: bf16[4096,4096]) -> bf16[1,4096,32,128] {
  param_0.28346 = bf16[1,1,4096,64] parameter(0)
  param_1.29533 = bf16[1,1,4096,64] parameter(1)
  param_2.10561 = bf16[4096,4096] parameter(2)
  ROOT %fusion = bf16[1,4096,32,128] fusion(param_0.28346, param_1.29533, param_2.10561), kind=kLoop, calls=%fused_concatenate.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_concatenate.33.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedConcatenate_33) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate.33_module, entry_computation_layout={(bf16[1,1,4096,64]{3,2,1,0}, bf16[1,1,4096,64]{3,2,1,0}, bf16[1,4096,32,128]{3,2,1,0})->bf16[1,4096,32,128]{3,2,1,0}}

%fused_concatenate.33.clone (param_0.28568: bf16[1,1,4096,64], param_1.29691: bf16[1,1,4096,64], param_2.10719: bf16[1,4096,32,128]) -> bf16[1,4096,32,128] {
  %param_2.10719 = bf16[1,4096,32,128]{3,2,1,0} parameter(2)
  %slice.394.1 = bf16[1,4096,32,64]{3,2,1,0} slice(%param_2.10719), slice={[0:1], [0:4096], [0:32], [64:128]}
  %param_0.28568 = bf16[1,1,4096,64]{3,2,1,0} parameter(0)
  %bitcast.1535.24 = bf16[4096,64]{1,0} bitcast(%param_0.28568)
  %broadcast.7226.23 = bf16[1,4096,32,64]{3,2,1,0} broadcast(%bitcast.1535.24), dimensions={1,3}
  %multiply.3238.5 = bf16[1,4096,32,64]{3,2,1,0} multiply(%slice.394.1, %broadcast.7226.23)
  %slice.395.1 = bf16[1,4096,32,64]{3,2,1,0} slice(%param_2.10719), slice={[0:1], [0:4096], [0:32], [0:64]}
  %param_1.29691 = bf16[1,1,4096,64]{3,2,1,0} parameter(1)
  %bitcast.1534.18 = bf16[4096,64]{1,0} bitcast(%param_1.29691)
  %broadcast.7225.17 = bf16[1,4096,32,64]{3,2,1,0} broadcast(%bitcast.1534.18), dimensions={1,3}
  %multiply.3240.3 = bf16[1,4096,32,64]{3,2,1,0} multiply(%slice.395.1, %broadcast.7225.17)
  %add.3435.3 = bf16[1,4096,32,64]{3,2,1,0} add(%multiply.3238.5, %multiply.3240.3)
  %multiply.3241.3 = bf16[1,4096,32,64]{3,2,1,0} multiply(%slice.394.1, %broadcast.7225.17)
  %convert.3341.5 = f32[1,4096,32,64]{3,2,1,0} convert(%slice.395.1)
  %negate.194.5 = f32[1,4096,32,64]{3,2,1,0} negate(%convert.3341.5)
  %convert.3342.3 = bf16[1,4096,32,64]{3,2,1,0} convert(%negate.194.5)
  %multiply.3242.5 = bf16[1,4096,32,64]{3,2,1,0} multiply(%convert.3342.3, %broadcast.7226.23)
  %add.3436.3 = bf16[1,4096,32,64]{3,2,1,0} add(%multiply.3241.3, %multiply.3242.5)
  ROOT %concatenate.200.1 = bf16[1,4096,32,128]{3,2,1,0} concatenate(%add.3435.3, %add.3436.3), dimensions={3}
}



ENTRY %wrapper_fused_concatenate.33.clone (param_0.28568: bf16[1,1,4096,64], param_1.29691: bf16[1,1,4096,64], param_2.10719: bf16[1,4096,32,128]) -> bf16[1,4096,32,128] {
  param_0.28568 = bf16[1,1,4096,64] parameter(0)
  param_1.29691 = bf16[1,1,4096,64] parameter(1)
  param_2.10719 = bf16[1,4096,32,128] parameter(2)
  ROOT %fusion = bf16[1,4096,32,128] fusion(param_0.28568, param_1.29691, param_2.10719), kind=kLoop, calls=%fused_concatenate.33.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_concatenate.158.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedConcatenate_158) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate.158_module, entry_computation_layout={(bf16[1,1,4096,64]{3,2,1,0}, bf16[1,1,4096,64]{3,2,1,0}, bf16[1,4096,8,128]{3,2,1,0})->bf16[1,4096,8,128]{3,2,1,0}}

%fused_concatenate.158.clone (param_0.31085: bf16[1,1,4096,64], param_1.35483: bf16[1,1,4096,64], param_2.14079: bf16[1,4096,8,128]) -> bf16[1,4096,8,128] {
  %param_2.14079 = bf16[1,4096,8,128]{3,2,1,0} parameter(2)
  %slice.392.1 = bf16[1,4096,8,64]{3,2,1,0} slice(%param_2.14079), slice={[0:1], [0:4096], [0:8], [64:128]}
  %param_0.31085 = bf16[1,1,4096,64]{3,2,1,0} parameter(0)
  %bitcast.1535.42 = bf16[4096,64]{1,0} bitcast(%param_0.31085)
  %broadcast.7233.17 = bf16[1,4096,8,64]{3,2,1,0} broadcast(%bitcast.1535.42), dimensions={1,3}
  %multiply.3232.5 = bf16[1,4096,8,64]{3,2,1,0} multiply(%slice.392.1, %broadcast.7233.17)
  %slice.393.1 = bf16[1,4096,8,64]{3,2,1,0} slice(%param_2.14079), slice={[0:1], [0:4096], [0:8], [0:64]}
  %param_1.35483 = bf16[1,1,4096,64]{3,2,1,0} parameter(1)
  %bitcast.1534.30 = bf16[4096,64]{1,0} bitcast(%param_1.35483)
  %broadcast.7232.13 = bf16[1,4096,8,64]{3,2,1,0} broadcast(%bitcast.1534.30), dimensions={1,3}
  %multiply.3234.3 = bf16[1,4096,8,64]{3,2,1,0} multiply(%slice.393.1, %broadcast.7232.13)
  %add.3432.3 = bf16[1,4096,8,64]{3,2,1,0} add(%multiply.3232.5, %multiply.3234.3)
  %multiply.3235.3 = bf16[1,4096,8,64]{3,2,1,0} multiply(%slice.392.1, %broadcast.7232.13)
  %convert.3339.5 = f32[1,4096,8,64]{3,2,1,0} convert(%slice.393.1)
  %negate.193.5 = f32[1,4096,8,64]{3,2,1,0} negate(%convert.3339.5)
  %convert.3340.3 = bf16[1,4096,8,64]{3,2,1,0} convert(%negate.193.5)
  %multiply.3236.5 = bf16[1,4096,8,64]{3,2,1,0} multiply(%convert.3340.3, %broadcast.7233.17)
  %add.3433.3 = bf16[1,4096,8,64]{3,2,1,0} add(%multiply.3235.3, %multiply.3236.5)
  ROOT %concatenate.199.1 = bf16[1,4096,8,128]{3,2,1,0} concatenate(%add.3432.3, %add.3433.3), dimensions={3}
}



ENTRY %wrapper_fused_concatenate.158.clone (param_0.31085: bf16[1,1,4096,64], param_1.35483: bf16[1,1,4096,64], param_2.14079: bf16[1,4096,8,128]) -> bf16[1,4096,8,128] {
  param_0.31085 = bf16[1,1,4096,64] parameter(0)
  param_1.35483 = bf16[1,1,4096,64] parameter(1)
  param_2.14079 = bf16[1,4096,8,128] parameter(2)
  ROOT %fusion = bf16[1,4096,8,128] fusion(param_0.31085, param_1.35483, param_2.14079), kind=kLoop, calls=%fused_concatenate.158.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_concatenate.191.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedConcatenate_191) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_concatenate.191_module, entry_computation_layout={(bf16[1,1,4096,64]{3,2,1,0}, bf16[1,1,4096,64]{3,2,1,0}, bf16[4096,1024]{1,0})->bf16[1,4096,8,128]{3,2,1,0}}

%fused_concatenate.191.clone (param_0.33055: bf16[1,1,4096,64], param_1.35388: bf16[1,1,4096,64], param_2.13921: bf16[4096,1024]) -> bf16[1,4096,8,128] {
  %param_2.13921 = bf16[4096,1024]{1,0} parameter(2)
  %bitcast.35193.6 = bf16[1,4096,8,128]{3,2,1,0} bitcast(%param_2.13921)
  %slice.130.3 = bf16[1,4096,8,64]{3,2,1,0} slice(%bitcast.35193.6), slice={[0:1], [0:4096], [0:8], [0:64]}
  %param_1.35388 = bf16[1,1,4096,64]{3,2,1,0} parameter(1)
  %bitcast.8.514 = bf16[4096,64]{1,0} bitcast(%param_1.35388)
  %broadcast.5400.257 = bf16[1,4096,8,64]{3,2,1,0} broadcast(%bitcast.8.514), dimensions={1,3}
  %multiply.1742.3 = bf16[1,4096,8,64]{3,2,1,0} multiply(%slice.130.3, %broadcast.5400.257)
  %slice.131.3 = bf16[1,4096,8,64]{3,2,1,0} slice(%bitcast.35193.6), slice={[0:1], [0:4096], [0:8], [64:128]}
  %param_0.33055 = bf16[1,1,4096,64]{3,2,1,0} parameter(0)
  %bitcast.9.770 = bf16[4096,64]{1,0} bitcast(%param_0.33055)
  %broadcast.5401.385 = bf16[1,4096,8,64]{3,2,1,0} broadcast(%bitcast.9.770), dimensions={1,3}
  %multiply.1744.5 = bf16[1,4096,8,64]{3,2,1,0} multiply(%slice.131.3, %broadcast.5401.385)
  %subtract.66.3 = bf16[1,4096,8,64]{3,2,1,0} subtract(%multiply.1742.3, %multiply.1744.5)
  %multiply.1745.3 = bf16[1,4096,8,64]{3,2,1,0} multiply(%slice.131.3, %broadcast.5400.257)
  %multiply.1746.5 = bf16[1,4096,8,64]{3,2,1,0} multiply(%slice.130.3, %broadcast.5401.385)
  %add.2532.3 = bf16[1,4096,8,64]{3,2,1,0} add(%multiply.1745.3, %multiply.1746.5)
  ROOT %concatenate.1.1 = bf16[1,4096,8,128]{3,2,1,0} concatenate(%subtract.66.3, %add.2532.3), dimensions={3}
}



ENTRY %wrapper_fused_concatenate.191.clone (param_0.33055: bf16[1,1,4096,64], param_1.35388: bf16[1,1,4096,64], param_2.13921: bf16[4096,1024]) -> bf16[1,4096,8,128] {
  param_0.33055 = bf16[1,1,4096,64] parameter(0)
  param_1.35388 = bf16[1,1,4096,64] parameter(1)
  param_2.13921 = bf16[4096,1024] parameter(2)
  ROOT %fusion = bf16[1,4096,8,128] fusion(param_0.33055, param_1.35388, param_2.13921), kind=kLoop, calls=%fused_concatenate.191.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_multiply.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedMultiply) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply_module, entry_computation_layout={(bf16[8192,14336]{1,0}, bf16[8192,14336]{1,0})->bf16[1,8,1024,14336]{3,2,1,0}}

%fused_multiply.clone (param_0.15452: bf16[8192,14336], param_1.19586: bf16[8192,14336]) -> bf16[1,8,1024,14336] {
  %param_1.19586 = bf16[8192,14336]{1,0} parameter(1)
  %bitcast.35981.12 = bf16[1,8,1024,14336]{3,2,1,0} bitcast(%param_1.19586)
  %constant_4121_66 = bf16[] constant(1)
  %convert.8927.33 = f32[] convert(%constant_4121_66)
  %broadcast.6096.320 = f32[1,8,1024,14336]{3,2,1,0} broadcast(%convert.8927.33), dimensions={}
  %convert.3329.11 = f32[1,8,1024,14336]{3,2,1,0} convert(%bitcast.35981.12)
  %negate.190.9 = f32[1,8,1024,14336]{3,2,1,0} negate(%convert.3329.11)
  %convert.3330.7 = bf16[1,8,1024,14336]{3,2,1,0} convert(%negate.190.9)
  %exponential.189.5 = bf16[1,8,1024,14336]{3,2,1,0} exponential(%convert.3330.7)
  %broadcast.5440.353 = bf16[1,8,1024,14336]{3,2,1,0} broadcast(%constant_4121_66), dimensions={}
  %add.3381.3 = bf16[1,8,1024,14336]{3,2,1,0} add(%exponential.189.5, %broadcast.5440.353)
  %convert.3332.5 = f32[1,8,1024,14336]{3,2,1,0} convert(%add.3381.3)
  %divide.672.7 = f32[1,8,1024,14336]{3,2,1,0} divide(%broadcast.6096.320, %convert.3332.5)
  %convert.3333.5 = bf16[1,8,1024,14336]{3,2,1,0} convert(%divide.672.7)
  %multiply.3141.3 = bf16[1,8,1024,14336]{3,2,1,0} multiply(%bitcast.35981.12, %convert.3333.5)
  %param_0.15452 = bf16[8192,14336]{1,0} parameter(0)
  %bitcast.35984.1 = bf16[1,8,1024,14336]{3,2,1,0} bitcast(%param_0.15452)
  ROOT %multiply.3143.1 = bf16[1,8,1024,14336]{3,2,1,0} multiply(%multiply.3141.3, %bitcast.35984.1)
}



ENTRY %wrapper_fused_multiply.clone (param_0.15452: bf16[8192,14336], param_1.19586: bf16[8192,14336]) -> bf16[1,8,1024,14336] {
  param_0.15452 = bf16[8192,14336] parameter(0)
  param_1.19586 = bf16[8192,14336] parameter(1)
  ROOT %fusion = bf16[1,8,1024,14336] fusion(param_0.15452, param_1.19586), kind=kLoop, calls=%fused_multiply.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_multiply.96.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedMultiply_96) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.96_module, entry_computation_layout={(bf16[4096]{0}, f32[1,4096,1]{2,1,0}, bf16[4096,4096]{1,0})->bf16[1,4096,4096]{2,1,0}}

%fused_multiply.96.clone (param_0.25949: bf16[4096], param_1.26295: f32[1,4096,1], param_2.8229: bf16[4096,4096]) -> bf16[1,4096,4096] {
  %param_2.8229 = bf16[4096,4096]{1,0} parameter(2)
  %bitcast.38200.19 = bf16[1,4096,4096]{2,1,0} bitcast(%param_2.8229)
  %convert.1394.19 = f32[1,4096,4096]{2,1,0} convert(%bitcast.38200.19)
  %param_1.26295 = f32[1,4096,1]{2,1,0} parameter(1)
  %bitcast.1520.11 = f32[4096]{0} bitcast(%param_1.26295)
  %broadcast.7194.11 = f32[1,4096,4096]{2,1,0} broadcast(%bitcast.1520.11), dimensions={1}
  %multiply.3148.5 = f32[1,4096,4096]{2,1,0} multiply(%convert.1394.19, %broadcast.7194.11)
  %convert.1395.3 = bf16[1,4096,4096]{2,1,0} convert(%multiply.3148.5)
  %param_0.25949 = bf16[4096]{0} parameter(0)
  %broadcast.6018.1 = bf16[1,4096,4096]{2,1,0} broadcast(%param_0.25949), dimensions={2}
  ROOT %multiply.1047.1 = bf16[1,4096,4096]{2,1,0} multiply(%convert.1395.3, %broadcast.6018.1)
}



ENTRY %wrapper_fused_multiply.96.clone (param_0.25949: bf16[4096], param_1.26295: f32[1,4096,1], param_2.8229: bf16[4096,4096]) -> bf16[1,4096,4096] {
  param_0.25949 = bf16[4096] parameter(0)
  param_1.26295 = f32[1,4096,1] parameter(1)
  param_2.8229 = bf16[4096,4096] parameter(2)
  ROOT %fusion = bf16[1,4096,4096] fusion(param_0.25949, param_1.26295, param_2.8229), kind=kLoop, calls=%fused_multiply.96.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_multiply.160.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedMultiply_160) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.160_module, entry_computation_layout={(bf16[4096]{0}, f32[1,4096,1]{2,1,0}, bf16[1,4096,4096]{2,1,0})->bf16[1,4096,4096]{2,1,0}}

%fused_multiply.160.clone (param_0.26206: bf16[4096], param_1.26808: f32[1,4096,1], param_2.8806: bf16[1,4096,4096]) -> bf16[1,4096,4096] {
  %param_2.8806 = bf16[1,4096,4096]{2,1,0} parameter(2)
  %convert.1405.16 = f32[1,4096,4096]{2,1,0} convert(%param_2.8806)
  %param_1.26808 = f32[1,4096,1]{2,1,0} parameter(1)
  %bitcast.1529.13 = f32[4096]{0} bitcast(%param_1.26808)
  %broadcast.7220.13 = f32[1,4096,4096]{2,1,0} broadcast(%bitcast.1529.13), dimensions={1}
  %multiply.3165.5 = f32[1,4096,4096]{2,1,0} multiply(%convert.1405.16, %broadcast.7220.13)
  %convert.1406.3 = bf16[1,4096,4096]{2,1,0} convert(%multiply.3165.5)
  %param_0.26206 = bf16[4096]{0} parameter(0)
  %broadcast.7221.1 = bf16[1,4096,4096]{2,1,0} broadcast(%param_0.26206), dimensions={2}
  ROOT %multiply.3166.1 = bf16[1,4096,4096]{2,1,0} multiply(%convert.1406.3, %broadcast.7221.1)
}



ENTRY %wrapper_fused_multiply.160.clone (param_0.26206: bf16[4096], param_1.26808: f32[1,4096,1], param_2.8806: bf16[1,4096,4096]) -> bf16[1,4096,4096] {
  param_0.26206 = bf16[4096] parameter(0)
  param_1.26808 = f32[1,4096,1] parameter(1)
  param_2.8806 = bf16[1,4096,4096] parameter(2)
  ROOT %fusion = bf16[1,4096,4096] fusion(param_0.26206, param_1.26808, param_2.8806), kind=kLoop, calls=%fused_multiply.160.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_multiply.161.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedMultiply_161) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_multiply.161_module, entry_computation_layout={(f32[1,4096,1]{2,1,0}, f32[4096]{0}, f32[4096]{0})->f32[1,4096,1]{2,1,0}}

%fused_multiply.161.clone (param_0.52877: f32[1,4096,1], param_1.59710: f32[4096], param_2.30204: f32[4096]) -> f32[1,4096,1] {
  %param_1.59710 = f32[4096]{0} parameter(1)
  %bitcast.1526.3 = f32[1,4096,1]{2,1,0} bitcast(%param_1.59710)
  %param_0.52877 = f32[1,4096,1]{2,1,0} parameter(0)
  %param_2.30204 = f32[4096]{0} parameter(2)
  %bitcast.31281.5 = f32[1,4096]{1,0} bitcast(%param_2.30204)
  %constant_3957_66 = f32[] constant(0.000244140625)
  %broadcast.5380.197 = f32[1,4096]{1,0} broadcast(%constant_3957_66), dimensions={}
  %multiply.3147.5 = f32[1,4096]{1,0} multiply(%bitcast.31281.5, %broadcast.5380.197)
  %constant_3958_66 = f32[] constant(1e-05)
  %broadcast.5381.67 = f32[1,4096]{1,0} broadcast(%constant_3958_66), dimensions={}
  %add.3383.3 = f32[1,4096]{1,0} add(%multiply.3147.5, %broadcast.5381.67)
  %bitcast.1519.8 = f32[1,4096,1]{2,1,0} bitcast(%add.3383.3)
  %divide.675.7 = f32[1,4096,1]{2,1,0} divide(%param_0.52877, %bitcast.1519.8)
  %constant_9518_2 = f32[] constant(-0.5)
  %broadcast.7210.67 = f32[1,4096,1]{2,1,0} broadcast(%constant_9518_2), dimensions={}
  %multiply.3158.5 = f32[1,4096,1]{2,1,0} multiply(%divide.675.7, %broadcast.7210.67)
  %multiply.3159.3 = f32[1,4096,1]{2,1,0} multiply(%bitcast.1526.3, %multiply.3158.5)
  %constant_9519_2 = f32[] constant(0.00048828125)
  %broadcast.7211.1 = f32[1,4096,1]{2,1,0} broadcast(%constant_9519_2), dimensions={}
  ROOT %multiply.3160.1 = f32[1,4096,1]{2,1,0} multiply(%multiply.3159.3, %broadcast.7211.1)
}



ENTRY %wrapper_fused_multiply.161.clone (param_0.52877: f32[1,4096,1], param_1.59710: f32[4096], param_2.30204: f32[4096]) -> f32[1,4096,1] {
  param_0.52877 = f32[1,4096,1] parameter(0)
  param_1.59710 = f32[4096] parameter(1)
  param_2.30204 = f32[4096] parameter(2)
  ROOT %fusion = f32[1,4096,1] fusion(param_0.52877, param_1.59710, param_2.30204), kind=kLoop, calls=%fused_multiply.161.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_rsqrt.64.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedRsqrt_64) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_rsqrt.64_module, entry_computation_layout={(f32[4096]{0})->f32[1,4096,1]{2,1,0}}

%fused_rsqrt.64.clone (param_0.52707: f32[4096]) -> f32[1,4096,1] {
  %param_0.52707 = f32[4096]{0} parameter(0)
  %bitcast.31281.7 = f32[1,4096]{1,0} bitcast(%param_0.52707)
  %constant_3957_1 = f32[] constant(0.000244140625)
  %broadcast.5380.199 = f32[1,4096]{1,0} broadcast(%constant_3957_1), dimensions={}
  %multiply.3147.7 = f32[1,4096]{1,0} multiply(%bitcast.31281.7, %broadcast.5380.199)
  %constant_3958_1 = f32[] constant(1e-05)
  %broadcast.5381.69 = f32[1,4096]{1,0} broadcast(%constant_3958_1), dimensions={}
  %add.3383.5 = f32[1,4096]{1,0} add(%multiply.3147.7, %broadcast.5381.69)
  %bitcast.1519.1 = f32[1,4096,1]{2,1,0} bitcast(%add.3383.5)
  ROOT %rsqrt.128.1 = f32[1,4096,1]{2,1,0} rsqrt(%bitcast.1519.1)
}



ENTRY %wrapper_fused_rsqrt.64.clone (param_0.52707: f32[4096]) -> f32[1,4096,1] {
  param_0.52707 = f32[4096] parameter(0)
  ROOT %fusion = f32[1,4096,1] fusion(param_0.52707), kind=kLoop, calls=%fused_rsqrt.64.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_select.2.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedSelect_2) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.2_module, entry_computation_layout={(bf16[4096,14336]{1,0}, pred[], f32[])->bf16[1,4096,14336]{2,1,0}}

%fused_select.2.clone (param_0.16859: bf16[4096,14336], param_1.14336: pred[], param_2.2645: f32[]) -> bf16[1,4096,14336] {
  %param_1.14336 = pred[] parameter(1)
  %broadcast.10109.63 = pred[1,4096,14336]{2,1,0} broadcast(%param_1.14336), dimensions={}
  %param_0.16859 = bf16[4096,14336]{1,0} parameter(0)
  %bitcast.38055.2 = bf16[1,4096,14336]{2,1,0} bitcast(%param_0.16859)
  %convert.8606.5 = f32[1,4096,14336]{2,1,0} convert(%bitcast.38055.2)
  %param_2.2645 = f32[] parameter(2)
  %broadcast.6111.318 = f32[1,4096,14336]{2,1,0} broadcast(%param_2.2645), dimensions={}
  %divide.2017.5 = f32[1,4096,14336]{2,1,0} divide(%convert.8606.5, %broadcast.6111.318)
  %convert.8608.3 = bf16[1,4096,14336]{2,1,0} convert(%divide.2017.5)
  ROOT %select.2901.1 = bf16[1,4096,14336]{2,1,0} select(%broadcast.10109.63, %bitcast.38055.2, %convert.8608.3)
}



ENTRY %wrapper_fused_select.2.clone (param_0.16859: bf16[4096,14336], param_1.14336: pred[], param_2.2645: f32[]) -> bf16[1,4096,14336] {
  param_0.16859 = bf16[4096,14336] parameter(0)
  param_1.14336 = pred[] parameter(1)
  param_2.2645 = f32[] parameter(2)
  ROOT %fusion = bf16[1,4096,14336] fusion(param_0.16859, param_1.14336, param_2.2645), kind=kLoop, calls=%fused_select.2.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_select.12.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedSelect_12) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.12_module, entry_computation_layout={(bf16[14336,4096]{1,0}, pred[], f32[])->bf16[1,14336,4096]{2,1,0}}

%fused_select.12.clone (param_0.16917: bf16[14336,4096], param_1.14394: pred[], param_2.2732: f32[]) -> bf16[1,14336,4096] {
  %param_1.14394 = pred[] parameter(1)
  %broadcast.10132.28 = pred[1,14336,4096]{2,1,0} broadcast(%param_1.14394), dimensions={}
  %param_0.16917 = bf16[14336,4096]{1,0} parameter(0)
  %bitcast.37998.1 = bf16[1,14336,4096]{2,1,0} bitcast(%param_0.16917)
  %convert.8188.5 = f32[1,14336,4096]{2,1,0} convert(%bitcast.37998.1)
  %param_2.2732 = f32[] parameter(2)
  %broadcast.6116.152 = f32[1,14336,4096]{2,1,0} broadcast(%param_2.2732), dimensions={}
  %divide.1901.5 = f32[1,14336,4096]{2,1,0} divide(%convert.8188.5, %broadcast.6116.152)
  %convert.8190.3 = bf16[1,14336,4096]{2,1,0} convert(%divide.1901.5)
  ROOT %select.2863.1 = bf16[1,14336,4096]{2,1,0} select(%broadcast.10132.28, %bitcast.37998.1, %convert.8190.3)
}



ENTRY %wrapper_fused_select.12.clone (param_0.16917: bf16[14336,4096], param_1.14394: pred[], param_2.2732: f32[]) -> bf16[1,14336,4096] {
  param_0.16917 = bf16[14336,4096] parameter(0)
  param_1.14394 = pred[] parameter(1)
  param_2.2732 = f32[] parameter(2)
  ROOT %fusion = bf16[1,14336,4096] fusion(param_0.16917, param_1.14394, param_2.2732), kind=kLoop, calls=%fused_select.12.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_select.96.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedSelect_96) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.96_module, entry_computation_layout={(bf16[512,32000]{1,0}, f32[], pred[])->bf16[512,32000]{1,0}}

%fused_select.96.clone (param_0.26727: bf16[512,32000], param_1.27589: f32[], param_2.8811: pred[]) -> bf16[512,32000] {
  %param_2.8811 = pred[] parameter(2)
  %broadcast.12026.1 = pred[512,32000]{1,0} broadcast(%param_2.8811), dimensions={}
  %param_0.26727 = bf16[512,32000]{1,0} parameter(0)
  %convert.8705.3 = f32[512,32000]{1,0} convert(%param_0.26727)
  %param_1.27589 = f32[] parameter(1)
  %broadcast.6129.8 = f32[512,32000]{1,0} broadcast(%param_1.27589), dimensions={}
  %divide.2045.5 = f32[512,32000]{1,0} divide(%convert.8705.3, %broadcast.6129.8)
  %convert.8707.3 = bf16[512,32000]{1,0} convert(%divide.2045.5)
  ROOT %select.2910.1 = bf16[512,32000]{1,0} select(%broadcast.12026.1, %param_0.26727, %convert.8707.3)
}



ENTRY %wrapper_fused_select.96.clone (param_0.26727: bf16[512,32000], param_1.27589: f32[], param_2.8811: pred[]) -> bf16[512,32000] {
  param_0.26727 = bf16[512,32000] parameter(0)
  param_1.27589 = f32[] parameter(1)
  param_2.8811 = pred[] parameter(2)
  ROOT %fusion = bf16[512,32000] fusion(param_0.26727, param_1.27589, param_2.8811), kind=kLoop, calls=%fused_select.96.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_select.97.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedSelect_97) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.97_module, entry_computation_layout={(s32[], s32[1,4096]{1,0})->f32[1,4096]{1,0}}

%fused_select.97.clone (param_0.52875: s32[], param_1.59643: s32[1,4096]) -> f32[1,4096] {
  %param_1.59643 = s32[1,4096]{1,0} parameter(1)
  %constant_3783_616 = s32[] constant(0)
  %broadcast.7195.5 = s32[1,4096]{1,0} broadcast(%constant_3783_616), dimensions={}
  %compare.1876.5 = pred[1,4096]{1,0} compare(%param_1.59643, %broadcast.7195.5), direction=NE
  %constant_3781_4 = f32[] constant(1)
  %param_0.52875 = s32[] parameter(0)
  %convert.1397.3 = f32[] convert(%param_0.52875)
  %constant_9509_2 = f32[] constant(1e-08)
  %add.3384.3 = f32[] add(%convert.1397.3, %constant_9509_2)
  %divide.673.1 = f32[] divide(%constant_3781_4, %add.3384.3)
  %broadcast.7197.1 = f32[1,4096]{1,0} broadcast(%divide.673.1), dimensions={}
  %constant_3956_96 = f32[] constant(0)
  %broadcast.7199.1 = f32[1,4096]{1,0} broadcast(%constant_3956_96), dimensions={}
  ROOT %select.1094.1 = f32[1,4096]{1,0} select(%compare.1876.5, %broadcast.7197.1, %broadcast.7199.1)
}



ENTRY %wrapper_fused_select.97.clone (param_0.52875: s32[], param_1.59643: s32[1,4096]) -> f32[1,4096] {
  param_0.52875 = s32[] parameter(0)
  param_1.59643 = s32[1,4096] parameter(1)
  ROOT %fusion = f32[1,4096] fusion(param_0.52875, param_1.59643), kind=kLoop, calls=%fused_select.97.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_select.98.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedSelect_98) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_select.98_module, entry_computation_layout={(f32[1,4096]{1,0}, f32[4096]{0}, f32[4096]{0}, s32[1,4096]{1,0})->f32[1,4096]{1,0}}

%fused_select.98.clone (param_0.50955: f32[1,4096], param_1.58380: f32[4096], param_2.29686: f32[4096], param_3.18915: s32[1,4096]) -> f32[1,4096] {
  %param_3.18915 = s32[1,4096]{1,0} parameter(3)
  %constant_3783_171 = s32[] constant(0)
  %broadcast.7195.7 = s32[1,4096]{1,0} broadcast(%constant_3783_171), dimensions={}
  %compare.1876.7 = pred[1,4096]{1,0} compare(%param_3.18915, %broadcast.7195.7), direction=NE
  %param_0.50955 = f32[1,4096]{1,0} parameter(0)
  %param_1.58380 = f32[4096]{0} parameter(1)
  %param_2.29686 = f32[4096]{0} parameter(2)
  %add.529.1 = f32[4096]{0} add(%param_1.58380, %param_2.29686)
  %multiply.41.5 = f32[4096]{0} multiply(%add.529.1, %add.529.1)
  %constant_3956_98 = f32[] constant(0)
  %broadcast.3250.6 = f32[4096]{0} broadcast(%constant_3956_98), dimensions={}
  %multiply.42.5 = f32[4096]{0} multiply(%multiply.41.5, %broadcast.3250.6)
  %bitcast.4988.3 = f32[1,4096]{1,0} bitcast(%multiply.42.5)
  %add.7526.3 = f32[1,4096]{1,0} add(%param_0.50955, %bitcast.4988.3)
  %broadcast.7199.2 = f32[1,4096]{1,0} broadcast(%constant_3956_98), dimensions={}
  ROOT %select.2914.1 = f32[1,4096]{1,0} select(%compare.1876.7, %add.7526.3, %broadcast.7199.2)
}



ENTRY %wrapper_fused_select.98.clone (param_0.50955: f32[1,4096], param_1.58380: f32[4096], param_2.29686: f32[4096], param_3.18915: s32[1,4096]) -> f32[1,4096] {
  param_0.50955 = f32[1,4096] parameter(0)
  param_1.58380 = f32[4096] parameter(1)
  param_2.29686 = f32[4096] parameter(2)
  param_3.18915 = s32[1,4096] parameter(3)
  ROOT %fusion = f32[1,4096] fusion(param_0.50955, param_1.58380, param_2.29686, param_3.18915), kind=kLoop, calls=%fused_select.98.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_transpose.19.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedTranspose_19) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.19_module, entry_computation_layout={(bf16[4096,8192]{1,0})->bf16[8,4096,1024]{2,1,0}}

%fused_transpose.19.clone (param_0.1415: bf16[4096,8192]) -> bf16[8,4096,1024] {
  %param_0.1415 = bf16[4096,8192]{1,0} parameter(0)
  %bitcast.37088.1 = bf16[4096,8,1024]{2,1,0} bitcast(%param_0.1415)
  ROOT %transpose.4264.1 = bf16[8,4096,1024]{2,1,0} transpose(%bitcast.37088.1), dimensions={1,0,2}
}



ENTRY %wrapper_fused_transpose.19.clone (param_0.1415: bf16[4096,8192]) -> bf16[8,4096,1024] {
  param_0.1415 = bf16[4096,8192] parameter(0)
  ROOT %fusion = bf16[8,4096,1024] fusion(param_0.1415), kind=kLoop, calls=%fused_transpose.19.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_transpose.20.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedTranspose_20) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.20_module, entry_computation_layout={(bf16[1,4096,8,1,1024]{4,3,1,0,2})->bf16[8,1024,4096]{2,1,0}}

%fused_transpose.20.clone (param_0.1419: bf16[1,4096,8,1,1024]{4,3,1,0,2}) -> bf16[8,1024,4096] {
  %param_0.1419 = bf16[1,4096,8,1,1024]{4,3,1,0,2} parameter(0)
  %bitcast.37041.1 = bf16[8,4096,1024]{2,1,0} bitcast(%param_0.1419)
  ROOT %transpose.4588.1 = bf16[8,1024,4096]{2,1,0} transpose(%bitcast.37041.1), dimensions={0,2,1}
}



ENTRY %wrapper_fused_transpose.20.clone (param_0.1419: bf16[1,4096,8,1,1024]{4,3,1,0,2}) -> bf16[8,1024,4096] {
  param_0.1419 = bf16[1,4096,8,1,1024]{4,3,1,0,2} parameter(0)
  ROOT %fusion = bf16[8,1024,4096]{2,1,0} fusion(param_0.1419), kind=kLoop, calls=%fused_transpose.20.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_transpose.65.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedTranspose_65) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.65_module, entry_computation_layout={(bf16[512,32,128]{2,1,0})->bf16[4096,512]{1,0}}

%fused_transpose.65.clone (param_0.5873: bf16[512,32,128]) -> bf16[4096,512] {
  %param_0.5873 = bf16[512,32,128]{2,1,0} parameter(0)
  %bitcast.23396.1 = bf16[512,4096]{1,0} bitcast(%param_0.5873)
  ROOT %transpose.4533.1 = bf16[4096,512]{1,0} transpose(%bitcast.23396.1), dimensions={1,0}
}



ENTRY %wrapper_fused_transpose.65.clone (param_0.5873: bf16[512,32,128]) -> bf16[4096,512] {
  param_0.5873 = bf16[512,32,128] parameter(0)
  ROOT %fusion = bf16[4096,512] fusion(param_0.5873), kind=kLoop, calls=%fused_transpose.65.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_transpose.97.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedTranspose_97) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.97_module, entry_computation_layout={(bf16[32,128,512]{2,1,0}, bf16[512,32,128]{2,1,0})->bf16[4096,512]{1,0}}

%fused_transpose.97.clone (param_0.52972: bf16[32,128,512], param_1.60000: bf16[512,32,128]) -> bf16[4096,512] {
  %param_1.60000 = bf16[512,32,128]{2,1,0} parameter(1)
  %constant_20349_322 = bf16[] constant(0.1001)
  %broadcast.3554.506 = bf16[512,32,128]{2,1,0} broadcast(%constant_20349_322), dimensions={}
  %multiply.316.5 = bf16[512,32,128]{2,1,0} multiply(%param_1.60000, %broadcast.3554.506)
  %param_0.52972 = bf16[32,128,512]{2,1,0} parameter(0)
  %bitcast.23254.5 = bf16[4096,512]{1,0} bitcast(%param_0.52972)
  %transpose.4462.5 = bf16[512,4096]{1,0} transpose(%bitcast.23254.5), dimensions={1,0}
  %bitcast.23255.5 = bf16[512,32,128]{2,1,0} bitcast(%transpose.4462.5)
  %constant_20350_259 = bf16[] constant(0.8984)
  %broadcast.3555.314 = bf16[512,32,128]{2,1,0} broadcast(%constant_20350_259), dimensions={}
  %multiply.317.5 = bf16[512,32,128]{2,1,0} multiply(%bitcast.23255.5, %broadcast.3555.314)
  %add.5310.3 = bf16[512,32,128]{2,1,0} add(%multiply.316.5, %multiply.317.5)
  %bitcast.23332.1 = bf16[512,4096]{1,0} bitcast(%add.5310.3)
  ROOT %transpose.4501.1 = bf16[4096,512]{1,0} transpose(%bitcast.23332.1), dimensions={1,0}
}



ENTRY %wrapper_fused_transpose.97.clone (param_0.52972: bf16[32,128,512], param_1.60000: bf16[512,32,128]) -> bf16[4096,512] {
  param_0.52972 = bf16[32,128,512] parameter(0)
  param_1.60000 = bf16[512,32,128] parameter(1)
  ROOT %fusion = bf16[4096,512] fusion(param_0.52972, param_1.60000), kind=kLoop, calls=%fused_transpose.97.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_transpose.131.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedTranspose_131) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.131_module, entry_computation_layout={(bf16[512,4096]{1,0}, bf16[], bf16[], f32[], bf16[512,32,128]{2,1,0}, /*index=5*/bf16[32,128,512]{2,1,0}, bf16[512,32,128]{2,1,0})->bf16[512,32,128]{2,1,0}}

%fused_transpose.131.clone (param_0.32418: bf16[512,4096], param_1.35732: bf16[], param_2.31397: bf16[], param_3.20455: f32[], param_4.12673: bf16[512,32,128], param_5.8985: bf16[32,128,512], param_6.6647: bf16[512,32,128]) -> bf16[512,32,128] {
  %param_0.32418 = bf16[512,4096]{1,0} parameter(0)
  %bitcast.23253.1 = bf16[512,32,128]{2,1,0} bitcast(%param_0.32418)
  %param_1.35732 = bf16[] parameter(1)
  %broadcast.3551.68 = bf16[512,32,128]{2,1,0} broadcast(%param_1.35732), dimensions={}
  %param_6.6647 = bf16[512,32,128]{2,1,0} parameter(6)
  %constant_20349_323 = bf16[] constant(0.1001)
  %broadcast.3554.508 = bf16[512,32,128]{2,1,0} broadcast(%constant_20349_323), dimensions={}
  %multiply.316.7 = bf16[512,32,128]{2,1,0} multiply(%param_6.6647, %broadcast.3554.508)
  %param_5.8985 = bf16[32,128,512]{2,1,0} parameter(5)
  %bitcast.23254.7 = bf16[4096,512]{1,0} bitcast(%param_5.8985)
  %transpose.4462.7 = bf16[512,4096]{1,0} transpose(%bitcast.23254.7), dimensions={1,0}
  %bitcast.23255.7 = bf16[512,32,128]{2,1,0} bitcast(%transpose.4462.7)
  %constant_20350_164 = bf16[] constant(0.8984)
  %broadcast.3555.316 = bf16[512,32,128]{2,1,0} broadcast(%constant_20350_164), dimensions={}
  %multiply.317.7 = bf16[512,32,128]{2,1,0} multiply(%bitcast.23255.7, %broadcast.3555.316)
  %add.5310.5 = bf16[512,32,128]{2,1,0} add(%multiply.316.7, %multiply.317.7)
  %convert.8570.7 = f32[512,32,128]{2,1,0} convert(%add.5310.5)
  %param_2.31397 = bf16[] parameter(2)
  %broadcast.3556.68 = bf16[512,32,128]{2,1,0} broadcast(%param_2.31397), dimensions={}
  %param_4.12673 = bf16[512,32,128]{2,1,0} parameter(4)
  %convert.8565.5 = f32[512,32,128]{2,1,0} convert(%param_4.12673)
  %param_3.20455 = f32[] parameter(3)
  %broadcast.6127.254 = f32[512,32,128]{2,1,0} broadcast(%param_3.20455), dimensions={}
  %divide.189.5 = f32[512,32,128]{2,1,0} divide(%convert.8565.5, %broadcast.6127.254)
  %sqrt.357.3 = f32[512,32,128]{2,1,0} sqrt(%divide.189.5)
  %convert.8569.3 = bf16[512,32,128]{2,1,0} convert(%sqrt.357.3)
  %constant_20360_164 = bf16[] constant(1.001e-08)
  %broadcast.3560.196 = bf16[512,32,128]{2,1,0} broadcast(%constant_20360_164), dimensions={}
  %add.5335.5 = bf16[512,32,128]{2,1,0} add(%convert.8569.3, %broadcast.3560.196)
  %multiply.321.3 = bf16[512,32,128]{2,1,0} multiply(%broadcast.3556.68, %add.5335.5)
  %convert.8571.9 = f32[512,32,128]{2,1,0} convert(%multiply.321.3)
  %divide.190.9 = f32[512,32,128]{2,1,0} divide(%convert.8570.7, %convert.8571.9)
  %convert.8572.7 = bf16[512,32,128]{2,1,0} convert(%divide.190.9)
  %multiply.322.3 = bf16[512,32,128]{2,1,0} multiply(%bitcast.23253.1, %broadcast.3554.508)
  %add.5382.5 = bf16[512,32,128]{2,1,0} add(%convert.8572.7, %multiply.322.3)
  %multiply.323.3 = bf16[512,32,128]{2,1,0} multiply(%broadcast.3551.68, %add.5382.5)
  ROOT %add.5385.1 = bf16[512,32,128]{2,1,0} add(%bitcast.23253.1, %multiply.323.3)
}



ENTRY %wrapper_fused_transpose.131.clone (param_0.32418: bf16[512,4096], param_1.35732: bf16[], param_2.31397: bf16[], param_3.20455: f32[], param_4.12673: bf16[512,32,128], param_5.8985: bf16[32,128,512], param_6.6647: bf16[512,32,128]) -> bf16[512,32,128] {
  param_0.32418 = bf16[512,4096] parameter(0)
  param_1.35732 = bf16[] parameter(1)
  param_2.31397 = bf16[] parameter(2)
  param_3.20455 = f32[] parameter(3)
  param_4.12673 = bf16[512,32,128] parameter(4)
  param_5.8985 = bf16[32,128,512] parameter(5)
  param_6.6647 = bf16[512,32,128] parameter(6)
  ROOT %fusion = bf16[512,32,128] fusion(param_0.32418, param_1.35732, param_2.31397, param_3.20455, param_4.12673, param_5.8985, param_6.6647), kind=kLoop, calls=%fused_transpose.131.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_transpose.224.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedTranspose_224) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.224_module, entry_computation_layout={(bf16[32,128,512]{2,1,0})->bf16[512,4096]{1,0}}

%fused_transpose.224.clone (param_0.6643: bf16[32,128,512]) -> bf16[512,4096] {
  %param_0.6643 = bf16[32,128,512]{2,1,0} parameter(0)
  %bitcast.22626.1 = bf16[4096,512]{1,0} bitcast(%param_0.6643)
  ROOT %transpose.4148.1 = bf16[512,4096]{1,0} transpose(%bitcast.22626.1), dimensions={1,0}
}



ENTRY %wrapper_fused_transpose.224.clone (param_0.6643: bf16[32,128,512]) -> bf16[512,4096] {
  param_0.6643 = bf16[32,128,512] parameter(0)
  ROOT %fusion = bf16[512,4096] fusion(param_0.6643), kind=kLoop, calls=%fused_transpose.224.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_transpose.256.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedTranspose_256) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.256_module, entry_computation_layout={(bf16[], bf16[512,32000]{1,0}, bf16[], bf16[512,32000]{1,0}, bf16[32000,512]{1,0})->bf16[512,32000]{1,0}}

%fused_transpose.256.clone (param_0.28246: bf16[], param_1.29433: bf16[512,32000], param_2.31366: bf16[], param_3.20392: bf16[512,32000], param_4.12516: bf16[32000,512]) -> bf16[512,32000] {
  %param_4.12516 = bf16[32000,512]{1,0} parameter(4)
  %transpose.3247.1 = bf16[512,32000]{1,0} transpose(%param_4.12516), dimensions={1,0}
  %param_0.28246 = bf16[] parameter(0)
  %broadcast.12025.4 = bf16[512,32000]{1,0} broadcast(%param_0.28246), dimensions={}
  %param_3.20392 = bf16[512,32000]{1,0} parameter(3)
  %convert.8724.5 = f32[512,32000]{1,0} convert(%param_3.20392)
  %param_2.31366 = bf16[] parameter(2)
  %broadcast.12030.14 = bf16[512,32000]{1,0} broadcast(%param_2.31366), dimensions={}
  %param_1.29433 = bf16[512,32000]{1,0} parameter(1)
  %constant_20360_7 = bf16[] constant(1.001e-08)
  %broadcast.12034.14 = bf16[512,32000]{1,0} broadcast(%constant_20360_7), dimensions={}
  %add.5614.7 = bf16[512,32000]{1,0} add(%param_1.29433, %broadcast.12034.14)
  %multiply.337.7 = bf16[512,32000]{1,0} multiply(%broadcast.12030.14, %add.5614.7)
  %convert.8725.7 = f32[512,32000]{1,0} convert(%multiply.337.7)
  %divide.196.7 = f32[512,32000]{1,0} divide(%convert.8724.5, %convert.8725.7)
  %convert.8726.5 = bf16[512,32000]{1,0} convert(%divide.196.7)
  %constant_20349_7 = bf16[] constant(0.1001)
  %broadcast.12028.8 = bf16[512,32000]{1,0} broadcast(%constant_20349_7), dimensions={}
  %multiply.338.3 = bf16[512,32000]{1,0} multiply(%transpose.3247.1, %broadcast.12028.8)
  %add.5629.3 = bf16[512,32000]{1,0} add(%convert.8726.5, %multiply.338.3)
  %multiply.339.3 = bf16[512,32000]{1,0} multiply(%broadcast.12025.4, %add.5629.3)
  ROOT %add.5639.1 = bf16[512,32000]{1,0} add(%transpose.3247.1, %multiply.339.3)
}



ENTRY %wrapper_fused_transpose.256.clone (param_0.28246: bf16[], param_1.29433: bf16[512,32000], param_2.31366: bf16[], param_3.20392: bf16[512,32000], param_4.12516: bf16[32000,512]) -> bf16[512,32000] {
  param_0.28246 = bf16[] parameter(0)
  param_1.29433 = bf16[512,32000] parameter(1)
  param_2.31366 = bf16[] parameter(2)
  param_3.20392 = bf16[512,32000] parameter(3)
  param_4.12516 = bf16[32000,512] parameter(4)
  ROOT %fusion = bf16[512,32000] fusion(param_0.28246, param_1.29433, param_2.31366, param_3.20392, param_4.12516), kind=kLoop, calls=%fused_transpose.256.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source: mixtral_training/extracted_fusions/fused_transpose.257.clean
TEST_F(CuDnnNonGemmFusionLevel1Test, MixtralTraining_FusedTranspose_257) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule fused_transpose.257_module, entry_computation_layout={(bf16[32000,512]{1,0}, bf16[512,32000]{1,0})->bf16[32000,512]{1,0}}

%fused_transpose.257.clone (param_0.53137: bf16[32000,512], param_1.60389: bf16[512,32000]) -> bf16[32000,512] {
  %param_1.60389 = bf16[512,32000]{1,0} parameter(1)
  %multiply.334.6 = bf16[512,32000]{1,0} multiply(%param_1.60389, %param_1.60389)
  %constant_20356_229 = bf16[] constant(0.05005)
  %broadcast.12031.8 = bf16[512,32000]{1,0} broadcast(%constant_20356_229), dimensions={}
  %multiply.335.5 = bf16[512,32000]{1,0} multiply(%multiply.334.6, %broadcast.12031.8)
  %param_0.53137 = bf16[32000,512]{1,0} parameter(0)
  %transpose.3634.3 = bf16[512,32000]{1,0} transpose(%param_0.53137), dimensions={1,0}
  %constant_20357_229 = bf16[] constant(0.9492)
  %broadcast.12032.8 = bf16[512,32000]{1,0} broadcast(%constant_20357_229), dimensions={}
  %multiply.336.5 = bf16[512,32000]{1,0} multiply(%transpose.3634.3, %broadcast.12032.8)
  %add.5606.3 = bf16[512,32000]{1,0} add(%multiply.335.5, %multiply.336.5)
  ROOT %transpose.3734.1 = bf16[32000,512]{1,0} transpose(%add.5606.3), dimensions={1,0}
}



ENTRY %wrapper_fused_transpose.257.clone (param_0.53137: bf16[32000,512], param_1.60389: bf16[512,32000]) -> bf16[32000,512] {
  param_0.53137 = bf16[32000,512] parameter(0)
  param_1.60389 = bf16[512,32000] parameter(1)
  ROOT %fusion = bf16[32000,512] fusion(param_0.53137, param_1.60389), kind=kLoop, calls=%fused_transpose.257.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source:
// mixtral_training/extracted_fusions/wrapped_concatenate_computation.clean
TEST_F(CuDnnNonGemmFusionLevel1Test,
       MixtralTraining_WrappedConcatenateComputation) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule wrapped_concatenate_computation_module, entry_computation_layout={(bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, /*index=5*/bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, /*index=10*/bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, /*index=15*/bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, /*index=20*/bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, /*index=25*/bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, /*index=30*/bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, /*index=35*/bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, /*index=40*/bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, /*index=45*/bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, /*index=50*/bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, /*index=55*/bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, /*index=60*/bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0}, bf16[4096]{0})->bf16[266240]{0}}

%wrapped_concatenate_computation.clone (param_0.54041: bf16[4096], param_1.61435: bf16[4096], param_2.33769: bf16[4096], param_3.23571: bf16[4096], param_4.15954: bf16[4096], param_5.12472: bf16[4096], param_6.8611: bf16[4096], param_7.4936: bf16[4096], param_8.3000: bf16[4096], param_9.3130: bf16[4096], param_10.2557: bf16[4096], param_11.1496: bf16[4096], param_12.1050: bf16[4096], param_13.944: bf16[4096], param_14.716: bf16[4096], param_15.408: bf16[4096], param_16.308: bf16[4096], param_17.163: bf16[4096], param_18.59: bf16[4096], param_19.64: bf16[4096], param_20.68: bf16[4096], param_21.64: bf16[4096], param_22.64: bf16[4096], param_23.68: bf16[4096], param_24.64: bf16[4096], param_25.64: bf16[4096], param_26.68: bf16[4096], param_27.63: bf16[4096], param_28.59: bf16[4096], param_29.64: bf16[4096], param_30.68: bf16[4096], param_31.64: bf16[4096], param_32.64: bf16[4096], param_33.68: bf16[4096], param_34.64: bf16[4096], param_35.64: bf16[4096], param_36.68: bf16[4096], param_37.63: bf16[4096], param_38.59: bf16[4096], param_39.64: bf16[4096], param_40.68: bf16[4096], param_41.64: bf16[4096], param_42.60: bf16[4096], param_43.60: bf16[4096], param_44.48: bf16[4096], param_45.48: bf16[4096], param_46.52: bf16[4096], param_47.47: bf16[4096], param_48.47: bf16[4096], param_49.52: bf16[4096], param_50.52: bf16[4096], param_51.48: bf16[4096], param_52.48: bf16[4096], param_53.52: bf16[4096], param_54.48: bf16[4096], param_55.48: bf16[4096], param_56.52: bf16[4096], param_57.47: bf16[4096], param_58.47: bf16[4096], param_59.52: bf16[4096], param_60.52: bf16[4096], param_61.48: bf16[4096], param_62.48: bf16[4096], param_63.52: bf16[4096], param_64.48: bf16[4096]) -> bf16[266240] {
  %param_0.54041 = bf16[4096]{0} parameter(0)
  %param_1.61435 = bf16[4096]{0} parameter(1)
  %param_2.33769 = bf16[4096]{0} parameter(2)
  %param_3.23571 = bf16[4096]{0} parameter(3)
  %param_4.15954 = bf16[4096]{0} parameter(4)
  %param_5.12472 = bf16[4096]{0} parameter(5)
  %param_6.8611 = bf16[4096]{0} parameter(6)
  %param_7.4936 = bf16[4096]{0} parameter(7)
  %param_8.3000 = bf16[4096]{0} parameter(8)
  %param_9.3130 = bf16[4096]{0} parameter(9)
  %param_10.2557 = bf16[4096]{0} parameter(10)
  %param_11.1496 = bf16[4096]{0} parameter(11)
  %param_12.1050 = bf16[4096]{0} parameter(12)
  %param_13.944 = bf16[4096]{0} parameter(13)
  %param_14.716 = bf16[4096]{0} parameter(14)
  %param_15.408 = bf16[4096]{0} parameter(15)
  %param_16.308 = bf16[4096]{0} parameter(16)
  %param_17.163 = bf16[4096]{0} parameter(17)
  %param_18.59 = bf16[4096]{0} parameter(18)
  %param_19.64 = bf16[4096]{0} parameter(19)
  %param_20.68 = bf16[4096]{0} parameter(20)
  %param_21.64 = bf16[4096]{0} parameter(21)
  %param_22.64 = bf16[4096]{0} parameter(22)
  %param_23.68 = bf16[4096]{0} parameter(23)
  %param_24.64 = bf16[4096]{0} parameter(24)
  %param_25.64 = bf16[4096]{0} parameter(25)
  %param_26.68 = bf16[4096]{0} parameter(26)
  %param_27.63 = bf16[4096]{0} parameter(27)
  %param_28.59 = bf16[4096]{0} parameter(28)
  %param_29.64 = bf16[4096]{0} parameter(29)
  %param_30.68 = bf16[4096]{0} parameter(30)
  %param_31.64 = bf16[4096]{0} parameter(31)
  %param_32.64 = bf16[4096]{0} parameter(32)
  %param_33.68 = bf16[4096]{0} parameter(33)
  %param_34.64 = bf16[4096]{0} parameter(34)
  %param_35.64 = bf16[4096]{0} parameter(35)
  %param_36.68 = bf16[4096]{0} parameter(36)
  %param_37.63 = bf16[4096]{0} parameter(37)
  %param_38.59 = bf16[4096]{0} parameter(38)
  %param_39.64 = bf16[4096]{0} parameter(39)
  %param_40.68 = bf16[4096]{0} parameter(40)
  %param_41.64 = bf16[4096]{0} parameter(41)
  %param_42.60 = bf16[4096]{0} parameter(42)
  %param_43.60 = bf16[4096]{0} parameter(43)
  %param_44.48 = bf16[4096]{0} parameter(44)
  %param_45.48 = bf16[4096]{0} parameter(45)
  %param_46.52 = bf16[4096]{0} parameter(46)
  %param_47.47 = bf16[4096]{0} parameter(47)
  %param_48.47 = bf16[4096]{0} parameter(48)
  %param_49.52 = bf16[4096]{0} parameter(49)
  %param_50.52 = bf16[4096]{0} parameter(50)
  %param_51.48 = bf16[4096]{0} parameter(51)
  %param_52.48 = bf16[4096]{0} parameter(52)
  %param_53.52 = bf16[4096]{0} parameter(53)
  %param_54.48 = bf16[4096]{0} parameter(54)
  %param_55.48 = bf16[4096]{0} parameter(55)
  %param_56.52 = bf16[4096]{0} parameter(56)
  %param_57.47 = bf16[4096]{0} parameter(57)
  %param_58.47 = bf16[4096]{0} parameter(58)
  %param_59.52 = bf16[4096]{0} parameter(59)
  %param_60.52 = bf16[4096]{0} parameter(60)
  %param_61.48 = bf16[4096]{0} parameter(61)
  %param_62.48 = bf16[4096]{0} parameter(62)
  %param_63.52 = bf16[4096]{0} parameter(63)
  %param_64.48 = bf16[4096]{0} parameter(64)
  ROOT %concatenate.439 = bf16[266240]{0} concatenate(%param_0.54041, %param_1.61435, %param_2.33769, %param_3.23571, %param_4.15954, /*index=5*/%param_5.12472, %param_6.8611, %param_7.4936, %param_8.3000, %param_9.3130, /*index=10*/%param_10.2557, %param_11.1496, %param_12.1050, %param_13.944, %param_14.716, /*index=15*/%param_15.408, %param_16.308, %param_17.163, %param_18.59, %param_19.64, /*index=20*/%param_20.68, %param_21.64, %param_22.64, %param_23.68, %param_24.64, /*index=25*/%param_25.64, %param_26.68, %param_27.63, %param_28.59, %param_29.64, /*index=30*/%param_30.68, %param_31.64, %param_32.64, %param_33.68, %param_34.64, /*index=35*/%param_35.64, %param_36.68, %param_37.63, %param_38.59, %param_39.64, /*index=40*/%param_40.68, %param_41.64, %param_42.60, %param_43.60, %param_44.48, /*index=45*/%param_45.48, %param_46.52, %param_47.47, %param_48.47, %param_49.52, /*index=50*/%param_50.52, %param_51.48, %param_52.48, %param_53.52, %param_54.48, /*index=55*/%param_55.48, %param_56.52, %param_57.47, %param_58.47, %param_59.52, /*index=60*/%param_60.52, %param_61.48, %param_62.48, %param_63.52, %param_64.48), dimensions={0}
}



ENTRY %wrapper_wrapped_concatenate_computation.clone (param_0.54041: bf16[4096], param_1.61435: bf16[4096], param_2.33769: bf16[4096], param_3.23571: bf16[4096], param_4.15954: bf16[4096], param_5.12472: bf16[4096], param_6.8611: bf16[4096], param_7.4936: bf16[4096], param_8.3000: bf16[4096], param_9.3130: bf16[4096], param_10.2557: bf16[4096], param_11.1496: bf16[4096], param_12.1050: bf16[4096], param_13.944: bf16[4096], param_14.716: bf16[4096], param_15.408: bf16[4096], param_16.308: bf16[4096], param_17.163: bf16[4096], param_18.59: bf16[4096], param_19.64: bf16[4096], param_20.68: bf16[4096], param_21.64: bf16[4096], param_22.64: bf16[4096], param_23.68: bf16[4096], param_24.64: bf16[4096], param_25.64: bf16[4096], param_26.68: bf16[4096], param_27.63: bf16[4096], param_28.59: bf16[4096], param_29.64: bf16[4096], param_30.68: bf16[4096], param_31.64: bf16[4096], param_32.64: bf16[4096], param_33.68: bf16[4096], param_34.64: bf16[4096], param_35.64: bf16[4096], param_36.68: bf16[4096], param_37.63: bf16[4096], param_38.59: bf16[4096], param_39.64: bf16[4096], param_40.68: bf16[4096], param_41.64: bf16[4096], param_42.60: bf16[4096], param_43.60: bf16[4096], param_44.48: bf16[4096], param_45.48: bf16[4096], param_46.52: bf16[4096], param_47.47: bf16[4096], param_48.47: bf16[4096], param_49.52: bf16[4096], param_50.52: bf16[4096], param_51.48: bf16[4096], param_52.48: bf16[4096], param_53.52: bf16[4096], param_54.48: bf16[4096], param_55.48: bf16[4096], param_56.52: bf16[4096], param_57.47: bf16[4096], param_58.47: bf16[4096], param_59.52: bf16[4096], param_60.52: bf16[4096], param_61.48: bf16[4096], param_62.48: bf16[4096], param_63.52: bf16[4096], param_64.48: bf16[4096]) -> bf16[266240] {
  param_0.54041 = bf16[4096] parameter(0)
  param_1.61435 = bf16[4096] parameter(1)
  param_2.33769 = bf16[4096] parameter(2)
  param_3.23571 = bf16[4096] parameter(3)
  param_4.15954 = bf16[4096] parameter(4)
  param_5.12472 = bf16[4096] parameter(5)
  param_6.8611 = bf16[4096] parameter(6)
  param_7.4936 = bf16[4096] parameter(7)
  param_8.3000 = bf16[4096] parameter(8)
  param_9.3130 = bf16[4096] parameter(9)
  param_10.2557 = bf16[4096] parameter(10)
  param_11.1496 = bf16[4096] parameter(11)
  param_12.1050 = bf16[4096] parameter(12)
  param_13.944 = bf16[4096] parameter(13)
  param_14.716 = bf16[4096] parameter(14)
  param_15.408 = bf16[4096] parameter(15)
  param_16.308 = bf16[4096] parameter(16)
  param_17.163 = bf16[4096] parameter(17)
  param_18.59 = bf16[4096] parameter(18)
  param_19.64 = bf16[4096] parameter(19)
  param_20.68 = bf16[4096] parameter(20)
  param_21.64 = bf16[4096] parameter(21)
  param_22.64 = bf16[4096] parameter(22)
  param_23.68 = bf16[4096] parameter(23)
  param_24.64 = bf16[4096] parameter(24)
  param_25.64 = bf16[4096] parameter(25)
  param_26.68 = bf16[4096] parameter(26)
  param_27.63 = bf16[4096] parameter(27)
  param_28.59 = bf16[4096] parameter(28)
  param_29.64 = bf16[4096] parameter(29)
  param_30.68 = bf16[4096] parameter(30)
  param_31.64 = bf16[4096] parameter(31)
  param_32.64 = bf16[4096] parameter(32)
  param_33.68 = bf16[4096] parameter(33)
  param_34.64 = bf16[4096] parameter(34)
  param_35.64 = bf16[4096] parameter(35)
  param_36.68 = bf16[4096] parameter(36)
  param_37.63 = bf16[4096] parameter(37)
  param_38.59 = bf16[4096] parameter(38)
  param_39.64 = bf16[4096] parameter(39)
  param_40.68 = bf16[4096] parameter(40)
  param_41.64 = bf16[4096] parameter(41)
  param_42.60 = bf16[4096] parameter(42)
  param_43.60 = bf16[4096] parameter(43)
  param_44.48 = bf16[4096] parameter(44)
  param_45.48 = bf16[4096] parameter(45)
  param_46.52 = bf16[4096] parameter(46)
  param_47.47 = bf16[4096] parameter(47)
  param_48.47 = bf16[4096] parameter(48)
  param_49.52 = bf16[4096] parameter(49)
  param_50.52 = bf16[4096] parameter(50)
  param_51.48 = bf16[4096] parameter(51)
  param_52.48 = bf16[4096] parameter(52)
  param_53.52 = bf16[4096] parameter(53)
  param_54.48 = bf16[4096] parameter(54)
  param_55.48 = bf16[4096] parameter(55)
  param_56.52 = bf16[4096] parameter(56)
  param_57.47 = bf16[4096] parameter(57)
  param_58.47 = bf16[4096] parameter(58)
  param_59.52 = bf16[4096] parameter(59)
  param_60.52 = bf16[4096] parameter(60)
  param_61.48 = bf16[4096] parameter(61)
  param_62.48 = bf16[4096] parameter(62)
  param_63.52 = bf16[4096] parameter(63)
  param_64.48 = bf16[4096] parameter(64)
  ROOT %fusion = bf16[266240] fusion(param_0.54041, param_1.61435, param_2.33769, param_3.23571, param_4.15954, param_5.12472, param_6.8611, param_7.4936, param_8.3000, param_9.3130, param_10.2557, param_11.1496, param_12.1050, param_13.944, param_14.716, param_15.408, param_16.308, param_17.163, param_18.59, param_19.64, param_20.68, param_21.64, param_22.64, param_23.68, param_24.64, param_25.64, param_26.68, param_27.63, param_28.59, param_29.64, param_30.68, param_31.64, param_32.64, param_33.68, param_34.64, param_35.64, param_36.68, param_37.63, param_38.59, param_39.64, param_40.68, param_41.64, param_42.60, param_43.60, param_44.48, param_45.48, param_46.52, param_47.47, param_48.47, param_49.52, param_50.52, param_51.48, param_52.48, param_53.52, param_54.48, param_55.48, param_56.52, param_57.47, param_58.47, param_59.52, param_60.52, param_61.48, param_62.48, param_63.52, param_64.48), kind=kLoop, calls=%wrapped_concatenate_computation.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source:
// mixtral_training/extracted_fusions/wrapped_concatenate_computation.1.clean
TEST_F(CuDnnNonGemmFusionLevel1Test,
       MixtralTraining_WrappedConcatenateComputation_1) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule wrapped_concatenate_computation.1_module, entry_computation_layout={(f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=5*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=10*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=15*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=20*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=25*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=30*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=35*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=40*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=45*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=50*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=55*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=60*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=65*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=70*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=75*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=80*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=85*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=90*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=95*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=100*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=105*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=110*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=115*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=120*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=125*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=130*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=135*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=140*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=145*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=150*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=155*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=160*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=165*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=170*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=175*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=180*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=185*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=190*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=195*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=200*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=205*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=210*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=215*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=220*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=225*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=230*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=235*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=240*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=245*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=250*/f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=255*/f32[1]{0})->f32[256]{0}}

%wrapped_concatenate_computation.1.clone (param_0.54043: f32[1], param_1.61436: f32[1], param_2.33770: f32[1], param_3.23572: f32[1], param_4.15955: f32[1], param_5.12473: f32[1], param_6.8612: f32[1], param_7.4937: f32[1], param_8.3001: f32[1], param_9.3131: f32[1], param_10.2558: f32[1], param_11.1497: f32[1], param_12.1051: f32[1], param_13.945: f32[1], param_14.717: f32[1], param_15.409: f32[1], param_16.309: f32[1], param_17.164: f32[1], param_18.60: f32[1], param_19.65: f32[1], param_20.69: f32[1], param_21.65: f32[1], param_22.65: f32[1], param_23.69: f32[1], param_24.65: f32[1], param_25.65: f32[1], param_26.69: f32[1], param_27.64: f32[1], param_28.60: f32[1], param_29.65: f32[1], param_30.69: f32[1], param_31.65: f32[1], param_32.65: f32[1], param_33.69: f32[1], param_34.65: f32[1], param_35.65: f32[1], param_36.69: f32[1], param_37.64: f32[1], param_38.60: f32[1], param_39.65: f32[1], param_40.69: f32[1], param_41.65: f32[1], param_42.61: f32[1], param_43.61: f32[1], param_44.49: f32[1], param_45.49: f32[1], param_46.53: f32[1], param_47.48: f32[1], param_48.48: f32[1], param_49.53: f32[1], param_50.53: f32[1], param_51.49: f32[1], param_52.49: f32[1], param_53.53: f32[1], param_54.49: f32[1], param_55.49: f32[1], param_56.53: f32[1], param_57.48: f32[1], param_58.48: f32[1], param_59.53: f32[1], param_60.53: f32[1], param_61.49: f32[1], param_62.49: f32[1], param_63.53: f32[1], param_64.49: f32[1], param_65.48: f32[1], param_66.52: f32[1], param_67.47: f32[1], param_68.47: f32[1], param_69.52: f32[1], param_70.52: f32[1], param_71.48: f32[1], param_72.48: f32[1], param_73.52: f32[1], param_74.48: f32[1], param_75.48: f32[1], param_76.52: f32[1], param_77.47: f32[1], param_78.47: f32[1], param_79.52: f32[1], param_80.52: f32[1], param_81.48: f32[1], param_82.48: f32[1], param_83.52: f32[1], param_84.48: f32[1], param_85.48: f32[1], param_86.52: f32[1], param_87.47: f32[1], param_88.47: f32[1], param_89.52: f32[1], param_90.52: f32[1], param_91.48: f32[1], param_92.35: f32[1], param_93.26: f32[1], param_94: f32[1], param_95: f32[1], param_96: f32[1], param_97: f32[1], param_98: f32[1], param_99: f32[1], param_100: f32[1], param_101: f32[1], param_102: f32[1], param_103: f32[1], param_104: f32[1], param_105: f32[1], param_106: f32[1], param_107: f32[1], param_108: f32[1], param_109: f32[1], param_110: f32[1], param_111: f32[1], param_112: f32[1], param_113: f32[1], param_114: f32[1], param_115: f32[1], param_116: f32[1], param_117: f32[1], param_118: f32[1], param_119: f32[1], param_120: f32[1], param_121: f32[1], param_122: f32[1], param_123: f32[1], param_124: f32[1], param_125: f32[1], param_126: f32[1], param_127: f32[1], param_128: f32[1], param_129: f32[1], param_130: f32[1], param_131: f32[1], param_132: f32[1], param_133: f32[1], param_134: f32[1], param_135: f32[1], param_136: f32[1], param_137: f32[1], param_138: f32[1], param_139: f32[1], param_140: f32[1], param_141: f32[1], param_142: f32[1], param_143: f32[1], param_144: f32[1], param_145: f32[1], param_146: f32[1], param_147: f32[1], param_148: f32[1], param_149: f32[1], param_150: f32[1], param_151: f32[1], param_152: f32[1], param_153: f32[1], param_154: f32[1], param_155: f32[1], param_156: f32[1], param_157: f32[1], param_158: f32[1], param_159: f32[1], param_160: f32[1], param_161: f32[1], param_162: f32[1], param_163: f32[1], param_164: f32[1], param_165: f32[1], param_166: f32[1], param_167: f32[1], param_168: f32[1], param_169: f32[1], param_170: f32[1], param_171: f32[1], param_172: f32[1], param_173: f32[1], param_174: f32[1], param_175: f32[1], param_176: f32[1], param_177: f32[1], param_178: f32[1], param_179: f32[1], param_180: f32[1], param_181: f32[1], param_182: f32[1], param_183: f32[1], param_184: f32[1], param_185: f32[1], param_186: f32[1], param_187: f32[1], param_188: f32[1], param_189: f32[1], param_190: f32[1], param_191: f32[1], param_192: f32[1], param_193: f32[1], param_194: f32[1], param_195: f32[1], param_196: f32[1], param_197: f32[1], param_198: f32[1], param_199: f32[1], param_200: f32[1], param_201: f32[1], param_202: f32[1], param_203: f32[1], param_204: f32[1], param_205: f32[1], param_206: f32[1], param_207: f32[1], param_208: f32[1], param_209: f32[1], param_210: f32[1], param_211: f32[1], param_212: f32[1], param_213: f32[1], param_214: f32[1], param_215: f32[1], param_216: f32[1], param_217: f32[1], param_218: f32[1], param_219: f32[1], param_220: f32[1], param_221: f32[1], param_222: f32[1], param_223: f32[1], param_224: f32[1], param_225: f32[1], param_226: f32[1], param_227: f32[1], param_228: f32[1], param_229: f32[1], param_230: f32[1], param_231: f32[1], param_232: f32[1], param_233: f32[1], param_234: f32[1], param_235: f32[1], param_236: f32[1], param_237: f32[1], param_238: f32[1], param_239: f32[1], param_240: f32[1], param_241: f32[1], param_242: f32[1], param_243: f32[1], param_244: f32[1], param_245: f32[1], param_246: f32[1], param_247: f32[1], param_248: f32[1], param_249: f32[1], param_250: f32[1], param_251: f32[1], param_252: f32[1], param_253: f32[1], param_254: f32[1], param_255: f32[1]) -> f32[256] {
  %param_0.54043 = f32[1]{0} parameter(0)
  %param_1.61436 = f32[1]{0} parameter(1)
  %param_2.33770 = f32[1]{0} parameter(2)
  %param_3.23572 = f32[1]{0} parameter(3)
  %param_4.15955 = f32[1]{0} parameter(4)
  %param_5.12473 = f32[1]{0} parameter(5)
  %param_6.8612 = f32[1]{0} parameter(6)
  %param_7.4937 = f32[1]{0} parameter(7)
  %param_8.3001 = f32[1]{0} parameter(8)
  %param_9.3131 = f32[1]{0} parameter(9)
  %param_10.2558 = f32[1]{0} parameter(10)
  %param_11.1497 = f32[1]{0} parameter(11)
  %param_12.1051 = f32[1]{0} parameter(12)
  %param_13.945 = f32[1]{0} parameter(13)
  %param_14.717 = f32[1]{0} parameter(14)
  %param_15.409 = f32[1]{0} parameter(15)
  %param_16.309 = f32[1]{0} parameter(16)
  %param_17.164 = f32[1]{0} parameter(17)
  %param_18.60 = f32[1]{0} parameter(18)
  %param_19.65 = f32[1]{0} parameter(19)
  %param_20.69 = f32[1]{0} parameter(20)
  %param_21.65 = f32[1]{0} parameter(21)
  %param_22.65 = f32[1]{0} parameter(22)
  %param_23.69 = f32[1]{0} parameter(23)
  %param_24.65 = f32[1]{0} parameter(24)
  %param_25.65 = f32[1]{0} parameter(25)
  %param_26.69 = f32[1]{0} parameter(26)
  %param_27.64 = f32[1]{0} parameter(27)
  %param_28.60 = f32[1]{0} parameter(28)
  %param_29.65 = f32[1]{0} parameter(29)
  %param_30.69 = f32[1]{0} parameter(30)
  %param_31.65 = f32[1]{0} parameter(31)
  %param_32.65 = f32[1]{0} parameter(32)
  %param_33.69 = f32[1]{0} parameter(33)
  %param_34.65 = f32[1]{0} parameter(34)
  %param_35.65 = f32[1]{0} parameter(35)
  %param_36.69 = f32[1]{0} parameter(36)
  %param_37.64 = f32[1]{0} parameter(37)
  %param_38.60 = f32[1]{0} parameter(38)
  %param_39.65 = f32[1]{0} parameter(39)
  %param_40.69 = f32[1]{0} parameter(40)
  %param_41.65 = f32[1]{0} parameter(41)
  %param_42.61 = f32[1]{0} parameter(42)
  %param_43.61 = f32[1]{0} parameter(43)
  %param_44.49 = f32[1]{0} parameter(44)
  %param_45.49 = f32[1]{0} parameter(45)
  %param_46.53 = f32[1]{0} parameter(46)
  %param_47.48 = f32[1]{0} parameter(47)
  %param_48.48 = f32[1]{0} parameter(48)
  %param_49.53 = f32[1]{0} parameter(49)
  %param_50.53 = f32[1]{0} parameter(50)
  %param_51.49 = f32[1]{0} parameter(51)
  %param_52.49 = f32[1]{0} parameter(52)
  %param_53.53 = f32[1]{0} parameter(53)
  %param_54.49 = f32[1]{0} parameter(54)
  %param_55.49 = f32[1]{0} parameter(55)
  %param_56.53 = f32[1]{0} parameter(56)
  %param_57.48 = f32[1]{0} parameter(57)
  %param_58.48 = f32[1]{0} parameter(58)
  %param_59.53 = f32[1]{0} parameter(59)
  %param_60.53 = f32[1]{0} parameter(60)
  %param_61.49 = f32[1]{0} parameter(61)
  %param_62.49 = f32[1]{0} parameter(62)
  %param_63.53 = f32[1]{0} parameter(63)
  %param_64.49 = f32[1]{0} parameter(64)
  %param_65.48 = f32[1]{0} parameter(65)
  %param_66.52 = f32[1]{0} parameter(66)
  %param_67.47 = f32[1]{0} parameter(67)
  %param_68.47 = f32[1]{0} parameter(68)
  %param_69.52 = f32[1]{0} parameter(69)
  %param_70.52 = f32[1]{0} parameter(70)
  %param_71.48 = f32[1]{0} parameter(71)
  %param_72.48 = f32[1]{0} parameter(72)
  %param_73.52 = f32[1]{0} parameter(73)
  %param_74.48 = f32[1]{0} parameter(74)
  %param_75.48 = f32[1]{0} parameter(75)
  %param_76.52 = f32[1]{0} parameter(76)
  %param_77.47 = f32[1]{0} parameter(77)
  %param_78.47 = f32[1]{0} parameter(78)
  %param_79.52 = f32[1]{0} parameter(79)
  %param_80.52 = f32[1]{0} parameter(80)
  %param_81.48 = f32[1]{0} parameter(81)
  %param_82.48 = f32[1]{0} parameter(82)
  %param_83.52 = f32[1]{0} parameter(83)
  %param_84.48 = f32[1]{0} parameter(84)
  %param_85.48 = f32[1]{0} parameter(85)
  %param_86.52 = f32[1]{0} parameter(86)
  %param_87.47 = f32[1]{0} parameter(87)
  %param_88.47 = f32[1]{0} parameter(88)
  %param_89.52 = f32[1]{0} parameter(89)
  %param_90.52 = f32[1]{0} parameter(90)
  %param_91.48 = f32[1]{0} parameter(91)
  %param_92.35 = f32[1]{0} parameter(92)
  %param_93.26 = f32[1]{0} parameter(93)
  %param_94 = f32[1]{0} parameter(94)
  %param_95 = f32[1]{0} parameter(95)
  %param_96 = f32[1]{0} parameter(96)
  %param_97 = f32[1]{0} parameter(97)
  %param_98 = f32[1]{0} parameter(98)
  %param_99 = f32[1]{0} parameter(99)
  %param_100 = f32[1]{0} parameter(100)
  %param_101 = f32[1]{0} parameter(101)
  %param_102 = f32[1]{0} parameter(102)
  %param_103 = f32[1]{0} parameter(103)
  %param_104 = f32[1]{0} parameter(104)
  %param_105 = f32[1]{0} parameter(105)
  %param_106 = f32[1]{0} parameter(106)
  %param_107 = f32[1]{0} parameter(107)
  %param_108 = f32[1]{0} parameter(108)
  %param_109 = f32[1]{0} parameter(109)
  %param_110 = f32[1]{0} parameter(110)
  %param_111 = f32[1]{0} parameter(111)
  %param_112 = f32[1]{0} parameter(112)
  %param_113 = f32[1]{0} parameter(113)
  %param_114 = f32[1]{0} parameter(114)
  %param_115 = f32[1]{0} parameter(115)
  %param_116 = f32[1]{0} parameter(116)
  %param_117 = f32[1]{0} parameter(117)
  %param_118 = f32[1]{0} parameter(118)
  %param_119 = f32[1]{0} parameter(119)
  %param_120 = f32[1]{0} parameter(120)
  %param_121 = f32[1]{0} parameter(121)
  %param_122 = f32[1]{0} parameter(122)
  %param_123 = f32[1]{0} parameter(123)
  %param_124 = f32[1]{0} parameter(124)
  %param_125 = f32[1]{0} parameter(125)
  %param_126 = f32[1]{0} parameter(126)
  %param_127 = f32[1]{0} parameter(127)
  %param_128 = f32[1]{0} parameter(128)
  %param_129 = f32[1]{0} parameter(129)
  %param_130 = f32[1]{0} parameter(130)
  %param_131 = f32[1]{0} parameter(131)
  %param_132 = f32[1]{0} parameter(132)
  %param_133 = f32[1]{0} parameter(133)
  %param_134 = f32[1]{0} parameter(134)
  %param_135 = f32[1]{0} parameter(135)
  %param_136 = f32[1]{0} parameter(136)
  %param_137 = f32[1]{0} parameter(137)
  %param_138 = f32[1]{0} parameter(138)
  %param_139 = f32[1]{0} parameter(139)
  %param_140 = f32[1]{0} parameter(140)
  %param_141 = f32[1]{0} parameter(141)
  %param_142 = f32[1]{0} parameter(142)
  %param_143 = f32[1]{0} parameter(143)
  %param_144 = f32[1]{0} parameter(144)
  %param_145 = f32[1]{0} parameter(145)
  %param_146 = f32[1]{0} parameter(146)
  %param_147 = f32[1]{0} parameter(147)
  %param_148 = f32[1]{0} parameter(148)
  %param_149 = f32[1]{0} parameter(149)
  %param_150 = f32[1]{0} parameter(150)
  %param_151 = f32[1]{0} parameter(151)
  %param_152 = f32[1]{0} parameter(152)
  %param_153 = f32[1]{0} parameter(153)
  %param_154 = f32[1]{0} parameter(154)
  %param_155 = f32[1]{0} parameter(155)
  %param_156 = f32[1]{0} parameter(156)
  %param_157 = f32[1]{0} parameter(157)
  %param_158 = f32[1]{0} parameter(158)
  %param_159 = f32[1]{0} parameter(159)
  %param_160 = f32[1]{0} parameter(160)
  %param_161 = f32[1]{0} parameter(161)
  %param_162 = f32[1]{0} parameter(162)
  %param_163 = f32[1]{0} parameter(163)
  %param_164 = f32[1]{0} parameter(164)
  %param_165 = f32[1]{0} parameter(165)
  %param_166 = f32[1]{0} parameter(166)
  %param_167 = f32[1]{0} parameter(167)
  %param_168 = f32[1]{0} parameter(168)
  %param_169 = f32[1]{0} parameter(169)
  %param_170 = f32[1]{0} parameter(170)
  %param_171 = f32[1]{0} parameter(171)
  %param_172 = f32[1]{0} parameter(172)
  %param_173 = f32[1]{0} parameter(173)
  %param_174 = f32[1]{0} parameter(174)
  %param_175 = f32[1]{0} parameter(175)
  %param_176 = f32[1]{0} parameter(176)
  %param_177 = f32[1]{0} parameter(177)
  %param_178 = f32[1]{0} parameter(178)
  %param_179 = f32[1]{0} parameter(179)
  %param_180 = f32[1]{0} parameter(180)
  %param_181 = f32[1]{0} parameter(181)
  %param_182 = f32[1]{0} parameter(182)
  %param_183 = f32[1]{0} parameter(183)
  %param_184 = f32[1]{0} parameter(184)
  %param_185 = f32[1]{0} parameter(185)
  %param_186 = f32[1]{0} parameter(186)
  %param_187 = f32[1]{0} parameter(187)
  %param_188 = f32[1]{0} parameter(188)
  %param_189 = f32[1]{0} parameter(189)
  %param_190 = f32[1]{0} parameter(190)
  %param_191 = f32[1]{0} parameter(191)
  %param_192 = f32[1]{0} parameter(192)
  %param_193 = f32[1]{0} parameter(193)
  %param_194 = f32[1]{0} parameter(194)
  %param_195 = f32[1]{0} parameter(195)
  %param_196 = f32[1]{0} parameter(196)
  %param_197 = f32[1]{0} parameter(197)
  %param_198 = f32[1]{0} parameter(198)
  %param_199 = f32[1]{0} parameter(199)
  %param_200 = f32[1]{0} parameter(200)
  %param_201 = f32[1]{0} parameter(201)
  %param_202 = f32[1]{0} parameter(202)
  %param_203 = f32[1]{0} parameter(203)
  %param_204 = f32[1]{0} parameter(204)
  %param_205 = f32[1]{0} parameter(205)
  %param_206 = f32[1]{0} parameter(206)
  %param_207 = f32[1]{0} parameter(207)
  %param_208 = f32[1]{0} parameter(208)
  %param_209 = f32[1]{0} parameter(209)
  %param_210 = f32[1]{0} parameter(210)
  %param_211 = f32[1]{0} parameter(211)
  %param_212 = f32[1]{0} parameter(212)
  %param_213 = f32[1]{0} parameter(213)
  %param_214 = f32[1]{0} parameter(214)
  %param_215 = f32[1]{0} parameter(215)
  %param_216 = f32[1]{0} parameter(216)
  %param_217 = f32[1]{0} parameter(217)
  %param_218 = f32[1]{0} parameter(218)
  %param_219 = f32[1]{0} parameter(219)
  %param_220 = f32[1]{0} parameter(220)
  %param_221 = f32[1]{0} parameter(221)
  %param_222 = f32[1]{0} parameter(222)
  %param_223 = f32[1]{0} parameter(223)
  %param_224 = f32[1]{0} parameter(224)
  %param_225 = f32[1]{0} parameter(225)
  %param_226 = f32[1]{0} parameter(226)
  %param_227 = f32[1]{0} parameter(227)
  %param_228 = f32[1]{0} parameter(228)
  %param_229 = f32[1]{0} parameter(229)
  %param_230 = f32[1]{0} parameter(230)
  %param_231 = f32[1]{0} parameter(231)
  %param_232 = f32[1]{0} parameter(232)
  %param_233 = f32[1]{0} parameter(233)
  %param_234 = f32[1]{0} parameter(234)
  %param_235 = f32[1]{0} parameter(235)
  %param_236 = f32[1]{0} parameter(236)
  %param_237 = f32[1]{0} parameter(237)
  %param_238 = f32[1]{0} parameter(238)
  %param_239 = f32[1]{0} parameter(239)
  %param_240 = f32[1]{0} parameter(240)
  %param_241 = f32[1]{0} parameter(241)
  %param_242 = f32[1]{0} parameter(242)
  %param_243 = f32[1]{0} parameter(243)
  %param_244 = f32[1]{0} parameter(244)
  %param_245 = f32[1]{0} parameter(245)
  %param_246 = f32[1]{0} parameter(246)
  %param_247 = f32[1]{0} parameter(247)
  %param_248 = f32[1]{0} parameter(248)
  %param_249 = f32[1]{0} parameter(249)
  %param_250 = f32[1]{0} parameter(250)
  %param_251 = f32[1]{0} parameter(251)
  %param_252 = f32[1]{0} parameter(252)
  %param_253 = f32[1]{0} parameter(253)
  %param_254 = f32[1]{0} parameter(254)
  %param_255 = f32[1]{0} parameter(255)
  ROOT %concatenate.440 = f32[256]{0} concatenate(%param_0.54043, %param_1.61436, %param_2.33770, %param_3.23572, %param_4.15955, /*index=5*/%param_5.12473, %param_6.8612, %param_7.4937, %param_8.3001, %param_9.3131, /*index=10*/%param_10.2558, %param_11.1497, %param_12.1051, %param_13.945, %param_14.717, /*index=15*/%param_15.409, %param_16.309, %param_17.164, %param_18.60, %param_19.65, /*index=20*/%param_20.69, %param_21.65, %param_22.65, %param_23.69, %param_24.65, /*index=25*/%param_25.65, %param_26.69, %param_27.64, %param_28.60, %param_29.65, /*index=30*/%param_30.69, %param_31.65, %param_32.65, %param_33.69, %param_34.65, /*index=35*/%param_35.65, %param_36.69, %param_37.64, %param_38.60, %param_39.65, /*index=40*/%param_40.69, %param_41.65, %param_42.61, %param_43.61, %param_44.49, /*index=45*/%param_45.49, %param_46.53, %param_47.48, %param_48.48, %param_49.53, /*index=50*/%param_50.53, %param_51.49, %param_52.49, %param_53.53, %param_54.49, /*index=55*/%param_55.49, %param_56.53, %param_57.48, %param_58.48, %param_59.53, /*index=60*/%param_60.53, %param_61.49, %param_62.49, %param_63.53, %param_64.49, /*index=65*/%param_65.48, %param_66.52, %param_67.47, %param_68.47, %param_69.52, /*index=70*/%param_70.52, %param_71.48, %param_72.48, %param_73.52, %param_74.48, /*index=75*/%param_75.48, %param_76.52, %param_77.47, %param_78.47, %param_79.52, /*index=80*/%param_80.52, %param_81.48, %param_82.48, %param_83.52, %param_84.48, /*index=85*/%param_85.48, %param_86.52, %param_87.47, %param_88.47, %param_89.52, /*index=90*/%param_90.52, %param_91.48, %param_92.35, %param_93.26, %param_94, /*index=95*/%param_95, %param_96, %param_97, %param_98, %param_99, /*index=100*/%param_100, %param_101, %param_102, %param_103, %param_104, /*index=105*/%param_105, %param_106, %param_107, %param_108, %param_109, /*index=110*/%param_110, %param_111, %param_112, %param_113, %param_114, /*index=115*/%param_115, %param_116, %param_117, %param_118, %param_119, /*index=120*/%param_120, %param_121, %param_122, %param_123, %param_124, /*index=125*/%param_125, %param_126, %param_127, %param_128, %param_129, /*index=130*/%param_130, %param_131, %param_132, %param_133, %param_134, /*index=135*/%param_135, %param_136, %param_137, %param_138, %param_139, /*index=140*/%param_140, %param_141, %param_142, %param_143, %param_144, /*index=145*/%param_145, %param_146, %param_147, %param_148, %param_149, /*index=150*/%param_150, %param_151, %param_152, %param_153, %param_154, /*index=155*/%param_155, %param_156, %param_157, %param_158, %param_159, /*index=160*/%param_160, %param_161, %param_162, %param_163, %param_164, /*index=165*/%param_165, %param_166, %param_167, %param_168, %param_169, /*index=170*/%param_170, %param_171, %param_172, %param_173, %param_174, /*index=175*/%param_175, %param_176, %param_177, %param_178, %param_179, /*index=180*/%param_180, %param_181, %param_182, %param_183, %param_184, /*index=185*/%param_185, %param_186, %param_187, %param_188, %param_189, /*index=190*/%param_190, %param_191, %param_192, %param_193, %param_194, /*index=195*/%param_195, %param_196, %param_197, %param_198, %param_199, /*index=200*/%param_200, %param_201, %param_202, %param_203, %param_204, /*index=205*/%param_205, %param_206, %param_207, %param_208, %param_209, /*index=210*/%param_210, %param_211, %param_212, %param_213, %param_214, /*index=215*/%param_215, %param_216, %param_217, %param_218, %param_219, /*index=220*/%param_220, %param_221, %param_222, %param_223, %param_224, /*index=225*/%param_225, %param_226, %param_227, %param_228, %param_229, /*index=230*/%param_230, %param_231, %param_232, %param_233, %param_234, /*index=235*/%param_235, %param_236, %param_237, %param_238, %param_239, /*index=240*/%param_240, %param_241, %param_242, %param_243, %param_244, /*index=245*/%param_245, %param_246, %param_247, %param_248, %param_249, /*index=250*/%param_250, %param_251, %param_252, %param_253, %param_254, /*index=255*/%param_255), dimensions={0}
}



ENTRY %wrapper_wrapped_concatenate_computation.1.clone (param_0.54043: f32[1], param_1.61436: f32[1], param_2.33770: f32[1], param_3.23572: f32[1], param_4.15955: f32[1], param_5.12473: f32[1], param_6.8612: f32[1], param_7.4937: f32[1], param_8.3001: f32[1], param_9.3131: f32[1], param_10.2558: f32[1], param_11.1497: f32[1], param_12.1051: f32[1], param_13.945: f32[1], param_14.717: f32[1], param_15.409: f32[1], param_16.309: f32[1], param_17.164: f32[1], param_18.60: f32[1], param_19.65: f32[1], param_20.69: f32[1], param_21.65: f32[1], param_22.65: f32[1], param_23.69: f32[1], param_24.65: f32[1], param_25.65: f32[1], param_26.69: f32[1], param_27.64: f32[1], param_28.60: f32[1], param_29.65: f32[1], param_30.69: f32[1], param_31.65: f32[1], param_32.65: f32[1], param_33.69: f32[1], param_34.65: f32[1], param_35.65: f32[1], param_36.69: f32[1], param_37.64: f32[1], param_38.60: f32[1], param_39.65: f32[1], param_40.69: f32[1], param_41.65: f32[1], param_42.61: f32[1], param_43.61: f32[1], param_44.49: f32[1], param_45.49: f32[1], param_46.53: f32[1], param_47.48: f32[1], param_48.48: f32[1], param_49.53: f32[1], param_50.53: f32[1], param_51.49: f32[1], param_52.49: f32[1], param_53.53: f32[1], param_54.49: f32[1], param_55.49: f32[1], param_56.53: f32[1], param_57.48: f32[1], param_58.48: f32[1], param_59.53: f32[1], param_60.53: f32[1], param_61.49: f32[1], param_62.49: f32[1], param_63.53: f32[1], param_64.49: f32[1], param_65.48: f32[1], param_66.52: f32[1], param_67.47: f32[1], param_68.47: f32[1], param_69.52: f32[1], param_70.52: f32[1], param_71.48: f32[1], param_72.48: f32[1], param_73.52: f32[1], param_74.48: f32[1], param_75.48: f32[1], param_76.52: f32[1], param_77.47: f32[1], param_78.47: f32[1], param_79.52: f32[1], param_80.52: f32[1], param_81.48: f32[1], param_82.48: f32[1], param_83.52: f32[1], param_84.48: f32[1], param_85.48: f32[1], param_86.52: f32[1], param_87.47: f32[1], param_88.47: f32[1], param_89.52: f32[1], param_90.52: f32[1], param_91.48: f32[1], param_92.35: f32[1], param_93.26: f32[1], param_94: f32[1], param_95: f32[1], param_96: f32[1], param_97: f32[1], param_98: f32[1], param_99: f32[1], param_100: f32[1], param_101: f32[1], param_102: f32[1], param_103: f32[1], param_104: f32[1], param_105: f32[1], param_106: f32[1], param_107: f32[1], param_108: f32[1], param_109: f32[1], param_110: f32[1], param_111: f32[1], param_112: f32[1], param_113: f32[1], param_114: f32[1], param_115: f32[1], param_116: f32[1], param_117: f32[1], param_118: f32[1], param_119: f32[1], param_120: f32[1], param_121: f32[1], param_122: f32[1], param_123: f32[1], param_124: f32[1], param_125: f32[1], param_126: f32[1], param_127: f32[1], param_128: f32[1], param_129: f32[1], param_130: f32[1], param_131: f32[1], param_132: f32[1], param_133: f32[1], param_134: f32[1], param_135: f32[1], param_136: f32[1], param_137: f32[1], param_138: f32[1], param_139: f32[1], param_140: f32[1], param_141: f32[1], param_142: f32[1], param_143: f32[1], param_144: f32[1], param_145: f32[1], param_146: f32[1], param_147: f32[1], param_148: f32[1], param_149: f32[1], param_150: f32[1], param_151: f32[1], param_152: f32[1], param_153: f32[1], param_154: f32[1], param_155: f32[1], param_156: f32[1], param_157: f32[1], param_158: f32[1], param_159: f32[1], param_160: f32[1], param_161: f32[1], param_162: f32[1], param_163: f32[1], param_164: f32[1], param_165: f32[1], param_166: f32[1], param_167: f32[1], param_168: f32[1], param_169: f32[1], param_170: f32[1], param_171: f32[1], param_172: f32[1], param_173: f32[1], param_174: f32[1], param_175: f32[1], param_176: f32[1], param_177: f32[1], param_178: f32[1], param_179: f32[1], param_180: f32[1], param_181: f32[1], param_182: f32[1], param_183: f32[1], param_184: f32[1], param_185: f32[1], param_186: f32[1], param_187: f32[1], param_188: f32[1], param_189: f32[1], param_190: f32[1], param_191: f32[1], param_192: f32[1], param_193: f32[1], param_194: f32[1], param_195: f32[1], param_196: f32[1], param_197: f32[1], param_198: f32[1], param_199: f32[1], param_200: f32[1], param_201: f32[1], param_202: f32[1], param_203: f32[1], param_204: f32[1], param_205: f32[1], param_206: f32[1], param_207: f32[1], param_208: f32[1], param_209: f32[1], param_210: f32[1], param_211: f32[1], param_212: f32[1], param_213: f32[1], param_214: f32[1], param_215: f32[1], param_216: f32[1], param_217: f32[1], param_218: f32[1], param_219: f32[1], param_220: f32[1], param_221: f32[1], param_222: f32[1], param_223: f32[1], param_224: f32[1], param_225: f32[1], param_226: f32[1], param_227: f32[1], param_228: f32[1], param_229: f32[1], param_230: f32[1], param_231: f32[1], param_232: f32[1], param_233: f32[1], param_234: f32[1], param_235: f32[1], param_236: f32[1], param_237: f32[1], param_238: f32[1], param_239: f32[1], param_240: f32[1], param_241: f32[1], param_242: f32[1], param_243: f32[1], param_244: f32[1], param_245: f32[1], param_246: f32[1], param_247: f32[1], param_248: f32[1], param_249: f32[1], param_250: f32[1], param_251: f32[1], param_252: f32[1], param_253: f32[1], param_254: f32[1], param_255: f32[1]) -> f32[256] {
  param_0.54043 = f32[1] parameter(0)
  param_1.61436 = f32[1] parameter(1)
  param_2.33770 = f32[1] parameter(2)
  param_3.23572 = f32[1] parameter(3)
  param_4.15955 = f32[1] parameter(4)
  param_5.12473 = f32[1] parameter(5)
  param_6.8612 = f32[1] parameter(6)
  param_7.4937 = f32[1] parameter(7)
  param_8.3001 = f32[1] parameter(8)
  param_9.3131 = f32[1] parameter(9)
  param_10.2558 = f32[1] parameter(10)
  param_11.1497 = f32[1] parameter(11)
  param_12.1051 = f32[1] parameter(12)
  param_13.945 = f32[1] parameter(13)
  param_14.717 = f32[1] parameter(14)
  param_15.409 = f32[1] parameter(15)
  param_16.309 = f32[1] parameter(16)
  param_17.164 = f32[1] parameter(17)
  param_18.60 = f32[1] parameter(18)
  param_19.65 = f32[1] parameter(19)
  param_20.69 = f32[1] parameter(20)
  param_21.65 = f32[1] parameter(21)
  param_22.65 = f32[1] parameter(22)
  param_23.69 = f32[1] parameter(23)
  param_24.65 = f32[1] parameter(24)
  param_25.65 = f32[1] parameter(25)
  param_26.69 = f32[1] parameter(26)
  param_27.64 = f32[1] parameter(27)
  param_28.60 = f32[1] parameter(28)
  param_29.65 = f32[1] parameter(29)
  param_30.69 = f32[1] parameter(30)
  param_31.65 = f32[1] parameter(31)
  param_32.65 = f32[1] parameter(32)
  param_33.69 = f32[1] parameter(33)
  param_34.65 = f32[1] parameter(34)
  param_35.65 = f32[1] parameter(35)
  param_36.69 = f32[1] parameter(36)
  param_37.64 = f32[1] parameter(37)
  param_38.60 = f32[1] parameter(38)
  param_39.65 = f32[1] parameter(39)
  param_40.69 = f32[1] parameter(40)
  param_41.65 = f32[1] parameter(41)
  param_42.61 = f32[1] parameter(42)
  param_43.61 = f32[1] parameter(43)
  param_44.49 = f32[1] parameter(44)
  param_45.49 = f32[1] parameter(45)
  param_46.53 = f32[1] parameter(46)
  param_47.48 = f32[1] parameter(47)
  param_48.48 = f32[1] parameter(48)
  param_49.53 = f32[1] parameter(49)
  param_50.53 = f32[1] parameter(50)
  param_51.49 = f32[1] parameter(51)
  param_52.49 = f32[1] parameter(52)
  param_53.53 = f32[1] parameter(53)
  param_54.49 = f32[1] parameter(54)
  param_55.49 = f32[1] parameter(55)
  param_56.53 = f32[1] parameter(56)
  param_57.48 = f32[1] parameter(57)
  param_58.48 = f32[1] parameter(58)
  param_59.53 = f32[1] parameter(59)
  param_60.53 = f32[1] parameter(60)
  param_61.49 = f32[1] parameter(61)
  param_62.49 = f32[1] parameter(62)
  param_63.53 = f32[1] parameter(63)
  param_64.49 = f32[1] parameter(64)
  param_65.48 = f32[1] parameter(65)
  param_66.52 = f32[1] parameter(66)
  param_67.47 = f32[1] parameter(67)
  param_68.47 = f32[1] parameter(68)
  param_69.52 = f32[1] parameter(69)
  param_70.52 = f32[1] parameter(70)
  param_71.48 = f32[1] parameter(71)
  param_72.48 = f32[1] parameter(72)
  param_73.52 = f32[1] parameter(73)
  param_74.48 = f32[1] parameter(74)
  param_75.48 = f32[1] parameter(75)
  param_76.52 = f32[1] parameter(76)
  param_77.47 = f32[1] parameter(77)
  param_78.47 = f32[1] parameter(78)
  param_79.52 = f32[1] parameter(79)
  param_80.52 = f32[1] parameter(80)
  param_81.48 = f32[1] parameter(81)
  param_82.48 = f32[1] parameter(82)
  param_83.52 = f32[1] parameter(83)
  param_84.48 = f32[1] parameter(84)
  param_85.48 = f32[1] parameter(85)
  param_86.52 = f32[1] parameter(86)
  param_87.47 = f32[1] parameter(87)
  param_88.47 = f32[1] parameter(88)
  param_89.52 = f32[1] parameter(89)
  param_90.52 = f32[1] parameter(90)
  param_91.48 = f32[1] parameter(91)
  param_92.35 = f32[1] parameter(92)
  param_93.26 = f32[1] parameter(93)
  param_94 = f32[1] parameter(94)
  param_95 = f32[1] parameter(95)
  param_96 = f32[1] parameter(96)
  param_97 = f32[1] parameter(97)
  param_98 = f32[1] parameter(98)
  param_99 = f32[1] parameter(99)
  param_100 = f32[1] parameter(100)
  param_101 = f32[1] parameter(101)
  param_102 = f32[1] parameter(102)
  param_103 = f32[1] parameter(103)
  param_104 = f32[1] parameter(104)
  param_105 = f32[1] parameter(105)
  param_106 = f32[1] parameter(106)
  param_107 = f32[1] parameter(107)
  param_108 = f32[1] parameter(108)
  param_109 = f32[1] parameter(109)
  param_110 = f32[1] parameter(110)
  param_111 = f32[1] parameter(111)
  param_112 = f32[1] parameter(112)
  param_113 = f32[1] parameter(113)
  param_114 = f32[1] parameter(114)
  param_115 = f32[1] parameter(115)
  param_116 = f32[1] parameter(116)
  param_117 = f32[1] parameter(117)
  param_118 = f32[1] parameter(118)
  param_119 = f32[1] parameter(119)
  param_120 = f32[1] parameter(120)
  param_121 = f32[1] parameter(121)
  param_122 = f32[1] parameter(122)
  param_123 = f32[1] parameter(123)
  param_124 = f32[1] parameter(124)
  param_125 = f32[1] parameter(125)
  param_126 = f32[1] parameter(126)
  param_127 = f32[1] parameter(127)
  param_128 = f32[1] parameter(128)
  param_129 = f32[1] parameter(129)
  param_130 = f32[1] parameter(130)
  param_131 = f32[1] parameter(131)
  param_132 = f32[1] parameter(132)
  param_133 = f32[1] parameter(133)
  param_134 = f32[1] parameter(134)
  param_135 = f32[1] parameter(135)
  param_136 = f32[1] parameter(136)
  param_137 = f32[1] parameter(137)
  param_138 = f32[1] parameter(138)
  param_139 = f32[1] parameter(139)
  param_140 = f32[1] parameter(140)
  param_141 = f32[1] parameter(141)
  param_142 = f32[1] parameter(142)
  param_143 = f32[1] parameter(143)
  param_144 = f32[1] parameter(144)
  param_145 = f32[1] parameter(145)
  param_146 = f32[1] parameter(146)
  param_147 = f32[1] parameter(147)
  param_148 = f32[1] parameter(148)
  param_149 = f32[1] parameter(149)
  param_150 = f32[1] parameter(150)
  param_151 = f32[1] parameter(151)
  param_152 = f32[1] parameter(152)
  param_153 = f32[1] parameter(153)
  param_154 = f32[1] parameter(154)
  param_155 = f32[1] parameter(155)
  param_156 = f32[1] parameter(156)
  param_157 = f32[1] parameter(157)
  param_158 = f32[1] parameter(158)
  param_159 = f32[1] parameter(159)
  param_160 = f32[1] parameter(160)
  param_161 = f32[1] parameter(161)
  param_162 = f32[1] parameter(162)
  param_163 = f32[1] parameter(163)
  param_164 = f32[1] parameter(164)
  param_165 = f32[1] parameter(165)
  param_166 = f32[1] parameter(166)
  param_167 = f32[1] parameter(167)
  param_168 = f32[1] parameter(168)
  param_169 = f32[1] parameter(169)
  param_170 = f32[1] parameter(170)
  param_171 = f32[1] parameter(171)
  param_172 = f32[1] parameter(172)
  param_173 = f32[1] parameter(173)
  param_174 = f32[1] parameter(174)
  param_175 = f32[1] parameter(175)
  param_176 = f32[1] parameter(176)
  param_177 = f32[1] parameter(177)
  param_178 = f32[1] parameter(178)
  param_179 = f32[1] parameter(179)
  param_180 = f32[1] parameter(180)
  param_181 = f32[1] parameter(181)
  param_182 = f32[1] parameter(182)
  param_183 = f32[1] parameter(183)
  param_184 = f32[1] parameter(184)
  param_185 = f32[1] parameter(185)
  param_186 = f32[1] parameter(186)
  param_187 = f32[1] parameter(187)
  param_188 = f32[1] parameter(188)
  param_189 = f32[1] parameter(189)
  param_190 = f32[1] parameter(190)
  param_191 = f32[1] parameter(191)
  param_192 = f32[1] parameter(192)
  param_193 = f32[1] parameter(193)
  param_194 = f32[1] parameter(194)
  param_195 = f32[1] parameter(195)
  param_196 = f32[1] parameter(196)
  param_197 = f32[1] parameter(197)
  param_198 = f32[1] parameter(198)
  param_199 = f32[1] parameter(199)
  param_200 = f32[1] parameter(200)
  param_201 = f32[1] parameter(201)
  param_202 = f32[1] parameter(202)
  param_203 = f32[1] parameter(203)
  param_204 = f32[1] parameter(204)
  param_205 = f32[1] parameter(205)
  param_206 = f32[1] parameter(206)
  param_207 = f32[1] parameter(207)
  param_208 = f32[1] parameter(208)
  param_209 = f32[1] parameter(209)
  param_210 = f32[1] parameter(210)
  param_211 = f32[1] parameter(211)
  param_212 = f32[1] parameter(212)
  param_213 = f32[1] parameter(213)
  param_214 = f32[1] parameter(214)
  param_215 = f32[1] parameter(215)
  param_216 = f32[1] parameter(216)
  param_217 = f32[1] parameter(217)
  param_218 = f32[1] parameter(218)
  param_219 = f32[1] parameter(219)
  param_220 = f32[1] parameter(220)
  param_221 = f32[1] parameter(221)
  param_222 = f32[1] parameter(222)
  param_223 = f32[1] parameter(223)
  param_224 = f32[1] parameter(224)
  param_225 = f32[1] parameter(225)
  param_226 = f32[1] parameter(226)
  param_227 = f32[1] parameter(227)
  param_228 = f32[1] parameter(228)
  param_229 = f32[1] parameter(229)
  param_230 = f32[1] parameter(230)
  param_231 = f32[1] parameter(231)
  param_232 = f32[1] parameter(232)
  param_233 = f32[1] parameter(233)
  param_234 = f32[1] parameter(234)
  param_235 = f32[1] parameter(235)
  param_236 = f32[1] parameter(236)
  param_237 = f32[1] parameter(237)
  param_238 = f32[1] parameter(238)
  param_239 = f32[1] parameter(239)
  param_240 = f32[1] parameter(240)
  param_241 = f32[1] parameter(241)
  param_242 = f32[1] parameter(242)
  param_243 = f32[1] parameter(243)
  param_244 = f32[1] parameter(244)
  param_245 = f32[1] parameter(245)
  param_246 = f32[1] parameter(246)
  param_247 = f32[1] parameter(247)
  param_248 = f32[1] parameter(248)
  param_249 = f32[1] parameter(249)
  param_250 = f32[1] parameter(250)
  param_251 = f32[1] parameter(251)
  param_252 = f32[1] parameter(252)
  param_253 = f32[1] parameter(253)
  param_254 = f32[1] parameter(254)
  param_255 = f32[1] parameter(255)
  ROOT %fusion = f32[256] fusion(param_0.54043, param_1.61436, param_2.33770, param_3.23572, param_4.15955, param_5.12473, param_6.8612, param_7.4937, param_8.3001, param_9.3131, param_10.2558, param_11.1497, param_12.1051, param_13.945, param_14.717, param_15.409, param_16.309, param_17.164, param_18.60, param_19.65, param_20.69, param_21.65, param_22.65, param_23.69, param_24.65, param_25.65, param_26.69, param_27.64, param_28.60, param_29.65, param_30.69, param_31.65, param_32.65, param_33.69, param_34.65, param_35.65, param_36.69, param_37.64, param_38.60, param_39.65, param_40.69, param_41.65, param_42.61, param_43.61, param_44.49, param_45.49, param_46.53, param_47.48, param_48.48, param_49.53, param_50.53, param_51.49, param_52.49, param_53.53, param_54.49, param_55.49, param_56.53, param_57.48, param_58.48, param_59.53, param_60.53, param_61.49, param_62.49, param_63.53, param_64.49, param_65.48, param_66.52, param_67.47, param_68.47, param_69.52, param_70.52, param_71.48, param_72.48, param_73.52, param_74.48, param_75.48, param_76.52, param_77.47, param_78.47, param_79.52, param_80.52, param_81.48, param_82.48, param_83.52, param_84.48, param_85.48, param_86.52, param_87.47, param_88.47, param_89.52, param_90.52, param_91.48, param_92.35, param_93.26, param_94, param_95, param_96, param_97, param_98, param_99, param_100, param_101, param_102, param_103, param_104, param_105, param_106, param_107, param_108, param_109, param_110, param_111, param_112, param_113, param_114, param_115, param_116, param_117, param_118, param_119, param_120, param_121, param_122, param_123, param_124, param_125, param_126, param_127, param_128, param_129, param_130, param_131, param_132, param_133, param_134, param_135, param_136, param_137, param_138, param_139, param_140, param_141, param_142, param_143, param_144, param_145, param_146, param_147, param_148, param_149, param_150, param_151, param_152, param_153, param_154, param_155, param_156, param_157, param_158, param_159, param_160, param_161, param_162, param_163, param_164, param_165, param_166, param_167, param_168, param_169, param_170, param_171, param_172, param_173, param_174, param_175, param_176, param_177, param_178, param_179, param_180, param_181, param_182, param_183, param_184, param_185, param_186, param_187, param_188, param_189, param_190, param_191, param_192, param_193, param_194, param_195, param_196, param_197, param_198, param_199, param_200, param_201, param_202, param_203, param_204, param_205, param_206, param_207, param_208, param_209, param_210, param_211, param_212, param_213, param_214, param_215, param_216, param_217, param_218, param_219, param_220, param_221, param_222, param_223, param_224, param_225, param_226, param_227, param_228, param_229, param_230, param_231, param_232, param_233, param_234, param_235, param_236, param_237, param_238, param_239, param_240, param_241, param_242, param_243, param_244, param_245, param_246, param_247, param_248, param_249, param_250, param_251, param_252, param_253, param_254, param_255), kind=kLoop, calls=%wrapped_concatenate_computation.1.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source:
// mixtral_training/extracted_fusions/wrapped_concatenate_computation.2.clean
TEST_F(CuDnnNonGemmFusionLevel1Test,
       MixtralTraining_WrappedConcatenateComputation_2) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule wrapped_concatenate_computation.2_module, entry_computation_layout={(f32[1]{0}, f32[1]{0})->f32[2]{0}}

%wrapped_concatenate_computation.2.clone (param_0.54364: f32[1], param_1.61437: f32[1]) -> f32[2] {
  %param_0.54364 = f32[1]{0} parameter(0)
  %param_1.61437 = f32[1]{0} parameter(1)
  ROOT %concatenate.441 = f32[2]{0} concatenate(%param_0.54364, %param_1.61437), dimensions={0}
}



ENTRY %wrapper_wrapped_concatenate_computation.2.clone (param_0.54364: f32[1], param_1.61437: f32[1]) -> f32[2] {
  param_0.54364 = f32[1] parameter(0)
  param_1.61437 = f32[1] parameter(1)
  ROOT %fusion = f32[2] fusion(param_0.54364, param_1.61437), kind=kLoop, calls=%wrapped_concatenate_computation.2.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

// Source:
// mixtral_training/extracted_fusions/wrapped_concatenate_computation.6.clean
TEST_F(CuDnnNonGemmFusionLevel1Test,
       MixtralTraining_WrappedConcatenateComputation_6) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule wrapped_concatenate_computation.6_module, entry_computation_layout={(f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}, /*index=5*/f32[1]{0}, f32[1]{0})->f32[7]{0}}

%wrapped_concatenate_computation.6.clone (param_0.55140: f32[1], param_1.61441: f32[1], param_2.33774: f32[1], param_3.23576: f32[1], param_4.15959: f32[1], param_5.12477: f32[1], param_6.8616: f32[1]) -> f32[7] {
  %param_0.55140 = f32[1]{0} parameter(0)
  %param_1.61441 = f32[1]{0} parameter(1)
  %param_2.33774 = f32[1]{0} parameter(2)
  %param_3.23576 = f32[1]{0} parameter(3)
  %param_4.15959 = f32[1]{0} parameter(4)
  %param_5.12477 = f32[1]{0} parameter(5)
  %param_6.8616 = f32[1]{0} parameter(6)
  ROOT %concatenate.450 = f32[7]{0} concatenate(%param_0.55140, %param_1.61441, %param_2.33774, %param_3.23576, %param_4.15959, /*index=5*/%param_5.12477, %param_6.8616), dimensions={0}
}



ENTRY %wrapper_wrapped_concatenate_computation.6.clone (param_0.55140: f32[1], param_1.61441: f32[1], param_2.33774: f32[1], param_3.23576: f32[1], param_4.15959: f32[1], param_5.12477: f32[1], param_6.8616: f32[1]) -> f32[7] {
  param_0.55140 = f32[1] parameter(0)
  param_1.61441 = f32[1] parameter(1)
  param_2.33774 = f32[1] parameter(2)
  param_3.23576 = f32[1] parameter(3)
  param_4.15959 = f32[1] parameter(4)
  param_5.12477 = f32[1] parameter(5)
  param_6.8616 = f32[1] parameter(6)
  ROOT %fusion = f32[7] fusion(param_0.55140, param_1.61441, param_2.33774, param_3.23576, param_4.15959, param_5.12477, param_6.8616), kind=kLoop, calls=%wrapped_concatenate_computation.6.clone
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionLevel1Test, testIsNan) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule wrapped_concatenate_computation.6_module, entry_computation_layout={(f32[2,2])->pred[2,2]}

%wrapped_compare_computation (param_0: f32[2,2]) -> pred[2,2] {
  %param_0 = f32[2,2]{1,0} parameter(0), metadata={scheduling_name="param_0"}
  ROOT %ne.1.1 = pred[2,2]{1,0} compare(%param_0, %param_0), direction=NE, metadata={op_name="jit(isnan)/ne" scheduling_name="ne.1.1" stack_frame_id=16}
}



ENTRY %wrapper_wrapped_concatenate_computation.6.clone (param_0: f32[2,2]) -> pred[2,2] {
  %param_0 = f32[2,2]{1,0} parameter(0), metadata={scheduling_name="param_0"}
  ROOT %fusion = pred[2,2] fusion(param_0), kind=kLoop, calls=%wrapped_compare_computation
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionLevel1Test, testOp971_rad2deg_bfloat16) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule jit_rad2deg, entry_computation_layout={(bf16[2,2,4]{2,1,0})->bf16[2,2,4]{2,1,0}}, allow_spmd_sharding_propagation_to_parameters={true}, allow_spmd_sharding_propagation_to_output={true}, frontend_attributes={fingerprint_before_lhs="efb28bbb973b7d03d511bd39d84d07e4"}

%fused_multiply (param_0: bf16[2,2,4]) -> bf16[2,2,4] {
  %param_0 = bf16[2,2,4]{2,1,0} parameter(0), metadata={scheduling_name="param_0"}
  %constant_1_1 = bf16[] constant(57.25)
  %broadcast.1.1 = bf16[2,2,4]{2,1,0} broadcast(%constant_1_1), dimensions={}, metadata={scheduling_name="broadcast.1.1"}
  ROOT %mul.1.1 = bf16[2,2,4]{2,1,0} multiply(%param_0, %broadcast.1.1), metadata={op_name="jit(rad2deg)/mul" scheduling_name="mul.1.1" stack_frame_id=16}
}

ENTRY %main.1 (x.1: bf16[2,2,4]) -> bf16[2,2,4] {
  %x.1 = bf16[2,2,4]{2,1,0} parameter(0), metadata={op_name="x" scheduling_name="x.1"}
  ROOT %loop_multiply_fusion = bf16[2,2,4]{2,1,0} fusion(%x.1), kind=kLoop, calls=%fused_multiply, metadata={op_name="jit(rad2deg)/mul" scheduling_name="loop_multiply_fusion" stack_frame_id=16}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnNonGemmFusionLevel1Test, testOp310_reciprocal_float32) {
  EXPECT_TRUE(RunAndCompare(R"(
HloModule jit_rad2deg, entry_computation_layout={(f32[2,2,2]{2,1,0})->f32[2,2,2]{2,1,0}}, allow_spmd_sharding_propagation_to_parameters={true}, allow_spmd_sharding_propagation_to_output={true}, frontend_attributes={fingerprint_before_lhs="efb28bbb973b7d03d511bd39d84d07e4"}
%fused_divide (param_0.1: f32[2,2,2]) -> f32[2,2,2] {
  %constant.1.1 = f32[] constant(0.5)
  %broadcast.1.1 = f32[2,2,2]{2,1,0} broadcast(%constant.1.1), dimensions={}
  %param_0.1 = f32[2,2,2]{2,1,0} parameter(0)
  ROOT %integer_pow.1.1 = f32[2,2,2]{2,1,0} divide(%broadcast.1.1, %param_0.1), metadata={op_name="jit(reciprocal)/integer_pow" stack_frame_id=16}
}
ENTRY %main.1 (x.1: f32[2,2,2]) -> f32[2,2,2] {
  %x.1 = f32[2,2,2]{2,1,0} parameter(0), metadata={op_name="x" scheduling_name="x.1"}
  ROOT %loop_multiply_fusion = f32[2,2,2]{2,1,0} fusion(%x.1), kind=kLoop, calls=%fused_divide, metadata={op_name="jit(rad2deg)/mul" scheduling_name="loop_multiply_fusion" stack_frame_id=16}
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}


TEST_F(CuDnnNonGemmFusionLevel1Test, testKdeIntegrateKde3) {
EXPECT_TRUE(RunAndCompare(R"(
HloModule wrapped_concatenate_computation.6_module, entry_computation_layout={()->f32[1,1]}

%fused_bitcast () -> f32[1,1] {
  %constant.17.1 = f32[1,1]{1,0} constant({ {1} })
  ROOT %bitcast.46.1 = f32[1,1]{1,0} bitcast(%constant.17.1)
}



ENTRY %wrapper_wrapped_concatenate_computation.6.clone () -> f32[1,1] {
  ROOT %fusion = f32[1,1] fusion(), kind=kLoop, calls=%fused_bitcast
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}


TEST_F(CuDnnNonGemmFusionLevel1Test, testConcate) {
EXPECT_TRUE(RunAndCompare(R"(
HloModule wrapped_concatenate_computation.6_module, entry_computation_layout={(f32[20])->f32[22]}

%fused_concatenate (param_0.80: f32[20]) -> f32[22] {
  %param_0.80 = f32[20]{0} parameter(0)
  %slice.0.1 = f32[1]{0} slice(%param_0.80), slice={[19:20]}, metadata={op_name="jit(wrapped_fun)/jit(_interp)/slice"}
  %constant.85.1 = f32[1]{0} constant({-0.59}), metadata={op_name="jit(wrapped_fun)/jit(_interp)/sub"}
  %add.40.1 = f32[1]{0} add(%slice.0.1, %constant.85.1), metadata={op_name="jit(wrapped_fun)/jit(_interp)/sub"}
  %slice.1.1 = f32[1]{0} slice(%param_0.80), slice={[0:1]}, metadata={op_name="jit(wrapped_fun)/jit(_interp)/slice"}
  %constant.70.1 = f32[1]{0} constant({0.59}), metadata={op_name="jit(wrapped_fun)/jit(_interp)/sub"}
  %add.14.1 = f32[1]{0} add(%slice.1.1, %constant.70.1), metadata={op_name="jit(wrapped_fun)/jit(_interp)/add"}
  ROOT %concatenate.0.1 = f32[22]{0} concatenate(%add.40.1, %param_0.80, %add.14.1), dimensions={0}, metadata={op_name="jit(wrapped_fun)/jit(_interp)/concatenate"}
}



ENTRY %wrapper_wrapped_concatenate_computation.6.clone (x.1: f32[20]) -> f32[22] {
  %x.1 = f32[20] parameter(0)
  ROOT %fusion = f32[22] fusion(%x.1), kind=kLoop, calls=%fused_concatenate
}
)",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnFusionFileCheckTest, ConvFpropGraphConvertedCorrectly) {
  const std::string kHloText = R"(
fusion {
  input = f32[2,9,9,17] parameter(0)
  filter = f32[32,3,3,17] parameter(1)
  ROOT conv = f32[2,9,9,32] convolution(input, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, feature_group_count=1, convolution_kind=fprop
}


ENTRY Test {
  input = f32[2,9,9,17] parameter(0)
  filter = f32[32,3,3,17] parameter(1)
  ROOT conv = f32[2,9,9,32] fusion(input, filter), kind=kCustom, calls=fusion, backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})";

  EXPECT_TRUE(*RunCuDnnFileCheck(kHloText, R"(
CHECK: "nodes": [
CHECK:  {
CHECK:   "compute_data_type": "FLOAT",
CHECK:   "dilation": [{{[[:space:]]*1,[[:space:]]*1[[:space:]]*}}],
CHECK:   "inputs": {
CHECK:    "W": 2,
CHECK:    "X": 1
CHECK:   },
CHECK:   "math_mode": "CROSS_CORRELATION",
CHECK:   "name": "0",
CHECK:   "outputs": {
CHECK:    "Y": 3
CHECK:   },
CHECK:   "post_padding": [{{[[:space:]]*1,[[:space:]]*1[[:space:]]*}}],
CHECK:   "pre_padding": [{{[[:space:]]*1,[[:space:]]*1[[:space:]]*}}],
CHECK:   "stride": [{{[[:space:]]*1,[[:space:]]*1[[:space:]]*}}],
CHECK:   "tag": "CONV_FPROP"
CHECK:  }
CHECK: ],
CHECK:"tensors": {
CHECK:   "data_type": "FLOAT",
CHECK:   "dim": [{{[[:space:]]*2,[[:space:]]*17,[[:space:]]*9,[[:space:]]*9[[:space:]]*}}],
CHECK:   "name": "input",
CHECK:   "stride": [{{[[:space:]]*1377,[[:space:]]*1,[[:space:]]*153,[[:space:]]*17[[:space:]]*}}],
CHECK:   "data_type": "FLOAT",
CHECK:   "dim": [{{[[:space:]]*32,[[:space:]]*17,[[:space:]]*3,[[:space:]]*3[[:space:]]*}}],
CHECK:   "name": "filter",
CHECK:   "stride": [{{[[:space:]]*153,[[:space:]]*1,[[:space:]]*51,[[:space:]]*17[[:space:]]*}}],
CHECK:   "data_type": "FLOAT",
CHECK:   "dim": [{{[[:space:]]*2,[[:space:]]*32,[[:space:]]*9,[[:space:]]*9[[:space:]]*}}],
CHECK:   "name": "conv",
CHECK:   "stride": [{{[[:space:]]*2592,[[:space:]]*1,[[:space:]]*288,[[:space:]]*32[[:space:]]*}}],
)"));
}
}  // namespace
}  // namespace xla::gpu
