/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/hlo_emitter_parameters_autotuner.h"

#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info.h"
#include "xla/service/gpu/instruction_fusion.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/env.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace gpu {
namespace {

class HloEmitterParametersAutotunerTest : public HloTestBase {
 public:
  HloEmitterParametersAutotunerTest()
      : HloTestBase(/*verifier_layout_sensitive=*/true,
                    /*allow_mixed_precision_in_hlo_verifier=*/false) {}

  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    return debug_options;
  }

  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }

  void CheckHloEmitterParametersAutotuning(absl::string_view hlo,
                                           absl::string_view expected) {
    GpuDeviceInfo gpu_device_info =
        GetGpuDeviceInfo(backend().default_stream_executor());
    HloPassPipeline pipeline("loop_fusion_autotuning");
    pipeline.AddPass<GpuInstructionFusion>(/*may_duplicate=*/true,
                                           gpu_device_info);
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "",
                                        tsl::port::MaxParallelism());
    DebugOptions opts;
    pipeline.AddPass<HloEmitterParametersAutotuner>(
        AutotuneConfig{DeviceConfig{backend().default_stream_executor(),
                                    backend().memory_allocator()},
                       opts},
        &thread_pool);

    RunAndFilecheckHloRewrite(
        hlo, std::move(pipeline), expected, [](const HloModule* m) {
          VLOG(5) << m->ToString();
          const HloInstruction* loop_fusion =
              m->entry_computation()->root_instruction();
          CHECK_EQ(loop_fusion->opcode(), HloOpcode::kFusion);
          CHECK_GT(loop_fusion->backend_config<xla::gpu::FusionBackendConfig>()
                       .value()
                       .loop_fusion_config()
                       .unroll_factor(),
                   0);
        });
  }
};

TEST_F(HloEmitterParametersAutotunerTest, AutotunePassAddsUnrollFactor) {
  const std::string hlo = R"(
    HloModule module
     add {
       rhs.1 = f32[] parameter(1)
       lhs.1 = f32[] parameter(0)
       ROOT add.1 = f32[] add(lhs.1, rhs.1)
     }

     fused_computation {
       param_0.2 = bf16[4,480,16]{2,1,0} parameter(0)
       convert.18 = f32[4,480,16]{2,1,0} convert(param_0.2)
       constant_1 = bf16[] constant(0)
       convert.17 = f32[] convert(constant_1)
       reduce.1 = f32[480,16]{1,0} reduce(convert.18, convert.17), dimensions={0},
         to_apply=add
       ROOT convert.16 = bf16[480,16]{1,0} convert(reduce.1)
     }

     ENTRY e {
       p0 = bf16[4,480,16]{2,1,0} parameter(0)
       ROOT fusion.1 = bf16[480,16]{1,0} fusion(p0), kind=kLoop,
         calls=fused_computation
     }
    )";
  CheckHloEmitterParametersAutotuning(hlo, R"(
// CHECK: backend_config={"kind":"","loop_fusion_config":{"unroll_factor"
)");

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(hlo, ErrorSpec{/*aabs=*/5e-3, /*arel=*/5e-3}));
}

TEST_F(HloEmitterParametersAutotunerTest,
       BackendPropagatesAndUsesUnrollFactor) {
  const std::string hlo = R"(
    HloModule module
     add {
       rhs.1 = f32[] parameter(1)
       lhs.1 = f32[] parameter(0)
       ROOT add.1 = f32[] add(lhs.1, rhs.1)
     }

     fused_computation {
       param_0.2 = bf16[4,480,16]{2,1,0} parameter(0)
       convert.18 = f32[4,480,16]{2,1,0} convert(param_0.2)
       constant_1 = bf16[] constant(0)
       convert.17 = f32[] convert(constant_1)
       reduce.1 = f32[480,16]{1,0} reduce(convert.18, convert.17), dimensions={0},
         to_apply=add
       ROOT convert.16 = bf16[480,16]{1,0} convert(reduce.1)
     }

     ENTRY e {
       p0 = bf16[4,480,16]{2,1,0} parameter(0)
       ROOT fusion.1 = bf16[480,16]{1,0} fusion(p0), kind=kLoop,
         calls=fused_computation, backend_config={"kind":"",
         "loop_fusion_config":{"unroll_factor":"16"}}
     }

    )";

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(hlo, ErrorSpec{/*aabs=*/5e-3, /*arel=*/5e-3}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
