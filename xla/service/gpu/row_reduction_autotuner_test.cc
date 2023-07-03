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

#include "xla/service/gpu/row_reduction_autotuner.h"

#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/instruction_fusion.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

class RowReductionAutotunerTest : public HloTestBase {
 public:
  RowReductionAutotunerTest()
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

  void CheckRowReductionAutotuning(absl::string_view hlo,
                                   absl::string_view expected) {
    GpuDeviceInfo gpu_device_info =
        GetGpuDeviceInfo(backend().default_stream_executor());
    HloPassPipeline pipeline("row_reduction_autotuning");
    pipeline.AddPass<GpuInstructionFusion>(/*may_duplicate=*/true,
                                           gpu_device_info);
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "",
                                        tsl::port::MaxParallelism());
    DebugOptions opts;
    pipeline.AddPass<RowReductionAutotuner>(
        AutotuneConfig{DeviceConfig{backend().default_stream_executor(),
                                    backend().memory_allocator()},
                       opts},
        &thread_pool);

    RunAndFilecheckHloRewrite(
        hlo, std::move(pipeline), expected, [](const HloModule* m) {
          VLOG(5) << m->ToString();
          const HloInstruction* reduction_fusion =
              m->entry_computation()->root_instruction();
          if (reduction_fusion->opcode() == HloOpcode::kReduce) {
            reduction_fusion = reduction_fusion->operand(0);
          }
          CHECK_EQ(reduction_fusion->opcode(), HloOpcode::kFusion);
          CHECK_GT(
              reduction_fusion->backend_config<xla::gpu::FusionBackendConfig>()
                  .value()
                  .row_reduction_config()
                  .tile_x(),
              0);
        });
  }
};

TEST_F(RowReductionAutotunerTest, SimpleReduction) {
  const std::string hlo = R"(
 HloModule m

 Sum {
   x.1 = f32[] parameter(0)
   y.1 = f32[] parameter(1)
   ROOT add.1 = f32[] add(x.1, y.1)
 }

 ENTRY reduce.1 {
   p = bf16[512,1024] parameter(0)
   c = f32[512,1024] convert(p)

   i = f32[] constant(0)
   ROOT reduce = f32[512] reduce(c, i), dimensions={1}, to_apply=Sum
 }
)";
  CheckRowReductionAutotuning(hlo, R"(
// CHECK: backend_config={"kind":"","row_reduction_config":{"tile_x"
)");

  EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{/*aabs=*/5e-3, /*arel=*/5e-3}));
}

TEST_F(RowReductionAutotunerTest, MultiOutputReduction) {
  const char* const hlo = R"(
  HloModule m

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT out = f32[] add(a, b)
  }

  fused_computation {
    p = f32[1024,1024]{1,0} parameter(0)
    s = f32[1024,1024]{1,0} sqrt(p)
    z = f32[] constant(0)
    r1 = f32[1024]{0} reduce(p, z), to_apply=add, dimensions={1}
    r2 = f32[1024]{0} reduce(s, z), to_apply=add, dimensions={1}
    ROOT out = (f32[1024]{0}, f32[1024]{0}) tuple(r1, r2)
  }

  ENTRY e {
    p = f32[1024,1024]{1,0} parameter(0)
    ROOT f = (f32[1024]{0}, f32[1024]{0}) fusion(p), kind=kInput, calls=fused_computation
  })";
  CheckRowReductionAutotuning(hlo, R"(
// CHECK: backend_config={"kind":"","row_reduction_config":{"tile_x"
)");

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(hlo, ErrorSpec{/*aabs=*/5e-3, /*arel=*/5e-3}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
