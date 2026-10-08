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

#include <gtest/gtest.h>
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_interpreter_reference_mixin.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

class TempBufferPackingTest
    : public HloInterpreterReferenceMixin<GpuPjRtCodegenTest> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions options = HloInterpreterReferenceMixin<
        GpuPjRtCodegenTest>::GetDebugOptionsForTest();
    options.set_xla_gpu_enable_heap_simulator_packing_search(true);
    return options;
  }
};

TEST_F(TempBufferPackingTest, ReusesStorageAcrossScheduledPhases) {
  // Explicit fusions and a fixed schedule keep four differently sized
  // temporaries alive across successive phases. Execute the assigned offsets
  // on one GPU and compare both outputs with the interpreter.
  constexpr char hlo[] = R"(
HloModule packing, is_scheduled=true

negate_b {
  p = f32[256]{0} parameter(0)
  ROOT b = f32[256]{0} negate(p)
}
abs_c {
  p = f32[320]{0} parameter(0)
  ROOT c = f32[320]{0} abs(p)
}
add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT sum = f32[] add(x, y)
}
reduce_320 {
  p = f32[320]{0} parameter(0)
  zero = f32[] constant(0)
  ROOT r = f32[] reduce(p, zero), dimensions={0}, to_apply=add
}
reduce_256 {
  p = f32[256]{0} parameter(0)
  zero = f32[] constant(0)
  ROOT r = f32[] reduce(p, zero), dimensions={0}, to_apply=add
}
reduce_192 {
  p = f32[192]{0} parameter(0)
  zero = f32[] constant(0)
  ROOT r = f32[] reduce(p, zero), dimensions={0}, to_apply=add
}
broadcast_a {
  p = f32[] parameter(0)
  ROOT a = f32[192]{0} broadcast(p), dimensions={}
}
broadcast_d {
  p = f32[] parameter(0)
  ROOT d = f32[320]{0} broadcast(p), dimensions={}
}
ENTRY main {
  p0 = f32[256]{0} parameter(0)
  p1 = f32[320]{0} parameter(1)
  b = f32[256]{0} fusion(p0), kind=kLoop, calls=negate_b
  c = f32[320]{0} fusion(p1), kind=kLoop, calls=abs_c
  use_c = f32[] fusion(c), kind=kInput, calls=reduce_320
  a = f32[192]{0} fusion(use_c), kind=kLoop, calls=broadcast_a
  use_b = f32[] fusion(b), kind=kInput, calls=reduce_256
  d = f32[320]{0} fusion(use_b), kind=kLoop, calls=broadcast_d
  use_a = f32[] fusion(a), kind=kInput, calls=reduce_192
  use_d = f32[] fusion(d), kind=kInput, calls=reduce_320
  ROOT result = (f32[], f32[]) tuple(use_a, use_d)
}
)";
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo, ErrorSpec{1e-3, 1e-5}));
}

}  // namespace
}  // namespace xla::gpu
