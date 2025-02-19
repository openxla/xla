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

#include "xla/service/gpu/transforms/gemm_fusion_swap_operands.h"

#include <gtest/gtest.h>
#include "xla/hlo/testlib/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class SwapOperandsTest : public HloTestBase {};

TEST_F(SwapOperandsTest, CodeGeneratingMovesToLhs) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule DotLayout

fcomp {
    p0 = bf16[64,768,320]{2,1,0} parameter(0)

    p1 = s4[64,448,320]{2,1,0} parameter(1)
    p1.c = bf16[64,448,320]{2,1,0} convert(p1)

    ROOT dot = bf16[64,768,448]{2,1,0} dot(p0, p1.c),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={2}
}

ENTRY main {
  p0 = bf16[64,768,320]{2,1,0} parameter(0)
  p1 = s4[64,448,320]{2,1,0} parameter(1)
  ROOT fusion = bf16[64,768,448]{2,1,0} fusion(p0, p1),
    kind=kCustom, calls=fcomp,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})");
  EXPECT_TRUE(GemmFusionSwapOperands().Run(module->get()).value());
  EXPECT_TRUE(*RunFileCheck(module->get()->ToString(), R"(
CHECK: bf16[64,448,768]{1,2,0} dot
CHECK-NEXT: bf16[64,768,448]{2,1,0} bitcast)"));
}

TEST_F(SwapOperandsTest, CodeGeneratingMovesToLhsMultipleNoncontracting) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule DotLayout

fcomp {
    p0 = bf16[768,96,320]{2,1,0} parameter(0)
    p0.r = bf16[73728,320]{1,0} reshape(p0)

    p1 = s4[448,320]{1,0} parameter(1)
    p1.c = bf16[448,320]{1,0} convert(p1)

    dot = bf16[73728,448]{1,0} dot(p0.r, p1.c),
      lhs_contracting_dims={1},
      rhs_contracting_dims={1}

    ROOT res = bf16[768,96,448]{2,1,0} bitcast(dot)
}

ENTRY main {
  p0 = bf16[768,96,320]{2,1,0} parameter(0)
  p1 = s4[448,320]{1,0} parameter(1)
  ROOT fusion = bf16[768,96,448]{2,1,0} fusion(p0, p1),
    kind=kCustom, calls=fcomp,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})");
  EXPECT_TRUE(GemmFusionSwapOperands().Run(module->get()).value());
  EXPECT_TRUE(*RunFileCheck(module->get()->ToString(), R"(
CHECK: bf16[448,73728]{0,1} dot
CHECK-NEXT: bf16[73728,448]{1,0} bitcast)"));
}

TEST_F(SwapOperandsTest, SplitNoncontractingIsKeptInLhs) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule DotLayout

fcomp {
    p0 = bf16[768,320,96]{2,1,0} parameter(0)

    p1 = s4[448,320]{1,0} parameter(1)
    p1.c = bf16[448,320]{1,0} convert(p1)

    ROOT dot = bf16[768,96,448]{2,1,0} dot(p0, p1.c),
      lhs_contracting_dims={1},
      rhs_contracting_dims={1}
}

ENTRY main {
  p0 = bf16[768,320,96]{2,1,0} parameter(0)
  p1 = s4[448,320]{1,0} parameter(1)
  ROOT fusion = bf16[768,96,448]{2,1,0} fusion(p0, p1),
    kind=kCustom, calls=fcomp,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})");
  EXPECT_FALSE(GemmFusionSwapOperands().Run(module->get()).value());
}

TEST_F(SwapOperandsTest, DoNotSwapSmallRhsNoncontracting) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule DotLayout

fcomp {
    p0 = bf16[64,768,320]{2,1,0} parameter(0)

    p1 = s4[64,32,320]{2,1,0} parameter(1)
    p1.c = bf16[64,32,320]{2,1,0} convert(p1)

    ROOT dot = bf16[64,768,32]{2,1,0} dot(p0, p1.c),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={2}
}

ENTRY main {
  p0 = bf16[64,768,320]{2,1,0} parameter(0)
  p1 = s4[64,32,320]{2,1,0} parameter(1)
  ROOT fusion = bf16[64,768,32]{2,1,0} fusion(p0, p1),
    kind=kCustom, calls=fcomp,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})");
  EXPECT_FALSE(GemmFusionSwapOperands().Run(module->get()).value());
}

TEST_F(SwapOperandsTest, BothNonCodeGeneratingSwapSmallLhs) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule DotLayout

fcomp {
    p0 = bf16[64,32,320]{2,1,0} parameter(0)
    p1 = bf16[64,448,320]{2,1,0} parameter(1)

    ROOT dot = bf16[64,32,448]{2,1,0} dot(p0, p1),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={2}
}

ENTRY main {
  p0 = bf16[64,32,320]{2,1,0} parameter(0)
  p1 = bf16[64,448,320]{2,1,0} parameter(1)
  ROOT fusion = bf16[64,32,448]{2,1,0} fusion(p0, p1),
    kind=kCustom, calls=fcomp,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})");
  EXPECT_TRUE(GemmFusionSwapOperands().Run(module->get()).value());
  EXPECT_TRUE(*RunFileCheck(module->get()->ToString(), R"(
CHECK: bf16[64,448,32]{1,2,0} dot
CHECK-NEXT: bf16[64,32,448]{2,1,0} bitcast)"));
}

TEST_F(SwapOperandsTest, BothCodeGeneratingSwapSmallLhs) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule DotLayout

fcomp {
    p0 = s4[64,32,320]{2,1,0} parameter(0)
    p0.c = bf16[64,32,320]{2,1,0} convert(p0)
    p1 = s4[64,448,320]{2,1,0} parameter(1)
    p1.c = bf16[64,448,320]{2,1,0} convert(p1)

    ROOT dot = bf16[64,32,448]{2,1,0} dot(p0.c, p1.c),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={2}
}

ENTRY main {
  p0 = s4[64,32,320]{2,1,0} parameter(0)
  p1 = s4[64,448,320]{2,1,0} parameter(1)
  ROOT fusion = bf16[64,32,448]{2,1,0} fusion(p0, p1),
    kind=kCustom, calls=fcomp,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})");
  EXPECT_TRUE(GemmFusionSwapOperands().Run(module->get()).value());
  EXPECT_TRUE(*RunFileCheck(module->get()->ToString(), R"(
CHECK: bf16[64,448,32]{1,2,0} dot
CHECK-NEXT: bf16[64,32,448]{2,1,0} bitcast)"));
}

TEST_F(SwapOperandsTest, BothNonCodeGeneratingDoNotSwapIfBothSmall) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule DotLayout

fcomp {
    p0 = bf16[64,32,320]{2,1,0} parameter(0)
    p1 = bf16[64,48,320]{2,1,0} parameter(1)

    ROOT dot = bf16[64,32,48]{2,1,0} dot(p0, p1),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={2}
}

ENTRY main {
  p0 = bf16[64,32,320]{2,1,0} parameter(0)
  p1 = bf16[64,48,320]{2,1,0} parameter(1)
  ROOT fusion = bf16[64,32,48]{2,1,0} fusion(p0, p1),
    kind=kCustom, calls=fcomp,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})");
  EXPECT_FALSE(GemmFusionSwapOperands().Run(module->get()).value());
}

TEST_F(SwapOperandsTest, MultipleParameterIsFine) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule MultipleParameterIsFine

fcomp {
  %parameter_0.7 = bf16[1,8,8,640]{3,2,1,0} parameter(0)
  %bitcast.67601 = bf16[8,5120]{1,0} bitcast(bf16[1,8,8,640]{3,2,1,0} %parameter_0.7)
  %parameter_1.7 = s8[5120,5120]{1,0} parameter(1)
  %parameter_2.2 = s8[5120,5120]{1,0} parameter(2)
  %concatenate.2342 = s8[5120,10240]{1,0} concatenate(s8[5120,5120]{1,0} %parameter_1.7, s8[5120,5120]{1,0} %parameter_2.2), dimensions={1}
  %convert.17053 = bf16[5120,10240]{1,0} convert(s8[5120,10240]{1,0} %concatenate.2342)
  ROOT %dot.2515 = bf16[8,10240]{1,0} dot(bf16[8,5120]{1,0} %bitcast.67601, bf16[5120,10240]{1,0} %convert.17053), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  %multiply.27185 = bf16[1,8,8,640]{3,2,1,0} parameter(0)
  %param.391 = s8[5120,5120]{1,0} parameter(1)
  %param.393 = s8[5120,5120]{1,0} parameter(2)
  ROOT %micro_kernel = bf16[8,10240]{1,0} fusion(bf16[1,8,8,640]{3,2,1,0} %multiply.27185, s8[5120,5120]{1,0} %param.391, s8[5120,5120]{1,0} %param.393), kind=kCustom, calls=fcomp,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})");
  EXPECT_TRUE(GemmFusionSwapOperands().Run(module->get()).value());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
