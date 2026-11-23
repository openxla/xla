/* Copyright 2022 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_interpreter_reference_mixin.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {
namespace {

class CpuWhileTest : public HloTestBase {};

// Verifies fix for b/233647273.
TEST_F(CpuWhileTest, While) {
  const std::string hlo_text = R"(
HloModule module

f1 {
  f1.p0 = s32[] parameter(0)
  ROOT f1.sum = s32[] add(f1.p0, f1.p0)
}

f2 {
  f2.p0 = s32[] parameter(0)
  f2.p1 = s32[] parameter(1)
  ROOT f2.sum = s32[] add(f2.p0, f2.p1)
}

body {
  body.p0 = s32[] parameter(0)
  sum2 = s32[] fusion(body.p0), kind=kLoop, calls=f1
  ROOT sum3 = s32[] fusion(sum2, body.p0), kind=kLoop, calls=f2
}

cond {
  cond.p0 = s32[] parameter(0)
  cond.c1 = s32[] constant(1)
  ROOT cond.root = pred[] compare(cond.p0, cond.c1), direction=EQ
}

ENTRY entry {
  entry.c1 = s32[] constant(1)
  ROOT entry.root = s32[] while(entry.c1), condition=cond, body=body
}
)";

  // Compile and execute the computation.
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(const Literal result, Execute(std::move(module), {}));

  // Check the output correctness.
  LiteralTestUtil::ExpectR0Equal(3, result);
}

// Add a small while loop that calls sort to verify that the small call emitter
// can deal with thread local buffers.
TEST_F(CpuWhileTest, WhileSort) {
  const std::string hlo_text = R"(
    HloModule sort_loop

    %sort_comparison (x: f32[], y: f32[]) -> pred[] {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %result = compare(x, y), direction=GT
    }

    %sort_body (input: (f32[4])) -> (f32[4]) {
      %input = f32[4] parameter(0)
      %sorted = sort(%input), dimensions={0}, is_stable=true, to_apply=%sort_comparison
      ROOT %result = tuple(%sorted)
    }

    %while_condition (loop_state: (s32[], f32[4])) -> pred[] {
      %loop_state = (s32[], f32[4]) parameter(0)
      %index = s32[] get-tuple-element(%loop_state), index=0
      %limit = s32[] constant(4)
      ROOT %result = compare(%index, %limit), direction=LT
    }

    %while_body (loop_state: (s32[], f32[4])) -> (s32[], f32[4]) {
      %loop_state = (s32[], f32[4]) parameter(0)
      %index = s32[] get-tuple-element(%loop_state), index=0
      %data = f32[4] get-tuple-element(%loop_state), index=1
      %increment = s32[] constant(1)
      %new_index = add(%index, %increment)
      %sorted_data = sort(%data), dimensions={0}, is_stable=true, to_apply=%sort_comparison
      ROOT %new_state = tuple(%new_index, %sorted_data)
    }

    ENTRY %main (input: f32[4]) -> f32[4] {
      %init_index = s32[] constant(0)
      %input = f32[4] parameter(0)
      %loop_state = (s32[], f32[4]) tuple(%init_index, %input)
      %while_loop = (s32[], f32[4]) while(%loop_state), condition=%while_condition, body=%while_body
      ROOT %final_sorted = f32[4] get-tuple-element(%while_loop), index=1
    }
  )";

  // Compile and execute the computation.
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  Literal input = LiteralUtil::CreateR1<float>({3, 1, 4, 2});

  ASSERT_OK_AND_ASSIGN(const Literal result,
                       Execute(std::move(module), {&input}));

  // Check the output correctness.
  LiteralTestUtil::ExpectR1Equal(absl::MakeConstSpan({4.0f, 3.0f, 2.0f, 1.0f}),
                                 result);
}

// Simply runs a HLO module that was seen to fail before a fix due to an
// attempted deref of a nullptr in eigen runtime dot call. This test is not
// intended to verify the correctness of the result, just that it does not
// result in an error.
TEST_F(CpuWhileTest, WhileDotDoesNotError) {
  constexpr absl::string_view hlo_text = R"(
    HloModule Complex_1.44, entry_computation_layout={(f32[1,2,2]{2,1,0}, f32[1,2,2]{2,1,0})->f32[]}

    %add_F32.39 (lhs.40: f32[], rhs.41: f32[]) -> f32[] {
      %lhs.40 = f32[] parameter(0)
      %rhs.41 = f32[] parameter(1)
      ROOT %add.42 = f32[] add(%lhs.40, %rhs.41)
    }

    %unblocked_body.14 (parameter.15: (s32[], c64[1,2,2], c64[1,2,2], pred[1,1,1])) -> (s32[], c64[1,2,2], c64[1,2,2], pred[1,1,1]) {
      %parameter.15 = (s32[], c64[1,2,2]{2,1,0}, c64[1,2,2]{2,1,0}, pred[1,1,1]{2,1,0}) parameter(0)
      %get-tuple-element.16 = s32[] get-tuple-element(%parameter.15), index=0
      %constant.20 = s32[] constant(1)
      %add.21 = s32[] add(%get-tuple-element.16, %constant.20)
      %get-tuple-element.17 = c64[1,2,2]{2,1,0} get-tuple-element(%parameter.15), index=1
      %iota.23 = s32[1,2,2]{2,1,0} iota(), iota_dimension=1
      %iota.22 = s32[1,2,2]{2,1,0} iota(), iota_dimension=2
      %compare.24 = pred[1,2,2]{2,1,0} compare(%iota.23, %iota.22), direction=GE
      %broadcast.25 = s32[1,2,2]{2,1,0} broadcast(%get-tuple-element.16), dimensions={}
      %compare.26 = pred[1,2,2]{2,1,0} compare(%iota.22, %broadcast.25), direction=EQ
      %and.27 = pred[1,2,2]{2,1,0} and(%compare.24, %compare.26)
      %get-tuple-element.18 = c64[1,2,2]{2,1,0} get-tuple-element(%parameter.15), index=2
      %real.30 = f32[1,2,2]{2,1,0} real(%get-tuple-element.18)
      %imag.31 = f32[1,2,2]{2,1,0} imag(%get-tuple-element.18)
      %negate.32 = f32[1,2,2]{2,1,0} negate(%imag.31)
      %complex.33 = c64[1,2,2]{2,1,0} complex(%real.30, %negate.32)
      %dot.34 = c64[1,2,2]{2,1,0} dot(%get-tuple-element.18, %complex.33), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={2}, operand_precision={highest,highest}, frontend_attributes={grad_x="false",grad_y="false"}
      %transpose.35 = c64[1,2,2]{2,1,0} transpose(%dot.34), dimensions={0,1,2}
      %subtract.36 = c64[1,2,2]{2,1,0} subtract(%get-tuple-element.17, %transpose.35)
      %constant.37 = s32[] constant(0)
      %dynamic-slice.38 = c64[1,1,1]{2,1,0} dynamic-slice(%subtract.36, %constant.37, %get-tuple-element.16, %get-tuple-element.16), dynamic_slice_sizes={1,1,1}
      %real.39 = f32[1,1,1]{2,1,0} real(%dynamic-slice.38)
      %sqrt.40 = f32[1,1,1]{2,1,0} sqrt(%real.39)
      %constant.41 = f32[] constant(0)
      %broadcast.42 = f32[1,1,1]{2,1,0} broadcast(%constant.41), dimensions={}
      %complex.43 = c64[1,1,1]{2,1,0} complex(%sqrt.40, %broadcast.42)
      %reshape.46 = c64[1]{0} reshape(%complex.43)
      %broadcast.47 = c64[1,2,2]{2,1,0} broadcast(%reshape.46), dimensions={0}
      %divide.48 = c64[1,2,2]{2,1,0} divide(%subtract.36, %broadcast.47)
      %constant.28 = c64[] constant((0, 0))
      %broadcast.29 = c64[1,2,2]{2,1,0} broadcast(%constant.28), dimensions={}
      %select.49 = c64[1,2,2]{2,1,0} select(%and.27, %divide.48, %broadcast.29)
      %add.50 = c64[1,2,2]{2,1,0} add(%select.49, %get-tuple-element.18)
      %get-tuple-element.19 = pred[1,1,1]{2,1,0} get-tuple-element(%parameter.15), index=3
      %compare.44 = pred[1,1,1]{2,1,0} compare(%sqrt.40, %sqrt.40), direction=NE
      %or.45 = pred[1,1,1]{2,1,0} or(%get-tuple-element.19, %compare.44)
      ROOT %tuple.51 = (s32[], c64[1,2,2]{2,1,0}, c64[1,2,2]{2,1,0}, pred[1,1,1]{2,1,0}) tuple(%add.21, %get-tuple-element.17, %add.50, %or.45)
    }

    %unblocked_condition.52 (parameter.53: (s32[], c64[1,2,2], c64[1,2,2], pred[1,1,1])) -> pred[] {
      %parameter.53 = (s32[], c64[1,2,2]{2,1,0}, c64[1,2,2]{2,1,0}, pred[1,1,1]{2,1,0}) parameter(0)
      %get-tuple-element.55 = c64[1,2,2]{2,1,0} get-tuple-element(%parameter.53), index=1
      %get-tuple-element.56 = c64[1,2,2]{2,1,0} get-tuple-element(%parameter.53), index=2
      %get-tuple-element.57 = pred[1,1,1]{2,1,0} get-tuple-element(%parameter.53), index=3
      %get-tuple-element.54 = s32[] get-tuple-element(%parameter.53), index=0
      %constant.58 = s32[] constant(2)
      ROOT %compare.59 = pred[] compare(%get-tuple-element.54, %constant.58), direction=LT
    }

    %xla.cholesky_c64_1_2_2__lower.76 (a.1: c64[1,2,2]) -> c64[1,2,2] {
      %constant.12 = s32[] constant(0)
      %a.1 = c64[1,2,2]{2,1,0} parameter(0)
      %slice.6 = c64[1,2,2]{2,1,0} slice(%a.1), slice={[0:1], [0:2], [0:2]}
      %slice.7 = c64[1,2,2]{2,1,0} slice(%slice.6), slice={[0:1], [0:2], [0:2]}
      %constant.8 = c64[] constant((0, 0))
      %broadcast.9 = c64[1,2,2]{2,1,0} broadcast(%constant.8), dimensions={}
      %constant.10 = pred[] constant(false)
      %broadcast.11 = pred[1,1,1]{2,1,0} broadcast(%constant.10), dimensions={}
      %tuple.13 = (s32[], c64[1,2,2]{2,1,0}, c64[1,2,2]{2,1,0}, pred[1,1,1]{2,1,0}) tuple(%constant.12, %slice.7, %broadcast.9, %broadcast.11)
      %while.60 = (s32[], c64[1,2,2]{2,1,0}, c64[1,2,2]{2,1,0}, pred[1,1,1]{2,1,0}) while(%tuple.13), condition=%unblocked_condition.52, body=%unblocked_body.14
      %get-tuple-element.61 = s32[] get-tuple-element(%while.60), index=0
      %get-tuple-element.62 = c64[1,2,2]{2,1,0} get-tuple-element(%while.60), index=1
      %constant.4 = pred[] constant(false)
      %broadcast.5 = pred[1,1,1]{2,1,0} broadcast(%constant.4), dimensions={}
      %get-tuple-element.64 = pred[1,1,1]{2,1,0} get-tuple-element(%while.60), index=3
      %or.65 = pred[1,1,1]{2,1,0} or(%broadcast.5, %get-tuple-element.64)
      %broadcast.70 = pred[1,1,1]{2,1,0} broadcast(%or.65), dimensions={0,1,2}
      %reshape.71 = pred[1]{0} reshape(%broadcast.70)
      %broadcast.72 = pred[1,2,2]{2,1,0} broadcast(%reshape.71), dimensions={0}
      %constant.73 = c64[] constant((nan, 0))
      %broadcast.74 = c64[1,2,2]{2,1,0} broadcast(%constant.73), dimensions={}
      %constant.2 = c64[] constant((0, 0))
      %broadcast.3 = c64[1,2,2]{2,1,0} broadcast(%constant.2), dimensions={}
      %get-tuple-element.63 = c64[1,2,2]{2,1,0} get-tuple-element(%while.60), index=2
      %constant.66 = s32[] constant(0)
      %constant.67 = s32[] constant(0)
      %constant.68 = s32[] constant(0)
      %dynamic-update-slice.69 = c64[1,2,2]{2,1,0} dynamic-update-slice(%broadcast.3, %get-tuple-element.63, %constant.66, %constant.67, %constant.68)
      ROOT %select.75 = c64[1,2,2]{2,1,0} select(%broadcast.72, %broadcast.74, %dynamic-update-slice.69)
    }

    ENTRY %Complex_1.44 (input_real.1: f32[1,2,2], input_imag.2: f32[1,2,2]) -> f32[] {
      %input_real.1 = f32[1,2,2]{2,1,0} parameter(0)
      %input_imag.2 = f32[1,2,2]{2,1,0} parameter(1)
      %complex.3 = c64[1,2,2]{2,1,0} complex(%input_real.1, %input_imag.2)
      %real.4 = f32[1,2,2]{2,1,0} real(%complex.3)
      %imag.5 = f32[1,2,2]{2,1,0} imag(%complex.3)
      %negate.6 = f32[1,2,2]{2,1,0} negate(%imag.5)
      %complex.7 = c64[1,2,2]{2,1,0} complex(%real.4, %negate.6)
      %transpose.8 = c64[1,2,2]{1,2,0} transpose(%complex.7), dimensions={0,2,1}
      %dot.9 = c64[1,2,2]{2,1,0} dot(%complex.3, %transpose.8), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}, operand_precision={highest,highest}, frontend_attributes={grad_x="false",grad_y="false"}
      %transpose.10 = c64[1,2,2]{2,1,0} transpose(%dot.9), dimensions={0,1,2}
      %iota.13 = s32[2]{0} iota(), iota_dimension=0
      %constant.14 = s32[] constant(0)
      %broadcast.15 = s32[2]{0} broadcast(%constant.14), dimensions={}
      %add.16 = s32[2]{0} add(%iota.13, %broadcast.15)
      %broadcast.18 = s32[2,2]{1,0} broadcast(%add.16), dimensions={0}
      %iota.12 = s32[2]{0} iota(), iota_dimension=0
      %broadcast.17 = s32[2,2]{1,0} broadcast(%iota.12), dimensions={1}
      %compare.19 = pred[2,2]{1,0} compare(%broadcast.18, %broadcast.17), direction=GE
      %broadcast.20 = pred[1,2,2]{2,1,0} broadcast(%compare.19), dimensions={1,2}
      %call = c64[1,2,2]{2,1,0} call(%transpose.10), to_apply=%xla.cholesky_c64_1_2_2__lower.76
      %constant.21 = c64[] constant((0, 0))
      %broadcast.22 = c64[1,2,2]{2,1,0} broadcast(%constant.21), dimensions={}
      %select.23 = c64[1,2,2]{2,1,0} select(%broadcast.20, %call, %broadcast.22)
      %real.24 = f32[1,2,2]{2,1,0} real(%select.23)
      %imag.25 = f32[1,2,2]{2,1,0} imag(%select.23)
      %negate.26 = f32[1,2,2]{2,1,0} negate(%imag.25)
      %complex.27 = c64[1,2,2]{2,1,0} complex(%real.24, %negate.26)
      %transpose.28 = c64[1,2,2]{1,2,0} transpose(%complex.27), dimensions={0,2,1}
      %dot.29 = c64[1,2,2]{2,1,0} dot(%select.23, %transpose.28), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}, operand_precision={highest,highest}, frontend_attributes={grad_x="false",grad_y="false"}
      %transpose.30 = c64[1,2,2]{2,1,0} transpose(%dot.29), dimensions={0,1,2}
      %subtract.31 = c64[1,2,2]{2,1,0} subtract(%transpose.10, %transpose.30)
      %real.32 = f32[1,2,2]{2,1,0} real(%subtract.31)
      %imag.33 = f32[1,2,2]{2,1,0} imag(%subtract.31)
      %negate.34 = f32[1,2,2]{2,1,0} negate(%imag.33)
      %complex.35 = c64[1,2,2]{2,1,0} complex(%real.32, %negate.34)
      %multiply.36 = c64[1,2,2]{2,1,0} multiply(%subtract.31, %complex.35)
      %abs.37 = f32[1,2,2]{2,1,0} abs(%multiply.36)
      %constant.38 = f32[] constant(0)
      ROOT %reduce.43 = f32[] reduce(%abs.37, %constant.38), dimensions={0,1,2}, to_apply=%add_F32.39
    }
  )";

  // No need to check the values of the Literal, just that it returns a valid
  // result.
  Literal input_real = LiteralUtil::CreateR3<float>({{{1, 2}, {3, 4}}});
  Literal input_imag = LiteralUtil::CreateR3<float>({{{9, 10}, {11, 12}}});

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  EXPECT_OK(Execute(std::move(module), {&input_real, &input_imag}));
}

TEST_F(CpuWhileTest, FusedDynamicUpdateSliceInSmallWhileLoop) {
  const std::string hlo_text = R"(
    HloModule FusedDynamicUpdateSliceInSmallWhileLoop

    fused_dus {
      param_data = f32[8] parameter(0)
      param_update = f32[2] parameter(1)
      param_idx = s32[] parameter(2)
      ROOT dus = f32[8] dynamic-update-slice(param_data, param_update, param_idx)
    }

    body {
      loop_state = (s32[], f32[8]) parameter(0)
      idx = s32[] get-tuple-element(loop_state), index=0
      data = f32[8] get-tuple-element(loop_state), index=1
      one = s32[] constant(1)
      next_idx = s32[] add(idx, one)
      update = f32[2] constant({10.0, 20.0})
      fused = f32[8] fusion(data, update, idx), kind=kLoop, calls=fused_dus
      ROOT new_state = (s32[], f32[8]) tuple(next_idx, fused)
    }

    cond {
      loop_state = (s32[], f32[8]) parameter(0)
      idx = s32[] get-tuple-element(loop_state), index=0
      limit = s32[] constant(3)
      ROOT cmp = pred[] compare(idx, limit), direction=LT
    }

    ENTRY main {
      init_idx = s32[] constant(0)
      zero = f32[] constant(0.0)
      init_data = f32[8] broadcast(zero), dimensions={}
      init_state = (s32[], f32[8]) tuple(init_idx, init_data)
      while_loop = (s32[], f32[8]) while(init_state), condition=cond, body=body
      ROOT result = f32[8] get-tuple-element(while_loop), index=1
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(const Literal result, Execute(std::move(module), {}));

  LiteralTestUtil::ExpectR1Equal(
      absl::MakeConstSpan({10.0f, 10.0f, 10.0f, 20.0f, 0.0f, 0.0f, 0.0f, 0.0f}),
      result);
}

// The two loops share one kernel, which must use each call's own buffers.
TEST_F(CpuWhileTest, IdenticalSmallWhileLoopsSharedKernelCorrect) {
  constexpr absl::string_view hlo_text = R"(
    HloModule IdenticalSmallWhileLoops

    body {
      state = (s32[], f32[4]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      x = f32[4] get-tuple-element(state), index=1
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      c = f32[] constant(2)
      cb = f32[4] broadcast(c), dimensions={}
      sum = f32[4] add(x, cb)
      next_x = f32[4] multiply(sum, cb)
      ROOT next_state = (s32[], f32[4]) tuple(next_i, next_x)
    }

    cond {
      state = (s32[], f32[4]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      limit = s32[] constant(5)
      ROOT lt = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      zero = s32[] constant(0)
      init0 = (s32[], f32[4]) tuple(zero, p0)
      init1 = (s32[], f32[4]) tuple(zero, p1)
      while0 = (s32[], f32[4]) while(init0), condition=cond, body=body
      while1 = (s32[], f32[4]) while(init1), condition=cond, body=body
      r0 = f32[4] get-tuple-element(while0), index=1
      r1 = f32[4] get-tuple-element(while1), index=1
      ROOT r = (f32[4], f32[4]) tuple(r0, r1)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  Literal p0 = LiteralUtil::CreateR1<float>({0, 1, 2, 3});
  Literal p1 = LiteralUtil::CreateR1<float>({10, 20, 30, 40});
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&p0, &p1}));

  // Five iterations of x = (x + 2) * 2 give 32 * x + 124.
  std::vector<Literal> results = result.DecomposeTuple();
  ASSERT_EQ(results.size(), 2);
  LiteralTestUtil::ExpectR1Equal(
      absl::MakeConstSpan({124.0f, 156.0f, 188.0f, 220.0f}), results[0]);
  LiteralTestUtil::ExpectR1Equal(
      absl::MakeConstSpan({444.0f, 764.0f, 1084.0f, 1404.0f}), results[1]);
}

// Small loops become one kernel running the thunks of the loop. These bodies
// exercise each kind of thunk inside that kernel.
class CpuSmallWhileLoopTest : public HloInterpreterReferenceMixin<HloTestBase> {
 protected:
  // Checks that the loop compiled to one kernel, since a fallback to a call
  // thunk would still produce correct results.
  void ExpectOneKernel(absl::string_view hlo_text,
                       std::optional<int64_t> intra_op_threads = std::nullopt,
                       bool is_aot_compile = false) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(hlo_text));
    if (intra_op_threads.has_value()) {
      module->mutable_config().set_intra_op_parallelism_threads(
          *intra_op_threads);
    }
    Compiler::CompileOptions options;
    options.is_aot_compile = is_aot_compile;
    CpuCompiler compiler;
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> optimized,
        compiler.RunHloPasses(std::move(module), /*stream_exec=*/nullptr,
                              options));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                         compiler.RunBackend(std::move(optimized),
                                             /*stream_exec=*/nullptr, options));
    int64_t num_kernels = 0;
    for (const auto& thunk : absl::down_cast<CpuExecutable*>(executable.get())
                                 ->thunks()
                                 .thunk_sequence()) {
      EXPECT_NE(thunk->kind(), Thunk::Kind::kWhile);
      EXPECT_NE(thunk->kind(), Thunk::Kind::kCall);
      num_kernels += thunk->kind() == Thunk::Kind::kKernel;
    }
    EXPECT_EQ(num_kernels, 1);
  }

  void ExpectBackendNoCallThunk(
      absl::string_view hlo_text, int64_t expected_kernels,
      std::optional<int64_t> intra_op_threads = std::nullopt,
      bool is_aot_compile = false) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(hlo_text));
    if (intra_op_threads.has_value()) {
      module->mutable_config().set_intra_op_parallelism_threads(
          *intra_op_threads);
    }
    Compiler::CompileOptions options;
    options.is_aot_compile = is_aot_compile;
    CpuCompiler compiler;
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                         compiler.RunBackend(std::move(module),
                                             /*stream_exec=*/nullptr, options));
    const ThunkSequence& thunks =
        absl::down_cast<CpuExecutable*>(executable.get())
            ->thunks()
            .thunk_sequence();
    int64_t num_kernels = 0;
    for (const auto& thunk : thunks) {
      num_kernels += thunk->kind() == Thunk::Kind::kKernel;
    }
    EXPECT_EQ(num_kernels, expected_kernels);
    thunks.ForEach([](const Thunk& thunk) {
      EXPECT_NE(thunk.kind(), Thunk::Kind::kCall);
    });
  }

  void ExpectBackendFallsBackToCallThunk(absl::string_view hlo_text,
                                         int64_t intra_op_threads = 4) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(hlo_text));
    module->mutable_config().set_intra_op_parallelism_threads(intra_op_threads);
    CpuCompiler compiler;
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Executable> executable,
        compiler.RunBackend(std::move(module), /*stream_exec=*/nullptr,
                            /*options=*/{}));
    const ThunkSequence& thunks =
        absl::down_cast<CpuExecutable*>(executable.get())
            ->thunks()
            .thunk_sequence();
    int64_t num_calls = 0;
    int64_t num_whiles = 0;
    thunks.ForEach([&](const Thunk& thunk) {
      num_calls += thunk.kind() == Thunk::Kind::kCall;
      num_whiles += thunk.kind() == Thunk::Kind::kWhile;
    });
    EXPECT_GE(num_calls, 1);
    EXPECT_EQ(num_whiles, 1);
  }

  // Checks that EmitSmallCallThunk falls back to a CallThunk wrapping a
  // WhileThunk so nested parallel or concurrent work executes on the intra op
  // thread pool instead of serially inside one kernel.
  void ExpectFallsBackToCallThunk(
      absl::string_view hlo_text, int64_t intra_op_threads = 4,
      std::optional<int64_t> byte_threshold = std::nullopt) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(hlo_text));
    module->mutable_config().set_intra_op_parallelism_threads(intra_op_threads);
    if (byte_threshold.has_value()) {
      (*module->mutable_config()
            .mutable_debug_options()
            .mutable_xla_backend_extra_options())
          ["xla_cpu_small_while_loop_byte_threshold"] =
              absl::StrCat(*byte_threshold);
    }
    CpuCompiler compiler;
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> optimized,
        compiler.RunHloPasses(std::move(module), /*stream_exec=*/nullptr,
                              /*options=*/{}));
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Executable> executable,
        compiler.RunBackend(std::move(optimized), /*stream_exec=*/nullptr,
                            /*options=*/{}));
    const ThunkSequence& thunks =
        absl::down_cast<CpuExecutable*>(executable.get())
            ->thunks()
            .thunk_sequence();
    int64_t num_calls = 0;
    int64_t num_whiles = 0;
    thunks.ForEach([&](const Thunk& thunk) {
      num_calls += thunk.kind() == Thunk::Kind::kCall;
      num_whiles += thunk.kind() == Thunk::Kind::kWhile;
    });
    EXPECT_GE(num_calls, 1);
    EXPECT_EQ(num_whiles, 1);
  }

  // Checks that SmallWhileLoopHoistingPass leaves the loop unhoisted as a top
  // level WhileThunk.
  void ExpectNotHoisted(absl::string_view hlo_text,
                        std::optional<int64_t> byte_threshold = std::nullopt) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(hlo_text));
    if (byte_threshold.has_value()) {
      (*module->mutable_config()
            .mutable_debug_options()
            .mutable_xla_backend_extra_options())
          ["xla_cpu_small_while_loop_byte_threshold"] =
              absl::StrCat(*byte_threshold);
    }
    CpuCompiler compiler;
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> optimized,
        compiler.RunHloPasses(std::move(module), /*stream_exec=*/nullptr,
                              /*options=*/{}));
    for (const HloComputation* comp : optimized->computations()) {
      for (const HloInstruction* instr : comp->instructions()) {
        EXPECT_NE(instr->get_frontend_attribute("xla_cpu_small_call"), "true")
            << instr->ToString();
      }
    }
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Executable> executable,
        compiler.RunBackend(std::move(optimized), /*stream_exec=*/nullptr,
                            /*options=*/{}));
    int64_t num_whiles = 0;
    for (const auto& thunk : absl::down_cast<CpuExecutable*>(executable.get())
                                 ->thunks()
                                 .thunk_sequence()) {
      num_whiles += thunk->kind() == Thunk::Kind::kWhile;
    }
    EXPECT_EQ(num_whiles, 1);
  }
};

TEST_F(CpuSmallWhileLoopTest, NestedWhileAndConditional) {
  constexpr absl::string_view hlo_text = R"(
    HloModule NestedWhileAndConditional

    inner_body {
      s = (s32[], f32[4]) parameter(0)
      j = s32[] get-tuple-element(s), index=0
      x = f32[4] get-tuple-element(s), index=1
      one = s32[] constant(1)
      c = f32[] constant(0.5)
      cb = f32[4] broadcast(c), dimensions={}
      ROOT r = (s32[], f32[4]) tuple(add(j, one), add(x, cb))
    }

    inner_cond {
      s = (s32[], f32[4]) parameter(0)
      j = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(j, s32[] constant(3)), direction=LT
    }

    double {
      x = f32[4] parameter(0)
      ROOT r = f32[4] add(x, x)
    }

    negate {
      x = f32[4] parameter(0)
      ROOT r = f32[4] negate(x)
    }

    body {
      s = (s32[], f32[4]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      x = f32[4] get-tuple-element(s), index=1
      zero = s32[] constant(0)
      inner = (s32[], f32[4]) while((s32[], f32[4]) tuple(zero, x)),
          condition=inner_cond, body=inner_body
      y = f32[4] get-tuple-element(inner), index=1
      two = s32[] constant(2)
      branch = s32[] remainder(i, two)
      z = f32[4] conditional(branch, y, y), branch_computations={double, negate}
      one = s32[] constant(1)
      ROOT r = (s32[], f32[4]) tuple(add(i, one), z)
    }

    cond {
      s = (s32[], f32[4]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(4)), direction=LT
    }

    ENTRY main {
      p = f32[4] parameter(0)
      zero = s32[] constant(0)
      w = (s32[], f32[4]) while((s32[], f32[4]) tuple(zero, p)),
          condition=cond, body=body
      ROOT r = f32[4] get-tuple-element(w), index=1
    }
  )";
  ExpectOneKernel(hlo_text);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5}));
}

TEST_F(CpuSmallWhileLoopTest, DotsAndScatter) {
  constexpr absl::string_view hlo_text = R"(
    HloModule DotsAndScatter

    add {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT r = f32[] add(a, b)
    }

    body {
      s = (s32[], bf16[16,16], bf16[16], f32[16,16], f32[16]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      m = bf16[16,16] get-tuple-element(s), index=1
      v = bf16[16] get-tuple-element(s), index=2
      a = f32[16,16] get-tuple-element(s), index=3
      acc = f32[16] get-tuple-element(s), index=4
      mv = bf16[16] dot(m, v), lhs_contracting_dims={1},
          rhs_contracting_dims={0}
      av = f32[16] dot(a, acc), lhs_contracting_dims={1},
          rhs_contracting_dims={0}
      sum = f32[16] add(av, acc)
      c = s32[] constant(5)
      idx = s32[3,1] broadcast(remainder(i, c)), dimensions={}
      offsets = s32[3,1] constant({{0}, {5}, {10}})
      indices = s32[3,1] add(idx, offsets)
      updates = f32[3] constant({1, 2, 3})
      scattered = f32[16] scatter(sum, indices, updates), to_apply=add,
          update_window_dims={}, inserted_window_dims={0},
          scatter_dims_to_operand_dims={0}, index_vector_dim=1
      one = s32[] constant(1)
      mn = bf16[16,16] negate(m)
      ROOT r = (s32[], bf16[16,16], bf16[16], f32[16,16], f32[16])
          tuple(add(i, one), mn, mv, a, scattered)
    }

    cond {
      s = (s32[], bf16[16,16], bf16[16], f32[16,16], f32[16]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(3)), direction=LT
    }

    ENTRY main {
      m = bf16[16,16] parameter(0)
      v = bf16[16] parameter(1)
      a = f32[16,16] parameter(2)
      acc = f32[16] parameter(3)
      zero = s32[] constant(0)
      init = (s32[], bf16[16,16], bf16[16], f32[16,16], f32[16])
          tuple(zero, m, v, a, acc)
      w = (s32[], bf16[16,16], bf16[16], f32[16,16], f32[16]) while(init),
          condition=cond, body=body
      mv = bf16[16] get-tuple-element(w), index=2
      out = f32[16] get-tuple-element(w), index=4
      ROOT r = (bf16[16], f32[16]) tuple(mv, out)
    }
  )";
  ExpectOneKernel(hlo_text);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-2, 1e-2}));
}

// An output fusion of a dot and an addend, which no HLO pass is left to
// create from the matrix vector product below.
TEST_F(CpuSmallWhileLoopTest, DotOutputFusion) {
  constexpr absl::string_view hlo_text = R"(
    HloModule DotOutputFusion

    fused_dot_add {
      a = f32[16,16] parameter(0)
      x = f32[16] parameter(1)
      addend = f32[16] parameter(2)
      d = f32[16] dot(a, x), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT r = f32[16] add(d, addend)
    }

    body {
      s = (s32[], f32[16,16], f32[16]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      a = f32[16,16] get-tuple-element(s), index=1
      x = f32[16] get-tuple-element(s), index=2
      y = f32[16] fusion(a, x, x), kind=kOutput, calls=fused_dot_add
      one = s32[] constant(1)
      ROOT r = (s32[], f32[16,16], f32[16]) tuple(add(i, one), a, y)
    }

    cond {
      s = (s32[], f32[16,16], f32[16]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(3)), direction=LT
    }

    ENTRY main {
      a = f32[16,16] parameter(0)
      x = f32[16] parameter(1)
      zero = s32[] constant(0)
      w = (s32[], f32[16,16], f32[16]) while((s32[], f32[16,16], f32[16])
          tuple(zero, a, x)), condition=cond, body=body
      ROOT r = f32[16] get-tuple-element(w), index=2
    }
  )";
  ExpectOneKernel(hlo_text);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

// EmitSmallCallThunk falls back to CallThunk for multi workgroup nested kernels
// when multi threading is enabled, and lowers them sequentially into one kernel
// when running single threaded or in AOT mode.
TEST_F(CpuSmallWhileLoopTest, MultipleWorkgroups) {
  constexpr absl::string_view hlo_text = R"hlo(
    HloModule MultipleWorkgroups

    add_step {
      x = f32[64,16]{1,0} parameter(0)
      i = s32[] parameter(1)
      fi = f32[] convert(i)
      b = f32[64,16]{1,0} broadcast(fi), dimensions={}
      iota = f32[64,16]{1,0} iota(), iota_dimension=0
      prod = f32[64,16]{1,0} multiply(iota, b)
      ROOT y = f32[64,16]{1,0} add(x, prod)
    }

    inc_i {
      i = s32[] parameter(0)
      one = s32[] constant(1)
      ROOT next = s32[] add(i, one)
    }

    cmp_i {
      i = s32[] parameter(0)
      limit = s32[] constant(4)
      ROOT lt = pred[] compare(i, limit), direction=LT
    }

    body {
      s = (s32[], f32[64,16]{1,0}) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      x = f32[64,16]{1,0} get-tuple-element(s), index=1
      y = f32[64,16]{1,0} fusion(x, i), kind=kLoop, calls=add_step,
          backend_config={"outer_dimension_partitions":["4"]}
      next_i = s32[] fusion(i), kind=kLoop, calls=inc_i,
          control-predecessors={y}
      ROOT r = (s32[], f32[64,16]{1,0}) tuple(next_i, y)
    }

    cond {
      s = (s32[], f32[64,16]{1,0}) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] fusion(i), kind=kLoop, calls=cmp_i
    }

    small_fn {
      zero = s32[] parameter(0)
      p = f32[64,16]{1,0} parameter(1)
      i_init = s32[] copy(zero)
      x_init = f32[64,16]{1,0} copy(p)
      init = (s32[], f32[64,16]{1,0}) tuple(i_init, x_init)
      w = (s32[], f32[64,16]{1,0}) while(init), condition=cond, body=body
      ROOT r = f32[64,16]{1,0} get-tuple-element(w), index=1
    }

    ENTRY main {
      p = f32[64,16]{1,0} parameter(0)
      zero = s32[] constant(0)
      ROOT r = f32[64,16]{1,0} call(zero, p), to_apply=small_fn,
          frontend_attributes={xla_cpu_small_call="true"}
    }
  )hlo";
  ExpectBackendFallsBackToCallThunk(hlo_text, /*intra_op_threads=*/4);
  ExpectBackendNoCallThunk(hlo_text, /*expected_kernels=*/1,
                           /*intra_op_threads=*/1);
  ExpectBackendNoCallThunk(hlo_text, /*expected_kernels=*/1,
                           /*intra_op_threads=*/4, /*is_aot_compile=*/true);
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text, ErrorSpec{1e-3, 1e-3}));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> single_thread_module,
                       ParseAndReturnVerifiedModule(hlo_text));
  single_thread_module->mutable_config().set_intra_op_parallelism_threads(1);
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(single_thread_module),
                                       ErrorSpec{1e-3, 1e-3}));
}

TEST_F(CpuSmallWhileLoopTest, PartitionedFusionFallsBackToCallThunk) {
  constexpr absl::string_view hlo_text = R"hlo(
    HloModule PartitionedFusionFallsBackToCallThunk

    body {
      s = (s32[], f64[8192]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      x = f64[8192] get-tuple-element(s), index=1
      half = f64[8192] broadcast(f64[] constant(0.5)), dimensions={}
      scaled = f64[8192] multiply(x, half)
      y = f64[8192] acos(scaled)
      ROOT r = (s32[], f64[8192]) tuple(add(i, s32[] constant(1)), y)
    }

    cond {
      s = (s32[], f64[8192]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(3)), direction=LT
    }

    ENTRY main {
      p = f64[8192] parameter(0)
      lo = f64[8192] broadcast(f64[] constant(-0.9)), dimensions={}
      hi = f64[8192] broadcast(f64[] constant(0.9)), dimensions={}
      clamped = f64[8192] clamp(lo, p, hi)
      w = (s32[], f64[8192]) while((s32[], f64[8192]) tuple(
          s32[] constant(0), clamped)), condition=cond, body=body
      ROOT r = f64[8192] get-tuple-element(w), index=1
    }
  )hlo";
  ExpectFallsBackToCallThunk(hlo_text, /*intra_op_threads=*/4,
                             /*byte_threshold=*/1LL << 30);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(CpuSmallWhileLoopTest, PartitionedIndexWithScatterFallsBackToCallThunk) {
  constexpr absl::string_view hlo_text = R"hlo(
    HloModule PartitionedIndexWithScatterFallsBackToCallThunk

    add_f32 {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT s = f32[] add(a, b)
    }

    body {
      s = (s32[], f32[16], f64[8192]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      hist = f32[16] get-tuple-element(s), index=1
      x = f64[8192] get-tuple-element(s), index=2
      angle = f64[8192] acos(x)
      step = f64[8192] broadcast(f64[] convert(i)), dimensions={}
      four = f64[8192] broadcast(f64[] constant(4.0)), dimensions={}
      scaled = f64[8192] add(f64[8192] multiply(angle, four), step)
      raw_idx = s32[8192] convert(scaled)
      idx_lo = s32[8192] broadcast(s32[] constant(0)), dimensions={}
      idx_hi = s32[8192] broadcast(s32[] constant(15)), dimensions={}
      clamped_idx = s32[8192] clamp(idx_lo, raw_idx, idx_hi)
      idx = s32[8192,1] reshape(clamped_idx)
      ones = f32[8192] broadcast(f32[] constant(1.0)), dimensions={}
      next_hist = f32[16] scatter(hist, idx, ones),
          update_window_dims={}, inserted_window_dims={0},
          scatter_dims_to_operand_dims={0}, index_vector_dim=1,
          to_apply=add_f32
      half = f64[8192] broadcast(f64[] constant(0.5)), dimensions={}
      next_x = f64[8192] multiply(x, half)
      ROOT r = (s32[], f32[16], f64[8192])
          tuple(add(i, s32[] constant(1)), next_hist, next_x)
    }

    cond {
      s = (s32[], f32[16], f64[8192]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(3)), direction=LT
    }

    ENTRY main {
      p = f64[8192] parameter(0)
      lo = f64[8192] broadcast(f64[] constant(-0.9)), dimensions={}
      hi = f64[8192] broadcast(f64[] constant(0.9)), dimensions={}
      x = f64[8192] clamp(lo, p, hi)
      hist = f32[16] broadcast(f32[] constant(0.0)), dimensions={}
      w = (s32[], f32[16], f64[8192]) while((s32[], f32[16], f64[8192])
          tuple(s32[] constant(0), hist, x)), condition=cond, body=body
      ROOT r = f32[16] get-tuple-element(w), index=1
    }
  )hlo";
  ExpectFallsBackToCallThunk(hlo_text, /*intra_op_threads=*/4,
                             /*byte_threshold=*/1LL << 30);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-3, 1e-3}));
}

TEST_F(CpuSmallWhileLoopTest, ParallelGemvBatchDotAndCopyFallBackToCallThunk) {
  constexpr absl::string_view gemv_hlo = R"hlo(
    HloModule ParallelGemvFallback

    body {
      s = (s32[], f32[1024,1024], f32[1024]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      a = f32[1024,1024] get-tuple-element(s), index=1
      v = f32[1024] get-tuple-element(s), index=2
      scale = f32[1024] broadcast(f32[] constant(0.001)), dimensions={}
      dv = f32[1024] dot(a, v),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
      next_v = f32[1024] multiply(dv, scale)
      ROOT r = (s32[], f32[1024,1024], f32[1024])
          tuple(add(i, s32[] constant(1)), a, next_v)
    }

    cond {
      s = (s32[], f32[1024,1024], f32[1024]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(3)), direction=LT
    }

    ENTRY main {
      a = f32[1024,1024] parameter(0)
      v = f32[1024] parameter(1)
      w = (s32[], f32[1024,1024], f32[1024]) while(
          (s32[], f32[1024,1024], f32[1024]) tuple(s32[] constant(0), a, v)),
          condition=cond, body=body
      ROOT r = f32[1024] get-tuple-element(w), index=2
    }
  )hlo";
  ExpectFallsBackToCallThunk(gemv_hlo, /*intra_op_threads=*/4,
                             /*byte_threshold=*/1LL << 30);
  ExpectOneKernel(gemv_hlo, /*intra_op_threads=*/1);
  ExpectOneKernel(gemv_hlo, /*intra_op_threads=*/4, /*is_aot_compile=*/true);
  EXPECT_TRUE(RunAndCompare(gemv_hlo, ErrorSpec{1e-2, 1e-2}));

  // A kOutput dot + addend fusion runs in one workgroup under DotKernelEmitter
  // and stays compiled into a single kernel even for large GEMV dimensions.
  constexpr absl::string_view gemv_addend_hlo = R"hlo(
    HloModule LargeDotAddendOutputFusionOneKernel

    fused_dot_add {
      a = f32[1024,1024] parameter(0)
      v = f32[1024] parameter(1)
      bias = f32[1024] parameter(2)
      d = f32[1024] dot(a, v),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT sum = f32[1024] add(d, bias)
    }

    body {
      s = (s32[], f32[1024,1024], f32[1024]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      a = f32[1024,1024] get-tuple-element(s), index=1
      v = f32[1024] get-tuple-element(s), index=2
      scale = f32[1024] broadcast(f32[] constant(0.001)), dimensions={}
      dv = f32[1024] fusion(a, v, v), kind=kOutput, calls=fused_dot_add
      next_v = f32[1024] multiply(dv, scale)
      ROOT r = (s32[], f32[1024,1024], f32[1024])
          tuple(add(i, s32[] constant(1)), a, next_v)
    }

    cond {
      s = (s32[], f32[1024,1024], f32[1024]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(3)), direction=LT
    }

    ENTRY main {
      a = f32[1024,1024] parameter(0)
      v = f32[1024] parameter(1)
      w = (s32[], f32[1024,1024], f32[1024]) while(
          (s32[], f32[1024,1024], f32[1024]) tuple(s32[] constant(0), a, v)),
          condition=cond, body=body
      ROOT r = f32[1024] get-tuple-element(w), index=2
    }
  )hlo";
  ExpectOneKernel(gemv_addend_hlo, /*intra_op_threads=*/4);
  EXPECT_TRUE(RunAndCompare(gemv_addend_hlo, ErrorSpec{1e-2, 1e-2}));

  constexpr absl::string_view batch_dot_hlo = R"hlo(
    HloModule ParallelBatchDotFallback

    body {
      s = (s32[], f32[4,512,512], f32[4,512]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      a = f32[4,512,512] get-tuple-element(s), index=1
      v = f32[4,512] get-tuple-element(s), index=2
      scale = f32[4,512] broadcast(f32[] constant(0.002)), dimensions={}
      dv = f32[4,512] dot(a, v),
          lhs_batch_dims={0}, rhs_batch_dims={0},
          lhs_contracting_dims={2}, rhs_contracting_dims={1}
      next_v = f32[4,512] multiply(dv, scale)
      ROOT r = (s32[], f32[4,512,512], f32[4,512])
          tuple(add(i, s32[] constant(1)), a, next_v)
    }

    cond {
      s = (s32[], f32[4,512,512], f32[4,512]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(3)), direction=LT
    }

    ENTRY main {
      a = f32[4,512,512] parameter(0)
      v = f32[4,512] parameter(1)
      w = (s32[], f32[4,512,512], f32[4,512]) while(
          (s32[], f32[4,512,512], f32[4,512]) tuple(s32[] constant(0), a, v)),
          condition=cond, body=body
      ROOT r = f32[4,512] get-tuple-element(w), index=2
    }
  )hlo";
  ExpectFallsBackToCallThunk(batch_dot_hlo, /*intra_op_threads=*/4,
                             /*byte_threshold=*/1LL << 30);
  EXPECT_TRUE(RunAndCompare(batch_dot_hlo, ErrorSpec{1e-2, 1e-2}));

  // Ring buffer rotation forces CopyInsertion to emit > 1 MiB CopyThunks inside
  // the loop body, which fall back under multi threading and stay one kernel
  // when single threaded or in AOT mode.
  constexpr absl::string_view copy_hlo = R"hlo(
    HloModule ParallelCopyFallback

    body {
      s = (s32[], f32[300000], f32[300000]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      a = f32[300000] get-tuple-element(s), index=1
      b = f32[300000] get-tuple-element(s), index=2
      ROOT r = (s32[], f32[300000], f32[300000])
          tuple(add(i, s32[] constant(1)), b, a)
    }

    cond {
      s = (s32[], f32[300000], f32[300000]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(3)), direction=LT
    }

    ENTRY main {
      a = f32[300000] parameter(0)
      b = f32[300000] parameter(1)
      w = (s32[], f32[300000], f32[300000]) while(
          (s32[], f32[300000], f32[300000]) tuple(s32[] constant(0), a, b)),
          condition=cond, body=body
      ROOT r = f32[300000] get-tuple-element(w), index=1
    }
  )hlo";
  ExpectFallsBackToCallThunk(copy_hlo, /*intra_op_threads=*/4,
                             /*byte_threshold=*/1LL << 30);
  ExpectOneKernel(copy_hlo, /*intra_op_threads=*/1);
  ExpectOneKernel(copy_hlo, /*intra_op_threads=*/4, /*is_aot_compile=*/true);
  EXPECT_TRUE(RunAndCompare(copy_hlo, ErrorSpec{0}));

  // Exactly 1 MiB copies run single threaded in CopyThunk and stay one kernel.
  constexpr absl::string_view small_copy_hlo = R"hlo(
    HloModule SingleBlockCopyOneKernel

    body {
      s = (s32[], f32[262144], f32[262144]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      a = f32[262144] get-tuple-element(s), index=1
      b = f32[262144] get-tuple-element(s), index=2
      ROOT r = (s32[], f32[262144], f32[262144])
          tuple(add(i, s32[] constant(1)), b, a)
    }

    cond {
      s = (s32[], f32[262144], f32[262144]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(3)), direction=LT
    }

    ENTRY main {
      a = f32[262144] parameter(0)
      b = f32[262144] parameter(1)
      w = (s32[], f32[262144], f32[262144]) while(
          (s32[], f32[262144], f32[262144]) tuple(s32[] constant(0), a, b)),
          condition=cond, body=body
      ROOT r = f32[262144] get-tuple-element(w), index=1
    }
  )hlo";
  ExpectOneKernel(small_copy_hlo, /*intra_op_threads=*/4);
}

TEST_F(CpuSmallWhileLoopTest, ConcurrentHeavyKernelsFallback) {
  auto build_loop_hlo = [](absl::string_view module_name, int num_vec_inputs,
                           bool chain_vec, int num_extra_scalars,
                           int64_t vec_elems = 32768) {
    const std::string vec_type = absl::StrCat("f32[", vec_elems, "]{0}");
    std::string fusions;
    absl::StrAppend(&fusions,
                    "inc_i {\n"
                    "  x = s32[] parameter(0)\n"
                    "  ROOT y = s32[] add(x, s32[] constant(1))\n"
                    "}\n"
                    "cmp_i {\n"
                    "  i = s32[] parameter(0)\n"
                    "  ROOT lt = pred[] compare(i, s32[] constant(3)),"
                    " direction=LT\n"
                    "}\n");
    const int num_vec_fusions = chain_vec ? 2 : num_vec_inputs;
    for (int k = 0; k < num_vec_fusions; ++k) {
      absl::StrAppend(&fusions, "inc_vec_", k, " {\n  x = ", vec_type,
                      " parameter(0)\n  one = ", vec_type,
                      " broadcast(f32[] constant(1.0)), dimensions={}\n"
                      "  ROOT y = ",
                      vec_type, " add(x, one)\n}\n");
    }
    for (int k = 0; k < num_extra_scalars; ++k) {
      absl::StrAppend(&fusions, "inc_s32_", k,
                      " {\n"
                      "  x = s32[] parameter(0)\n"
                      "  ROOT y = s32[] add(x, s32[] constant(1))\n"
                      "}\n");
    }

    std::string state_tuple = "(s32[]";
    for (int k = 0; k < num_vec_inputs; ++k) {
      absl::StrAppend(&state_tuple, ", ", vec_type);
    }
    for (int k = 0; k < num_extra_scalars; ++k) {
      absl::StrAppend(&state_tuple, ", s32[]");
    }
    absl::StrAppend(&state_tuple, ")");

    std::string body_ops = "  i = s32[] get-tuple-element(s), index=0\n";
    std::string body_root_args = "next_i";
    int idx = 1;
    for (int k = 0; k < num_vec_inputs; ++k, ++idx) {
      absl::StrAppend(&body_ops, "  v", k, " = ", vec_type,
                      " get-tuple-element(s), index=", idx, "\n");
    }
    for (int k = 0; k < num_extra_scalars; ++k, ++idx) {
      absl::StrAppend(&body_ops, "  t", k,
                      " = s32[] get-tuple-element(s), index=", idx, "\n");
    }
    absl::StrAppend(&body_ops,
                    "  next_i = s32[] fusion(i), kind=kLoop, calls=inc_i\n");
    if (chain_vec) {
      absl::StrAppend(&body_ops, "  v0_mid = ", vec_type,
                      " fusion(v0), kind=kLoop, calls=inc_vec_0\n"
                      "  next_v0 = ",
                      vec_type,
                      " fusion(v0_mid), kind=kLoop, calls=inc_vec_1\n");
      absl::StrAppend(&body_root_args, ", next_v0");
    } else {
      for (int k = 0; k < num_vec_inputs; ++k) {
        absl::StrAppend(&body_ops, "  next_v", k, " = ", vec_type, " fusion(v",
                        k, "), kind=kLoop, calls=inc_vec_", k, "\n");
        absl::StrAppend(&body_root_args, ", next_v", k);
      }
    }
    for (int k = 0; k < num_extra_scalars; ++k) {
      absl::StrAppend(&body_ops, "  next_t", k, " = s32[] fusion(t", k,
                      "), kind=kLoop, calls=inc_s32_", k, "\n");
      absl::StrAppend(&body_root_args, ", next_t", k);
    }

    std::string small_params = "  z = s32[] parameter(0)\n";
    std::string small_copies = "  ci = s32[] copy(z)\n";
    std::string init_args = "ci";
    std::string entry_params;
    std::string call_args = "zero";
    for (int k = 0; k < num_vec_inputs; ++k) {
      absl::StrAppend(&small_params, "  pv", k, " = ", vec_type, " parameter(",
                      k + 1, ")\n");
      absl::StrAppend(&small_copies, "  cv", k, " = ", vec_type, " copy(pv", k,
                      ")\n");
      absl::StrAppend(&init_args, ", cv", k);
      absl::StrAppend(&entry_params, "  p", k, " = ", vec_type, " parameter(",
                      k, ")\n");
      absl::StrAppend(&call_args, ", p", k);
    }
    for (int k = 0; k < num_extra_scalars; ++k) {
      absl::StrAppend(&small_copies, "  cs", k, " = s32[] copy(z)\n");
      absl::StrAppend(&init_args, ", cs", k);
    }

    std::string small_root;
    std::string out_shape;
    if (num_vec_inputs == 1) {
      out_shape = vec_type;
      small_root = absl::StrCat("  ROOT out = ", vec_type,
                                " get-tuple-element(w), index=1\n");
    } else {
      out_shape = absl::StrCat("(", vec_type, ", ", vec_type, ")");
      small_root = absl::StrCat("  r0 = ", vec_type,
                                " get-tuple-element(w), index=1\n"
                                "  r1 = ",
                                vec_type,
                                " get-tuple-element(w), index=2\n"
                                "  ROOT out = ",
                                out_shape, " tuple(r0, r1)\n");
    }

    return absl::StrCat(
        "HloModule ", module_name, "\n", fusions, "body {\n  s = ", state_tuple,
        " parameter(0)\n", body_ops, "  ROOT r = ", state_tuple, " tuple(",
        body_root_args, ")\n}\ncond {\n  s = ", state_tuple,
        " parameter(0)\n  i = s32[] get-tuple-element(s), index=0\n"
        "  ROOT lt = pred[] fusion(i), kind=kLoop, calls=cmp_i\n"
        "}\nsmall_fn {\n",
        small_params, small_copies, "  init = ", state_tuple, " tuple(",
        init_args, ")\n  w = ", state_tuple,
        " while(init), condition=cond, body=body\n", small_root,
        "}\nENTRY main {\n", entry_params,
        "  zero = s32[] constant(0)\n  ROOT r = ", out_shape, " call(",
        call_args,
        "), to_apply=small_fn,"
        " frontend_attributes={xla_cpu_small_call=\"true\"}\n}\n");
  };

  // Two independent single workgroup kernels each touching 256 KiB (128 KiB
  // read + 128 KiB write) alongside 7 independent scalar kernels (10 thunks >
  // 8 so ThunkExecutor runs the body concurrently) fall back to CallThunk
  // under multi threading, and stay one kernel when single threaded or AOT.
  const std::string independent_heavy_hlo =
      build_loop_hlo("IndependentHeavyKernelsFallback", /*num_vec_inputs=*/2,
                     /*chain_vec=*/false, /*num_extra_scalars=*/7);
  ExpectBackendFallsBackToCallThunk(independent_heavy_hlo,
                                    /*intra_op_threads=*/4);
  ExpectBackendNoCallThunk(independent_heavy_hlo, /*expected_kernels=*/1,
                           /*intra_op_threads=*/1);
  ExpectBackendNoCallThunk(independent_heavy_hlo, /*expected_kernels=*/1,
                           /*intra_op_threads=*/4, /*is_aot_compile=*/true);
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(independent_heavy_hlo, ErrorSpec{1e-4, 1e-4}));

  // Two independent kernels just below the 256 KiB threshold (32767 * 4 * 2 =
  // 262136 bytes each) stay compiled into one kernel.
  const std::string below_threshold_hlo =
      build_loop_hlo("BelowThresholdIndependentKernelsOneKernel",
                     /*num_vec_inputs=*/2, /*chain_vec=*/false,
                     /*num_extra_scalars=*/7, /*vec_elems=*/32767);
  ExpectBackendNoCallThunk(below_threshold_hlo, /*expected_kernels=*/1,
                           /*intra_op_threads=*/4);

  // Two dependent 256 KiB kernels in a chain alongside 7 independent scalar
  // kernels (10 thunks > 8 so ThunkExecutor is concurrent, but the heavy
  // kernels cannot run in parallel) stay compiled into one kernel.
  const std::string dependent_heavy_hlo =
      build_loop_hlo("DependentHeavyKernelsOneKernel", /*num_vec_inputs=*/1,
                     /*chain_vec=*/true, /*num_extra_scalars=*/7);
  ExpectBackendNoCallThunk(dependent_heavy_hlo, /*expected_kernels=*/1,
                           /*intra_op_threads=*/4);

  // Only one 256 KiB kernel plus 8 independent scalar kernels stays one kernel.
  const std::string single_heavy_hlo =
      build_loop_hlo("SingleHeavyKernelOneKernel", /*num_vec_inputs=*/1,
                     /*chain_vec=*/false, /*num_extra_scalars=*/8);
  ExpectBackendNoCallThunk(single_heavy_hlo, /*expected_kernels=*/1,
                           /*intra_op_threads=*/4);

  // Two independent 256 KiB kernels in a body with <= 8 thunks (ThunkExecutor
  // executes sequentially regardless) stay compiled into one kernel.
  const std::string sequential_executor_hlo =
      build_loop_hlo("SequentialExecutorHeavyKernelsOneKernel",
                     /*num_vec_inputs=*/2, /*chain_vec=*/false,
                     /*num_extra_scalars=*/0);
  ExpectBackendNoCallThunk(sequential_executor_hlo, /*expected_kernels=*/1,
                           /*intra_op_threads=*/4);
}

TEST_F(CpuSmallWhileLoopTest, ManyReductionBodyNotHoisted) {
  // A body with 40 fused array reductions and 40 array updates (> 64 array loop
  // kernels, as in issue #52122) exceeds the hoisted body complexity bound even
  // under a large byte threshold.
  std::string body_chain;
  for (int k = 0; k < 40; ++k) {
    const std::string prev = k == 0 ? "x" : absl::StrCat("x", k - 1);
    absl::StrAppend(
        &body_chain, "  c1_", k, " = f32[64] broadcast(f32[] constant(",
        0.01 * (k + 1), ")), dimensions={}\n  c2_", k,
        " = f32[64] broadcast(f32[] constant(", 0.01 * (k + 2),
        ")), dimensions={}\n  t_", k, " = f32[64] add(multiply(sn, c1_", k,
        "), multiply(cs, c2_", k, "))\n  red_", k, " = f32[] reduce(t_", k,
        ", f32[] constant(0.0)), dimensions={0}, to_apply=add_f32\n  rb_", k,
        " = f32[64] broadcast(red_", k, "), dimensions={}\n  x", k,
        " = f32[64] add(", prev, ", rb_", k, ")\n");
  }
  const std::string many_reductions_hlo = absl::StrCat(
      "HloModule ManyReductionsNotHoisted\n"
      "add_f32 {\n"
      "  a = f32[] parameter(0)\n"
      "  b = f32[] parameter(1)\n"
      "  ROOT s = f32[] add(a, b)\n"
      "}\n"
      "body {\n"
      "  st = (s32[], f32[64]) parameter(0)\n"
      "  i = s32[] get-tuple-element(st), index=0\n"
      "  x = f32[64] get-tuple-element(st), index=1\n"
      "  sn = f32[64] sine(x)\n"
      "  cs = f32[64] cosine(x)\n",
      body_chain,
      "  ROOT r = (s32[], f32[64]) tuple(add(i, s32[] constant(1)), x39)\n"
      "}\n"
      "cond {\n"
      "  st = (s32[], f32[64]) parameter(0)\n"
      "  i = s32[] get-tuple-element(st), index=0\n"
      "  ROOT lt = pred[] compare(i, s32[] constant(2)), direction=LT\n"
      "}\n"
      "ENTRY main {\n"
      "  x = f32[64] parameter(0)\n"
      "  w = (s32[], f32[64]) while((s32[], f32[64])"
      " tuple(s32[] constant(0), x)), condition=cond, body=body\n"
      "  ROOT r = f32[64] get-tuple-element(w), index=1\n"
      "}\n");
  ExpectNotHoisted(many_reductions_hlo, /*byte_threshold=*/1LL << 30);
  EXPECT_TRUE(RunAndCompare(many_reductions_hlo, ErrorSpec{1e-2, 1e-2}));
}

TEST_F(CpuSmallWhileLoopTest, InPlaceDynamicUpdateSlice) {
  constexpr absl::string_view hlo_text = R"(
    HloModule InPlaceDynamicUpdateSlice

    body {
      s = (s32[], s32[], f32[32]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      acc = s32[] get-tuple-element(s), index=1
      arr = f32[32] get-tuple-element(s), index=2
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      next_acc = s32[] add(acc, next_i)
      slice = f32[1] dynamic-slice(arr, i), dynamic_slice_sizes={1}
      delta = f32[1] reshape(f32[] convert(next_acc))
      upd = f32[1] add(slice, delta)
      next_arr = f32[32] dynamic-update-slice(arr, upd, i)
      ROOT r = (s32[], s32[], f32[32]) tuple(next_i, next_acc, next_arr)
    }

    cond {
      s = (s32[], s32[], f32[32]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(16)), direction=LT
    }

    ENTRY main {
      p = f32[32] parameter(0)
      z = s32[] constant(0)
      w = (s32[], s32[], f32[32]) while((s32[], s32[], f32[32]) tuple(z, z, p)),
          condition=cond, body=body
      ROOT r = f32[32] get-tuple-element(w), index=2
    }
  )";
  ExpectOneKernel(hlo_text);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5}));
}

TEST_F(CpuSmallWhileLoopTest, ReusedTempBuffers) {
  constexpr absl::string_view hlo_text = R"(
    HloModule ReusedTempBuffers

    inner_body {
      s = (s32[], u32[2], u32[4]) parameter(0)
      j = s32[] get-tuple-element(s), index=0
      v = u32[2] get-tuple-element(s), index=1
      k = u32[4] get-tuple-element(s), index=2
      k0 = u32[2] slice(k), slice={[0:2]}
      k1 = u32[2] slice(k), slice={[2:4]}
      v1 = u32[2] xor(add(v, k0), k1)
      k_next = u32[4] concatenate(k1, k0), dimensions={0}
      ROOT r = (s32[], u32[2], u32[4]) tuple(add(j, s32[] constant(1)), v1, k_next)
    }

    inner_cond {
      s = (s32[], u32[2], u32[4]) parameter(0)
      j = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(j, s32[] constant(4)), direction=LT
    }

    body {
      s = (s32[], u32[2]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      v = u32[2] get-tuple-element(s), index=1
      k = u32[4] constant({13, 15, 26, 6})
      inner = (s32[], u32[2], u32[4]) while(
          (s32[], u32[2], u32[4]) tuple(s32[] constant(0), v, k)),
          condition=inner_cond, body=inner_body
      v_next = u32[2] get-tuple-element(inner), index=1
      ROOT r = (s32[], u32[2]) tuple(add(i, s32[] constant(1)), v_next)
    }

    cond {
      s = (s32[], u32[2]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(5)), direction=LT
    }

    ENTRY main {
      p = u32[2] parameter(0)
      w = (s32[], u32[2]) while((s32[], u32[2]) tuple(s32[] constant(0), p)),
          condition=cond, body=body
      ROOT r = u32[2] get-tuple-element(w), index=1
    }
  )";
  ExpectOneKernel(hlo_text);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{0}));
}

TEST_F(CpuSmallWhileLoopTest, NestedFusionKernelNameCollision) {
  // Fusion and concatenate emitted before enclosing small call.
  constexpr absl::string_view kBackward = R"(
    HloModule NestedFusionKernelNameCollision

    f_add {
      x = s32[4] parameter(0)
      y = s32[4] parameter(1)
      ROOT s = s32[4] add(x, y)
    }

    f_mul {
      x = s32[4] parameter(0)
      ROOT y = s32[4] multiply(x, s32[4] constant({2, 2, 2, 2}))
    }

    small_fn {
      v = s32[4] parameter(0)
      cat = s32[4] concatenate(s32[2] constant({1, 2}),
                               s32[2] constant({3, 4})), dimensions={0}
      call_kernel = s32[4] fusion(v, cat), kind=kLoop, calls=f_add
      ROOT call_kernel_body = s32[4] fusion(call_kernel), kind=kLoop, calls=f_mul
    }

    ENTRY main {
      p = s32[4] parameter(0)
      ROOT call = s32[4] call(p), to_apply=small_fn,
          frontend_attributes={xla_cpu_small_call="true"}
    }
  )";
  ExpectBackendNoCallThunk(kBackward, /*expected_kernels=*/1);
  EXPECT_TRUE(RunAndCompareNoHloPasses(kBackward, ErrorSpec{0}));

  // Unfused concatenate before small call `c`, `c` before top level fusion
  // `c_kernel`, and nested small call `n` before sibling fusion `n_kernel`.
  constexpr absl::string_view kForward = R"(
    HloModule ForwardKernelNameCollision

    f_add {
      x = s32[4] parameter(0)
      ROOT y = s32[4] add(x, s32[4] constant({1, 2, 3, 4}))
    }

    f_mul_outer {
      x = s32[4] parameter(0)
      ROOT y = s32[4] multiply(x, s32[4] constant({3, 3, 3, 3}))
    }

    f_mul_entry {
      x = s32[4] parameter(0)
      y = s32[4] parameter(1)
      ROOT z = s32[4] add(x, y)
    }

    inner_fn {
      v = s32[4] parameter(0)
      ROOT a = s32[4] fusion(v), kind=kLoop, calls=f_add
    }

    outer_fn {
      v = s32[4] parameter(0)
      n = s32[4] call(v), to_apply=inner_fn,
          frontend_attributes={xla_cpu_small_call="true"}
      ROOT n_kernel = s32[4] fusion(n), kind=kLoop, calls=f_mul_outer
    }

    ENTRY main {
      p = s32[4] parameter(0)
      pre = s32[4] concatenate(s32[2] constant({1, 2}),
                               s32[2] constant({3, 4})), dimensions={0}
      c = s32[4] call(p), to_apply=outer_fn,
          frontend_attributes={xla_cpu_small_call="true"}
      ROOT c_kernel = s32[4] fusion(c, pre), kind=kLoop, calls=f_mul_entry
    }
  )";
  ExpectBackendNoCallThunk(kForward, /*expected_kernels=*/3);
  EXPECT_TRUE(RunAndCompareNoHloPasses(kForward, ErrorSpec{0}));

  // Sort comparator computation named like small call `c`'s kernel.
  constexpr absl::string_view kComparator = R"(
    HloModule ComparatorKernelNameCollision

    c_kernel {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT lt = pred[] compare(a, b), direction=LT
    }

    f_add {
      x = s32[4] parameter(0)
      ROOT y = s32[4] add(x, s32[4] constant({1, 2, 3, 4}))
    }

    small_fn {
      v = s32[4] parameter(0)
      ROOT a = s32[4] fusion(v), kind=kLoop, calls=f_add
    }

    ENTRY main {
      p = s32[4] parameter(0)
      s = s32[4] sort(p), dimensions={0}, to_apply=c_kernel
      ROOT c = s32[4] call(s), to_apply=small_fn,
          frontend_attributes={xla_cpu_small_call="true"}
    }
  )";
  ExpectBackendNoCallThunk(kComparator, /*expected_kernels=*/1);
  EXPECT_TRUE(RunAndCompareNoHloPasses(kComparator, ErrorSpec{0}));
}

TEST_F(CpuSmallWhileLoopTest, BatchedMatrixVectorDot) {
  constexpr absl::string_view hlo_text = R"(
    HloModule BatchedMatrixVectorDot

    body {
      s = (s32[], f32[2,16,16], f32[2,16]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      a = f32[2,16,16] get-tuple-element(s), index=1
      x = f32[2,16] get-tuple-element(s), index=2
      y = f32[2,16] dot(a, x), lhs_batch_dims={0}, rhs_batch_dims={0},
          lhs_contracting_dims={2}, rhs_contracting_dims={1}
      next_i = s32[] add(i, s32[] constant(1))
      ROOT r = (s32[], f32[2,16,16], f32[2,16]) tuple(next_i, a, y)
    }

    cond {
      s = (s32[], f32[2,16,16], f32[2,16]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(3)), direction=LT
    }

    ENTRY main {
      a = f32[2,16,16] parameter(0)
      x = f32[2,16] parameter(1)
      w = (s32[], f32[2,16,16], f32[2,16]) while(
          (s32[], f32[2,16,16], f32[2,16]) tuple(s32[] constant(0), a, x)),
          condition=cond, body=body
      ROOT out = f32[2,16] get-tuple-element(w), index=2
    }
  )";
  ExpectOneKernel(hlo_text);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-4}));
}

TEST_F(CpuSmallWhileLoopTest, ZeroSizedArrayInLoopState) {
  constexpr absl::string_view hlo_text = R"(
    HloModule ZeroSizedArrayInLoopState

    body {
      s = (s32[], f32[0], f32[4]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      e = f32[0] get-tuple-element(s), index=1
      x = f32[4] get-tuple-element(s), index=2
      next_i = s32[] add(i, s32[] constant(1))
      next_e = f32[0] negate(e)
      next_x = f32[4] add(x, x)
      ROOT r = (s32[], f32[0], f32[4]) tuple(next_i, next_e, next_x)
    }

    cond {
      s = (s32[], f32[0], f32[4]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(3)), direction=LT
    }

    ENTRY main {
      e = f32[0] parameter(0)
      p = f32[4] parameter(1)
      w = (s32[], f32[0], f32[4]) while((s32[], f32[0], f32[4]) tuple(
          s32[] constant(0), e, p)), condition=cond, body=body
      e_out = f32[0] get-tuple-element(w), index=1
      x_out = f32[4] get-tuple-element(w), index=2
      ROOT r = (f32[0], f32[4]) tuple(e_out, x_out)
    }
  )";
  ExpectOneKernel(hlo_text);
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{0}));
}

TEST_F(CpuSmallWhileLoopTest, ReadOnlyWhileConditionEmitsNoCallThunk) {
  constexpr absl::string_view kHlo = R"(
    HloModule ReadOnlyWhileCondition

    f_double {
      x = s32[4] parameter(0)
      ROOT y = s32[4] add(x, x)
    }

    cond {
      s = (pred[], s32[4]) parameter(0)
      ROOT c = pred[] get-tuple-element(s), index=0
    }

    body {
      s = (pred[], s32[4]) parameter(0)
      c = pred[] get-tuple-element(s), index=0
      x = s32[4] get-tuple-element(s), index=1
      nx = s32[4] fusion(x), kind=kLoop, calls=f_double
      ROOT r = (pred[], s32[4]) tuple(c, nx)
    }

    small_fn {
      c = pred[] parameter(0)
      x = s32[4] parameter(1)
      x_copy = s32[4] copy(x)
      init = (pred[], s32[4]) tuple(c, x_copy)
      w = (pred[], s32[4]) while(init), condition=cond, body=body
      ROOT out = s32[4] get-tuple-element(w), index=1
    }

    ENTRY main {
      p = s32[4] parameter(0)
      f = pred[] constant(false)
      ROOT call = s32[4] call(f, p), to_apply=small_fn,
          frontend_attributes={xla_cpu_small_call="true"}
    }
  )";
  ExpectBackendNoCallThunk(kHlo, /*expected_kernels=*/1);
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHlo, ErrorSpec{0}));
}

int64_t TotalOuterDimensionPartitions(const HloInstruction& instr) {
  absl::StatusOr<BackendConfig> config = instr.backend_config<BackendConfig>();
  if (!config.ok()) {
    return 1;
  }
  int64_t total = 1;
  for (int64_t p : config->outer_dimension_partitions()) {
    if (p > 1) {
      total *= p;
    }
  }
  return total;
}

TEST_F(CpuSmallWhileLoopTest, SmallSingleWorkgroupScatterWhileLoopStillHoists) {
  constexpr int64_t kNumBins = 32;
  constexpr int64_t kNumUpdates = 8;
  constexpr int64_t kNumIters = 4;

  constexpr absl::string_view kHlo = R"hlo(
    HloModule SmallSingleWorkgroupScatterWhileLoopStillHoists

    add_s32 {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }

    body {
      s = (s32[], s32[32], s32[8,1], s32[8]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      hist = s32[32] get-tuple-element(s), index=1
      idx = s32[8,1] get-tuple-element(s), index=2
      upd = s32[8] get-tuple-element(s), index=3
      step = s32[8] broadcast(i), dimensions={}
      upd_i = s32[8] add(upd, step)
      next_hist = s32[32] scatter(hist, idx, upd_i),
          update_window_dims={},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_s32
      next_i = s32[] add(i, s32[] constant(1))
      ROOT r = (s32[], s32[32], s32[8,1], s32[8])
          tuple(next_i, next_hist, idx, upd)
    }

    cond {
      s = (s32[], s32[32], s32[8,1], s32[8]) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(4)), direction=LT
    }

    ENTRY main {
      hist = s32[32] parameter(0)
      idx = s32[8,1] parameter(1)
      upd = s32[8] parameter(2)
      init = (s32[], s32[32], s32[8,1], s32[8])
          tuple(s32[] constant(0), hist, idx, upd)
      w = (s32[], s32[32], s32[8,1], s32[8]) while(init),
          condition=cond, body=body
      ROOT out = s32[32] get-tuple-element(w), index=1
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> inspect_module,
                       ParseAndReturnVerifiedModule(kHlo));
  inspect_module->mutable_config().set_intra_op_parallelism_threads(8);
  CpuCompiler compiler;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> optimized,
      compiler.RunHloPasses(std::move(inspect_module), /*stream_exec=*/nullptr,
                            /*options=*/{}));

  int64_t scatter_fusion_count = 0;
  int64_t small_call_count = 0;
  for (const HloComputation* comp : optimized->computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      if (instr->get_frontend_attribute("xla_cpu_small_call") == "true") {
        ++small_call_count;
      }
      if (instr->opcode() == HloOpcode::kFusion &&
          instr->fused_expression_root()->opcode() == HloOpcode::kScatter) {
        ++scatter_fusion_count;
        EXPECT_EQ(TotalOuterDimensionPartitions(*instr), 1)
            << instr->ToString();
      }
    }
  }
  EXPECT_EQ(scatter_fusion_count, 1);
  EXPECT_EQ(small_call_count, 1);

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      compiler.RunBackend(std::move(optimized), /*stream_exec=*/nullptr,
                          /*options=*/{}));
  int64_t num_kernels = 0;
  for (const auto& thunk : absl::down_cast<CpuExecutable*>(executable.get())
                               ->thunks()
                               .thunk_sequence()) {
    EXPECT_NE(thunk->kind(), Thunk::Kind::kWhile);
    EXPECT_NE(thunk->kind(), Thunk::Kind::kCall);
    if (thunk->kind() == Thunk::Kind::kKernel) {
      ++num_kernels;
    }
  }
  EXPECT_EQ(num_kernels, 1);

  std::vector<int32_t> init_vals(kNumBins);
  std::vector<int32_t> expected(kNumBins);
  for (int64_t b = 0; b < kNumBins; ++b) {
    init_vals[b] = static_cast<int32_t>(b * 10);
    expected[b] = init_vals[b];
  }
  const int32_t kIndices[kNumUpdates] = {3, 7, 3, -1, 31, 0, 7, 32};
  const int32_t kUpdates[kNumUpdates] = {2, 5, 11, 13, 17, 19, 23, 29};
  Literal hist = LiteralUtil::CreateR1<int32_t>(init_vals);
  Literal idx = LiteralUtil::CreateR2<int32_t>(
      {{3}, {7}, {3}, {-1}, {31}, {0}, {7}, {32}});
  Literal upd = LiteralUtil::CreateR1<int32_t>(kUpdates);
  constexpr int32_t kIterOffsetSum =
      static_cast<int32_t>(kNumIters * (kNumIters - 1) / 2);
  for (int64_t k = 0; k < kNumUpdates; ++k) {
    if (kIndices[k] >= 0 && kIndices[k] < kNumBins) {
      expected[kIndices[k]] +=
          static_cast<int32_t>(kNumIters) * kUpdates[k] + kIterOffsetSum;
    }
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> exec_module,
                       ParseAndReturnVerifiedModule(kHlo));
  exec_module->mutable_config().set_intra_op_parallelism_threads(8);
  ASSERT_OK_AND_ASSIGN(Literal result,
                       Execute(std::move(exec_module), {&hist, &idx, &upd}));
  LiteralTestUtil::ExpectR1Equal<int32_t>(expected, result);
}

// Verifies that ThunkSequenceKernelEmitter runs all 7 workgroups of a
// partitioned simplified scatter fusion (with non divisible num_updates = 257)
// for both non unique s32 add and unique f32 overwrite inside an explicit
// xla_cpu_small_call while loop and a direct xla_cpu_small_call.
TEST_F(CpuSmallWhileLoopTest, ExplicitSmallCallExecutesMultiWorkgroupScatter) {
  constexpr int64_t kHistBins = 64;
  constexpr int64_t kBufSize = 300;
  constexpr int64_t kNumUpdates = 257;
  constexpr int64_t kWhileIters = 3;
  constexpr int32_t kDirectStep = 5;

  constexpr absl::string_view kHlo = R"hlo(
    HloModule ExplicitSmallCallExecutesMultiWorkgroupScatter

    add_s32 {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }

    overwrite_f32 {
      lhs = f32[] parameter(0)
      ROOT rhs = f32[] parameter(1)
    }

    scatter_add_s32 {
      dst = s32[64]{0} parameter(0)
      idx = s32[257,1]{1,0} parameter(1)
      upd = s32[257,1]{1,0} parameter(2)
      step_scalar = s32[] parameter(3)
      step = s32[257,1]{1,0} broadcast(step_scalar), dimensions={}
      upd_step = s32[257,1]{1,0} add(upd, step)
      ROOT s = s32[64]{0} scatter(dst, idx, upd_step),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_s32
    }

    scatter_overwrite_f32 {
      dst = f32[300]{0} parameter(0)
      idx = s32[257,1]{1,0} parameter(1)
      upd = f32[257,1]{1,0} parameter(2)
      step_scalar = s32[] parameter(3)
      fstep = f32[] convert(step_scalar)
      step = f32[257,1]{1,0} broadcast(fstep), dimensions={}
      upd_step = f32[257,1]{1,0} add(upd, step)
      ROOT s = f32[300]{0} scatter(dst, idx, upd_step),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          unique_indices=true,
          to_apply=overwrite_f32
    }

    scatter_add_s32_call {
      dst = s32[64]{0} parameter(0)
      idx = s32[257,1]{1,0} parameter(1)
      upd = s32[257,1]{1,0} parameter(2)
      step_scalar = s32[] parameter(3)
      step = s32[257,1]{1,0} broadcast(step_scalar), dimensions={}
      upd_step = s32[257,1]{1,0} add(upd, step)
      ROOT s = s32[64]{0} scatter(dst, idx, upd_step),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_s32
    }

    scatter_overwrite_f32_call {
      dst = f32[300]{0} parameter(0)
      idx = s32[257,1]{1,0} parameter(1)
      upd = f32[257,1]{1,0} parameter(2)
      step_scalar = s32[] parameter(3)
      fstep = f32[] convert(step_scalar)
      step = f32[257,1]{1,0} broadcast(fstep), dimensions={}
      upd_step = f32[257,1]{1,0} add(upd, step)
      ROOT s = f32[300]{0} scatter(dst, idx, upd_step),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          unique_indices=true,
          to_apply=overwrite_f32
    }

    inc_i {
      i = s32[] parameter(0)
      ROOT next = s32[] add(i, s32[] constant(1))
    }

    cmp_i {
      i = s32[] parameter(0)
      ROOT lt = pred[] compare(i, s32[] constant(3)), direction=LT
    }

    body {
      st = (s32[], s32[64]{0}, f32[300]{0}, s32[257,1]{1,0}, s32[257,1]{1,0},
            s32[257,1]{1,0}, f32[257,1]{1,0}) parameter(0)
      i = s32[] get-tuple-element(st), index=0
      hist = s32[64]{0} get-tuple-element(st), index=1
      buf = f32[300]{0} get-tuple-element(st), index=2
      add_idx = s32[257,1]{1,0} get-tuple-element(st), index=3
      add_upd = s32[257,1]{1,0} get-tuple-element(st), index=4
      ow_idx = s32[257,1]{1,0} get-tuple-element(st), index=5
      ow_upd = f32[257,1]{1,0} get-tuple-element(st), index=6
      next_hist = s32[64]{0} fusion(hist, add_idx, add_upd, i), kind=kLoop,
          calls=scatter_add_s32,
          backend_config={"outer_dimension_partitions":["7"]}
      next_buf = f32[300]{0} fusion(buf, ow_idx, ow_upd, i), kind=kLoop,
          calls=scatter_overwrite_f32,
          backend_config={"outer_dimension_partitions":["7"]}
      next_i = s32[] fusion(i), kind=kLoop, calls=inc_i,
          control-predecessors={next_hist, next_buf}
      ROOT r = (s32[], s32[64]{0}, f32[300]{0}, s32[257,1]{1,0},
                s32[257,1]{1,0}, s32[257,1]{1,0}, f32[257,1]{1,0})
          tuple(next_i, next_hist, next_buf, add_idx, add_upd, ow_idx, ow_upd)
    }

    cond {
      st = (s32[], s32[64]{0}, f32[300]{0}, s32[257,1]{1,0}, s32[257,1]{1,0},
            s32[257,1]{1,0}, f32[257,1]{1,0}) parameter(0)
      i = s32[] get-tuple-element(st), index=0
      ROOT lt = pred[] fusion(i), kind=kLoop, calls=cmp_i
    }

    small_while_fn {
      zero = s32[] parameter(0)
      hist_p = s32[64]{0} parameter(1)
      buf_p = f32[300]{0} parameter(2)
      add_idx = s32[257,1]{1,0} parameter(3)
      add_upd = s32[257,1]{1,0} parameter(4)
      ow_idx = s32[257,1]{1,0} parameter(5)
      ow_upd = f32[257,1]{1,0} parameter(6)
      i_init = s32[] copy(zero)
      hist_init = s32[64]{0} copy(hist_p)
      buf_init = f32[300]{0} copy(buf_p)
      init = (s32[], s32[64]{0}, f32[300]{0}, s32[257,1]{1,0}, s32[257,1]{1,0},
              s32[257,1]{1,0}, f32[257,1]{1,0})
          tuple(i_init, hist_init, buf_init, add_idx, add_upd, ow_idx, ow_upd)
      w = (s32[], s32[64]{0}, f32[300]{0}, s32[257,1]{1,0}, s32[257,1]{1,0},
           s32[257,1]{1,0}, f32[257,1]{1,0})
          while(init), condition=cond, body=body,
          frontend_attributes={xla_cpu_small_call="true"}
      out_hist = s32[64]{0} get-tuple-element(w), index=1
      out_buf = f32[300]{0} get-tuple-element(w), index=2
      ROOT r = (s32[64]{0}, f32[300]{0}) tuple(out_hist, out_buf)
    }

    small_call_fn {
      step = s32[] parameter(0)
      hist_p = s32[64]{0} parameter(1)
      buf_p = f32[300]{0} parameter(2)
      add_idx = s32[257,1]{1,0} parameter(3)
      add_upd = s32[257,1]{1,0} parameter(4)
      ow_idx = s32[257,1]{1,0} parameter(5)
      ow_upd = f32[257,1]{1,0} parameter(6)
      hist_copy = s32[64]{0} copy(hist_p)
      buf_copy = f32[300]{0} copy(buf_p)
      out_hist = s32[64]{0} fusion(hist_copy, add_idx, add_upd, step),
          kind=kLoop, calls=scatter_add_s32_call,
          backend_config={"outer_dimension_partitions":["7"]}
      out_buf = f32[300]{0} fusion(buf_copy, ow_idx, ow_upd, step),
          kind=kLoop, calls=scatter_overwrite_f32_call,
          backend_config={"outer_dimension_partitions":["7"]}
      ROOT r = (s32[64]{0}, f32[300]{0}) tuple(out_hist, out_buf)
    }

    ENTRY main {
      hist = s32[64]{0} parameter(0)
      buf = f32[300]{0} parameter(1)
      add_idx = s32[257,1]{1,0} parameter(2)
      add_upd = s32[257,1]{1,0} parameter(3)
      ow_idx = s32[257,1]{1,0} parameter(4)
      ow_upd = f32[257,1]{1,0} parameter(5)
      zero = s32[] constant(0)
      five = s32[] constant(5)
      while_res = (s32[64]{0}, f32[300]{0})
          call(zero, hist, buf, add_idx, add_upd, ow_idx, ow_upd),
          to_apply=small_while_fn,
          frontend_attributes={xla_cpu_small_call="true"}
      call_res = (s32[64]{0}, f32[300]{0})
          call(five, hist, buf, add_idx, add_upd, ow_idx, ow_upd),
          to_apply=small_call_fn,
          frontend_attributes={xla_cpu_small_call="true"}
      wh = s32[64]{0} get-tuple-element(while_res), index=0
      wb = f32[300]{0} get-tuple-element(while_res), index=1
      ch = s32[64]{0} get-tuple-element(call_res), index=0
      cb = f32[300]{0} get-tuple-element(call_res), index=1
      ROOT out = (s32[64]{0}, f32[300]{0}, s32[64]{0}, f32[300]{0})
          tuple(wh, wb, ch, cb)
    }
  )hlo";

  ExpectBackendFallsBackToCallThunk(kHlo, /*intra_op_threads=*/4);
  ExpectBackendNoCallThunk(kHlo, /*expected_kernels=*/2,
                           /*intra_op_threads=*/1);
  ExpectBackendNoCallThunk(kHlo, /*expected_kernels=*/2,
                           /*intra_op_threads=*/4, /*is_aot_compile=*/true);

  Literal hist(ShapeUtil::MakeShape(S32, {kHistBins}));
  Literal buf(ShapeUtil::MakeShape(F32, {kBufSize}));
  Literal add_idx(ShapeUtil::MakeShape(S32, {kNumUpdates, 1}));
  Literal add_upd(ShapeUtil::MakeShape(S32, {kNumUpdates, 1}));
  Literal ow_idx(ShapeUtil::MakeShape(S32, {kNumUpdates, 1}));
  Literal ow_upd(ShapeUtil::MakeShape(F32, {kNumUpdates, 1}));

  absl::Span<int32_t> hist_data = hist.data<int32_t>();
  absl::Span<float> buf_data = buf.data<float>();
  std::vector<int32_t> expected_while_hist(kHistBins);
  std::vector<int32_t> expected_call_hist(kHistBins);
  std::vector<float> expected_while_buf(kBufSize);
  std::vector<float> expected_call_buf(kBufSize);

  for (int64_t b = 0; b < kHistBins; ++b) {
    hist_data[b] = static_cast<int32_t>(100 + b);
    expected_while_hist[b] = hist_data[b];
    expected_call_hist[b] = hist_data[b];
  }
  for (int64_t i = 0; i < kBufSize; ++i) {
    buf_data[i] = -static_cast<float>(i + 1);
    expected_while_buf[i] = buf_data[i];
    expected_call_buf[i] = buf_data[i];
  }

  absl::Span<int32_t> add_idx_data = add_idx.data<int32_t>();
  absl::Span<int32_t> add_upd_data = add_upd.data<int32_t>();
  absl::Span<int32_t> ow_idx_data = ow_idx.data<int32_t>();
  absl::Span<float> ow_upd_data = ow_upd.data<float>();
  constexpr int32_t kWhileStepSum =
      static_cast<int32_t>(kWhileIters * (kWhileIters - 1) / 2);
  for (int64_t k = 0; k < kNumUpdates; ++k) {
    // Colliding bins across all 7 workgroups, including negative and positive
    // out of bounds entries.
    int32_t bin = 0;
    if (k == 19) {
      bin = -3;
    } else if (k == 255) {
      bin = static_cast<int32_t>(kHistBins + 5);
    } else {
      bin = static_cast<int32_t>((k * 17 + 5) % kHistBins);
    }
    const int32_t delta = static_cast<int32_t>(k * 3 + 1);
    add_idx_data[k] = bin;
    add_upd_data[k] = delta;
    if (bin >= 0 && bin < kHistBins) {
      expected_while_hist[bin] +=
          static_cast<int32_t>(kWhileIters) * delta + kWhileStepSum;
      expected_call_hist[bin] += delta + kDirectStep;
    }

    // 107 is coprime to 300, so every k in [0, 257) maps to a distinct index
    // in [0, 300), including the final partial workgroup covering k = 256.
    const int32_t dst = static_cast<int32_t>((k * 107 + 13) % kBufSize);
    const float val = static_cast<float>(k * 4 + 10);
    ow_idx_data[k] = dst;
    ow_upd_data[k] = val;
    expected_while_buf[dst] = val + static_cast<float>(kWhileIters - 1);
    expected_call_buf[dst] = val + static_cast<float>(kDirectStep);
  }

  for (int64_t threads : {1, 4}) {
    SCOPED_TRACE(threads);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                         ParseAndReturnVerifiedModule(kHlo));
    module->mutable_config().set_intra_op_parallelism_threads(threads);
    ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module),
                                                 {&hist, &buf, &add_idx,
                                                  &add_upd, &ow_idx, &ow_upd},
                                                 /*run_hlo_passes=*/false));
    std::vector<Literal> outputs = result.DecomposeTuple();
    ASSERT_EQ(outputs.size(), 4);
    LiteralTestUtil::ExpectR1Equal<int32_t>(expected_while_hist, outputs[0]);
    LiteralTestUtil::ExpectR1Equal<float>(expected_while_buf, outputs[1]);
    LiteralTestUtil::ExpectR1Equal<int32_t>(expected_call_hist, outputs[2]);
    LiteralTestUtil::ExpectR1Equal<float>(expected_call_buf, outputs[3]);
  }
}

}  // namespace
}  // namespace cpu
}  // namespace xla
