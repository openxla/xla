/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/backends/cpu/transforms/small_while_loop_hoisting_pass.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/dot_op_emitter.h"
#include "xla/service/cpu/target_machine_features_stub.h"
#include "xla/service/hlo_module_config.h"
#include "xla/xla.pb.h"

namespace xla {
namespace {

class SmallWhileLoopHoistingPassTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<bool> RunSmallWhileLoopHoistingPass(
      HloModule* module, int64_t small_buffer_access_size = 1024) {
    return cpu::SmallWhileLoopHoistingPass(small_buffer_access_size,
                                           &target_machine_features_)
        .Run(module);
  }

  cpu::TargetMachineFeaturesStub target_machine_features_{[](int64_t) {
    return cpu::TargetMachineFeatures::kEigenExpectedTensorAlignment;
  }};
};

TEST_F(SmallWhileLoopHoistingPassTest, SmallWhileLoopHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule simple_while_loop

    while_body {
      counter = s32[] parameter(0)
      increment = s32[] constant(1)
      ROOT incremented_counter = s32[] add(counter, increment)
    }

    while_condition {
      counter = s32[] parameter(0)
      limit = s32[] constant(10)
      ROOT less_than = pred[] compare(counter, limit), direction=LT
    }

    ENTRY main {
      initial_counter = s32[] constant(0)
      ROOT while_loop = s32[] while(initial_counter), condition=while_condition, body=while_body
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call_instr = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call_instr, nullptr);
  std::optional<std::string> maybe_small_call =
      call_instr->get_frontend_attribute("xla_cpu_small_call");
  ASSERT_NE(maybe_small_call, std::nullopt);
  EXPECT_EQ(*maybe_small_call, "true");

  EXPECT_EQ(call_instr->to_apply()->root_instruction()->opcode(),
            HloOpcode::kWhile);
}

TEST_F(SmallWhileLoopHoistingPassTest, NoBigWhileLoopHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule simple_while_loop

    reduce_fn {
      x = s32[] parameter(0)
      y = s32[] parameter(1)
      ROOT add = s32[] add(x, y)
    }

    while_body {
      counter = s32[] parameter(0)
      dummy_constant = s32[1000000] constant({...})
      // The big constant must be in the call graph to be considered in the cost
      // analysis, hence the reduce.
      element_reduce = s32[] reduce(dummy_constant, counter), dimensions={0}, to_apply=reduce_fn
      ROOT incremented_counter = s32[] add(counter, element_reduce)
    }

    while_condition {
      counter = s32[] parameter(0)
      limit = s32[] constant(10)
      ROOT less_than = pred[] compare(counter, limit), direction=LT
    }

    ENTRY main {
      initial_counter = s32[] constant(0)
      ROOT while_loop = s32[] while(initial_counter), condition=while_condition, body=while_body
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SmallWhileLoopHoistingPassTest, NoInOutFeedWhileLoopHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule in_out_feed_while_loop, entry_computation_layout={(pred[])->(pred[])}

    body_fn (T.4: (pred[])) -> (pred[]) {
      T.4 = (pred[]) parameter(0)
      after-all.5 = token[] after-all()
      infeed.6 = ((f32[1,3]{1,0}, pred[], u32[]), token[]) infeed(token[] after-all.5)
      get-tuple-element.7 = token[] get-tuple-element(((f32[1,3]{1,0}, pred[], u32[]), token[]) infeed.6), index=1
      get-tuple-element.8 = (f32[1,3]{1,0}, pred[], u32[]) get-tuple-element(((f32[1,3]{1,0}, pred[], u32[]), token[]) infeed.6), index=0
      get-tuple-element.11 = f32[1,3]{1,0} get-tuple-element((f32[1,3]{1,0}, pred[], u32[]) get-tuple-element.8), index=0
      constant.12 = f32[] constant(1)
      broadcast.13 = f32[1,3]{1,0} broadcast(f32[] constant.12), dimensions={}
      multiply.14 = f32[1,3]{1,0} multiply(f32[1,3]{1,0} get-tuple-element.11, f32[1,3]{1,0} broadcast.13)
      concatenate.15 = f32[1,6]{1,0} concatenate(f32[1,3]{1,0} multiply.14, f32[1,3]{1,0} multiply.14), dimensions={1}
      get-tuple-element.10 = u32[] get-tuple-element((f32[1,3]{1,0}, pred[], u32[]) get-tuple-element.8), index=2
      tuple.16 = (f32[1,6]{1,0}, u32[]) tuple(f32[1,6]{1,0} concatenate.15, u32[] get-tuple-element.10)
      after-all.17 = token[] after-all()
      outfeed.18 = token[] outfeed((f32[1,6]{1,0}, u32[]) tuple.16, token[] after-all.17), outfeed_shape=(f32[1,6]{1,0}, u32[])
      tuple.19 = () tuple()
      get-tuple-element.9 = pred[] get-tuple-element((f32[1,3]{1,0}, pred[], u32[]) get-tuple-element.8), index=1
      ROOT tuple.20 = (pred[]) tuple(pred[] get-tuple-element.9)
    }

    condition_fn (T.22: (pred[])) -> pred[] {
      T.22 = (pred[]) parameter(0)
      ROOT get-tuple-element.23 = pred[] get-tuple-element((pred[]) T.22), index=0
    }

    ENTRY main (prev0.1: pred[]) -> (pred[]) {
      prev0.1 = pred[] parameter(0)
      tuple.2 = (pred[]) tuple(pred[] prev0.1)
      ROOT tuple.26 = (pred[]) while((pred[]) tuple.2), condition=condition_fn, body=body_fn
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SmallWhileLoopHoistingPassTest, NoFftWhileLoopHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule fft_module

    %body_comp (arg_tuple.3: (s32[], c64[30])) -> (s32[], c64[30]) {
      %arg_tuple.3 = (s32[], c64[30]{0}) parameter(0)
      %get-tuple-element.4 = s32[] get-tuple-element(%arg_tuple.3), index=0
      %constant.6 = s32[] constant(1)
      %add.14 = s32[] add(%get-tuple-element.4, %constant.6)
      %get-tuple-element.5 = c64[30]{0} get-tuple-element(%arg_tuple.3), index=1
      %fft.10 = c64[30]{0} fft(%get-tuple-element.5), fft_type=FFT, fft_length={30}
      ROOT %tuple.15 = (s32[], c64[30]{0}) tuple(%add.14, %get-tuple-element.5)
    }

    %condition_comp (arg_tuple.17: (s32[], c64[30])) -> pred[] {
      %arg_tuple.17 = (s32[], c64[30]{0}) parameter(0)
      %get-tuple-element.18 = s32[] get-tuple-element(%arg_tuple.17), index=0
      %constant.20 = s32[] constant(10)
      ROOT %lt.21 = pred[] compare(%get-tuple-element.18, %constant.20), direction=LT
    }

    ENTRY %main.27 (args_0_.1: c64[30]) -> c64[30] {
      %constant.2 = s32[] constant(0)
      %args_0_.1 = c64[30]{0} parameter(0)
      %while.23 = (s32[], c64[30]{0}) tuple(%constant.2, %args_0_.1)
      %while.24 = (s32[], c64[30]{0}) while(%while.23), condition=%condition_comp, body=%body_comp
      %while.25 = s32[] get-tuple-element(%while.24), index=0
      ROOT %while.26 = c64[30]{0} get-tuple-element(%while.24), index=1
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SmallWhileLoopHoistingPassTest, NoYnnWhileLoopHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule ynn_module

    %ynn_comp (lhs: f32[8,3], rhs: f32[3,3]) -> f32[8,3] {
      %lhs = f32[8,3] parameter(0)
      %rhs = f32[3,3] parameter(1)
      ROOT %dot = f32[8,3] dot(%lhs, %rhs), lhs_contracting_dims={1},
                                            rhs_contracting_dims={0}
    }

    %closed_call (x: f32[8,3]) -> (f32[8,3], f32[3], f32[3,3]) {
      %x = f32[8,3] parameter(0)
      %one = f32[] constant(1)
      %one_2d = f32[3,3] broadcast(%one), dimensions={}
      %ynn_fusion = f32[8,3] fusion(%x, %one_2d), kind=kCustom, calls=%ynn_comp,
        backend_config={
            "outer_dimension_partitions":[],
            "fusion_config":{"kind":"__ynn_fusion"}
          }
      %zero = f32[] constant(0)
      %zero_1d = f32[3]{0} broadcast(%zero), dimensions={}
      ROOT %tuple = (f32[8,3], f32[3], f32[3,3]) tuple(%ynn_fusion,
                                                       %zero_1d, %one_2d)
    }

    %body_comp (state: (s32[], f32[8,3], f32[8,3], f32[8,3,3])) ->
                       (s32[], f32[8,3], f32[8,3], f32[8,3,3]) {
      %state = (s32[], f32[8,3], f32[8,3], f32[8,3,3]) parameter(0)
      %idx = s32[] get-tuple-element(%state), index=0
      %one = s32[] constant(1)
      %new_idx = s32[] add(%idx, %one)
      %x = f32[8,3] get-tuple-element(%state), index=1
      %call = (f32[8,3], f32[3], f32[3,3]) call(%x), to_apply=%closed_call
      %in2 = f32[8,3] get-tuple-element(%state), index=2
      %in3 = f32[8,3,3] get-tuple-element(%state), index=3
      ROOT tuple = (s32[], f32[8,3], f32[8,3], f32[8,3,3])
                      tuple(%new_idx, %x, %in2, %in3)
    }

    %cond_comp (state: (s32[], f32[8,3], f32[8,3], f32[8,3,3])) -> pred[] {
      %state = (s32[], f32[8,3], f32[8,3], f32[8,3,3]) parameter(0)
      %idx = s32[] get-tuple-element(%state), index=0
      %eight = s32[] constant(8)
      ROOT %lt.1 = pred[] compare(%idx, %eight), direction=LT
    }

    ENTRY %main (x: f32[8,3]) -> (f32[8,3], f32[8,3], f32[8,3,3]) {
      %zero_int = s32[] constant(0)
      %x = f32[8,3] parameter(0)
      %zero_float = f32[] constant(0)
      %zero_2d = f32[8,3] broadcast(%zero_float), dimensions={}
      %zero_3d = f32[8,3,3] broadcast(%zero_float), dimensions={}
      %init_state = (s32[], f32[8,3], f32[8,3], f32[8,3,3])
                      tuple(%zero_int, %x, %zero_2d, %zero_3d)
      %while = (s32[], f32[8,3], f32[8,3], f32[8,3,3])
                  while(%init_state), condition=%cond_comp,
                  body=%body_comp
      %res_2d = f32[8,3] get-tuple-element(%while), index=2
      %res_3d = f32[8,3,3] get-tuple-element(%while), index=3
      ROOT %tuple = (f32[8,3], f32[8,3], f32[8,3,3]) tuple(%x, %res_2d, %res_3d)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SmallWhileLoopHoistingPassTest, DotAndConvolutionWhileLoopHoisting) {
  constexpr absl::string_view kHloTemplate = R"(
    HloModule while_loop

    while_body {
      state = (s32[], $lhs, $rhs) parameter(0)
      counter = s32[] get-tuple-element(state), index=0
      one = s32[] constant(1)
      next = s32[] add(counter, one)
      lhs = $lhs get-tuple-element(state), index=1
      rhs = $rhs get-tuple-element(state), index=2
      op = $op
      ROOT tuple = (s32[], $lhs, $rhs) tuple(next, lhs, rhs)
    }

    while_condition {
      state = (s32[], $lhs, $rhs) parameter(0)
      counter = s32[] get-tuple-element(state), index=0
      limit = s32[] constant(10)
      ROOT less_than = pred[] compare(counter, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      lhs = $lhs parameter(0)
      rhs = $rhs parameter(1)
      init = (s32[], $lhs, $rhs) tuple(zero, lhs, rhs)
      ROOT while_loop = (s32[], $lhs, $rhs) while(init),
          condition=while_condition, body=while_body
    }
    )";

  struct TestCase {
    absl::string_view lhs;
    absl::string_view rhs;
    absl::string_view op;
    bool expect_hoisted;
  };
  // Small single workgroup matrix vector products are emitted as kernels.
  // Larger dots and convolutions either launch multiple workgroups or call
  // Eigen.
  const TestCase kTestCases[] = {
      {"f32[64,64]", "f32[64,1]",
       "f32[64,1] dot(lhs, rhs), lhs_contracting_dims={1}, "
       "rhs_contracting_dims={0}",
       true},
      {"f32[512,256]", "f32[256,256]",
       "f32[512,256] dot(lhs, rhs), lhs_contracting_dims={1}, "
       "rhs_contracting_dims={0}",
       false},
      {"bf16[256,256]", "bf16[256,1]",
       "bf16[256,1] dot(lhs, rhs), lhs_contracting_dims={1}, "
       "rhs_contracting_dims={0}",
       false},
      {"bf16[1,8,8,4]", "bf16[3,3,4,4]",
       "bf16[1,8,8,4] convolution(lhs, rhs), window={size=3x3 pad=1_1x1_1}, "
       "dim_labels=b01f_01io->b01f",
       true},
      {"f32[8,512,256]", "f32[8,256,256]",
       "f32[8,512,256] dot(lhs, rhs), lhs_batch_dims={0}, "
       "lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}",
       false},
      {"f32[1,32,32,16]", "f32[3,3,16,16]",
       "f32[1,32,32,16] convolution(lhs, rhs), window={size=3x3 pad=1_1x1_1}, "
       "dim_labels=b01f_01io->b01f",
       false},
  };
  for (const TestCase& test_case : kTestCases) {
    SCOPED_TRACE(test_case.op);
    std::string hlo_string =
        absl::StrReplaceAll(kHloTemplate, {{"$lhs", test_case.lhs},
                                           {"$rhs", test_case.rhs},
                                           {"$op", test_case.op}});
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                         ParseAndReturnVerifiedModule(hlo_string));
    ASSERT_OK_AND_ASSIGN(bool changed,
                         RunSmallWhileLoopHoistingPass(
                             m.get(), /*small_buffer_access_size=*/1000000000));
    EXPECT_EQ(changed, test_case.expect_hoisted);
  }
}

TEST_F(SmallWhileLoopHoistingPassTest, LargeDimGemmNotClassifiedAsSmallTiled) {
  constexpr absl::string_view kHloTemplate = R"(
    HloModule while_loop

    while_body {
      state = (s32[], $lhs, $rhs) parameter(0)
      counter = s32[] get-tuple-element(state), index=0
      one = s32[] constant(1)
      next = s32[] add(counter, one)
      lhs = $lhs get-tuple-element(state), index=1
      rhs = $rhs get-tuple-element(state), index=2
      dot = $out dot(lhs, rhs), lhs_contracting_dims={1},
          rhs_contracting_dims={0}
      ROOT tuple = (s32[], $lhs, $rhs) tuple(next, lhs, rhs)
    }

    while_condition {
      state = (s32[], $lhs, $rhs) parameter(0)
      counter = s32[] get-tuple-element(state), index=0
      limit = s32[] constant(10)
      ROOT less_than = pred[] compare(counter, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      lhs = $lhs parameter(0)
      rhs = $rhs parameter(1)
      init = (s32[], $lhs, $rhs) tuple(zero, lhs, rhs)
      ROOT while_loop = (s32[], $lhs, $rhs) while(init),
          condition=while_condition, body=while_body
    }
    )";

  struct TestCase {
    absl::string_view lhs;
    absl::string_view rhs;
    absl::string_view out;
    cpu::DotImplementationStrategy expected_strategy;
    bool expect_hoisted;
  };
  const TestCase kTestCases[] = {
      // Genuine small GEMM under single-threaded Eigen uses kTiledLlvmIrGemm.
      {"f32[8,8]", "f32[8,8]", "f32[8,8]",
       cpu::DotImplementationStrategy::kTiledLlvmIrGemm, true},
      // >= 2^31 (wraps negative in 32-bit int) and > 2^32 (wraps modulo 2^32
      // into <= 32) across m, k, and n must use kEigen, not kTiledLlvmIrGemm.
      {"f32[2147483648,8]", "f32[8,8]", "f32[2147483648,8]",
       cpu::DotImplementationStrategy::kEigen, false},
      {"f32[4294967304,8]", "f32[8,8]", "f32[4294967304,8]",
       cpu::DotImplementationStrategy::kEigen, false},
      {"f32[8,2147483648]", "f32[2147483648,8]", "f32[8,8]",
       cpu::DotImplementationStrategy::kEigen, false},
      {"f32[8,4294967297]", "f32[4294967297,8]", "f32[8,8]",
       cpu::DotImplementationStrategy::kEigen, false},
      {"f32[8,8]", "f32[8,2147483648]", "f32[8,2147483648]",
       cpu::DotImplementationStrategy::kEigen, false},
      {"f32[8,8]", "f32[8,4294967304]", "f32[8,4294967304]",
       cpu::DotImplementationStrategy::kEigen, false},
  };

  for (const TestCase& tc : kTestCases) {
    SCOPED_TRACE(absl::StrReplaceAll("$lhs x $rhs",
                                     {{"$lhs", tc.lhs}, {"$rhs", tc.rhs}}));
    std::string hlo_string = absl::StrReplaceAll(
        kHloTemplate, {{"$lhs", tc.lhs}, {"$rhs", tc.rhs}, {"$out", tc.out}});
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                         ParseAndReturnVerifiedModule(hlo_string));
    m->mutable_config().mutable_debug_options().set_xla_cpu_multi_thread_eigen(
        false);

    const HloInstruction* dot = FindInstruction(m.get(), HloOpcode::kDot);
    ASSERT_NE(dot, nullptr);
    EXPECT_EQ(cpu::GetDotImplementationStrategy(m->config(), *dot,
                                                target_machine_features_,
                                                /*allow_runtime_calls=*/true),
              tc.expected_strategy);

    ASSERT_OK_AND_ASSIGN(
        bool changed,
        RunSmallWhileLoopHoistingPass(m.get(),
                                      /*small_buffer_access_size=*/1LL << 50));
    EXPECT_EQ(changed, tc.expect_hoisted);
  }
}

// A fused dot that would call Eigen outside a fusion would run on one thread
// inside a hoisted loop kernel.
TEST_F(SmallWhileLoopHoistingPassTest, NoFusedEigenSizedDotHoisting) {
  constexpr absl::string_view kHlo = R"(
    HloModule fused_eigen_sized_dot

    fused_dot {
      lhs = f32[64,64] parameter(0)
      rhs = f32[64,64] parameter(1)
      ROOT dot = f32[64,64] dot(lhs, rhs),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    while_body {
      state = (s32[], f32[64,64]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      x = f32[64,64] get-tuple-element(state), index=1
      next_i = s32[] add(i, s32[] constant(1))
      next_x = f32[64,64] fusion(x, x), kind=kLoop, calls=fused_dot
      ROOT r = (s32[], f32[64,64]) tuple(next_i, next_x)
    }

    while_condition {
      state = (s32[], f32[64,64]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      ROOT lt = pred[] compare(i, s32[] constant(2)), direction=LT
    }

    ENTRY main {
      p0 = f32[64,64] parameter(0)
      init = (s32[], f32[64,64]) tuple(s32[] constant(0), p0)
      ROOT w = (s32[], f32[64,64]) while(init),
          condition=while_condition, body=while_body
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(kHlo));
  m->mutable_config().mutable_debug_options().set_xla_cpu_multi_thread_eigen(
      false);

  const HloInstruction* dot = FindInstruction(m.get(), HloOpcode::kDot);
  ASSERT_NE(dot, nullptr);
  EXPECT_EQ(cpu::GetDotImplementationStrategy(m->config(), *dot,
                                              target_machine_features_,
                                              /*allow_runtime_calls=*/true),
            cpu::DotImplementationStrategy::kEigen);

  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunSmallWhileLoopHoistingPass(m.get(),
                                    /*small_buffer_access_size=*/1LL << 20));
  EXPECT_FALSE(changed);
}

TEST_F(SmallWhileLoopHoistingPassTest, PreservesControlDependencies) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule preserves_control_deps

    while_body {
      s = s32[] parameter(0)
      c1 = s32[] constant(1)
      ROOT next = s32[] add(s, c1)
    }

    while_condition {
      s = s32[] parameter(0)
      limit = s32[] constant(3)
      ROOT lt = pred[] compare(s, limit), direction=LT
    }

    ENTRY main {
      p0 = s32[] parameter(0)
      c1 = s32[] constant(1)
      pred_op = s32[] negate(p0)
      inner = s32[] while(p0), condition=while_condition, body=while_body,
          control-predecessors={pred_op}
      succ_op = s32[] add(p0, c1), control-predecessors={inner}
      ROOT out = (s32[], s32[], s32[]) tuple(pred_op, inner, succ_op)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call_instr = FindInstruction(m.get(), HloOpcode::kCall);
  const HloInstruction* pred_op = FindInstruction(m.get(), "pred_op");
  const HloInstruction* succ_op = FindInstruction(m.get(), "succ_op");
  ASSERT_NE(call_instr, nullptr);
  ASSERT_EQ(call_instr->control_predecessors().size(), 1);
  EXPECT_EQ(call_instr->control_predecessors()[0], pred_op);
  ASSERT_EQ(succ_op->control_predecessors().size(), 1);
  EXPECT_EQ(succ_op->control_predecessors()[0], call_instr);
}

TEST_F(SmallWhileLoopHoistingPassTest, PartitionedFusionPreventsHoisting) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule partitioned_fusion

    fused_acos {
      p0 = f64[4096]{0} parameter(0)
      ROOT r = f64[4096]{0} acos(p0)
    }

    while_body {
      s = (s32[], f64[4096]{0}) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      x = f64[4096]{0} get-tuple-element(s), index=1
      one = s32[] constant(1)
      next_i = s32[] add(i, one)
      next_x = f64[4096]{0} fusion(x), kind=kLoop, calls=fused_acos,
          backend_config={"outer_dimension_partitions":["4"]}
      ROOT out = (s32[], f64[4096]{0}) tuple(next_i, next_x)
    }

    while_cond {
      s = (s32[], f64[4096]{0}) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(8)), direction=LT
    }

    ENTRY main {
      x = f64[4096]{0} parameter(0)
      init = (s32[], f64[4096]{0}) tuple(s32[] constant(0), x)
      ROOT w = (s32[], f64[4096]{0}) while(init),
          condition=while_cond, body=while_body
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunSmallWhileLoopHoistingPass(m.get(),
                                    /*small_buffer_access_size=*/1LL << 30));
  EXPECT_FALSE(changed);
}

TEST_F(SmallWhileLoopHoistingPassTest,
       SmallScatterHoistsUnlessPairedWithPartitionedFusion) {
  constexpr absl::string_view kSmallScatterOnlyHlo = R"hlo(
    HloModule scatter_loop

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    scatter_fusion {
      hist = f32[64]{0} parameter(0)
      idx = s32[128,1]{1,0} parameter(1)
      upd = f32[128]{0} parameter(2)
      ROOT s = f32[64]{0} scatter(hist, idx, upd),
          update_window_dims={}, inserted_window_dims={0},
          scatter_dims_to_operand_dims={0}, index_vector_dim=1,
          to_apply=add_f32
    }

    index_fusion {
      x = f64[128]{0} parameter(0)
      a = f64[128]{0} acos(x)
      c = s32[128]{0} convert(a)
      ROOT idx = s32[128,1]{1,0} bitcast(c)
    }

    while_body {
      s = (s32[], f32[64]{0}, f64[128]{0}, f32[128]{0}) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      hist = f32[64]{0} get-tuple-element(s), index=1
      x = f64[128]{0} get-tuple-element(s), index=2
      upd = f32[128]{0} get-tuple-element(s), index=3
      idx_fusion = s32[128,1]{1,0} fusion(x), kind=kLoop, calls=index_fusion
      next_hist = f32[64]{0} fusion(hist, idx_fusion, upd),
          kind=kLoop, calls=scatter_fusion
      next_i = s32[] add(i, s32[] constant(1))
      ROOT out = (s32[], f32[64]{0}, f64[128]{0}, f32[128]{0})
          tuple(next_i, next_hist, x, upd)
    }

    while_cond {
      s = (s32[], f32[64]{0}, f64[128]{0}, f32[128]{0}) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(5)), direction=LT
    }

    ENTRY main {
      hist = f32[64]{0} parameter(0)
      x = f64[128]{0} parameter(1)
      upd = f32[128]{0} parameter(2)
      init = (s32[], f32[64]{0}, f64[128]{0}, f32[128]{0})
          tuple(s32[] constant(0), hist, x, upd)
      ROOT w = (s32[], f32[64]{0}, f64[128]{0}, f32[128]{0}) while(init),
          condition=while_cond, body=while_body
    }
  )hlo";

  // Unpartitioned index fusion plus single threaded scatter fusion hoists.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_unpartitioned,
                       ParseAndReturnVerifiedModule(kSmallScatterOnlyHlo));
  ASSERT_OK_AND_ASSIGN(
      bool hoisted,
      RunSmallWhileLoopHoistingPass(m_unpartitioned.get(),
                                    /*small_buffer_access_size=*/2LL << 20));
  EXPECT_TRUE(hoisted);

  // When the sibling index fusion carries outer_dimension_partitions > 1, the
  // loop is not hoisted so the index fusion executes on the thread pool.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_partitioned,
                       ParseAndReturnVerifiedModule(kSmallScatterOnlyHlo));
  HloInstruction* idx = FindInstruction(m_partitioned.get(), "idx_fusion");
  ASSERT_NE(idx, nullptr);
  cpu::BackendConfig config;
  config.add_outer_dimension_partitions(8);
  ASSERT_OK(idx->set_backend_config(config));
  ASSERT_OK_AND_ASSIGN(
      bool hoisted_with_parallel_idx,
      RunSmallWhileLoopHoistingPass(m_partitioned.get(),
                                    /*small_buffer_access_size=*/1LL << 30));
  EXPECT_FALSE(hoisted_with_parallel_idx);
}

TEST_F(SmallWhileLoopHoistingPassTest,
       ParallelGemvAndBatchDotAndOutputDotFusionPreventHoisting) {
  // 1024x1024 matrix times 1024 vector lowers to multi workgroup tiled GEMV.
  constexpr absl::string_view kParallelGemvHlo = R"hlo(
    HloModule parallel_gemv

    while_body {
      s = (s32[], f32[1024,1024]{1,0}, f32[1024]{0}) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      a = f32[1024,1024]{1,0} get-tuple-element(s), index=1
      v = f32[1024]{0} get-tuple-element(s), index=2
      next_v = f32[1024]{0} dot(a, v),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT out = (s32[], f32[1024,1024]{1,0}, f32[1024]{0})
          tuple(add(i, s32[] constant(1)), a, next_v)
    }

    while_cond {
      s = (s32[], f32[1024,1024]{1,0}, f32[1024]{0}) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(4)), direction=LT
    }

    ENTRY main {
      a = f32[1024,1024]{1,0} parameter(0)
      v = f32[1024]{0} parameter(1)
      init = (s32[], f32[1024,1024]{1,0}, f32[1024]{0})
          tuple(s32[] constant(0), a, v)
      ROOT w = (s32[], f32[1024,1024]{1,0}, f32[1024]{0}) while(init),
          condition=while_cond, body=while_body
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_gemv,
                       ParseAndReturnVerifiedModule(kParallelGemvHlo));
  const HloInstruction* gemv = FindInstruction(m_gemv.get(), HloOpcode::kDot);
  ASSERT_NE(gemv, nullptr);
  EXPECT_TRUE(cpu::HasMultipleDotWorkGroups(cpu::GetDotOpWorkGroupDim(
      m_gemv->config(), *gemv, target_machine_features_)));
  ASSERT_OK_AND_ASSIGN(
      bool gemv_changed,
      RunSmallWhileLoopHoistingPass(m_gemv.get(),
                                    /*small_buffer_access_size=*/1LL << 30));
  EXPECT_FALSE(gemv_changed);

  // A bare dot wrapped in a kOutput fusion emits the same multiple workgroups.
  constexpr absl::string_view kParallelOutputFusionHlo = R"hlo(
    HloModule parallel_output_dot_fusion

    dot_fusion {
      a = f32[1024,1024]{1,0} parameter(0)
      v = f32[1024]{0} parameter(1)
      ROOT d = f32[1024]{0} dot(a, v),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    while_body {
      s = (s32[], f32[1024,1024]{1,0}, f32[1024]{0}) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      a = f32[1024,1024]{1,0} get-tuple-element(s), index=1
      v = f32[1024]{0} get-tuple-element(s), index=2
      next_v = f32[1024]{0} fusion(a, v), kind=kOutput, calls=dot_fusion
      ROOT out = (s32[], f32[1024,1024]{1,0}, f32[1024]{0})
          tuple(add(i, s32[] constant(1)), a, next_v)
    }

    while_cond {
      s = (s32[], f32[1024,1024]{1,0}, f32[1024]{0}) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(4)), direction=LT
    }

    ENTRY main {
      a = f32[1024,1024]{1,0} parameter(0)
      v = f32[1024]{0} parameter(1)
      init = (s32[], f32[1024,1024]{1,0}, f32[1024]{0})
          tuple(s32[] constant(0), a, v)
      ROOT w = (s32[], f32[1024,1024]{1,0}, f32[1024]{0}) while(init),
          condition=while_cond, body=while_body
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_out_fusion,
                       ParseAndReturnVerifiedModule(kParallelOutputFusionHlo));
  ASSERT_OK_AND_ASSIGN(
      bool out_fusion_changed,
      RunSmallWhileLoopHoistingPass(m_out_fusion.get(),
                                    /*small_buffer_access_size=*/1LL << 30));
  EXPECT_FALSE(out_fusion_changed);

  // Batched GEMV whose inner LHS has 65536 > 32768 elements parallelizes across
  // the batch dimension.
  constexpr absl::string_view kParallelBatchDotHlo = R"hlo(
    HloModule parallel_batch_dot

    while_body {
      s = (s32[], f32[4,256,256]{2,1,0}, f32[4,256]{1,0}) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      a = f32[4,256,256]{2,1,0} get-tuple-element(s), index=1
      v = f32[4,256]{1,0} get-tuple-element(s), index=2
      next_v = f32[4,256]{1,0} dot(a, v),
          lhs_batch_dims={0}, rhs_batch_dims={0},
          lhs_contracting_dims={2}, rhs_contracting_dims={1}
      ROOT out = (s32[], f32[4,256,256]{2,1,0}, f32[4,256]{1,0})
          tuple(add(i, s32[] constant(1)), a, next_v)
    }

    while_cond {
      s = (s32[], f32[4,256,256]{2,1,0}, f32[4,256]{1,0}) parameter(0)
      i = s32[] get-tuple-element(s), index=0
      ROOT lt = pred[] compare(i, s32[] constant(4)), direction=LT
    }

    ENTRY main {
      a = f32[4,256,256]{2,1,0} parameter(0)
      v = f32[4,256]{1,0} parameter(1)
      init = (s32[], f32[4,256,256]{2,1,0}, f32[4,256]{1,0})
          tuple(s32[] constant(0), a, v)
      ROOT w = (s32[], f32[4,256,256]{2,1,0}, f32[4,256]{1,0}) while(init),
          condition=while_cond, body=while_body
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_batch,
                       ParseAndReturnVerifiedModule(kParallelBatchDotHlo));
  const HloInstruction* bdot = FindInstruction(m_batch.get(), HloOpcode::kDot);
  ASSERT_NE(bdot, nullptr);
  EXPECT_EQ(cpu::GetDotOpWorkGroupDim(m_batch->config(), *bdot,
                                      target_machine_features_)
                .x,
            4);
  ASSERT_OK_AND_ASSIGN(
      bool batch_changed,
      RunSmallWhileLoopHoistingPass(m_batch.get(),
                                    /*small_buffer_access_size=*/1LL << 30));
  EXPECT_FALSE(batch_changed);
}

TEST_F(SmallWhileLoopHoistingPassTest,
       ParallelAndTransposingCopiesPreventHoisting) {
  auto run_copy_test =
      [&](absl::string_view src_shape,
          absl::string_view dst_shape) -> absl::StatusOr<bool> {
    const std::string hlo = absl::StrCat(
        "HloModule copy_loop\n"
        "while_body {\n"
        "  s = (s32[], ",
        src_shape, ", ", dst_shape,
        ") parameter(0)\n"
        "  i = s32[] get-tuple-element(s), index=0\n"
        "  a = ",
        src_shape,
        " get-tuple-element(s), index=1\n"
        "  c = ",
        dst_shape,
        " copy(a)\n"
        "  ROOT out = (s32[], ",
        src_shape, ", ", dst_shape,
        ") tuple(add(i, s32[] constant(1)), a, c)\n"
        "}\n"
        "while_cond {\n"
        "  s = (s32[], ",
        src_shape, ", ", dst_shape,
        ") parameter(0)\n"
        "  i = s32[] get-tuple-element(s), index=0\n"
        "  ROOT lt = pred[] compare(i, s32[] constant(3)), direction=LT\n"
        "}\n"
        "ENTRY main {\n"
        "  a = ",
        src_shape,
        " parameter(0)\n"
        "  b = ",
        dst_shape,
        " parameter(1)\n"
        "  init = (s32[], ",
        src_shape, ", ", dst_shape,
        ") tuple(s32[] constant(0), a, b)\n"
        "  ROOT w = (s32[], ",
        src_shape, ", ", dst_shape,
        ") while(init), condition=while_cond, body=while_body\n"
        "}\n");
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo));
    return RunSmallWhileLoopHoistingPass(
        m.get(), /*small_buffer_access_size=*/1LL << 30);
  };

  // Exactly 1 MiB runs as a single block in CopyThunk and remains hoistable.
  ASSERT_OK_AND_ASSIGN(bool hoisted_1mib,
                       run_copy_test("f32[262144]{0}", "f32[262144]{0}"));
  EXPECT_TRUE(hoisted_1mib);

  // Just over 1 MiB splits into multiple parallel copy blocks in CopyThunk.
  ASSERT_OK_AND_ASSIGN(bool hoisted_over_1mib,
                       run_copy_test("f32[262145]{0}", "f32[262145]{0}"));
  EXPECT_FALSE(hoisted_over_1mib);

  // Layout changing copies use TransposePlan and cannot be emitted inline.
  ASSERT_OK_AND_ASSIGN(bool hoisted_transpose,
                       run_copy_test("f32[32,32]{0,1}", "f32[32,32]{1,0}"));
  EXPECT_FALSE(hoisted_transpose);
}

TEST_F(SmallWhileLoopHoistingPassTest,
       ManyArrayLoopFusionsRejectedWhileScalarSliceUpdatesHoist) {
  auto build_many_reductions_hlo = [](int num_reductions) {
    std::string fusions;
    std::string chained;
    for (int k = 0; k < num_reductions; ++k) {
      absl::StrAppend(&fusions, "reduce_fusion_", k,
                      " {\n"
                      "  p = f32[232]{0} parameter(0)\n"
                      "  c = f32[] constant(",
                      k + 1,
                      ")\n"
                      "  cb = f32[232]{0} broadcast(c), dimensions={}\n"
                      "  m = f32[232]{0} multiply(p, cb)\n"
                      "  ROOT r = f32[] reduce(m, f32[] constant(0)),"
                      " dimensions={0}, to_apply=add_f32\n"
                      "}\n"
                      "add_scalar_fusion_",
                      k,
                      " {\n"
                      "  v = f32[232]{0} parameter(0)\n"
                      "  s = f32[] parameter(1)\n"
                      "  sb = f32[232]{0} broadcast(s), dimensions={}\n"
                      "  ROOT out = f32[232]{0} add(v, sb)\n"
                      "}\n");
      const std::string prev = k == 0 ? "x" : absl::StrCat("x", k - 1);
      absl::StrAppend(&chained, "  r", k, " = f32[] fusion(", prev,
                      "), kind=kLoop, calls=reduce_fusion_", k, "\n  x", k,
                      " = f32[232]{0} fusion(", prev, ", r", k,
                      "), kind=kLoop, calls=add_scalar_fusion_", k, "\n");
    }
    return absl::StrCat(
        "HloModule many_reductions\n"
        "add_f32 {\n"
        "  a = f32[] parameter(0)\n"
        "  b = f32[] parameter(1)\n"
        "  ROOT s = f32[] add(a, b)\n"
        "}\n",
        fusions,
        "while_body {\n"
        "  st = (s32[], f32[232]{0}) parameter(0)\n"
        "  i = s32[] get-tuple-element(st), index=0\n"
        "  x = f32[232]{0} get-tuple-element(st), index=1\n",
        chained,
        "  ROOT out = (s32[], f32[232]{0}) tuple(add(i, s32[]"
        " constant(1)), x",
        num_reductions - 1,
        ")\n}\n"
        "while_cond {\n"
        "  st = (s32[], f32[232]{0}) parameter(0)\n"
        "  i = s32[] get-tuple-element(st), index=0\n"
        "  ROOT lt = pred[] compare(i, s32[] constant(10)), direction=LT\n"
        "}\n"
        "ENTRY main {\n"
        "  x = f32[232]{0} parameter(0)\n"
        "  init = (s32[], f32[232]{0}) tuple(s32[] constant(0), x)\n"
        "  ROOT w = (s32[], f32[232]{0}) while(init),"
        " condition=while_cond, body=while_body\n"
        "}\n");
  };

  // A loop with 80 reduction fusions (160 array loop kernels, as in issue
  // #52122) exceeds the hoisted loop kernel budget even under a large byte cap.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_large_body,
                       ParseAndReturnVerifiedModule(
                           build_many_reductions_hlo(/*num_reductions=*/80)));
  ASSERT_OK_AND_ASSIGN(
      bool large_body_changed,
      RunSmallWhileLoopHoistingPass(m_large_body.get(),
                                    /*small_buffer_access_size=*/1LL << 30));
  EXPECT_FALSE(large_body_changed);

  // By contrast, 100 scalar in place dynamic-update-slice fusions (as in
  // jax.issue.40150.grad) contain zero array loop kernels and still hoist.
  std::string dus_fusions;
  std::string dus_chain;
  for (int k = 0; k < 100; ++k) {
    absl::StrAppend(&dus_fusions, "scalar_dus_fusion_", k,
                    " {\n"
                    "  arr = f32[614]{0} parameter(0)\n"
                    "  v = f32[1]{0} parameter(1)\n"
                    "  idx = s32[] parameter(2)\n"
                    "  ROOT d = f32[614]{0} dynamic-update-slice(arr, v, idx)\n"
                    "}\n");
    const std::string prev = k == 0 ? "buf" : absl::StrCat("buf", k - 1);
    absl::StrAppend(&dus_chain, "  buf", k, " = f32[614]{0} fusion(", prev,
                    ", val, i), kind=kLoop, calls=scalar_dus_fusion_", k, "\n");
  }
  const std::string scalar_dus_hlo = absl::StrCat(
      "HloModule many_scalar_dus\n", dus_fusions,
      "while_body {\n"
      "  st = (s32[], f32[614]{0}, f32[1]{0}) parameter(0)\n"
      "  i = s32[] get-tuple-element(st), index=0\n"
      "  buf = f32[614]{0} get-tuple-element(st), index=1\n"
      "  val = f32[1]{0} get-tuple-element(st), index=2\n",
      dus_chain,
      "  ROOT out = (s32[], f32[614]{0}, f32[1]{0})"
      " tuple(add(i, s32[] constant(1)), buf99, val)\n"
      "}\n"
      "while_cond {\n"
      "  st = (s32[], f32[614]{0}, f32[1]{0}) parameter(0)\n"
      "  i = s32[] get-tuple-element(st), index=0\n"
      "  ROOT lt = pred[] compare(i, s32[] constant(614)), direction=LT\n"
      "}\n"
      "ENTRY main {\n"
      "  buf = f32[614]{0} parameter(0)\n"
      "  val = f32[1]{0} parameter(1)\n"
      "  init = (s32[], f32[614]{0}, f32[1]{0})"
      " tuple(s32[] constant(0), buf, val)\n"
      "  ROOT w = (s32[], f32[614]{0}, f32[1]{0}) while(init),"
      " condition=while_cond, body=while_body\n"
      "}\n");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_scalar_dus,
                       ParseAndReturnVerifiedModule(scalar_dus_hlo));
  ASSERT_OK_AND_ASSIGN(
      bool scalar_dus_changed,
      RunSmallWhileLoopHoistingPass(m_scalar_dus.get(),
                                    /*small_buffer_access_size=*/2LL << 20));
  EXPECT_TRUE(scalar_dus_changed);
}

}  // namespace
}  // namespace xla
