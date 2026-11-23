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

#include "xla/service/cpu/parallel_task_assignment.h"

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/codegen/device_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/target_machine_features_stub.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/xla.pb.h"

namespace xla {
namespace {

int64_t TotalPartitions(const cpu::BackendConfig& config) {
  return absl::c_accumulate(config.outer_dimension_partitions(), int64_t{1},
                            std::multiplies<int64_t>());
}

absl::StatusOr<int64_t> GetAssignedPartitions(const HloInstruction* instr) {
  ABSL_ASSIGN_OR_RETURN(cpu::BackendConfig config,
                        instr->backend_config<cpu::BackendConfig>());
  return TotalPartitions(config);
}

class ParallelTaskAssignmentTest : public HloHardwareIndependentTestBase {
 protected:
  const int max_parallelism_ = 10;

  cpu::TargetMachineFeaturesStub target_machine_features_;

  ParallelTaskAssignmentTest()
      : target_machine_features_([](int64_t shape_size) {
          return cpu::TargetMachineFeatures::kEigenExpectedTensorAlignment;
        }) {}

  absl::StatusOr<bool> RunParallelTaskAssigner(HloModule* module) {
    return RunParallelTaskAssigner(module, max_parallelism_,
                                   &target_machine_features_);
  }

  absl::StatusOr<bool> RunParallelTaskAssigner(
      HloModule* module, int64_t max_parallelism,
      const cpu::TargetMachineFeatures* features) {
    return cpu::ParallelTaskAssigner(max_parallelism, shape_size_func_,
                                     features)
        .Run(module);
  }

  static cpu::TargetMachineFeaturesStub MakeTargetFeatures(
      int32_t vector_register_byte_size, std::string enabled_features = "") {
    return cpu::TargetMachineFeaturesStub(
        [](int64_t) {
          return cpu::TargetMachineFeatures::kEigenExpectedTensorAlignment;
        },
        vector_register_byte_size, std::move(enabled_features));
  }

  absl::StatusOr<int64_t> RunAndGetRootScatterTasks(
      absl::string_view hlo, int64_t max_parallelism = 32,
      std::function<void(DebugOptions&)> configure_debug_options = nullptr) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo));
    if (configure_debug_options != nullptr) {
      configure_debug_options(m->mutable_config().mutable_debug_options());
    }
    ABSL_ASSIGN_OR_RETURN(bool changed,
                          RunParallelTaskAssigner(m.get(), max_parallelism,
                                                  &target_machine_features_));
    const HloInstruction* root = m->entry_computation()->root_instruction();
    ABSL_ASSIGN_OR_RETURN(cpu::BackendConfig config,
                          root->backend_config<cpu::BackendConfig>());
    if (!changed) {
      EXPECT_TRUE(config.outer_dimension_partitions().empty());
      return 1;
    }
    EXPECT_EQ(config.outer_dimension_partitions_size(), 1);
    return config.outer_dimension_partitions(0);
  }

  static std::string BuildSimplified1DScatterHlo(
      absl::string_view elem_type, int64_t num_bins, int64_t num_updates,
      absl::string_view reducer_body, bool unique_indices = false,
      absl::string_view elem_attr = "") {
    const std::string dst_layout =
        elem_attr.empty() ? "" : absl::StrCat("{0", elem_attr, "}");
    const std::string upd_layout =
        elem_attr.empty() ? "" : absl::StrCat("{1,0", elem_attr, "}");
    const std::string dst_shape =
        absl::StrCat(elem_type, "[", num_bins, "]", dst_layout);
    const std::string upd_shape =
        absl::StrCat(elem_type, "[", num_updates, ",1]", upd_layout);
    const std::string idx_shape = absl::StrCat(
        "s32[", num_updates, ",1]", elem_attr.empty() ? "" : "{1,0}");
    const absl::string_view unique_attr =
        unique_indices ? "unique_indices=true,\n          " : "";
    const std::string reducer_lines =
        absl::StrContains(reducer_body, "parameter(1)")
            ? absl::StrCat("ROOT rhs = ", elem_type, "[] parameter(1)")
            : absl::StrCat("rhs = ", elem_type, "[] parameter(1)\n        ",
                           reducer_body);
    return absl::StrReplaceAll(
        R"hlo(
      HloModule scatter_1d

      reducer {
        lhs = $ty[] parameter(0)
        $reducer_lines
      }

      wrapped_scatter {
        operand = $dst parameter(0)
        indices = $idx parameter(1)
        updates = $upd parameter(2)
        ROOT scatter = $dst scatter(operand, indices, updates),
            update_window_dims={1},
            inserted_window_dims={},
            scatter_dims_to_operand_dims={0},
            index_vector_dim=1,
            $uniqueto_apply=reducer
      }

      ENTRY main {
        operand = $dst parameter(0)
        indices = $idx parameter(1)
        updates = $upd parameter(2)
        ROOT fusion = $dst fusion(operand, indices, updates),
            kind=kLoop, calls=wrapped_scatter
      }
    )hlo",
        {{"$ty", elem_type},
         {"$reducer_lines", reducer_lines},
         {"$dst", dst_shape},
         {"$idx", idx_shape},
         {"$upd", upd_shape},
         {"$unique", unique_attr}});
  }

  const HloCostAnalysis::ShapeSizeFunction shape_size_func_ =
      cpu::CpuExecutable::ShapeSizeBytes;
};

TEST_F(ParallelTaskAssignmentTest, ReduceWindowParallelized) {
  constexpr absl::string_view hlo_string = R"hlo(
    HloModule m
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY e {
      p0 = f32[2048,512] parameter(0)
      p1 = f32[] parameter(1)
      ROOT reduce-window = f32[64,512] reduce-window(p0, p1),
          window={size=32x1 stride=32x1}, to_apply=add
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* reduce_window =
      FindInstruction(m.get(), HloOpcode::kReduceWindow);
  ASSERT_OK_AND_ASSIGN(int64_t partitions,
                       GetAssignedPartitions(reduce_window));
  EXPECT_GT(partitions, 1);
  EXPECT_LE(partitions, max_parallelism_);

  // A 64 KiB reduce-window finishes in under 20 microseconds and should not
  // pay thread pool dispatch overhead.
  constexpr absl::string_view small_rw_hlo = R"hlo(
    HloModule small_rw
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }
    ENTRY e {
      p0 = f32[128,128] parameter(0)
      p1 = f32[] parameter(1)
      ROOT rw = f32[4,128] reduce-window(p0, p1),
          window={size=32x1 stride=32x1}, to_apply=add
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_small,
                       ParseAndReturnVerifiedModule(small_rw_hlo));
  ASSERT_OK_AND_ASSIGN(bool small_changed,
                       RunParallelTaskAssigner(m_small.get()));
  EXPECT_FALSE(small_changed);
}

TEST_F(ParallelTaskAssignmentTest, DotOperationNotParallelized) {
  const std::string hlo_string = R"(
    HloModule TestTaskParallel_Dot
    ENTRY Dot {
      dot_lhs = f32[196614,2]{1,0} parameter(0)
      dot_rhs = f32[2,1]{1,0} parameter(1)
      ROOT dot = f32[196614,1]{1,0} dot(dot_lhs, dot_rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest,
       FusedComputationWithDotOperationNotParallelized) {
  const std::string hlo_string = R"(
    HloModule TestTaskParallel_DotNestedInFusedComp
    fused_computation.0 {
      parameter.0 = f32[196614,2]{1,0} parameter(0)
      parameter.0.1 = f32[2,1]{1,0} parameter(1)
      parameter.0.2 = f32[196614,1]{1,0} parameter(2)
      dot.0 = f32[196614,1]{1,0} dot(parameter.0, parameter.0.1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT add.0 = f32[196614,1]{1,0} add(dot.0, parameter.0.2)

    }
    ENTRY DotNestedInFusedComp {
      parameter = f32[196614,2]{1,0} parameter(0)
      parameter.1 = f32[2,1]{1,0} parameter(1)
      parameter.2 = f32[196614,1]{1,0} parameter(2)
      ROOT fusion = f32[196614,1]{1,0} fusion(parameter, parameter.1,
        parameter.2), kind=kOutput, calls=fused_computation.0
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, RngOperationNotParallelized) {
  const std::string hlo_string = R"(
    HloModule TestTaskParallel_rng
    ENTRY Rng {
      src0 = f32[] parameter(0)
      src1 = f32[] parameter(1)
      ROOT rng0 = f32[1234567,2]{1,0} rng(f32[] src0, f32[] src1),
      distribution=rng_uniform
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, InfeedOutfeedOperationNotParallelized) {
  const std::string hlo_string = R"(
    HloModule TestTaskParallel_infeed_outfeed
    ENTRY InfeedOutfeed {
      token0 = token[] after-all()
      infeed0 = (u32[12345678,2]{1,0}, token[]) infeed(token0)
      infeed0.data = u32[12345678,2]{1,0} get-tuple-element((u32[12345678,2]{1,0}, token[]) infeed0), index=0
      ROOT outfeed0 = token[] outfeed(infeed0.data, token0)
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, InPlaceDynamicUpdateSliceNotParallelized) {
  // A dynamic-update-slice within a while loop.  This construction is an easy
  // way to make a DUS which can be run "in-place" (i.e. the input and output
  // are the same buffer, and running the DUS only writes to the updated
  // elements).
  const std::string hlo_string = R"(
  HloModule test

  body {
    zero = s32[] constant(0)
    one = s32[] constant(1)
    ten = s32[] constant(10)
    loop_carry = (s32[], u32[1,100], u32[10000,100]) parameter(0)
    i = s32[] get-tuple-element(loop_carry), index=0
    i_plus_ten = s32[] add(i, ten)
    update = u32[1,100] get-tuple-element(loop_carry), index=1
    data = u32[10000,100] get-tuple-element(loop_carry), index=2
    new_data = u32[10000,100] dynamic-update-slice(data, update, i_plus_ten, zero)
    new_i = s32[] add(i, one)
    ROOT tuple = (s32[], u32[1,100], u32[10000,100]) tuple(new_i, update, new_data)
  }

  cond {
    loop_carry = (s32[], u32[1,100], u32[10000,100]) parameter(0)
    two = s32[] constant(2)
    i = s32[] get-tuple-element(loop_carry), index=0
    ROOT less-than = pred[] compare(i, two), direction=LT
  }

  ENTRY test {
    zero = s32[] constant(0)
    initial_i = s32[] parameter(0)
    update = u32[1,100] parameter(1)
    data = u32[10000,100] parameter(2)
    tuple = (s32[], u32[1,100], u32[10000,100]) tuple(initial_i, update, data)
    ROOT while = (s32[], u32[1,100], u32[10000,100]) while(tuple), condition=cond, body=body
  }
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, AllReduceNotParallelized) {
  constexpr char hlo_string[] = R"(
  HloModule TestTaskParallel_allreduce
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY CRS {
      input = f32[1234567] parameter(0)
      ROOT crs = f32[1234567] all-reduce(input), replica_groups={}, to_apply=add
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, ConstantNotParallelized) {
  constexpr char hlo_string[] = R"(
  HloModule TestTaskParallel_constant
    ENTRY const {
      ROOT constant = f32[1234567] constant({...})
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, CustomFusionUnchanged) {
  constexpr absl::string_view hlo_string = R"(
HloModule jit_ynn_bin_ops

fused_computation (matrix_a: f32[1000,1000], matrix_b: f32[1000,1000]) -> f32[1000,1000] {
  matrix_a = f32[1000,1000] parameter(0)
  matrix_b = f32[1000,1000] parameter(1)
  add_result = f32[1000,1000] add(matrix_a, matrix_b)
  subtract_result = f32[1000,1000] subtract(matrix_a, matrix_b)
  ROOT multiply_result = f32[1000,1000] multiply(add_result, subtract_result)
}

ENTRY main (input_x: f32[1000,1000], input_y: f32[1000,1000]) -> f32[1000,1000] {
  input_x = f32[1000,1000] parameter(0)
  input_y = f32[1000,1000] parameter(1)
  ROOT fused_result = f32[1000,1000] fusion(input_x, input_y), kind=kCustom, calls=fused_computation, backend_config={"outer_dimension_partitions":[],"fusion_config":{"kind":"__ynn_fusion"}}
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, NestedTupleParallelized) {
  constexpr absl::string_view hlo_string = R"(
HloModule nested_tuple

fused_computation (lhs: f32[1000,1000], rhs: f32[1000,1000]) -> ((f32[1000,1000], f32[1000,1000])) {
  lhs = f32[1000,1000] parameter(0)
  rhs = f32[1000,1000] parameter(1)
  add_result = f32[1000,1000] add(lhs, rhs)
  subtract_result = f32[1000,1000] subtract(lhs, rhs)
  tuple_0 = (f32[1000,1000], f32[1000,1000]) tuple(add_result, subtract_result)
  ROOT tuple_result = ((f32[1000,1000], f32[1000,1000])) tuple(tuple_0)
}

ENTRY main (lhs: f32[1000,1000], rhs: f32[1000,1000]) -> ((f32[1000,1000], f32[1000,1000])) {
  lhs = f32[1000,1000] parameter(0)
  rhs = f32[1000,1000] parameter(1)
  ROOT fused_result = ((f32[1000,1000], f32[1000,1000])) fusion(lhs, rhs), kind=kLoop, calls=fused_computation
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(m.get()));
  EXPECT_TRUE(changed);
}

TEST_F(ParallelTaskAssignmentTest, UnsimplifiedScatterFusionNotParallelized) {
  constexpr absl::string_view hlo_string = R"hlo(
    HloModule unsimplified_scatter_fusion

    add_f64 {
      lhs = f64[] parameter(0)
      rhs = f64[] parameter(1)
      ROOT sum = f64[] add(lhs, rhs)
    }

    wrapped_scatter {
      operand = f64[4194304] parameter(0)
      indices = s32[262144,1] parameter(1)
      updates = f64[262144] parameter(2)
      ROOT scatter = f64[4194304] scatter(operand, indices, updates),
          update_window_dims={},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_f64
    }

    ENTRY main {
      operand = f64[4194304] parameter(0)
      indices = s32[262144,1] parameter(1)
      updates = f64[262144] parameter(2)
      ROOT fusion = f64[4194304] fusion(operand, indices, updates),
          kind=kLoop, calls=wrapped_scatter
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(int64_t tasks, RunAndGetRootScatterTasks(hlo_string));
  EXPECT_EQ(tasks, 1);
}

TEST_F(ParallelTaskAssignmentTest, SimplifiedScatterFusionParallelized) {
  // Regression check for issue #53475: simplified f64[49152] histogram scatter
  // with N=2097152 updates and fused acos index computation partitions across
  // multiple workgroups along num_updates, while a standalone L2 resident CAS
  // histogram without fused compute stays single threaded and a standalone DRAM
  // resident f64[4194304] histogram partitions across workgroups.
  constexpr absl::string_view fused_acos_hlo = R"hlo(
    HloModule simplified_scatter_fusion

    add_f64 {
      lhs = f64[] parameter(0)
      rhs = f64[] parameter(1)
      ROOT sum = f64[] add(lhs, rhs)
    }

    wrapped_scatter {
      operand = f64[49152] parameter(0)
      raw_indices = f64[2097152,1] parameter(1)
      updates = f64[2097152,1] parameter(2)
      angles = f64[2097152,1] acos(raw_indices)
      indices = s32[2097152,1] convert(angles)
      ROOT scatter = f64[49152] scatter(operand, indices, updates),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_f64
    }

    ENTRY main {
      operand = f64[49152] parameter(0)
      raw_indices = f64[2097152,1] parameter(1)
      updates = f64[2097152,1] parameter(2)
      ROOT fusion = f64[49152] fusion(operand, raw_indices, updates),
          kind=kLoop, calls=wrapped_scatter
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(
      int64_t fused_tasks_32,
      RunAndGetRootScatterTasks(fused_acos_hlo, /*max_parallelism=*/32));
  EXPECT_GE(fused_tasks_32, 16);
  EXPECT_LE(fused_tasks_32, 32);

  ASSERT_OK_AND_ASSIGN(
      int64_t fused_tasks_64,
      RunAndGetRootScatterTasks(fused_acos_hlo, /*max_parallelism=*/64));
  EXPECT_GT(fused_tasks_64, fused_tasks_32);
  EXPECT_LE(fused_tasks_64, 64);

  ASSERT_OK_AND_ASSIGN(
      int64_t standalone_l2_tasks,
      RunAndGetRootScatterTasks(
          BuildSimplified1DScatterHlo("f64", /*num_bins=*/49152,
                                      /*num_updates=*/2097152,
                                      "ROOT sum = f64[] add(lhs, rhs)"),
          /*max_parallelism=*/32));
  EXPECT_EQ(standalone_l2_tasks, 1);

  ASSERT_OK_AND_ASSIGN(
      int64_t standalone_dram_tasks,
      RunAndGetRootScatterTasks(
          BuildSimplified1DScatterHlo("f64", /*num_bins=*/4194304,
                                      /*num_updates=*/262144,
                                      "ROOT sum = f64[] add(lhs, rhs)"),
          /*max_parallelism=*/32));
  EXPECT_GE(standalone_dram_tasks, 16);
  EXPECT_LE(standalone_dram_tasks, 32);
}

TEST_F(ParallelTaskAssignmentTest,
       SmallIntegerBitwiseFusionNotParallelizedDespiteHighOpDensity) {
  constexpr absl::string_view hlo_string = R"hlo(
    HloModule threefry_round

    fused_threefry {
      x0 = u32[51200] parameter(0)
      x1 = u32[51200] parameter(1)
      sh0 = u32[51200] broadcast(u32[] constant(13)), dimensions={}
      d0 = u32[51200] broadcast(u32[] constant(19)), dimensions={}
      a0 = u32[51200] add(x0, x1)
      y0 = u32[51200] xor(a0, or(shift-left(x1, sh0), shift-right-logical(x1, d0)))
      sh1 = u32[51200] broadcast(u32[] constant(15)), dimensions={}
      d1 = u32[51200] broadcast(u32[] constant(17)), dimensions={}
      a1 = u32[51200] add(a0, y0)
      y1 = u32[51200] xor(a1, or(shift-left(y0, sh1), shift-right-logical(y0, d1)))
      sh2 = u32[51200] broadcast(u32[] constant(26)), dimensions={}
      d2 = u32[51200] broadcast(u32[] constant(6)), dimensions={}
      a2 = u32[51200] add(a1, y1)
      y2 = u32[51200] xor(a2, or(shift-left(y1, sh2), shift-right-logical(y1, d2)))
      sh3 = u32[51200] broadcast(u32[] constant(6)), dimensions={}
      d3 = u32[51200] broadcast(u32[] constant(26)), dimensions={}
      a3 = u32[51200] add(a2, y2)
      ROOT y3 = u32[51200] xor(a3, or(shift-left(y2, sh3), shift-right-logical(y2, d3)))
    }

    ENTRY main {
      x0 = u32[51200] parameter(0)
      x1 = u32[51200] parameter(1)
      ROOT out = u32[51200] fusion(x0, x1), kind=kLoop, calls=fused_threefry
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed,
                       RunParallelTaskAssigner(m.get(), /*max_parallelism=*/64,
                                               &target_machine_features_));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, VectorWidthControlsComputeParallelism) {
  constexpr absl::string_view hlo_string = R"hlo(
    HloModule bitwise_fusion

    fused_comp {
      x0 = u32[393216] parameter(0)
      x1 = u32[393216] parameter(1)
      s0 = u32[393216] broadcast(u32[] constant(13)), dimensions={}
      d0 = u32[393216] broadcast(u32[] constant(19)), dimensions={}
      a0 = u32[393216] add(x0, x1)
      y0 = u32[393216] xor(a0, or(shift-left(x1, s0), shift-right-logical(x1, d0)))
      a1 = u32[393216] add(a0, y0)
      y1 = u32[393216] xor(a1, or(shift-left(y0, s0), shift-right-logical(y0, d0)))
      a2 = u32[393216] add(a1, y1)
      y2 = u32[393216] xor(a2, or(shift-left(y1, s0), shift-right-logical(y1, d0)))
      a3 = u32[393216] add(a2, y2)
      y3 = u32[393216] xor(a3, or(shift-left(y2, s0), shift-right-logical(y2, d0)))
      a4 = u32[393216] add(a3, y3)
      y4 = u32[393216] xor(a4, or(shift-left(y3, s0), shift-right-logical(y3, d0)))
      a5 = u32[393216] add(a4, y4)
      y5 = u32[393216] xor(a5, or(shift-left(y4, s0), shift-right-logical(y4, d0)))
      a6 = u32[393216] add(a5, y5)
      ROOT y6 = u32[393216] xor(a6, or(shift-left(y5, s0), shift-right-logical(y5, d0)))
    }

    ENTRY main {
      x0 = u32[393216] parameter(0)
      x1 = u32[393216] parameter(1)
      ROOT out = u32[393216] fusion(x0, x1), kind=kLoop, calls=fused_comp
    }
  )hlo";

  auto run_with_vec_bytes =
      [&](int32_t vec_bytes, int32_t prefer_bits) -> absl::StatusOr<int64_t> {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
    m->mutable_config().mutable_debug_options().set_xla_cpu_prefer_vector_width(
        prefer_bits);
    cpu::TargetMachineFeaturesStub features =
        MakeTargetFeatures(vec_bytes, vec_bytes >= 64 ? "+avx,+avx512f" : "");
    ABSL_ASSIGN_OR_RETURN(
        bool changed,
        RunParallelTaskAssigner(m.get(), /*max_parallelism=*/64, &features));
    EXPECT_TRUE(changed);
    return GetAssignedPartitions(m->entry_computation()->root_instruction());
  };

  ASSERT_OK_AND_ASSIGN(int64_t scalar_tasks, run_with_vec_bytes(0, 256));
  ASSERT_OK_AND_ASSIGN(int64_t neon_tasks, run_with_vec_bytes(16, 256));
  ASSERT_OK_AND_ASSIGN(int64_t avx2_tasks, run_with_vec_bytes(32, 256));
  ASSERT_OK_AND_ASSIGN(int64_t avx512_tasks, run_with_vec_bytes(64, 512));
  EXPECT_GT(scalar_tasks, neon_tasks);
  EXPECT_GT(neon_tasks, avx2_tasks);
  EXPECT_GT(avx2_tasks, avx512_tasks);
}

TEST_F(ParallelTaskAssignmentTest, SmallAddAndSmallGatherStaySingleThreaded) {
  constexpr absl::string_view add_hlo = R"hlo(
    HloModule small_f64_add

    fused_add {
      s = f64[32768] parameter(0)
      x = f64[8,32768] parameter(1)
      i = s32[] parameter(2)
      sl = f64[1,32768] dynamic-slice(x, i, s32[] constant(0)),
          dynamic_slice_sizes={1,32768}
      ROOT sum = f64[32768] add(s, f64[32768] reshape(sl))
    }

    ENTRY main {
      s = f64[32768] parameter(0)
      x = f64[8,32768] parameter(1)
      i = s32[] parameter(2)
      ROOT out = f64[32768] fusion(s, x, i), kind=kLoop, calls=fused_add
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_add,
                       ParseAndReturnVerifiedModule(add_hlo));
  ASSERT_OK_AND_ASSIGN(
      bool add_changed,
      RunParallelTaskAssigner(m_add.get(), /*max_parallelism=*/32,
                              &target_machine_features_));
  EXPECT_FALSE(add_changed);

  // Small token embedding gather out of a large table must not be sized by the
  // entire table.
  constexpr absl::string_view gather_hlo = R"hlo(
    HloModule small_embedding_gather

    fused_gather {
      table = f32[262144,128] parameter(0)
      indices = s32[8,1] parameter(1)
      ROOT g = f32[8,128] gather(table, indices),
          offset_dims={1},
          collapsed_slice_dims={0},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={1,128}
    }

    ENTRY main {
      table = f32[262144,128] parameter(0)
      indices = s32[8,1] parameter(1)
      ROOT out = f32[8,128] fusion(table, indices),
          kind=kLoop, calls=fused_gather
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_gather,
                       ParseAndReturnVerifiedModule(gather_hlo));
  ASSERT_OK_AND_ASSIGN(
      bool gather_changed,
      RunParallelTaskAssigner(m_gather.get(), /*max_parallelism=*/32,
                              &target_machine_features_));
  EXPECT_FALSE(gather_changed);
}

TEST_F(ParallelTaskAssignmentTest, ScalarLibmFp64AcosFusionParallelized) {
  // Expanded FP64 acos (sqrt + unvectorized atan2) on 16K elements has a tiny
  // memory footprint but hundreds of microseconds of scalar libm work.
  constexpr absl::string_view f64_acos_hlo = R"hlo(
    HloModule f64_acos

    fused_acos {
      x = f64[16384] parameter(0)
      one = f64[16384] broadcast(f64[] constant(1.0)), dimensions={}
      x2 = f64[16384] multiply(x, x)
      rad = f64[16384] sqrt(f64[16384] subtract(one, x2))
      ROOT out = f64[16384] atan2(rad, x)
    }

    ENTRY main {
      x = f64[16384] parameter(0)
      ROOT out = f64[16384] fusion(x), kind=kLoop, calls=fused_acos
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_acos,
                       ParseAndReturnVerifiedModule(f64_acos_hlo));
  ASSERT_OK_AND_ASSIGN(
      bool acos_changed,
      RunParallelTaskAssigner(m_acos.get(), /*max_parallelism=*/32,
                              &target_machine_features_));
  EXPECT_TRUE(acos_changed);
  ASSERT_OK_AND_ASSIGN(
      int64_t acos_tasks,
      GetAssignedPartitions(m_acos->entry_computation()->root_instruction()));
  EXPECT_GE(acos_tasks, 8);
}

TEST_F(ParallelTaskAssignmentTest, SmallUnaryAtan2OneStaysSingleThreaded) {
  // atan2(y, 1.0) lowers to vectorized atan and stays single threaded at 4K
  // elements, unlike general atan2(y, x) which calls scalar libm.
  constexpr absl::string_view unary_atan_hlo = R"hlo(
    HloModule f64_unary_atan

    fused_atan {
      y = f64[4096] parameter(0)
      one = f64[4096] broadcast(f64[] constant(1.0)), dimensions={}
      ROOT out = f64[4096] atan2(y, one)
    }

    ENTRY main {
      y = f64[4096] parameter(0)
      ROOT out = f64[4096] fusion(y), kind=kLoop, calls=fused_atan
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_atan,
                       ParseAndReturnVerifiedModule(unary_atan_hlo));
  ASSERT_OK_AND_ASSIGN(
      bool atan_changed,
      RunParallelTaskAssigner(m_atan.get(), /*max_parallelism=*/16,
                              &target_machine_features_));
  EXPECT_FALSE(atan_changed);
}

TEST_F(ParallelTaskAssignmentTest, TinyVectorizedExpLogStaysSingleThreaded) {
  constexpr absl::string_view tiny_transcendental_hlo = R"hlo(
    HloModule tiny_f32_exp

    fused_exp {
      x = f32[4096] parameter(0)
      ROOT out = f32[4096] exponential(f32[4096] log(x))
    }

    ENTRY main {
      x = f32[4096] parameter(0)
      ROOT out = f32[4096] fusion(x), kind=kLoop, calls=fused_exp
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_tiny,
                       ParseAndReturnVerifiedModule(tiny_transcendental_hlo));
  ASSERT_OK_AND_ASSIGN(
      bool tiny_changed,
      RunParallelTaskAssigner(m_tiny.get(), /*max_parallelism=*/32,
                              &target_machine_features_));
  EXPECT_FALSE(tiny_changed);
}

TEST_F(ParallelTaskAssignmentTest, LargeVectorizedF32ExpParallelized) {
  constexpr absl::string_view large_f32_exp_hlo = R"hlo(
    HloModule large_f32_exp

    fused_exp {
      x = f32[262144] parameter(0)
      s = f32[262144] parameter(1)
      ROOT out = f32[262144] add(s, f32[262144] exponential(x))
    }

    ENTRY main {
      x = f32[262144] parameter(0)
      s = f32[262144] parameter(1)
      ROOT out = f32[262144] fusion(x, s), kind=kLoop, calls=fused_exp
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_large_exp,
                       ParseAndReturnVerifiedModule(large_f32_exp_hlo));
  ASSERT_OK_AND_ASSIGN(
      bool large_exp_changed,
      RunParallelTaskAssigner(m_large_exp.get(), /*max_parallelism=*/32,
                              &target_machine_features_));
  EXPECT_TRUE(large_exp_changed);
  ASSERT_OK_AND_ASSIGN(
      int64_t large_exp_tasks,
      GetAssignedPartitions(
          m_large_exp->entry_computation()->root_instruction()));
  EXPECT_GE(large_exp_tasks, 4);
}

TEST_F(ParallelTaskAssignmentTest,
       BroadcastScalarLibmOpDoesNotTriggerParallelization) {
  // A scalar libm op hoisted out of a broadcast by LICM must not trigger
  // parallelization on a small output shape.
  constexpr absl::string_view hoisted_scalar_atan2_hlo = R"hlo(
    HloModule hoisted_scalar_atan2

    fused_bcast {
      x = f64[16384] parameter(0)
      a = f64[] parameter(1)
      b = f64[] parameter(2)
      angle = f64[] atan2(a, b)
      bcast = f64[16384] broadcast(angle), dimensions={}
      ROOT out = f64[16384] add(x, bcast)
    }

    ENTRY main {
      x = f64[16384] parameter(0)
      a = f64[] parameter(1)
      b = f64[] parameter(2)
      ROOT out = f64[16384] fusion(x, a, b), kind=kLoop, calls=fused_bcast
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_scalar_bcast,
                       ParseAndReturnVerifiedModule(hoisted_scalar_atan2_hlo));
  ASSERT_OK_AND_ASSIGN(
      bool scalar_bcast_changed,
      RunParallelTaskAssigner(m_scalar_bcast.get(), /*max_parallelism=*/32,
                              &target_machine_features_));
  EXPECT_FALSE(scalar_bcast_changed);
}

TEST_F(ParallelTaskAssignmentTest, LargeMemoryBoundFusionBandwidthCapped) {
  constexpr absl::string_view mem_bound_hlo = R"hlo(
    HloModule large_mem_bound

    fused_add {
      a = f64[4194304] parameter(0)
      b = f64[4194304] parameter(1)
      ROOT out = f64[4194304] add(a, b)
    }

    ENTRY main {
      a = f64[4194304] parameter(0)
      b = f64[4194304] parameter(1)
      ROOT out = f64[4194304] fusion(a, b), kind=kLoop, calls=fused_add
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_mem,
                       ParseAndReturnVerifiedModule(mem_bound_hlo));
  ASSERT_OK_AND_ASSIGN(
      bool mem_changed,
      RunParallelTaskAssigner(m_mem.get(), /*max_parallelism=*/64,
                              &target_machine_features_));
  EXPECT_TRUE(mem_changed);
  ASSERT_OK_AND_ASSIGN(
      int64_t mem_tasks,
      GetAssignedPartitions(m_mem->entry_computation()->root_instruction()));
  EXPECT_EQ(mem_tasks, 8);
}

TEST_F(ParallelTaskAssignmentTest,
       LargeComputeBoundFusionScalesPastMemoryBandwidthCap) {
  constexpr absl::string_view compute_bound_hlo = R"hlo(
    HloModule large_compute_bound

    fused_atan2 {
      y = f64[262144] parameter(0)
      x = f64[262144] parameter(1)
      ROOT out = f64[262144] atan2(y, x)
    }

    ENTRY main {
      y = f64[262144] parameter(0)
      x = f64[262144] parameter(1)
      ROOT out = f64[262144] fusion(y, x), kind=kLoop, calls=fused_atan2
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_comp,
                       ParseAndReturnVerifiedModule(compute_bound_hlo));
  ASSERT_OK_AND_ASSIGN(
      bool comp_changed,
      RunParallelTaskAssigner(m_comp.get(), /*max_parallelism=*/64,
                              &target_machine_features_));
  EXPECT_TRUE(comp_changed);
  ASSERT_OK_AND_ASSIGN(
      int64_t comp_tasks,
      GetAssignedPartitions(m_comp->entry_computation()->root_instruction()));
  EXPECT_GT(comp_tasks, 8);
}

TEST_F(ParallelTaskAssignmentTest, WhileBodySelectiveParallelization) {
  // Inside a while body, a tiny threefry fusion and a narrow contended
  // f64[1024] scatter stay single threaded while an expensive FP64 atan2 index
  // fusion and a DRAM resident f64[4194304] simplified scatter both get
  // partitioned.
  constexpr absl::string_view hlo_string = R"hlo(
    HloModule while_selective

    add_f64 {
      lhs = f64[] parameter(0)
      rhs = f64[] parameter(1)
      ROOT sum = f64[] add(lhs, rhs)
    }

    tiny_bitwise {
      x = u32[20,200] parameter(0)
      y = u32[20,200] parameter(1)
      sh = u32[20,200] broadcast(u32[] constant(13)), dimensions={}
      dr = u32[20,200] broadcast(u32[] constant(19)), dimensions={}
      a = u32[20,200] add(x, y)
      ROOT r = u32[20,200] xor(a, or(shift-left(y, sh), shift-right-logical(y, dr)))
    }

    heavy_index {
      p = f64[262144] parameter(0)
      one = f64[262144] broadcast(f64[] constant(1.0)), dimensions={}
      rad = f64[262144] sqrt(f64[262144] subtract(one, multiply(p, p)))
      ang = f64[262144] atan2(rad, p)
      ROOT idx = s32[262144,1] reshape(s32[262144] convert(ang))
    }

    small_scatter_comp {
      hist = f64[1024] parameter(0)
      idx = s32[262144,1] parameter(1)
      upd = f64[262144,1] parameter(2)
      ROOT s = f64[1024] scatter(hist, idx, upd),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_f64
    }

    wide_scatter_comp {
      hist = f64[4194304] parameter(0)
      idx = s32[262144,1] parameter(1)
      upd = f64[262144,1] parameter(2)
      ROOT s = f64[4194304] scatter(hist, idx, upd),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_f64
    }

    body {
      st = (s32[], u32[20,200], f64[1024], f64[4194304], f64[262144], f64[262144,1]) parameter(0)
      i = s32[] get-tuple-element(st), index=0
      rng = u32[20,200] get-tuple-element(st), index=1
      small_hist = f64[1024] get-tuple-element(st), index=2
      hist = f64[4194304] get-tuple-element(st), index=3
      p = f64[262144] get-tuple-element(st), index=4
      upd = f64[262144,1] get-tuple-element(st), index=5
      next_i = s32[] add(i, s32[] constant(1))
      next_rng = u32[20,200] fusion(rng, rng), kind=kLoop, calls=tiny_bitwise
      index_fusion = s32[262144,1] fusion(p), kind=kLoop, calls=heavy_index
      next_small_hist = f64[1024] fusion(small_hist, index_fusion, upd), kind=kLoop, calls=small_scatter_comp
      next_hist = f64[4194304] fusion(hist, index_fusion, upd), kind=kLoop, calls=wide_scatter_comp
      ROOT out = (s32[], u32[20,200], f64[1024], f64[4194304], f64[262144], f64[262144,1])
          tuple(next_i, next_rng, next_small_hist, next_hist, p, upd)
    }

    cond {
      st = (s32[], u32[20,200], f64[1024], f64[4194304], f64[262144], f64[262144,1]) parameter(0)
      i = s32[] get-tuple-element(st), index=0
      ROOT lt = pred[] compare(i, s32[] constant(8)), direction=LT
    }

    ENTRY main {
      init = (s32[], u32[20,200], f64[1024], f64[4194304], f64[262144], f64[262144,1]) parameter(0)
      ROOT res = (s32[], u32[20,200], f64[1024], f64[4194304], f64[262144], f64[262144,1])
          while(init), condition=cond, body=body
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed,
                       RunParallelTaskAssigner(m.get(), /*max_parallelism=*/32,
                                               &target_machine_features_));
  EXPECT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(int64_t rng_tasks, GetAssignedPartitions(FindInstruction(
                                              m.get(), "next_rng")));
  ASSERT_OK_AND_ASSIGN(int64_t idx_tasks, GetAssignedPartitions(FindInstruction(
                                              m.get(), "index_fusion")));
  ASSERT_OK_AND_ASSIGN(
      int64_t small_scatter_tasks,
      GetAssignedPartitions(FindInstruction(m.get(), "next_small_hist")));
  ASSERT_OK_AND_ASSIGN(
      int64_t scatter_tasks,
      GetAssignedPartitions(FindInstruction(m.get(), "next_hist")));
  EXPECT_EQ(rng_tasks, 1);
  EXPECT_GT(idx_tasks, 1);
  EXPECT_EQ(small_scatter_tasks, 1);
  EXPECT_GT(scatter_tasks, 1);
}

TEST_F(ParallelTaskAssignmentTest, LayoutChangingCopyParallelized) {
  // LayoutAssignment typically represents a physical matrix transpose as a
  // bitcast kTranspose followed by a layout changing kCopy.
  constexpr absl::string_view transpose_copy_hlo = R"hlo(
    HloModule transpose_copy

    fused_transpose {
      p0 = bf16[512,512]{1,0} parameter(0)
      t = bf16[512,512]{0,1} transpose(p0), dimensions={1,0}
      ROOT c = bf16[512,512]{1,0} copy(t)
    }

    ENTRY main {
      p0 = bf16[512,512]{1,0} parameter(0)
      ROOT out = bf16[512,512]{1,0} fusion(p0),
          kind=kLoop, calls=fused_transpose
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_transpose,
                       ParseAndReturnVerifiedModule(transpose_copy_hlo));
  ASSERT_OK_AND_ASSIGN(
      bool transpose_changed,
      RunParallelTaskAssigner(m_transpose.get(), /*max_parallelism=*/32,
                              &target_machine_features_));
  EXPECT_TRUE(transpose_changed);
  ASSERT_OK_AND_ASSIGN(
      int64_t transpose_tasks,
      GetAssignedPartitions(
          m_transpose->entry_computation()->root_instruction()));
  EXPECT_GE(transpose_tasks, 2);
}

TEST_F(ParallelTaskAssignmentTest, MultiDimensionBf16ReduceParallelized) {
  // A loop emitter BF16 multi dimension reduction over 131K elements runs a
  // scalar extend, F32 add, and round to nearest even truncate chain and
  // benefits strongly from parallelization despite its 256 KiB footprint.
  constexpr absl::string_view bf16_reduce_hlo = R"hlo(
    HloModule bf16_reduce

    add_bf16 {
      lhs = bf16[] parameter(0)
      rhs = bf16[] parameter(1)
      ROOT sum = bf16[] add(lhs, rhs)
    }

    fused_reduce {
      p0 = bf16[256,32,16]{2,1,0} parameter(0)
      init = bf16[] constant(0)
      ROOT r = bf16[256]{0} reduce(p0, init), dimensions={1,2}, to_apply=add_bf16
    }

    ENTRY main {
      p0 = bf16[256,32,16]{2,1,0} parameter(0)
      ROOT out = bf16[256]{0} fusion(p0), kind=kLoop, calls=fused_reduce
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_reduce,
                       ParseAndReturnVerifiedModule(bf16_reduce_hlo));
  ASSERT_OK_AND_ASSIGN(
      bool reduce_changed,
      RunParallelTaskAssigner(m_reduce.get(), /*max_parallelism=*/32,
                              &target_machine_features_));
  EXPECT_TRUE(reduce_changed);
  ASSERT_OK_AND_ASSIGN(
      int64_t reduce_tasks,
      GetAssignedPartitions(m_reduce->entry_computation()->root_instruction()));
  EXPECT_GE(reduce_tasks, 8);
}

TEST_F(ParallelTaskAssignmentTest,
       ManyOperandNarrowConcatenateStaysSingleThreaded) {
  // Partitioning a concatenate of many scalar width columns introduces dynamic
  // workgroup bounds around hundreds of scalar copies, inflating LLVM compile
  // time with negligible runtime benefit.
  std::string params;
  std::string fparams;
  std::string args;
  std::string entry_args;
  for (int i = 0; i < 216; ++i) {
    if (i > 0) {
      absl::StrAppend(&args, ", ");
      absl::StrAppend(&entry_args, ", ");
    }
    absl::StrAppend(&params, "  p", i, " = f32[198,1]{1,0} parameter(", i,
                    ")\n");
    absl::StrAppend(&fparams, "  fp", i, " = f32[198,1]{1,0} parameter(", i,
                    ")\n");
    absl::StrAppend(&args, "fp", i);
    absl::StrAppend(&entry_args, "p", i);
  }
  const std::string concat_hlo =
      absl::StrCat("HloModule wide_concat\n\nfused_concat {\n", fparams,
                   "  ROOT c = f32[198,216]{1,0} concatenate(", args,
                   "), dimensions={1}\n}\n\nENTRY main {\n", params,
                   "  ROOT out = f32[198,216]{1,0} fusion(", entry_args,
                   "), kind=kLoop, calls=fused_concat\n}\n");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_concat,
                       ParseAndReturnVerifiedModule(concat_hlo));
  ASSERT_OK_AND_ASSIGN(
      bool concat_changed,
      RunParallelTaskAssigner(m_concat.get(), /*max_parallelism=*/64,
                              &target_machine_features_));
  EXPECT_FALSE(concat_changed);
}

TEST_F(ParallelTaskAssignmentTest, LargeFewOperandConcatenateParallelized) {
  constexpr absl::string_view large_concat_hlo = R"hlo(
    HloModule large_concat

    fused_concat {
      p0 = f32[256,128,64]{2,1,0} parameter(0)
      p1 = f32[256,128,64]{2,1,0} parameter(1)
      ROOT c = f32[256,128,128]{2,1,0} concatenate(p0, p1), dimensions={2}
    }

    ENTRY main {
      p0 = f32[256,128,64]{2,1,0} parameter(0)
      p1 = f32[256,128,64]{2,1,0} parameter(1)
      ROOT out = f32[256,128,128]{2,1,0} fusion(p0, p1),
          kind=kLoop, calls=fused_concat
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_concat,
                       ParseAndReturnVerifiedModule(large_concat_hlo));
  ASSERT_OK_AND_ASSIGN(
      bool concat_changed,
      RunParallelTaskAssigner(m_concat.get(), /*max_parallelism=*/64,
                              &target_machine_features_));
  EXPECT_TRUE(concat_changed);
  ASSERT_OK_AND_ASSIGN(
      int64_t concat_tasks,
      GetAssignedPartitions(m_concat->entry_computation()->root_instruction()));
  EXPECT_EQ(concat_tasks, 8);
}

TEST_F(ParallelTaskAssignmentTest,
       TiledGemvDotFusionStaysMemoryBandwidthCapped) {
  constexpr absl::string_view gemv_hlo = R"hlo(
    HloModule tiled_gemv

    wrapped_dot {
      lhs = f32[1152,1]{1,0} parameter(0)
      rhs = f32[1152,13824]{1,0} parameter(1)
      ROOT dot = f32[1,13824]{1,0} dot(lhs, rhs),
          lhs_contracting_dims={0}, rhs_contracting_dims={0}
    }

    ENTRY main {
      lhs = f32[1152,1]{1,0} parameter(0)
      rhs = f32[1152,13824]{1,0} parameter(1)
      ROOT out = f32[1,13824]{1,0} fusion(lhs, rhs),
          kind=kLoop, calls=wrapped_dot
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(gemv_hlo));
  ASSERT_OK_AND_ASSIGN(bool changed,
                       RunParallelTaskAssigner(m.get(), /*max_parallelism=*/64,
                                               &target_machine_features_));
  EXPECT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(
      int64_t tasks,
      GetAssignedPartitions(m->entry_computation()->root_instruction()));
  EXPECT_EQ(tasks, 8);
}

TEST_F(ParallelTaskAssignmentTest,
       BatchedSmallRowGatherAvoidsOverPartitioning) {
  constexpr absl::string_view batched_gather_hlo = R"hlo(
    HloModule batched_gather

    fused_gather {
      x = bf16[1024,512]{1,0} parameter(0)
      idx = s32[1024,512,1]{2,1,0} parameter(1)
      ROOT g = bf16[1024,512]{1,0} gather(x, idx),
          offset_dims={},
          collapsed_slice_dims={1},
          start_index_map={1},
          operand_batching_dims={0},
          start_indices_batching_dims={0},
          index_vector_dim=2,
          slice_sizes={1,1}
    }

    ENTRY main {
      x = bf16[1024,512]{1,0} parameter(0)
      idx = s32[1024,512,1]{2,1,0} parameter(1)
      ROOT out = bf16[1024,512]{1,0} fusion(x, idx),
          kind=kLoop, calls=fused_gather
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(batched_gather_hlo));
  ASSERT_OK_AND_ASSIGN(bool changed,
                       RunParallelTaskAssigner(m.get(), /*max_parallelism=*/64,
                                               &target_machine_features_));
  EXPECT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(
      int64_t tasks,
      GetAssignedPartitions(m->entry_computation()->root_instruction()));
  EXPECT_EQ(1024 % tasks, 0);
  EXPECT_LE(tasks, 8);
}

TEST_F(ParallelTaskAssignmentTest, StridedNarrowSliceFusionParallelized) {
  constexpr absl::string_view strided_slice_hlo = R"hlo(
    HloModule strided_slice

    fused_add_slice {
      p0 = f32[6272,6,8]{2,1,0} parameter(0)
      p1 = f32[6272,6,8]{2,1,0} parameter(1)
      sum = f32[6272,6,8]{2,1,0} add(p0, p1)
      ROOT s = f32[6272,1,8]{2,1,0} slice(sum), slice={[0:6272], [5:6], [0:8]}
    }

    ENTRY main {
      p0 = f32[6272,6,8]{2,1,0} parameter(0)
      p1 = f32[6272,6,8]{2,1,0} parameter(1)
      ROOT out = f32[6272,1,8]{2,1,0} fusion(p0, p1),
          kind=kLoop, calls=fused_add_slice
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(strided_slice_hlo));
  ASSERT_OK_AND_ASSIGN(bool changed,
                       RunParallelTaskAssigner(m.get(), /*max_parallelism=*/64,
                                               &target_machine_features_));
  EXPECT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(
      int64_t tasks,
      GetAssignedPartitions(m->entry_computation()->root_instruction()));
  EXPECT_GE(tasks, 2);
  EXPECT_EQ(6272 % tasks, 0);
}

TEST_F(ParallelTaskAssignmentTest, PrefersExactDivisorAndSkipsBorderlineWork) {
  // A 512 KiB elementwise add sits near the dispatch break even point and
  // stays single threaded under the wall and CPU trade off objective.
  constexpr absl::string_view small_add_hlo = R"hlo(
    HloModule borderline_add

    fused_add {
      p0 = f32[512,256]{1,0} parameter(0)
      p1 = f32[256]{0} parameter(1)
      b = f32[512,256]{1,0} broadcast(p1), dimensions={1}
      ROOT out = f32[512,256]{1,0} add(p0, b)
    }

    ENTRY main {
      p0 = f32[512,256]{1,0} parameter(0)
      p1 = f32[256]{0} parameter(1)
      ROOT out = f32[512,256]{1,0} fusion(p0, p1), kind=kLoop, calls=fused_add
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_small,
                       ParseAndReturnVerifiedModule(small_add_hlo));
  ASSERT_OK_AND_ASSIGN(
      bool small_changed,
      RunParallelTaskAssigner(m_small.get(), /*max_parallelism=*/64,
                              &target_machine_features_));
  EXPECT_FALSE(small_changed);

  // A compute heavy fusion on a power of two outer dimension chooses an exact
  // divisor so LoopFusionKernelEmitter avoids emitting an in bounds guard.
  constexpr absl::string_view log1p_hlo = R"hlo(
    HloModule power_of_two_log1p

    fused_log1p {
      p0 = f32[512,256,1]{2,1,0} parameter(0)
      z = f32[512,256,1]{2,1,0} broadcast(f32[] constant(0)), dimensions={}
      m = f32[512,256,1]{2,1,0} maximum(p0, z)
      ROOT out = f32[512,256,1]{2,1,0} log-plus-one(m)
    }

    ENTRY main {
      p0 = f32[512,256,1]{2,1,0} parameter(0)
      ROOT out = f32[512,256,1]{2,1,0} fusion(p0), kind=kLoop, calls=fused_log1p
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_log1p,
                       ParseAndReturnVerifiedModule(log1p_hlo));
  ASSERT_OK_AND_ASSIGN(
      bool log1p_changed,
      RunParallelTaskAssigner(m_log1p.get(), /*max_parallelism=*/64,
                              &target_machine_features_));
  EXPECT_TRUE(log1p_changed);
  ASSERT_OK_AND_ASSIGN(
      int64_t log1p_tasks,
      GetAssignedPartitions(m_log1p->entry_computation()->root_instruction()));
  EXPECT_EQ(512 % log1p_tasks, 0);
  EXPECT_LE(log1p_tasks, 8);
}

TEST_F(ParallelTaskAssignmentTest,
       SmallPredAndPackedSubByteFusionsStaySingleThreaded) {
  // Small PRED and sub byte S4 elementwise fusions use StorageBitWidth and
  // layout element_size_in_bits without overestimating memory or undercounting
  // SIMD lanes.
  constexpr absl::string_view small_pred_s4_hlo = R"hlo(
    HloModule small_pred_s4

    fused_pred {
      p0 = pred[65536]{0} parameter(0)
      p1 = pred[65536]{0} parameter(1)
      ROOT a = pred[65536]{0} and(p0, p1)
    }

    fused_s4 {
      q0 = s4[65536]{0:E(4)} parameter(0)
      q1 = s4[65536]{0:E(4)} parameter(1)
      ROOT b = s4[65536]{0:E(4)} add(q0, q1)
    }

    ENTRY main {
      p0 = pred[65536]{0} parameter(0)
      p1 = pred[65536]{0} parameter(1)
      q0 = s4[65536]{0:E(4)} parameter(2)
      q1 = s4[65536]{0:E(4)} parameter(3)
      r0 = pred[65536]{0} fusion(p0, p1), kind=kLoop, calls=fused_pred
      r1 = s4[65536]{0:E(4)} fusion(q0, q1), kind=kLoop, calls=fused_s4
      ROOT out = (pred[65536]{0}, s4[65536]{0:E(4)}) tuple(r0, r1)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_pred_s4,
                       ParseAndReturnVerifiedModule(small_pred_s4_hlo));
  ASSERT_OK_AND_ASSIGN(
      bool pred_s4_changed,
      RunParallelTaskAssigner(m_pred_s4.get(), /*max_parallelism=*/32,
                              &target_machine_features_));
  EXPECT_FALSE(pred_s4_changed);
}

TEST_F(ParallelTaskAssignmentTest,
       BitcastAndTupleInPlaceDynamicUpdateSliceFusionsNotParallelized) {
  // Both a DUS fusion with a bitcast root and a tuple output fusion of multiple
  // DUS roots lower via the in place DUS emitter and must be skipped even when
  // the destination buffers are large.
  constexpr absl::string_view in_place_dus_hlo = R"hlo(
    HloModule bitcast_and_tuple_dus

    bitcast_dus {
      buf = f32[4,1048576]{1,0} parameter(0)
      upd = f32[1,16]{1,0} parameter(1)
      i = s32[] parameter(2)
      z = s32[] constant(0)
      d = f32[4,1048576]{1,0} dynamic-update-slice(buf, upd, i, z)
      ROOT bc = f32[4194304]{0} bitcast(d)
    }

    tuple_dus {
      a = f32[4194304]{0} parameter(0)
      b = f32[4194304]{0} parameter(1)
      ua = f32[16]{0} parameter(2)
      ub = f32[16]{0} parameter(3)
      idx = s32[] parameter(4)
      da = f32[4194304]{0} dynamic-update-slice(a, ua, idx)
      db = f32[4194304]{0} dynamic-update-slice(b, ub, idx)
      ROOT t = (f32[4194304]{0}, f32[4194304]{0}) tuple(da, db)
    }

    ENTRY main {
      p0 = f32[4,1048576]{1,0} parameter(0)
      p1 = f32[4194304]{0} parameter(1)
      u2d = f32[1,16]{1,0} parameter(2)
      u1d = f32[16]{0} parameter(3)
      i = s32[] parameter(4)
      f0 = f32[4194304]{0} fusion(p0, u2d, i), kind=kLoop, calls=bitcast_dus
      f1 = (f32[4194304]{0}, f32[4194304]{0}) fusion(f0, p1, u1d, u1d, i),
          kind=kLoop, calls=tuple_dus
      ROOT out = (f32[4194304]{0}, f32[4194304]{0}) copy(f1)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_dus,
                       ParseAndReturnVerifiedModule(in_place_dus_hlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed, RunParallelTaskAssigner(m_dus.get(), /*max_parallelism=*/32,
                                            &target_machine_features_));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest,
       ScatterPartitionsAlongNumUpdatesNotDestinationShape) {
  // Large DRAM tier f32[8388608] destination with num_updates = 5 and heavy
  // fused update compute partitions across updates and never exceeds 5 tasks.
  constexpr absl::string_view kHeavyFewUpdatesTemplate = R"hlo(
    HloModule heavy_few_updates

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    fused_scatter {
      dst = f32[8388608] parameter(0)
      idx = s32[$n,1] parameter(1)
      src = f32[$n,16384] parameter(2)
      a0 = f32[$n,16384] acos(src)
      a1 = f32[$n,16384] acos(a0)
      red = f32[$n] reduce(a1, f32[] constant(0)),
          dimensions={1}, to_apply=add_f32
      upd = f32[$n,1] reshape(red)
      ROOT s = f32[8388608] scatter(dst, idx, upd),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_f32
    }

    ENTRY main {
      dst = f32[8388608] parameter(0)
      idx = s32[$n,1] parameter(1)
      src = f32[$n,16384] parameter(2)
      ROOT f = f32[8388608] fusion(dst, idx, src),
          kind=kLoop, calls=fused_scatter
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(
      int64_t five_update_tasks,
      RunAndGetRootScatterTasks(
          absl::StrReplaceAll(kHeavyFewUpdatesTemplate, {{"$n", "5"}}),
          /*max_parallelism=*/32));
  EXPECT_GT(five_update_tasks, 1);
  EXPECT_LE(five_update_tasks, 5);

  // Tiny num_updates = 64 into a DRAM tier destination stays at 1 task.
  ASSERT_OK_AND_ASSIGN(
      int64_t tiny_tasks,
      RunAndGetRootScatterTasks(
          BuildSimplified1DScatterHlo("f32", /*num_bins=*/8388608,
                                      /*num_updates=*/64,
                                      "ROOT r = f32[] add(lhs, rhs)"),
          /*max_parallelism=*/32));
  EXPECT_EQ(tiny_tasks, 1);

  // Non divisible num_updates values partition cleanly within (1, num_updates]
  // without spawning extra tasks that leave the max chunk size unchanged.
  ASSERT_OK_AND_ASSIGN(
      int64_t thirteen_on_eight_tasks,
      RunAndGetRootScatterTasks(
          absl::StrReplaceAll(kHeavyFewUpdatesTemplate, {{"$n", "13"}}),
          /*max_parallelism=*/8));
  EXPECT_GT(thirteen_on_eight_tasks, 1);
  EXPECT_LE(thirteen_on_eight_tasks, 7);

  ASSERT_OK_AND_ASSIGN(
      int64_t prime_update_tasks,
      RunAndGetRootScatterTasks(
          BuildSimplified1DScatterHlo("f32", /*num_bins=*/8388608,
                                      /*num_updates=*/131071,
                                      "ROOT r = f32[] add(lhs, rhs)"),
          /*max_parallelism=*/32));
  EXPECT_GT(prime_update_tasks, 1);
  EXPECT_LE(prime_update_tasks, 32);
}

TEST_F(ParallelTaskAssignmentTest, ScatterContentionModel) {
  // Rank 0 destinations stay at 1 task when updates are plain values because
  // every update contends on one cache line.
  constexpr absl::string_view kRank0Template = R"hlo(
    HloModule rank0_scatter

    add_fn {
      lhs = $ty[] parameter(0)
      rhs = $ty[] parameter(1)
      ROOT sum = $ty[] add(lhs, rhs)
    }

    fused_scatter {
      dst = $ty[] parameter(0)
      idx = s32[262144,0] parameter(1)
      upd = $ty[262144] parameter(2)
      ROOT s = $ty[] scatter(dst, idx, upd),
          update_window_dims={},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={},
          index_vector_dim=1,
          to_apply=add_fn
    }

    ENTRY main {
      dst = $ty[] parameter(0)
      idx = s32[262144,0] parameter(1)
      upd = $ty[262144] parameter(2)
      ROOT f = $ty[] fusion(dst, idx, upd), kind=kLoop, calls=fused_scatter
    }
  )hlo";

  for (absl::string_view ty : {"f64", "s32"}) {
    SCOPED_TRACE(ty);
    ASSERT_OK_AND_ASSIGN(
        int64_t tasks, RunAndGetRootScatterTasks(
                           absl::StrReplaceAll(kRank0Template, {{"$ty", ty}})));
    EXPECT_EQ(tasks, 1);
  }

  // Narrow 1 through 1024 bin non unique histograms with N = 262144 stay
  // single threaded to avoid atomic cache line contention and RFO overhead.
  for (int64_t narrow_bins : {1, 8, 32, 256, 1024}) {
    for (absl::string_view ty : {"f64", "s32"}) {
      SCOPED_TRACE(absl::StrCat(ty, "[", narrow_bins, "]"));
      const std::string body =
          absl::StrCat("ROOT r = ", ty, "[] add(lhs, rhs)");
      ASSERT_OK_AND_ASSIGN(
          int64_t tasks, RunAndGetRootScatterTasks(BuildSimplified1DScatterHlo(
                             ty, narrow_bins, /*num_updates=*/262144, body)));
      EXPECT_EQ(tasks, 1);
    }
  }

  // Narrow s8 indices into a DRAM tier destination address at most 128 non
  // negative bins and stay single threaded despite the wide destination shape.
  ASSERT_OK_AND_ASSIGN(
      int64_t s8_idx_tasks,
      RunAndGetRootScatterTasks(absl::StrReplaceAll(
          BuildSimplified1DScatterHlo("s32", /*num_bins=*/8388608,
                                      /*num_updates=*/262144,
                                      "ROOT r = s32[] add(lhs, rhs)"),
          {{"indices = s32[262144,1]", "indices = s8[262144,1]"}})));
  EXPECT_EQ(s8_idx_tasks, 1);

  // Standalone cache resident non unique CAS histograms stay at 1 task because
  // pipelined non atomic RMW beats cross CCD cmpxchg.
  struct CacheCasCase {
    absl::string_view type;
    int64_t bins;
    int64_t updates;
  };
  for (const CacheCasCase& tc : {
           CacheCasCase{"f32", 65536, 262144},
           CacheCasCase{"f64", 49152, 262144},
           CacheCasCase{"f32", 1048576, 524288},
       }) {
    SCOPED_TRACE(absl::StrCat(tc.type, "[", tc.bins, "]"));
    ASSERT_OK_AND_ASSIGN(
        int64_t tasks,
        RunAndGetRootScatterTasks(
            BuildSimplified1DScatterHlo(
                tc.type, tc.bins, tc.updates,
                absl::StrCat("ROOT r = ", tc.type, "[] add(lhs, rhs)")),
            /*max_parallelism=*/32));
    EXPECT_EQ(tasks, 1);
  }

  // Far L3 s32 direct atomicrmw add stays near a single CCD on 32 cores and
  // scales further on 64 cores; 1 MiB L3 direct atomic overwrite caps near a
  // single CCD domain.
  const std::string far_l3_s32_add = BuildSimplified1DScatterHlo(
      "s32", /*num_bins=*/1048576, /*num_updates=*/524288,
      "ROOT r = s32[] add(lhs, rhs)");
  ASSERT_OK_AND_ASSIGN(
      int64_t s32_l3_32,
      RunAndGetRootScatterTasks(far_l3_s32_add, /*max_parallelism=*/32));
  ASSERT_OK_AND_ASSIGN(
      int64_t s32_l3_64,
      RunAndGetRootScatterTasks(far_l3_s32_add, /*max_parallelism=*/64));
  EXPECT_GE(s32_l3_32, 16);
  EXPECT_GT(s32_l3_64, s32_l3_32);

  // Near L3 (1 MiB) direct atomic overwrites parallelize within a single 8 core
  // cache domain, while s32/f32 stay single workgroup on 32 core multi domain
  // pools where cross domain RFO outweighs the short non atomic loop.
  CpuDeviceSpec ldst128_spec;
  ldst128_spec.has_atomic_load_store_128 = true;
  target_machine_features_.set_cpu_device_spec(ldst128_spec);
  for (const auto& [ty, bins] :
       std::initializer_list<std::pair<absl::string_view, int64_t>>{
           {"s32", 262144}, {"f32", 262144}, {"c128", 65536}}) {
    SCOPED_TRACE(absl::StrCat("l3_overwrite_", ty));
    const std::string hlo = BuildSimplified1DScatterHlo(
        ty, bins, /*num_updates=*/262144,
        absl::StrCat("ROOT r = ", ty, "[] parameter(1)"));
    ASSERT_OK_AND_ASSIGN(int64_t tasks_8,
                         RunAndGetRootScatterTasks(hlo, /*max_parallelism=*/8));
    EXPECT_EQ(tasks_8, 8);
    if (ty != "c128") {
      ASSERT_OK_AND_ASSIGN(int64_t tasks_32, RunAndGetRootScatterTasks(
                                                 hlo, /*max_parallelism=*/32));
      EXPECT_EQ(tasks_32, 1);
    }
  }
  ASSERT_OK_AND_ASSIGN(
      int64_t c128_512k_overwrite_32,
      RunAndGetRootScatterTasks(
          BuildSimplified1DScatterHlo("c128", /*num_bins=*/65536,
                                      /*num_updates=*/524288,
                                      "ROOT r = c128[] parameter(1)"),
          /*max_parallelism=*/32));
  EXPECT_GE(c128_512k_overwrite_32, 16);
  target_machine_features_.set_cpu_device_spec(CpuDeviceSpec{});

  // DRAM resident (> 16 MiB) histograms scale to high task counts across
  // integer atomicrmw, CAS floating point and complex types, pred, and direct
  // atomic overwrite reducers.
  struct DramCase {
    absl::string_view type;
    int64_t num_bins;
    absl::string_view reducer_body;
  };
  const DramCase kDramCases[] = {
      {"s32", 8388608, "ROOT r = s32[] add(lhs, rhs)"},
      {"f32", 8388608, "ROOT r = f32[] add(lhs, rhs)"},
      {"f64", 4194304, "ROOT r = f64[] add(lhs, rhs)"},
      {"bf16", 16777216, "ROOT r = bf16[] add(lhs, rhs)"},
      {"c64", 4194304, "ROOT r = c64[] add(lhs, rhs)"},
      {"pred", 33554432, "ROOT r = pred[] or(lhs, rhs)"},
      {"f32", 8388608, "ROOT r = f32[] parameter(1)"},
      {"s32", 8388608, "ROOT r = s32[] parameter(1)"},
  };
  for (const DramCase& tc : kDramCases) {
    SCOPED_TRACE(
        absl::StrCat(tc.type, "[", tc.num_bins, "] ", tc.reducer_body));
    ASSERT_OK_AND_ASSIGN(
        int64_t tasks,
        RunAndGetRootScatterTasks(BuildSimplified1DScatterHlo(
                                      tc.type, tc.num_bins,
                                      /*num_updates=*/262144, tc.reducer_body),
                                  /*max_parallelism=*/32));
    EXPECT_GE(tasks, 16);
  }

  // Rank 0 destination with heavy fused transcendental compute outweighs
  // single line atomic contention and partitions along num_updates despite the
  // rank 0 output shape.
  constexpr absl::string_view kRank0HeavyComputeHlo = R"hlo(
    HloModule rank0_heavy_scatter

    add_f64 {
      lhs = f64[] parameter(0)
      rhs = f64[] parameter(1)
      ROOT sum = f64[] add(lhs, rhs)
    }

    fused_scatter {
      dst = f64[] parameter(0)
      idx = s32[65536,0] parameter(1)
      x = f64[65536,16] parameter(2)
      a0 = f64[65536,16] acos(x)
      a1 = f64[65536,16] acos(a0)
      upd = f64[65536] reduce(a1, f64[] constant(0)),
          dimensions={1}, to_apply=add_f64
      ROOT s = f64[] scatter(dst, idx, upd),
          update_window_dims={},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={},
          index_vector_dim=1,
          to_apply=add_f64
    }

    ENTRY main {
      dst = f64[] parameter(0)
      idx = s32[65536,0] parameter(1)
      x = f64[65536,16] parameter(2)
      ROOT f = f64[] fusion(dst, idx, x), kind=kLoop, calls=fused_scatter
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(int64_t rank0_heavy_tasks,
                       RunAndGetRootScatterTasks(kRank0HeavyComputeHlo,
                                                 /*max_parallelism=*/32));
  EXPECT_GE(rank0_heavy_tasks, 8);
}

TEST_F(ParallelTaskAssignmentTest,
       ScatterUniqueIndicesWindowSlicesAndFusedProducers) {
  // Unique indices parallelize cleanly at moderate N = 65536 on 8 cores and
  // assign at least as many tasks as non unique CAS updates.
  ASSERT_OK_AND_ASSIGN(
      int64_t unique_64k,
      RunAndGetRootScatterTasks(
          BuildSimplified1DScatterHlo("f32", /*num_bins=*/65536,
                                      /*num_updates=*/65536,
                                      "ROOT r = f32[] add(lhs, rhs)",
                                      /*unique_indices=*/true),
          /*max_parallelism=*/8));
  ASSERT_OK_AND_ASSIGN(int64_t cas_64k, RunAndGetRootScatterTasks(
                                            BuildSimplified1DScatterHlo(
                                                "f32", /*num_bins=*/65536,
                                                /*num_updates=*/65536,
                                                "ROOT r = f32[] add(lhs, rhs)",
                                                /*unique_indices=*/false),
                                            /*max_parallelism=*/8));
  EXPECT_GT(unique_64k, 1);
  EXPECT_EQ(cas_64k, 1);

  // Small unique_indices=true scatter with N = 32768 stays below the 85 us
  // minimum single thread threshold, while L3 resident N = 262144 caps near a
  // single CCD domain and DRAM resident f32[8388608] scales further.
  ASSERT_OK_AND_ASSIGN(
      int64_t unique_32k,
      RunAndGetRootScatterTasks(
          BuildSimplified1DScatterHlo("f32", /*num_bins=*/65536,
                                      /*num_updates=*/32768,
                                      "ROOT r = f32[] add(lhs, rhs)",
                                      /*unique_indices=*/true),
          /*max_parallelism=*/32));
  EXPECT_EQ(unique_32k, 1);

  ASSERT_OK_AND_ASSIGN(
      int64_t unique_256k_l3,
      RunAndGetRootScatterTasks(
          BuildSimplified1DScatterHlo("f32", /*num_bins=*/262144,
                                      /*num_updates=*/262144,
                                      "ROOT r = f32[] add(lhs, rhs)",
                                      /*unique_indices=*/true),
          /*max_parallelism=*/32));
  EXPECT_GE(unique_256k_l3, 16);

  const std::string dram_unique_hlo = BuildSimplified1DScatterHlo(
      "f32", /*num_bins=*/8388608, /*num_updates=*/524288,
      "ROOT r = f32[] add(lhs, rhs)", /*unique_indices=*/true);
  ASSERT_OK_AND_ASSIGN(
      int64_t unique_512k_dram_32,
      RunAndGetRootScatterTasks(dram_unique_hlo, /*max_parallelism=*/32));
  ASSERT_OK_AND_ASSIGN(
      int64_t unique_512k_dram_64,
      RunAndGetRootScatterTasks(dram_unique_hlo, /*max_parallelism=*/64));
  EXPECT_GE(unique_512k_dram_32, 16);
  EXPECT_GT(unique_512k_dram_64, unique_512k_dram_32);

  // Window slice scatter into f32[512,128] with contiguous [1,128] row slices.
  constexpr absl::string_view kUniqueWindowSliceHlo = R"hlo(
    HloModule window_slice_unique

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    fused_scatter {
      dst = f32[512,128] parameter(0)
      idx = s32[4096,1] parameter(1)
      upd = f32[4096,1,128] parameter(2)
      ROOT s = f32[512,128] scatter(dst, idx, upd),
          update_window_dims={1,2},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          unique_indices=true,
          to_apply=add_f32
    }

    ENTRY main {
      dst = f32[512,128] parameter(0)
      idx = s32[4096,1] parameter(1)
      upd = f32[4096,1,128] parameter(2)
      ROOT f = f32[512,128] fusion(dst, idx, upd),
          kind=kLoop, calls=fused_scatter
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(
      int64_t unique_slice_tasks,
      RunAndGetRootScatterTasks(kUniqueWindowSliceHlo, /*max_parallelism=*/16));
  EXPECT_GE(unique_slice_tasks, 4);

  constexpr absl::string_view kOverwriteWindowSliceHlo = R"hlo(
    HloModule window_slice_overwrite

    overwrite_f32 {
      lhs = f32[] parameter(0)
      ROOT rhs = f32[] parameter(1)
    }

    fused_scatter {
      dst = f32[2048,128] parameter(0)
      idx = s32[8192,1] parameter(1)
      upd = f32[8192,1,128] parameter(2)
      ROOT s = f32[2048,128] scatter(dst, idx, upd),
          update_window_dims={1,2},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=overwrite_f32
    }

    ENTRY main {
      dst = f32[2048,128] parameter(0)
      idx = s32[8192,1] parameter(1)
      upd = f32[8192,1,128] parameter(2)
      ROOT f = f32[2048,128] fusion(dst, idx, upd),
          kind=kLoop, calls=fused_scatter
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(int64_t overwrite_slice_tasks,
                       RunAndGetRootScatterTasks(kOverwriteWindowSliceHlo,
                                                 /*max_parallelism=*/16));
  EXPECT_GE(overwrite_slice_tasks, 4);

  // Fused update and index producers drive parallelization even when the raw
  // scatter write count alone falls below the dispatch threshold.
  ASSERT_OK_AND_ASSIGN(
      int64_t plain_8k_tasks,
      RunAndGetRootScatterTasks(
          BuildSimplified1DScatterHlo("f32", /*num_bins=*/16384,
                                      /*num_updates=*/8192,
                                      "ROOT r = f32[] add(lhs, rhs)",
                                      /*unique_indices=*/true),
          /*max_parallelism=*/16));
  EXPECT_EQ(plain_8k_tasks, 1);

  constexpr absl::string_view kFusedUpdateProducerHlo = R"hlo(
    HloModule fused_update_producer

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    fused_scatter {
      dst = f32[16384] parameter(0)
      idx = s32[8192,1] parameter(1)
      raw_upd = f64[8192,1] parameter(2)
      upd = f32[8192,1] convert(f64[8192,1] acos(raw_upd))
      ROOT s = f32[16384] scatter(dst, idx, upd),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          unique_indices=true,
          to_apply=add_f32
    }

    ENTRY main {
      dst = f32[16384] parameter(0)
      idx = s32[8192,1] parameter(1)
      raw_upd = f64[8192,1] parameter(2)
      ROOT f = f32[16384] fusion(dst, idx, raw_upd),
          kind=kLoop, calls=fused_scatter
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(int64_t fused_upd_tasks,
                       RunAndGetRootScatterTasks(kFusedUpdateProducerHlo,
                                                 /*max_parallelism=*/16));
  EXPECT_GT(fused_upd_tasks, plain_8k_tasks);

  constexpr absl::string_view kFusedIndexProducerHlo = R"hlo(
    HloModule fused_index_producer

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    fused_scatter {
      dst = f32[16384] parameter(0)
      raw_idx = f64[8192,1] parameter(1)
      upd = f32[8192,1] parameter(2)
      idx = s32[8192,1] convert(f64[8192,1] acos(raw_idx))
      ROOT s = f32[16384] scatter(dst, idx, upd),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          unique_indices=true,
          to_apply=add_f32
    }

    ENTRY main {
      dst = f32[16384] parameter(0)
      raw_idx = f64[8192,1] parameter(1)
      upd = f32[8192,1] parameter(2)
      ROOT f = f32[16384] fusion(dst, raw_idx, upd),
          kind=kLoop, calls=fused_scatter
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(int64_t fused_idx_tasks,
                       RunAndGetRootScatterTasks(kFusedIndexProducerHlo,
                                                 /*max_parallelism=*/16));
  EXPECT_GT(fused_idx_tasks, plain_8k_tasks);

  // Heavy fused acos compute into an L2 resident non unique f64[49152] CAS
  // histogram scales across 8, 16, 32, and 64 cores.
  constexpr absl::string_view kFusedAcosL2HistHlo = R"hlo(
    HloModule fused_acos_l2_hist

    add_f64 {
      lhs = f64[] parameter(0)
      rhs = f64[] parameter(1)
      ROOT sum = f64[] add(lhs, rhs)
    }

    fused_scatter {
      dst = f64[49152] parameter(0)
      raw_idx = f64[262144,1] parameter(1)
      upd = f64[262144,1] parameter(2)
      idx = s32[262144,1] convert(f64[262144,1] acos(raw_idx))
      ROOT s = f64[49152] scatter(dst, idx, upd),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_f64
    }

    ENTRY main {
      dst = f64[49152] parameter(0)
      raw_idx = f64[262144,1] parameter(1)
      upd = f64[262144,1] parameter(2)
      ROOT f = f64[49152] fusion(dst, raw_idx, upd),
          kind=kLoop, calls=fused_scatter
    }
  )hlo";
  int64_t prev_tasks = 1;
  for (int64_t max_cores : {8, 16, 32, 64}) {
    SCOPED_TRACE(absl::StrCat("max_cores=", max_cores));
    ASSERT_OK_AND_ASSIGN(int64_t tasks, RunAndGetRootScatterTasks(
                                            kFusedAcosL2HistHlo, max_cores));
    EXPECT_GT(tasks, prev_tasks);
    prev_tasks = tasks;
  }
}

TEST_F(ParallelTaskAssignmentTest, ScatterVariadicAndPackedSubByte) {
  constexpr absl::string_view kVariadicTemplate = R"hlo(
    HloModule variadic_scatter

    add_pair {
      a0 = f32[] parameter(0)
      b0 = s32[] parameter(1)
      a1 = f32[] parameter(2)
      b1 = s32[] parameter(3)
      ROOT out = (f32[], s32[]) tuple(f32[] add(a0, a1), s32[] add(b0, b1))
    }

    fused_scatter {
      d0 = f32[8388608] parameter(0)
      d1 = s32[8388608] parameter(1)
      idx = s32[262144,1] parameter(2)
      upd0 = f32[262144,1] parameter(3)
      upd1 = s32[262144,1] parameter(4)
      ROOT s = (f32[8388608], s32[8388608]) scatter(d0, d1, idx, upd0, upd1),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          $uniqueto_apply=add_pair
    }

    ENTRY main {
      d0 = f32[8388608] parameter(0)
      d1 = s32[8388608] parameter(1)
      idx = s32[262144,1] parameter(2)
      upd0 = f32[262144,1] parameter(3)
      upd1 = s32[262144,1] parameter(4)
      ROOT f = (f32[8388608], s32[8388608]) fusion(d0, d1, idx, upd0, upd1),
          kind=kLoop, calls=fused_scatter
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(
      int64_t variadic_unique_tasks,
      RunAndGetRootScatterTasks(absl::StrReplaceAll(
          kVariadicTemplate, {{"$unique", "unique_indices=true, "}})));
  EXPECT_GT(variadic_unique_tasks, 1);

  ASSERT_OK_AND_ASSIGN(int64_t variadic_non_unique_tasks,
                       RunAndGetRootScatterTasks(absl::StrReplaceAll(
                           kVariadicTemplate, {{"$unique", ""}})));
  EXPECT_EQ(variadic_non_unique_tasks, 1);

  // Variadic scatter with packed sub byte s4 stays at 1 even with
  // unique_indices=true and DRAM tier buffers because distinct elements share
  // storage bytes.
  constexpr absl::string_view kVariadicS4Hlo = R"hlo(
    HloModule variadic_s4_scatter

    add_pair {
      a0 = s4[] parameter(0)
      b0 = s32[] parameter(1)
      a1 = s4[] parameter(2)
      b1 = s32[] parameter(3)
      ROOT out = (s4[], s32[]) tuple(s4[] add(a0, a1), s32[] add(b0, b1))
    }

    fused_scatter {
      d0 = s4[67108864]{0:E(4)} parameter(0)
      d1 = s32[8388608]{0} parameter(1)
      idx = s32[262144,1]{1,0} parameter(2)
      upd0 = s4[262144,1]{1,0:E(4)} parameter(3)
      upd1 = s32[262144,1]{1,0} parameter(4)
      ROOT s = (s4[67108864]{0:E(4)}, s32[8388608]{0})
          scatter(d0, d1, idx, upd0, upd1),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          unique_indices=true,
          to_apply=add_pair
    }

    ENTRY main {
      d0 = s4[67108864]{0:E(4)} parameter(0)
      d1 = s32[8388608]{0} parameter(1)
      idx = s32[262144,1]{1,0} parameter(2)
      upd0 = s4[262144,1]{1,0:E(4)} parameter(3)
      upd1 = s32[262144,1]{1,0} parameter(4)
      ROOT f = (s4[67108864]{0:E(4)}, s32[8388608]{0})
          fusion(d0, d1, idx, upd0, upd1), kind=kLoop, calls=fused_scatter
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(int64_t variadic_s4_tasks,
                       RunAndGetRootScatterTasks(kVariadicS4Hlo));
  EXPECT_EQ(variadic_s4_tasks, 1);

  // Single operand packed s4 parallelizes via byte wide CAS on a DRAM tier
  // buffer and stays at 1 on L2 resident and tiny contended buffers.
  ASSERT_OK_AND_ASSIGN(
      int64_t dram_s4_tasks,
      RunAndGetRootScatterTasks(BuildSimplified1DScatterHlo(
          "s4", /*num_bins=*/67108864, /*num_updates=*/262144,
          "ROOT r = s4[] add(lhs, rhs)", /*unique_indices=*/false, ":E(4)")));
  EXPECT_GT(dram_s4_tasks, 1);

  ASSERT_OK_AND_ASSIGN(
      int64_t l2_s4_tasks,
      RunAndGetRootScatterTasks(BuildSimplified1DScatterHlo(
          "s4", /*num_bins=*/1048576, /*num_updates=*/262144,
          "ROOT r = s4[] add(lhs, rhs)", /*unique_indices=*/false, ":E(4)")));
  EXPECT_EQ(l2_s4_tasks, 1);

  ASSERT_OK_AND_ASSIGN(
      int64_t tiny_s4_tasks,
      RunAndGetRootScatterTasks(BuildSimplified1DScatterHlo(
          "s4", /*num_bins=*/32, /*num_updates=*/262144,
          "ROOT r = s4[] add(lhs, rhs)", /*unique_indices=*/false, ":E(4)")));
  EXPECT_EQ(tiny_s4_tasks, 1);
}

TEST_F(ParallelTaskAssignmentTest, ScatterCpuDeviceSpecAtomicCapabilityGating) {
  auto make_spec = [](bool cmpxchg_128, bool load_store_128, bool rmw_64,
                      bool load_store_64) {
    CpuDeviceSpec spec;
    spec.has_atomic_cmpxchg_128 = cmpxchg_128;
    spec.has_atomic_load_store_128 = load_store_128;
    spec.has_atomic_rmw_64 = rmw_64;
    spec.has_atomic_load_store_64 = load_store_64;
    return spec;
  };

  const std::string c128_add = BuildSimplified1DScatterHlo(
      "c128", /*num_bins=*/2097152, /*num_updates=*/262144,
      "ROOT r = c128[] add(lhs, rhs)", /*unique_indices=*/false);
  const std::string c128_overwrite = BuildSimplified1DScatterHlo(
      "c128", /*num_bins=*/2097152, /*num_updates=*/262144,
      "ROOT r = c128[] parameter(1)", /*unique_indices=*/false);
  const std::string c128_unique = BuildSimplified1DScatterHlo(
      "c128", /*num_bins=*/2097152, /*num_updates=*/262144,
      "ROOT r = c128[] add(lhs, rhs)", /*unique_indices=*/true);

  // Both 128 bit flags false: non unique c128 add and overwrite stay at 1,
  // while unique_indices=true uses plain loads and stores and parallelizes.
  target_machine_features_.set_cpu_device_spec(
      make_spec(/*cmpxchg_128=*/false, /*load_store_128=*/false,
                /*rmw_64=*/true, /*load_store_64=*/true));
  ASSERT_OK_AND_ASSIGN(int64_t tasks, RunAndGetRootScatterTasks(c128_add));
  EXPECT_EQ(tasks, 1);
  ASSERT_OK_AND_ASSIGN(tasks, RunAndGetRootScatterTasks(c128_overwrite));
  EXPECT_EQ(tasks, 1);
  ASSERT_OK_AND_ASSIGN(tasks, RunAndGetRootScatterTasks(c128_unique));
  EXPECT_GT(tasks, 1);

  // Only has_atomic_load_store_128=true: c128 add still stays at 1, while c128
  // overwrite parallelizes via direct 128 bit atomic store.
  target_machine_features_.set_cpu_device_spec(
      make_spec(/*cmpxchg_128=*/false, /*load_store_128=*/true,
                /*rmw_64=*/true, /*load_store_64=*/true));
  ASSERT_OK_AND_ASSIGN(tasks, RunAndGetRootScatterTasks(c128_add));
  EXPECT_EQ(tasks, 1);
  ASSERT_OK_AND_ASSIGN(tasks, RunAndGetRootScatterTasks(c128_overwrite));
  EXPECT_GT(tasks, 1);

  // Only has_atomic_cmpxchg_128=true: both c128 add and c128 overwrite
  // parallelize.
  target_machine_features_.set_cpu_device_spec(
      make_spec(/*cmpxchg_128=*/true, /*load_store_128=*/false,
                /*rmw_64=*/true, /*load_store_64=*/true));
  ASSERT_OK_AND_ASSIGN(tasks, RunAndGetRootScatterTasks(c128_add));
  EXPECT_GT(tasks, 1);
  ASSERT_OK_AND_ASSIGN(tasks, RunAndGetRootScatterTasks(c128_overwrite));
  EXPECT_GT(tasks, 1);

  // AArch64 outline atomics without LSE (has_atomic_rmw_64=false,
  // has_atomic_load_store_64=true): integer/float RMW and all packed s4
  // scatters (including overwrite and unique_indices=true) stay at 1, while
  // unique_indices=true f32 and non unique f32 overwrite still parallelize.
  target_machine_features_.set_cpu_device_spec(
      make_spec(/*cmpxchg_128=*/false, /*load_store_128=*/false,
                /*rmw_64=*/false, /*load_store_64=*/true));
  ASSERT_OK_AND_ASSIGN(tasks,
                       RunAndGetRootScatterTasks(BuildSimplified1DScatterHlo(
                           "s32", /*num_bins=*/8388608, /*num_updates=*/262144,
                           "ROOT r = s32[] add(lhs, rhs)")));
  EXPECT_EQ(tasks, 1);
  ASSERT_OK_AND_ASSIGN(tasks,
                       RunAndGetRootScatterTasks(BuildSimplified1DScatterHlo(
                           "f32", /*num_bins=*/8388608, /*num_updates=*/262144,
                           "ROOT r = f32[] add(lhs, rhs)")));
  EXPECT_EQ(tasks, 1);
  for (bool unique : {false, true}) {
    for (absl::string_view body :
         {"ROOT r = s4[] add(lhs, rhs)", "ROOT r = s4[] parameter(1)"}) {
      SCOPED_TRACE(absl::StrCat("s4 unique=", unique, " ", body));
      ASSERT_OK_AND_ASSIGN(
          tasks, RunAndGetRootScatterTasks(BuildSimplified1DScatterHlo(
                     "s4", /*num_bins=*/67108864, /*num_updates=*/262144, body,
                     unique, ":E(4)")));
      EXPECT_EQ(tasks, 1);
    }
  }
  ASSERT_OK_AND_ASSIGN(
      tasks, RunAndGetRootScatterTasks(BuildSimplified1DScatterHlo(
                 "f32", /*num_bins=*/262144, /*num_updates=*/262144,
                 "ROOT r = f32[] add(lhs, rhs)", /*unique_indices=*/true)));
  EXPECT_GT(tasks, 1);
  ASSERT_OK_AND_ASSIGN(
      tasks, RunAndGetRootScatterTasks(BuildSimplified1DScatterHlo(
                 "f32", /*num_bins=*/8388608, /*num_updates=*/262144,
                 "ROOT r = f32[] parameter(1)", /*unique_indices=*/false)));
  EXPECT_GT(tasks, 1);
}

TEST_F(ParallelTaskAssignmentTest, ScatterBufferAliasingFallback) {
  // Fused update producer reading from the scatter operand parameter.
  constexpr absl::string_view kSharedUpdateParamHlo = R"hlo(
    HloModule shared_update_param

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    fused_scatter {
      dst = f32[8388608] parameter(0)
      idx = s32[8388608,1] parameter(1)
      doubled = f32[8388608] add(dst, dst)
      upd = f32[8388608,1] reshape(doubled)
      ROOT s = f32[8388608] scatter(dst, idx, upd),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_f32
    }

    ENTRY main {
      dst = f32[8388608] parameter(0)
      idx = s32[8388608,1] parameter(1)
      ROOT f = f32[8388608] fusion(dst, idx), kind=kLoop, calls=fused_scatter
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(int64_t tasks,
                       RunAndGetRootScatterTasks(kSharedUpdateParamHlo));
  EXPECT_EQ(tasks, 1);

  // Fused index producer reading from the scatter operand parameter.
  constexpr absl::string_view kSharedIndexParamHlo = R"hlo(
    HloModule shared_index_param

    add_s32 {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }

    fused_scatter {
      dst = s32[8388608] parameter(0)
      upd = s32[8388608,1] parameter(1)
      idx = s32[8388608,1] reshape(dst)
      ROOT s = s32[8388608] scatter(dst, idx, upd),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_s32
    }

    ENTRY main {
      dst = s32[8388608] parameter(0)
      upd = s32[8388608,1] parameter(1)
      ROOT f = s32[8388608] fusion(dst, upd), kind=kLoop, calls=fused_scatter
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(tasks, RunAndGetRootScatterTasks(kSharedIndexParamHlo));
  EXPECT_EQ(tasks, 1);

  // Separate fusion parameters fed by the same outer operand directly.
  constexpr absl::string_view kSameOuterDirectHlo = R"hlo(
    HloModule same_outer_direct

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    fused_scatter {
      dst = f32[8388608] parameter(0)
      idx = s32[8388608,1] parameter(1)
      src = f32[8388608] parameter(2)
      upd = f32[8388608,1] reshape(src)
      ROOT s = f32[8388608] scatter(dst, idx, upd),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_f32
    }

    ENTRY main {
      x = f32[8388608] parameter(0)
      idx = s32[8388608,1] parameter(1)
      ROOT f = f32[8388608] fusion(x, idx, x), kind=kLoop, calls=fused_scatter
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(tasks, RunAndGetRootScatterTasks(kSameOuterDirectHlo));
  EXPECT_EQ(tasks, 1);

  // Separate fusion parameters fed by the same outer operand through bitcast.
  constexpr absl::string_view kSameOuterBitcastHlo = R"hlo(
    HloModule same_outer_bitcast

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    fused_scatter {
      dst = f32[8388608]{0} parameter(0)
      idx = s32[8388608,1]{1,0} parameter(1)
      upd = f32[8388608,1]{1,0} parameter(2)
      ROOT s = f32[8388608]{0} scatter(dst, idx, upd),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_f32
    }

    ENTRY main {
      x = f32[8388608]{0} parameter(0)
      idx = s32[8388608,1]{1,0} parameter(1)
      upd_bc = f32[8388608,1]{1,0} bitcast(x)
      ROOT f = f32[8388608]{0} fusion(x, idx, upd_bc),
          kind=kLoop, calls=fused_scatter
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(tasks, RunAndGetRootScatterTasks(kSameOuterBitcastHlo));
  EXPECT_EQ(tasks, 1);

  // Matching get-tuple-element index stays at 1, while distinct indices
  // parallelize.
  constexpr absl::string_view kTupleOuterTemplate = R"hlo(
    HloModule tuple_outer_scatter

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    fused_scatter {
      dst = f32[8388608]{0} parameter(0)
      idx = s32[8388608,1]{1,0} parameter(1)
      upd = f32[8388608,1]{1,0} parameter(2)
      ROOT s = f32[8388608]{0} scatter(dst, idx, upd),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_f32
    }

    ENTRY main {
      t = (f32[8388608]{0}, f32[8388608]{0}, s32[8388608,1]{1,0}) parameter(0)
      dst = f32[8388608]{0} get-tuple-element(t), index=0
      src = f32[8388608]{0} get-tuple-element(t), index=$src_idx
      upd = f32[8388608,1]{1,0} bitcast(src)
      idx = s32[8388608,1]{1,0} get-tuple-element(t), index=2
      ROOT f = f32[8388608]{0} fusion(dst, idx, upd),
          kind=kLoop, calls=fused_scatter
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(tasks, RunAndGetRootScatterTasks(absl::StrReplaceAll(
                                  kTupleOuterTemplate, {{"$src_idx", "0"}})));
  EXPECT_EQ(tasks, 1);
  ASSERT_OK_AND_ASSIGN(tasks, RunAndGetRootScatterTasks(absl::StrReplaceAll(
                                  kTupleOuterTemplate, {{"$src_idx", "1"}})));
  EXPECT_GT(tasks, 1);

  // Forwarding through kTuple, kGetTupleElement, and kOptimizationBarrier is
  // peeled so aliasing is still detected before buffer assignment.
  constexpr absl::string_view kTupleForwardAliasHlo = R"hlo(
    HloModule tuple_forward_alias

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    fused_scatter {
      dst = f32[8388608]{0} parameter(0)
      idx = s32[8388608,1]{1,0} parameter(1)
      upd = f32[8388608,1]{1,0} parameter(2)
      ROOT s = f32[8388608]{0} scatter(dst, idx, upd),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_f32
    }

    ENTRY main {
      dst = f32[8388608]{0} parameter(0)
      other = f32[8388608]{0} parameter(1)
      idx = s32[8388608,1]{1,0} parameter(2)
      guarded = f32[8388608]{0} opt-barrier(dst)
      packed = (f32[8388608]{0}, f32[8388608]{0}) tuple(guarded, other)
      unpacked = f32[8388608]{0} get-tuple-element(packed), index=0
      upd = f32[8388608,1]{1,0} bitcast(unpacked)
      ROOT f = f32[8388608]{0} fusion(dst, idx, upd),
          kind=kLoop, calls=fused_scatter
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(tasks, RunAndGetRootScatterTasks(kTupleForwardAliasHlo));
  EXPECT_EQ(tasks, 1);

  // Explicit output_to_operand_aliasing pointing at a producer parameter.
  constexpr absl::string_view kExplicitProducerAliasHlo = R"hlo(
    HloModule explicit_producer_alias

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    fused_scatter {
      dst = f32[8388608] parameter(0)
      idx = s32[8388608,1] parameter(1)
      src = f32[8388608] parameter(2)
      upd = f32[8388608,1] reshape(src)
      ROOT s = f32[8388608] scatter(dst, idx, upd),
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_f32
    }

    ENTRY main {
      dst = f32[8388608] parameter(0)
      idx = s32[8388608,1] parameter(1)
      src = f32[8388608] parameter(2)
      ROOT f = f32[8388608] fusion(dst, idx, src),
          kind=kLoop, calls=fused_scatter,
          output_to_operand_aliasing={{}: (2, {})}
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(tasks,
                       RunAndGetRootScatterTasks(kExplicitProducerAliasHlo));
  EXPECT_EQ(tasks, 1);
}

TEST_F(ParallelTaskAssignmentTest, ScatterDeterminismFallback) {
  // Enable 128 bit cmpxchg so c128 add is gated only by determinism settings.
  CpuDeviceSpec spec;
  spec.has_atomic_cmpxchg_128 = true;
  spec.has_atomic_load_store_128 = true;
  spec.has_atomic_rmw_64 = true;
  spec.has_atomic_load_store_64 = true;
  target_machine_features_.set_cpu_device_spec(spec);

  struct DeterminismCase {
    absl::string_view label;
    absl::string_view type;
    int64_t num_bins;
    absl::string_view reducer_body;
    bool unique_indices;
    bool expect_parallel_when_deterministic;
  };
  const DeterminismCase kCases[] = {
      {"f32_add", "f32", 8388608, "ROOT r = f32[] add(lhs, rhs)", false, false},
      {"f64_add", "f64", 4194304, "ROOT r = f64[] add(lhs, rhs)", false, false},
      {"c128_add", "c128", 2097152, "ROOT r = c128[] add(lhs, rhs)", false,
       false},
      {"f32_min", "f32", 8388608, "ROOT r = f32[] minimum(lhs, rhs)", false,
       false},
      {"s32_overwrite", "s32", 8388608, "ROOT r = s32[] parameter(1)", false,
       false},
      {"s32_subtract", "s32", 8388608, "ROOT r = s32[] subtract(lhs, rhs)",
       false, false},
      {"s32_multi_op", "s32", 8388608,
       "sum = s32[] add(lhs, rhs)\n        one = s32[] constant(1)\n        "
       "ROOT r = s32[] add(sum, one)",
       false, false},
      {"s32_add", "s32", 8388608, "ROOT r = s32[] add(lhs, rhs)", false, true},
      {"s32_mul", "s32", 8388608, "ROOT r = s32[] multiply(lhs, rhs)", false,
       true},
      {"s32_min", "s32", 8388608, "ROOT r = s32[] minimum(lhs, rhs)", false,
       true},
      {"s32_max", "s32", 8388608, "ROOT r = s32[] maximum(lhs, rhs)", false,
       true},
      {"s32_xor", "s32", 8388608, "ROOT r = s32[] xor(lhs, rhs)", false, true},
      {"pred_or", "pred", 33554432, "ROOT r = pred[] or(lhs, rhs)", false,
       true},
      {"f32_unique_add", "f32", 262144, "ROOT r = f32[] add(lhs, rhs)", true,
       true},
  };

  struct FlagMode {
    absl::string_view name;
    std::function<void(DebugOptions&)> apply;
    bool deterministic;
  };
  const FlagMode kFlagModes[] = {
      {"default", [](DebugOptions&) {}, false},
      {"exclude_nondeterministic_ops",
       [](DebugOptions& opts) {
         opts.set_xla_gpu_exclude_nondeterministic_ops(true);
       },
       true},
      {"deterministic_ops",
       [](DebugOptions& opts) { opts.set_xla_gpu_deterministic_ops(true); },
       true},
  };

  for (const FlagMode& mode : kFlagModes) {
    for (const DeterminismCase& tc : kCases) {
      SCOPED_TRACE(absl::StrCat(mode.name, "/", tc.label));
      const std::string hlo = BuildSimplified1DScatterHlo(
          tc.type, tc.num_bins,
          /*num_updates=*/262144, tc.reducer_body, tc.unique_indices);
      ASSERT_OK_AND_ASSIGN(
          int64_t tasks,
          RunAndGetRootScatterTasks(hlo, /*max_parallelism=*/32, mode.apply));
      if (!mode.deterministic || tc.expect_parallel_when_deterministic) {
        EXPECT_GT(tasks, 1);
      } else {
        EXPECT_EQ(tasks, 1);
      }
    }
  }
}

}  // namespace
}  // namespace xla
