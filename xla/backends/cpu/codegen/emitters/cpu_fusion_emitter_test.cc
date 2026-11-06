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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "xla/backends/cpu/codegen/emitters/cpu_scatter_emitter.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/backends/cpu/codegen/fusion_emitter.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/logical_buffer.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace cpu {
namespace {

using ::testing::ElementsAreArray;

struct GetWorkDimensionsTestCase {
  std::string test_name;
  absl::string_view shape_and_layout;
  int64_t outer_dimension_partitions;
  uint64_t expected_num_work_groups;
  std::vector<int64_t> expected_work_tile_size;
  absl::string_view root_op = "copy(p0)";
};

class CpuFusionEmitterTest
    : public HloHardwareIndependentTestBase,
      public ::testing::WithParamInterface<GetWorkDimensionsTestCase> {
 protected:
  absl::StatusOr<std::unique_ptr<BufferAssignment>> RunBufferAssignment(
      const HloModule& hlo) {
    return BufferAssigner::Run(
        &hlo, std::make_unique<DependencyHloOrdering>(&hlo),
        [](const BufferValue& buffer) {
          return CpuExecutable::ShapeSizeBytes(buffer.shape());
        },
        &alias_info_, [](LogicalBuffer::Color) { return /*alignment=*/1; },
        BufferAssigner::Options{});
  }

  AliasInfo alias_info_;
};

static constexpr absl::string_view kScatterHlo = R"(
  add {
    %lhs = f32[] parameter(0)
    %rhs = f32[] parameter(1)
    ROOT %add.2 = f32[] add(%lhs, %rhs)
  }

  scatter_computation {
    %operand = f32[50,64,8] parameter(0)
    %indices = s32[500,1]{1,0} parameter(1)
    %updates = f32[500,1,64,8] parameter(2)
    ROOT %scatter = f32[50,64,8] scatter(%operand, %indices, %updates),
      update_window_dims={1,2,3},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1,
      to_apply=add
  }

  ENTRY main {
    %p = f32[50,64,8]{2,1,0} parameter(0)
    %p.1 = s32[500,1]{1,0} parameter(1)
    %p.2 = f32[500,1,64,8]{3,2,1,0} parameter(2)
    ROOT %wrapped_scatter = f32[50,64,8]{2,1,0} fusion(%p, %p.1, %p.2),
      kind=kLoop,
      calls=%scatter_computation
  }
)";

TEST_F(CpuFusionEmitterTest, ScatterMlir) {
  constexpr absl::string_view kExpected = R"(
    CHECK:       module @wrapped_scatter attributes {{{.*}}xla.extra_backend_options = #xla<extra_backend_options["xla_cpu_disable_loop_unrolling"]>{{.*}}}
    CHECK:       @wrapped_scatter(
    CHECK-SAME:    xla.entry
    CHECK:           %[[XLA_LOOP:.+]] = xla.loop
    CHECK:           xla.pure_call
    CHECK:           scf.if
    CHECK:             xla.pure_call
    CHECK:             tensor.extract
    CHECK:             arith.addf
    CHECK:           return %[[XLA_LOOP]]
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kScatterHlo));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  CpuScatterFusion emitter(*buffer_assignment, fusion, mlir_context.get());
  TF_ASSERT_OK_AND_ASSIGN(KernelDefinition kernel_definition,
                          emitter.EmitKernelDefinition());
  const auto& mlir_source = kernel_definition.source();
  auto mlir_dump = mlir_source.ToString();
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(mlir_dump, kExpected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(CpuFusionEmitterTest, ScatterLlvm) {
  constexpr absl::string_view kExpected = R"(
    CHECK-NOT:  @wrapped_scatter_entry(
    CHECK-NOT:  @wrapped_scatter_kernel(
    CHECK:      @wrapped_scatter(
    CHECK:      uwtable "frame-pointer"="all"
    CHECK-SAME: "prefer-vector-width"="512"
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kScatterHlo));
  auto& debug_options = hlo_module->mutable_config().mutable_debug_options();
  debug_options.set_xla_cpu_prefer_vector_width(512);
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  CpuScatterFusion emitter(*buffer_assignment, fusion, mlir_context.get());
  TF_ASSERT_OK_AND_ASSIGN(KernelDefinition kernel_definition,
                          emitter.EmitKernelDefinition());
  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  TF_ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource llvm_source,
      compiler.Compile(std::move(kernel_definition).TakeSource()));
  auto llvm_dump = llvm_source.ToString();
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(llvm_dump, kExpected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(CpuFusionEmitterTest,
       ScatterReorderedOperandsWritesToResultSliceAndHonorsNonInvariant) {
  constexpr absl::string_view kHlo = R"(
    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }

    scatter_computation {
      indices = s32[2,1] parameter(0)
      updates = s32[2,1] parameter(1)
      operand = s32[4] parameter(2)
      ROOT s = s32[4] scatter(operand, indices, updates),
        update_window_dims={1},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=add
    }

    ENTRY main {
      p0 = s32[2,1] parameter(0)
      p1 = s32[2,1] parameter(1)
      p2 = s32[4] parameter(2)
      ROOT wrapped_scatter = s32[4] fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation,
        frontend_attributes={xla.no_invariant_operands="0"}
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  const auto* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  CpuScatterFusion emitter(*buffer_assignment, fusion, mlir_context.get());
  ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> kernel_definition,
                       emitter.EmitKernelDefinition());
  EXPECT_FALSE(kernel_definition.spec().invariant_arguments().contains(0));
  EXPECT_TRUE(kernel_definition.spec().invariant_arguments().contains(1));
  EXPECT_FALSE(kernel_definition.spec().invariant_arguments().contains(2));
  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource llvm_source,
      compiler.Compile(std::move(kernel_definition).TakeSource()));
  const llvm::Function* wrapped =
      llvm_source.module()->getFunction("wrapped_scatter_wrapped");
  ASSERT_NE(wrapped, nullptr);
  // Operand 0 is marked not invariant, operand 1 is read only, operand 2 and
  // arg 3 share the writable result slice, and args 4..6 are the workgroup IDs.
  ASSERT_EQ(wrapped->arg_size(), 7);
  EXPECT_FALSE(wrapped->getArg(0)->onlyReadsMemory());
  EXPECT_FALSE(wrapped->getArg(0)->hasNoAliasAttr());
  EXPECT_TRUE(wrapped->getArg(1)->onlyReadsMemory());
  EXPECT_TRUE(wrapped->getArg(1)->hasNoAliasAttr());
  EXPECT_FALSE(wrapped->getArg(2)->onlyReadsMemory());
  EXPECT_TRUE(wrapped->getArg(2)->hasNoAliasAttr());
  EXPECT_FALSE(wrapped->getArg(2)->use_empty());
  EXPECT_TRUE(wrapped->getArg(3)->use_empty());
}

TEST_F(CpuFusionEmitterTest, SortComparatorMlirAndLlvm) {
  constexpr absl::string_view kSortHlo = R"(
    HloModule sort_module
    compare {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT lt = pred[] compare(lhs, rhs), direction=LT
    }
    ENTRY main {
      p0 = f32[8] parameter(0)
      ROOT sorted = f32[8] sort(p0), dimensions={0}, to_apply=compare
    }
  )";
  constexpr absl::string_view kExpectedMlir = R"(
    CHECK:      func.func @compare(
    CHECK-SAME:   %[[ARG0:.+]]: tensor<f32>, %[[ARG1:.+]]: tensor<f32>) -> i1
    CHECK-SAME:   xla.cpu.is_comparator
    CHECK-NOT:    xla.entry
    CHECK:        %[[LHS:.+]] = tensor.extract %[[ARG0]][] : tensor<f32>
    CHECK:        %[[RHS:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
    CHECK:        %[[CMP:.+]] = xla.pure_call @compare_lt(%[[LHS]], %[[RHS]]) : (f32, f32) -> i8
    CHECK:        %[[RES:.+]] = arith.trunci %[[CMP]] : i8 to i1
    CHECK:        return %[[RES]] : i1
  )";
  constexpr absl::string_view kExpectedLlvm = R"(
    CHECK:      define zeroext i1 @compare(ptr noundef nonnull %{{.+}}) #[[ATTR:[0-9]+]]
    CHECK:      attributes #[[ATTR]] = {{{.*}}uwtable{{[ }]}}
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kSortHlo));
  const HloComputation* comparator =
      hlo_module->entry_computation()->root_instruction()->to_apply();
  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      MlirKernelSource mlir_source,
      EmitSortComparator(*mlir_context, *comparator, "compare"));
  EXPECT_THAT(RunFileCheck(mlir_source.ToString(), kExpectedMlir),
              absl_testing::IsOkAndHolds(true));

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource llvm_source,
                       compiler.Compile(std::move(mlir_source)));
  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), kExpectedLlvm),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest, WrittenDonatedParameterDropsEntryNoAlias) {
  constexpr absl::string_view kHlo = R"(
    HloModule m, input_output_alias={ {0}: (0, {}, may-alias), {1}: (2, {}, may-alias) }

    fused_a {
      p0 = f32[8] parameter(0)
      p1 = f32[8] parameter(1)
      ROOT sum = f32[8] add(p0, p1)
    }

    fused_b {
      p0 = f32[8] parameter(0)
      p1 = f32[8] parameter(1)
      ROOT sum = f32[8] add(p0, p1)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      p1 = f32[8] parameter(1)
      p2 = f32[8] parameter(2)
      f0 = f32[8] fusion(p0, p1), kind=kLoop, calls=fused_a
      t1 = f32[8] negate(p1)
      f1 = f32[8] fusion(p2, t1), kind=kLoop, calls=fused_b
      ROOT out = (f32[8], f32[8]) tuple(f0, f1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  const HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  const auto* f0 = Cast<HloFusionInstruction>(root->operand(0));
  const auto* f1 = Cast<HloFusionInstruction>(root->operand(1));

  ASSERT_OK_AND_ASSIGN(std::string fp0,
                       GetFusionFingerprint(*f0, *buffer_assignment,
                                            GetDefaultBufferAlignment()));
  ASSERT_OK_AND_ASSIGN(std::string fp1,
                       GetFusionFingerprint(*f1, *buffer_assignment,
                                            GetDefaultBufferAlignment()));
  EXPECT_NE(fp0, fp1);

  auto compile =
      [&](const HloFusionInstruction* f) -> absl::StatusOr<LlvmKernelSource> {
    auto mlir_context = FusionCompiler::CreateContext();
    ABSL_ASSIGN_OR_RETURN(
        KernelDefinition<MlirKernelSource> def,
        EmitFusionKernel(*mlir_context, *f, buffer_assignment.get(),
                         /*use_unique_c_name=*/false,
                         /*enable_tiled_emitter=*/false));
    FusionCompiler compiler(mlir_context.get(),
                            FusionCompiler::Options{512, 1, true});
    return compiler.Compile(std::move(def).TakeSource());
  };

  // In f0, both p0 (written donated) and p1 (read entry parameter) omit
  // noalias, while p1 stays readonly.
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource src0, compile(f0));
  const llvm::Function* wrapped0 = src0.module()->getFunction("f0_wrapped");
  ASSERT_NE(wrapped0, nullptr);
  EXPECT_FALSE(wrapped0->getArg(0)->hasNoAliasAttr());
  EXPECT_FALSE(wrapped0->getArg(1)->hasNoAliasAttr());
  EXPECT_TRUE(wrapped0->getArg(1)->onlyReadsMemory());

  // In f1, p2 (written donated) omits noalias, while t1 (temporary) keeps
  // noalias and readonly.
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource src1, compile(f1));
  const llvm::Function* wrapped1 = src1.module()->getFunction("f1_wrapped");
  ASSERT_NE(wrapped1, nullptr);
  EXPECT_FALSE(wrapped1->getArg(0)->hasNoAliasAttr());
  EXPECT_TRUE(wrapped1->getArg(1)->hasNoAliasAttr());
  EXPECT_TRUE(wrapped1->getArg(1)->onlyReadsMemory());
}

TEST_F(CpuFusionEmitterTest,
       WrittenDonatedParameterDropsEntryNoAliasOnConcatDusAndScatter) {
  constexpr absl::string_view kHlo = R"(
    HloModule m, input_output_alias={
      {0}: (0, {}, may-alias),
      {1}: (1, {}, may-alias),
      {2}: (2, {}, may-alias),
      {3}: (3, {}, may-alias)
    }

    add_scalar {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    concat_fn {
      a = f32[4] parameter(0)
      b = f32[4] parameter(1)
      ROOT c = f32[8] concatenate(a, b), dimensions={0}
    }

    dus_fn {
      dst = f32[8] parameter(0)
      upd = f32[4] parameter(1)
      ROOT d = f32[8] dynamic-update-slice(dst, upd, s32[] constant(2))
    }

    scatter_fn_a {
      op = f32[8] parameter(0)
      idx = s32[2,1] parameter(1)
      upd = f32[2,1] parameter(2)
      ROOT s = f32[8] scatter(op, idx, upd),
        update_window_dims={1},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=add_scalar
    }

    scatter_fn_b {
      op = f32[8] parameter(0)
      idx = s32[2,1] parameter(1)
      upd = f32[2,1] parameter(2)
      ROOT s = f32[8] scatter(op, idx, upd),
        update_window_dims={1},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=add_scalar
    }

    ENTRY main {
      d_concat = f32[8] parameter(0)
      d_dus = f32[8] parameter(1)
      d_scatter_a = f32[8] parameter(2)
      d_scatter_b = f32[8] parameter(3)
      e_half = f32[4] parameter(4)
      e_idx = s32[2,1] parameter(5)
      e_upd = f32[2,1] parameter(6)

      f_concat = f32[8] fusion(e_half, e_half), kind=kLoop, calls=concat_fn
      f_dus = f32[8] fusion(d_dus, e_half), kind=kLoop, calls=dus_fn
      f_scatter_a = f32[8] fusion(d_scatter_a, e_idx, e_upd), kind=kLoop, calls=scatter_fn_a
      t_upd = f32[2,1] negate(e_upd)
      f_scatter_b = f32[8] fusion(d_scatter_b, e_idx, t_upd), kind=kLoop, calls=scatter_fn_b
      ROOT out = (f32[8], f32[8], f32[8], f32[8]) tuple(
          f_concat, f_dus, f_scatter_a, f_scatter_b)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  const HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  const auto* f_concat = Cast<HloFusionInstruction>(root->operand(0));
  const auto* f_dus = Cast<HloFusionInstruction>(root->operand(1));
  const auto* f_scatter_a = Cast<HloFusionInstruction>(root->operand(2));
  const auto* f_scatter_b = Cast<HloFusionInstruction>(root->operand(3));

  auto compile =
      [&](const HloFusionInstruction* f) -> absl::StatusOr<LlvmKernelSource> {
    auto mlir_context = FusionCompiler::CreateContext();
    ABSL_ASSIGN_OR_RETURN(
        KernelDefinition<MlirKernelSource> def,
        EmitFusionKernel(*mlir_context, *f, buffer_assignment.get(),
                         /*use_unique_c_name=*/false,
                         /*enable_tiled_emitter=*/false));
    FusionCompiler compiler(mlir_context.get(),
                            FusionCompiler::Options{512, 1, true});
    return compiler.Compile(std::move(def).TakeSource());
  };

  for (const HloFusionInstruction* f : {f_concat, f_dus, f_scatter_a}) {
    ASSERT_OK_AND_ASSIGN(LlvmKernelSource src, compile(f));
    const llvm::Function* wrapped =
        src.module()->getFunction(absl::StrCat(f->name(), "_wrapped"));
    ASSERT_NE(wrapped, nullptr);
    for (const llvm::Argument& arg : wrapped->args()) {
      if (arg.getType()->isPointerTy()) {
        EXPECT_FALSE(arg.hasNoAliasAttr()) << f->name();
      }
    }
  }

  ASSERT_OK_AND_ASSIGN(std::string fp_a,
                       GetFusionFingerprint(*f_scatter_a, *buffer_assignment,
                                            GetDefaultBufferAlignment()));
  ASSERT_OK_AND_ASSIGN(std::string fp_b,
                       GetFusionFingerprint(*f_scatter_b, *buffer_assignment,
                                            GetDefaultBufferAlignment()));
  EXPECT_NE(fp_a, fp_b);
}

TEST_F(CpuFusionEmitterTest, SortComparatorUnsupportedOp) {
  constexpr absl::string_view kSortHlo = R"(
    HloModule sort_module
    cond {
      state = s32[] parameter(0)
      limit = s32[] constant(4)
      ROOT lt = pred[] compare(state, limit), direction=LT
    }
    body {
      state = s32[] parameter(0)
      one = s32[] constant(1)
      ROOT next = s32[] add(state, one)
    }
    compare {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      loop = s32[] while(lhs), condition=cond, body=body
      ROOT lt = pred[] compare(loop, rhs), direction=LT
    }
    ENTRY main {
      p0 = s32[8] parameter(0)
      ROOT sorted = s32[8] sort(p0), dimensions={0}, to_apply=compare
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kSortHlo));
  const HloComputation* comparator =
      hlo_module->entry_computation()->root_instruction()->to_apply();
  auto mlir_context = FusionCompiler::CreateContext();
  EXPECT_THAT(EmitSortComparator(*mlir_context, *comparator, "compare"),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_P(CpuFusionEmitterTest, WorkDimensionsTest) {
  const GetWorkDimensionsTestCase& tc = GetParam();
  constexpr absl::string_view kHloTemplate = R"(
    HloModule test
    fused_computation {
      p0 = $0 parameter(0)
      ROOT res = $0 $2
    }
    ENTRY main {
      p0 = $0 parameter(0)
      ROOT fusion = $0 fusion(p0), kind=kLoop,
        calls=fused_computation,
        backend_config={"outer_dimension_partitions":["$1"]}
    }
  )";
  std::string hlo = absl::Substitute(kHloTemplate, tc.shape_and_layout,
                                     tc.outer_dimension_partitions, tc.root_op);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*module));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());

  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> kernel_definition,
      EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));
  const WorkDimensions& dims = kernel_definition.spec().work_dimensions();
  EXPECT_EQ(dims.num_work_groups.x, tc.expected_num_work_groups);
  EXPECT_THAT(dims.work_tile_size.dimensions,
              ElementsAreArray(tc.expected_work_tile_size));
}

INSTANTIATE_TEST_SUITE_P(
    WorkDimensions, CpuFusionEmitterTest,
    ::testing::ValuesIn<GetWorkDimensionsTestCase>({
        {
            /*test_name=*/"NonSubByteDoesNotRoundOddTile",
            /*shape_and_layout=*/"f32[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{3, 3},
        },
        {
            // CeilOfRatio(6, 2) = 3 is rounded up to 4 (multiple of 2
            // elements/byte), giving 4 * 3 = 12 elements (6 whole bytes) per
            // workgroup.
            /*test_name=*/"SubByteS4RoundsOddSplitTileToMultipleOfTwo",
            /*shape_and_layout=*/"s4[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{4, 3},
        },
        {
            // CeilOfRatio(5, 2) = 3 is also rounded up to 4, so workgroup 0
            // writes 12 elements (6 whole bytes) and workgroup 1 writes the
            // remaining 3 elements starting at byte 6.
            /*test_name=*/
            "SubByteS4OddRowsStillRoundsSplitTileToMultipleOfTwo",
            /*shape_and_layout=*/"s4[5,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{4, 3},
        },
        {
            // CeilOfRatio(6, 2) = 3 is rounded up to 4 (multiple of 4
            // elements/byte), giving 4 * 3 = 12 elements (3 whole bytes) per
            // workgroup.
            /*test_name=*/"SubByteS2RoundsSplitTileToMultipleOfFour",
            /*shape_and_layout=*/"s2[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{4, 3},
        },
        {
            // CeilOfRatio(12, 2) = 6 is rounded up to 8 (multiple of 8
            // elements/byte), giving 8 * 3 = 24 elements (3 whole bytes) per
            // workgroup.
            /*test_name=*/"SubByteS1RoundsSplitTileToMultipleOfEight",
            /*shape_and_layout=*/"s1[12,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{8, 3},
        },
        {
            // PRED is stored 1 element per byte, so odd split tiles are
            // not rounded.
            /*test_name=*/"PredIsUnpackedAndDoesNotRoundOddTile",
            /*shape_and_layout=*/"pred[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{3, 3},
        },
        {
            // CeilOfRatio(6, 4) = 2 (already a multiple of 2 elems/byte);
            // 3 workgroups of size 2 cover all 6 rows, so num_work_groups is
            // shrunk from 4 to 3.
            /*test_name=*/"SubByteRecomputesNumWorkGroupsAfterRounding",
            /*shape_and_layout=*/"s4[6,3]{1,0}",
            /*outer_dimension_partitions=*/4,
            /*expected_num_work_groups=*/3,
            /*expected_work_tile_size=*/{2, 3},
        },
        {
            // Concatenate fusions with packed sub-byte outputs must run in a
            // single workgroup covering the entire shape, because concatenated
            // operands can start at non-byte-aligned offsets.
            /*test_name=*/"ConcatenateSubByteForcesSingleWorkGroup",
            /*shape_and_layout=*/"s4[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/1,
            /*expected_work_tile_size=*/{6, 3},
            /*root_op=*/"concatenate(p0), dimensions={0}",
        },
        {
            // DynamicUpdateSlice fusions with packed sub-byte outputs must also
            // run in a single workgroup covering the entire shape, because the
            // update index may land at a non-byte aligned offset.
            /*test_name=*/"DynamicUpdateSliceSubByteForcesSingleWorkGroup",
            /*shape_and_layout=*/"s4[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/1,
            /*expected_work_tile_size=*/{6, 3},
            /*root_op=*/
            "dynamic-update-slice(p0, p0, s32[] constant(0), s32[] "
            "constant(0))",
        },
        {
            // 3D shape where the leading dim (2) is smaller than 4 workgroups:
            // dim 0 (2) is folded with dim 1 (6) into 12 slices of 3 elements;
            // CeilOfRatio(12, 4) = 3 rounds up to 4, leaving 3 workgroups of
            // tile {4, 3}.
            /*test_name=*/"ThreeDimsFoldsLeadingDimAndRoundsSplitTile",
            /*shape_and_layout=*/"s4[2,6,3]{2,1,0}",
            /*outer_dimension_partitions=*/4,
            /*expected_num_work_groups=*/3,
            /*expected_work_tile_size=*/{4, 3},
        },
        {
            // 3D shape where the leading dim (6) is large enough to split
            // directly: CeilOfRatio(6, 2) = 3 rounds up to 4, keeping both
            // minor dims (4 and 3) intact.
            /*test_name=*/"ThreeDimsSplitsOutermostDimAndKeepsMinorDims",
            /*shape_and_layout=*/"s4[6,4,3]{2,1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{4, 4, 3},
        },
        {
            // More workgroups (8) than the product of the first two dimensions
            // (2 * 2 = 4): all three dimensions fold together (2 * 2 * 3 = 12),
            // and CeilOfRatio(12, 8) = 2 yields 6 workgroups of 1D tile {2}.
            /*test_name=*/"FoldsAllDimensionsIntoSingleTileDim",
            /*shape_and_layout=*/"s4[2,2,3]{2,1,0}",
            /*outer_dimension_partitions=*/8,
            /*expected_num_work_groups=*/6,
            /*expected_work_tile_size=*/{2},
        },
        {
            // As many workgroups (12) as total elements (2 * 2 * 3 = 12): all
            // three dimensions fold together, and CeilOfRatio(12, 12) = 1 is
            // rounded up to 2 elements (1 byte) per workgroup, shrinking
            // num_work_groups from 12 to 6.
            /*test_name=*/"OneWorkGroupPerElementRoundsToWholeByte",
            /*shape_and_layout=*/"s4[2,2,3]{2,1,0}",
            /*outer_dimension_partitions=*/12,
            /*expected_num_work_groups=*/6,
            /*expected_work_tile_size=*/{2},
        },
        {
            // Column-major layout {0,1}: dim 1 (size 6) is the most-major
            // physical dimension and gets split and rounded from 3 to 4, while
            // dim 0 (size 3) is minor.
            /*test_name=*/"SubByteColumnMajorLayoutSplitsPhysicalMajorDim",
            /*shape_and_layout=*/"s4[3,6]{0,1}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{4, 3},
        },
        {
            // Single workgroup does not round odd dimensions.
            /*test_name=*/"SingleWorkGroupDoesNotRoundOddSubByteDims",
            /*shape_and_layout=*/"s4[5,3]{1,0}",
            /*outer_dimension_partitions=*/1,
            /*expected_num_work_groups=*/1,
            /*expected_work_tile_size=*/{5, 3},
        },
        {
            // f6 types are stored one element per byte and are not rounded.
            /*test_name=*/"NonPowerOfTwoSubByteF6DoesNotRoundOddTile",
            /*shape_and_layout=*/"f6e3m2fn[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{3, 3},
        },
        {
            // More workgroups (16) than elements (12): clamped to 12, then
            // rounded to 2 elements (1 byte) per workgroup.
            /*test_name=*/"MoreWorkGroupsThanElementsSubByteRoundsToWholeByte",
            /*shape_and_layout=*/"s4[2,2,3]{2,1,0}",
            /*outer_dimension_partitions=*/16,
            /*expected_num_work_groups=*/6,
            /*expected_work_tile_size=*/{2},
        },
        {
            // More workgroups (8) than elements (6): clamped to 6.
            /*test_name=*/
            "MoreWorkGroupsThanElementsNonSubByteClampsToElements",
            /*shape_and_layout=*/"f32[2,3]{1,0}",
            /*outer_dimension_partitions=*/8,
            /*expected_num_work_groups=*/6,
            /*expected_work_tile_size=*/{1},
        },
        {
            // Zero-element shapes get a single workgroup and an empty tile.
            /*test_name=*/"ZeroElementsUsesSingleWorkGroup",
            /*shape_and_layout=*/"f32[0,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/1,
            /*expected_work_tile_size=*/{},
        },
    }),
    [](const ::testing::TestParamInfo<GetWorkDimensionsTestCase>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace cpu
}  // namespace xla
