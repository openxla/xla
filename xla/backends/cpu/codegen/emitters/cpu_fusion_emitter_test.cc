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

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/backends/cpu/codegen/emitters/cpu_scatter_emitter.h"
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/backends/cpu/codegen/fusion_emitter.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/codegen/target_machine_test_base.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/codegen/device_spec.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/ir_emission_utils.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/status_macros.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/types.h"
#include "xla/xla.pb.h"

namespace xla {
namespace cpu {
namespace {

using ::testing::ContainsRegex;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::UnorderedElementsAre;

bool ScatterConstraintsSatisfied(const IndexingMap& map, int64_t thread_id,
                                 int64_t index_id, int64_t vector_id,
                                 int64_t vector_element_id) {
  mlir::MLIRContext* ctx = map.GetMLIRContext();
  return map.ConstraintsSatisfied(
      {CreateSymbolicConstant(thread_id, ctx)},
      {CreateSymbolicConstant(index_id, ctx),
       CreateSymbolicConstant(vector_id, ctx),
       CreateSymbolicConstant(vector_element_id, ctx)});
}

llvm::SmallVector<int64_t, 4> EvaluateScatterMap(const IndexingMap& map,
                                                 int64_t thread_id,
                                                 int64_t index_id,
                                                 int64_t vector_id,
                                                 int64_t vector_element_id) {
  mlir::MLIRContext* ctx = map.GetMLIRContext();
  return map.Evaluate({CreateSymbolicConstant(thread_id, ctx)},
                      {CreateSymbolicConstant(index_id, ctx),
                       CreateSymbolicConstant(vector_id, ctx),
                       CreateSymbolicConstant(vector_element_id, ctx)});
}

struct LoweredScatterLoopIr {
  std::string after_peel;
  std::string after_scf;
};

absl::StatusOr<LoweredScatterLoopIr> RunPeelAndLowerScatterLoop(
    const IndexingMap& map, int64_t num_threads, mlir::MLIRContext* ctx) {
  mlir::OpBuilder builder(ctx);
  mlir::Location loc = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      llvm_ir::CreateMlirModuleOp(loc, "scatter_loop_module");

  builder.setInsertionPointToStart(module->getBody());
  auto index_type = builder.getIndexType();
  auto func = mlir::func::FuncOp::create(
      builder, loc, "visit_scatter",
      builder.getFunctionType({index_type}, {index_type}));

  mlir::ImplicitLocOpBuilder b(loc, func);
  b.setInsertionPointToStart(func.addEntryBlock());
  WorkGroupIdOp workgroup_id = WorkGroupIdOp::create(b, WorkGroupDimension::x);
  workgroup_id->setAttr("xla.range", b.getIndexArrayAttr({0, num_threads - 1}));

  mlir::ValueRange results = emitters::EmitXlaLoopOp(
      b, {workgroup_id}, {func.getArgument(0)}, map,
      [](mlir::ImplicitLocOpBuilder& nested_b, mlir::ValueRange ivs,
         mlir::ValueRange map_results,
         mlir::ValueRange iter_args) -> llvm::SmallVector<mlir::Value> {
        return {iter_args[0]};
      });
  mlir::func::ReturnOp::create(b, results);

  LoweredScatterLoopIr out;
  {
    mlir::PassManager peel_pm(ctx);
    peel_pm.enableVerifier(true);
    peel_pm.addNestedPass<mlir::func::FuncOp>(
        cpu::createPeelWorkgroupLoopPass());
    tsl::StatusScopedDiagnosticHandler diag(ctx);
    ABSL_RETURN_IF_ERROR(diag.consumeStatus(peel_pm.run(*module)));
  }
  out.after_peel = mlir::debugString(*module);

  {
    mlir::PassManager scf_pm(ctx);
    scf_pm.enableVerifier(true);
    scf_pm.addNestedPass<mlir::func::FuncOp>(
        emitters::createLowerXlaLoopsToScfPass());
    tsl::StatusScopedDiagnosticHandler diag(ctx);
    ABSL_RETURN_IF_ERROR(diag.consumeStatus(scf_pm.run(*module)));
  }
  out.after_scf = mlir::debugString(*module);
  return out;
}

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

TEST_F(CpuFusionEmitterTest, ScatterIndexingMapExactDivisibility) {
  std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();

  // num_updates = 6, num_threads = 3 => num_updates_per_thread = 2.
  // Simplify eliminates all constraints so neither workgroup peeling nor inner
  // bounds checks are emitted.
  IndexingMap map = GetScatterIndexingMap(
      /*updates_operand_shape=*/{6, 4}, /*num_threads=*/3, /*vector_size=*/4,
      ctx.get());
  EXPECT_FALSE(map.IsKnownEmpty());
  EXPECT_TRUE(ScatterConstraintsSatisfied(map, /*thread_id=*/2, /*index_id=*/1,
                                          /*vector_id=*/0,
                                          /*vector_element_id=*/3));

  map.Simplify();
  EXPECT_FALSE(map.IsKnownEmpty());
  EXPECT_EQ(map.GetDimensionBound(0), (Interval{0, 2}));
  EXPECT_THAT(map.GetSymbolBounds(),
              ElementsAre(Interval{0, 1}, Interval{0, 0}, Interval{0, 3}));
  EXPECT_EQ(map.GetConstraintsCount(), 0);
  EXPECT_THAT(map.ComputeResultRanges(),
              ElementsAre(Interval{0, 5}, Interval{0, 3}));

  ASSERT_OK_AND_ASSIGN(
      LoweredScatterLoopIr ir,
      RunPeelAndLowerScatterLoop(map, /*num_threads=*/3, ctx.get()));
  EXPECT_THAT(RunFileCheck(ir.after_peel, R"(
    CHECK-NOT: scf.if
    CHECK:     xla.loop
  )"),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(ir.after_scf, R"(
    CHECK-NOT: xla.loop
    CHECK-NOT: scf.if
    CHECK-NOT: arith.cmpi
    CHECK:     scf.for
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest, ScatterIndexingMapNonDivisibleNumUpdates) {
  std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();

  // num_updates = 5, num_threads = 3 => num_updates_per_thread = 2.
  // Workgroup 2 has index_id 0 -> updates_id 4 (last valid update) and
  // index_id 1 -> updates_id 5 (must be rejected).
  IndexingMap map = GetScatterIndexingMap(
      /*updates_operand_shape=*/{5, 4}, /*num_threads=*/3, /*vector_size=*/4,
      ctx.get());
  EXPECT_FALSE(map.IsKnownEmpty());
  EXPECT_TRUE(ScatterConstraintsSatisfied(map, /*thread_id=*/0, /*index_id=*/0,
                                          /*vector_id=*/0,
                                          /*vector_element_id=*/0));
  EXPECT_TRUE(ScatterConstraintsSatisfied(map, /*thread_id=*/1, /*index_id=*/1,
                                          /*vector_id=*/0,
                                          /*vector_element_id=*/3));
  EXPECT_EQ(EvaluateScatterMap(map, 2, 0, 0, 3),
            (llvm::SmallVector<int64_t, 4>{4, 3}));
  EXPECT_TRUE(ScatterConstraintsSatisfied(map, /*thread_id=*/2, /*index_id=*/0,
                                          /*vector_id=*/0,
                                          /*vector_element_id=*/3));
  EXPECT_EQ(EvaluateScatterMap(map, 2, 1, 0, 0),
            (llvm::SmallVector<int64_t, 4>{5, 0}));
  EXPECT_FALSE(ScatterConstraintsSatisfied(map, /*thread_id=*/2, /*index_id=*/1,
                                           /*vector_id=*/0,
                                           /*vector_element_id=*/0));

  map.Simplify();
  EXPECT_FALSE(map.IsKnownEmpty());
  EXPECT_EQ(map.GetDimensionBound(0), (Interval{0, 2}));
  EXPECT_THAT(map.GetSymbolBounds(),
              ElementsAre(Interval{0, 1}, Interval{0, 0}, Interval{0, 3}));
  ASSERT_EQ(map.GetConstraintsCount(), 1);
  EXPECT_EQ(map.GetSymbolicConstraints().front().second, (Interval{0, 4}));
  EXPECT_THAT(map.ComputeResultRanges(),
              ElementsAre(Interval{0, 4}, Interval{0, 3}));

  // PeelWorkgroupLoopPass peels workgroups 0..1 without constraints and
  // narrows workgroup 2's index_id domain to [0, 0] (updates_id == 4 only).
  ASSERT_OK_AND_ASSIGN(
      LoweredScatterLoopIr ir,
      RunPeelAndLowerScatterLoop(map, /*num_threads=*/3, ctx.get()));
  EXPECT_THAT(RunFileCheck(ir.after_peel, R"(
    CHECK: scf.if
    CHECK:   xla.loop {{.*}} in #xla.indexing_map<"(thread_id)[index_id, vector_id, vector_element_id] -> (thread_id * 2 + index_id, vector_element_id), domain: thread_id in [0, 1], index_id in [0, 1], vector_id in [0, 0], vector_element_id in [0, 3]">
    CHECK: else
    CHECK:   xla.loop {{.*}} in #xla.indexing_map<"(thread_id)[index_id, vector_id, vector_element_id] -> (4, vector_element_id), domain: thread_id in [2, 2], index_id in [0, 0], vector_id in [0, 0], vector_element_id in [0, 3]">
  )"),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(ir.after_scf, R"(
    CHECK-NOT: xla.loop
    CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
    CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
    CHECK:     scf.if
    CHECK:       scf.for %{{.+}} = %{{.+}} to %[[C2]]
    CHECK:     else
    CHECK:       scf.for %{{.+}} = %{{.+}} to %[[C1]]
  )"),
              absl_testing::IsOkAndHolds(true));

  // Two threads with 5 updates (3 updates per thread): workgroup 1 checks
  // `updates_id <= 4` directly in SCF without peeling, rejecting 5.
  IndexingMap two_thread_map = GetScatterIndexingMap(
      {5, 4}, /*num_threads=*/2, /*vector_size=*/4, ctx.get());
  EXPECT_TRUE(ScatterConstraintsSatisfied(two_thread_map, 1, 1, 0, 0));
  EXPECT_FALSE(ScatterConstraintsSatisfied(two_thread_map, 1, 2, 0, 0));
  two_thread_map.Simplify();
  ASSERT_EQ(two_thread_map.GetConstraintsCount(), 1);
  EXPECT_EQ(two_thread_map.GetSymbolicConstraints().front().second,
            (Interval{0, 4}));
  ASSERT_OK_AND_ASSIGN(
      LoweredScatterLoopIr two_thread_ir,
      RunPeelAndLowerScatterLoop(two_thread_map, /*num_threads=*/2, ctx.get()));
  EXPECT_THAT(RunFileCheck(two_thread_ir.after_scf, R"(
    CHECK-NOT: xla.loop
    CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
    CHECK:     scf.for
    CHECK:       arith.cmpi sle, %{{.+}}, %[[C4]]
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest,
       ScatterIndexingMapMultiUpdatePerThreadWithIdleWorkgroups) {
  std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();

  // num_updates = 4, num_threads = 3 => num_updates_per_thread = 2.
  // Workgroups 0..1 are full; workgroup 2 is completely idle even at
  // index_id = 0 (updates_id == 4 must be rejected).
  IndexingMap map_4_3 = GetScatterIndexingMap(
      /*updates_operand_shape=*/{4, 4}, /*num_threads=*/3, /*vector_size=*/4,
      ctx.get());
  EXPECT_TRUE(ScatterConstraintsSatisfied(map_4_3, 1, 1, 0, 3));
  EXPECT_EQ(EvaluateScatterMap(map_4_3, 2, 0, 0, 0),
            (llvm::SmallVector<int64_t, 4>{4, 0}));
  EXPECT_FALSE(ScatterConstraintsSatisfied(map_4_3, 2, 0, 0, 0));
  EXPECT_FALSE(ScatterConstraintsSatisfied(map_4_3, 2, 1, 0, 0));

  map_4_3.Simplify();
  EXPECT_EQ(map_4_3.GetDimensionBound(0), (Interval{0, 1}));
  EXPECT_THAT(map_4_3.GetSymbolBounds(),
              ElementsAre(Interval{0, 1}, Interval{0, 0}, Interval{0, 3}));
  EXPECT_THAT(map_4_3.GetSymbolicConstraints(), IsEmpty());
  EXPECT_THAT(map_4_3.ComputeResultRanges(),
              ElementsAre(Interval{0, 3}, Interval{0, 3}));

  ASSERT_OK_AND_ASSIGN(
      LoweredScatterLoopIr ir_4_3,
      RunPeelAndLowerScatterLoop(map_4_3, /*num_threads=*/3, ctx.get()));
  EXPECT_THAT(RunFileCheck(ir_4_3.after_scf, R"(
    CHECK-NOT: xla.loop
    CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
    CHECK-DAG: %[[WG:.+]] = xla.workgroup_id x
    CHECK:     scf.for
    CHECK:       arith.cmpi sle, %[[WG]], %[[C1]]
  )"),
              absl_testing::IsOkAndHolds(true));

  // num_updates = 5, num_threads = 4 => num_updates_per_thread = 2.
  // Workgroups 0..1 are full, workgroup 2 is partial (updates_id == 4 only),
  // and workgroup 3 is completely idle (updates_id == 6 rejected).
  IndexingMap map_5_4 = GetScatterIndexingMap(
      /*updates_operand_shape=*/{5, 4}, /*num_threads=*/4, /*vector_size=*/4,
      ctx.get());
  EXPECT_TRUE(ScatterConstraintsSatisfied(map_5_4, 2, 0, 0, 0));
  EXPECT_FALSE(ScatterConstraintsSatisfied(map_5_4, 2, 1, 0, 0));
  EXPECT_FALSE(ScatterConstraintsSatisfied(map_5_4, 3, 0, 0, 0));

  map_5_4.Simplify();
  EXPECT_EQ(map_5_4.GetDimensionBound(0), (Interval{0, 3}));
  ASSERT_EQ(map_5_4.GetConstraintsCount(), 1);
  EXPECT_EQ(map_5_4.GetSymbolicConstraints().front().second, (Interval{0, 4}));
  EXPECT_THAT(map_5_4.ComputeResultRanges(),
              ElementsAre(Interval{0, 4}, Interval{0, 3}));

  ASSERT_OK_AND_ASSIGN(
      LoweredScatterLoopIr ir_5_4,
      RunPeelAndLowerScatterLoop(map_5_4, /*num_threads=*/4, ctx.get()));
  EXPECT_THAT(RunFileCheck(ir_5_4.after_peel, R"(
    CHECK: scf.if
    CHECK:   xla.loop {{.*}} thread_id in [0, 1], index_id in [0, 1]
    CHECK: else
    CHECK:   xla.loop {{.*}} thread_id in [2, 2], index_id in [0, 0]
  )"),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(ir_5_4.after_scf, R"(
    CHECK-NOT: xla.loop
    CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
    CHECK-DAG: %[[WG:.+]] = xla.workgroup_id x
    CHECK:     scf.if
    CHECK:     else
    CHECK:       arith.cmpi eq, %[[WG]], %[[C2]]
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest, ScatterIndexingMapMoreThreadsThanUpdates) {
  std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();

  IndexingMap map = GetScatterIndexingMap(
      /*updates_operand_shape=*/{3, 4}, /*num_threads=*/8, /*vector_size=*/4,
      ctx.get());
  EXPECT_FALSE(map.IsKnownEmpty());
  EXPECT_EQ(map.GetDimensionBound(0), (Interval{0, 7}));
  EXPECT_THAT(map.GetSymbolBounds(),
              ElementsAre(Interval{0, 0}, Interval{0, 0}, Interval{0, 3}));
  for (int64_t tid = 0; tid < 3; ++tid) {
    EXPECT_EQ(EvaluateScatterMap(map, tid, 0, 0, 0),
              (llvm::SmallVector<int64_t, 4>{tid, 0}));
    EXPECT_TRUE(ScatterConstraintsSatisfied(map, tid, 0, 0, 0));
  }
  // updates_id == num_updates (tid == 3) and all higher threads are rejected.
  for (int64_t tid = 3; tid < 8; ++tid) {
    EXPECT_FALSE(ScatterConstraintsSatisfied(map, tid, 0, 0, 0));
  }

  map.Simplify();
  EXPECT_FALSE(map.IsKnownEmpty());
  EXPECT_EQ(map.GetDimensionBound(0), (Interval{0, 2}));
  EXPECT_THAT(map.GetSymbolBounds(),
              ElementsAre(Interval{0, 0}, Interval{0, 0}, Interval{0, 3}));
  EXPECT_THAT(map.GetSymbolicConstraints(), IsEmpty());
  EXPECT_THAT(map.ComputeResultRanges(),
              ElementsAre(Interval{0, 2}, Interval{0, 3}));

  ASSERT_OK_AND_ASSIGN(
      LoweredScatterLoopIr ir,
      RunPeelAndLowerScatterLoop(map, /*num_threads=*/8, ctx.get()));
  EXPECT_THAT(RunFileCheck(ir.after_scf, R"(
    CHECK-NOT: xla.loop
    CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
    CHECK-DAG: %[[WG:.+]] = xla.workgroup_id x
    CHECK:     scf.for
    CHECK:       arith.cmpi sle, %[[WG]], %[[C2]]
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest, ScatterIndexingMapNonDivisibleVectorSize) {
  std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();

  // num_slice_elements = 3 * 2 = 6, vector_size = 4 => 2 vectors per slice;
  // (vector_id=1, vector_element_id=1) -> linear 5 is valid, linear 6 and 7
  // are rejected.
  IndexingMap map = GetScatterIndexingMap(
      /*updates_operand_shape=*/{5, 3, 2}, /*num_threads=*/3,
      /*vector_size=*/4, ctx.get());
  EXPECT_FALSE(map.IsKnownEmpty());
  EXPECT_THAT(map.GetSymbolBounds(),
              ElementsAre(Interval{0, 1}, Interval{0, 1}, Interval{0, 3}));
  EXPECT_EQ(EvaluateScatterMap(map, 2, 0, 1, 1),
            (llvm::SmallVector<int64_t, 4>{4, 2, 1}));
  EXPECT_TRUE(ScatterConstraintsSatisfied(map, 2, 0, 1, 1));
  EXPECT_FALSE(ScatterConstraintsSatisfied(map, 0, 0, 1, 2));
  EXPECT_FALSE(ScatterConstraintsSatisfied(map, 0, 0, 1, 3));
  EXPECT_FALSE(ScatterConstraintsSatisfied(map, 2, 1, 0, 0));

  map.Simplify();
  EXPECT_FALSE(map.IsKnownEmpty());
  ASSERT_EQ(map.GetConstraintsCount(), 2);
  std::vector<Interval> constraint_intervals;
  for (const auto& [expr, interval] : map.GetSymbolicConstraints()) {
    constraint_intervals.push_back(interval);
  }
  EXPECT_THAT(constraint_intervals,
              UnorderedElementsAre(Interval{0, 4}, Interval{0, 5}));
  EXPECT_EQ(map.ComputeResultRanges()[0], (Interval{0, 4}));

  // PeelWorkgroupLoopPass must not strip the slice tail constraint.
  ASSERT_OK_AND_ASSIGN(
      LoweredScatterLoopIr ir,
      RunPeelAndLowerScatterLoop(map, /*num_threads=*/3, ctx.get()));
  EXPECT_THAT(RunFileCheck(ir.after_peel, R"(
    CHECK-NOT: scf.if
    CHECK:     xla.loop
  )"),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(ir.after_scf, R"(
    CHECK-NOT: xla.loop
    CHECK-DAG: arith.constant 4 : index
    CHECK-DAG: arith.constant 5 : index
    CHECK:     scf.if
  )"),
              absl_testing::IsOkAndHolds(true));

  // When vector_size > num_slice_elements, Simplify folds the slice constraint
  // into vector_element_id's domain, enabling workgroup peeling on updates_id.
  IndexingMap small_slice_map = GetScatterIndexingMap(
      {5, 3}, /*num_threads=*/3, /*vector_size=*/4, ctx.get());
  EXPECT_TRUE(ScatterConstraintsSatisfied(small_slice_map, 2, 0, 0, 2));
  EXPECT_FALSE(ScatterConstraintsSatisfied(small_slice_map, 0, 0, 0, 3));
  EXPECT_FALSE(ScatterConstraintsSatisfied(small_slice_map, 2, 1, 0, 0));
  small_slice_map.Simplify();
  EXPECT_THAT(small_slice_map.GetSymbolBounds(),
              ElementsAre(Interval{0, 1}, Interval{0, 0}, Interval{0, 2}));
  ASSERT_EQ(small_slice_map.GetConstraintsCount(), 1);
  EXPECT_EQ(small_slice_map.GetSymbolicConstraints().front().second,
            (Interval{0, 4}));
  ASSERT_OK_AND_ASSIGN(LoweredScatterLoopIr small_slice_ir,
                       RunPeelAndLowerScatterLoop(
                           small_slice_map, /*num_threads=*/3, ctx.get()));
  EXPECT_THAT(RunFileCheck(small_slice_ir.after_peel, R"(
    CHECK: scf.if
    CHECK:   xla.loop
    CHECK: else
    CHECK:   xla.loop
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest, ScatterIndexingMapZeroUpdatesAndZeroSlices) {
  std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();

  struct EmptyCase {
    std::vector<int64_t> updates_shape;
    int64_t num_threads;
    int64_t vector_size;
  };
  const std::vector<EmptyCase> cases = {
      {{0, 4, 8}, 1, 4},  // num_updates == 0, single thread
      {{0, 4, 8}, 4, 4},  // num_updates == 0, multi thread
      {{0}, 2, 1},        // num_updates == 0, rank 0 operand
      {{5, 0, 8}, 1, 1},  // num_slice_elements == 0, single thread
      {{5, 4, 0}, 3, 4},  // num_slice_elements == 0, multi thread, vector > 1
      {{0, 0}, 2, 1},     // both num_updates == 0 and num_slice_elements == 0
  };

  for (const EmptyCase& tc : cases) {
    SCOPED_TRACE(absl::StrCat(
        "num_threads=", tc.num_threads, " vector_size=", tc.vector_size,
        " rank=", tc.updates_shape.size(), " dim0=", tc.updates_shape.front()));
    IndexingMap map = GetScatterIndexingMap(tc.updates_shape, tc.num_threads,
                                            tc.vector_size, ctx.get());
    EXPECT_TRUE(map.IsKnownEmpty());
    EXPECT_EQ(map.GetNumResults(),
              static_cast<int64_t>(tc.updates_shape.size()));
    EXPECT_FALSE(ScatterConstraintsSatisfied(map, 0, 0, 0, 0));

    map.Simplify();
    EXPECT_TRUE(map.IsKnownEmpty());
    EXPECT_EQ(map.GetDimensionBound(0), (Interval{0, -1}));
    EXPECT_THAT(map.GetSymbolBounds(),
                ElementsAre(Interval{0, -1}, Interval{0, -1}, Interval{0, -1}));
    EXPECT_THAT(map.GetSymbolicConstraints(), IsEmpty());
  }
}

TEST_F(CpuFusionEmitterTest, ScatterIndexingMapRankZeroScalarOperand) {
  std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();

  // Rank 0 scatter operand has updates shape [num_updates], so slice_shape is
  // empty and num_slice_elements == 1. Test both vector_size == 1 and
  // vector_size > 1 (where vector_element_id >= 1 must be rejected so each
  // scalar update is applied once).
  for (int64_t vector_size : {int64_t{1}, int64_t{4}}) {
    SCOPED_TRACE(absl::StrCat("vector_size=", vector_size));
    IndexingMap map = GetScatterIndexingMap(
        /*updates_operand_shape=*/{5}, /*num_threads=*/3, vector_size,
        ctx.get());
    EXPECT_FALSE(map.IsKnownEmpty());
    EXPECT_EQ(map.GetNumResults(), 1);
    EXPECT_EQ(EvaluateScatterMap(map, 2, 0, 0, 0),
              (llvm::SmallVector<int64_t, 4>{4}));
    EXPECT_TRUE(ScatterConstraintsSatisfied(map, 2, 0, 0, 0));
    EXPECT_FALSE(ScatterConstraintsSatisfied(map, 2, 1, 0, 0));
    if (vector_size > 1) {
      EXPECT_FALSE(ScatterConstraintsSatisfied(map, 0, 0, 0, 1));
    }

    map.Simplify();
    EXPECT_FALSE(map.IsKnownEmpty());
    EXPECT_EQ(map.GetDimensionBound(0), (Interval{0, 2}));
    EXPECT_THAT(map.GetSymbolBounds(),
                ElementsAre(Interval{0, 1}, Interval{0, 0}, Interval{0, 0}));
    ASSERT_EQ(map.GetConstraintsCount(), 1);
    EXPECT_EQ(map.GetSymbolicConstraints().front().second, (Interval{0, 4}));
    EXPECT_THAT(map.ComputeResultRanges(), ElementsAre(Interval{0, 4}));

    if (vector_size == 1) {
      ASSERT_OK_AND_ASSIGN(
          LoweredScatterLoopIr ir,
          RunPeelAndLowerScatterLoop(map, /*num_threads=*/3, ctx.get()));
      EXPECT_THAT(RunFileCheck(ir.after_peel, R"(
        CHECK: scf.if
        CHECK:   xla.loop {{.*}} thread_id in [0, 1], index_id in [0, 1]
        CHECK: else
        CHECK:   xla.loop {{.*}} thread_id in [2, 2], index_id in [0, 0]
      )"),
                  absl_testing::IsOkAndHolds(true));
      EXPECT_THAT(RunFileCheck(ir.after_scf, "CHECK-NOT: xla.loop"),
                  absl_testing::IsOkAndHolds(true));
    }
  }
}

TEST_F(CpuFusionEmitterTest, ScatterEmptyShapesShortCircuit) {
  constexpr absl::string_view kZeroUpdatesHlo = R"hlo(
    HloModule scatter_zero_updates
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }
    scatter_computation {
      operand = f32[2,2]{1,0} parameter(0)
      indices = s32[0,1]{1,0} parameter(1)
      updates = f32[0,1,2]{2,1,0} parameter(2)
      ROOT s = f32[2,2]{1,0} scatter(operand, indices, updates),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=add
    }
    ENTRY main {
      p0 = f32[2,2]{1,0} parameter(0)
      p1 = s32[0,1]{1,0} parameter(1)
      p2 = f32[0,1,2]{2,1,0} parameter(2)
      ROOT wrapped_scatter = f32[2,2]{1,0} fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  constexpr absl::string_view kZeroSliceHlo = R"hlo(
    HloModule scatter_zero_slice
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }
    scatter_computation {
      operand = f32[2,2]{1,0} parameter(0)
      indices = s32[2,2]{1,0} parameter(1)
      updates = f32[2,1,0]{2,1,0} parameter(2)
      ROOT s = f32[2,2]{1,0} scatter(operand, indices, updates),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0,1},
        index_vector_dim=1,
        to_apply=add
    }
    ENTRY main {
      p0 = f32[2,2]{1,0} parameter(0)
      p1 = s32[2,2]{1,0} parameter(1)
      p2 = f32[2,1,0]{2,1,0} parameter(2)
      ROOT wrapped_scatter = f32[2,2]{1,0} fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  for (const auto& [name, hlo] :
       {std::pair<absl::string_view, absl::string_view>{"zero_updates",
                                                        kZeroUpdatesHlo},
        std::pair<absl::string_view, absl::string_view>{"zero_slice",
                                                        kZeroSliceHlo}}) {
    SCOPED_TRACE(name);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(hlo));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                         RunBufferAssignment(*module));
    const auto* fusion = Cast<HloFusionInstruction>(
        module->entry_computation()->root_instruction());
    std::unique_ptr<mlir::MLIRContext> mlir_context =
        FusionCompiler::CreateContext();
    CpuScatterFusion emitter(*buffer_assignment, fusion, mlir_context.get());
    ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> kernel_definition,
                         emitter.EmitKernelDefinition());
    EXPECT_THAT(RunFileCheck(kernel_definition.source().ToString(), R"(
      CHECK:     func.func @wrapped_scatter(
      CHECK-NOT: xla.loop
      CHECK:     return
    )"),
                absl_testing::IsOkAndHolds(true));
  }
}

TEST_F(CpuFusionEmitterTest, ScatterRankZeroOperandEmitsLoop) {
  constexpr absl::string_view kMultiUpdateScalarHlo = R"hlo(
    HloModule scatter_rank0_multi
    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }
    scatter_computation {
      operand = s32[] parameter(0)
      indices = s32[3,0]{1,0} parameter(1)
      updates = s32[3]{0} parameter(2)
      ROOT s = s32[] scatter(operand, indices, updates),
        update_window_dims={},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={},
        index_vector_dim=1,
        to_apply=add
    }
    ENTRY main {
      p0 = s32[] parameter(0)
      p1 = s32[3,0]{1,0} parameter(1)
      p2 = s32[3]{0} parameter(2)
      ROOT wrapped_scatter = s32[] fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kMultiUpdateScalarHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*module));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> kernel_definition,
      EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));
  EXPECT_THAT(RunFileCheck(kernel_definition.source().ToString(), R"(
    CHECK: func.func @wrapped_scatter(
    CHECK:   xla.loop
    CHECK:     tensor.extract %{{.+}}[] : tensor<i32>
    CHECK:     arith.addi
    CHECK:     tensor.insert %{{.+}} into %{{.+}}[] : tensor<i32>
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest, ScatterVectorSizeUsesWidestOperandOrUpdateType) {
  // Operand 0 is s8 (1 byte) and operand 1 is c128 (16 bytes). With a 64 byte
  // vectorization budget, max_vectorized_elements must be 64 / 16 = 4 (giving
  // vector_id in [0, 3], vector_element_id in [0, 3] for slice size 16), not 16
  // from s8 alone.
  constexpr absl::string_view kVariadicHlo = R"hlo(
    HloModule variadic_scatter_vector_size
    reducer {
      lhs0 = s8[] parameter(0)
      lhs1 = c128[] parameter(1)
      rhs0 = s8[] parameter(2)
      rhs1 = c128[] parameter(3)
      sum0 = s8[] add(lhs0, rhs0)
      sum1 = c128[] add(lhs1, rhs1)
      ROOT out = (s8[], c128[]) tuple(sum0, sum1)
    }
    scatter_computation {
      op0 = s8[4,16]{1,0} parameter(0)
      op1 = c128[4,16]{1,0} parameter(1)
      indices = s32[2,1]{1,0} parameter(2)
      upd0 = s8[2,1,16]{2,1,0} parameter(3)
      upd1 = c128[2,1,16]{2,1,0} parameter(4)
      ROOT s = (s8[4,16]{1,0}, c128[4,16]{1,0}) scatter(
          op0, op1, indices, upd0, upd1),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=reducer
    }
    ENTRY main {
      p0 = s8[4,16]{1,0} parameter(0)
      p1 = c128[4,16]{1,0} parameter(1)
      p2 = s32[2,1]{1,0} parameter(2)
      p3 = s8[2,1,16]{2,1,0} parameter(3)
      p4 = c128[2,1,16]{2,1,0} parameter(4)
      ROOT wrapped_scatter = (s8[4,16]{1,0}, c128[4,16]{1,0}) fusion(
          p0, p1, p2, p3, p4),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  // Mixed floating point precision where updates are wider than the operand:
  // operand is bf16 (2 bytes) and updates are f64 (8 bytes), so
  // max_vectorized_elements must be 64 / 8 = 8 (vector_id in [0, 1],
  // vector_element_id in [0, 7]), not 16 from bf16 alone.
  constexpr absl::string_view kWiderUpdateHlo = R"hlo(
    HloModule wider_update_scatter_vector_size
    reducer {
      lhs = bf16[] parameter(0)
      rhs = f64[] parameter(1)
      lhs_f64 = f64[] convert(lhs)
      sum = f64[] add(lhs_f64, rhs)
      ROOT out = bf16[] convert(sum)
    }
    scatter_computation {
      op = bf16[4,16]{1,0} parameter(0)
      indices = s32[2,1]{1,0} parameter(1)
      upd = f64[2,1,16]{2,1,0} parameter(2)
      ROOT s = bf16[4,16]{1,0} scatter(op, indices, upd),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=reducer
    }
    ENTRY main {
      p0 = bf16[4,16]{1,0} parameter(0)
      p1 = s32[2,1]{1,0} parameter(1)
      p2 = f64[2,1,16]{2,1,0} parameter(2)
      ROOT wrapped_scatter = bf16[4,16]{1,0} fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  struct VectorSizeCase {
    absl::string_view name;
    absl::string_view hlo;
    absl::string_view expected_domain;
  };
  for (const VectorSizeCase& tc : std::initializer_list<VectorSizeCase>{
           {"variadic_s8_c128", kVariadicHlo,
            "CHECK: domain: thread_id in [0, 0], index_id in [0, 1], "
            "vector_id in [0, 3], vector_element_id in [0, 3]"},
           {"wider_update_bf16_f64", kWiderUpdateHlo,
            "CHECK: domain: thread_id in [0, 0], index_id in [0, 1], "
            "vector_id in [0, 1], vector_element_id in [0, 7]"}}) {
    SCOPED_TRACE(tc.name);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(tc.hlo));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                         RunBufferAssignment(*module));
    const auto* fusion = Cast<HloFusionInstruction>(
        module->entry_computation()->root_instruction());
    std::unique_ptr<mlir::MLIRContext> mlir_context =
        FusionCompiler::CreateContext();
    CpuScatterFusion emitter(*buffer_assignment, fusion, mlir_context.get());
    ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> kernel_definition,
                         emitter.EmitKernelDefinition());
    EXPECT_THAT(
        RunFileCheck(kernel_definition.source().ToString(), tc.expected_domain),
        absl_testing::IsOkAndHolds(true));
  }
}

TEST_F(CpuFusionEmitterTest, ScatterEmitterRejectsUnsupportedScatters) {
  constexpr absl::string_view kNonSimplified1DHlo = R"hlo(
    HloModule non_simplified_1d
    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }
    scatter_computation {
      operand = s32[4]{0} parameter(0)
      indices = s32[2,1]{1,0} parameter(1)
      updates = s32[2]{0} parameter(2)
      ROOT s = s32[4]{0} scatter(operand, indices, updates),
        update_window_dims={},
        inserted_window_dims={0},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=add
    }
    ENTRY main {
      p0 = s32[4]{0} parameter(0)
      p1 = s32[2,1]{1,0} parameter(1)
      p2 = s32[2]{0} parameter(2)
      ROOT wrapped_scatter = s32[4]{0} fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  constexpr absl::string_view kUnsimplifiedScalarSingleUpdateHlo = R"hlo(
    HloModule unsimplified_scalar_single_update
    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }
    scatter_computation {
      operand = s32[] parameter(0)
      indices = s32[0]{0} parameter(1)
      updates = s32[] parameter(2)
      ROOT s = s32[] scatter(operand, indices, updates),
        update_window_dims={},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={},
        index_vector_dim=0,
        to_apply=add
    }
    ENTRY main {
      p0 = s32[] parameter(0)
      p1 = s32[0]{0} parameter(1)
      p2 = s32[] parameter(2)
      ROOT wrapped_scatter = s32[] fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  constexpr absl::string_view kBatchingDimsHlo = R"hlo(
    HloModule batching_dims_scatter
    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }
    scatter_computation {
      operand = s32[2,4]{1,0} parameter(0)
      indices = s32[2,1]{1,0} parameter(1)
      updates = s32[2,1]{1,0} parameter(2)
      ROOT s = s32[2,4]{1,0} scatter(operand, indices, updates),
        update_window_dims={1},
        inserted_window_dims={},
        input_batching_dims={0},
        scatter_indices_batching_dims={0},
        scatter_dims_to_operand_dims={1},
        index_vector_dim=1,
        to_apply=add
    }
    ENTRY main {
      p0 = s32[2,4]{1,0} parameter(0)
      p1 = s32[2,1]{1,0} parameter(1)
      p2 = s32[2,1]{1,0} parameter(2)
      ROOT wrapped_scatter = s32[2,4]{1,0} fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  constexpr absl::string_view kMismatchedVariadicDimsHlo = R"hlo(
    HloModule mismatched_variadic_dims
    reducer {
      lhs0 = s32[] parameter(0)
      lhs1 = s32[] parameter(1)
      rhs0 = s32[] parameter(2)
      rhs1 = s32[] parameter(3)
      sum0 = s32[] add(lhs0, rhs0)
      sum1 = s32[] add(lhs1, rhs1)
      ROOT out = (s32[], s32[]) tuple(sum0, sum1)
    }
    scatter_computation {
      op0 = s32[4]{0} parameter(0)
      op1 = s32[8]{0} parameter(1)
      indices = s32[2,1]{1,0} parameter(2)
      upd0 = s32[2,1]{1,0} parameter(3)
      upd1 = s32[2,1]{1,0} parameter(4)
      ROOT s = (s32[4]{0}, s32[8]{0}) scatter(op0, op1, indices, upd0, upd1),
        update_window_dims={1},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=reducer
    }
    ENTRY main {
      p0 = s32[4]{0} parameter(0)
      p1 = s32[8]{0} parameter(1)
      p2 = s32[2,1]{1,0} parameter(2)
      p3 = s32[2,1]{1,0} parameter(3)
      p4 = s32[2,1]{1,0} parameter(4)
      ROOT wrapped_scatter = (s32[4]{0}, s32[8]{0}) fusion(p0, p1, p2, p3, p4),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  constexpr absl::string_view kUnsupported1BitIntHlo = R"hlo(
    HloModule unsupported_1bit_int
    add {
      lhs = s1[] parameter(0)
      rhs = s1[] parameter(1)
      ROOT sum = s1[] add(lhs, rhs)
    }
    scatter_computation {
      operand = s1[4]{0:E(1)} parameter(0)
      indices = s32[2,1]{1,0} parameter(1)
      updates = s1[2,1]{1,0:E(1)} parameter(2)
      ROOT s = s1[4]{0:E(1)} scatter(operand, indices, updates),
        update_window_dims={1},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=add
    }
    ENTRY main {
      p0 = s1[4]{0:E(1)} parameter(0)
      p1 = s32[2,1]{1,0} parameter(1)
      p2 = s1[2,1]{1,0:E(1)} parameter(2)
      ROOT wrapped_scatter = s1[4]{0:E(1)} fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  constexpr absl::string_view kUnsupported1BitIntInToApplyHlo = R"hlo(
    HloModule unsupported_1bit_int_in_to_apply
    add_via_s1 {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      rhs_s1 = s1[] convert(rhs)
      rhs_s32 = s32[] convert(rhs_s1)
      ROOT sum = s32[] add(lhs, rhs_s32)
    }
    scatter_computation {
      operand = s32[4]{0} parameter(0)
      indices = s32[2,1]{1,0} parameter(1)
      updates = s32[2,1]{1,0} parameter(2)
      ROOT s = s32[4]{0} scatter(operand, indices, updates),
        update_window_dims={1},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=add_via_s1
    }
    ENTRY main {
      p0 = s32[4]{0} parameter(0)
      p1 = s32[2,1]{1,0} parameter(1)
      p2 = s32[2,1]{1,0} parameter(2)
      ROOT wrapped_scatter = s32[4]{0} fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  constexpr absl::string_view kUnsupported1BitIntFusedProducerHlo = R"hlo(
    HloModule unsupported_1bit_int_fused_producer
    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }
    scatter_computation {
      operand = s32[4]{0} parameter(0)
      indices = s32[2,1]{1,0} parameter(1)
      raw_updates = u1[2,1]{1,0:E(1)} parameter(2)
      updates = s32[2,1]{1,0} convert(raw_updates)
      ROOT s = s32[4]{0} scatter(operand, indices, updates),
        update_window_dims={1},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=add
    }
    ENTRY main {
      p0 = s32[4]{0} parameter(0)
      p1 = s32[2,1]{1,0} parameter(1)
      p2 = u1[2,1]{1,0:E(1)} parameter(2)
      ROOT wrapped_scatter = s32[4]{0} fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  constexpr absl::string_view kUnsupported1BitIntIndexHlo = R"hlo(
    HloModule unsupported_1bit_int_index
    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }
    scatter_computation {
      operand = s32[4]{0} parameter(0)
      indices = u1[2,1]{1,0:E(1)} parameter(1)
      updates = s32[2,1]{1,0} parameter(2)
      ROOT s = s32[4]{0} scatter(operand, indices, updates),
        update_window_dims={1},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=add
    }
    ENTRY main {
      p0 = s32[4]{0} parameter(0)
      p1 = u1[2,1]{1,0:E(1)} parameter(1)
      p2 = s32[2,1]{1,0} parameter(2)
      ROOT wrapped_scatter = s32[4]{0} fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  struct RejectCase {
    absl::string_view name;
    absl::string_view hlo;
    absl::StatusCode expected_code;
    absl::string_view expected_error_substr;
  };
  for (const RejectCase& tc : std::initializer_list<RejectCase>{
           {"non_simplified_1d", kNonSimplified1DHlo,
            absl::StatusCode::kInternal,
            "Non-simplified HLO Scatter is not supported."},
           {"unsimplified_scalar_single_update",
            kUnsimplifiedScalarSingleUpdateHlo, absl::StatusCode::kInternal,
            "Non-simplified HLO Scatter is not supported."},
           {"batching_dims", kBatchingDimsHlo, absl::StatusCode::kInternal,
            "Non-simplified HLO Scatter is not supported."},
           {"mismatched_variadic_dims", kMismatchedVariadicDimsHlo,
            absl::StatusCode::kInternal,
            "Variadic scatter operands must have matching dimensions."},
           {"unsupported_1bit_int", kUnsupported1BitIntHlo,
            absl::StatusCode::kUnimplemented, "1 bit packed integer type s1"},
           {"unsupported_1bit_int_in_to_apply", kUnsupported1BitIntInToApplyHlo,
            absl::StatusCode::kUnimplemented, "1 bit packed integer type s1"},
           {"unsupported_1bit_int_fused_producer",
            kUnsupported1BitIntFusedProducerHlo,
            absl::StatusCode::kUnimplemented, "1 bit packed integer type u1"},
           {"unsupported_1bit_int_index", kUnsupported1BitIntIndexHlo,
            absl::StatusCode::kUnimplemented,
            "1 bit packed integer type u1"}}) {
    SCOPED_TRACE(tc.name);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(tc.hlo));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                         RunBufferAssignment(*module));
    const auto* fusion = Cast<HloFusionInstruction>(
        module->entry_computation()->root_instruction());
    std::unique_ptr<mlir::MLIRContext> mlir_context =
        FusionCompiler::CreateContext();
    CpuScatterFusion emitter(*buffer_assignment, fusion, mlir_context.get());
    EXPECT_THAT(emitter.EmitKernelDefinition(),
                absl_testing::StatusIs(tc.expected_code,
                                       HasSubstr(tc.expected_error_substr)));
  }
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

TEST_F(CpuFusionEmitterTest, SortComparatorWeakAndTotalOrder) {
  constexpr absl::string_view kWeakSortHlo = R"hlo(
    HloModule sort_weak_module
    compare {
      lhs = bf16[] parameter(0)
      rhs = bf16[] parameter(1)
      ROOT lt = pred[] compare(lhs, rhs), direction=LT, order=WEAK
    }
    ENTRY main {
      p0 = bf16[8] parameter(0)
      ROOT sorted = bf16[8] sort(p0), dimensions={0}, to_apply=compare
    }
  )hlo";
  constexpr absl::string_view kExpectedWeakMlir = R"(
    CHECK:      func.func private @compare_lt(
    CHECK-DAG:    arith.cmpf ult
    CHECK-DAG:    arith.cmpf ord
    CHECK:        arith.andi
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> weak_module,
                       ParseAndReturnVerifiedModule(kWeakSortHlo));
  const HloComputation* weak_comparator =
      weak_module->entry_computation()->root_instruction()->to_apply();
  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      MlirKernelSource weak_mlir,
      EmitSortComparator(*mlir_context, *weak_comparator, "compare_weak"));
  EXPECT_THAT(RunFileCheck(weak_mlir.ToString(), kExpectedWeakMlir),
              absl_testing::IsOkAndHolds(true));

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource weak_llvm,
                       compiler.Compile(std::move(weak_mlir)));
  EXPECT_THAT(RunFileCheck(weak_llvm.ToString(),
                           "CHECK: define zeroext i1 @compare_weak("),
              absl_testing::IsOkAndHolds(true));

  constexpr absl::string_view kTotalSortHlo = R"hlo(
    HloModule sort_total_module
    compare {
      lhs = bf16[] parameter(0)
      rhs = bf16[] parameter(1)
      ROOT lt = pred[] compare(lhs, rhs), direction=LT, order=TOTAL
    }
    ENTRY main {
      p0 = bf16[8] parameter(0)
      ROOT sorted = bf16[8] sort(p0), dimensions={0}, to_apply=compare
    }
  )hlo";
  constexpr absl::string_view kExpectedTotalMlir = R"(
    CHECK:      func.func private @compare_lt(
    CHECK:        arith.bitcast %{{.+}} : bf16 to i16
    CHECK:        arith.cmpi slt
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> total_module,
                       ParseAndReturnVerifiedModule(kTotalSortHlo));
  const HloComputation* total_comparator =
      total_module->entry_computation()->root_instruction()->to_apply();
  ASSERT_OK_AND_ASSIGN(
      MlirKernelSource total_mlir,
      EmitSortComparator(*mlir_context, *total_comparator, "compare_total"));
  EXPECT_THAT(RunFileCheck(total_mlir.ToString(), kExpectedTotalMlir),
              absl_testing::IsOkAndHolds(true));
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource total_llvm,
                       compiler.Compile(std::move(total_mlir)));
  EXPECT_THAT(RunFileCheck(total_llvm.ToString(),
                           "CHECK: define zeroext i1 @compare_total("),
              absl_testing::IsOkAndHolds(true));
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

TEST_F(CpuFusionEmitterTest, ConcatenateMatchingLayoutEmitsMemcpy) {
  constexpr absl::string_view kHlo = R"(
    HloModule concat_module
    fused_computation {
      p0 = f32[4,2,8]{2,1,0} parameter(0)
      p1 = f32[4,3,8]{2,1,0} parameter(1)
      ROOT concat = f32[4,5,8]{2,1,0} concatenate(p0, p1), dimensions={1}
    }
    ENTRY main {
      p0 = f32[4,2,8]{2,1,0} parameter(0)
      p1 = f32[4,3,8]{2,1,0} parameter(1)
      ROOT wrapped_concat = f32[4,5,8]{2,1,0} fusion(p0, p1),
        kind=kLoop, calls=fused_computation,
        backend_config={"outer_dimension_partitions":["2"]}
    }
  )";
  constexpr absl::string_view kExpectedMlir = R"(
    CHECK:      func.func @wrapped_concat(
    CHECK-SAME:   xla.entry
    CHECK-NOT:    xla.loop
    CHECK:        scf.for
    CHECK:          xla.copy_slice {{.*}} count(16)
    CHECK:          xla.copy_slice {{.*}} count(24)
  )";
  std::string expected_llvm = absl::Substitute(R"(
    CHECK:      define ptr @wrapped_concat(
    CHECK-NOT:    call void @llvm.memcpy
    CHECK:        call void @llvm.assume(i1 true) [ "align"(ptr %{{.*}}, i64 $0) ]
    CHECK:      define internal void @wrapped_concat_wrapped(
    CHECK:        call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %{{.*}}, i64 64, i1 false)
    CHECK:        call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %{{.*}}, i64 96, i1 false)
  )",
                                               MinAlign());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  const auto* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> kernel_definition,
      EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));
  EXPECT_EQ(kernel_definition.spec().work_dimensions().num_work_groups.x, 2);
  EXPECT_THAT(
      RunFileCheck(kernel_definition.source().ToString(), kExpectedMlir),
      absl_testing::IsOkAndHolds(true));

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource llvm_source,
      compiler.Compile(std::move(kernel_definition).TakeSource()));
  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), expected_llvm),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest,
       ConcatenateNoOuterDimsForcesSingleWorkGroupAndEmitsMemcpy) {
  constexpr absl::string_view kHlo = R"(
    HloModule concat_module
    fused_computation {
      p0 = f32[8,16]{1,0} parameter(0)
      p1 = f32[12,16]{1,0} parameter(1)
      ROOT concat = f32[20,16]{1,0} concatenate(p0, p1), dimensions={0}
    }
    ENTRY main {
      p0 = f32[8,16]{1,0} parameter(0)
      p1 = f32[12,16]{1,0} parameter(1)
      ROOT wrapped_concat = f32[20,16]{1,0} fusion(p0, p1),
        kind=kLoop, calls=fused_computation,
        backend_config={"outer_dimension_partitions":["4"]}
    }
  )";
  std::string expected_llvm = absl::Substitute(R"(
    CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align $0 %{{.*}}, ptr align $0 %{{.*}}, i64 512, i1 false)
    CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align $0 %{{.*}}, ptr align $0 %{{.*}}, i64 768, i1 false)
  )",
                                               MinAlign());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  const auto* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> kernel_definition,
      EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));
  EXPECT_EQ(kernel_definition.spec().work_dimensions().num_work_groups.x, 1);

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource llvm_source,
      compiler.Compile(std::move(kernel_definition).TakeSource()));
  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), expected_llvm),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest, ConcatenateMismatchedLayoutFallsBackToLoop) {
  constexpr absl::string_view kHlo = R"(
    HloModule concat_module
    fused_computation {
      p0 = f32[4,8]{0,1} parameter(0)
      p1 = f32[6,8]{1,0} parameter(1)
      ROOT concat = f32[10,8]{1,0} concatenate(p0, p1), dimensions={0}
    }
    ENTRY main {
      p0 = f32[4,8]{0,1} parameter(0)
      p1 = f32[6,8]{1,0} parameter(1)
      ROOT wrapped_concat = f32[10,8]{1,0} fusion(p0, p1),
        kind=kLoop, calls=fused_computation
    }
  )";
  constexpr absl::string_view kExpectedMlir = R"(
    CHECK-NOT: xla.copy_slice
    CHECK:     xla.loop
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  const auto* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> kernel_definition,
      EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));
  EXPECT_THAT(
      RunFileCheck(kernel_definition.source().ToString(), kExpectedMlir),
      absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest,
       ConcatenateNonDefaultMatchingLayoutUsesPhysicalOrderMemcpy) {
  constexpr absl::string_view kHlo = R"(
    HloModule concat_module
    fused_computation {
      p0 = f32[2,4,3]{0,2,1} parameter(0)
      p1 = f32[2,4,5]{0,2,1} parameter(1)
      ROOT concat = f32[2,4,8]{0,2,1} concatenate(p0, p1), dimensions={2}
    }
    ENTRY main {
      p0 = f32[2,4,3]{0,2,1} parameter(0)
      p1 = f32[2,4,5]{0,2,1} parameter(1)
      ROOT wrapped_concat = f32[2,4,8]{0,2,1} fusion(p0, p1),
        kind=kLoop, calls=fused_computation,
        backend_config={"outer_dimension_partitions":["2"]}
    }
  )";
  constexpr absl::string_view kExpectedMlir = R"(
    CHECK:      func.func @wrapped_concat(
    CHECK-NOT:    xla.loop
    CHECK:        scf.for
    CHECK:          xla.copy_slice {{.*}} count(6)
    CHECK:          xla.copy_slice {{.*}} count(10)
  )";
  constexpr absl::string_view kExpectedLlvm = R"(
    CHECK: call void @llvm.memcpy{{.*}}i64 24
    CHECK: call void @llvm.memcpy{{.*}}i64 40
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  const auto* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> kernel_definition,
      EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));
  EXPECT_EQ(kernel_definition.spec().work_dimensions().num_work_groups.x, 2);
  EXPECT_THAT(
      RunFileCheck(kernel_definition.source().ToString(), kExpectedMlir),
      absl_testing::IsOkAndHolds(true));

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource llvm_source,
      compiler.Compile(std::move(kernel_definition).TakeSource()));
  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), kExpectedLlvm),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest, ConcatenateComplexAndNarrowFloatCompileToMemcpy) {
  for (absl::string_view elem_type : {"c64", "f6e3m2fn"}) {
    std::string hlo = absl::Substitute(R"(
      HloModule concat_module
      fused_computation {
        p0 = $0[4,3]{1,0} parameter(0)
        p1 = $0[4,5]{1,0} parameter(1)
        ROOT concat = $0[4,8]{1,0} concatenate(p0, p1), dimensions={1}
      }
      ENTRY main {
        p0 = $0[4,3]{1,0} parameter(0)
        p1 = $0[4,5]{1,0} parameter(1)
        ROOT wrapped_concat = $0[4,8]{1,0} fusion(p0, p1),
          kind=kLoop, calls=fused_computation
      }
    )",
                                       elem_type);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                         ParseAndReturnVerifiedModule(hlo));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                         RunBufferAssignment(*hlo_module));
    const auto* fusion = Cast<HloFusionInstruction>(
        hlo_module->entry_computation()->root_instruction());
    auto mlir_context = FusionCompiler::CreateContext();
    ASSERT_OK_AND_ASSIGN(
        KernelDefinition<MlirKernelSource> kernel_definition,
        EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                         /*use_unique_c_name=*/false,
                         /*enable_tiled_emitter=*/false));
    FusionCompiler compiler(mlir_context.get(),
                            FusionCompiler::Options{512, 1, true});
    ASSERT_OK_AND_ASSIGN(
        LlvmKernelSource llvm_source,
        compiler.Compile(std::move(kernel_definition).TakeSource()));
    EXPECT_THAT(
        RunFileCheck(llvm_source.ToString(),
                     elem_type == "c64"
                         ? "CHECK: call void @llvm.memcpy{{.*}}i64 24\nCHECK: "
                           "call void @llvm.memcpy{{.*}}i64 40"
                         : "CHECK: call void @llvm.memcpy{{.*}}i64 3\nCHECK: "
                           "call void @llvm.memcpy{{.*}}i64 5"),
        absl_testing::IsOkAndHolds(true));
  }
}

ABSL_ATTRIBUTE_NO_SANITIZE_CFI XLA_CPU_KernelError* CallCpuKernel(
    FunctionLibrary::Kernel* kernel, XLA_CPU_KernelCallFrame* frame) {
  return (*kernel)(frame);
}

TEST_F(CpuFusionEmitterTest,
       AtomicCasEmitsAlignedUnorderedLoadsAndJitExecutes) {
  constexpr absl::string_view kMlir = R"(
    module attributes {xla.cpu_memory_region_name = "cpu_fusion_emitter_test"} {
      func.func @cpu_atomic_cas(
          %s4_buf: tensor<4xi4> {llvm.align = 16 : index, xla.slice_index = 0},
          %f16_buf: tensor<2xf16> {llvm.align = 16 : index, xla.slice_index = 1},
          %f32_buf: tensor<2xf32> {llvm.align = 16 : index, xla.slice_index = 2},
          %c64_buf: tensor<2xcomplex<f32>> {llvm.align = 16 : index, xla.slice_index = 3})
          -> (tensor<4xi4>, tensor<2xf16>, tensor<2xf32>,
              tensor<2xcomplex<f32>>)
          attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %v_i4 = arith.constant 5 : i4
        %v_f16 = arith.constant 2.5 : f16
        %v_f32 = arith.constant 3.5 : f32
        %v_c64 = complex.constant [4.0 : f32, 6.0 : f32] : complex<f32>

        %out_s4 = xla.atomic_store %v_i4 into %s4_buf[%c1] : tensor<4xi4>
        %out_f16 = xla.atomic_rmw %f16_buf[%c1] : tensor<2xf16> {
          ^bb0(%cur: f16):
            %sum = arith.addf %cur, %v_f16 : f16
            xla.yield %sum : f16
        }
        %out_f32 = xla.atomic_rmw %f32_buf[%c0] : tensor<2xf32> {
          ^bb0(%cur: f32):
            %max = arith.maximumf %cur, %v_f32 : f32
            xla.yield %max : f32
        }
        %out_c64 = xla.atomic_rmw %c64_buf[%c1] : tensor<2xcomplex<f32>> {
          ^bb0(%cur: complex<f32>):
            %sum = complex.add %cur, %v_c64 : complex<f32>
            xla.yield %sum : complex<f32>
        }
        return %out_s4, %out_f16, %out_f32, %out_c64
            : tensor<4xi4>, tensor<2xf16>, tensor<2xf32>,
              tensor<2xcomplex<f32>>
      }
    }
  )";

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  mlir::MLIRContext* context_ptr = mlir_context.get();
  ASSERT_OK_AND_ASSIGN(
      MlirKernelSource mlir_source,
      MlirKernelSource::ParseFromString(kMlir, std::move(mlir_context)));
  FusionCompiler compiler(context_ptr, FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource llvm_source,
                       compiler.Compile(std::move(mlir_source)));

  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), R"(
    CHECK: load atomic i8, ptr %[[S4_ADDR:.*]] monotonic, align 1
    CHECK: cmpxchg ptr %[[S4_ADDR]], i8 %{{.*}}, i8 %{{.*}} monotonic monotonic
    CHECK: load atomic i16, ptr %[[F16_ADDR:.*]] monotonic, align 2
    CHECK: cmpxchg ptr %[[F16_ADDR]], i16 %{{.*}}, i16 %{{.*}} monotonic monotonic
    CHECK: load atomic i32, ptr %[[F32_ADDR:.*]] monotonic, align 4
    CHECK: cmpxchg ptr %[[F32_ADDR]], i32 %{{.*}}, i32 %{{.*}} monotonic monotonic
    CHECK: load atomic i64, ptr %[[C64_ADDR:.*]] monotonic, align 8
    CHECK: cmpxchg ptr %[[C64_ADDR]], i64 %{{.*}}, i64 %{{.*}} monotonic monotonic
  )"),
              absl_testing::IsOkAndHolds(true));

  std::unique_ptr<IrCompiler> ir_compiler =
      IrCompiler::Create({},
                         IrCompiler::Options{
                             /*opt_level=*/llvm::CodeGenOptLevel::Default,
                             /*optimize_for_size=*/false,
                             TargetMachineOptions(GetDebugOptionsFromFlags()),
                         },
                         IrCompiler::CompilationHooks());
  ASSERT_OK_AND_ASSIGN(JitCompiler jit,
                       JitCompiler::Create({}, std::move(ir_compiler)));
  ASSERT_OK(jit.AddModule(std::move(llvm_source).thread_safe_module()));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FunctionLibrary> library,
      std::move(jit).Compile(
          {FunctionLibrary::Sym<FunctionLibrary::Kernel>("cpu_atomic_cas")}));
  ASSERT_OK_AND_ASSIGN(
      FunctionLibrary::Kernel * kernel,
      library->ResolveFunction<FunctionLibrary::Kernel>("cpu_atomic_cas"));

  alignas(16) std::array<uint8_t, 2> s4_bytes = {0xa3, 0x72};
  alignas(16) std::array<half, 2> f16_vals = {half(1.0f), half(1.5f)};
  alignas(16) std::array<float, 2> f32_vals = {1.25f, 9.0f};
  alignas(16) std::array<complex64, 2> c64_vals = {complex64(1.0f, 2.0f),
                                                   complex64(3.0f, 5.0f)};
  std::vector<XLA_CPU_KernelArg> args = {
      {s4_bytes.data(), s4_bytes.size()},
      {f16_vals.data(), f16_vals.size() * sizeof(half)},
      {f32_vals.data(), f32_vals.size() * sizeof(float)},
      {c64_vals.data(), c64_vals.size() * sizeof(complex64)},
  };
  XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
  XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
  XLA_CPU_KernelCallFrame frame = {&num_workgroups, &workgroup_id, args.size(),
                                   args.data()};
  EXPECT_EQ(CallCpuKernel(kernel, &frame), nullptr);

  EXPECT_EQ(s4_bytes[0], 0x53);
  EXPECT_EQ(s4_bytes[1], 0x72);
  EXPECT_EQ(f16_vals[0], half(1.0f));
  EXPECT_EQ(f16_vals[1], half(4.0f));
  EXPECT_EQ(f32_vals[0], 3.5f);
  EXPECT_EQ(f32_vals[1], 9.0f);
  EXPECT_EQ(c64_vals[0], complex64(1.0f, 2.0f));
  EXPECT_EQ(c64_vals[1], complex64(7.0f, 11.0f));
}

TEST_F(CpuFusionEmitterTest,
       AtomicStoreContiguousLoopEmitsMonotonicAndExecutesInJit) {
  constexpr absl::string_view kMlir = R"(
    module attributes {xla.cpu_memory_region_name = "cpu_atomic_store_loop"} {
      func.func @cpu_atomic_store_loop(
          %src: tensor<128xf32> {llvm.align = 16 : index, xla.invariant, xla.slice_index = 0},
          %dst: tensor<128xf32> {llvm.align = 16 : index, xla.slice_index = 1})
          -> tensor<128xf32>
          attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c128 = arith.constant 128 : index
        %res = scf.for %i = %c0 to %c128 step %c1 iter_args(%cur = %dst) -> (tensor<128xf32>) {
          %v = tensor.extract %src[%i] : tensor<128xf32>
          %next = xla.atomic_store %v into %cur[%i] : tensor<128xf32>
          scf.yield %next : tensor<128xf32>
        }
        return %res : tensor<128xf32>
      }
    }
  )";

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  mlir::MLIRContext* context_ptr = mlir_context.get();
  ASSERT_OK_AND_ASSIGN(
      MlirKernelSource mlir_source,
      MlirKernelSource::ParseFromString(kMlir, std::move(mlir_context)));
  FusionCompiler compiler(context_ptr, FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource llvm_source,
                       compiler.Compile(std::move(mlir_source)));
  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), R"(
    CHECK: store atomic float %{{.*}}, ptr %{{.*}} monotonic, align 4
  )"),
              absl_testing::IsOkAndHolds(true));

  // Optimizing and JIT linking fails with an unresolved
  // __llvm_memcpy_element_unordered_atomic_4 symbol on x86_64 if unordered
  // ordering is used instead of monotonic.
  std::unique_ptr<IrCompiler> ir_compiler =
      IrCompiler::Create({},
                         IrCompiler::Options{
                             /*opt_level=*/llvm::CodeGenOptLevel::Default,
                             /*optimize_for_size=*/false,
                             TargetMachineOptions(GetDebugOptionsFromFlags()),
                         },
                         IrCompiler::CompilationHooks());
  ASSERT_OK_AND_ASSIGN(JitCompiler jit,
                       JitCompiler::Create({}, std::move(ir_compiler)));
  ASSERT_OK(jit.AddModule(std::move(llvm_source).thread_safe_module()));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FunctionLibrary> library,
      std::move(jit).Compile({FunctionLibrary::Sym<FunctionLibrary::Kernel>(
          "cpu_atomic_store_loop")}));
  ASSERT_OK_AND_ASSIGN(FunctionLibrary::Kernel * kernel,
                       library->ResolveFunction<FunctionLibrary::Kernel>(
                           "cpu_atomic_store_loop"));

  alignas(16) std::array<float, 128> src;
  alignas(16) std::array<float, 128> dst;
  for (int i = 0; i < 128; ++i) {
    src[i] = static_cast<float>(i * 3 + 1);
    dst[i] = -1.0f;
  }
  std::array<XLA_CPU_KernelArg, 2> args = {{
      {src.data(), sizeof(src)},
      {dst.data(), sizeof(dst)},
  }};
  XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
  XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
  XLA_CPU_KernelCallFrame frame = {&num_workgroups, &workgroup_id, args.size(),
                                   args.data()};
  EXPECT_EQ(CallCpuKernel(kernel, &frame), nullptr);
  EXPECT_EQ(dst, src);
}

constexpr absl::string_view kCpuAtomic128StoreMlir = R"(
  module attributes {xla.cpu_memory_region_name = "cpu_atomic_128_store"} {
    func.func @cpu_atomic_128_store(
        %i128_buf: tensor<4xi128> {llvm.align = 16 : index, xla.slice_index = 0},
        %i128_src: tensor<1xi128> {llvm.align = 16 : index, xla.invariant, xla.slice_index = 1},
        %c128_buf: tensor<4xcomplex<f64>> {llvm.align = 16 : index, xla.slice_index = 2},
        %c128_src: tensor<1xcomplex<f64>> {llvm.align = 16 : index, xla.invariant, xla.slice_index = 3})
        -> (tensor<4xi128>, tensor<4xcomplex<f64>>)
        attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %v_i128 = tensor.extract %i128_src[%c0] : tensor<1xi128>
      %v_c128 = tensor.extract %c128_src[%c0] : tensor<1xcomplex<f64>>
      %out_i128 = xla.atomic_store %v_i128 into %i128_buf[%c1] : tensor<4xi128>
      %out_c128 = xla.atomic_store %v_c128 into %c128_buf[%c2] : tensor<4xcomplex<f64>>
      return %out_i128, %out_c128 : tensor<4xi128>, tensor<4xcomplex<f64>>
    }
  }
)";

constexpr absl::string_view kCpuAtomic128RmwMlir = R"(
  module attributes {xla.cpu_memory_region_name = "cpu_atomic_128_rmw"} {
    func.func @cpu_atomic_128_rmw(
        %i128_buf: tensor<4xi128> {llvm.align = 16 : index, xla.slice_index = 0},
        %i128_delta: tensor<1xi128> {llvm.align = 16 : index, xla.invariant, xla.slice_index = 1},
        %c128_buf: tensor<4xcomplex<f64>> {llvm.align = 16 : index, xla.slice_index = 2},
        %c128_delta: tensor<1xcomplex<f64>> {llvm.align = 16 : index, xla.invariant, xla.slice_index = 3})
        -> (tensor<4xi128>, tensor<4xcomplex<f64>>)
        attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %d_i128 = tensor.extract %i128_delta[%c0] : tensor<1xi128>
      %d_c128 = tensor.extract %c128_delta[%c0] : tensor<1xcomplex<f64>>
      %out_i128 = xla.atomic_rmw %i128_buf[%c1] : tensor<4xi128> {
        ^bb0(%cur: i128):
          %sum = arith.addi %cur, %d_i128 : i128
          xla.yield %sum : i128
      }
      %out_c128 = xla.atomic_rmw %c128_buf[%c2] : tensor<4xcomplex<f64>> {
        ^bb0(%cur: complex<f64>):
          %sum = complex.add %cur, %d_c128 : complex<f64>
          xla.yield %sum : complex<f64>
      }
      return %out_i128, %out_c128 : tensor<4xi128>, tensor<4xcomplex<f64>>
    }
  }
)";

struct CompiledAtomic128Artifacts {
  std::string llvm_ir;
  std::string optimized_llvm_ir;
  std::string assembly;
  std::vector<std::string> undefined_symbols;
  std::vector<std::string> helper_symbols;
};

absl::StatusOr<LlvmKernelSource> LowerAtomic128ForTarget(
    absl::string_view mlir_text, const llvm::TargetMachine& tm) {
  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  ABSL_ASSIGN_OR_RETURN(
      MlirKernelSource mlir_source,
      MlirKernelSource::ParseFromString(mlir_text, std::move(mlir_context)));

  FusionCompiler::Options options;
  options.vector_width = 256;
  options.verification_level = 1;
  options.fast_min_max = true;
  options.cpu_features = tm.getTargetFeatureString().str();
  options.cpu_device_spec = CpuDeviceSpecFromTargetMachine(&tm);

  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  FusionCompiler compiler(mlir_source.module().getContext(),
                          std::move(options));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<llvm::Module> llvm_module,
                        compiler.Compile(*llvm_context, mlir_source.module()));
  return LlvmKernelSource(std::move(llvm_context), std::move(llvm_module));
}

absl::StatusOr<CompiledAtomic128Artifacts> CompileLlvmModuleForTarget(
    LlvmKernelSource llvm_source, llvm::TargetMachine& tm,
    absl::string_view triple_str, absl::string_view cpu,
    absl::string_view features, llvm::CodeGenOptLevel opt_level) {
  CompiledAtomic128Artifacts out;
  out.llvm_ir = llvm_source.ToString();

  std::unique_ptr<llvm::Module> optimized_clone;
  IrCompiler::CompilationHooks hooks;
  hooks.post_optimization = [&](const llvm::Module& m) {
    optimized_clone = llvm::CloneModule(m);
  };
  std::string triple_copy(triple_str);
  std::string cpu_copy(cpu);
  std::string features_copy(features);
  IrCompiler ir_compiler(
      [triple_copy, cpu_copy, features_copy, opt_level]() {
        return TargetMachineTestBase::CreateTargetMachineWithOptions(
            triple_copy, cpu_copy, features_copy, llvm::Reloc::PIC_, opt_level);
      },
      IrCompiler::Options{
          opt_level,
          /*optimize_for_size=*/false,
          TargetMachineOptions(triple_str, cpu, features),
      },
      std::move(hooks));

  llvm::orc::ThreadSafeModule tsm = std::move(llvm_source).thread_safe_module();
  llvm::Module* module = tsm.getModuleUnlocked();
  module->setTargetTriple(tm.getTargetTriple());
  module->setDataLayout(tm.createDataLayout());

  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> obj_buf =
      ir_compiler(*module);
  if (!obj_buf) {
    return absl::InternalError(llvm::toString(obj_buf.takeError()));
  }
  llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> obj_file =
      llvm::object::ObjectFile::createObjectFile((*obj_buf)->getMemBufferRef());
  if (!obj_file) {
    return absl::InternalError(llvm::toString(obj_file.takeError()));
  }
  for (const llvm::object::SymbolRef& sym : (*obj_file)->symbols()) {
    llvm::Expected<uint32_t> flags = sym.getFlags();
    if (!flags) {
      return absl::InternalError(llvm::toString(flags.takeError()));
    }
    llvm::Expected<llvm::StringRef> name = sym.getName();
    if (!name) {
      return absl::InternalError(llvm::toString(name.takeError()));
    }
    if (name->empty()) {
      continue;
    }
    if ((*flags & llvm::object::SymbolRef::SF_Undefined) != 0) {
      out.undefined_symbols.push_back(name->str());
    }
    if (name->contains("__atomic_") || name->contains("__aarch64_") ||
        name->contains("__sync_") ||
        name->contains("__llvm_memcpy_element_unordered_atomic")) {
      out.helper_symbols.push_back(name->str());
    }
  }

  TF_RET_CHECK(optimized_clone != nullptr);
  {
    llvm::raw_string_ostream ir_stream(out.optimized_llvm_ir);
    optimized_clone->print(ir_stream, nullptr);
  }
  llvm::SmallVector<char, 0> asm_chars;
  llvm::raw_svector_ostream asm_stream(asm_chars);
  llvm::legacy::PassManager asm_pm;
  TF_RET_CHECK(!tm.addPassesToEmitFile(asm_pm, asm_stream, nullptr,
                                       llvm::CodeGenFileType::AssemblyFile));
  asm_pm.run(*optimized_clone);
  optimized_clone.reset();
  out.assembly.assign(asm_chars.data(), asm_chars.size());
  return out;
}

absl::StatusOr<CompiledAtomic128Artifacts> CompileAtomic128ForTarget(
    absl::string_view mlir_text, absl::string_view triple_str,
    absl::string_view cpu, absl::string_view features,
    llvm::CodeGenOptLevel opt_level = llvm::CodeGenOptLevel::Default) {
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<llvm::TargetMachine> tm,
      TargetMachineTestBase::CreateTargetMachineWithOptions(
          triple_str, cpu, features, llvm::Reloc::PIC_, opt_level));
  ABSL_ASSIGN_OR_RETURN(LlvmKernelSource llvm_source,
                        LowerAtomic128ForTarget(mlir_text, *tm));
  return CompileLlvmModuleForTarget(std::move(llvm_source), *tm, triple_str,
                                    cpu, features, opt_level);
}

::testing::AssertionResult HasNoAtomicRuntimeHelpers(
    const CompiledAtomic128Artifacts& artifacts) {
  if (!artifacts.undefined_symbols.empty()) {
    return ::testing::AssertionFailure()
           << "unexpected undefined symbols: "
           << absl::StrJoin(artifacts.undefined_symbols, ", ") << "\n"
           << artifacts.assembly;
  }
  if (!artifacts.helper_symbols.empty()) {
    return ::testing::AssertionFailure()
           << "unexpected atomic helper symbols: "
           << absl::StrJoin(artifacts.helper_symbols, ", ") << "\n"
           << artifacts.assembly;
  }
  for (absl::string_view prefix : {"__atomic_", "__aarch64_", "__sync_",
                                   "__llvm_memcpy_element_unordered_atomic"}) {
    if (absl::StrContains(artifacts.assembly, prefix)) {
      return ::testing::AssertionFailure()
             << "unexpected runtime helper prefix '" << prefix
             << "' in assembly:\n"
             << artifacts.assembly;
    }
  }
  return ::testing::AssertionSuccess();
}

TEST_F(CpuFusionEmitterTest, Atomic128X64Codegen) {
  constexpr absl::string_view kTriple = "x86_64-unknown-linux-gnu";

  // x86_64 with +cx16 (without +avx): both store and RMW lower via 16 byte
  // aligned CAS loops to native lock cmpxchg16b with zero libcalls.
  for (llvm::CodeGenOptLevel opt :
       {llvm::CodeGenOptLevel::None, llvm::CodeGenOptLevel::Default}) {
    SCOPED_TRACE(absl::StrCat("cx16 opt=", static_cast<int>(opt)));
    ASSERT_OK_AND_ASSIGN(
        CompiledAtomic128Artifacts store_cx16,
        CompileAtomic128ForTarget(kCpuAtomic128StoreMlir, kTriple, "x86-64",
                                  "+cx16", opt));
    EXPECT_TRUE(HasNoAtomicRuntimeHelpers(store_cx16));
    EXPECT_THAT(RunFileCheck(store_cx16.llvm_ir, R"(
      CHECK: load atomic i128, ptr %[[I_ADDR:.*]] monotonic, align 16
      CHECK: cmpxchg ptr %[[I_ADDR]], i128 %{{.*}}, i128 %{{.*}} monotonic monotonic, align 16
      CHECK: load atomic i128, ptr %[[C_ADDR:.*]] monotonic, align 16
      CHECK: cmpxchg ptr %[[C_ADDR]], i128 %{{.*}}, i128 %{{.*}} monotonic monotonic, align 16
    )"),
                absl_testing::IsOkAndHolds(true));
    EXPECT_THAT(store_cx16.assembly,
                ContainsRegex("lock[[:space:]]+cmpxchg16b"));
    EXPECT_THAT(store_cx16.assembly, Not(HasSubstr("vmovdqa")));

    ASSERT_OK_AND_ASSIGN(
        CompiledAtomic128Artifacts rmw_cx16,
        CompileAtomic128ForTarget(kCpuAtomic128RmwMlir, kTriple, "x86-64",
                                  "+cx16", opt));
    EXPECT_TRUE(HasNoAtomicRuntimeHelpers(rmw_cx16));
    EXPECT_THAT(rmw_cx16.assembly, ContainsRegex("lock[[:space:]]+cmpxchg16b"));
  }

  // x86_64 with +cx16,+avx (explicit and implied by haswell and
  // skylake-avx512): store lowers to direct vmovdqa (no cmpxchg16b), and RMW
  // uses vmovdqa for the initial load plus lock cmpxchg16b for the CAS update.
  for (const auto& [cpu, features] :
       std::vector<std::pair<absl::string_view, absl::string_view>>{
           {"x86-64", "+cx16,+avx"}, {"haswell", ""}, {"skylake-avx512", ""}}) {
    for (llvm::CodeGenOptLevel opt :
         {llvm::CodeGenOptLevel::None, llvm::CodeGenOptLevel::Default}) {
      SCOPED_TRACE(absl::StrCat("cpu=", cpu, " features=", features,
                                " opt=", static_cast<int>(opt)));
      ASSERT_OK_AND_ASSIGN(
          CompiledAtomic128Artifacts store_avx,
          CompileAtomic128ForTarget(kCpuAtomic128StoreMlir, kTriple, cpu,
                                    features, opt));
      EXPECT_TRUE(HasNoAtomicRuntimeHelpers(store_avx));
      EXPECT_THAT(RunFileCheck(store_avx.llvm_ir, R"(
        CHECK-NOT: cmpxchg
        CHECK: store atomic i128 %{{.*}}, ptr %{{.*}} monotonic, align 16
        CHECK-NOT: cmpxchg
        CHECK: store atomic i128 %{{.*}}, ptr %{{.*}} monotonic, align 16
        CHECK-NOT: cmpxchg
      )"),
                  absl_testing::IsOkAndHolds(true));
      // SelectionDAG emits vmovdqa for 16 byte aligned 128 bit atomic stores;
      // at O2 X86ExecutionDomainFix may replace a pure copy store with the
      // equivalent aligned vmovaps/vmovapd encoding.
      if (opt == llvm::CodeGenOptLevel::None) {
        EXPECT_THAT(store_avx.assembly,
                    ContainsRegex("vmovdqa(32|64)?[[:space:]]+%xmm[0-9]+,"));
      }
      EXPECT_THAT(store_avx.assembly,
                  ContainsRegex(
                      "vmov(dqa(32|64)?|ap[sd])[[:space:]]+%xmm[0-9]+, 16\\("));
      EXPECT_THAT(store_avx.assembly,
                  ContainsRegex(
                      "vmov(dqa(32|64)?|ap[sd])[[:space:]]+%xmm[0-9]+, 32\\("));
      EXPECT_THAT(store_avx.assembly,
                  Not(ContainsRegex(
                      "vmov(dqu(8|16|32|64)?|up[sd])[[:space:]]+%xmm[0-9]+,")));
      EXPECT_THAT(store_avx.assembly, Not(HasSubstr("cmpxchg16b")));

      ASSERT_OK_AND_ASSIGN(
          CompiledAtomic128Artifacts rmw_avx,
          CompileAtomic128ForTarget(kCpuAtomic128RmwMlir, kTriple, cpu,
                                    features, opt));
      EXPECT_TRUE(HasNoAtomicRuntimeHelpers(rmw_avx));
      EXPECT_THAT(RunFileCheck(rmw_avx.llvm_ir, R"(
        CHECK: load atomic i128, ptr %[[I_ADDR:.*]] monotonic, align 16
        CHECK: cmpxchg ptr %[[I_ADDR]], i128 %{{.*}}, i128 %{{.*}} monotonic monotonic, align 16
        CHECK: load atomic i128, ptr %[[C_ADDR:.*]] monotonic, align 16
        CHECK: cmpxchg ptr %[[C_ADDR]], i128 %{{.*}}, i128 %{{.*}} monotonic monotonic, align 16
      )"),
                  absl_testing::IsOkAndHolds(true));
      EXPECT_THAT(rmw_avx.assembly, HasSubstr("vmovdqa"));
      EXPECT_THAT(rmw_avx.assembly,
                  ContainsRegex("lock[[:space:]]+cmpxchg16b"));
    }
  }

  // x86_64 with -cx16 (with or without +avx): cleanly rejected during lowering
  // rather than emitting __atomic_*_16 libcalls.
  for (absl::string_view features :
       {absl::string_view("-cx16"), absl::string_view("-cx16,+avx")}) {
    SCOPED_TRACE(features);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<llvm::TargetMachine> tm_no_cx16,
                         TargetMachineTestBase::CreateTargetMachineWithOptions(
                             kTriple, "haswell", features, llvm::Reloc::PIC_,
                             llvm::CodeGenOptLevel::Default));
    EXPECT_THAT(LowerAtomic128ForTarget(kCpuAtomic128StoreMlir, *tm_no_cx16),
                absl_testing::StatusIs(
                    absl::StatusCode::kUnknown,
                    HasSubstr("failed to lower to LLVM atomic operation")));
    EXPECT_THAT(LowerAtomic128ForTarget(kCpuAtomic128RmwMlir, *tm_no_cx16),
                absl_testing::StatusIs(
                    absl::StatusCode::kUnknown,
                    HasSubstr("failed to lower to LLVM atomic operation")));
  }
}

TEST_F(CpuFusionEmitterTest, Atomic128AArch64Codegen) {
  constexpr absl::string_view kTriple = "aarch64-unknown-linux-gnu";

  // Base AArch64 (no +lse, no +lse2, no +outline-atomics): lowers via native
  // ldxp/stxp loops with zero helper calls.
  for (llvm::CodeGenOptLevel opt :
       {llvm::CodeGenOptLevel::None, llvm::CodeGenOptLevel::Default}) {
    SCOPED_TRACE(absl::StrCat("base opt=", static_cast<int>(opt)));
    ASSERT_OK_AND_ASSIGN(
        CompiledAtomic128Artifacts store_base,
        CompileAtomic128ForTarget(kCpuAtomic128StoreMlir, kTriple, "generic",
                                  "", opt));
    EXPECT_TRUE(HasNoAtomicRuntimeHelpers(store_base));
    EXPECT_THAT(store_base.assembly, HasSubstr("ldxp"));
    EXPECT_THAT(store_base.assembly, HasSubstr("stxp"));
    EXPECT_THAT(store_base.assembly, Not(HasSubstr("casp")));

    ASSERT_OK_AND_ASSIGN(
        CompiledAtomic128Artifacts rmw_base,
        CompileAtomic128ForTarget(kCpuAtomic128RmwMlir, kTriple, "generic", "",
                                  opt));
    EXPECT_TRUE(HasNoAtomicRuntimeHelpers(rmw_base));
    EXPECT_THAT(rmw_base.assembly, HasSubstr("ldxp"));
    EXPECT_THAT(rmw_base.assembly, HasSubstr("stxp"));
    EXPECT_THAT(rmw_base.assembly, Not(HasSubstr("casp")));
  }

  // AArch64 with +lse (including +lse,+outline-atomics and neoverse-n1): lowers
  // inline to casp with zero __aarch64_cas16_* calls.
  for (const auto& [cpu, features] :
       std::vector<std::pair<absl::string_view, absl::string_view>>{
           {"generic", "+lse"},
           {"generic", "+lse,+outline-atomics"},
           {"neoverse-n1", ""}}) {
    for (llvm::CodeGenOptLevel opt :
         {llvm::CodeGenOptLevel::None, llvm::CodeGenOptLevel::Default}) {
      SCOPED_TRACE(absl::StrCat("cpu=", cpu, " features=", features,
                                " opt=", static_cast<int>(opt)));
      ASSERT_OK_AND_ASSIGN(
          CompiledAtomic128Artifacts store_lse,
          CompileAtomic128ForTarget(kCpuAtomic128StoreMlir, kTriple, cpu,
                                    features, opt));
      EXPECT_TRUE(HasNoAtomicRuntimeHelpers(store_lse));
      EXPECT_THAT(store_lse.assembly, HasSubstr("casp"));
      EXPECT_THAT(store_lse.assembly, Not(HasSubstr("ldxp")));
      EXPECT_THAT(store_lse.assembly, Not(HasSubstr("stxp")));

      ASSERT_OK_AND_ASSIGN(
          CompiledAtomic128Artifacts rmw_lse,
          CompileAtomic128ForTarget(kCpuAtomic128RmwMlir, kTriple, cpu,
                                    features, opt));
      EXPECT_TRUE(HasNoAtomicRuntimeHelpers(rmw_lse));
      EXPECT_THAT(rmw_lse.assembly, HasSubstr("casp"));
      EXPECT_THAT(rmw_lse.assembly, Not(HasSubstr("ldxp")));
      EXPECT_THAT(rmw_lse.assembly, Not(HasSubstr("stxp")));
    }
  }

  constexpr absl::string_view kNonSpPairStoreRegex =
      R"(stp[[:space:]]+x[0-9]+,[[:space:]]*x[0-9]+,[[:space:]]*\[x[0-9]+(,[[:space:]]*#[0-9]+)?\])";
  constexpr absl::string_view kNonSpPairLoadRegex =
      R"(ldp[[:space:]]+x[0-9]+,[[:space:]]*x[0-9]+,[[:space:]]*\[x[0-9]+(,[[:space:]]*#[0-9]+)?\])";

  // AArch64 with +lse,+lse2 (including +outline-atomics and neoverse-v2):
  // atomic_store lowers to single copy atomic stp (no casp or ldxp/stxp), and
  // atomic_rmw uses ldp for the initial load plus casp for the CAS update.
  for (const auto& [cpu, features] :
       std::vector<std::pair<absl::string_view, absl::string_view>>{
           {"generic", "+lse,+lse2"},
           {"generic", "+lse,+lse2,+outline-atomics"},
           {"neoverse-v2", ""}}) {
    for (llvm::CodeGenOptLevel opt :
         {llvm::CodeGenOptLevel::None, llvm::CodeGenOptLevel::Default}) {
      SCOPED_TRACE(absl::StrCat("cpu=", cpu, " features=", features,
                                " opt=", static_cast<int>(opt)));
      ASSERT_OK_AND_ASSIGN(
          CompiledAtomic128Artifacts store_lse2,
          CompileAtomic128ForTarget(kCpuAtomic128StoreMlir, kTriple, cpu,
                                    features, opt));
      EXPECT_TRUE(HasNoAtomicRuntimeHelpers(store_lse2));
      EXPECT_THAT(RunFileCheck(store_lse2.llvm_ir, R"(
        CHECK-NOT: cmpxchg
        CHECK: store atomic i128 %{{.*}}, ptr %{{.*}} monotonic, align 16
        CHECK-NOT: cmpxchg
        CHECK: store atomic i128 %{{.*}}, ptr %{{.*}} monotonic, align 16
        CHECK-NOT: cmpxchg
      )"),
                  absl_testing::IsOkAndHolds(true));
      EXPECT_THAT(store_lse2.assembly, ContainsRegex(kNonSpPairStoreRegex));
      EXPECT_THAT(store_lse2.assembly, Not(HasSubstr("casp")));
      EXPECT_THAT(store_lse2.assembly, Not(HasSubstr("ldxp")));
      EXPECT_THAT(store_lse2.assembly, Not(HasSubstr("stxp")));

      ASSERT_OK_AND_ASSIGN(
          CompiledAtomic128Artifacts rmw_lse2,
          CompileAtomic128ForTarget(kCpuAtomic128RmwMlir, kTriple, cpu,
                                    features, opt));
      EXPECT_TRUE(HasNoAtomicRuntimeHelpers(rmw_lse2));
      EXPECT_THAT(RunFileCheck(rmw_lse2.llvm_ir, R"(
        CHECK: load atomic i128, ptr %[[I_ADDR:.*]] monotonic, align 16
        CHECK: cmpxchg ptr %[[I_ADDR]], i128 %{{.*}}, i128 %{{.*}} monotonic monotonic, align 16
        CHECK: load atomic i128, ptr %[[C_ADDR:.*]] monotonic, align 16
        CHECK: cmpxchg ptr %[[C_ADDR]], i128 %{{.*}}, i128 %{{.*}} monotonic monotonic, align 16
      )"),
                  absl_testing::IsOkAndHolds(true));
      EXPECT_THAT(rmw_lse2.assembly, ContainsRegex(kNonSpPairLoadRegex));
      EXPECT_THAT(rmw_lse2.assembly, HasSubstr("casp"));
      EXPECT_THAT(rmw_lse2.assembly, Not(HasSubstr("ldxp")));
      EXPECT_THAT(rmw_lse2.assembly, Not(HasSubstr("stxp")));
    }
  }

  // AArch64 with +outline-atomics and without +lse/+lse2: cleanly rejected so
  // no unresolved __aarch64_cas16_* symbol can be emitted.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<llvm::TargetMachine> tm_outline,
                       TargetMachineTestBase::CreateTargetMachineWithOptions(
                           kTriple, "generic", "+outline-atomics",
                           llvm::Reloc::PIC_, llvm::CodeGenOptLevel::Default));
  EXPECT_THAT(LowerAtomic128ForTarget(kCpuAtomic128StoreMlir, *tm_outline),
              absl_testing::StatusIs(
                  absl::StatusCode::kUnknown,
                  HasSubstr("failed to lower to LLVM atomic operation")));
  EXPECT_THAT(LowerAtomic128ForTarget(kCpuAtomic128RmwMlir, *tm_outline),
              absl_testing::StatusIs(
                  absl::StatusCode::kUnknown,
                  HasSubstr("failed to lower to LLVM atomic operation")));

  // AArch64 with +lse2,+outline-atomics (no +lse): direct store lowers to
  // native stp without helper calls, while RMW is cleanly rejected.
  for (llvm::CodeGenOptLevel opt :
       {llvm::CodeGenOptLevel::None, llvm::CodeGenOptLevel::Default}) {
    SCOPED_TRACE(absl::StrCat("lse2_outline opt=", static_cast<int>(opt)));
    ASSERT_OK_AND_ASSIGN(
        CompiledAtomic128Artifacts store_lse2_outline,
        CompileAtomic128ForTarget(kCpuAtomic128StoreMlir, kTriple, "generic",
                                  "+lse2,+outline-atomics", opt));
    EXPECT_TRUE(HasNoAtomicRuntimeHelpers(store_lse2_outline));
    EXPECT_THAT(RunFileCheck(store_lse2_outline.llvm_ir, R"(
      CHECK-NOT: cmpxchg
      CHECK: store atomic i128 %{{.*}}, ptr %{{.*}} monotonic, align 16
      CHECK-NOT: cmpxchg
      CHECK: store atomic i128 %{{.*}}, ptr %{{.*}} monotonic, align 16
      CHECK-NOT: cmpxchg
    )"),
                absl_testing::IsOkAndHolds(true));
    EXPECT_THAT(store_lse2_outline.assembly,
                ContainsRegex(kNonSpPairStoreRegex));
    EXPECT_THAT(store_lse2_outline.assembly, Not(HasSubstr("casp")));

    ASSERT_OK_AND_ASSIGN(std::unique_ptr<llvm::TargetMachine> tm_lse2_outline,
                         TargetMachineTestBase::CreateTargetMachineWithOptions(
                             kTriple, "generic", "+lse2,+outline-atomics",
                             llvm::Reloc::PIC_, opt));
    EXPECT_THAT(LowerAtomic128ForTarget(kCpuAtomic128RmwMlir, *tm_lse2_outline),
                absl_testing::StatusIs(
                    absl::StatusCode::kUnknown,
                    HasSubstr("failed to lower to LLVM atomic operation")));
  }
}

TEST_F(CpuFusionEmitterTest, Atomic128RejectsUnderalignedC128OnCapableCpu) {
  constexpr absl::string_view kUnderalignedStore = R"(
    module attributes {xla.cpu_memory_region_name = "c128_align8_store"} {
      func.func @c128_align8_store(
          %buf: tensor<4xcomplex<f64>> {llvm.align = 8 : index, xla.slice_index = 0})
          -> tensor<4xcomplex<f64>>
          attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
        %c0 = arith.constant 0 : index
        %v = complex.constant [1.0 : f64, 2.0 : f64] : complex<f64>
        %out = xla.atomic_store %v into %buf[%c0] : tensor<4xcomplex<f64>>
        return %out : tensor<4xcomplex<f64>>
      }
    }
  )";
  constexpr absl::string_view kUnderalignedRmw = R"(
    module attributes {xla.cpu_memory_region_name = "c128_align8_rmw"} {
      func.func @c128_align8_rmw(
          %buf: tensor<4xcomplex<f64>> {llvm.align = 8 : index, xla.slice_index = 0})
          -> tensor<4xcomplex<f64>>
          attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
        %c0 = arith.constant 0 : index
        %v = complex.constant [1.0 : f64, 2.0 : f64] : complex<f64>
        %out = xla.atomic_rmw %buf[%c0] : tensor<4xcomplex<f64>> {
          ^bb0(%cur: complex<f64>):
            %sum = complex.add %cur, %v : complex<f64>
            xla.yield %sum : complex<f64>
        }
        return %out : tensor<4xcomplex<f64>>
      }
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<llvm::TargetMachine> tm,
                       TargetMachineTestBase::CreateTargetMachineWithOptions(
                           "x86_64-unknown-linux-gnu", "haswell", "",
                           llvm::Reloc::PIC_, llvm::CodeGenOptLevel::Default));
  EXPECT_THAT(LowerAtomic128ForTarget(kUnderalignedStore, *tm),
              absl_testing::StatusIs(
                  absl::StatusCode::kUnknown,
                  HasSubstr("failed to lower to LLVM atomic operation")));
  EXPECT_THAT(LowerAtomic128ForTarget(kUnderalignedRmw, *tm),
              absl_testing::StatusIs(
                  absl::StatusCode::kUnknown,
                  HasSubstr("failed to lower to LLVM atomic operation")));
}

TEST_F(CpuFusionEmitterTest, Atomic128HostJitExecutesC128StoreAndAdd) {
  std::unique_ptr<IrCompiler> ir_compiler =
      IrCompiler::Create({},
                         IrCompiler::Options{
                             /*opt_level=*/llvm::CodeGenOptLevel::Default,
                             /*optimize_for_size=*/false,
                             TargetMachineOptions(GetDebugOptionsFromFlags()),
                         },
                         IrCompiler::CompilationHooks());
  ASSERT_OK_AND_ASSIGN(JitCompiler jit,
                       JitCompiler::Create({}, std::move(ir_compiler)));
  CpuDeviceSpec host_spec =
      CpuDeviceSpecFromTargetMachine(jit.target_machine());
  if (!host_spec.has_atomic_cmpxchg_128) {
    GTEST_SKIP() << "Host CPU does not support native 128 bit atomic CAS.";
  }

  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource store_llvm,
      LowerAtomic128ForTarget(kCpuAtomic128StoreMlir, *jit.target_machine()));
  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource rmw_llvm,
      LowerAtomic128ForTarget(kCpuAtomic128RmwMlir, *jit.target_machine()));
  ASSERT_OK(jit.AddModule(std::move(store_llvm).thread_safe_module()));
  ASSERT_OK(jit.AddModule(std::move(rmw_llvm).thread_safe_module()));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FunctionLibrary> library,
      std::move(jit).Compile({FunctionLibrary::Sym<FunctionLibrary::Kernel>(
                                  "cpu_atomic_128_store"),
                              FunctionLibrary::Sym<FunctionLibrary::Kernel>(
                                  "cpu_atomic_128_rmw")}));
  ASSERT_OK_AND_ASSIGN(FunctionLibrary::Kernel * store_kernel,
                       library->ResolveFunction<FunctionLibrary::Kernel>(
                           "cpu_atomic_128_store"));
  ASSERT_OK_AND_ASSIGN(
      FunctionLibrary::Kernel * rmw_kernel,
      library->ResolveFunction<FunctionLibrary::Kernel>("cpu_atomic_128_rmw"));

  struct alignas(16) Uint128Words {
    uint64_t lo;
    uint64_t hi;
  };
  static_assert(sizeof(Uint128Words) == 16 && alignof(Uint128Words) == 16);

  alignas(16) std::array<Uint128Words, 4> i128_buf = {
      Uint128Words{1, 2}, Uint128Words{3, 4}, Uint128Words{5, 6},
      Uint128Words{7, 8}};
  alignas(16) Uint128Words i128_store_val{0xffffffffffffffffULL, 0x10ULL};
  alignas(16) std::array<complex128, 4> c128_buf = {
      complex128(1.0, 2.0), complex128(3.0, 4.0), complex128(5.0, 6.0),
      complex128(7.0, 8.0)};
  alignas(16) complex128 c128_store_val(12.5, -3.25);

  std::array<XLA_CPU_KernelArg, 4> store_args = {{
      {i128_buf.data(), sizeof(i128_buf)},
      {&i128_store_val, sizeof(i128_store_val)},
      {c128_buf.data(), sizeof(c128_buf)},
      {&c128_store_val, sizeof(c128_store_val)},
  }};
  XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
  XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
  XLA_CPU_KernelCallFrame store_frame = {&num_workgroups, &workgroup_id,
                                         store_args.size(), store_args.data()};
  ASSERT_EQ(CallCpuKernel(store_kernel, &store_frame), nullptr);

  EXPECT_EQ(i128_buf[0].lo, 1u);
  EXPECT_EQ(i128_buf[0].hi, 2u);
  EXPECT_EQ(i128_buf[1].lo, 0xffffffffffffffffULL);
  EXPECT_EQ(i128_buf[1].hi, 0x10ULL);
  EXPECT_EQ(c128_buf[1], complex128(3.0, 4.0));
  EXPECT_EQ(c128_buf[2], complex128(12.5, -3.25));

  // Verify multi threaded atomic add across both 64 bit halves (including carry
  // propagation across bit 63 for i128 and exact accumulation for c128).
  constexpr int kNumThreads = 8;
  constexpr int kItersPerThread = 500;
  alignas(16) Uint128Words i128_delta{1ULL, 2ULL};
  alignas(16) complex128 c128_delta(1.5, -0.5);
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "atomic128_pool",
                                 kNumThreads);
    for (int t = 0; t < kNumThreads; ++t) {
      pool.Schedule([&]() {
        std::array<XLA_CPU_KernelArg, 4> rmw_args = {{
            {i128_buf.data(), sizeof(i128_buf)},
            {&i128_delta, sizeof(i128_delta)},
            {c128_buf.data(), sizeof(c128_buf)},
            {&c128_delta, sizeof(c128_delta)},
        }};
        XLA_CPU_NumWorkGroups nwg = {1, 1, 1};
        XLA_CPU_WorkGroupId wgid = {0, 0, 0};
        XLA_CPU_KernelCallFrame rmw_frame = {&nwg, &wgid, rmw_args.size(),
                                             rmw_args.data()};
        for (int i = 0; i < kItersPerThread; ++i) {
          EXPECT_EQ(CallCpuKernel(rmw_kernel, &rmw_frame), nullptr);
        }
      });
    }
  }

  constexpr uint64_t kTotalUpdates = kNumThreads * kItersPerThread;
  // Initial i128_buf[1] was (2^64 - 1) + 0x10 * 2^64; adding (1 + 2 * 2^64)
  // kTotalUpdates times carries 1 into the high 64 bit word.
  EXPECT_EQ(i128_buf[1].lo, kTotalUpdates - 1);
  EXPECT_EQ(i128_buf[1].hi, 0x10ULL + 1ULL + 2ULL * kTotalUpdates);
  EXPECT_EQ(c128_buf[2], complex128(12.5 + 1.5 * kTotalUpdates,
                                    -3.25 - 0.5 * kTotalUpdates));
  // Neighboring elements must be untouched.
  EXPECT_EQ(i128_buf[0].lo, 1u);
  EXPECT_EQ(i128_buf[2].lo, 5u);
  EXPECT_EQ(c128_buf[1], complex128(3.0, 4.0));
  EXPECT_EQ(c128_buf[3], complex128(7.0, 8.0));
}

struct EmittedScatterInfo {
  uint64_t num_work_groups = 0;
  std::string mlir;
};

constexpr CpuDeviceSpec kFullAtomicCpuSpec = {
    /*has_atomic_cmpxchg_128=*/true,
    /*has_atomic_load_store_128=*/true,
    /*has_atomic_rmw_64=*/true,
    /*has_atomic_load_store_64=*/true,
};

absl::StatusOr<EmittedScatterInfo> EmitScatterInfo(
    const HloModule& hlo_module, const BufferAssignment& buffer_assignment,
    const CpuDeviceSpec& cpu_device_spec = kFullAtomicCpuSpec) {
  const auto* fusion = Cast<HloFusionInstruction>(
      hlo_module.entry_computation()->root_instruction());
  std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();
  ABSL_ASSIGN_OR_RETURN(
      KernelDefinition<MlirKernelSource> kernel_def,
      EmitFusionKernel(*ctx, *fusion, &buffer_assignment,
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false, cpu_device_spec));
  const int64_t expected_wg = GetCpuScatterEffectiveWorkGroups(
      *fusion, GetCpuScatterRequestedWorkGroups(*fusion), cpu_device_spec,
      &buffer_assignment);
  TF_RET_CHECK(kernel_def.spec().work_dimensions().num_work_groups.x ==
               static_cast<uint64_t>(expected_wg));
  return EmittedScatterInfo{
      kernel_def.spec().work_dimensions().num_work_groups.x,
      kernel_def.source().ToString()};
}

TEST_F(CpuFusionEmitterTest, ScatterWorkGroupClampingAndVariadic) {
  auto emit_hlo =
      [&](absl::string_view hlo) -> absl::StatusOr<EmittedScatterInfo> {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> mod,
                          ParseAndReturnVerifiedModule(hlo));
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> ba,
                          RunBufferAssignment(*mod));
    return EmitScatterInfo(*mod, *ba);
  };
  auto make_1d = [](int64_t n, absl::string_view parts, absl::string_view body,
                    absl::string_view attrs) {
    return absl::Substitute(
        R"hlo(
      HloModule m
      reducer {
        lhs = f32[] parameter(0)
        rhs = f32[] parameter(1)
        ROOT out = $0
      }
      sc {
        op0 = f32[64]{0} parameter(0)
        idx = s32[$1,1]{1,0} parameter(1)
        upd0 = f32[$1,1]{1,0} parameter(2)
        ROOT res = f32[64]{0} scatter(op0, idx, upd0),
          update_window_dims={1}, inserted_window_dims={},
          scatter_dims_to_operand_dims={0}, index_vector_dim=1,
          to_apply=reducer$2
      }
      ENTRY main {
        p0 = f32[64]{0} parameter(0)
        p1 = s32[$1,1]{1,0} parameter(1)
        p2 = f32[$1,1]{1,0} parameter(2)
        ROOT wrapped_scatter = f32[64]{0} fusion(p0, p1, p2), kind=kLoop,
          calls=sc, backend_config={"outer_dimension_partitions":["$3"]}
      })hlo",
        body, n, attrs, parts);
  };

  ASSERT_OK_AND_ASSIGN(EmittedScatterInfo add32,
                       emit_hlo(make_1d(32, "4", "f32[] add(lhs, rhs)", "")));
  EXPECT_EQ(add32.num_work_groups, 4u);
  EXPECT_THAT(add32.mlir, ::testing::HasSubstr("xla.atomic_rmw"));

  ASSERT_OK_AND_ASSIGN(EmittedScatterInfo store32,
                       emit_hlo(make_1d(32, "4", "f32[] copy(rhs)", "")));
  EXPECT_EQ(store32.num_work_groups, 4u);
  EXPECT_THAT(store32.mlir, ::testing::HasSubstr("xla.atomic_store"));

  ASSERT_OK_AND_ASSIGN(EmittedScatterInfo uniq32,
                       emit_hlo(make_1d(32, "4", "f32[] add(lhs, rhs)",
                                        ", unique_indices=true")));
  EXPECT_EQ(uniq32.num_work_groups, 4u);
  EXPECT_THAT(uniq32.mlir, ::testing::HasSubstr("tensor.insert"));
  EXPECT_THAT(uniq32.mlir, ::testing::Not(::testing::HasSubstr("xla.atomic_")));

  ASSERT_OK_AND_ASSIGN(EmittedScatterInfo clamp5,
                       emit_hlo(make_1d(5, "16", "f32[] add(lhs, rhs)", "")));
  EXPECT_EQ(clamp5.num_work_groups, 5u);
  ASSERT_OK_AND_ASSIGN(EmittedScatterInfo single1,
                       emit_hlo(make_1d(1, "4", "f32[] add(lhs, rhs)", "")));
  EXPECT_EQ(single1.num_work_groups, 1u);
  ASSERT_OK_AND_ASSIGN(EmittedScatterInfo empty0,
                       emit_hlo(make_1d(0, "4", "f32[] add(lhs, rhs)", "")));
  EXPECT_EQ(empty0.num_work_groups, 1u);

  ASSERT_OK_AND_ASSIGN(EmittedScatterInfo zero_slice, emit_hlo(R"hlo(
    HloModule zero_slice
    reducer {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT out = f32[] add(lhs, rhs)
    }
    sc {
      op0 = f32[8,0]{1,0} parameter(0)
      idx = s32[16,1]{1,0} parameter(1)
      upd0 = f32[16,1,0]{2,1,0} parameter(2)
      ROOT res = f32[8,0]{1,0} scatter(op0, idx, upd0),
        update_window_dims={1,2}, inserted_window_dims={},
        scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=reducer
    }
    ENTRY main {
      p0 = f32[8,0]{1,0} parameter(0)
      p1 = s32[16,1]{1,0} parameter(1)
      p2 = f32[16,1,0]{2,1,0} parameter(2)
      ROOT wrapped_scatter = f32[8,0]{1,0} fusion(p0, p1, p2), kind=kLoop,
        calls=sc, backend_config={"outer_dimension_partitions":["4"]}
    })hlo"));
  EXPECT_EQ(zero_slice.num_work_groups, 1u);

  EXPECT_THAT(emit_hlo(R"hlo(
    HloModule unsimplified_scalar
    reducer {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT out = f32[] add(lhs, rhs)
    }
    sc {
      op0 = f32[] parameter(0)
      idx = s32[0]{0} parameter(1)
      upd0 = f32[] parameter(2)
      ROOT res = f32[] scatter(op0, idx, upd0),
        update_window_dims={}, inserted_window_dims={},
        scatter_dims_to_operand_dims={}, index_vector_dim=0, to_apply=reducer
    }
    ENTRY main {
      p0 = f32[] parameter(0)
      p1 = s32[0]{0} parameter(1)
      p2 = f32[] parameter(2)
      ROOT wrapped_scatter = f32[] fusion(p0, p1, p2), kind=kLoop,
        calls=sc, backend_config={"outer_dimension_partitions":["4"]}
    })hlo"),
              absl_testing::StatusIs(absl::StatusCode::kInternal));

  auto make_variadic = [](absl::string_view t0, absl::string_view lay0,
                          absl::string_view lay_upd0, absl::string_view attrs) {
    return absl::Substitute(
        R"hlo(
      HloModule variadic
      reducer {
        lhs0 = $0[] parameter(0)
        lhs1 = f32[] parameter(1)
        rhs0 = $0[] parameter(2)
        rhs1 = f32[] parameter(3)
        sum0 = $0[] add(lhs0, rhs0)
        sum1 = f32[] add(lhs1, rhs1)
        ROOT out = ($0[], f32[]) tuple(sum0, sum1)
      }
      sc {
        op0 = $0[64]$1 parameter(0)
        op1 = f32[64]{0} parameter(1)
        idx = s32[32,1]{1,0} parameter(2)
        upd0 = $0[32,1]$2 parameter(3)
        upd1 = f32[32,1]{1,0} parameter(4)
        ROOT res = ($0[64]$1, f32[64]{0}) scatter(op0, op1, idx, upd0, upd1),
          update_window_dims={1}, inserted_window_dims={},
          scatter_dims_to_operand_dims={0}, index_vector_dim=1,
          to_apply=reducer$3
      }
      ENTRY main {
        p0 = $0[64]$1 parameter(0)
        p1 = f32[64]{0} parameter(1)
        p2 = s32[32,1]{1,0} parameter(2)
        p3 = $0[32,1]$2 parameter(3)
        p4 = f32[32,1]{1,0} parameter(4)
        ROOT wrapped_scatter = ($0[64]$1, f32[64]{0})
          fusion(p0, p1, p2, p3, p4), kind=kLoop, calls=sc,
          backend_config={"outer_dimension_partitions":["4"]}
      })hlo",
        t0, lay0, lay_upd0, attrs);
  };
  ASSERT_OK_AND_ASSIGN(
      EmittedScatterInfo var_uniq,
      emit_hlo(make_variadic("s32", "{0}", "{1,0}", ", unique_indices=true")));
  EXPECT_EQ(var_uniq.num_work_groups, 4u);
  ASSERT_OK_AND_ASSIGN(EmittedScatterInfo var_non_uniq,
                       emit_hlo(make_variadic("s32", "{0}", "{1,0}", "")));
  EXPECT_EQ(var_non_uniq.num_work_groups, 1u);
  ASSERT_OK_AND_ASSIGN(EmittedScatterInfo var_s4_uniq,
                       emit_hlo(make_variadic("s4", "{0:E(4)}", "{1,0:E(4)}",
                                              ", unique_indices=true")));
  EXPECT_EQ(var_s4_uniq.num_work_groups, 1u);
}

struct CompiledHostScatterKernel {
  std::unique_ptr<FunctionLibrary> library;
  FunctionLibrary::Kernel* kernel = nullptr;
  uint64_t num_work_groups = 0;
  CpuDeviceSpec host_spec;
};

absl::StatusOr<CompiledHostScatterKernel> CompileHostScatterKernel(
    const HloModule& hlo_module, const BufferAssignment& buffer_assignment) {
  std::unique_ptr<IrCompiler> ir_compiler = IrCompiler::Create(
      {},
      IrCompiler::Options{/*opt_level=*/llvm::CodeGenOptLevel::Default,
                          /*optimize_for_size=*/false,
                          TargetMachineOptions(GetDebugOptionsFromFlags())},
      IrCompiler::CompilationHooks());
  ABSL_ASSIGN_OR_RETURN(JitCompiler jit,
                        JitCompiler::Create({}, std::move(ir_compiler)));
  const llvm::TargetMachine* tm = jit.target_machine();
  TF_RET_CHECK(tm != nullptr);
  const CpuDeviceSpec host_spec = CpuDeviceSpecFromTargetMachine(tm);
  const auto* fusion = Cast<HloFusionInstruction>(
      hlo_module.entry_computation()->root_instruction());
  std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();
  ABSL_ASSIGN_OR_RETURN(
      KernelDefinition<MlirKernelSource> kernel_def,
      EmitFusionKernel(*ctx, *fusion, &buffer_assignment,
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false, host_spec));
  const std::string kernel_name(kernel_def.spec().name());
  const uint64_t num_wg = kernel_def.spec().work_dimensions().num_work_groups.x;
  FusionCompiler::Options opts;
  opts.vector_width = 256;
  opts.verification_level = 1;
  opts.fast_min_max = true;
  opts.cpu_features = tm->getTargetFeatureString().str();
  opts.cpu_device_spec = host_spec;
  FusionCompiler compiler(ctx.get(), std::move(opts));
  ABSL_ASSIGN_OR_RETURN(LlvmKernelSource llvm_src,
                        compiler.Compile(std::move(kernel_def).TakeSource()));
  ABSL_RETURN_IF_ERROR(jit.AddModule(std::move(llvm_src).thread_safe_module()));
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<FunctionLibrary> lib,
      std::move(jit).Compile(
          {FunctionLibrary::Sym<FunctionLibrary::Kernel>(kernel_name)}));
  ABSL_ASSIGN_OR_RETURN(
      FunctionLibrary::Kernel * fn,
      lib->ResolveFunction<FunctionLibrary::Kernel>(kernel_name));
  return CompiledHostScatterKernel{std::move(lib), fn, num_wg, host_spec};
}

void RunScatterKernelConcurrently(FunctionLibrary::Kernel* kernel,
                                  uint64_t num_work_groups,
                                  absl::Span<const XLA_CPU_KernelArg> args) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "scatter_wg_pool",
                               static_cast<int>(num_work_groups));
  std::atomic<uint64_t> ready{0};
  for (uint64_t wg = 0; wg < num_work_groups; ++wg) {
    pool.Schedule([&, wg]() {
      ready.fetch_add(1, std::memory_order_acq_rel);
      for (int spin = 0; spin < 10000 && ready.load(std::memory_order_acquire) <
                                             num_work_groups;
           ++spin) {
      }
      std::vector<XLA_CPU_KernelArg> local_args(args.begin(), args.end());
      XLA_CPU_NumWorkGroups nwg = {num_work_groups, 1, 1};
      XLA_CPU_WorkGroupId wgid = {wg, 0, 0};
      XLA_CPU_KernelCallFrame frame = {&nwg, &wgid, local_args.size(),
                                       local_args.data()};
      EXPECT_EQ(CallCpuKernel(kernel, &frame), nullptr);
    });
  }
}

TEST_F(CpuFusionEmitterTest, ScatterPackedSubByteAtomicCasAndExecution) {
  auto make_packed = [](absl::string_view ty, int bits, absl::string_view op,
                        bool unique) {
    const std::string body =
        op == "overwrite" ? absl::Substitute("$0[] copy(rhs)", ty)
                          : absl::Substitute("$0[] $1(lhs, rhs)", ty, op);
    return absl::Substitute(
        R"hlo(
      HloModule packed
      reducer {
        lhs = $0[] parameter(0)
        rhs = $0[] parameter(1)
        ROOT out = $1
      }
      sc {
        op0 = $0[64]{0:E($2)} parameter(0)
        idx = s32[64,1]{1,0} parameter(1)
        upd0 = $0[64,1]{1,0:E($2)} parameter(2)
        ROOT res = $0[64]{0:E($2)} scatter(op0, idx, upd0),
          update_window_dims={1}, inserted_window_dims={},
          scatter_dims_to_operand_dims={0}, index_vector_dim=1,
          to_apply=reducer$3
      }
      ENTRY main {
        p0 = $0[64]{0:E($2)} parameter(0)
        p1 = s32[64,1]{1,0} parameter(1)
        p2 = $0[64,1]{1,0:E($2)} parameter(2)
        ROOT wrapped_scatter = $0[64]{0:E($2)} fusion(p0, p1, p2), kind=kLoop,
          calls=sc, backend_config={"outer_dimension_partitions":["4"]}
      })hlo",
        ty, body, bits, unique ? ", unique_indices=true" : "");
  };

  for (const auto& [ty, bits] : std::vector<std::pair<absl::string_view, int>>{
           {"s4", 4}, {"u4", 4}, {"s2", 2}, {"u2", 2}}) {
    for (bool unique : {false, true}) {
      ASSERT_OK_AND_ASSIGN(
          std::unique_ptr<HloModule> add_mod,
          ParseAndReturnVerifiedModule(make_packed(ty, bits, "add", unique)));
      ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> add_ba,
                           RunBufferAssignment(*add_mod));
      ASSERT_OK_AND_ASSIGN(EmittedScatterInfo add_info,
                           EmitScatterInfo(*add_mod, *add_ba));
      EXPECT_EQ(add_info.num_work_groups, 4u) << ty << " unique=" << unique;
      EXPECT_THAT(add_info.mlir, ::testing::HasSubstr("xla.atomic_rmw"));
      EXPECT_THAT(add_info.mlir,
                  ::testing::Not(::testing::HasSubstr("tensor.insert")));

      ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> st_mod,
                           ParseAndReturnVerifiedModule(
                               make_packed(ty, bits, "overwrite", unique)));
      ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> st_ba,
                           RunBufferAssignment(*st_mod));
      ASSERT_OK_AND_ASSIGN(EmittedScatterInfo st_info,
                           EmitScatterInfo(*st_mod, *st_ba));
      EXPECT_EQ(st_info.num_work_groups, 4u) << ty << " unique=" << unique;
      EXPECT_THAT(st_info.mlir, ::testing::HasSubstr("xla.atomic_store"));
      EXPECT_THAT(st_info.mlir,
                  ::testing::Not(::testing::HasSubstr("tensor.insert")));
    }
  }

  auto pack_nibbles = [](const std::array<uint8_t, 64>& vals) {
    std::array<uint8_t, 32> bytes{};
    for (int i = 0; i < 64; ++i) {
      bytes[i / 2] |= static_cast<uint8_t>((vals[i] & 0xfU) << ((i % 2) * 4));
    }
    return bytes;
  };
  auto get_nibble = [](const std::array<uint8_t, 32>& bytes, int i) -> uint8_t {
    return static_cast<uint8_t>((bytes[i / 2] >> ((i % 2) * 4)) & 0xfU);
  };
  auto sign_ext4 = [](uint8_t v) -> int {
    return (v & 0x8U) != 0 ? static_cast<int>(v) - 16 : static_cast<int>(v);
  };

  // Unique overwrite on s4 and u4: workgroups (0, 1) and (2, 3) concurrently
  // update even and odd nibbles sharing every destination byte.
  for (absl::string_view ty : {"s4", "u4"}) {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> mod,
        ParseAndReturnVerifiedModule(make_packed(ty, 4, "overwrite", true)));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> ba,
                         RunBufferAssignment(*mod));
    ASSERT_OK_AND_ASSIGN(CompiledHostScatterKernel ck,
                         CompileHostScatterKernel(*mod, *ba));
    if (!ck.host_spec.has_atomic_rmw_64) {
      GTEST_SKIP() << "Host lacks inline 8 through 64 bit atomic RMW.";
    }
    ASSERT_EQ(ck.num_work_groups, 4u);
    alignas(16) std::array<int32_t, 64> idx{};
    std::array<uint8_t, 64> upd_nibbles{};
    std::array<uint8_t, 64> expected{};
    for (int k = 0; k < 64; ++k) {
      const int pair_group = k / 32;
      const int parity = (k / 16) % 2;
      const int lane = k % 16;
      idx[k] = pair_group * 32 + lane * 2 + parity;
      upd_nibbles[k] = static_cast<uint8_t>((k * 5 + 3) & 0xfU);
      expected[idx[k]] = upd_nibbles[k];
    }
    alignas(16) std::array<uint8_t, 32> upd_buf = pack_nibbles(upd_nibbles);
    for (int round = 0; round < 16; ++round) {
      alignas(16) std::array<uint8_t, 32> out_buf{};
      out_buf.fill(0xa5U);
      std::array<XLA_CPU_KernelArg, 4> args = {{
          {out_buf.data(), out_buf.size()},
          {idx.data(), sizeof(idx)},
          {upd_buf.data(), upd_buf.size()},
          {out_buf.data(), out_buf.size()},
      }};
      RunScatterKernelConcurrently(ck.kernel, ck.num_work_groups, args);
      for (int i = 0; i < 64; ++i) {
        EXPECT_EQ(get_nibble(out_buf, i), expected[i]) << ty << " i=" << i;
      }
    }
  }

  // Non unique colliding add and maximum on s4 and u4 across adjacent even and
  // odd nibbles.
  for (const auto& [ty, op] :
       std::vector<std::pair<absl::string_view, absl::string_view>>{
           {"s4", "add"},
           {"u4", "add"},
           {"s4", "maximum"},
           {"u4", "maximum"}}) {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> mod,
        ParseAndReturnVerifiedModule(make_packed(ty, 4, op, false)));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> ba,
                         RunBufferAssignment(*mod));
    ASSERT_OK_AND_ASSIGN(CompiledHostScatterKernel ck,
                         CompileHostScatterKernel(*mod, *ba));
    ASSERT_EQ(ck.num_work_groups, 4u);
    std::array<uint8_t, 64> init_nibbles{};
    for (int i = 0; i < 64; ++i) {
      init_nibbles[i] = static_cast<uint8_t>(ty == "s4" ? 0x9U : 0x2U);
    }
    alignas(16) std::array<int32_t, 64> idx{};
    std::array<uint8_t, 64> upd_nibbles{};
    std::array<uint8_t, 64> expected = init_nibbles;
    for (int k = 0; k < 64; ++k) {
      idx[k] = k % 16;
      upd_nibbles[k] = static_cast<uint8_t>((k * 3 + 1) & 0xfU);
      if (op == "add") {
        expected[idx[k]] =
            static_cast<uint8_t>((expected[idx[k]] + upd_nibbles[k]) & 0xfU);
      } else if (ty == "s4") {
        if (sign_ext4(upd_nibbles[k]) > sign_ext4(expected[idx[k]])) {
          expected[idx[k]] = upd_nibbles[k];
        }
      } else {
        expected[idx[k]] = std::max(expected[idx[k]], upd_nibbles[k]);
      }
    }
    alignas(16) std::array<uint8_t, 32> upd_buf = pack_nibbles(upd_nibbles);
    for (int round = 0; round < 16; ++round) {
      alignas(16) std::array<uint8_t, 32> out_buf = pack_nibbles(init_nibbles);
      std::array<XLA_CPU_KernelArg, 4> args = {{
          {out_buf.data(), out_buf.size()},
          {idx.data(), sizeof(idx)},
          {upd_buf.data(), upd_buf.size()},
          {out_buf.data(), out_buf.size()},
      }};
      RunScatterKernelConcurrently(ck.kernel, ck.num_work_groups, args);
      for (int i = 0; i < 64; ++i) {
        EXPECT_EQ(get_nibble(out_buf, i), expected[i])
            << ty << " " << op << " i=" << i;
      }
    }
  }
}

struct CrossCompiledScatter {
  uint64_t num_work_groups = 0;
  CompiledAtomic128Artifacts artifacts;
};

TEST_F(CpuFusionEmitterTest, ScatterCrossTargetX64AndAArch64Codegen) {
  constexpr absl::string_view kX86 = "x86_64-unknown-linux-gnu";
  constexpr absl::string_view kArm = "aarch64-unknown-linux-gnu";
  auto compile_for =
      [&](absl::string_view hlo, absl::string_view triple,
          absl::string_view cpu,
          absl::string_view features) -> absl::StatusOr<CrossCompiledScatter> {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> mod,
                          ParseAndReturnVerifiedModule(hlo));
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> ba,
                          RunBufferAssignment(*mod));
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<llvm::TargetMachine> tm,
                          TargetMachineTestBase::CreateTargetMachineWithOptions(
                              triple, cpu, features, llvm::Reloc::PIC_,
                              llvm::CodeGenOptLevel::Default));
    const CpuDeviceSpec spec = CpuDeviceSpecFromTargetMachine(tm.get());
    const auto* fusion = Cast<HloFusionInstruction>(
        mod->entry_computation()->root_instruction());
    std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();
    CpuScatterFusion emitter(*ba, fusion, ctx.get(), spec);
    ABSL_ASSIGN_OR_RETURN(KernelDefinition<MlirKernelSource> def,
                          emitter.EmitKernelDefinition());
    const uint64_t num_wg = def.spec().work_dimensions().num_work_groups.x;
    FusionCompiler::Options opts;
    opts.vector_width = 256;
    opts.verification_level = 1;
    opts.fast_min_max = true;
    opts.cpu_features = tm->getTargetFeatureString().str();
    opts.cpu_device_spec = spec;
    FusionCompiler compiler(ctx.get(), std::move(opts));
    ABSL_ASSIGN_OR_RETURN(LlvmKernelSource llvm_src,
                          compiler.Compile(std::move(def).TakeSource()));
    ABSL_ASSIGN_OR_RETURN(
        CompiledAtomic128Artifacts art,
        CompileLlvmModuleForTarget(std::move(llvm_src), *tm, triple, cpu,
                                   features, llvm::CodeGenOptLevel::Default));
    EXPECT_TRUE(HasNoAtomicRuntimeHelpers(art));
    EXPECT_THAT(art.optimized_llvm_ir,
                Not(HasSubstr("__llvm_memcpy_element_unordered_atomic")));
    return CrossCompiledScatter{num_wg, std::move(art)};
  };
  auto make_1d = [](absl::string_view ty, absl::string_view op, bool unique) {
    const absl::string_view lay0 = ty == "s4" ? "{0:E(4)}" : "{0}";
    const absl::string_view lay1 = ty == "s4" ? "{1,0:E(4)}" : "{1,0}";
    const std::string reducer_lines =
        op == "keep_lhs"
            ? absl::Substitute(
                  "ROOT lhs = $0[] parameter(0)\n        rhs = $0[] "
                  "parameter(1)",
                  ty)
            : absl::Substitute(
                  "lhs = $0[] parameter(0)\n        rhs = $0[] parameter(1)\n"
                  "        ROOT out = $1",
                  ty,
                  op == "overwrite"
                      ? absl::Substitute("$0[] copy(rhs)", ty)
                      : absl::Substitute("$0[] $1(lhs, rhs)", ty, op));
    return absl::Substitute(
        R"hlo(
      HloModule cross
      reducer {
        $1
      }
      sc {
        op0 = $0[64]$2 parameter(0)
        idx = s32[32,1]{1,0} parameter(1)
        upd0 = $0[32,1]$3 parameter(2)
        ROOT res = $0[64]$2 scatter(op0, idx, upd0),
          update_window_dims={1}, inserted_window_dims={},
          scatter_dims_to_operand_dims={0}, index_vector_dim=1,
          to_apply=reducer$4
      }
      ENTRY main {
        p0 = $0[64]$2 parameter(0)
        p1 = s32[32,1]{1,0} parameter(1)
        p2 = $0[32,1]$3 parameter(2)
        ROOT wrapped_scatter = $0[64]$2 fusion(p0, p1, p2), kind=kLoop,
          calls=sc, backend_config={"outer_dimension_partitions":["4"]}
      })hlo",
        ty, reducer_lines, lay0, lay1, unique ? ", unique_indices=true" : "");
  };

  // Contiguous window slice overwrite stays per element monotonic atomic store.
  constexpr absl::string_view kWindowOverwrite = R"hlo(
    HloModule window_overwrite
    reducer {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT out = f32[] copy(rhs)
    }
    sc {
      op0 = f32[64,128]{1,0} parameter(0)
      idx = s32[32,1]{1,0} parameter(1)
      upd0 = f32[32,1,128]{2,1,0} parameter(2)
      ROOT res = f32[64,128]{1,0} scatter(op0, idx, upd0),
        update_window_dims={1,2}, inserted_window_dims={},
        scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=reducer
    }
    ENTRY main {
      p0 = f32[64,128]{1,0} parameter(0)
      p1 = s32[32,1]{1,0} parameter(1)
      p2 = f32[32,1,128]{2,1,0} parameter(2)
      ROOT wrapped_scatter = f32[64,128]{1,0} fusion(p0, p1, p2), kind=kLoop,
        calls=sc, backend_config={"outer_dimension_partitions":["4"]}
    })hlo";
  for (const auto& [triple, cpu] :
       std::vector<std::pair<absl::string_view, absl::string_view>>{
           {kX86, "haswell"}, {kArm, "neoverse-v2"}}) {
    ASSERT_OK_AND_ASSIGN(CrossCompiledScatter win,
                         compile_for(kWindowOverwrite, triple, cpu, ""));
    EXPECT_EQ(win.num_work_groups, 4u);
    EXPECT_THAT(
        win.artifacts.llvm_ir,
        ::testing::ContainsRegex("store atomic float .* monotonic, align 4"));
    EXPECT_THAT(win.artifacts.optimized_llvm_ir,
                ::testing::ContainsRegex(
                    "store atomic (float|i32) .* monotonic, align 4"));
  }

  // x86_64 subtarget matrix: -cx16, +cx16 without avx, and +cx16 with avx.
  for (absl::string_view op : {"add", "overwrite"}) {
    ASSERT_OK_AND_ASSIGN(
        CrossCompiledScatter no_cx16,
        compile_for(make_1d("c128", op, false), kX86, "haswell", "-cx16"));
    EXPECT_EQ(no_cx16.num_work_groups, 1u);
    ASSERT_OK_AND_ASSIGN(
        CrossCompiledScatter cx16_no_avx,
        compile_for(make_1d("c128", op, false), kX86, "x86-64", "+cx16,-avx"));
    EXPECT_EQ(cx16_no_avx.num_work_groups, 4u);
    EXPECT_THAT(cx16_no_avx.artifacts.assembly,
                ::testing::ContainsRegex("lock[[:space:]]+cmpxchg16b"));
  }
  ASSERT_OK_AND_ASSIGN(
      CrossCompiledScatter x86_s32,
      compile_for(make_1d("s32", "add", false), kX86, "haswell", "-cx16"));
  EXPECT_EQ(x86_s32.num_work_groups, 4u);
  ASSERT_OK_AND_ASSIGN(
      CrossCompiledScatter x86_c128_uniq,
      compile_for(make_1d("c128", "add", true), kX86, "haswell", "-cx16"));
  EXPECT_EQ(x86_c128_uniq.num_work_groups, 4u);
  ASSERT_OK_AND_ASSIGN(CrossCompiledScatter cx16_avx_st,
                       compile_for(make_1d("c128", "overwrite", false), kX86,
                                   "x86-64", "+cx16,+avx"));
  EXPECT_EQ(cx16_avx_st.num_work_groups, 4u);
  EXPECT_THAT(
      cx16_avx_st.artifacts.llvm_ir,
      ::testing::ContainsRegex("store atomic i128 .* monotonic, align 16"));
  EXPECT_THAT(cx16_avx_st.artifacts.llvm_ir,
              ::testing::Not(::testing::HasSubstr("cmpxchg")));

  // AArch64 subtarget matrix: outline atomics without lse, +lse2, and +lse.
  for (absl::string_view ty : {"s32", "f32", "c64", "c128", "s4"}) {
    ASSERT_OK_AND_ASSIGN(CrossCompiledScatter arm_add,
                         compile_for(make_1d(ty, "add", false), kArm, "generic",
                                     "-lse,+outline-atomics"));
    EXPECT_EQ(arm_add.num_work_groups, 1u) << ty;
  }
  for (bool uniq : {false, true}) {
    ASSERT_OK_AND_ASSIGN(CrossCompiledScatter arm_s4,
                         compile_for(make_1d("s4", "overwrite", uniq), kArm,
                                     "generic", "-lse,+outline-atomics"));
    EXPECT_EQ(arm_s4.num_work_groups, 1u);
  }
  ASSERT_OK_AND_ASSIGN(CrossCompiledScatter arm_f32_uniq,
                       compile_for(make_1d("f32", "add", true), kArm, "generic",
                                   "-lse,+outline-atomics"));
  EXPECT_EQ(arm_f32_uniq.num_work_groups, 4u);
  ASSERT_OK_AND_ASSIGN(CrossCompiledScatter arm_f32_st,
                       compile_for(make_1d("f32", "overwrite", false), kArm,
                                   "generic", "-lse,+outline-atomics"));
  EXPECT_EQ(arm_f32_st.num_work_groups, 4u);

  ASSERT_OK_AND_ASSIGN(CrossCompiledScatter arm_lse2_st,
                       compile_for(make_1d("c128", "overwrite", false), kArm,
                                   "generic", "-lse,+lse2,+outline-atomics"));
  EXPECT_EQ(arm_lse2_st.num_work_groups, 4u);
  EXPECT_THAT(
      arm_lse2_st.artifacts.llvm_ir,
      ::testing::ContainsRegex("store atomic i128 .* monotonic, align 16"));
  EXPECT_THAT(arm_lse2_st.artifacts.assembly, ::testing::HasSubstr("stp"));
  EXPECT_THAT(arm_lse2_st.artifacts.assembly,
              ::testing::Not(::testing::HasSubstr("casp")));
  EXPECT_THAT(arm_lse2_st.artifacts.assembly,
              ::testing::Not(::testing::HasSubstr("ldxp")));
  ASSERT_OK_AND_ASSIGN(CrossCompiledScatter arm_lse2_add,
                       compile_for(make_1d("c128", "add", false), kArm,
                                   "generic", "-lse,+lse2,+outline-atomics"));
  EXPECT_EQ(arm_lse2_add.num_work_groups, 1u);
  // Keep destination reducers (`ROOT lhs = parameter(0)`) have zero users on
  // parameter 0 yet still lower as atomic RMW, so outline atomics without LSE
  // must clamp both 32 bit and 128 bit variants to 1 workgroup.
  ASSERT_OK_AND_ASSIGN(CrossCompiledScatter arm_keep_lhs_s32,
                       compile_for(make_1d("s32", "keep_lhs", false), kArm,
                                   "generic", "-lse,+outline-atomics"));
  EXPECT_EQ(arm_keep_lhs_s32.num_work_groups, 1u);
  ASSERT_OK_AND_ASSIGN(CrossCompiledScatter arm_lse2_keep_lhs_c128,
                       compile_for(make_1d("c128", "keep_lhs", false), kArm,
                                   "generic", "-lse,+lse2,+outline-atomics"));
  EXPECT_EQ(arm_lse2_keep_lhs_c128.num_work_groups, 1u);

  for (absl::string_view feat : {"+lse", "+lse,+lse2"}) {
    for (absl::string_view op : {"add", "overwrite"}) {
      ASSERT_OK_AND_ASSIGN(
          CrossCompiledScatter arm_lse,
          compile_for(make_1d("c128", op, false), kArm, "generic", feat));
      EXPECT_EQ(arm_lse.num_work_groups, 4u) << feat << " " << op;
    }
  }
}

TEST_F(CpuFusionEmitterTest, ScatterAlignmentAliasingAndDeterminismFallbacks) {
  auto size_fn = [](const BufferValue& buf) {
    return CpuExecutable::ShapeSizeBytes(buf.shape());
  };
  auto emit_at_offset = [&](absl::string_view ty,
                            int64_t offset) -> absl::StatusOr<uint64_t> {
    const std::string hlo = absl::Substitute(
        R"hlo(
      HloModule aligned
      reducer {
        lhs = $0[] parameter(0)
        rhs = $0[] parameter(1)
        ROOT out = $0[] add(lhs, rhs)
      }
      sc {
        op0 = $0[16]{0} parameter(0)
        idx = s32[16,1]{1,0} parameter(1)
        upd0 = $0[16,1]{1,0} parameter(2)
        ROOT res = $0[16]{0} scatter(op0, idx, upd0),
          update_window_dims={1}, inserted_window_dims={},
          scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=reducer
      }
      ENTRY main {
        p0 = $0[16]{0} parameter(0)
        base_op = $0[16]{0} copy(p0)
        p1 = s32[16,1]{1,0} parameter(1)
        p2 = $0[16,1]{1,0} parameter(2)
        ROOT wrapped_scatter = $0[16]{0} fusion(base_op, p1, p2), kind=kLoop,
          calls=sc, backend_config={"outer_dimension_partitions":["4"]}
      })hlo",
        ty);
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> mod,
                          ParseAndReturnVerifiedModule(hlo));
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> base_ba,
                          RunBufferAssignment(*mod));
    const HloInstruction* root = mod->entry_computation()->root_instruction();
    ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice res_slice,
                          base_ba->GetUniqueSlice(root, {}));
    BufferAssignmentProto proto = base_ba->ToProto();
    BufferAllocationProto* alloc =
        proto.mutable_buffer_allocations(res_slice.index());
    alloc->set_size(offset + res_slice.size());
    for (BufferAllocationProto::Assigned& a : *alloc->mutable_assigned()) {
      a.set_offset(offset);
    }
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<BufferAssignment> custom_ba,
        BufferAssignment::FromProto(proto, mod.get(), size_fn, &alias_info_));
    ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice shifted,
                          custom_ba->GetUniqueSlice(root, {}));
    TF_RET_CHECK(shifted.offset() == offset);
    ABSL_ASSIGN_OR_RETURN(EmittedScatterInfo info,
                          EmitScatterInfo(*mod, *custom_ba));
    return info.num_work_groups;
  };
  EXPECT_THAT(emit_at_offset("c128", 8), absl_testing::IsOkAndHolds(1u));
  EXPECT_THAT(emit_at_offset("c128", 16), absl_testing::IsOkAndHolds(4u));
  EXPECT_THAT(emit_at_offset("c64", 4), absl_testing::IsOkAndHolds(1u));
  EXPECT_THAT(emit_at_offset("c64", 8), absl_testing::IsOkAndHolds(4u));

  // Fused producer reading the scatter operand parameter falls back to 1.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> alias_mod,
                       ParseAndReturnVerifiedModule(R"hlo(
    HloModule producer_alias
    reducer {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT out = f32[] add(lhs, rhs)
    }
    sc {
      op0 = f32[16]{0} parameter(0)
      idx = s32[16,1]{1,0} parameter(1)
      raw_upd = f32[16,1]{1,0} parameter(2)
      op_col = f32[16,1]{1,0} reshape(op0)
      upd0 = f32[16,1]{1,0} add(raw_upd, op_col)
      ROOT res = f32[16]{0} scatter(op0, idx, upd0),
        update_window_dims={1}, inserted_window_dims={},
        scatter_dims_to_operand_dims={0}, index_vector_dim=1,
        unique_indices=true, to_apply=reducer
    }
    ENTRY main {
      p0 = f32[16]{0} parameter(0)
      p1 = s32[16,1]{1,0} parameter(1)
      p2 = f32[16,1]{1,0} parameter(2)
      ROOT wrapped_scatter = f32[16]{0} fusion(p0, p1, p2), kind=kLoop,
        calls=sc, backend_config={"outer_dimension_partitions":["4"]}
    })hlo"));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> alias_ba,
                       RunBufferAssignment(*alias_mod));
  ASSERT_OK_AND_ASSIGN(EmittedScatterInfo alias_info,
                       EmitScatterInfo(*alias_mod, *alias_ba));
  EXPECT_EQ(alias_info.num_work_groups, 1u);

  // Distinct HLO producer whose assigned buffer slice overlaps the result.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> ov_mod,
                       ParseAndReturnVerifiedModule(R"hlo(
    HloModule slice_overlap
    reducer {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT out = f32[] add(lhs, rhs)
    }
    sc {
      op0 = f32[16]{0} parameter(0)
      idx = s32[16,1]{1,0} parameter(1)
      upd0 = f32[16,1]{1,0} parameter(2)
      ROOT res = f32[16]{0} scatter(op0, idx, upd0),
        update_window_dims={1}, inserted_window_dims={},
        scatter_dims_to_operand_dims={0}, index_vector_dim=1,
        unique_indices=true, to_apply=reducer
    }
    ENTRY main {
      p0 = f32[16]{0} parameter(0)
      base_op = f32[16]{0} copy(p0)
      p1 = s32[16,1]{1,0} parameter(1)
      p2 = f32[16,1]{1,0} parameter(2)
      base_upd = f32[16,1]{1,0} copy(p2)
      ROOT wrapped_scatter = f32[16]{0} fusion(base_op, p1, base_upd),
        kind=kLoop, calls=sc,
        backend_config={"outer_dimension_partitions":["4"]}
    })hlo"));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> ov_base,
                       RunBufferAssignment(*ov_mod));
  ASSERT_OK_AND_ASSIGN(EmittedScatterInfo ov_disjoint,
                       EmitScatterInfo(*ov_mod, *ov_base));
  EXPECT_EQ(ov_disjoint.num_work_groups, 4u);
  const HloInstruction* ov_root =
      ov_mod->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(BufferAllocation::Slice ov_res,
                       ov_base->GetUniqueSlice(ov_root, {}));
  ASSERT_OK_AND_ASSIGN(BufferAllocation::Slice ov_upd,
                       ov_base->GetUniqueSlice(ov_root->operand(2), {}));
  ASSERT_NE(ov_res.index(), ov_upd.index());
  BufferAssignmentProto ov_proto = ov_base->ToProto();
  ov_proto.clear_peak_buffers();
  BufferAllocationProto* dst_alloc =
      ov_proto.mutable_buffer_allocations(ov_res.index());
  BufferAllocationProto* src_alloc =
      ov_proto.mutable_buffer_allocations(ov_upd.index());
  dst_alloc->set_size(std::max(dst_alloc->size(), ov_upd.size() + 8));
  for (const BufferAllocationProto::Assigned& a : src_alloc->assigned()) {
    BufferAllocationProto::Assigned* moved = dst_alloc->add_assigned();
    *moved = a;
    moved->set_offset(8);
  }
  src_alloc->clear_assigned();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> ov_ba,
                       BufferAssignment::FromProto(ov_proto, ov_mod.get(),
                                                   size_fn, &alias_info_));
  ASSERT_OK_AND_ASSIGN(EmittedScatterInfo ov_info,
                       EmitScatterInfo(*ov_mod, *ov_ba));
  EXPECT_EQ(ov_info.num_work_groups, 1u);

  // Determinism flags restrict parallel scatter to order independent reducers.
  auto emit_det = [&](absl::string_view ty, absl::string_view body, bool unique,
                      bool exclude_nondet) -> absl::StatusOr<uint64_t> {
    const std::string hlo = absl::Substitute(
        R"hlo(
      HloModule det
      reducer {
        lhs = $0[] parameter(0)
        rhs = $0[] parameter(1)
        ROOT out = $1
      }
      sc {
        op0 = $0[32]{0} parameter(0)
        idx = s32[16,1]{1,0} parameter(1)
        upd0 = $0[16,1]{1,0} parameter(2)
        ROOT res = $0[32]{0} scatter(op0, idx, upd0),
          update_window_dims={1}, inserted_window_dims={},
          scatter_dims_to_operand_dims={0}, index_vector_dim=1,
          to_apply=reducer$2
      }
      ENTRY main {
        p0 = $0[32]{0} parameter(0)
        p1 = s32[16,1]{1,0} parameter(1)
        p2 = $0[16,1]{1,0} parameter(2)
        ROOT wrapped_scatter = $0[32]{0} fusion(p0, p1, p2), kind=kLoop,
          calls=sc, backend_config={"outer_dimension_partitions":["4"]}
      })hlo",
        ty, body, unique ? ", unique_indices=true" : "");
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> mod,
                          ParseAndReturnVerifiedModule(hlo));
    DebugOptions& dbg = mod->mutable_config().mutable_debug_options();
    if (exclude_nondet) {
      dbg.set_xla_gpu_exclude_nondeterministic_ops(true);
    } else {
      dbg.set_xla_gpu_deterministic_ops(true);
    }
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> ba,
                          RunBufferAssignment(*mod));
    ABSL_ASSIGN_OR_RETURN(EmittedScatterInfo info, EmitScatterInfo(*mod, *ba));
    return info.num_work_groups;
  };
  for (bool exclude_nondet : {true, false}) {
    EXPECT_THAT(emit_det("f32", "f32[] add(lhs, rhs)", false, exclude_nondet),
                absl_testing::IsOkAndHolds(1u));
    EXPECT_THAT(emit_det("c128", "c128[] add(lhs, rhs)", false, exclude_nondet),
                absl_testing::IsOkAndHolds(1u));
    EXPECT_THAT(emit_det("s32", "s32[] copy(rhs)", false, exclude_nondet),
                absl_testing::IsOkAndHolds(1u));
    EXPECT_THAT(emit_det("s32", "s32[] add(lhs, rhs)", false, exclude_nondet),
                absl_testing::IsOkAndHolds(4u));
    EXPECT_THAT(
        emit_det("s32", "s32[] maximum(lhs, rhs)", false, exclude_nondet),
        absl_testing::IsOkAndHolds(4u));
    EXPECT_THAT(emit_det("pred", "pred[] or(lhs, rhs)", false, exclude_nondet),
                absl_testing::IsOkAndHolds(4u));
    EXPECT_THAT(emit_det("f32", "f32[] add(lhs, rhs)", true, exclude_nondet),
                absl_testing::IsOkAndHolds(4u));
  }
}

TEST_F(CpuFusionEmitterTest, ScatterConcurrentHostJitExecution) {
  auto build_kernel =
      [&](absl::string_view hlo) -> absl::StatusOr<CompiledHostScatterKernel> {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> mod,
                          ParseAndReturnVerifiedModule(hlo));
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> ba,
                          RunBufferAssignment(*mod));
    return CompileHostScatterKernel(*mod, *ba);
  };
  auto make_1d = [](absl::string_view ty, int64_t m, int64_t n, int64_t wg,
                    absl::string_view body) {
    return absl::Substitute(
        R"hlo(
      HloModule jit1d
      reducer {
        lhs = $0[] parameter(0)
        rhs = $0[] parameter(1)
        ROOT out = $1
      }
      sc {
        op0 = $0[$2]{0} parameter(0)
        idx = s32[$3,1]{1,0} parameter(1)
        upd0 = $0[$3,1]{1,0} parameter(2)
        ROOT res = $0[$2]{0} scatter(op0, idx, upd0),
          update_window_dims={1}, inserted_window_dims={},
          scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=reducer
      }
      ENTRY main {
        p0 = $0[$2]{0} parameter(0)
        p1 = s32[$3,1]{1,0} parameter(1)
        p2 = $0[$3,1]{1,0} parameter(2)
        ROOT wrapped_scatter = $0[$2]{0} fusion(p0, p1, p2), kind=kLoop,
          calls=sc, backend_config={"outer_dimension_partitions":["$4"]}
      })hlo",
        ty, body, m, n, wg);
  };

  // Colliding c64 add across 8 workgroups.
  ASSERT_OK_AND_ASSIGN(
      CompiledHostScatterKernel c64_ck,
      build_kernel(make_1d("c64", 4, 128, 8, "c64[] add(lhs, rhs)")));
  if (!c64_ck.host_spec.has_atomic_rmw_64) {
    GTEST_SKIP() << "Host lacks inline 8 through 64 bit atomic RMW.";
  }
  ASSERT_EQ(c64_ck.num_work_groups, 8u);
  {
    alignas(16) std::array<complex64, 4> out = {
        complex64(1.0f, -2.0f), complex64(3.0f, 4.0f), complex64(-5.0f, 6.0f),
        complex64(7.0f, -8.0f)};
    std::array<complex64, 4> expected = out;
    alignas(16) std::array<int32_t, 128> idx{};
    alignas(16) std::array<complex64, 128> upd{};
    for (int k = 0; k < 128; ++k) {
      idx[k] = k % 4;
      upd[k] = complex64(static_cast<float>(k + 1), -0.5f * (k + 1));
      expected[idx[k]] += upd[k];
    }
    std::array<XLA_CPU_KernelArg, 4> args = {{
        {out.data(), sizeof(out)},
        {idx.data(), sizeof(idx)},
        {upd.data(), sizeof(upd)},
        {out.data(), sizeof(out)},
    }};
    RunScatterKernelConcurrently(c64_ck.kernel, c64_ck.num_work_groups, args);
    EXPECT_THAT(out, ElementsAreArray(expected));
  }

  // Rank 0 s32[] scatter with 257 non divisible updates across 7 workgroups.
  ASSERT_OK_AND_ASSIGN(CompiledHostScatterKernel r0_ck, build_kernel(R"hlo(
    HloModule rank0_jit
    reducer {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT out = s32[] add(lhs, rhs)
    }
    sc {
      op0 = s32[] parameter(0)
      idx = s32[257,0]{1,0} parameter(1)
      upd0 = s32[257]{0} parameter(2)
      ROOT res = s32[] scatter(op0, idx, upd0),
        update_window_dims={}, inserted_window_dims={},
        scatter_dims_to_operand_dims={}, index_vector_dim=1, to_apply=reducer
    }
    ENTRY main {
      p0 = s32[] parameter(0)
      p1 = s32[257,0]{1,0} parameter(1)
      p2 = s32[257]{0} parameter(2)
      ROOT wrapped_scatter = s32[] fusion(p0, p1, p2), kind=kLoop, calls=sc,
        backend_config={"outer_dimension_partitions":["7"]}
    })hlo"));
  ASSERT_EQ(r0_ck.num_work_groups, 7u);
  {
    alignas(16) int32_t scalar_out = 10;
    alignas(16) int32_t empty_idx = 0;
    alignas(16) std::array<int32_t, 257> upd{};
    for (int k = 0; k < 257; ++k) {
      upd[k] = k + 1;
    }
    std::array<XLA_CPU_KernelArg, 4> args = {{
        {&scalar_out, sizeof(scalar_out)},
        {&empty_idx, 0},
        {upd.data(), sizeof(upd)},
        {&scalar_out, sizeof(scalar_out)},
    }};
    RunScatterKernelConcurrently(r0_ck.kernel, r0_ck.num_work_groups, args);
    EXPECT_EQ(scalar_out, 10 + 257 * 258 / 2);
  }

  // Negative and out of bounds indices are ignored under concurrency.
  ASSERT_OK_AND_ASSIGN(
      CompiledHostScatterKernel oob_ck,
      build_kernel(make_1d("s32", 8, 64, 4, "s32[] add(lhs, rhs)")));
  ASSERT_EQ(oob_ck.num_work_groups, 4u);
  {
    alignas(16) std::array<int32_t, 8> out = {1, 2, 3, 4, 5, 6, 7, 8};
    std::array<int32_t, 8> expected = out;
    const std::array<int32_t, 6> bad = {
        std::numeric_limits<int32_t>::min(), -99, -1, 8, 9,
        std::numeric_limits<int32_t>::max()};
    alignas(16) std::array<int32_t, 64> idx{};
    alignas(16) std::array<int32_t, 64> upd{};
    for (int k = 0; k < 64; ++k) {
      idx[k] = (k % 3 == 0) ? bad[(k / 3) % bad.size()] : (k % 8);
      upd[k] = k + 10;
      if (idx[k] >= 0 && idx[k] < 8) {
        expected[idx[k]] += upd[k];
      }
    }
    std::array<XLA_CPU_KernelArg, 4> args = {{
        {out.data(), sizeof(out)},
        {idx.data(), sizeof(idx)},
        {upd.data(), sizeof(upd)},
        {out.data(), sizeof(out)},
    }};
    RunScatterKernelConcurrently(oob_ck.kernel, oob_ck.num_work_groups, args);
    EXPECT_THAT(out, ElementsAreArray(expected));
  }

  // Column major {0,1} window slice add with overlapping and out of bounds
  // window starts.
  ASSERT_OK_AND_ASSIGN(CompiledHostScatterKernel col_ck, build_kernel(R"hlo(
    HloModule col_major_jit
    reducer {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT out = s32[] add(lhs, rhs)
    }
    sc {
      op0 = s32[6,4]{0,1} parameter(0)
      idx = s32[48,1]{1,0} parameter(1)
      upd0 = s32[48,2,4]{2,1,0} parameter(2)
      ROOT res = s32[6,4]{0,1} scatter(op0, idx, upd0),
        update_window_dims={1,2}, inserted_window_dims={},
        scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=reducer
    }
    ENTRY main {
      p0 = s32[6,4]{0,1} parameter(0)
      p1 = s32[48,1]{1,0} parameter(1)
      p2 = s32[48,2,4]{2,1,0} parameter(2)
      ROOT wrapped_scatter = s32[6,4]{0,1} fusion(p0, p1, p2), kind=kLoop,
        calls=sc, backend_config={"outer_dimension_partitions":["4"]}
    })hlo"));
  ASSERT_EQ(col_ck.num_work_groups, 4u);
  {
    alignas(16) std::array<int32_t, 24> out{};
    for (int i = 0; i < 24; ++i) {
      out[i] = i + 1;
    }
    std::array<int32_t, 24> expected = out;
    const std::array<int32_t, 7> rows = {-1, 0, 1, 2, 3, 4, 5};
    alignas(16) std::array<int32_t, 48> idx{};
    alignas(16) std::array<int32_t, 48 * 8> upd{};
    for (int k = 0; k < 48; ++k) {
      idx[k] = rows[k % rows.size()];
      for (int dr = 0; dr < 2; ++dr) {
        for (int c = 0; c < 4; ++c) {
          const int32_t val = k * 10 + dr * 4 + c + 1;
          upd[k * 8 + dr * 4 + c] = val;
          if (idx[k] >= 0 && idx[k] + 2 <= 6) {
            expected[c * 6 + (idx[k] + dr)] += val;
          }
        }
      }
    }
    std::array<XLA_CPU_KernelArg, 4> args = {{
        {out.data(), sizeof(out)},
        {idx.data(), sizeof(idx)},
        {upd.data(), sizeof(upd)},
        {out.data(), sizeof(out)},
    }};
    RunScatterKernelConcurrently(col_ck.kernel, col_ck.num_work_groups, args);
    EXPECT_THAT(out, ElementsAreArray(expected));
  }

  // Colliding c128 add and tear free c128 overwrite (im == -re - 1).
  if (c64_ck.host_spec.has_atomic_cmpxchg_128) {
    ASSERT_OK_AND_ASSIGN(
        CompiledHostScatterKernel c128_add,
        build_kernel(make_1d("c128", 4, 128, 8, "c128[] add(lhs, rhs)")));
    ASSERT_EQ(c128_add.num_work_groups, 8u);
    alignas(16) std::array<complex128, 4> out = {
        complex128(2.0, -1.0), complex128(-3.0, 5.0), complex128(4.0, -6.0),
        complex128(-7.0, 8.0)};
    std::array<complex128, 4> expected = out;
    alignas(16) std::array<int32_t, 128> idx{};
    alignas(16) std::array<complex128, 128> upd{};
    for (int k = 0; k < 128; ++k) {
      idx[k] = k % 4;
      upd[k] = complex128(static_cast<double>(k + 1), -0.25 * (k + 1));
      expected[idx[k]] += upd[k];
    }
    std::array<XLA_CPU_KernelArg, 4> args = {{
        {out.data(), sizeof(out)},
        {idx.data(), sizeof(idx)},
        {upd.data(), sizeof(upd)},
        {out.data(), sizeof(out)},
    }};
    RunScatterKernelConcurrently(c128_add.kernel, c128_add.num_work_groups,
                                 args);
    EXPECT_THAT(out, ElementsAreArray(expected));
  }
  if (c64_ck.host_spec.has_atomic_cmpxchg_128 ||
      c64_ck.host_spec.has_atomic_load_store_128) {
    ASSERT_OK_AND_ASSIGN(
        CompiledHostScatterKernel c128_st,
        build_kernel(make_1d("c128", 4, 256, 8, "c128[] copy(rhs)")));
    ASSERT_EQ(c128_st.num_work_groups, 8u);
    alignas(16) std::array<int32_t, 256> idx{};
    alignas(16) std::array<complex128, 256> upd{};
    for (int k = 0; k < 256; ++k) {
      idx[k] = k % 4;
      const double re = static_cast<double>(k + 1);
      upd[k] = complex128(re, -re - 1.0);
    }
    for (int round = 0; round < 32; ++round) {
      alignas(16) std::array<complex128, 4> out{};
      out.fill(complex128(0.0, -1.0));
      std::array<XLA_CPU_KernelArg, 4> args = {{
          {out.data(), sizeof(out)},
          {idx.data(), sizeof(idx)},
          {upd.data(), sizeof(upd)},
          {out.data(), sizeof(out)},
      }};
      RunScatterKernelConcurrently(c128_st.kernel, c128_st.num_work_groups,
                                   args);
      for (int i = 0; i < 4; ++i) {
        const double re = out[i].real();
        const double im = out[i].imag();
        EXPECT_GE(re, 1.0);
        EXPECT_LE(re, 256.0);
        EXPECT_EQ(re, static_cast<double>(static_cast<int>(re)));
        EXPECT_EQ(im, -re - 1.0) << "Torn c128 store at slot " << i;
      }
    }
  }
}

}  // namespace
}  // namespace cpu
}  // namespace xla
