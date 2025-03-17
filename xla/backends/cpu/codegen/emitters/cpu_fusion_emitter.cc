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

#include "xla/backends/cpu/codegen/emitters/cpu_fusion_emitter.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Linker/Linker.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_dialect.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_ops.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_types.h"
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/mlir/tools/mlir_replay/public/compiler_trace.pb.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/dump.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {
namespace {

using llvm::SmallVector;
using mlir::func::FuncOp;

bool Needs64Bits(const Shape& shape) {
  return shape.IsArray() ? !IsInt32(ShapeUtil::ElementsIn(shape))
                         : absl::c_any_of(shape.tuple_shapes(), Needs64Bits);
}

bool Is64BitIndex(const HloInstruction* instr, int operand) {
  const auto& shape = instr->operand(operand)->shape();
  return shape.element_type() == PrimitiveType::S64 ||
         shape.element_type() == PrimitiveType::U64;
}

bool Needs64BitIndices(const HloComputation* computation) {
  for (auto* instr : computation->instructions()) {
    // Check if any HLO instructions directly take 64 bit indices as operands.
    switch (instr->opcode()) {
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice:
        for (int i = 1; i < instr->operand_count(); ++i) {
          if (Is64BitIndex(instr, i)) return true;
        }
        break;
      case HloOpcode::kGather:
      case HloOpcode::kScatter: {
        int indices_operand_index = instr->operand_count() / 2;
        if (Is64BitIndex(instr, indices_operand_index)) return true;
        break;
      }
      default:
        break;
    }

    if (Needs64Bits(instr->shape()) ||
        absl::c_any_of(instr->called_computations(), Needs64BitIndices)) {
      return true;
    }
  }
  return false;
}
}  // namespace

absl::StatusOr<CpuFusionEmissionResult> CpuFusionEmitterBase::Emit() const {
  // Single-threaded for now.
  TF_ASSIGN_OR_RETURN(auto module,
                      CreateLLVMModule(*mlir_context_, *llvm_context_, *fusion_,
                                       buffer_assignment_));

  const HloModule* hlo_module = fusion_->GetModule();
  if (hlo_module == nullptr) {
    return Internal("HloModule is null");
  }
  // Create a Kernel API Builder and a throwaway kernel prototype in order to
  // extract useful info from them, e.g. noalias, invariant_arguments and
  // entry function attributes.
  // TODO(ecg): find a way to obtain the same info without wasting work by
  // creating a throwaway module. All of this additional info should probably be
  // explicit in the generated MLIR, not added afterwards like we're doing here.
  // TODO(ecg): some attributes on the final loads are missing wrt those
  // generated via KernelApiIrBuilder, e.g. noalias. Add them.
  KernelApiIrBuilder kernel_api_ir_builder(
      *llvm_context_,
      KernelApiIrBuilder::Options::FromHloModuleConfig(hlo_module->config()));
  std::unique_ptr<llvm::Module> throwaway_llvm_module =
      KernelApiIrBuilder::CreateModule(
          absl::StrCat(fusion_->name(), "_throwaway_module"), *llvm_context_);
  TF_ASSIGN_OR_RETURN(KernelApiIrBuilder::KernelPrototype kernel_prototype,
                      kernel_api_ir_builder.EmitKernelPrototype(
                          *throwaway_llvm_module, fusion_, &buffer_assignment_,
                          "_throwaway_kernel_prototype"));
  llvm::Function* kernel_function = module->getFunction(fusion_->name());
  kernel_api_ir_builder.SetKernelFunctionAttributes(kernel_function);

  CpuFusionEmissionResult result;
  result.llvm_module = std::move(module);
  result.invariant_arguments = std::move(kernel_prototype.invariant_arguments);
  return result;
}

absl::StatusOr<std::unique_ptr<llvm::Module>>
CpuFusionEmitterBase::CreateLLVMModule(
    mlir::MLIRContext& mlir_context, llvm::LLVMContext& llvm_context,
    const HloFusionInstruction& fusion,
    const BufferAssignment& buffer_assignment) const {
  TF_ASSIGN_OR_RETURN(auto module,
                      CreateMLIRModule(mlir_context, fusion,
                                       std::string(fusion.name()) + "_entry",
                                       buffer_assignment));

  mlir::PassManager pm(&mlir_context);
  if (VLOG_IS_ON(5)) {
    mlir_context.disableMultithreading();
    pm.enableIRPrinting();
  }
  AddXlaOpsOptimizationPasses(pm);
  AddLoopTransformationPasses(pm);
  AddLoweringPasses(pm);
  auto pipeline_status = RunPassPipeline(module.get(), pm, nullptr);
  TF_RETURN_IF_ERROR(pipeline_status);

  // At the end of the MLIR pipeline we must have just one function definition.
  // This helps later compilation stages, where each thunk is assumed to be a
  // standalone function.
  auto num_llvm_function_defs = [](mlir::ModuleOp m) {
    int count = 0;
    m.walk([&count](mlir::LLVM::LLVMFuncOp func) {
      if (!func.getBody().empty()) {
        count++;
      }
      return mlir::WalkResult::advance();
    });
    return count;
  };
  if (int num_funcs = num_llvm_function_defs(module.get()); num_funcs != 1) {
    return Internal("The module must have just one function definition; has %d",
                    num_funcs);
  }

  auto llvm_module = mlir::translateModuleToLLVMIR(module.get(), llvm_context);
  TF_RET_CHECK(llvm_module != nullptr)
      << "Failed to translate module to LLVM IR.";
  llvm_module->setDataLayout(llvm_module->getDataLayout());

  return llvm_module;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
CpuFusionEmitterBase::CreateMLIRModule(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion,
    const std::string& entry_function_name,
    const BufferAssignment& buffer_assignment,
    mlir::interpreter::MlirCompilationTrace* trace) const {
  context.loadDialect<mlir::DLTIDialect, mlir::affine::AffineDialect,
                      mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                      mlir::func::FuncDialect, mlir::math::MathDialect,
                      xla::cpu::XlaCpuDialect, mlir::mhlo::MhloDialect,
                      mlir::scf::SCFDialect, mlir::LLVM::LLVMDialect,
                      mlir::tensor::TensorDialect, mlir::vector::VectorDialect,
                      xla::XlaDialect>();
  mlir::DialectRegistry registry;
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::func::registerInlinerExtension(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  context.appendDialectRegistry(registry);

  mlir::OpBuilder builder(&context);
  auto loc = mlir::NameLoc::get(builder.getStringAttr(fusion.name()));
  mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(loc);

  // Create the entry function.
  TF_ASSIGN_OR_RETURN(
      std::vector<KernelApiIrBuilder::KernelParameter> arguments,
      KernelApiIrBuilder::GetKernelArgumentsParameters(&fusion,
                                                       &buffer_assignment));
  TF_ASSIGN_OR_RETURN(std::vector<KernelApiIrBuilder::KernelParameter> results,
                      KernelApiIrBuilder::GetKernelResultsParameters(
                          &fusion, &buffer_assignment));

  // TBD: Annotate tensors with the buffer indices. This way, the buffer
  // propagation pass can clean them up later.
  auto get_arg_attrs = [&](int index, BufferAllocation::Slice& slice,
                           bool is_result) -> absl::StatusOr<mlir::Attribute> {
    SmallVector<mlir::NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr(
        "xla.slice_index",
        builder.getIndexAttr(index + (is_result ? arguments.size() : 0))));
    attrs.push_back(builder.getNamedAttr(
        mlir::LLVM::LLVMDialect::getDereferenceableAttrName(),
        builder.getIndexAttr(slice.size())));
    attrs.push_back(builder.getNamedAttr(
        mlir::LLVM::LLVMDialect::getAlignAttrName(), builder.getIndexAttr(32)));
    return builder.getDictionaryAttr(attrs);
  };

  // First argument is the thread id.
  SmallVector<mlir::Attribute> arg_attrs{builder.getDictionaryAttr(
      builder.getNamedAttr("xla.invariant", builder.getUnitAttr()))};
  SmallVector<mlir::Type> param_types{builder.getIndexType()};

  for (const auto& [index, arg] : llvm::enumerate(arguments)) {
    param_types.push_back(emitters::TensorShapeToMlirType(arg.shape, builder));
    TF_ASSIGN_OR_RETURN(
        arg_attrs.emplace_back(),
        get_arg_attrs(index - 1, arg.slice, /*is_result=*/false));
  }

  auto result_types = emitters::ShapeToMlirTypes(fusion.shape(), builder);
  param_types.append(result_types.begin(), result_types.end());
  for (const auto& [index, result] : llvm::enumerate(results)) {
    TF_ASSIGN_OR_RETURN(arg_attrs.emplace_back(),
                        get_arg_attrs(index, result.slice, /*is_result=*/true));
  }

  builder.setInsertionPointToStart(module->getBody());
  auto entry_func = builder.create<FuncOp>(
      loc, entry_function_name,
      mlir::FunctionType::get(&context, param_types, result_types),
      /*sym_visibility=*/mlir::StringAttr{},
      mlir::ArrayAttr::get(&context, arg_attrs),
      /*res_attrs=*/mlir::ArrayAttr{});
  entry_func->setAttr("xla.entry", mlir::UnitAttr::get(&context));
  SetBackendKind(&context, entry_func, xla::BackendKind::kCpu);
  entry_func.setPrivate();

  // Create wrapper for the entry function. This function has one call_frame
  // argument and call the entry function.
  auto error_type = cpu::ErrorType::get(&context);
  auto call_frame_type = CallFrameType::get(mlir_context_);
  auto call_frame_func = builder.create<FuncOp>(
      loc, fusion.name(),
      builder.getFunctionType(/*arg_types=*/{call_frame_type},
                              /*result_types=*/{error_type}));
  builder.setInsertionPointToStart(call_frame_func.addEntryBlock());
  mlir::Value call_frame_arg = call_frame_func.getArgument(0);
  SmallVector<mlir::Value> extracted_values;
  extracted_values.reserve(arguments.size() + results.size() + 1);
  extracted_values.push_back(builder.create<cpu::ThreadIdOp>(
      loc, builder.getIndexType(), call_frame_arg));

  for (int i = 1; i < param_types.size(); ++i) {
    extracted_values.push_back(builder.create<cpu::LoadOp>(
        loc, param_types[i], call_frame_arg, i - 1));
  }
  auto call_results =
      builder.create<xla::PureCallOp>(loc, entry_func, extracted_values);
  call_results->setAttr("noinline", mlir::UnitAttr::get(&context));
  for (auto [index, call_result] : llvm::enumerate(call_results.getResults())) {
    builder.create<cpu::StoreOp>(loc, call_result, call_frame_arg,
                                 index + arguments.size());
  }
  auto error = builder.create<cpu::SuccessOp>(loc, error_type);
  builder.create<mlir::func::ReturnOp>(loc, error.getResult());

  TF_RETURN_IF_ERROR(EmitMlir(module.get(), entry_func, fusion));
  return module;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
absl::Status CpuFusionEmitterBase::EmitMlir(
    mlir::ModuleOp module, FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  std::vector<emitters::EpilogueSpecification> epilogues =
      GetEpilogues(fusion, module->getContext());
  emitters::PartitionedComputations computations(
      fusion.fused_instructions_computation(), module->getContext(),
      /*epilogues=*/epilogues);
  auto subgraph_to_mlir_fn = computations.DeclareFunctions(module);

  // Erase subgraphs for all heroes that aren't used anywhere else. This is
  // necessary because the instructions may not have elemental implementations
  // (scatter).
  for (const auto& epilogue : epilogues) {
    for (auto* custom : epilogue.heroes) {
      if (custom->user_count() == 0) {
        subgraph_to_mlir_fn.extract(&computations.FindSubgraph(custom))
            .mapped()
            .erase();
      }
    }
  }

  // The epilogue functions replace the root tuple.
  auto* root = fusion.fused_instructions_computation()->root_instruction();
  if (root->opcode() == HloOpcode::kTuple && !epilogues.empty()) {
    subgraph_to_mlir_fn.extract(&computations.FindSubgraph(root))
        .mapped()
        .erase();
  }

  auto call_targets =
      computations.CreateCallTargetProvider(subgraph_to_mlir_fn);
  for (const auto& comp : computations.partitioned_computations()) {
    for (const auto& subgraph : comp.subgraphs()) {
      if (subgraph_to_mlir_fn.contains(&subgraph)) {
        TF_RETURN_IF_ERROR(emitters::SubgraphToMlirFunction(
            comp, subgraph, subgraph_to_mlir_fn[&subgraph], call_targets));
      }
    }
  }
  for (const auto& epilogue : computations.epilogues()) {
    if (epilogue.roots.empty()) continue;
    TF_RETURN_IF_ERROR(emitters::SubgraphToMlirFunction(
        computations.FindPartitionedComputation(
            fusion.fused_instructions_computation()),
        epilogue, subgraph_to_mlir_fn[&epilogue], call_targets));
  }

  int index_bitwidth =
      Needs64BitIndices(fusion.fused_instructions_computation()) ? 64 : 32;
  mlir::OpBuilder b(module->getContext());
  auto index_layout = mlir::DataLayoutEntryAttr::get(
      b.getIndexType(), b.getI32IntegerAttr(index_bitwidth));
  module->setAttr(
      mlir::DLTIDialect::kDataLayoutAttrName,
      mlir::DataLayoutSpecAttr::get(module->getContext(), {index_layout}));

  return EmitEntryFunction(computations, call_targets, entry_function, fusion);
}

absl::Status CpuFusionEmitterBase::RunPassPipeline(
    mlir::ModuleOp module, mlir::PassManager& pm,
    mlir::interpreter::MlirCompilationTrace* trace) const {
  if (VLOG_IS_ON(5)) {
    module.getContext()->disableMultithreading();
    pm.enableIRPrinting();
  }
  tsl::StatusScopedDiagnosticHandler diagnostic_handler(module.getContext());
  (void)pm.run(module);
  return diagnostic_handler.consumeStatus();
}

void AddXlaOpsOptimizationPasses(mlir::OpPassManager& pm) {
  pm.addNestedPass<FuncOp>(emitters::CreateSimplifyArithPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(emitters::CreateEraseDeadFunctionsPass());
  pm.addPass(mlir::createCSEPass());
}

void AddLoopTransformationPasses(mlir::OpPassManager& pm) {
  pm.addNestedPass<FuncOp>(emitters::CreateLowerXlaToScfPass());
  pm.addPass(mlir::createInlinerPass({}, [&](mlir::OpPassManager& pm) {
    // CSE after inlining because inlining can introduce duplicates.
    pm.addPass(mlir::createCSEPass());
  }));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<FuncOp>(emitters::CreateLowerXlaLoopsToScfPass());
  pm.addPass(mlir::mhlo::createConvertToSignlessPass());
  pm.addPass(emitters::CreatePropagateSliceIndicesPass());
  pm.addPass(emitters::CreateFlattenTensorsPass());
  // We need LICM before unswitching loops, because our loop unswitcher only
  // detects for loops with a single if inside them.
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addNestedPass<FuncOp>(emitters::CreateUnswitchLoopsPass());
  // We need LICM again after unswitching, because that can introduce new
  // opportunities for LICM. This would not be necessary if LICM also moved
  // instructions over ifs.
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addNestedPass<FuncOp>(
      emitters::CreateVectorizeLoadsAndStoresPass(/*target_type=*/"cpu"));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void AddLoweringPasses(mlir::OpPassManager& pm) {
  pm.addNestedPass<FuncOp>(emitters::CreateConvertPureCallOpsPass());
  pm.addPass(cpu::CreateLowerToLLVMPass());
  pm.addPass(emitters::CreateLowerTensorsPass(/*target_type=*/"cpu"));
  pm.addPass(mlir::createConvertComplexToStandardPass());
  pm.addPass(emitters::CreateMergePointersToSameSlicePass());

  // LowerTensors creates new affine.apply ops. Fold and CSE them so
  // simplify-affine has maximally folded expressions to work with.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<FuncOp>(emitters::CreateSimplifyArithPass());
  pm.addPass(emitters::CreateSimplifyAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());

  // simplify-affine lowers most affine.apply ops, but if it can't prove a
  // division or modulo is unsigned, affine.apply ops will remain.
  pm.addPass(mlir::createLowerAffinePass());

  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCSEPass());

  pm.addPass(emitters::CreateExpandFloatOpsPass());
  pm.addPass(emitters::CreateEraseDeadFunctionsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(emitters::CreateLowerToLLVMPass(/*target_type=*/"cpu"));
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

using mlir::AffineExpr;

IndexingMap GetDefaultIndexingMap(absl::Span<const int64_t> thread_tile_sizes,
                                  absl::Span<const int64_t> shape,
                                  mlir::MLIRContext* mlir_context) {
  CHECK_EQ(thread_tile_sizes.size(), shape.size())
      << "thread_tile_sizes and shape must have the same size";
  SmallVector<int64_t> thread_tile_counts;
  thread_tile_counts.reserve(thread_tile_sizes.size());
  for (auto [tile_size, dim_size] : llvm::zip(thread_tile_sizes, shape)) {
    thread_tile_counts.push_back(CeilDiv(dim_size, tile_size));
  }
  // Delinearize thread_expr w.r.t. number of thread tiles per dimension.
  auto thread_expr = mlir::getAffineDimExpr(0, mlir_context);
  SmallVector<AffineExpr, 4> thread_ids =
      DelinearizeInBoundsIndex(thread_expr, thread_tile_counts);
  SmallVector<AffineExpr, 4> result;
  result.reserve(thread_ids.size());
  auto linear_index = mlir::getAffineSymbolExpr(0, mlir_context);
  SmallVector<AffineExpr, 4> indices_in_tile =
      DelinearizeInBoundsIndex(linear_index, thread_tile_sizes);
  SmallVector<std::pair<AffineExpr, Interval>, 4> constraints;
  constraints.reserve(thread_ids.size());
  for (auto [tile_size, thread_id, index_in_tile, dim] :
       llvm::zip(thread_tile_sizes, thread_ids, indices_in_tile, shape)) {
    result.push_back(thread_id * tile_size + index_in_tile);
    constraints.push_back(std::make_pair(result.back(), Interval{0, dim - 1}));
  }
  int64_t num_threads = Product(thread_tile_counts);
  int64_t num_tile_elements = Product(thread_tile_sizes);

  auto affine_map = mlir::AffineMap::get(/*num_dims=*/1, /*num_symbols=*/1,
                                         result, mlir_context);
  return IndexingMap(
      affine_map, {IndexingMap::Variable({0, num_threads - 1, "thread_id"})},
      {IndexingMap::Variable({0, num_tile_elements - 1, "linear_index"})}, {},
      constraints);
}

int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
}  // namespace cpu
}  // namespace xla
