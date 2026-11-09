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

#include "xla/backends/gpu/codegen/rng_get_and_update_state.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/numeric/bits.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

std::vector<uint8_t> RngStateInitialBytes() {
  std::vector<uint8_t> bytes(kRngStateSizeBytes, 0);
  absl::little_endian::Store64(bytes.data(), kRngStateInitialLow);
  return bytes;
}

absl::StatusOr<emitters::KernelArguments>
CreateRngGetAndUpdateStateKernelArguments(
    const HloRngGetAndUpdateStateInstruction* instr,
    const BufferAllocation::Slice& state_slice,
    const BufferAllocation::Slice& out_slice) {
  TF_RET_CHECK(instr != nullptr);
  TF_RET_CHECK(state_slice.size() == kRngStateSizeBytes);
  TF_RET_CHECK(out_slice.size() == kRngStateSizeBytes);
  TF_RET_CHECK(!state_slice.OverlapsWith(out_slice));
  const int64_t alignment =
      GetDefaultBufferAlignment().xla_allocated_buffer_align_bytes;
  auto make_argument = [&](const BufferAllocation::Slice& slice,
                           int64_t slice_index, bool written, bool aliased) {
    emitters::KernelArgument argument(instr->shape(), slice);
    argument.set_slice_index(slice_index);
    argument.set_alignment(alignment);
    argument.set_written(written);
    argument.set_aliased(aliased);
    argument.set_invariant(!aliased);
    return argument;
  };
  // The state is read and written, so it is passed twice: the read only copy
  // reports BufferUse::Read (defined on input), the written copy
  // BufferUse::Write. Both share the slice and must not be noalias.
  std::vector<emitters::KernelArgument> arguments;
  arguments.push_back(make_argument(state_slice, /*slice_index=*/0,
                                    /*written=*/false, /*aliased=*/true));
  arguments.push_back(make_argument(state_slice, /*slice_index=*/0,
                                    /*written=*/true, /*aliased=*/true));
  arguments.push_back(make_argument(out_slice, /*slice_index=*/1,
                                    /*written=*/true, /*aliased=*/false));
  return emitters::KernelArguments(std::move(arguments));
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
CreateRngGetAndUpdateStateMlirModule(
    mlir::MLIRContext& mlir_context,
    const HloRngGetAndUpdateStateInstruction* instr,
    const emitters::KernelArguments& kernel_arguments,
    absl::string_view entry_function_name) {
  TF_RET_CHECK(instr != nullptr);
  TF_RET_CHECK(instr->opcode() == HloOpcode::kRngGetAndUpdateState)
      << "Expected RngGetAndUpdateState instruction: " << instr->ToString();
  TF_RET_CHECK(instr->operand_count() == 0);
  TF_RET_CHECK(Shape::Equal().IgnoreLayout()(instr->shape(),
                                             ShapeUtil::MakeShape(U64, {2})))
      << "Expected u64[2] output shape: " << instr->ToString();
  const std::vector<emitters::KernelArgument>& args = kernel_arguments.args();
  TF_RET_CHECK(args.size() == 3);
  TF_RET_CHECK(args[0].slice() == args[1].slice() && !args[0].written() &&
               args[1].written() && args[2].written())
      << "Expected (state input, state output, output) kernel arguments";

  mlir::OpBuilder builder(&mlir_context);
  mlir::Location loc = mlir::NameLoc::get(builder.getStringAttr(instr->name()));
  mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(loc);
  emitters::SetIndexDataLayout(*module, *instr);

  mlir::func::FuncOp entry_func = emitters::EmitKernelApi(
      *module, kernel_arguments, {instr->shape(), instr->shape()},
      entry_function_name);
  SetBackendKind(&mlir_context, entry_func, BackendKind::kGpu);

  mlir::ImplicitLocOpBuilder b(loc, &mlir_context);
  b.setInsertionPointToStart(entry_func.addEntryBlock());

  mlir::Value state_in = entry_func.getArgument(0);
  mlir::Value state_out = entry_func.getArgument(1);
  mlir::Value out = entry_func.getArgument(2);

  mlir::Type i64_type = b.getI64Type();
  mlir::Value c0 = mlir::arith::ConstantIndexOp::create(b, 0);
  mlir::Value c1 = mlir::arith::ConstantIndexOp::create(b, 1);
  mlir::Value old_low =
      mlir::tensor::ExtractOp::create(b, state_in, mlir::ValueRange{c0});
  mlir::Value old_high =
      mlir::tensor::ExtractOp::create(b, state_in, mlir::ValueRange{c1});

  mlir::Value delta =
      mlir::arith::ConstantIntOp::create(b, i64_type, instr->delta());
  auto add_lo = mlir::arith::AddUIExtendedOp::create(b, old_low, delta);
  mlir::Value new_low = add_lo.getSum();
  mlir::Value carry =
      mlir::arith::ExtUIOp::create(b, i64_type, add_lo.getOverflow());
  mlir::Value new_high = mlir::arith::AddIOp::create(b, old_high, carry);

  mlir::Value updated_state = mlir::tensor::InsertOp::create(
      b, new_low, state_out, mlir::ValueRange{c0});
  updated_state = mlir::tensor::InsertOp::create(b, new_high, updated_state,
                                                 mlir::ValueRange{c1});
  mlir::Value updated_out =
      mlir::tensor::InsertOp::create(b, old_low, out, mlir::ValueRange{c0});
  updated_out = mlir::tensor::InsertOp::create(b, old_high, updated_out,
                                               mlir::ValueRange{c1});
  mlir::func::ReturnOp::create(b, mlir::ValueRange{updated_state, updated_out});
  return module;
}

AsyncThunkSequence EmitRngGetAndUpdateState(
    const HloRngGetAndUpdateStateInstruction* instr,
    IrEmitterContext* ir_emitter_context) {
  TF_RET_CHECK(instr != nullptr);
  TF_RET_CHECK(ir_emitter_context != nullptr);

  std::optional<BufferAllocation::Slice> rng_state_slice =
      ir_emitter_context->rng_state_slice();
  TF_RET_CHECK(rng_state_slice.has_value())
      << "Missing persistent RNG state allocation in IrEmitterContext.";

  ABSL_ASSIGN_OR_RETURN(
      BufferAllocation::Slice out_slice,
      ir_emitter_context->buffer_assignment().GetUniqueTopLevelSlice(instr));
  ABSL_ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                        CreateRngGetAndUpdateStateKernelArguments(
                            instr, *rng_state_slice, out_slice));

  LaunchDimensions launch_dimensions;
  std::string sanitized_kernel_name =
      ir_emitter_context->GetSanitizedUniqueName(std::string(instr->name()));
  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      instr, ir_emitter_context->GetNextThunkId());

  BorrowedMlirContext borrowed_context =
      ir_emitter_context->BorrowMlirContext();
  mlir::MLIRContext* mlir_context = borrowed_context->get();
  mlir_context->appendDialectRegistry(MlirKernelEmitter::GetDialectRegistry());
  mlir_context->loadAllAvailableDialects();

  ABSL_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      CreateRngGetAndUpdateStateMlirModule(
          *mlir_context, instr, kernel_arguments, sanitized_kernel_name));

  bool use_pdl = IsPdlEnabled(
      instr->GetModule()->config().debug_options(),
      ir_emitter_context->gpu_device_info().gpu_compute_capability());
  GpuKernelSpec spec(std::move(sanitized_kernel_name), launch_dimensions,
                     std::move(kernel_arguments), std::nullopt, use_pdl);
  MlirKernelDefinition kernel_def(
      std::move(spec), MlirKernelSource(nullptr, std::move(mlir_module)));
  return ir_emitter_context->kernel_compiler()
      ->Compile(std::move(thunk_info), std::move(kernel_def),
                std::move(borrowed_context), *instr->GetModule())
      .Map([](std::unique_ptr<Thunk> thunk) {
        return ThunkSequence::Of(std::move(thunk));
      });
}

}  // namespace xla::gpu
