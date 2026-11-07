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

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"  // IWYU pragma: keep
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"

namespace xla::cpu {

#define GEN_PASS_DEF_LEGALIZEI1STORAGEPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

bool IsI1MemRef(mlir::Type type) {
  auto memref_type = mlir::dyn_cast<mlir::MemRefType>(type);
  return memref_type && memref_type.getElementType().isInteger(1);
}

mlir::LogicalResult RewriteEntryFuncSignature(xtile::EntryFuncOp op,
                                              mlir::PatternRewriter& rewriter) {
  auto input_types = op.getFunctionType().getInputs();
  if (llvm::none_of(input_types, IsI1MemRef)) {
    return rewriter.notifyMatchFailure(op, "no i1 memref arguments");
  }
  if (op.getBody().empty()) {
    return rewriter.notifyMatchFailure(op, "empty function body");
  }

  mlir::Location loc = op.getLoc();
  mlir::Block* entry_block = &op.getBody().front();
  llvm::SmallVector<mlir::Type> new_operand_types(input_types);

  rewriter.setInsertionPointToStart(entry_block);
  for (auto&& [index, operand_type] : llvm::enumerate(new_operand_types)) {
    if (!IsI1MemRef(operand_type)) {
      continue;
    }
    auto memref_type = mlir::cast<mlir::MemRefType>(operand_type);
    mlir::BlockArgument func_argument = op.getArgument(index);
    auto cast_to_i1 = mlir::UnrealizedConversionCastOp::create(
        rewriter, loc, memref_type, func_argument);
    rewriter.replaceAllUsesExcept(func_argument, cast_to_i1.getResult(0),
                                  cast_to_i1);
    operand_type = mlir::MemRefType::Builder(memref_type)
                       .setElementType(xtile::MemoryStorageType(
                           memref_type.getElementType()));
  }

  rewriter.modifyOpInPlace(op, [&] {
    for (auto [arg, arg_type] :
         llvm::zip(entry_block->getArguments(), new_operand_types)) {
      arg.setType(arg_type);
    }
    op.setType(rewriter.getFunctionType(new_operand_types,
                                        op.getFunctionType().getResults()));
  });
  return mlir::success();
}

mlir::LogicalResult RewriteExtractTileOp(xtile::ExtractTileOp op,
                                         mlir::PatternRewriter& rewriter) {
  mlir::RankedTensorType result_type = op.getType();
  if (!result_type.getElementType().isInteger(1)) {
    return rewriter.notifyMatchFailure(op, "not an i1 tile extract");
  }
  auto cast_op =
      op.getSource().getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!cast_op || cast_op.getInputs().size() != 1) {
    return rewriter.notifyMatchFailure(op,
                                       "source is not an i8 -> i1 memref cast");
  }

  mlir::Location loc = op.getLoc();
  mlir::Value i8_source = cast_op.getInputs().front();
  mlir::RankedTensorType i8_tensor_type =
      result_type.clone(xtile::MemoryStorageType(result_type.getElementType()));
  auto loaded_i8 = xtile::ExtractTileOp::create(
      rewriter, loc, i8_tensor_type, i8_source, op.getOffsets(),
      op.getFullTileShapeAttr(), op.getStridesAttr());
  rewriter.replaceOp(
      op, xtile::UnpackFromMemoryStorage(rewriter, loc, loaded_i8,
                                         result_type.getElementType()));
  return mlir::success();
}

mlir::LogicalResult RewriteInsertTileOp(xtile::InsertTileOp op,
                                        mlir::PatternRewriter& rewriter) {
  mlir::RankedTensorType source_type = op.getSource().getType();
  if (!source_type.getElementType().isInteger(1)) {
    return rewriter.notifyMatchFailure(op, "not an i1 tile insert");
  }
  auto cast_op =
      op.getDestination().getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!cast_op || cast_op.getInputs().size() != 1) {
    return rewriter.notifyMatchFailure(
        op, "destination is not an i8 -> i1 memref cast");
  }

  mlir::Location loc = op.getLoc();
  mlir::Value i8_dest = cast_op.getInputs().front();
  mlir::Value ext_i8 =
      xtile::PackToMemoryStorage(rewriter, loc, op.getSource());
  rewriter.replaceOpWithNewOp<xtile::InsertTileOp>(
      op, ext_i8, i8_dest, op.getOffsets(), op.getFullTileShapeAttr(),
      op.getStridesAttr());
  return mlir::success();
}

struct LegalizeI1StoragePass
    : public impl::LegalizeI1StoragePassBase<LegalizeI1StoragePass> {
  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add(RewriteEntryFuncSignature);
    patterns.add(RewriteExtractTileOp);
    patterns.add(RewriteInsertTileOp);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    mlir::WalkResult walk_result = getOperation().walk([](mlir::Operation* op) {
      if (auto entry = mlir::dyn_cast<xtile::EntryFuncOp>(op)) {
        if (llvm::any_of(entry.getFunctionType().getInputs(), IsI1MemRef)) {
          entry.emitOpError("failed to legalize i1 memref argument");
          return mlir::WalkResult::interrupt();
        }
      }
      if (auto extract = mlir::dyn_cast<xtile::ExtractTileOp>(op);
          extract && extract.getType().getElementType().isInteger(1)) {
        extract.emitOpError("failed to legalize i1 tile extract");
        return mlir::WalkResult::interrupt();
      }
      if (auto insert = mlir::dyn_cast<xtile::InsertTileOp>(op);
          insert &&
          insert.getSource().getType().getElementType().isInteger(1)) {
        insert.emitOpError("failed to legalize i1 tile insert");
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (walk_result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace xla::cpu
