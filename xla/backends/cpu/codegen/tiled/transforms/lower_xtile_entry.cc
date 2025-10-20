/* Copyright 2025 The OpenXLA Authors.

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

#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_ops.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_types.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/xtile/ir/xtile_attrs.h"
#include "xla/codegen/xtile/ir/xtile_dialect.h"  // IWYU pragma: keep
#include "xla/codegen/xtile/ir/xtile_ops.h"

namespace xla::cpu {

#define GEN_PASS_DECL_LOWERXTILEENTRYPASS
#define GEN_PASS_DEF_LOWERXTILEENTRYPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

struct LowerXtileEntry : mlir::OpRewritePattern<xtile::EntryFuncOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      xtile::EntryFuncOp op, mlir::PatternRewriter& rewriter) const override {
    llvm::SmallVector<mlir::NamedAttribute> filtered_attrs;
    for (const auto& attr : op->getAttrs()) {
      if (!absl::c_linear_search(mlir::func::FuncOp::getAttributeNames(),
                                 attr.getName())) {
        filtered_attrs.push_back(attr);
      }
    }

    auto new_func_op = rewriter.create<mlir::func::FuncOp>(
        op->getLoc(), op.getSymName(), op.getFunctionType(), filtered_attrs);
    new_func_op.setArgAttrsAttr(op.getArgAttrsAttr());

    // Move the region from the old function to the new one.
    rewriter.inlineRegionBefore(op.getBody(), new_func_op.getBody(),
                                new_func_op.getBody().end());

    // Replace the original operation. Since a function definition does not
    // produce any results, we replace it with an empty list of values.
    rewriter.replaceOp(op, new_func_op);

    return mlir::success();
  }
};

struct LowerXTileEntryReturn
    : mlir::OpRewritePattern<xtile::EntryFuncReturnOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      xtile::EntryFuncReturnOp op,
      mlir::PatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, rewriter.create<mlir::func::ReturnOp>(op->getLoc()));
    return mlir::success();
  }
};

class LowerXTileEntryPass
    : public impl::LowerXTileEntryPassBase<LowerXTileEntryPass> {
 public:
  using LowerXTileEntryPassBase::LowerXTileEntryPassBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    if (WrapInCallFrame(module).failed()) {
      signalPassFailure();
      return;
    }

    patterns.add<LowerXtileEntry, LowerXTileEntryReturn>(context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

 private:
  // Wrap the entry function in another func that abides by the XLA:CPU ABI.
  mlir::LogicalResult WrapInCallFrame(mlir::ModuleOp module) {
    mlir::MLIRContext* context = module.getContext();
    mlir::ImplicitLocOpBuilder builder(module->getLoc(), module);

    for (auto entry_func : module.getOps<xtile::EntryFuncOp>()) {
      if (!entry_func.symbolKnownUseEmpty(module)) {
        module->emitError() << "entry function is itself called.";
        return mlir::failure();
      }

      llvm::StringRef kernel_name = entry_func.getName();
      std::string kernel_impl_name =
          absl::StrCat(absl::AlphaNum(entry_func.getName()), "_impl");
      entry_func.setName(kernel_impl_name);
      entry_func.setPrivate();
      entry_func->setAttr(
          "llvm.linkage",
          mlir::LLVM::LinkageAttr::get(context, mlir::LLVM::Linkage::Internal));
      entry_func->setAttr("always_inline", builder.getUnitAttr());

      auto call_frame_type = CallFrameType::get(context);
      auto error_type = ErrorType::get(context);
      builder.setInsertionPointToStart(module.getBody());
      mlir::func::FuncOp kernel_func = builder.create<mlir::func::FuncOp>(
          kernel_name,
          builder.getFunctionType({call_frame_type}, {error_type}));

      builder.setInsertionPointToStart(kernel_func.addEntryBlock());

      auto call_frame = mlir::cast<mlir::TypedValue<CallFrameType>>(
          kernel_func.getArgument(0));
      llvm::SmallVector<mlir::Value> call_args;
      for (const auto& [idx, arg] :
           llvm::enumerate(entry_func.getBufferArgs())) {
        LoadOp load = builder.create<LoadOp>(arg.getType(), call_frame, idx);
        call_args.push_back(load);
      }

      auto tile_info = entry_func->getAttrOfType<xla::xtile::TilingInfoAttr>(
          "xtile.tiling_info");

      if (!tile_info) {
        entry_func->emitError() << "missing tiling info.";
        return mlir::failure();
      }
      int32_t tile_count = tile_info.getTileCount();
      int32_t tiles_per_workgroup = tile_info.getTilesPerWorkgroup();

      mlir::Value tile_count_value =
          builder.create<mlir::arith::ConstantIndexOp>(tile_count);
      mlir::Value tiles_per_workgroup_value =
          builder.create<mlir::arith::ConstantIndexOp>(tiles_per_workgroup);
      mlir::Value workgroup_id = builder.create<ExtractWorkgroupIdOp>(
          builder.getIndexType(), call_frame, WorkGroupDimension::x);

      mlir::Value start_tile_id = builder.create<mlir::arith::MulIOp>(
          builder.getIndexType(), workgroup_id, tiles_per_workgroup_value);
      mlir::Value bounded_start_tile_id = builder.create<mlir::arith::MinSIOp>(
          builder.getIndexType(), start_tile_id, tile_count_value);

      mlir::Value end_tile_id = builder.create<mlir::arith::AddIOp>(
          builder.getIndexType(), start_tile_id, tiles_per_workgroup_value);
      mlir::Value bounded_end_tile_id = builder.create<mlir::arith::MinSIOp>(
          builder.getIndexType(), end_tile_id, tile_count_value);

      mlir::Value step = builder.create<mlir::arith::ConstantIndexOp>(1);

      auto for_op = builder.create<mlir::scf::ForOp>(bounded_start_tile_id,
                                                     bounded_end_tile_id, step);
      {
        mlir::ImplicitLocOpBuilder body_builder(entry_func->getLoc(),
                                                entry_func);
        body_builder.setInsertionPointToStart(for_op.getBody());

        call_args.push_back(for_op.getInductionVar());

        body_builder.create<mlir::func::CallOp>(kernel_impl_name,
                                                mlir::TypeRange(), call_args);
      }

      auto error = builder.create<cpu::SuccessOp>(error_type);
      builder.create<mlir::func::ReturnOp>(error.getResult());
    }

    return mlir::success();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateLowerXTileEntryPass() {
  return std::make_unique<LowerXTileEntryPass>();
}

}  // namespace xla::cpu
