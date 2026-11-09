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

#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLALOWERXTILEPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

namespace ttir = ::mlir::triton;
namespace ma = ::mlir::arith;
namespace xtile = ::xla::xtile;

// Get the new arg types of the lowered function by translating memrefs to the
// corresponding pointer types.
llvm::SmallVector<mlir::Type> GetTransformedArgTypes(
    xtile::EntryFuncOp& entry_op) {
  llvm::SmallVector<mlir::Type> arg_types;
  // Tile id is not carried over hence -1.
  arg_types.reserve(entry_op.getNumArguments() - 1U);
  for (const auto& arg : entry_op.getBufferArgs()) {
    mlir::MemRefType memref_type = mlir::cast<mlir::MemRefType>(arg.getType());
    arg_types.push_back(ttir::getPointerType(
        xtile::MemoryStorageType(memref_type.getElementType())));
  }
  mlir::TypeRange opaque_args(entry_op.getOpaqueArgs());
  arg_types.append(opaque_args.begin(), opaque_args.end());
  return arg_types;
}

MemrefToPtrOp CreateMemrefToPtr(mlir::OpBuilder& builder,
                                mlir::TypedValue<mlir::MemRefType> memref) {
  mlir::Type ptr_type = ttir::getPointerType(memref.getType().getElementType());
  return MemrefToPtrOp::create(builder, memref.getLoc(), ptr_type, memref);
}

mlir::Value CreateStoragePtr(mlir::OpBuilder& builder,
                             mlir::TypedValue<mlir::MemRefType> memref) {
  mlir::Value ptr = CreateMemrefToPtr(builder, memref);
  mlir::Type elem_type = memref.getType().getElementType();
  mlir::Type storage_elem_type = xtile::MemoryStorageType(elem_type);
  if (storage_elem_type == elem_type) {
    return ptr;
  }
  return ttir::BitcastOp::create(builder, memref.getLoc(),
                                 ttir::getPointerType(storage_elem_type), ptr);
}

// Rewrite a xtile entry to a func.func with the same body, but with memref
// arguments replaced by pointers.
class XTileEntryToTriton : public mlir::OpRewritePattern<xtile::EntryFuncOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      xtile::EntryFuncOp entry_op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::ModuleOp module = entry_op->getParentOfType<mlir::ModuleOp>();
    mlir::ImplicitLocOpBuilder builder(module->getLoc(), module);
    builder.setInsertionPointToStart(module.getBody());

    const int64_t num_buffer_args = entry_op.getBufferArgs().size();
    mlir::Attribute tile_id_range =
        entry_op.getArgAttr(entry_op.getNumArguments() - 1, "xla.range");
    auto new_arg_types = GetTransformedArgTypes(entry_op);
    auto new_func_op =
        mlir::func::FuncOp::create(builder, entry_op.getName(),
                                   builder.getFunctionType(new_arg_types, {}));

    // Move the old function's body to the new function
    rewriter.inlineRegionBefore(
        entry_op.getBody(), new_func_op.getFunctionBody(), new_func_op.end());

    Block& entry_block = new_func_op.front();
    builder.setInsertionPointToStart(&entry_block);

    SmallVector<BlockArgument> old_args(entry_block.getArguments());
    SmallVector<BlockArgument> new_args(entry_block.addArguments(
        new_arg_types,
        SmallVector<Location>(new_arg_types.size(), entry_op.getLoc())));

    BlockArgument tile_id_arg = old_args.back();

    auto pid = ttir::GetProgramIdOp::create(builder, ttir::ProgramIDDim::X);
    auto pid_idx =
        ma::IndexCastOp::create(builder, builder.getIndexType(), pid);
    if (tile_id_range) {
      pid_idx->setAttr("xla.range", tile_id_range);
    }
    rewriter.replaceAllUsesWith(tile_id_arg, pid_idx);

    // Handle memref arguments.
    for (auto [old_arg, new_arg] :
         llvm::zip(old_args,
                   mlir::ValueRange(new_args).take_front(num_buffer_args))) {
      if (old_arg.use_empty()) {
        continue;
      }
      mlir::MemRefType memref_type =
          mlir::cast<mlir::MemRefType>(old_arg.getType());

      mlir::Value ptr = new_arg;
      mlir::Type elem_type = memref_type.getElementType();
      if (xtile::MemoryStorageType(elem_type) != elem_type) {
        ptr = ttir::BitcastOp::create(builder, ttir::getPointerType(elem_type),
                                      ptr);
      }
      mlir::Value memref_cast =
          PtrToMemrefOp::create(builder, memref_type, ptr);

      // Replace all uses of the old argument with the result of the cast.
      rewriter.replaceAllUsesWith(old_arg, memref_cast);
    }
    // For opaque arguments, we can simply replace all uses with the new
    // argument.
    for (auto [old_arg, new_arg] :
         llvm::zip(mlir::ValueRange(old_args).drop_front(num_buffer_args),
                   mlir::ValueRange(new_args).drop_front(num_buffer_args))) {
      rewriter.replaceAllUsesWith(old_arg, new_arg);
    }

    entry_block.eraseArguments(0, old_args.size());

    rewriter.setInsertionPointToEnd(&entry_block);

    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(
        entry_block.getTerminator());

    rewriter.eraseOp(entry_op);
    return success();
  }
};

class XTileSelectBufferToTriton
    : public mlir::OpRewritePattern<xtile::SelectBufferOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      xtile::SelectBufferOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::TypedValue<mlir::MemRefType> buffer = op.getSource();
    mlir::Value replica_id = op.getReplicaId();
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // Make it a triton pointer.
    // -> !tt.ptr<i64>
    mlir::Value memref_to_ptr = CreateMemrefToPtr(rewriter, buffer);
    mlir::Value replica_id_i64 =
        mlir::arith::IndexCastOp::create(b, b.getI64Type(), replica_id);
    // Add the replica id to the pointer.
    auto ptr_addr = mlir::triton::AddPtrOp::create(
        b, memref_to_ptr.getType(), memref_to_ptr, replica_id_i64);
    // Dereference the outer pointer.
    auto addr_i64 = mlir::triton::LoadOp::create(b, ptr_addr.getResult(),
                                                 /*isVolatile=*/false);
    auto result_memref_type = mlir::cast<mlir::MemRefType>(op.getType());
    mlir::Type elem_type = result_memref_type.getElementType();
    mlir::Type storage_elem_type = xtile::MemoryStorageType(elem_type);
    mlir::NamedAttribute divisibility_attr = xtile::GetDivisibilityAttr(b);
    // The ptr -> i64 -> memref dance.
    mlir::Value final_ptr = mlir::triton::IntToPtrOp::create(
        b, ttir::getPointerType(storage_elem_type), addr_i64.getResult(),
        llvm::ArrayRef<mlir::NamedAttribute>{divisibility_attr});
    if (storage_elem_type != elem_type) {
      final_ptr = ttir::BitcastOp::create(b, ttir::getPointerType(elem_type),
                                          final_ptr);
    }
    mlir::Value ptr_to_memref =
        PtrToMemrefOp::create(b, op.getType(), final_ptr);
    rewriter.replaceOp(op, ptr_to_memref);
    return mlir::success();
  }
};

// Rewrite a xtile extract to a triton_xla extract.
class XTileExtractToTriton
    : public mlir::OpRewritePattern<xtile::ExtractTileOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      xtile::ExtractTileOp extract_op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::Location loc = extract_op.getLoc();
    mlir::MemRefType source_type = extract_op.getSource().getType();
    mlir::RankedTensorType result_type = extract_op.getType();
    mlir::Type logical_elem_type = result_type.getElementType();
    mlir::Type storage_elem_type = xtile::MemoryStorageType(logical_elem_type);

    mlir::Value storage_ptr =
        CreateStoragePtr(rewriter, extract_op.getSource());

    if (result_type.getRank() == 0) {
      mlir::Value scalar_value =
          ttir::LoadOp::create(rewriter, loc, storage_ptr,
                               /*isVolatile=*/false);
      scalar_value = xtile::UnpackFromMemoryStorage(rewriter, loc, scalar_value,
                                                    logical_elem_type);

      rewriter.replaceOpWithNewOp<mlir::tensor::FromElementsOp>(
          extract_op, result_type, scalar_value);
      return mlir::success();
    }

    absl::StatusOr<SmallVector<int64_t>> minor_to_major_or =
        xtile::GetPermutationMinorToMajor(source_type);
    if (!minor_to_major_or.ok()) {
      return rewriter.notifyMatchFailure(extract_op,
                                         minor_to_major_or.status().ToString());
    }
    const SmallVector<int64_t>& minor_to_major = *minor_to_major_or;
    mlir::RankedTensorType storage_tensor_type =
        result_type.clone(storage_elem_type);
    mlir::Value extracted = ExtractOp::create(
        rewriter, loc, storage_tensor_type, storage_ptr,
        extract_op.getOffsets(), extract_op.getFullTileShape(),
        extract_op.getStrides(), source_type.getShape(), minor_to_major);
    extracted = xtile::UnpackFromMemoryStorage(rewriter, loc, extracted,
                                               logical_elem_type);

    rewriter.replaceOp(extract_op, extracted);

    return mlir::success();
  }
};

// Rewrite a xtile insert to a triton_xla insert.
class XTileInsertToTriton : public mlir::OpRewritePattern<xtile::InsertTileOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      xtile::InsertTileOp insert_op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::Location loc = insert_op.getLoc();
    mlir::MemRefType destination_type = insert_op.getDestination().getType();
    mlir::RankedTensorType source_type = insert_op.getSource().getType();

    mlir::Value storage_ptr =
        CreateStoragePtr(rewriter, insert_op.getDestination());

    if (source_type.getRank() == 0) {
      mlir::Value scalar_value =
          mlir::tensor::ExtractOp::create(rewriter, loc, insert_op.getSource());
      scalar_value = xtile::PackToMemoryStorage(rewriter, loc, scalar_value);

      rewriter.replaceOpWithNewOp<ttir::StoreOp>(
          insert_op, storage_ptr, scalar_value, /*mask=*/nullptr);
      return mlir::success();
    }

    absl::StatusOr<SmallVector<int64_t>> minor_to_major_or =
        xtile::GetPermutationMinorToMajor(destination_type);
    if (!minor_to_major_or.ok()) {
      return rewriter.notifyMatchFailure(insert_op,
                                         minor_to_major_or.status().ToString());
    }
    const SmallVector<int64_t>& minor_to_major = *minor_to_major_or;
    mlir::Value tile_to_store =
        xtile::PackToMemoryStorage(rewriter, loc, insert_op.getSource());
    auto triton_insert_op = InsertOp::create(
        rewriter, loc, tile_to_store, storage_ptr, insert_op.getOffsets(),
        insert_op.getFullTileShape(), insert_op.getStrides(),
        destination_type.getShape(), minor_to_major);

    rewriter.replaceOp(insert_op, triton_insert_op);

    return mlir::success();
  }
};

class XTileMaskToTriton : public mlir::OpRewritePattern<xtile::MaskOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      xtile::MaskOp op, mlir::PatternRewriter& rewriter) const override {
    llvm::SmallVector<int64_t> masked_dimensions = op.getMaskedDimensions();
    if (masked_dimensions.size() != 1) {
      return op.emitOpError(
          "triton masking only supports masking over a single dimension");
    }

    int64_t mask_dimension = masked_dimensions.front();
    int64_t mask_bound = op.getBounds()[mask_dimension];
    int64_t masked_dim_size = op.getType().getDimSize(mask_dimension);
    auto iota_type =
        mlir::RankedTensorType::get(masked_dim_size, rewriter.getI32Type());
    auto range = stablehlo::IotaOp::create(rewriter, op.getLoc(), iota_type, 0);
    auto bcast_type = mlir::RankedTensorType::get(op.getType().getShape(),
                                                  iota_type.getElementType());
    auto bcast = stablehlo::BroadcastInDimOp::create(
        rewriter, op.getLoc(), bcast_type, range, {mask_dimension});
    auto constant = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(),
        mlir::DenseElementsAttr::get(bcast_type,
                                     rewriter.getI32IntegerAttr(mask_bound)));
    Value mask = arith::CmpIOp::create(
        rewriter, op.getLoc(), arith::CmpIPredicate::slt, bcast, constant);

    auto mask_value_tensor = mlir::tensor::FromElementsOp::create(
        rewriter, op.getLoc(),
        mlir::RankedTensorType::get({}, op.getValue().getType()),
        op.getValue());
    auto neutral = stablehlo::BroadcastInDimOp::create(
        rewriter, op.getLoc(), op.getType(), mask_value_tensor,
        ArrayRef<int64_t>{});

    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, mask, op.getSource(),
                                                 neutral);

    return mlir::success();
  }
};

class FoldIntoMemrefToPtr : public mlir::OpRewritePattern<MemrefToPtrOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      MemrefToPtrOp op, mlir::PatternRewriter& rewriter) const override {
    // As a transpose doesn't add any offset we can simply fold it into the
    // memref_to_ptr.
    auto transpose = op.getSrc().getDefiningOp<mlir::memref::TransposeOp>();
    if (!transpose) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<MemrefToPtrOp>(op, op.getType(),
                                               transpose.getIn());
    return mlir::success();
  }
};

class FoldBitcastRoundTrip : public mlir::OpRewritePattern<ttir::BitcastOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ttir::BitcastOp op, mlir::PatternRewriter& rewriter) const override {
    auto producer = op.getSrc().getDefiningOp<ttir::BitcastOp>();
    if (!producer || producer.getSrc().getType() != op.getType()) {
      return mlir::failure();
    }
    rewriter.replaceOp(op, producer.getSrc());
    return mlir::success();
  }
};

class TritonXLALowerXTilePass
    : public impl::TritonXLALowerXTilePassBase<TritonXLALowerXTilePass> {
 public:
  using TritonXLALowerXTilePassBase::TritonXLALowerXTilePassBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext* context = &getContext();

    mlir::RewritePatternSet patterns(context);

    patterns.add<XTileEntryToTriton, XTileSelectBufferToTriton,
                 XTileExtractToTriton, XTileInsertToTriton, XTileMaskToTriton,
                 FoldIntoMemrefToPtr, FoldBitcastRoundTrip>(context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

}  // namespace mlir::triton::xla
