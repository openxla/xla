/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/mosaic/dialect/tpu/transforms/serde.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/serde.h"

namespace mlir::tpu {

namespace {

constexpr StringRef kMangledDialect = "stable_mosaic.";
constexpr StringRef kVersionAttrName = "stable_mosaic.version";

using UpgradeRuleType = jaxlib::mosaic::UpgradeRuleType;
using DowngradeRuleType = jaxlib::mosaic::DowngradeRuleType;

std::optional<int64_t> getSignedIntAttr(Attribute attr) {
  auto int_attr = dyn_cast_if_present<IntegerAttr>(attr);
  if (!int_attr || !int_attr.getType().isSignlessInteger() ||
      int_attr.getValue().getBitWidth() == 0) {
    return std::nullopt;
  }
  return int_attr.getValue().trySExtValue();
}

FailureOr<ArrayRef<int32_t>> getValidatedSegmentSizes(
    Operation* op, size_t expected_size, ArrayRef<size_t> optional_segs = {}) {
  auto segment_attr =
      op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
  if (!segment_attr) {
    return op->emitError("Missing or invalid AttrSizedOperandSegments");
  }
  ArrayRef<int32_t> sizes = segment_attr.asArrayRef();
  if (sizes.size() != expected_size) {
    return op->emitError("Unexpected size of AttrSizedOperandSegments");
  }
  int64_t total = 0;
  for (int32_t size : sizes) {
    if (size < 0) {
      return op->emitError("Negative entry in AttrSizedOperandSegments");
    }
    total += size;
  }
  if (total != static_cast<int64_t>(op->getNumOperands())) {
    return op->emitError(
        "AttrSizedOperandSegments sum does not match operand count");
  }
  for (size_t seg : optional_segs) {
    if (sizes[seg] > 1) {
      return op->emitError(
          "Invalid optional segment count in AttrSizedOperandSegments");
    }
  }
  return sizes;
}

unsigned getSegmentOffset(ArrayRef<int32_t> sizes, size_t seg) {
  unsigned offset = 0;
  for (size_t i = 0; i < seg; ++i) {
    offset += static_cast<unsigned>(sizes[i]);
  }
  return offset;
}

CoreType getTargetCoreType(Operation* op, ArrayRef<int32_t> sizes,
                           size_t target_sem_seg) {
  if (sizes[target_sem_seg] == 1) {
    Value target_sem = op->getOperand(getSegmentOffset(sizes, target_sem_seg));
    if (auto memref_type = dyn_cast<MemRefType>(target_sem.getType())) {
      if (auto space = dyn_cast_if_present<MemorySpaceAttr>(
              memref_type.getMemorySpace())) {
        if (std::optional<CoreType> core = space.getCoreType()) {
          return *core;
        }
      }
    }
  }
  return GetCoreTypeOfParentOp(*op);
}

LogicalResult store_upgrade(Operation* op, int version, bool&) {
  if (version < 11) {
    op->setAttr("add", mlir::BoolAttr::get(op->getContext(), false));
  }
  return success();
}

LogicalResult store_downgrade(Operation* op, int version) {
  if (version < 11) {
    auto add_attr = op->getAttrOfType<BoolAttr>("add");
    if (!add_attr) {
      return op->emitError("Missing or invalid add attribute");
    }
    if (add_attr.getValue()) {
      return op->emitError(
          "Can only downgrade below version 11 when add is not set");
    }
    op->removeAttr("add");
  }
  return success();
}

LogicalResult dynamic_gather_upgrade(Operation* op, int version, bool&) {
  if (version < 5) {
    auto dimension_attr = op->getAttrOfType<IntegerAttr>("dimension");
    const std::optional<int64_t> dimension = getSignedIntAttr(dimension_attr);
    if (!dimension_attr || !dimension_attr.getType().isSignlessInteger(32) ||
        !dimension.has_value()) {
      return op->emitError("Missing or invalid dimension attribute");
    }
    op->removeAttr("dimension");
    op->setAttr("dimensions",
                DenseI32ArrayAttr::get(op->getContext(),
                                       {static_cast<int32_t>(*dimension)}));
  }
  return success();
}

LogicalResult dynamic_gather_downgrade(Operation* op, int version) {
  if (version < 5) {
    auto dimensions_attr = op->getAttrOfType<DenseI32ArrayAttr>("dimensions");
    if (!dimensions_attr) {
      return op->emitError("Missing or invalid dimensions attribute");
    }
    const ArrayRef<int32_t> dimensions = dimensions_attr.asArrayRef();
    if (dimensions.size() != 1) {
      return op->emitError(
          "Can only downgrade below version 5 when a single dimension is "
          "specified.");
    }
    const int32_t dimension = dimensions.front();
    op->removeAttr("dimensions");
    op->setAttr("dimension",
                mlir::IntegerAttr::get(
                    mlir::IntegerType::get(op->getContext(), 32), dimension));
  }
  return success();
}

LogicalResult upgrade_reduction_kind(Operation* op, ReductionKind int_max_kind,
                                     ReductionKind int_min_kind) {
  if (op->getNumOperands() == 0) {
    return op->emitError("Missing input operand");
  }
  const auto input_type = dyn_cast<VectorType>(op->getOperand(0).getType());
  if (!input_type) {
    return op->emitError("Invalid input type");
  }
  const Type elem_type = input_type.getElementType();
  auto kind_attr = op->getAttrOfType<ReductionKindAttr>("kind");
  if (!kind_attr) {
    return op->emitError("Missing or invalid kind attribute");
  }
  std::optional<ReductionKind> new_kind;
  switch (kind_attr.getValue()) {
    case ReductionKind::kMax_DEPRECATED:
      if (isa<FloatType>(elem_type)) {
        new_kind = ReductionKind::kMaxF;
      } else if (isa<IntegerType>(elem_type)) {
        new_kind = int_max_kind;
      } else {
        return op->emitError("Unsupported element type for max reduction");
      }
      break;
    case ReductionKind::kMin_DEPRECATED:
      if (isa<FloatType>(elem_type)) {
        new_kind = ReductionKind::kMinF;
      } else if (isa<IntegerType>(elem_type)) {
        new_kind = int_min_kind;
      } else {
        return op->emitError("Unsupported element type for min reduction");
      }
      break;
    default:
      break;
  }
  if (new_kind.has_value()) {
    op->setAttr("kind", ReductionKindAttr::get(op->getContext(), *new_kind));
  }
  return success();
}

LogicalResult downgrade_reduction_kind(Operation* op,
                                       ReductionKind int_max_kind,
                                       ReductionKind int_min_kind) {
  auto kind_attr = op->getAttrOfType<ReductionKindAttr>("kind");
  if (!kind_attr) {
    return op->emitError("Missing or invalid kind attribute");
  }
  const ReductionKind kind = kind_attr.getValue();
  std::optional<ReductionKind> new_kind;
  if (kind == ReductionKind::kMaxF || kind == int_max_kind) {
    new_kind = ReductionKind::kMax_DEPRECATED;
  } else if (kind == ReductionKind::kMinF || kind == int_min_kind) {
    new_kind = ReductionKind::kMin_DEPRECATED;
  } else if (kind == ReductionKind::kMaxSI || kind == ReductionKind::kMinSI ||
             kind == ReductionKind::kMaxUI || kind == ReductionKind::kMinUI) {
    return op->emitOpError("Cannot downgrade reduction kind ")
           << stringifyReductionKind(kind) << " below version 18";
  }
  if (new_kind.has_value()) {
    op->setAttr("kind", ReductionKindAttr::get(op->getContext(), *new_kind));
  }
  return success();
}

LogicalResult scan_upgrade(Operation* op, int version, bool&) {
  if (version < 17) {
    if (op->getNumOperands() == 0) {
      return op->emitError("Missing input operand for scan operation");
    }
    const auto input_type = dyn_cast<VectorType>(op->getOperand(0).getType());
    if (!input_type || input_type.getRank() <= 0) {
      return op->emitError("Invalid input type for scan operation");
    }
    const Attribute dimension_attr = IntegerAttr::get(
        IntegerType::get(op->getContext(), 64), input_type.getRank() - 1);
    op->setAttr("dimension", dimension_attr);
  }
  if (version < 18) {
    if (failed(upgrade_reduction_kind(op, ReductionKind::kMaxUI,
                                      ReductionKind::kMinUI))) {
      return failure();
    }
  }
  return success();
}

LogicalResult scan_downgrade(Operation* op, int version) {
  if (version < 18) {
    if (failed(downgrade_reduction_kind(op, ReductionKind::kMaxUI,
                                        ReductionKind::kMinUI))) {
      return failure();
    }
  }
  if (version < 17) {
    if (op->getNumOperands() == 0) {
      return op->emitError("Missing input operand for scan operation");
    }
    const auto input_type = dyn_cast<VectorType>(op->getOperand(0).getType());
    if (!input_type || input_type.getRank() <= 0) {
      return op->emitError("Invalid input type for scan operation");
    }
    const std::optional<int64_t> dimension =
        getSignedIntAttr(op->getAttr("dimension"));
    if (!dimension.has_value()) {
      return op->emitError("Missing or invalid dimensions attribute");
    }
    if (*dimension != input_type.getRank() - 1) {
      return op->emitError(
          "Can only downgrade below version 16 when dimension is the last "
          "dimension");
    }
    op->removeAttr("dimension");
  }
  return success();
}

LogicalResult reduce_upgrade(Operation* op, int version, bool&) {
  if (version < 18) {
    return upgrade_reduction_kind(op, ReductionKind::kMaxSI,
                                  ReductionKind::kMinSI);
  }
  return success();
}

LogicalResult reduce_downgrade(Operation* op, int version) {
  if (version < 18) {
    return downgrade_reduction_kind(op, ReductionKind::kMaxSI,
                                    ReductionKind::kMinSI);
  }
  return success();
}

// Changes the type of the result of `op` to `new_type` and routes its uses
// through a reshape back to the original type. RunSerde does not visit ops that
// rules insert, and it runs upgrade rules only when deserializing and downgrade
// rules only when serializing. So the reshape gets its final name here: plain
// after an upgrade, mangled after a downgrade.
void retype_result_through_reshape(Operation* op, VectorType new_type,
                                   bool mangle_reshape) {
  Value result = op->getResult(0);
  OpBuilder builder(op->getContext());
  builder.setInsertionPointAfter(op);
  OperationState state(op->getLoc(),
                       ((mangle_reshape ? kMangledDialect : StringRef()) +
                        ReshapeOp::getOperationName())
                           .str());
  state.addTypes(result.getType());
  state.addOperands(result);
  Operation* reshape = builder.create(state);
  result.setType(new_type);
  result.replaceAllUsesExcept(reshape->getResult(0), reshape);
}

// Before version 20, tpu.reduce_index dropped the reduced dimension instead of
// keeping it with size 1.
LogicalResult reduce_index_upgrade(Operation* op, int version, bool&) {
  if (version < 20) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return op->emitError(
          "Unexpected operand/result count in tpu.reduce_index");
    }
    const auto input_type = dyn_cast<VectorType>(op->getOperand(0).getType());
    const auto result_type = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!input_type || !result_type || input_type.isScalable() ||
        result_type.isScalable() || input_type.getRank() < 2) {
      return op->emitError("Invalid operand/result type in tpu.reduce_index");
    }
    const std::optional<int64_t> axis = getSignedIntAttr(op->getAttr("axis"));
    if (!axis.has_value()) {
      return op->emitError("Missing or invalid axis attribute");
    }
    if (*axis < 0 || *axis >= input_type.getRank()) {
      return op->emitError("Invalid axis in tpu.reduce_index");
    }
    SmallVector<int64_t> dropped_shape(input_type.getShape());
    dropped_shape.erase(dropped_shape.begin() + *axis);
    if (result_type.getShape() != ArrayRef<int64_t>(dropped_shape)) {
      return op->emitError("Invalid result shape in tpu.reduce_index");
    }
    SmallVector<int64_t> shape(input_type.getShape());
    shape[*axis] = 1;
    retype_result_through_reshape(
        op, VectorType::get(shape, result_type.getElementType()),
        /*mangle_reshape=*/false);
  }
  return success();
}

LogicalResult reduce_index_downgrade(Operation* op, int version) {
  if (version < 20) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return op->emitError(
          "Unexpected operand/result count in tpu.reduce_index");
    }
    const auto input_type = dyn_cast<VectorType>(op->getOperand(0).getType());
    const auto result_type = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!input_type || !result_type || input_type.isScalable() ||
        result_type.isScalable() || input_type.getRank() < 1) {
      return op->emitError("Invalid operand/result type in tpu.reduce_index");
    }
    const std::optional<int64_t> axis = getSignedIntAttr(op->getAttr("axis"));
    if (!axis.has_value()) {
      return op->emitError("Missing or invalid axis attribute");
    }
    if (*axis < 0 || *axis >= input_type.getRank()) {
      return op->emitError("Invalid axis in tpu.reduce_index");
    }
    SmallVector<int64_t> expected_result(input_type.getShape());
    expected_result[*axis] = 1;
    if (result_type.getShape() != ArrayRef<int64_t>(expected_result)) {
      return op->emitError("Invalid result shape in tpu.reduce_index");
    }
    if (input_type.getRank() == 1) {
      OpBuilder builder(op);
      OperationState state(
          op->getLoc(),
          (kMangledDialect + ReshapeOp::getOperationName()).str());
      state.addTypes(VectorType::get({1, input_type.getDimSize(0)},
                                     input_type.getElementType()));
      state.addOperands(op->getOperand(0));
      Operation* reshape = builder.create(state);
      op->setOperand(0, reshape->getResult(0));
      op->setAttr("axis", builder.getI32IntegerAttr(1));
      return success();
    }
    SmallVector<int64_t> shape(input_type.getShape());
    shape.erase(shape.begin() + *axis);
    retype_result_through_reshape(
        op, VectorType::get(shape, result_type.getElementType()),
        /*mangle_reshape=*/true);
  }
  return success();
}

// Upgrades an operation from a version without subcore_id (v13) to HEAD (v14+)
// by delinearizing a single core_id into separate core_id and subcore_id.
// This runs during deserialization (`serialize = false`), where operations have
// already been demangled from `stable_mosaic.<op>` back to regular MLIR ops.
// Therefore, we emit standard unmangled constant and arithmetic operations.
LogicalResult delinearize_subcore(Operation* op, size_t target_sem_seg,
                                  size_t core_id_seg, size_t subcore_id_seg) {
  FailureOr<ArrayRef<int32_t>> sizes = getValidatedSegmentSizes(
      op, subcore_id_seg, {target_sem_seg, core_id_seg});
  if (failed(sizes)) {
    return failure();
  }
  SmallVector<int32_t> new_sizes(*sizes);
  new_sizes.push_back(0);
  const CoreType target_core = getTargetCoreType(op, *sizes, target_sem_seg);
  if (target_core == CoreType::kScVectorSubcore && (*sizes)[core_id_seg] != 0) {
    const unsigned core_id_op_idx = getSegmentOffset(*sizes, core_id_seg);
    const unsigned subcore_id_op_idx = getSegmentOffset(*sizes, subcore_id_seg);
    Value core_id_val = op->getOperand(core_id_op_idx);
    if (!core_id_val.getType().isSignlessInteger(32)) {
      return op->emitError("Expected i32 core_id operand");
    }
    // For version < 14, supported SparseCore vector subcores use 16 subcores
    // per core.
    constexpr int32_t kNumSubcores = 16;
    OpBuilder builder(op);
    Location loc = op->getLoc();
    Value c16 = arith::ConstantOp::create(
        builder, loc, builder.getI32IntegerAttr(kNumSubcores));
    Value new_core_id = arith::DivUIOp::create(builder, loc, core_id_val, c16);
    Value new_subcore_id =
        arith::RemUIOp::create(builder, loc, core_id_val, c16);
    op->setOperand(core_id_op_idx, new_core_id);
    op->insertOperands(subcore_id_op_idx, {new_subcore_id});
    new_sizes[subcore_id_seg] = 1;
  }
  op->setAttr("operandSegmentSizes",
              DenseI32ArrayAttr::get(op->getContext(), new_sizes));
  return success();
}

// Downgrades an operation from HEAD (v14+) to a version without subcore_id
// (v13) by combining separate core_id and subcore_id into a single linearized
// core_id. This runs during serialization (`serialize = true`), where every
// operation in the module must be mangled with `stable_mosaic.`. Because MLIR's
// post-order walk has already advanced past the insertion point of new
// constants before downgrade rules execute, `RunSerde` will not visit or
// auto-mangle them. We therefore explicitly create pre-mangled
// `stable_mosaic.arith.*` operations.
LogicalResult linearize_subcore(Operation* op, size_t target_sem_seg,
                                size_t core_id_seg, size_t subcore_id_seg) {
  FailureOr<ArrayRef<int32_t>> sizes = getValidatedSegmentSizes(
      op, subcore_id_seg + 1, {target_sem_seg, core_id_seg, subcore_id_seg});
  if (failed(sizes)) {
    return failure();
  }
  const CoreType target_core = getTargetCoreType(op, *sizes, target_sem_seg);
  if ((*sizes)[subcore_id_seg] != 0) {
    if (target_core != CoreType::kScVectorSubcore) {
      return op->emitError(
          "subcore_id linearization requested for non-SC vector subcore type");
    }
    if ((*sizes)[core_id_seg] == 0) {
      return op->emitError(
          "Expected both core_id and subcore_id to be present for "
          "linearization");
    }
    const unsigned core_id_op_idx = getSegmentOffset(*sizes, core_id_seg);
    const unsigned subcore_id_op_idx = getSegmentOffset(*sizes, subcore_id_seg);
    Value core_id_val = op->getOperand(core_id_op_idx);
    Value subcore_id_val = op->getOperand(subcore_id_op_idx);
    if (!core_id_val.getType().isSignlessInteger(32) ||
        !subcore_id_val.getType().isSignlessInteger(32)) {
      return op->emitError("Expected i32 core_id and subcore_id operands");
    }
    OpBuilder builder(op);
    Location loc = op->getLoc();
    constexpr int32_t kNumSubcores = 16;
    OperationState const_state(
        loc, (kMangledDialect + arith::ConstantOp::getOperationName()).str());
    const_state.addTypes(builder.getI32Type());
    const_state.addAttribute("value", builder.getI32IntegerAttr(kNumSubcores));
    Value c16 = builder.create(const_state)->getResult(0);
    OperationState mul_state(
        loc, (kMangledDialect + arith::MulIOp::getOperationName()).str());
    mul_state.addTypes(builder.getI32Type());
    mul_state.addOperands({core_id_val, c16});
    Value mul_val = builder.create(mul_state)->getResult(0);
    OperationState add_state(
        loc, (kMangledDialect + arith::AddIOp::getOperationName()).str());
    add_state.addTypes(builder.getI32Type());
    add_state.addOperands({mul_val, subcore_id_val});
    op->setOperand(core_id_op_idx, builder.create(add_state)->getResult(0));
    op->eraseOperand(subcore_id_op_idx);
  } else if (target_core == CoreType::kScVectorSubcore &&
             (*sizes)[core_id_seg] != 0) {
    return op->emitError(
        "Cannot downgrade SC vector subcore operation with core_id and no "
        "subcore_id below version 14");
  }
  op->setAttr("operandSegmentSizes",
              DenseI32ArrayAttr::get(op->getContext(), sizes->drop_back()));
  return success();
}

LogicalResult enqueue_dma_upgrade(Operation* op, int version, bool&) {
  // Added AttrSizedOperandSegments and core_id in version 2.
  if (version < 2) {
    if (op->getNumOperands() == 3) {  // Local DMA.
      op->setAttr(
          OpTrait::AttrSizedOperandSegments<
              EnqueueDMAOp>::getOperandSegmentSizeAttr(),
          mlir::DenseI32ArrayAttr::get(op->getContext(), {1, 0, 1, 1, 0, 0}));
    } else if (op->getNumOperands() == 5) {  // Remote DMA.
      op->setAttr(
          OpTrait::AttrSizedOperandSegments<
              EnqueueDMAOp>::getOperandSegmentSizeAttr(),
          mlir::DenseI32ArrayAttr::get(op->getContext(), {1, 1, 1, 1, 1, 0}));
    } else {
      return op->emitError("Unexpected operand count in tpu.enqueue_dma: ")
             << op->getNumOperands();
    }
  }
  if (version < 4) {
    op->setAttr("priority",
                mlir::IntegerAttr::get(
                    mlir::IntegerType::get(op->getContext(), 32), 0));
  }
  if (version < 14) {
    return delinearize_subcore(op, /*target_sem_seg=*/3, /*core_id_seg=*/5,
                               /*subcore_id_seg=*/6);
  }
  return success();
}

LogicalResult enqueue_dma_downgrade(Operation* op, int version) {
  if (version < 14) {
    if (failed(linearize_subcore(op, /*target_sem_seg=*/3, /*core_id_seg=*/5,
                                 /*subcore_id_seg=*/6))) {
      return failure();
    }
  }
  if (version < 12) {
    FailureOr<ArrayRef<int32_t>> sizes =
        getValidatedSegmentSizes(op, /*expected_size=*/6, {3});
    if (failed(sizes)) {
      return failure();
    }
    if ((*sizes)[3] == 0) {
      return op->emitError(
          "Cannot downgrade enqueue_dma without target_semaphore to version "
          "< 12");
    }
  }
  if (version < 8) {
    auto ordering_attr = op->getAttrOfType<BoolAttr>("strict_ordering");
    if (ordering_attr != nullptr) {
      if (ordering_attr.getValue()) {
        return op->emitError(
            "Can only downgrade below version 8 when strict ordering is not "
            "set to True");
      }
      op->removeAttr("strict_ordering");
    }
  }
  if (version < 4) {
    if (Attribute priority_attr = op->getAttr("priority")) {
      const std::optional<int64_t> priority = getSignedIntAttr(priority_attr);
      if (!priority.has_value() || *priority != 0) {
        return op->emitError(
            "Can only downgrade below version 4 when priority is 0");
      }
      op->removeAttr("priority");
    }
  }
  if (version < 2) {
    return op->emitError("Downgrade to version ") << version << " unsupported";
  }
  return success();
}

LogicalResult iota_upgrade(Operation* op, int version, bool&) {
  if (version < 6) {
    auto dimension_attr = op->getAttrOfType<IntegerAttr>("dimension");
    const std::optional<int64_t> dimension = getSignedIntAttr(dimension_attr);
    if (!dimension_attr || !dimension_attr.getType().isSignlessInteger(32) ||
        !dimension.has_value()) {
      return op->emitError("Missing or invalid dimension attribute");
    }
    op->removeAttr("dimension");
    op->setAttr("dimensions",
                DenseI32ArrayAttr::get(op->getContext(),
                                       {static_cast<int32_t>(*dimension)}));
  }
  return success();
}

LogicalResult iota_downgrade(Operation* op, int version) {
  if (version < 6) {
    auto dimensions_attr = op->getAttrOfType<DenseI32ArrayAttr>("dimensions");
    if (!dimensions_attr) {
      return op->emitError("Missing or invalid dimensions attribute");
    }
    const ArrayRef<int32_t> dimensions = dimensions_attr.asArrayRef();
    if (dimensions.size() != 1) {
      return op->emitError(
          "Can only downgrade below version 5 when a single dimension is "
          "specified.");
    }
    const int32_t dimension = dimensions.front();
    op->removeAttr("dimensions");
    op->setAttr("dimension",
                mlir::IntegerAttr::get(
                    mlir::IntegerType::get(op->getContext(), 32), dimension));
  }
  return success();
}

LogicalResult wait_dma2_upgrade(Operation* op, int version, bool&) {
  if (version < 7) {
    if (op->getNumOperands() != 3) {
      return op->emitError("Unexpected operand count in tpu.wait_dma2: ")
             << op->getNumOperands();
    }
    op->setAttr(
        OpTrait::AttrSizedOperandSegments<
            EnqueueDMAOp>::getOperandSegmentSizeAttr(),
        mlir::DenseI32ArrayAttr::get(op->getContext(), {1, 1, 1, 0, 0}));
  }
  return success();
}

LogicalResult wait_dma2_downgrade(Operation* op, int version) {
  if (version < 7) {
    auto operands = op->getAttrOfType<mlir::DenseI32ArrayAttr>(
        OpTrait::AttrSizedOperandSegments<
            EnqueueDMAOp>::getOperandSegmentSizeAttr());
    if (!operands || operands.size() != 5) {
      return op->emitError("Missing or invalid AttrSizedOperandSegments");
    }
    if (operands[3] || operands[4]) {
      return op->emitError("Downgrade to version ")
             << version << " impossible: device_id and/or core_id is set";
    }
    op->removeAttr(OpTrait::AttrSizedOperandSegments<
                   EnqueueDMAOp>::getOperandSegmentSizeAttr());
  }
  if (version < 3) {
    return op->emitError("Downgrade to version ") << version << " unsupported";
  }
  return success();
}

LogicalResult wait_dma_upgrade(Operation* op, int version, bool&) {
  if (version < 14) {
    return delinearize_subcore(op, /*target_sem_seg=*/3, /*core_id_seg=*/5,
                               /*subcore_id_seg=*/6);
  }
  return success();
}

LogicalResult wait_dma_downgrade(Operation* op, int version) {
  if (version < 14) {
    return linearize_subcore(op, /*target_sem_seg=*/3, /*core_id_seg=*/5,
                             /*subcore_id_seg=*/6);
  }
  return success();
}

LogicalResult semaphore_signal_upgrade(Operation* op, int version, bool&) {
  // Added AttrSizedOperandSegments and core_id in version 2.
  if (version < 2) {
    if (op->getNumOperands() == 2) {  // Local signal.
      op->setAttr(OpTrait::AttrSizedOperandSegments<
                      EnqueueDMAOp>::getOperandSegmentSizeAttr(),
                  mlir::DenseI32ArrayAttr::get(op->getContext(), {1, 1, 0, 0}));
    } else if (op->getNumOperands() == 3) {  // Remote signal.
      op->setAttr(OpTrait::AttrSizedOperandSegments<
                      EnqueueDMAOp>::getOperandSegmentSizeAttr(),
                  mlir::DenseI32ArrayAttr::get(op->getContext(), {1, 1, 1, 0}));
    } else {
      return op->emitError("Unexpected operand count in tpu.semaphore_signal");
    }
  }
  if (version < 14) {
    return delinearize_subcore(op, /*target_sem_seg=*/0, /*core_id_seg=*/3,
                               /*subcore_id_seg=*/4);
  }
  return success();
}

LogicalResult semaphore_signal_downgrade(Operation* op, int version) {
  if (version < 14) {
    if (failed(linearize_subcore(op, /*target_sem_seg=*/0, /*core_id_seg=*/3,
                                 /*subcore_id_seg=*/4))) {
      return failure();
    }
  }
  if (version < 2) {
    auto operands = op->getAttrOfType<mlir::DenseI32ArrayAttr>(
        OpTrait::AttrSizedOperandSegments<
            EnqueueDMAOp>::getOperandSegmentSizeAttr());
    if (!operands || operands.size() != 4) {
      return op->emitError("Missing or invalid AttrSizedOperandSegments");
    }
    if (operands[3]) {
      return op->emitError("Downgrade to version ")
             << version << " impossible: core_id is set";
    }
    op->removeAttr(OpTrait::AttrSizedOperandSegments<
                   EnqueueDMAOp>::getOperandSegmentSizeAttr());
  }
  return success();
}

LogicalResult vector_multi_dim_reduce_upgrade(Operation* op, int version,
                                              bool&) {
  // Changed reductions_dims from ArrayAttr of IntegerAttrs to DenseI64ArrayAttr
  // in version 3.
  if (version < 3) {
    Attribute reduction_dims_attr = op->getAttr("reduction_dims");
    if (!reduction_dims_attr) {
      return op->emitError("Missing reduction_dims attribute");
    }
    ArrayAttr reduction_dims_array = dyn_cast<ArrayAttr>(reduction_dims_attr);
    if (!reduction_dims_array) {
      return op->emitOpError("reduction_dims attribute is not an ArrayAttr");
    }
    std::vector<int64_t> reduction_dims;
    reduction_dims.reserve(reduction_dims_array.size());
    for (Attribute reduction_dim : reduction_dims_array) {
      const std::optional<int64_t> dim = getSignedIntAttr(reduction_dim);
      if (!dim.has_value()) {
        return op->emitOpError(
            "reduction_dims attribute contains a non-IntegerAttr");
      }
      reduction_dims.push_back(*dim);
    }
    op->setAttr("reduction_dims",
                DenseI64ArrayAttr::get(op->getContext(), reduction_dims));
  }
  return success();
}

LogicalResult vector_multi_dim_reduce_downgrade(Operation* op, int version) {
  if (version < 3) {
    return op->emitOpError("Downgrade to version ")
           << version << " unsupported";
  }
  return success();
}

LogicalResult arith_constant_upgrade(Operation* op, int version, bool&) {
  return success();
}

LogicalResult arith_constant_downgrade(Operation* op, int version) {
  // The encoding of the boolean dense elements attr changed in version 10.
  // Before, it used to be a bitpacked value, but a splat true was represented
  // by a single byte with all bits set to 1. After the change, a splat true is
  // represented by a byte with the LSB set to 1. The change happens to be
  // backwards-compatible, but not forwards-compatible, which is why we need to
  // have the downgrade rule only.
  if (version < 10) {
    auto value_attr = op->getAttrOfType<DenseElementsAttr>("value");
    // Only i1 dense elements attrs are affected.
    if (!value_attr || value_attr.getElementType() !=
                           mlir::IntegerType::get(op->getContext(), 1)) {
      return success();
    }
    if (!value_attr.isSplat()) {
      return op->emitOpError("Downgrade to version ")
             << version << " not implemented: value is not a splat";
    }
    char new_splat = value_attr.getSplatValue<bool>() ? 0xff : 0x00;
    op->setAttr("value", mlir::SplatElementsAttr::getFromRawBuffer(
                             value_attr.getType(), {new_splat}));
  }
  return success();
}
LogicalResult reinterpret_cast_upgrade(Operation* op, int version, bool&) {
  if (version < 15) {
    const unsigned num_operands = op->getNumOperands();
    if (num_operands != 1 && num_operands != 2) {
      return op->emitError("Unexpected operand count in tpu.reinterpret_cast");
    }
    op->setAttr("operandSegmentSizes",
                mlir::DenseI32ArrayAttr::get(
                    op->getContext(), {1, num_operands == 2 ? 1 : 0, 0}));
  }
  if (version < 16) {
    FailureOr<ArrayRef<int32_t>> sizes =
        getValidatedSegmentSizes(op, /*expected_size=*/3, {1});
    if (failed(sizes)) {
      return failure();
    }
    if ((*sizes)[0] != 1) {
      return op->emitError(
          "Unexpected source segment size in tpu.reinterpret_cast");
    }
    llvm::SmallVector<int32_t> new_sizes(*sizes);
    new_sizes.push_back(0);
    op->setAttr("operandSegmentSizes",
                mlir::DenseI32ArrayAttr::get(op->getContext(), new_sizes));
  }
  return success();
}

LogicalResult reinterpret_cast_downgrade(Operation* op, int version) {
  if (version < 16) {
    FailureOr<ArrayRef<int32_t>> sizes =
        getValidatedSegmentSizes(op, /*expected_size=*/4, {1});
    if (failed(sizes)) {
      return failure();
    }
    if ((*sizes)[0] != 1) {
      return op->emitError(
          "Unexpected source segment size in tpu.reinterpret_cast");
    }
    if ((*sizes)[3] != 0) {
      return op->emitOpError(
          "Can only downgrade below version 16 when dynamic_strides is "
          "empty");
    }
    op->setAttr(
        "operandSegmentSizes",
        mlir::DenseI32ArrayAttr::get(op->getContext(), sizes->drop_back()));
  }

  if (version < 15) {
    FailureOr<ArrayRef<int32_t>> sizes =
        getValidatedSegmentSizes(op, /*expected_size=*/3, {1});
    if (failed(sizes)) {
      return failure();
    }
    if ((*sizes)[0] != 1) {
      return op->emitError(
          "Unexpected source segment size in tpu.reinterpret_cast");
    }
    if ((*sizes)[2] != 0) {
      return op->emitOpError(
          "Can only downgrade below version 15 when dynamic_sizes is empty");
    }
    op->removeAttr("operandSegmentSizes");
  }

  if (version < 12) {
    if (op->getNumOperands() != 1) {
      return op->emitOpError(
          "Can only downgrade below version 12 when dynamic_offset is not set");
    }
  }
  return success();
}

LogicalResult matmul_upgrade(Operation* op, int version, bool&) {
  if (version < 13) {
    op->setAttr("transpose_lhs_hint",
                mlir::BoolAttr::get(op->getContext(), false));
  }
  return success();
}

LogicalResult matmul_downgrade(Operation* op, int version) {
  if (version < 13) {
    auto transpose_lhs_hint_attr =
        op->getAttrOfType<BoolAttr>("transpose_lhs_hint");
    if (!transpose_lhs_hint_attr) {
      return op->emitOpError("Missing transpose_lhs_hint attribute");
    }
    if (transpose_lhs_hint_attr.getValue()) {
      return op->emitOpError(
          "Can only downgrade below version 12 when transpose_lhs_hint is "
          "False ");
    }
    op->removeAttr("transpose_lhs_hint");
  }
  return success();
}

LogicalResult repeat_upgrade(Operation* op, int version, bool& erased) {
  (void)version;
  if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
    return op->emitError("Unexpected operand/result count in tpu.repeat");
  }
  const auto source_type = dyn_cast<VectorType>(op->getOperand(0).getType());
  const auto result_type = dyn_cast<VectorType>(op->getResult(0).getType());
  if (!source_type || !result_type || source_type.isScalable() ||
      result_type.isScalable() || source_type.getRank() < 1 ||
      result_type.getRank() != source_type.getRank() ||
      result_type.getElementType() != source_type.getElementType()) {
    return op->emitError("Invalid operand/result type in tpu.repeat");
  }
  const std::optional<int64_t> dimension =
      getSignedIntAttr(op->getAttr("dimension"));
  const std::optional<int64_t> times = getSignedIntAttr(op->getAttr("times"));
  if (!dimension.has_value() || !times.has_value()) {
    return op->emitError("Missing dimension or times attribute in tpu.repeat");
  }
  if (*dimension < 0 || *dimension >= source_type.getRank() ||
      !llvm::isUInt<32>(*dimension)) {
    return op->emitError("Invalid dimension attribute in tpu.repeat");
  }
  if (*times < 1 || !llvm::isInt<32>(*times)) {
    return op->emitError("Invalid times attribute in tpu.repeat");
  }
  for (int64_t d = 0; d < source_type.getRank(); ++d) {
    const int64_t src_dim = source_type.getDimSize(d);
    const int64_t dst_dim = result_type.getDimSize(d);
    if (d == *dimension) {
      if (src_dim <= 0 || dst_dim <= 0 || dst_dim % src_dim != 0 ||
          dst_dim / src_dim != *times) {
        return op->emitError("Result shape mismatch in tpu.repeat");
      }
    } else if (src_dim != dst_dim) {
      return op->emitError("Result shape mismatch in tpu.repeat");
    }
  }
  Value source = op->getOperand(0);
  if (*times == 1) {
    op->replaceAllUsesWith(ValueRange{source});
  } else {
    OpBuilder builder(op);
    SmallVector<Value> operands(static_cast<size_t>(*times), source);
    Value concat =
        ConcatenateOp::create(builder, op->getLoc(), result_type, operands,
                              static_cast<uint32_t>(*dimension));
    op->replaceAllUsesWith(ValueRange{concat});
  }
  op->erase();
  erased = true;
  return success();
}

const llvm::StringMap<UpgradeRuleType>& upgrade_rules() {
  static auto rules = new llvm::StringMap<UpgradeRuleType>{
      {EnqueueDMAOp::getOperationName(), enqueue_dma_upgrade},
      {WaitDMA2Op::getOperationName(), wait_dma2_upgrade},
      {WaitDMAOp::getOperationName(), wait_dma_upgrade},
      {DynamicGatherOp::getOperationName(), dynamic_gather_upgrade},
      {ScanOp::getOperationName(), scan_upgrade},
      {ReduceOp::getOperationName(), reduce_upgrade},
      {ReduceIndexOp::getOperationName(), reduce_index_upgrade},
      {IotaOp::getOperationName(), iota_upgrade},
      {SemaphoreSignalOp::getOperationName(), semaphore_signal_upgrade},
      {vector::MultiDimReductionOp::getOperationName(),
       vector_multi_dim_reduce_upgrade},
      {StoreOp::getOperationName(), store_upgrade},
      {arith::ConstantOp::getOperationName(), arith_constant_upgrade},
      {ReinterpretCastOp::getOperationName(), reinterpret_cast_upgrade},
      {MatmulOp::getOperationName(), matmul_upgrade},
      {"tpu.repeat", repeat_upgrade},
  };
  return *rules;
}

const llvm::StringMap<DowngradeRuleType>& downgrade_rules() {
  static auto rules = new llvm::StringMap<DowngradeRuleType>{
      {EnqueueDMAOp::getOperationName(), enqueue_dma_downgrade},
      {WaitDMA2Op::getOperationName(), wait_dma2_downgrade},
      {WaitDMAOp::getOperationName(), wait_dma_downgrade},
      {DynamicGatherOp::getOperationName(), dynamic_gather_downgrade},
      {ScanOp::getOperationName(), scan_downgrade},
      {ReduceOp::getOperationName(), reduce_downgrade},
      {ReduceIndexOp::getOperationName(), reduce_index_downgrade},
      {IotaOp::getOperationName(), iota_downgrade},
      {SemaphoreSignalOp::getOperationName(), semaphore_signal_downgrade},
      {StoreOp::getOperationName(), store_downgrade},
      {vector::MultiDimReductionOp::getOperationName(),
       vector_multi_dim_reduce_downgrade},
      {arith::ConstantOp::getOperationName(), arith_constant_downgrade},
      {ReinterpretCastOp::getOperationName(), reinterpret_cast_downgrade},
      {MatmulOp::getOperationName(), matmul_downgrade},
  };
  return *rules;
}

}  // namespace

void MosaicSerdePass::runOnOperation() {
  ModuleOp module = getOperation();
  if (!serialize.hasValue()) {
    module.emitError("serialize option must be specified");
    return signalPassFailure();
  }
  int serialize_version = -1;
  if (serialize) {
    serialize_version = target_version.hasValue() ? target_version : kVersion;
  }
  if (failed(jaxlib::mosaic::RunSerde(
          module, upgrade_rules(), downgrade_rules(), serialize,
          {.dialect_prefix = kMangledDialect,
           .highest_version = kVersion,
           .version_attr_name = kVersionAttrName,
           .serialize_version = serialize_version},
          /*keep_version_attr=*/keep_version_attr))) {
    signalPassFailure();
  }
}

}  // namespace mlir::tpu
