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

#include <cstdint>
#include <optional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"  // IWYU pragma: keep

namespace xla::cpu {

#define GEN_PASS_DEF_HOISTGEMVREDUCTIONPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

namespace ma = ::mlir::arith;
namespace ms = ::mlir::scf;
namespace mv = ::mlir::vector;

using ::mlir::FloatType;
using ::mlir::Location;
using ::mlir::OpBuilder;
using ::mlir::Value;
using ::mlir::VectorType;

// Cap the number of live vector<K x T> accumulators carried across the loop so
// they stay in SIMD registers without spilling (e.g. <= 8 of 16 YMM regs).
constexpr int64_t kMaxHoistedRows = 8;

struct KMinorGemvOperands {
  Value mat;
  Value vec;
};

// Matches a K-minor GEMV `vector.contract` of shape
// `vector<M x K x T> x vector<K x T> -> vector<M x T>` (with either operand as
// the 2D matrix), where `M` is the parallel dim and `K` is the minor-most
// reduction dim.
std::optional<KMinorGemvOperands> MatchKMinorGemv(mv::ContractionOp contract) {
  if (!contract.isAddKind() || contract.isMasked()) {
    return std::nullopt;
  }
  VectorType lhs_type = contract.getLhsType();
  VectorType rhs_type = contract.getRhsType();
  auto acc_type = mlir::dyn_cast<VectorType>(contract.getAccType());
  if (!acc_type || acc_type.getRank() != 1 || lhs_type.isScalable() ||
      rhs_type.isScalable() || acc_type.isScalable()) {
    return std::nullopt;
  }
  mlir::Type elem_type = acc_type.getElementType();
  if (lhs_type.getElementType() != elem_type ||
      !mlir::isa<FloatType, mlir::IntegerType>(elem_type)) {
    return std::nullopt;
  }

  // `vector.contract` verifies projected-permutation indexing maps and at least
  // one contracting dim pair, so a rank-2 x rank-1 -> rank-1 contraction has
  // exactly one contracting dim (the 1D operand's sole dim) and one free dim.
  bool lhs_is_mat = lhs_type.getRank() == 2 && rhs_type.getRank() == 1;
  bool rhs_is_mat = rhs_type.getRank() == 2 && lhs_type.getRank() == 1;
  if (!lhs_is_mat && !rhs_is_mat) {
    return std::nullopt;
  }
  auto [lhs_k_dim, rhs_k_dim] = contract.getContractingDimMap().front();
  if ((lhs_is_mat ? lhs_k_dim : rhs_k_dim) != 1) {
    return std::nullopt;
  }
  Value mat = lhs_is_mat ? contract.getLhs() : contract.getRhs();
  Value vec = lhs_is_mat ? contract.getRhs() : contract.getLhs();
  auto mat_type = mlir::cast<VectorType>(mat.getType());
  int64_t m = mat_type.getDimSize(0);
  int64_t k = mat_type.getDimSize(1);
  if (m <= 1 || m > kMaxHoistedRows || k <= 1) {
    return std::nullopt;
  }
  return KMinorGemvOperands{mat, vec};
}

struct HoistableGemv {
  unsigned iter_arg_idx;
  mv::ContractionOp contract;
  Value mat;
  Value vec;
};

// Rewrites loop-carried K-minor GEMV `vector.contract` ops inside `scf.for`
// into 2D `vector<M x K x T>` partial-sum accumulations inside the loop
// followed by a post-loop `vector.multi_reduction`:
//
//   %zero = arith.constant dense<0> : vector<M x K x T>
//   %partial = scf.for ... iter_args(%acc = %zero : vector<M x K x T>) {
//     %bcast = vector.broadcast %vec : vector<K x T> to vector<M x K x T>
//     %next = vector.fma %mat, %bcast, %acc : vector<M x K x T>
//     scf.yield %next : vector<M x K x T>
//   }
//   %res = vector.multi_reduction <add>, %partial, %init [1]
//          : vector<M x K x T> to vector<M x T>
mlir::LogicalResult HoistKMinorGemvContract(ms::ForOp for_op,
                                            mlir::PatternRewriter& rewriter) {
  mlir::Block* body = for_op.getBody();
  auto yield_op = mlir::cast<ms::YieldOp>(body->getTerminator());

  llvm::SmallVector<HoistableGemv> matches;
  for (unsigned i = 0; i < for_op.getNumRegionIterArgs(); ++i) {
    Value acc = for_op.getRegionIterArg(i);
    if (!acc.hasOneUse()) {
      continue;
    }
    auto contract = mlir::dyn_cast<mv::ContractionOp>(*acc.getUsers().begin());
    if (!contract || contract->getBlock() != body || contract.getAcc() != acc ||
        !contract.getResult().hasOneUse() ||
        yield_op.getOperand(i) != contract.getResult()) {
      continue;
    }
    if (std::optional<KMinorGemvOperands> gemv = MatchKMinorGemv(contract)) {
      matches.push_back(HoistableGemv{i, contract, gemv->mat, gemv->vec});
    }
  }
  if (matches.empty()) {
    return mlir::failure();
  }

  Location loc = for_op.getLoc();
  llvm::SmallVector<Value> new_inits = llvm::to_vector(for_op.getInitArgs());
  for (const HoistableGemv& match : matches) {
    auto mat_type = mlir::cast<VectorType>(match.mat.getType());
    new_inits[match.iter_arg_idx] = ma::ConstantOp::create(
        rewriter, loc, mat_type, rewriter.getZeroAttr(mat_type));
  }

  auto new_for = ms::ForOp::create(
      rewriter, loc, for_op.getLowerBound(), for_op.getUpperBound(),
      for_op.getStep(), new_inits,
      [](OpBuilder&, Location, Value, mlir::ValueRange) {},
      for_op.getUnsignedCmp());
  new_for->setDiscardableAttrs(for_op->getDiscardableAttrDictionary());

  {
    OpBuilder::InsertionGuard guard(rewriter);
    for (HoistableGemv& match : matches) {
      Value new_acc = new_for.getRegionIterArg(match.iter_arg_idx);
      Location cloc = match.contract.getLoc();
      auto mat_type = mlir::cast<VectorType>(match.mat.getType());
      rewriter.setInsertionPoint(match.contract);
      Value bcast_vec =
          mv::BroadcastOp::create(rewriter, cloc, mat_type, match.vec);
      Value updated;
      if (mlir::isa<FloatType>(mat_type.getElementType())) {
        updated =
            mv::FMAOp::create(rewriter, cloc, match.mat, bcast_vec, new_acc);
      } else {
        Value prod = ma::MulIOp::create(rewriter, cloc, match.mat, bcast_vec);
        updated = ma::AddIOp::create(rewriter, cloc, prod, new_acc);
      }
      rewriter.modifyOpInPlace(yield_op, [&]() {
        yield_op.setOperand(match.iter_arg_idx, updated);
      });
      rewriter.eraseOp(match.contract);
    }
  }

  rewriter.mergeBlocks(body, new_for.getBody(),
                       new_for.getBody()->getArguments());

  rewriter.setInsertionPointAfter(new_for);
  llvm::SmallVector<Value> replacements =
      llvm::to_vector_of<Value>(new_for.getResults());
  for (const HoistableGemv& match : matches) {
    unsigned idx = match.iter_arg_idx;
    replacements[idx] = mv::MultiDimReductionOp::create(
        rewriter, loc, mv::CombiningKind::ADD, new_for.getResult(idx),
        for_op.getInitArgs()[idx], llvm::ArrayRef<int64_t>{1});
  }
  rewriter.replaceOp(for_op, replacements);
  return mlir::success();
}

class HoistGemvReductionPass
    : public impl::HoistGemvReductionPassBase<HoistGemvReductionPass> {
 public:
  using HoistGemvReductionPassBase::HoistGemvReductionPassBase;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    mv::populateDropUnitDimWithShapeCastPatterns(patterns);
    patterns.add(HoistKMinorGemvContract);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace xla::cpu
