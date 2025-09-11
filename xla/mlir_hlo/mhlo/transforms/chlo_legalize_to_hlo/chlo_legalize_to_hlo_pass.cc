/* Copyright 2020 The OpenXLA Authors.

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

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mhlo/utils/type_conversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_CHLOLEGALIZETOHLOPASS
#define GEN_PASS_DEF_CHLOLEGALIZETOHIGHLEVELMHLOPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

ChloLegalizeToHighLevelMhloPassOptions FromPassOptions(bool enableAcosh) {
  ChloLegalizeToHighLevelMhloPassOptions options;
  options.enable_acosh_ = enableAcosh;
  return options;
}

bool isLegalAcosh(chlo::AcoshOp op) {
  return !llvm::isa<FloatType>(getElementTypeOrSelf(op.getType()));
}

struct ChloLegalizeToHighLevelMhloPass
    : public impl::ChloLegalizeToHighLevelMhloPassBase<
          ChloLegalizeToHighLevelMhloPass> {
  ChloLegalizeToHighLevelMhloPass() = default;
  explicit ChloLegalizeToHighLevelMhloPass(
      ChloLegalizeToHighLevelMhloPassOptions options)
      : impl::ChloLegalizeToHighLevelMhloPassBase<
            ChloLegalizeToHighLevelMhloPass>(options) {}

  void runOnOperation() override {
    MLIRContext &context = getContext();
    ConversionTarget conversionTarget(context);
    RewritePatternSet conversionPatterns(&context);

    chlo::populateChloToHighLevelMhloOpPatterns(&context, &conversionPatterns,
                                                FromPassOptions(enable_acosh_));

    // Consider the mhlo dialect legal for tests. Also add helper dialects
    // that are needed by the patterns.
    conversionTarget.addLegalDialect<chlo::ChloDialect, mhlo::MhloDialect>();
    if (enable_acosh_) {
      conversionTarget.addDynamicallyLegalOp<chlo::AcoshOp>(isLegalAcosh);
    }
    conversionTarget.addIllegalOp<chlo::ErfOp>();
    conversionTarget.addIllegalOp<chlo::RaggedDotOp>();
    conversionTarget.addIllegalOp<chlo::TopKOp>();
    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(conversionPatterns)))) {
      return signalPassFailure();
    }
  }
};

struct ChloLegalizeToHloPass
    : public impl::ChloLegalizeToHloPassBase<ChloLegalizeToHloPass> {
  using ChloLegalizeToHloPassBase::ChloLegalizeToHloPassBase;

  void runOnOperation() override {
    MLIRContext &context = getContext();
    ConversionTarget conversionTarget(context);
    RewritePatternSet conversionPatterns(&context);

    stablehlo::StablehloToHloTypeConverter typeConverter;
    chlo::populateChloToHloPatterns(&context, &typeConverter,
                                    &conversionPatterns);

    // Consider the mhlo dialect legal for tests. Also add helper dialects
    // that are needed by the patterns.
    conversionTarget
        .addIllegalDialect<chlo::ChloDialect, stablehlo::StablehloDialect>();
    conversionTarget.addLegalDialect<
        MhloDialect, mlir::arith::ArithDialect, mlir::func::FuncDialect,
        mlir::tensor::TensorDialect, mlir::shape::ShapeDialect>();

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(conversionPatterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

ChloLegalizeToHighLevelMhloPassOptions getDefaultChloToHighLevelMhloOptions() {
  ChloLegalizeToHighLevelMhloPassOptions opts;
  opts.enable_acosh_ = false;
  return opts;
}

}  // namespace mhlo

namespace chlo {
namespace {

LogicalResult RaggedDotChloToMhlo(chlo::RaggedDotOp raggedDotOp,
                                  PatternRewriter& rewriter) {
  auto moduleOp = raggedDotOp->getParentOfType<ModuleOp>();

  OpBuilder builder(moduleOp.getBodyRegion());
  builder.setInsertionPointToStart(&moduleOp.getBodyRegion().front());

  auto chloRaggedDotDimNums = raggedDotOp.getRaggedDotDimensionNumbers();
  auto dotDimNums = mhlo::DotDimensionNumbersAttr::get(
      builder.getContext(), chloRaggedDotDimNums.getLhsBatchingDimensions(),
      chloRaggedDotDimNums.getRhsBatchingDimensions(),
      chloRaggedDotDimNums.getLhsContractingDimensions(),
      chloRaggedDotDimNums.getRhsContractingDimensions());
  auto raggedDotDimNums = mhlo::RaggedDotDimensionNumbersAttr::get(
      builder.getContext(), dotDimNums,
      chloRaggedDotDimNums.getLhsRaggedDimensions(),
      chloRaggedDotDimNums.getRhsGroupDimensions());

  auto mhloPrecision =
      [](chlo::Precision precision) -> std::optional<mhlo::Precision> {
    switch (precision) {
      case chlo::Precision::DEFAULT:
        return mhlo::Precision::DEFAULT;
      case chlo::Precision::HIGH:
        return mhlo::Precision::HIGH;
      case chlo::Precision::HIGHEST:
        return mhlo::Precision::HIGHEST;
    }
  };
  ArrayAttr precisionConfig = rewriter.getArrayAttr({});
  if (raggedDotOp.getPrecisionConfig().has_value()) {
    SmallVector<Attribute> vector;
    for (auto configValue : raggedDotOp.getPrecisionConfig()
                                .value()
                                .getAsRange<chlo::PrecisionAttr>()) {
      vector.push_back(mhlo::PrecisionAttr::get(
          raggedDotOp.getContext(),
          mhloPrecision(configValue.getValue()).value()));
    }
    precisionConfig = rewriter.getArrayAttr(vector);
  }

  mhlo::RaggedDotOp mhloOp = rewriter.create<mhlo::RaggedDotOp>(
      raggedDotOp.getLoc(), raggedDotOp.getResult().getType(),
      raggedDotOp.getLhs(), raggedDotOp.getRhs(), raggedDotOp.getGroupSizes(),
      raggedDotDimNums, precisionConfig);
  std::optional<NamedAttribute> frontendAttributes =
      raggedDotOp->getAttrDictionary().getNamed("mhlo.frontend_attributes");
  if (frontendAttributes.has_value()) {
    std::vector<NamedAttribute> attributes =
        mhloOp->getDiscardableAttrDictionary().getValue().vec();
    attributes.push_back(frontendAttributes.value());
    mhloOp->setDiscardableAttrs(rewriter.getDictionaryAttr(attributes));
  }

  rewriter.replaceOp(raggedDotOp, mhloOp.getOperation());
  return success();
}

LogicalResult AcoshChloToMhlo(chlo::AcoshOp op, PatternRewriter& rewriter) {
  if (mhlo::isLegalAcosh(op)) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<mhlo::AcoshOp>(op, op->getOperands());
  return success();
}

LogicalResult ErfChloToMhlo(chlo::ErfOp op, PatternRewriter& rewriter) {
  rewriter.replaceOpWithNewOp<mhlo::ErfOp>(op, op->getOperands());
  return success();
}

LogicalResult TopKChloToMhlo(chlo::TopKOp op, PatternRewriter& rewriter) {
  if (!mlir::isa<RankedTensorType>(op.getResult(0).getType())) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<mhlo::TopKOp>(op, op.getOperand(), op.getK());
  return success();
}

}  // namespace

void populateChloToHighLevelMhloOpPatterns(
    MLIRContext*, RewritePatternSet* patterns,
    const mhlo::ChloLegalizeToHighLevelMhloPassOptions& options) {
  constexpr unsigned kBenefit = 10;
  if (options.enable_acosh_) {
    patterns->add(AcoshChloToMhlo, kBenefit);
  }
  patterns->add(ErfChloToMhlo, kBenefit);
  patterns->add(RaggedDotChloToMhlo, kBenefit);
  patterns->add(TopKChloToMhlo, kBenefit);
}

void populateChloToHloPatterns(MLIRContext* context,
                               TypeConverter* typeConverter,
                               RewritePatternSet* patterns) {
  populateChloToHighLevelMhloOpPatterns(
      context, patterns, mhlo::getDefaultChloToHighLevelMhloOptions());
  stablehlo::populateChloToStablehloPatterns(context, patterns);
  stablehlo::populateStablehloToHloPatterns(patterns, typeConverter, context);
}

}  // namespace chlo
}  // namespace mlir
