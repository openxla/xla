/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <utility>

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"  // NOLINT: Required to register quantization dialect.
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

#define DEBUG_TYPE "stablehlo-compose-uniform-quantization"

namespace mlir {
namespace mhlo {
namespace {

using quant::UniformQuantizedPerAxisType;
using quant::UniformQuantizedType;

#define GEN_PASS_DEF_STABLEHLOCOMPOSEUNIFORMQUANTIZATIONPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

constexpr StringRef kUniformQuantizeFunctionNameSubstring = "uniform_quantize";
constexpr StringRef kUniformDequantizeFunctionNameSubstring =
    "uniform_dequantize";

class StablehloComposeUniformQuantizationPass
    : public impl::StablehloComposeUniformQuantizationPassBase<
          StablehloComposeUniformQuantizationPass> {
 private:
  void runOnOperation() override;
};

class UniformQuantizeFunctionCallPattern {
 public:
  static FailureOr<UniformQuantizeFunctionCallPattern> match(
      func::CallOp callOp) {
    if (!callOp.getCallee().contains(kUniformQuantizeFunctionNameSubstring)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match uniformQuantizeCallOp - name doesn't "
                    "contain uniform_quantize\n");
      return failure();
    }

    Value inputValue = callOp.getOperand(0);
    if (!inputValue) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match UniformQuantizedFunctionCallPattern. "
                    "Input value is empty.\n");
      return failure();
    }

    auto inputValueType = inputValue.getType().dyn_cast_or_null<TensorType>();
    if (!inputValueType) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match UniformQuantizedFunctionCallPattern. "
                    "Input value's type must be a TensorType.\n");
      return failure();
    }

    if (Type inputElementType = inputValueType.getElementType();
        !inputElementType.isa<FloatType>()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match UniformQuantizedFunctionCallPattern. "
                    "Input value's element type must be a float. Got: "
                 << inputElementType << "\n");
      return failure();
    }

    Value zeroPointsValue = callOp.getOperand(1);
    auto zeroPointConstantOp = dyn_cast_or_null<stablehlo::ConstantOp>(
        zeroPointsValue.getDefiningOp());
    if (!zeroPointConstantOp) {
      llvm::dbgs() << "Failed to match zeroPointsValue\n";
      return failure();
    }

    Value inverseScalesValue = callOp.getOperand(2);
    auto inverseScaleConstantOp = dyn_cast_or_null<stablehlo::ConstantOp>(
        inverseScalesValue.getDefiningOp());
    if (!inverseScaleConstantOp) {
      llvm::dbgs() << "Failed to match inputInverseScalesConstantOp\n";
      return failure();
    }

    return UniformQuantizeFunctionCallPattern(callOp);
  }

  Value getInputValue() { return callOp.getOperand(0); }

  Value getZeroPointsValue() { return callOp.getOperand(1); }

  Value getInverseScalesValue() { return callOp.getOperand(2); }

  stablehlo::ConstantOp getZeroPointsConstantOp() {
    return cast<stablehlo::ConstantOp>(getZeroPointsValue().getDefiningOp());
  }

  stablehlo::ConstantOp getInverseScalesConstantOp() {
    return cast<stablehlo::ConstantOp>(getInverseScalesValue().getDefiningOp());
  }

  ElementsAttr getZeroPointsValueAttr() {
    return getZeroPointsConstantOp().getValue();
  }

  ElementsAttr getInverseScalesValueAttr() {
    return getInverseScalesConstantOp().getValue();
  }

  func::CallOp getCallOp() { return callOp; }

  FlatSymbolRefAttr getFunction() { return callOp.getCalleeAttr(); }

 private:
  explicit UniformQuantizeFunctionCallPattern(func::CallOp callOp)
      : callOp(callOp) {}

  func::CallOp callOp;
};

// Pattern for uniform_dequantize function call.
class UniformDequantizeFunctionCallPattern {
 public:
  // Returns Failure if it doesn't match. Returns the "wrapper" for the uniform
  // dequantization function call pattern when matched.
  static FailureOr<UniformDequantizeFunctionCallPattern> match(
      func::CallOp callOp) {
    if (!callOp.getCallee().contains(kUniformDequantizeFunctionNameSubstring)) {
      llvm::dbgs() << "Failed to match uniformDequantizeCallOp - name doesn't "
                      "contain uniform_quantize\n";
      return failure();
    }

    Value zeroPointsValue = callOp.getOperand(1);
    auto zeroPointConstantOp = dyn_cast_or_null<stablehlo::ConstantOp>(
        zeroPointsValue.getDefiningOp());
    if (!zeroPointConstantOp) {
      llvm::dbgs() << "Failed to match zeroPointsValue\n";
      return failure();
    }

    Value inverseScalesValue = callOp.getOperand(2);
    auto inverseScaleConstantOp = dyn_cast_or_null<stablehlo::ConstantOp>(
        inverseScalesValue.getDefiningOp());
    if (!inverseScaleConstantOp) {
      llvm::dbgs() << "Failed to match inputInverseScalesConstantOp\n";
      return failure();
    }

    return UniformDequantizeFunctionCallPattern(callOp);
  }

  Value getInputValue() { return callOp.getOperand(0); }

  Value getZeroPointsValue() { return callOp.getOperand(1); }

  Value getInverseScalesValue() { return callOp.getOperand(2); }

  stablehlo::ConstantOp getZeroPointsConstantOp() {
    return cast<stablehlo::ConstantOp>(getZeroPointsValue().getDefiningOp());
  }

  stablehlo::ConstantOp getInverseScalesConstantOp() {
    return cast<stablehlo::ConstantOp>(getInverseScalesValue().getDefiningOp());
  }

  ElementsAttr getZeroPointsValueAttr() {
    return getZeroPointsConstantOp().getValue();
  }

  ElementsAttr getInverseScalesValueAttr() {
    return getInverseScalesConstantOp().getValue();
  }

  func::CallOp getCallOp() { return callOp; }

  FlatSymbolRefAttr getFunction() { return callOp.getCalleeAttr(); }

 private:
  explicit UniformDequantizeFunctionCallPattern(func::CallOp callOp)
      : callOp(callOp) {}

  func::CallOp callOp;
};

// Matches the pattern for quantized convolution op and rewrites it to use
// uniform quantized types.
//
// Currently assumes asymmetric per-tensor quantization for activations and
// symmetric per-channel quantization for filters.
//
// This pattern represents the following derived equation, where:
// * rn = real (expressed) value for tensor n
// * qn = quantized value for tensor n
// * sn = scale for tensor n
// * zn = zero point for tensor n
//
// r3 = r1 * r2
//    = s1 (q1 - z1) * s2 (q2 - z2)
//    = s1 s2 (q1 q2 - q1 z2 - q2 z1 + z1 z2)
//
// * z2 is zero, because it assumes symmetric quantization for the filter:
//
//    = s1 s2 (q1 q2 - q2 z1)
//
// In StableHLO text representation, the pattern is as the following
// (simplified):
//
// ```
// %3 = call @uniform_quantize(%0, %1, %2)  // Quantize input (q1).
// %4 = stablehlo.convert %0  // i8 -> f32 cast trick for input.
// %6 = stablehlo.convert %5  // Optional: i8 -> f32 cast trick for filter.
// %7 = stablehlo.convolution(%4, %6)  // q1 * q2 (disguised in f32).
// %8 = stablehlo.reshape %1  // z1
// %9 = stablehlo.broadcast_in_dim %8
// %10 = stablehlo.convert %9  // i8 -> f32 cast trick for z1.
// %11 = stablehlo.convert %5  // i8 -> f32 cast trick for filter.
// %12 = stablehlo.convolution(%10, %11)  // q2 * z1
// %13 = stablehlo.subtract %7, %12  // q1 * q2 - q2 * z1
// %15 = stablehlo.broadcast_in_dim %14  // s1 * s2, precalculated.
// %16 = stablehlo.multiply %13 %15  // s1 s2 (q1 q2 - q2 z1)
//
// The following quant -> dequant pattern is a no-op, but is required to
// retrieve the quantization parameters for the output tensor.
//
// %19 = call @uniform_quantize_0(%16, %17, %18)
// %20 = call @uniform_dequantize(%19, %17, %18)
// ```
//
// The rewritten pattern looks like:
//
// ```
// %1 = stablehlo.constant  // Filter uniform quantized type.
// %2 = stablehlo.uniform_quantize %0  // Input f32 -> uniform quantized type.
// %3 = stablehlo.convolution(%1, %2)   // In uniform quantized type.
// %4 = stablehlo.uniform_dequantize %3  // Dequantize the output.
// ```
class ComposeUniformQuantizedConvolutionOp
    : public OpRewritePattern<stablehlo::ConvolutionOp> {
 public:
  using OpRewritePattern<stablehlo::ConvolutionOp>::OpRewritePattern;

  LogicalResult match(stablehlo::ConvolutionOp op) const final {
    auto inputI8ToF32ConvertOp = dyn_cast_or_null<stablehlo::ConvertOp>(
        op.getOperand(0).getDefiningOp());
    if (!inputI8ToF32ConvertOp) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match: Input is not defined by a "
                                 "stablehlo::ConvertOp.\n");
      return failure();
    }

    auto uniformQuantizeCallOp = dyn_cast_or_null<func::CallOp>(
        inputI8ToF32ConvertOp.getOperand().getDefiningOp());
    if (!uniformQuantizeCallOp) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match: Input is not quantized by a "
                                 "uniform_quantize function.\n");
      return failure();
    }

    auto uniformQuantizeCallPatternForInput =
        UniformQuantizeFunctionCallPattern::match(uniformQuantizeCallOp);
    if (failed(uniformQuantizeCallPatternForInput)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match uniform_quantized call op for input.\n");
      return failure();
    }

    // Go downstream.
    Value convOutputValue = op.getResult();
    if (auto outputElementType =
            convOutputValue.getType().cast<TensorType>().getElementType();
        !outputElementType.isa<FloatType>()) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Failed to match. Output type is expected to be a float. Got: "
          << outputElementType << "\n");
      return failure();
    }

    if (!op->hasOneUse()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match op - doesn't have a single use.\n");
      return failure();
    }

    auto subtractOp =
        dyn_cast_or_null<stablehlo::SubtractOp>(*convOutputValue.user_begin());
    if (!subtractOp) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match subtractOp\n");
      return failure();
    }
    if (!subtractOp->hasOneUse()) {
      llvm::dbgs() << "Failed to match op - doesn't have a single use.\n";
      return failure();
    }

    auto otherConvOp = dyn_cast_or_null<stablehlo::ConvolutionOp>(
        subtractOp.getOperand(1).getDefiningOp());
    if (!otherConvOp) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match otherConvOp\n");
      return failure();
    }

    if (!isa<stablehlo::ConstantOp>(
            otherConvOp.getOperand(1).getDefiningOp()) &&
        !isa<stablehlo::ConvertOp>(otherConvOp.getOperand(1).getDefiningOp())) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match filter of otherConvOp\n");
      return failure();
    }

    auto otherZpI8ToF32ConvertOp = dyn_cast_or_null<stablehlo::ConvertOp>(
        otherConvOp.getOperand(0).getDefiningOp());
    if (!otherZpI8ToF32ConvertOp) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match otherZpI8ToF32ConvertOp\n");
      return failure();
    }

    auto otherZpBroadcastInDimOp =
        dyn_cast_or_null<stablehlo::BroadcastInDimOp>(
            otherZpI8ToF32ConvertOp.getOperand().getDefiningOp());
    if (!otherZpBroadcastInDimOp) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match otherZpBroadcastInDimOp\n");
      return failure();
    }

    auto otherZpReshapeOp = dyn_cast_or_null<stablehlo::ReshapeOp>(
        otherZpBroadcastInDimOp.getOperand().getDefiningOp());
    if (!otherZpReshapeOp) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match otherZpReshapeOp\n");
      return failure();
    }

    auto otherInputZeroPointsConstantOp =
        dyn_cast_or_null<stablehlo::ConstantOp>(
            otherZpReshapeOp.getOperand().getDefiningOp());
    if (!otherInputZeroPointsConstantOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match otherInputZeroPointsConstantOp\n");
      return failure();
    }

    auto combinedScaleMultiplyOp = dyn_cast_or_null<stablehlo::MulOp>(
        *subtractOp.getResult().user_begin());
    if (!combinedScaleMultiplyOp) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match combinedScaleMultiplyOp\n");
      return failure();
    }
    if (!combinedScaleMultiplyOp->hasOneUse()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match op - doesn't have a single use.\n");
      return failure();
    }

    auto scaleCombinedBroadcastInDimOp =
        dyn_cast_or_null<stablehlo::BroadcastInDimOp>(
            combinedScaleMultiplyOp.getOperand(1).getDefiningOp());
    if (!scaleCombinedBroadcastInDimOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match scaleCombinedBroadcastInDimOp\n");
      return failure();
    }

    // s1 * s2
    auto combinedScaleConstantOp = dyn_cast_or_null<stablehlo::ConstantOp>(
        scaleCombinedBroadcastInDimOp.getOperand().getDefiningOp());
    if (!combinedScaleConstantOp) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match combinedScaleConstantOp.\n");
      return failure();
    }

    // Quantize -> Dequantize following r3.
    auto outputUniformQuantizeCallOp = dyn_cast_or_null<func::CallOp>(
        *combinedScaleMultiplyOp.getResult().user_begin());
    if (!outputUniformQuantizeCallOp->hasOneUse()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match op - doesn't have a single use.\n");
      return failure();
    }

    if (failed(UniformQuantizeFunctionCallPattern::match(
            outputUniformQuantizeCallOp))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match op outputUniformQuantizeCallOp\n");
      return failure();
    }

    auto outputUniformDequantizeCallOp = dyn_cast_or_null<func::CallOp>(
        *outputUniformQuantizeCallOp.getResult(0).user_begin());
    if (!outputUniformDequantizeCallOp) {
      return failure();
    }
    if (failed(UniformDequantizeFunctionCallPattern::match(
            outputUniformDequantizeCallOp))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to match outputUniformDequantizeCallOp.\n");
      return failure();
    }

    Operation* filterOp = op.getOperand(1).getDefiningOp();
    if (!isa<stablehlo::ConstantOp>(filterOp) &&
        !(isa<stablehlo::ConvertOp>(filterOp) &&
          isa<stablehlo::ConstantOp>(
              filterOp->getOperand(0).getDefiningOp()))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to match filterConstantOp\n");
      return failure();
    }

    return success();
  }

  void rewrite(stablehlo::ConvolutionOp op,
               PatternRewriter& rewriter) const final {
    // Rewrite `call @uniform_quantize` -> `stablehlo.uniform_quantize`.
    auto inputI8ToF32ConvertOp =
        cast<stablehlo::ConvertOp>(op.getOperand(0).getDefiningOp());
    auto uniformQuantizeCallOp =
        cast<func::CallOp>(inputI8ToF32ConvertOp.getOperand().getDefiningOp());

    auto uniformQuantizeCallPatternForInput =
        *UniformQuantizeFunctionCallPattern::match(uniformQuantizeCallOp);
    const float inputInverseScalesValue =
        uniformQuantizeCallPatternForInput.getInverseScalesValueAttr()
            .getSplatValue<APFloat>()
            .convertToFloat();
    const float inputScaleValue = 1.0 / inputInverseScalesValue;
    const int64_t inputZeroPointValue =
        uniformQuantizeCallPatternForInput.getZeroPointsValueAttr()
            .getSplatValue<APInt>()
            .getSExtValue();

    Value inputValue = uniformQuantizeCallPatternForInput.getInputValue();
    UniformQuantizedType inputQuantizedElementType =
        createI8F32UniformQuantizedType(uniformQuantizeCallOp.getLoc(),
                                        rewriter, inputScaleValue,
                                        inputZeroPointValue);
    auto inputUniformQuantizeOp = rewriter.create<stablehlo::UniformQuantizeOp>(
        uniformQuantizeCallOp.getLoc(),
        /*result=*/
        inputValue.getType().cast<TensorType>().clone(
            inputQuantizedElementType),
        /*operand=*/inputValue);

    rewriter.replaceAllUsesWith(inputI8ToF32ConvertOp.getResult(),
                                inputUniformQuantizeOp.getResult());

    // Rewrite filter constant.
    Operation* filterOp = op.getOperand(1).getDefiningOp();

    // Retrieve the i8 filter values.
    DenseElementsAttr filterI8ValueAttr = nullptr;
    if (auto filterConstantOp =
            dyn_cast_or_null<stablehlo::ConstantOp>(filterOp);
        filterConstantOp) {
      // This is i8 values disguised as f32 (due to the upcast trick). Simply
      // cast them to i8.
      ElementsAttr filterValue = filterConstantOp.getValue();
      filterI8ValueAttr = filterValue.cast<DenseFPElementsAttr>().mapValues(
          rewriter.getI8Type(), [](const APFloat& val) -> APInt {
            APSInt convertedInt(/*BitWidth=*/8, /*isUnsigned=*/false);
            bool ignored;
            val.convertToInteger(convertedInt, APFloat::rmTowardZero, &ignored);
            return convertedInt;
          });

    } else if (isa<stablehlo::ConvertOp>(filterOp) &&
               isa<stablehlo::ConstantOp>(
                   filterOp->getOperand(0).getDefiningOp())) {
      filterI8ValueAttr =
          cast<stablehlo::ConstantOp>(filterOp->getOperand(0).getDefiningOp())
              .getValue()
              .cast<DenseIntElementsAttr>();
    }

    // Create Uniform Quantized constant for the filter.
    auto subtractOp = cast<stablehlo::SubtractOp>(*op.getResult().user_begin());
    auto otherConvOp = cast<stablehlo::ConvolutionOp>(
        subtractOp.getOperand(1).getDefiningOp());
    auto combinedScaleMultiplyOp =
        cast<stablehlo::MulOp>(*subtractOp.getResult().user_begin());

    auto scaleCombinedBroadcastInDimOp = cast<stablehlo::BroadcastInDimOp>(
        combinedScaleMultiplyOp.getOperand(1).getDefiningOp());
    auto combinedScaleConstantOp = cast<stablehlo::ConstantOp>(
        scaleCombinedBroadcastInDimOp.getOperand().getDefiningOp());

    SmallVector<double> filterScaleValues;
    for (const auto combinedScaleValue : combinedScaleConstantOp.getValue()
                                             .cast<DenseFPElementsAttr>()
                                             .getValues<float>()) {
      const float filterScaleValue =
          combinedScaleValue * inputInverseScalesValue;
      filterScaleValues.emplace_back(filterScaleValue);
    }

    // Assumes it is symmetric.
    SmallVector<int64_t> filterZeroPointValues(
        /*Size=*/filterScaleValues.size(), /*Value=*/0);

    // Use quantization dimension = 3 that corresponds to the output channel
    // dimension, assuming the filter format is `[0, 1, i, o]`.
    // TODO: b/291029962 - Lift the assumption above and retrieve the
    // quantization dimension from the `dimension_numbers` attribute.
    UniformQuantizedPerAxisType filterQuantizedElementType =
        createI8F32UniformQuantizedPerAxisType(filterOp->getLoc(), rewriter,
                                               filterScaleValues,
                                               filterZeroPointValues,
                                               /*quantizationDimension=*/3);

    // Create a new constant op for the filter in i8.
    auto quantizedFilterConstantOp = rewriter.create<stablehlo::ConstantOp>(
        rewriter.getUnknownLoc(),
        /*output=*/
        filterI8ValueAttr.getType().clone(filterQuantizedElementType),
        /*value=*/filterI8ValueAttr);

    // Replace filter uses with uniform quantized filter.
    rewriter.replaceAllUsesWith(filterOp->getResult(0),
                                quantizedFilterConstantOp.getResult());

    // Replace conv op with a new convolution op that has quantized output type.
    // Quantize -> Dequantize following r3.
    auto outputUniformQuantizeCallOp =
        cast<func::CallOp>(*combinedScaleMultiplyOp.getResult().user_begin());

    auto outputUniformQuantizeCallPattern =
        *UniformQuantizeFunctionCallPattern::match(outputUniformQuantizeCallOp);

    const int outputZeroPointValue =
        outputUniformQuantizeCallPattern.getZeroPointsValueAttr()
            .getSplatValue<APInt>()
            .getSExtValue();
    const float outputInverseScaleValue =
        outputUniformQuantizeCallPattern.getInverseScalesValueAttr()
            .getSplatValue<APFloat>()
            .convertToFloat();

    UniformQuantizedType outputUniformQuantizedType =
        createI8F32UniformQuantizedType(
            outputUniformQuantizeCallOp.getLoc(), rewriter,
            /*scale=*/1.0 / outputInverseScaleValue, outputZeroPointValue);

    Value convOutputValue = op.getResult();
    auto outputUniformQuantizedTensorType = RankedTensorType::getChecked(
        rewriter.getUnknownLoc(),
        /*shape=*/convOutputValue.getType().cast<TensorType>().getShape(),
        outputUniformQuantizedType);

    SmallVector<Type> newConvOutputTypes = {outputUniformQuantizedTensorType};
    auto newConvOpWithOutputType = rewriter.create<stablehlo::ConvolutionOp>(
        op.getLoc(), newConvOutputTypes, op.getOperands(), op->getAttrs());

    rewriter.replaceAllUsesWith(op.getResult(),
                                newConvOpWithOutputType.getResult());

    auto newOutputDequantOp = rewriter.create<stablehlo::UniformDequantizeOp>(
        rewriter.getUnknownLoc(),
        /*operand=*/newConvOpWithOutputType);

    auto outputUniformDequantizeCallOp = cast<func::CallOp>(
        *outputUniformQuantizeCallOp.getResult(0).user_begin());

    rewriter.replaceAllUsesWith(outputUniformDequantizeCallOp.getResult(0),
                                newOutputDequantOp.getResult());

    // Erase unused ops in the reverse order.
    rewriter.eraseOp(outputUniformDequantizeCallOp);
    rewriter.eraseOp(outputUniformQuantizeCallOp);
    rewriter.eraseOp(combinedScaleMultiplyOp);
    rewriter.eraseOp(subtractOp);
    rewriter.eraseOp(otherConvOp);
    rewriter.eraseOp(op);
    rewriter.eraseOp(inputI8ToF32ConvertOp);
    rewriter.eraseOp(uniformQuantizeCallOp);
  }

 private:
  // Creates a `UniformQuantizedType` with the given `scale` and `zeroPoint`
  // values. The produced type has f32 as its expressed type and i8 as its
  // storage type with default storage type min and max values, set to -128 and
  // 127, respectively.
  static UniformQuantizedType createI8F32UniformQuantizedType(
      Location loc, PatternRewriter& rewriter, const float scale,
      const int64_t zeroPoint) {
    return UniformQuantizedType::getChecked(
        loc, /*flags=*/true, /*storageType=*/rewriter.getI8Type(),
        /*expressedType=*/rewriter.getF32Type(), scale, zeroPoint,
        /*storageTypeMin=*/-128, /*storageTypeMax=*/127);
  }

  // Creates a `UniformQuantizedPerAxisType` with the given `scales` and
  // `zeroPoints` values. The produced type has f32 as its expressed type and i8
  // as its storage type with default storage type min and max values, set to
  // -128 and 127, respectively.
  static UniformQuantizedPerAxisType createI8F32UniformQuantizedPerAxisType(
      Location loc, PatternRewriter& rewriter, const ArrayRef<double> scales,
      const ArrayRef<int64_t> zeroPoints, const int quantizationDimension) {
    return UniformQuantizedPerAxisType::getChecked(
        loc, /*flags=*/true, /*storageType=*/rewriter.getI8Type(),
        /*expressedType=*/rewriter.getF32Type(), scales, zeroPoints,
        /*quantizedDimension=*/quantizationDimension, /*storageTypeMin=*/-128,
        /*storageTypeMax=*/127);
  }
};

void StablehloComposeUniformQuantizationPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext& ctx = getContext();

  RewritePatternSet patterns(&ctx);
  patterns.add<ComposeUniformQuantizedConvolutionOp>(&ctx);

  if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
    moduleOp.emitError()
        << "Failed to compose stablehlo uniform quantized ops.";
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createStablehloComposeUniformQuantizationPass() {
  return std::make_unique<StablehloComposeUniformQuantizationPass>();
}

}  // namespace mhlo
}  // namespace mlir
