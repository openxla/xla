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

#ifndef XLA_CODEGEN_EMITTERS_TRANSFORMS_LOWERING_UTILS_H_
#define XLA_CODEGEN_EMITTERS_TRANSFORMS_LOWERING_UTILS_H_

#include "absl/strings/str_format.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Operation.h"
#include "mlir/lib/Conversion/GPUCommon/GPUOpsLowering.h"
#include "mlir/lib/Conversion/GPUCommon/OpToFuncCallLowering.h"

namespace xla {
namespace emitters {

// Ensure AMDGPU allocas use address space 5 (private).
// AMDGPU requires allocas in AS5, but MLIR lowering creates them in AS0.
void EnsureAMDGPUAllocasUseAS5(mlir::Operation* operation);

namespace spirv {
namespace mm = ::mlir::math;
template <typename... Ops>
struct SPIRVMathOps {};

inline auto getSPIRVMathOps() {
  return SPIRVMathOps<
      mm::AcosOp, mm::AcoshOp, mm::AsinOp, mm::AsinhOp, mm::Atan2Op, mm::AtanOp,
      mm::AtanhOp, mm::CosOp, mm::CoshOp, mm::ExpM1Op, mm::ExpOp, mm::Log1pOp,
      mm::LogOp, mm::SinOp, mm::SinhOp, mm::TanOp, mm::TanhOp>{};
}

// Lowers math ops to SPIR-V OCL driver intrinsics to preserve accuracy,
// particularly for small-magnitude inputs where generic MLIR lowering may
// lose precision.
template <typename Op>
inline void populateLLVMSPVMathOpPatterns(mlir::LLVMTypeConverter& converter,
                                          mlir::RewritePatternSet& patterns) {
  auto op_name =
      mlir::OperationName(Op::getOperationName(), &converter.getContext())
          .stripDialect();
  auto base =
      absl::StrFormat("_Z%u__spirv_ocl_%s", op_name.size() + 12, op_name.str());

  patterns.add<mlir::ScalarizeVectorOpLowering<Op>>(converter, 1);
  patterns.add<mlir::OpToFuncCallLowering<Op>>(converter, base + "f",
                                               base + "d", "", "", "", 1);
}

template <typename... Ops>
void populateMathToLLVMSPVConversionPatterns(
    spirv::SPIRVMathOps<Ops...>, mlir::LLVMTypeConverter& converter,
    mlir::RewritePatternSet& patterns) {
  (spirv::populateLLVMSPVMathOpPatterns<Ops>(converter, patterns), ...);
}

}  // namespace spirv
}  // namespace emitters
}  // namespace xla

#endif  // XLA_CODEGEN_EMITTERS_TRANSFORMS_LOWERING_UTILS_H_
