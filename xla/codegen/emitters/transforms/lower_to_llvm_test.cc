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

#include <gtest/gtest.h>

#include <string>

#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"

// Validates LowerToLLVMPass for Intel GPU targets. This lives as a gtest rather
// than a lit test because the current gpu_device_info option only understands
// CUDA and ROCm compute capabilities, so we can't express oneAPI metadata there
// yet.
namespace xla::emitters {
namespace {

stream_executor::DeviceDescription CreateIntelDeviceDescription() {
  stream_executor::GpuDeviceInfoProto proto;
  stream_executor::DeviceDescription device =
      stream_executor::DeviceDescription::FromProto(proto).value();
  device.set_name("Intel(R) Arc(TM) B580 Graphics");
  device.set_device_vendor("Intel");
  return device;
}

TEST(LowerToLLVMPassTest, Log1p) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::math::MathDialect,
                      mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect>();

  constexpr absl::string_view kModuleStr = R"mlir(
    module {
      func.func @log1p(%arg0: f32) -> f32 {
        %0 = math.log1p %arg0 : f32
        return %0 : f32
      }
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kModuleStr, &context);
  ASSERT_TRUE(module) << "Failed to parse test module";
  mlir::ModuleOp module_op = *module;

  mlir::PassManager pm(&context);
  pm.addPass(CreateLowerToLLVMPass(CreateIntelDeviceDescription()));
  ASSERT_TRUE(mlir::succeeded(pm.run(module_op.getOperation())));
}

}  // namespace
}  // namespace xla::emitters
