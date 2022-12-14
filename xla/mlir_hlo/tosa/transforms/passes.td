/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

include "mlir/Pass/PassBase.td"

def TosaLegalizeMhloPass : Pass<"tosa-legalize-mhlo", "mlir::func::FuncOp"> {
  let summary = "Legalize from MHLO to TOSA";
  let constructor = "createLegalizeMhloPass()";
  let dependentDialects = ["::mlir::tosa::TosaDialect"];
}

def TosaPrepareMhloPass : Pass<"tosa-prepare-mhlo", "mlir::func::FuncOp"> {
  let summary = "Prepare MHLO for lowering to TOSA";
  let description = [{
    This pass adds rewriters to make MHLO ops more compatible with TOSA ops.
    Currently simplifies mhlo.dot_general into mhlo.dot for easier lowering.
  }];
  let constructor = "createPrepareMhloPass()";
  let dependentDialects = ["::mlir::tosa::TosaDialect"];
}