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


#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_IR_TRITON_XLA_DIALECT_TD_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_IR_TRITON_XLA_DIALECT_TD_

include "mlir/IR/DialectBase.td"

def XlaTritonDialect : Dialect {
  let name = "triton_xla";

  let description = [{
    This dialect contains ops included in the xla extension point for Triton.
  }];

  let cppNamespace = "::mlir::triton::xla";
  let useDefaultAttributePrinterParser = 1;
}

#endif // XLA_BACKENDS_GPU_CODEGEN_TRITON_IR_TRITON_XLA_DIALECT_TD_
