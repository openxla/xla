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

#ifndef XLA_GPU_EXPERIMENTAL_PASSES
#define XLA_GPU_EXPERIMENTAL_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Conversion from LMHLO dialects to Experimental runtime
//===----------------------------------------------------------------------===//

def ConvertToExperimental : Pass<"xla-gpu-to-experimental", "mlir::ModuleOp"> {
  let summary = "Converts LMHLO operations to Experimental runtime";

  let constructor = "xla::gpu::createConvertToExperimentalPass()";

  let dependentDialects = [
    "mlir::iree_compiler::IREE::Input::IREEInputDialect",
    "mlir::scf::SCFDialect",
    "xla::gpu::XlaGpuDialect",
  ];

  let options = [
    Option<"backend", "backend", "std::string", "\"hal\"",
           "The backend for launching kernels: hal or streamexecutor">,
  ];
}

#endif  // XLA_GPU_EXPERIMENTAL_PASSES
