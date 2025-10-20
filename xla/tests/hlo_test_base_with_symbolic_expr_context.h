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

#ifndef XLA_TESTS_HLO_TEST_BASE_WITH_SYMBOLIC_EXPR_CONTEXT_H_
#define XLA_TESTS_HLO_TEST_BASE_WITH_SYMBOLIC_EXPR_CONTEXT_H_

#include "mlir/IR/MLIRContext.h"
#include "xla/service/gpu/model/experimental/symbolic_expr.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {

class HloTestBaseWithSymbolicExprContext : public HloTestBase {
 public:
  gpu::SymbolicExprContext* symbolic_expr_context() {
    return &symbolic_expr_context_;
  }

 private:
  mlir::MLIRContext mlir_context_;
  gpu::SymbolicExprContext symbolic_expr_context_{&mlir_context_};
};

}  // namespace xla

#endif  // XLA_TESTS_HLO_TEST_BASE_WITH_SYMBOLIC_EXPR_CONTEXT_H_
