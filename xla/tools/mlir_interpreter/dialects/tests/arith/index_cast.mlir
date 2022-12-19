// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @i32() -> (index) {
  %c1 = arith.constant 42 : i32
  %index = arith.index_cast %c1 : i32 to index
  return %index : index
}

// CHECK-LABEL: @i32
// CHECK{LITERAL}: 42

func.func @i64() -> (index) {
  %c1 = arith.constant 43 : i64
  %index = arith.index_cast %c1 : i64 to index
  return %index : index
}

// CHECK-LABEL: @i64
// CHECK{LITERAL}: 43
