// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @tensor() -> tensor<2xi16> {
  %cst = arith.constant dense<[42, 43]> : tensor<2xi16>
  return %cst : tensor<2xi16>
}

// CHECK-LABEL: @tensor
// CHECK{LITERAL}: [42, 43]

func.func @tensor_splat() -> tensor<2xi32> {
  %cst = arith.constant dense<42> : tensor<2xi32>
  return %cst : tensor<2xi32>
}

// CHECK-LABEL: @tensor_splat
// CHECK{LITERAL}: [42, 42]

func.func @scalar() -> i1 {
  %cst = arith.constant true
  return %cst : i1
}

// CHECK-LABEL: @scalar
// CHECK{LITERAL}: 1


