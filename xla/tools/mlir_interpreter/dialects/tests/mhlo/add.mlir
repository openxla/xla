// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @tensor() -> tensor<2x3xi32> {
  %lhs = mhlo.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>
  %rhs = mhlo.constant dense<[[10, 20, 30], [40, 50, 60]]> : tensor<2x3xi32>
  %result = mhlo.add %lhs, %rhs : tensor<2x3xi32>
  return %result : tensor<2x3xi32>
}

// CHECK-LABEL: @tensor
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[10, 21, 32], [43, 54, 65]]

func.func @scalar() -> tensor<i32> {
  %lhs = mhlo.constant dense<40> : tensor<i32>
  %rhs = mhlo.constant dense<2> : tensor<i32>
  %result = mhlo.add %lhs, %rhs : tensor<i32>
  return %result : tensor<i32>
}

// CHECK-LABEL: @scalar
// CHECK-NEXT: Results
// CHECK-NEXT: 42
