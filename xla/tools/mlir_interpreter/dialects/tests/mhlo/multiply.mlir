// RUN: mlir-interpreter-runner %s | FileCheck %s

func.func @main() -> tensor<2x3xf32> {
  %lhs = mhlo.constant dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>
  %rhs = mhlo.constant dense<[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]> : tensor<2x3xf32>
  %result = mhlo.multiply %lhs, %rhs : tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// CHECK{LITERAL}: [[0.000000e+00, 2.000000e+01, 6.000000e+01], [1.200000e+02, 2.000000e+02, 3.000000e+02]]