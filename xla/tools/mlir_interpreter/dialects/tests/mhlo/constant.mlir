// RUN: mlir-interpreter-runner %s | FileCheck %s

func.func @main() -> (tensor<2x3xi32>, tensor<2x3xf32>, tensor<0x0x3xi16>, tensor<f64>) {
  %i32 = mhlo.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>
  %f32 = mhlo.constant dense<[[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]> : tensor<2x3xf32>
  %empty = mhlo.constant dense<> : tensor<0x0x3xi16>
  %scalar = mhlo.constant dense<3.14> : tensor<f64>
  return %i32, %f32, %empty, %scalar : tensor<2x3xi32>, tensor<2x3xf32>, tensor<0x0x3xi16>, tensor<f64>
}

// CHECK{LITERAL}: [[0, 1, 2], [3, 4, 5]]
// CHECK{LITERAL}: [[0.000000e+00, 1.000000e-01, 2.000000e-01], [3.000000e-01, 4.000000e-01, 5.000000e-01]]
// CHECK{LITERAL}: []
// CHECK{LITERAL}: 3.140000e+00