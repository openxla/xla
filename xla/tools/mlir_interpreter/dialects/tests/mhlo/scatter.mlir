// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @bounds_check() -> tensor<10xi32> {
  %operand = mhlo.constant dense<0> : tensor<10xi32>
  %indices = mhlo.constant dense<[[1], [8], [-1]]> : tensor<3x1xi32>
  %updates = mhlo.constant dense<[[4, 5, 6], [6, 7, 8], [8, 9, 10]]> : tensor<3x3xi32>
  %scatter = "mhlo.scatter"(%operand, %indices, %updates) ({
  ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
    %add = mhlo.add %lhs, %rhs : tensor<i32>
    "mhlo.return"(%add) : (tensor<i32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [0],
      inserted_window_dims = [],
      index_vector_dim = 1,
      scatter_dims_to_operand_dims = [0]
    >
  } : (tensor<10xi32>, tensor<3x1xi32>, tensor<3x3xi32>) -> tensor<10xi32>
  return %scatter : tensor<10xi32>
}

// CHECK-LABEL: @bounds_check
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [0, 4, 6, 8, 0, 0, 0, 0, 0, 0]