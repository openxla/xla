// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

#matmul_trait = {
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

func.func @matmul() -> (tensor<2x2xi32>, tensor<2x2xi32>) {
  %lhs = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %rhs = arith.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  %init = tensor.empty() : tensor<2x2xi32>
  %ret = linalg.generic #matmul_trait
    ins(%lhs, %rhs : tensor<2x3xi32>, tensor<3x2xi32>)
    outs(%init : tensor<2x2xi32>) {
    ^bb(%a: i32, %b: i32, %c: i32):
      %d = arith.muli %a, %b: i32
      %e = arith.addi %c, %d: i32
      linalg.yield %e : i32
    } -> tensor<2x2xi32>
  return %ret, %init : tensor<2x2xi32>, tensor<2x2xi32>
}

// CHECK-LABEL: @matmul
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[22, 28], [49, 64]]
// CHECK-NEXT{LITERAL}: [[0, 0], [0, 0]]

func.func @bufferized() -> memref<2x2xi32> {
  %lhs = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %rhs = arith.constant dense<[[2, 1], [3, 4], [5, 6]]> : tensor<3x2xi32>
  %alloc = memref.alloc() : memref<2x2xi32>
  linalg.generic #matmul_trait
    ins(%lhs, %rhs : tensor<2x3xi32>, tensor<3x2xi32>)
    outs(%alloc : memref<2x2xi32>) {
    ^bb(%a: i32, %b: i32, %c: i32):
      %d = arith.muli %a, %b: i32
      %e = arith.addi %c, %d: i32
      linalg.yield %e : i32
    }
  return %alloc : memref<2x2xi32>
}

// CHECK-LABEL: @bufferized
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[23, 27], [53, 60]]
