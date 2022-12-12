
// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @for() -> memref<4xi64> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %alloc = memref.alloc() : memref<4xi64>
  scf.for %i = %c0 to %c4 step %c2 {
    %1 = arith.index_cast %i: index to i64
    memref.store %1, %alloc[%i]: memref<4xi64>
  }
  return %alloc : memref<4xi64>
}

// CHECK-LABE: @for
// CHECK: Results
// CHECK-NEXT{LITERAL}: [0, 0, 2, 0]
