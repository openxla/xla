// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @eq() -> (tensor<i1>, tensor<i1>) {
  %c1 = arith.constant dense<1> : tensor<i32>
  %c2 = arith.constant dense<2> : tensor<i32>
  %0 = mhlo.compare EQ, %c1, %c2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = mhlo.compare EQ, %c1, %c1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %0, %1 : tensor<i1>, tensor<i1>
}

// CHECK-LABEL: @eq
// CHECK-NEXT: Results
// CHECK-NEXT: false
// CHECK-NEXT: true

func.func @ne() -> (tensor<i1>, tensor<i1>) {
  %c1 = arith.constant dense<1> : tensor<i32>
  %c2 = arith.constant dense<2> : tensor<i32>
  %0 = mhlo.compare NE, %c1, %c2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = mhlo.compare NE, %c1, %c1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %0, %1 : tensor<i1>, tensor<i1>
}

// CHECK-LABEL: @ne
// CHECK-NEXT: Results
// CHECK-NEXT: true
// CHECK-NEXT: false

func.func @ge() -> (tensor<i1>, tensor<i1>, tensor<i1>) {
  %c1 = arith.constant dense<1> : tensor<i32>
  %c2 = arith.constant dense<2> : tensor<i32>
  %0 = mhlo.compare GE, %c1, %c2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = mhlo.compare GE, %c1, %c1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = mhlo.compare GE, %c2, %c1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %0, %1, %2 : tensor<i1>, tensor<i1>, tensor<i1>
}

// CHECK-LABEL: @ge
// CHECK-NEXT: Results
// CHECK-NEXT: false
// CHECK-NEXT: true
// CHECK-NEXT: true

func.func @gt() -> (tensor<i1>, tensor<i1>, tensor<i1>) {
  %c1 = arith.constant dense<1> : tensor<i32>
  %c2 = arith.constant dense<2> : tensor<i32>
  %0 = mhlo.compare GT, %c1, %c2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = mhlo.compare GT, %c1, %c1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = mhlo.compare GT, %c2, %c1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %0, %1, %2 : tensor<i1>, tensor<i1>, tensor<i1>
}

// CHECK-LABEL: @gt
// CHECK-NEXT: Results
// CHECK-NEXT: false
// CHECK-NEXT: false
// CHECK-NEXT: true

func.func @le() -> (tensor<i1>, tensor<i1>, tensor<i1>) {
  %c1 = arith.constant dense<1> : tensor<i32>
  %c2 = arith.constant dense<2> : tensor<i32>
  %0 = mhlo.compare LE, %c1, %c2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = mhlo.compare LE, %c1, %c1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = mhlo.compare LE, %c2, %c1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %0, %1, %2 : tensor<i1>, tensor<i1>, tensor<i1>
}

// CHECK-LABEL: @le
// CHECK-NEXT: Results
// CHECK-NEXT: true
// CHECK-NEXT: true
// CHECK-NEXT: false

func.func @lt() -> (tensor<i1>, tensor<i1>, tensor<i1>) {
  %c1 = arith.constant dense<1> : tensor<i32>
  %c2 = arith.constant dense<2> : tensor<i32>
  %0 = mhlo.compare LT, %c1, %c2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = mhlo.compare LT, %c1, %c1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = mhlo.compare LT, %c2, %c1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %0, %1, %2 : tensor<i1>, tensor<i1>, tensor<i1>
}

// CHECK-LABEL: @lt
// CHECK-NEXT: Results
// CHECK-NEXT: true
// CHECK-NEXT: false
// CHECK-NEXT: false
