// RUN: fusion_compiler_opt %s -xtile-cpu-legalize-i1-storage -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: xtile.entry_func @i1_nd_extract_insert(
// CHECK-SAME:    %[[IN:[^:]+]]: memref<16x32xi8>,
// CHECK-SAME:    %[[OUT:[^:]+]]: memref<16x32xi8>,
// CHECK-SAME:    %[[PID:[^:]+]]: index)
xtile.entry_func @i1_nd_extract_insert(
    %in: memref<16x32xi1>,
    %out: memref<16x32xi1>,
    %pid: index) {
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0> : tensor<4x8xi8>
  // CHECK: %[[LOAD:.*]] = xtile.extract %[[IN]][%[[PID]], %{{.*}}] [4, 8] [1, 1] : memref<16x32xi8> -> tensor<4x8xi8>
  // CHECK: %[[CMP:.*]] = arith.cmpi ne, %[[LOAD]], %[[ZERO]] : tensor<4x8xi8>
  // CHECK: %[[EXT:.*]] = arith.extui %[[CMP]] : tensor<4x8xi1> to tensor<4x8xi8>
  // CHECK: xtile.insert %[[EXT]] into %[[OUT]][%[[PID]], %{{.*}}] [4, 8] [1, 1] : tensor<4x8xi8> -> memref<16x32xi8>
  %tile = xtile.extract %in[%pid, %c0] [4, 8] [1, 1] : memref<16x32xi1> -> tensor<4x8xi1>
  xtile.insert %tile into %out[%pid, %c0] [4, 8] [1, 1] : tensor<4x8xi1> -> memref<16x32xi1>
  xtile.return
}

// -----

// CHECK-LABEL: xtile.entry_func @i1_scalar_extract_insert(
// CHECK-SAME:    %[[IN:[^:]+]]: memref<i8>,
// CHECK-SAME:    %[[OUT:[^:]+]]: memref<i8>,
// CHECK-SAME:    %[[PID:[^:]+]]: index)
xtile.entry_func @i1_scalar_extract_insert(
    %in: memref<i1>,
    %out: memref<i1>,
    %pid: index) {
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0> : tensor<i8>
  // CHECK: %[[LOAD:.*]] = xtile.extract %[[IN]][] [] [] : memref<i8> -> tensor<i8>
  // CHECK: %[[CMP:.*]] = arith.cmpi ne, %[[LOAD]], %[[ZERO]] : tensor<i8>
  // CHECK: %[[EXT:.*]] = arith.extui %[[CMP]] : tensor<i1> to tensor<i8>
  // CHECK: xtile.insert %[[EXT]] into %[[OUT]][] [] [] : tensor<i8> -> memref<i8>
  %tile = xtile.extract %in[] [] [] : memref<i1> -> tensor<i1>
  xtile.insert %tile into %out[] [] [] : tensor<i1> -> memref<i1>
  xtile.return
}

// -----

func.func @unlegalized_func_arg_i1_extract(%in: memref<4xi1>) -> tensor<4xi1> {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{'xtile.extract' op failed to legalize i1 tile extract}}
  %tile = xtile.extract %in[%c0] [4] [1] : memref<4xi1> -> tensor<4xi1>
  return %tile : tensor<4xi1>
}
