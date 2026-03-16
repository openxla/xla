// RUN: xla-opt %s -triton-xla-atomics-generic | FileCheck %s

// Test lowering of AtomicWriteOp to pure Triton AtomicRMWOp using generic Triton ops

// CHECK-LABEL: @test_atomic_write
tt.func @test_atomic_write(%ptr: !tt.ptr<i32>, %value: i32) {
  // CHECK-NOT: triton_xla.atomic_write
  // CHECK: tt.atomic_rmw exch
  triton_xla.atomic_write sys, release, %ptr, %value : (!tt.ptr<i32>, i32) -> ()
  tt.return
}

// CHECK-LABEL: @test_atomic_write_with_mask
tt.func @test_atomic_write_with_mask(%ptr: !tt.ptr<i32>, %value: i32, %mask: i1) {
  // CHECK-NOT: triton_xla.atomic_write
  // CHECK: tt.atomic_rmw exch
  triton_xla.atomic_write sys, release, %ptr, %value, %mask : (!tt.ptr<i32>, i32, i1) -> ()
  tt.return
}

// Scalar atomic_spin_wait is not supported by the generic pass
// (only tensor pointers are supported)
// These tests verify that scalar operations pass through unchanged
// CHECK-LABEL: @test_atomic_spin_wait
tt.func @test_atomic_spin_wait(%ptr: !tt.ptr<i32>, %expected: i32) {
  // CHECK: triton_xla.atomic_spin_wait
  triton_xla.atomic_spin_wait sys, acquire, %ptr, less_than, %expected : (!tt.ptr<i32>, i32) -> ()
  tt.return
}

// CHECK-LABEL: @test_atomic_spin_wait_eq
tt.func @test_atomic_spin_wait_eq(%ptr: !tt.ptr<i32>, %expected: i32) {
  // CHECK: triton_xla.atomic_spin_wait
  triton_xla.atomic_spin_wait sys, acquire, %ptr, equal_to, %expected : (!tt.ptr<i32>, i32) -> ()
  tt.return
}

// CHECK-LABEL: @test_atomic_write_vectorized
tt.func @test_atomic_write_vectorized(%ptr: tensor<4x!tt.ptr<i32>>, %value: i32) {
  // CHECK-NOT: triton_xla.atomic_write
  // CHECK: [[SPLAT:%.*]] = tt.splat %arg1 : i32 -> tensor<4xi32>
  // CHECK: tt.atomic_rmw exch, release, sys, %arg0, [[SPLAT]] : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
  triton_xla.atomic_write sys, release, %ptr, %value : (tensor<4x!tt.ptr<i32>>, i32) -> ()
  tt.return
}

// CHECK-LABEL: @test_atomic_write_vectorized_with_mask
tt.func @test_atomic_write_vectorized_with_mask(%ptr: tensor<4x!tt.ptr<i32>>, %value: i32, %mask: tensor<4xi1>) {
  // CHECK-NOT: triton_xla.atomic_write
  // CHECK: [[SPLAT:%.*]] = tt.splat %arg1 : i32 -> tensor<4xi32>
  // CHECK: tt.atomic_rmw exch, release, sys, %arg0, [[SPLAT]], %arg2 : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
  triton_xla.atomic_write sys, release, %ptr, %value, %mask : (tensor<4x!tt.ptr<i32>>, i32, tensor<4xi1>) -> ()
  tt.return
}

// CHECK-LABEL: @test_atomic_spin_wait_vectorized
tt.func @test_atomic_spin_wait_vectorized(%ptr: tensor<4x!tt.ptr<i32>>, %expected: i32) {
  // CHECK-NOT: triton_xla.atomic_spin_wait
  // CHECK: arith.constant dense<0> : tensor<4xi32>
  // CHECK: tt.splat {{%.*}} : i32 -> tensor<4xi32>
  // CHECK: scf.while
  // CHECK: tt.atomic_rmw add, acquire, sys
  // CHECK: arith.cmpi slt
  // CHECK: tt.reduce
  // CHECK: arith.ori
  // CHECK: scf.yield
  triton_xla.atomic_spin_wait sys, acquire, %ptr, less_than, %expected : (tensor<4x!tt.ptr<i32>>, i32) -> ()
  tt.return
}

// CHECK-LABEL: @test_atomic_spin_wait_vectorized_with_mask
tt.func @test_atomic_spin_wait_vectorized_with_mask(%ptr: tensor<4x!tt.ptr<i32>>, %expected: i32, %mask: tensor<4xi1>) {
  // CHECK-NOT: triton_xla.atomic_spin_wait
  // CHECK: arith.constant dense<false> : tensor<4xi1>
  // CHECK: arith.constant dense<0> : tensor<4xi32>
  // CHECK: tt.splat {{%.*}} : i32 -> tensor<4xi32>
  // CHECK: scf.while
  // CHECK: tt.atomic_rmw add, acquire, sys, {{%.*}}, {{%.*}}, %arg2
  // CHECK: arith.cmpi eq
  // CHECK: arith.select %arg2
  // CHECK: tt.reduce
  // CHECK: arith.ori
  // CHECK: scf.yield
  triton_xla.atomic_spin_wait sys, acquire, %ptr, equal_to, %expected, %mask : (tensor<4x!tt.ptr<i32>>, i32, tensor<4xi1>) -> ()
  tt.return
}
