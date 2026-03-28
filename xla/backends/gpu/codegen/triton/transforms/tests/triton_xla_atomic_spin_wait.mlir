// RUN: xla-opt %s --triton-xla-lower-extern-atomics | FileCheck %s

// Test lowering of AtomicSpinWaitOp to tt.extern_elementwise

// CHECK-LABEL: tt.func @nomask_kernel
// CHECK-SAME:    %[[ARG0:.*]]: !tt.ptr<i32>
// CHECK-SAME:    %[[ARG1:.*]]: i32
tt.func @nomask_kernel(%ptr : !tt.ptr<i32>, %expected : i32) {
// CHECK-NEXT:  %[[RES:.+]] = tt.extern_elementwise %[[ARG0]], %[[ARG1]]
// CHECK-SAME:  {libname = "", libpath = "", pure = false, symbol = "xla_atomic_spin_wait_relaxed_gpu_eq"}
// CHECK-SAME:  : (!tt.ptr<i32>, i32) -> i32
  triton_xla.atomic_spin_wait gpu, relaxed, %ptr, equal_to, %expected  : (!tt.ptr<i32>, i32) -> ()
  tt.return
}

// CHECK-LABEL: tt.func @masked_kernel
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4x!tt.ptr<i32>>
// CHECK-SAME:    %[[ARG1:.*]]: tensor<4xi1>
// CHECK-SAME:    %[[ARG2:.*]]: i32
tt.func @masked_kernel(
  %ptr: tensor<4x!tt.ptr<i32>>,
  %mask: tensor<4xi1>,
  %expected: i32
) {
// CHECK:         %[[RES:.+]] = tt.extern_elementwise %[[ARG0]], %[[ARG2]], %[[ARG1]]
// CHECK-SAME:    {libname = "", libpath = "", pure = false, symbol = "xla_atomic_spin_wait_acquire_gpu_lt"}
// CHECK-SAME:    : (tensor<4x!tt.ptr<i32>>, i32, tensor<4xi1>) -> tensor<4xi32>
  triton_xla.atomic_spin_wait gpu, acquire, %ptr, less_than, %expected, %mask
      : (tensor<4x!tt.ptr<i32>>, i32, tensor<4xi1>) -> ()
  tt.return
}

// CHECK-LABEL: tt.func @test_vectorized_lt
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4x!tt.ptr<i32>>
// CHECK-SAME:    %[[ARG1:.*]]: i32
tt.func @test_vectorized_lt(%ptr: tensor<4x!tt.ptr<i32>>, %expected: i32) {
  // CHECK-NOT: triton_xla.atomic_spin_wait
  // CHECK: tt.extern_elementwise %[[ARG0]], %[[ARG1]]
  // CHECK-SAME: symbol = "xla_atomic_spin_wait_acquire_system_lt"
  triton_xla.atomic_spin_wait sys, acquire, %ptr, less_than, %expected : (tensor<4x!tt.ptr<i32>>, i32) -> ()
  tt.return
}

// CHECK-LABEL: tt.func @test_vectorized_eq
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4x!tt.ptr<i32>>
// CHECK-SAME:    %[[ARG1:.*]]: i32
tt.func @test_vectorized_eq(%ptr: tensor<4x!tt.ptr<i32>>, %expected: i32) {
  // CHECK-NOT: triton_xla.atomic_spin_wait
  // CHECK: tt.extern_elementwise %[[ARG0]], %[[ARG1]]
  // CHECK-SAME: symbol = "xla_atomic_spin_wait_acquire_system_eq"
  triton_xla.atomic_spin_wait sys, acquire, %ptr, equal_to, %expected : (tensor<4x!tt.ptr<i32>>, i32) -> ()
  tt.return
}
