// RUN: mlir-hlo-opt %s --split-input-file --hlo-canonicalize-async | FileCheck %s

func.func private @send(%arg0: tensor<i32>, %arg1: !mhlo.token) -> !mhlo.token attributes {execution_thread = "main"} {
  %0 = "mhlo.send"(%arg0, %arg1) {channel_handle = #mhlo.channel_handle<handle = 3, type = 2>} : (tensor<i32>, !mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

// CHECK-LABEL: func @test_send
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<i32>, %[[ARG1:.*]]: !mhlo.token)
func.func @test_send(%arg0: tensor<i32>, %arg1: !mhlo.token) -> !mhlo.token {
  // CHECK-NOT: mhlo.async_start
  // CHECK-NOT: mhlo.async_done
  // CHECK: %[[RESULT:.*]] = call @send(%[[ARG0]], %[[ARG1]])
  %0 = "mhlo.async_start"(%arg0, %arg1) {called_computation = @send, execution_thread = "main"} : (tensor<i32>, !mhlo.token) -> !mhlo.async_bundle<tuple<tensor<i32>, !mhlo.token>, !mhlo.token, tensor<ui32>>
  %1 = "mhlo.async_done"(%0) {called_computation = @send, execution_thread = "main"} : (!mhlo.async_bundle<tuple<tensor<i32>, !mhlo.token>, !mhlo.token, tensor<ui32>>) -> !mhlo.token
  // CHECK: return %[[RESULT]]
  func.return %1 : !mhlo.token
}

// -----

func.func private @recv(%token: !mhlo.token) -> (tensor<i32>, !mhlo.token) attributes {execution_thread = "main"} {
  %0:2 = "mhlo.recv"(%token) {channel_handle = #mhlo.channel_handle<handle = 5, type = 3>} : (!mhlo.token) -> (tensor<i32>, !mhlo.token)
  func.return %0#0, %0#1 : tensor<i32>, !mhlo.token
}

// CHECK-LABEL: func @test_recv
// CHECK-SAME:    (%[[ARG0:.*]]: !mhlo.token)
func.func @test_recv(%token: !mhlo.token) -> (tensor<i32>, !mhlo.token) {
  // CHECK-NOT: mhlo.async_start
  // CHECK-NOT: mhlo.async_done
  // CHECK: %[[RESULT:.*]]:2 = call @recv(%[[ARG0]])
  %0 = "mhlo.async_start"(%token) {called_computation = @recv, execution_thread = "main"} : (!mhlo.token) -> !mhlo.async_bundle<!mhlo.token, tuple<tensor<i32>, !mhlo.token>, tensor<i32>>
  %1:2 = "mhlo.async_done"(%0) {called_computation = @recv, execution_thread = "main"} : (!mhlo.async_bundle<!mhlo.token, tuple<tensor<i32>, !mhlo.token>, tensor<i32>>) -> (tensor<i32>, !mhlo.token)
  // CHECK: return %[[RESULT]]#0, %[[RESULT]]#1
  return %1#0, %1#1 : tensor<i32>, !mhlo.token
}

// -----

func.func private @send_with_barrier(%arg0: tensor<i32>, %arg1: !mhlo.token) -> !mhlo.token attributes {execution_thread = "main"} {
  %0 = "mhlo.send"(%arg0, %arg1) {channel_handle = #mhlo.channel_handle<handle = 3, type = 2>} : (tensor<i32>, !mhlo.token) -> !mhlo.token
  %1 = "mhlo.optimization_barrier"(%0) : (!mhlo.token) -> !mhlo.token
  func.return %1 : !mhlo.token
}

// CHECK-LABEL: func @test_ineligible_ops
func.func @test_ineligible_ops(%arg0: tensor<i32>, %arg1: !mhlo.token) -> !mhlo.token {
  // CHECK: mhlo.async_start
  // CHECK: mhlo.async_done
  %0 = "mhlo.async_start"(%arg0, %arg1) {called_computation = @send_with_barrier, execution_thread = "main"} : (tensor<i32>, !mhlo.token) -> !mhlo.async_bundle<tuple<tensor<i32>, !mhlo.token>, !mhlo.token, tensor<ui32>>
  %1 = "mhlo.async_done"(%0) {called_computation = @send_with_barrier, execution_thread = "main"} : (!mhlo.async_bundle<tuple<tensor<i32>, !mhlo.token>, !mhlo.token, tensor<ui32>>) -> !mhlo.token
  func.return %1 : !mhlo.token
}

// -----

func.func private @send(%arg0: tensor<i32>, %arg1: !mhlo.token) -> !mhlo.token attributes {execution_thread = "main"} {
  %0 = "mhlo.send"(%arg0, %arg1) {channel_handle = #mhlo.channel_handle<handle = 3, type = 2>} : (tensor<i32>, !mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

// CHECK-LABEL: func @test_has_async_update
func.func @test_has_async_update(%arg0: tensor<i32>, %arg1: !mhlo.token) -> !mhlo.token {
  // CHECK: mhlo.async_start
  // CHECK: mhlo.async_update
  // CHECK: mhlo.async_done
  %0 = "mhlo.async_start"(%arg0, %arg1) {called_computation = @send, execution_thread = "main"} : (tensor<i32>, !mhlo.token) -> !mhlo.async_bundle<tuple<tensor<i32>, !mhlo.token>, !mhlo.token, tensor<ui32>>
  %1 = "mhlo.async_update"(%0) {called_computation = @send, execution_thread = "main"} : (!mhlo.async_bundle<tuple<tensor<i32>, !mhlo.token>, !mhlo.token, tensor<ui32>>) -> !mhlo.async_bundle<tuple<tensor<i32>, !mhlo.token>, !mhlo.token, tensor<ui32>>
  %2 = "mhlo.async_done"(%1) {called_computation = @send, execution_thread = "main"} : (!mhlo.async_bundle<tuple<tensor<i32>, !mhlo.token>, !mhlo.token, tensor<ui32>>) -> !mhlo.token
  func.return %2 : !mhlo.token
}

