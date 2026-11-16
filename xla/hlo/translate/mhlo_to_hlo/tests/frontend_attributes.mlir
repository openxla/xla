// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

// Tests parameters with frontend_attributes have such attributes set correctly
// in HloModule

func.func @main(%arg0: tensor<3x4xf32> {mhlo.frontend_attributes = {_test = "a"}}, %arg1: tensor<3x4xf32>) -> tuple<tensor<3x4xf32>, tensor<3x4xf32>> {
  %0 = "mhlo.tuple"(%arg0, %arg1) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tuple<tensor<3x4xf32>, tensor<3x4xf32>>
  func.return %0 : tuple<tensor<3x4xf32>, tensor<3x4xf32>>
}

// CHECK:  ENTRY
// CHECK:  %[[P0:.*]] = f32[3,4] parameter(0), frontend_attributes={_test="a"}
// CHECK:  %[[P1:.*]] = f32[3,4] parameter(1)
// CHECK-NOT: frontend_attributes=

// -----

// Tests call ops with frontend_attributes have such attributes set correctly
// in HloModule

func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = call @add(%arg0, %arg0) {mhlo.frontend_attributes = {_xla_compute_type = "dense"}} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
func.func private @add(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = mhlo.add %arg0, %arg1 : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK:  ENTRY
// CHECK:       %[[CALL:.*]] = f32[] call
// CHECK-SAME:  frontend_attributes={_xla_compute_type="dense"}

// -----

// Tests that the get-tuple-elements unpacking a multi-result call do not
// inherit the call's frontend_attributes.

func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  %0:2 = call @two(%arg0) {mhlo.frontend_attributes = {_xla_stream_annotation = "collective"}} : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  %1 = mhlo.add %0#0, %0#1 : tensor<f32>
  func.return %1 : tensor<f32>
}
func.func private @two(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0 = mhlo.negate %arg0 : tensor<f32>
  return %arg0, %0 : tensor<f32>, tensor<f32>
}

// CHECK:  ENTRY
// CHECK:       %[[CALL:.*]] = (f32[], f32[]) call
// CHECK-SAME:  frontend_attributes={_xla_stream_annotation="collective"}
// CHECK:       get-tuple-element(%[[CALL]]), index=0
// CHECK-NOT:   frontend_attributes={
// CHECK:       get-tuple-element(%[[CALL]]), index=1
// CHECK-NOT:   frontend_attributes={
// CHECK:       ROOT

// -----

// Tests that the get-tuple-elements unpacking a multi-result op other than a
// call do not inherit its frontend_attributes either.

func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0:2 = "mhlo.all_reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %2 = mhlo.add %arg2, %arg3 : tensor<f32>
    mhlo.return %2 : tensor<f32>
  }) {replica_groups = dense<> : tensor<0x0xi64>, mhlo.frontend_attributes = {_xla_stream_annotation = "collective"}} : (tensor<8xf32>, tensor<f32>) -> (tensor<8xf32>, tensor<f32>)
  %1 = mhlo.add %0#1, %0#1 : tensor<f32>
  func.return %1 : tensor<f32>
}

// CHECK:  ENTRY
// CHECK:       %[[ALL_REDUCE:.*]] = (f32[8], f32[]) all-reduce
// CHECK-SAME:  frontend_attributes={_xla_stream_annotation="collective"}
// CHECK:       get-tuple-element(%[[ALL_REDUCE]]), index=0
// CHECK-NOT:   frontend_attributes={
// CHECK:       get-tuple-element(%[[ALL_REDUCE]]), index=1
// CHECK-NOT:   frontend_attributes={
// CHECK:       ROOT

// -----

// Tests that the get-tuple-elements unpacking an infeed do not inherit its
// frontend_attributes.

func.func @main(%arg0: !mhlo.token) -> tensor<3x3xi32> {
  %0:2 = "mhlo.infeed"(%arg0) <{infeed_config = "foobar", layout = [[0, 1]]}> {mhlo.frontend_attributes = {_xla_stream_annotation = "collective"}} : (!mhlo.token) -> (tensor<3x3xi32>, !mhlo.token)
  func.return %0#0 : tensor<3x3xi32>
}

// CHECK:  ENTRY
// CHECK:       %[[INFEED:.*]] = ((s32[3,3]), token[]) infeed
// CHECK-SAME:  frontend_attributes={_xla_stream_annotation="collective"}
// CHECK:       get-tuple-element(%[[INFEED]]), index=0
// CHECK-NOT:   frontend_attributes={
// CHECK:       get-tuple-element({{.*}}), index=0
// CHECK-NOT:   frontend_attributes={
// CHECK:       get-tuple-element(%[[INFEED]]), index=1
// CHECK-NOT:   frontend_attributes={

// -----

// Tests that the get-tuple-elements unpacking a recv-done do not inherit the
// recv's frontend_attributes.

func.func @main(%arg0: !mhlo.token) -> tensor<3x4xi32> {
  %0:2 = "mhlo.recv"(%arg0) {
    channel_handle = #mhlo.channel_handle<handle = 5, type = 3>,
    is_host_transfer = true,
    mhlo.frontend_attributes = {_xla_stream_annotation = "collective"}
  } : (!mhlo.token) -> (tensor<3x4xi32>, !mhlo.token)
  func.return %0#0 : tensor<3x4xi32>
}

// CHECK:  ENTRY
// CHECK:       %[[RECV_DONE:.*]] = (s32[3,4], token[]) recv-done
// CHECK:       get-tuple-element(%[[RECV_DONE]]), index=0
// CHECK-NOT:   frontend_attributes={
// CHECK:       get-tuple-element(%[[RECV_DONE]]), index=1
// CHECK-NOT:   frontend_attributes={
