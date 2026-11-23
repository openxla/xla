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
// inherit any of the call's frontend_attributes: neither the call-boundary
// stream annotation nor the other attributes.

func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0:2 = call @two_results(%arg0) {mhlo.frontend_attributes = {_xla_stream_annotation = "locality_domain:0", inlineable = "false", _xla_compute_type = "dense"}} : (tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  func.return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
}
func.func private @two_results(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
  %1 = mhlo.multiply %arg0, %arg0 : tensor<4xf32>
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK:      ENTRY
// CHECK:      %[[CALL:.*]] = (f32[4], f32[4]) call
// CHECK-SAME: frontend_attributes={_xla_compute_type="dense",_xla_stream_annotation="locality_domain:0",inlineable="false"}
// CHECK:      get-tuple-element(%[[CALL]]), index=0
// CHECK-NOT:  frontend_attributes={
// CHECK:      get-tuple-element(%[[CALL]]), index=1
// CHECK-NOT:  frontend_attributes={
// CHECK:      ROOT

// -----

// Tests that a regular stream annotation is also kept only on the call, along
// with the call's other frontend_attributes.

func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0:2 = call @two_results(%arg0) {mhlo.frontend_attributes = {_xla_stream_annotation = "1", inlineable = "false"}} : (tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  func.return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
}
func.func private @two_results(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
  %1 = mhlo.multiply %arg0, %arg0 : tensor<4xf32>
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK:      ENTRY
// CHECK:      %[[CALL:.*]] = (f32[4], f32[4]) call
// CHECK-SAME: frontend_attributes={_xla_stream_annotation="1",inlineable="false"}
// CHECK:      get-tuple-element(%[[CALL]]), index=0
// CHECK-NOT:  frontend_attributes={
// CHECK:      get-tuple-element(%[[CALL]]), index=1
// CHECK-NOT:  frontend_attributes={
// CHECK:      ROOT

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
// CHECK-NOT:   frontend_attributes={
// CHECK:       %[[ALL_REDUCE:.*]] = (f32[8], f32[]) all-reduce
// CHECK-SAME:  frontend_attributes={_xla_stream_annotation="collective"}
// CHECK:       get-tuple-element(%[[ALL_REDUCE]]), index=0
// CHECK-NOT:   frontend_attributes={
// CHECK:       get-tuple-element(%[[ALL_REDUCE]]), index=1
// CHECK-NOT:   frontend_attributes={
// CHECK:       ROOT

// -----

// Tests that the tuple and get-tuple-elements packing the operands of a
// multi-operand all-gather, and the get-tuple-elements unpacking its results,
// do not inherit its frontend_attributes.

func.func @main(%arg0: tensor<8x2xf32>, %arg1: tensor<8x4xf32>) -> tensor<8x16xf32> {
  %0:2 = "mhlo.all_gather"(%arg0, %arg1) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    use_global_device_ids,
    mhlo.frontend_attributes = {_xla_stream_annotation = "collective"}
  } : (tensor<8x2xf32>, tensor<8x4xf32>) -> (tensor<8x8xf32>, tensor<8x16xf32>)
  %1 = mhlo.add %0#1, %0#1 : tensor<8x16xf32>
  func.return %1 : tensor<8x16xf32>
}

// CHECK:  ENTRY
// CHECK-NOT:   frontend_attributes={
// CHECK:       %[[ALL_GATHER:.*]] = (f32[8,8], f32[8,16]) all-gather
// CHECK-SAME:  frontend_attributes={_xla_stream_annotation="collective"}
// CHECK:       get-tuple-element(%[[ALL_GATHER]]), index=0
// CHECK-NOT:   frontend_attributes={
// CHECK:       get-tuple-element(%[[ALL_GATHER]]), index=1
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
// CHECK:       %[[RECV:.*]] = (s32[3,4], u32[], token[]) recv
// CHECK-SAME:  frontend_attributes={{.*}}_xla_stream_annotation="collective"
// CHECK:       %[[RECV_DONE:.*]] = (s32[3,4], token[]) recv-done(%[[RECV]])
// CHECK-SAME:  frontend_attributes={{.*}}_xla_stream_annotation="collective"
// CHECK:       get-tuple-element(%[[RECV_DONE]]), index=0
// CHECK-NOT:   frontend_attributes={
// CHECK:       get-tuple-element(%[[RECV_DONE]]), index=1
// CHECK-NOT:   frontend_attributes={

// -----

// Tests that the tuple packing the operands of a multi-operand op, and the
// get-tuple-elements unpacking its results, do not inherit its
// frontend_attributes.

func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0:2 = "mhlo.optimization_barrier"(%arg0, %arg1) {mhlo.frontend_attributes = {_xla_stream_annotation = "collective"}} : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  %1 = mhlo.add %0#0, %0#1 : tensor<f32>
  func.return %1 : tensor<f32>
}

// CHECK:  ENTRY
// CHECK:       %[[TUPLE:.*]] = (f32[], f32[]) tuple
// CHECK-NOT:   frontend_attributes={
// CHECK:       %[[BARRIER:.*]] = (f32[], f32[]) opt-barrier(%[[TUPLE]])
// CHECK-SAME:  frontend_attributes={_xla_stream_annotation="collective"}
// CHECK:       get-tuple-element(%[[BARRIER]]), index=0
// CHECK-NOT:   frontend_attributes={
// CHECK:       get-tuple-element(%[[BARRIER]]), index=1
// CHECK-NOT:   frontend_attributes={
// CHECK:       ROOT
