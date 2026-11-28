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
// RUN: xla-translate --print-sugar=false -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

// Regions run on the execution thread of their enclosing function.

// CHECK-LABEL: HloModule main
// CHECK:       %[[IF_TRUE:region_[0-9.]+]] ({{.*}}) -> f32[] {
// CHECK:         ROOT {{.*}} negate(
// CHECK-NEXT:  }, execution_thread="sparsecore"
// CHECK:       %[[IF_FALSE:region_[0-9.]+]] ({{.*}}) -> f32[] {
// CHECK-NEXT:    ROOT {{.*}} parameter(0)
// CHECK-NEXT:  }, execution_thread="sparsecore"
// CHECK:       %[[CASE_0:region_[0-9.]+]] ({{.*}}) -> f32[] {
// CHECK:         ROOT {{.*}} abs(
// CHECK-NEXT:  }, execution_thread="sparsecore"
// CHECK:       %[[CASE_1:region_[0-9.]+]] ({{.*}}) -> f32[] {
// CHECK-NEXT:    ROOT {{.*}} parameter(0)
// CHECK-NEXT:  }, execution_thread="sparsecore"
// CHECK:       %[[NESTED_IF_TRUE:region_[0-9.]+]] ({{.*}}) -> f32[] {
// CHECK:         ROOT {{.*}} exponential(
// CHECK-NEXT:  }, execution_thread="sparsecore"
// CHECK:       %[[NESTED_IF_FALSE:region_[0-9.]+]] ({{.*}}) -> f32[] {
// CHECK-NEXT:    ROOT {{.*}} parameter(0)
// CHECK-NEXT:  }, execution_thread="sparsecore"
// CHECK:       %[[BODY:region_[0-9.]+]] ({{.*}}) -> f32[] {
// CHECK:         ROOT {{.*}} conditional({{.*}}), true_computation=%[[NESTED_IF_TRUE]], false_computation=%[[NESTED_IF_FALSE]]
// CHECK-NEXT:  }, execution_thread="sparsecore"
// CHECK:       %[[COND:region_[0-9.]+]] ({{.*}}) -> pred[] {
// CHECK:         ROOT {{.*}} compare(
// CHECK-NEXT:  }, execution_thread="sparsecore"
// CHECK:       %[[REDUCER:region_[0-9.]+]] ({{.*}}) -> f32[] {
// CHECK:         ROOT {{.*}} add(
// CHECK-NEXT:  }, execution_thread="sparsecore"
// CHECK:       %sparsecore.{{[0-9]+}} (
// CHECK:         conditional({{.*}}), true_computation=%[[IF_TRUE]], false_computation=%[[IF_FALSE]]
// CHECK:         conditional({{.*}}), branch_computations={%[[CASE_0]], %[[CASE_1]]}
// CHECK:         while({{.*}}), condition=%[[COND]], body=%[[BODY]]
// CHECK:         reduce({{.*}}), dimensions={0}, to_apply=%[[REDUCER]]
// CHECK:       }, execution_thread="sparsecore"
// CHECK:       ENTRY
// CHECK-NOT:   execution_thread=
func.func private @sparsecore(%pred: tensor<i1>, %index: tensor<i32>, %arg: tensor<f32>, %vec: tensor<4xf32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) attributes {execution_thread = "sparsecore"} {
  %if = "stablehlo.if"(%pred) ({
    %0 = stablehlo.negate %arg : tensor<f32>
    stablehlo.return %0 : tensor<f32>
  }, {
    stablehlo.return %arg : tensor<f32>
  }) : (tensor<i1>) -> tensor<f32>
  %case = "stablehlo.case"(%index) ({
    %0 = stablehlo.abs %arg : tensor<f32>
    stablehlo.return %0 : tensor<f32>
  }, {
    stablehlo.return %arg : tensor<f32>
  }) : (tensor<i32>) -> tensor<f32>
  %while = stablehlo.while(%iter = %arg) : tensor<f32>
  cond {
    %0 = stablehlo.compare LT, %iter, %iter : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %0 : tensor<i1>
  } do {
    %0 = stablehlo.compare GT, %iter, %iter : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %1 = "stablehlo.if"(%0) ({
      %2 = stablehlo.exponential %iter : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }, {
      stablehlo.return %iter : tensor<f32>
    }) : (tensor<i1>) -> tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %reduce = stablehlo.reduce(%vec init: %init) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
  func.return %if, %case, %while, %reduce : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
}

func.func @main(%pred: tensor<i1>, %index: tensor<i32>, %arg: tensor<f32>, %vec: tensor<4xf32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
  %0:4 = func.call @sparsecore(%pred, %index, %arg, %vec) : (tensor<i1>, tensor<i32>, tensor<f32>, tensor<4xf32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>)
  func.return %0#0, %0#1, %0#2, %0#3 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
}

// -----

// CHECK-LABEL: HloModule main
// CHECK:       %[[FUSED:region_[0-9.]+]] ({{.*}}) -> f32[] {
// CHECK:         ROOT {{.*}} negate(
// CHECK-NEXT:  }, execution_thread="sparsecore"
// CHECK:       %sparsecore.{{[0-9]+}} (
// CHECK:         fusion({{.*}}), kind=kLoop, calls=%[[FUSED]]
// CHECK-NEXT:  }, execution_thread="sparsecore"
// CHECK:       ENTRY
// CHECK-NOT:   execution_thread=
func.func private @sparsecore(%arg: tensor<f32>) -> tensor<f32> attributes {execution_thread = "sparsecore"} {
  %0 = "mhlo.fusion"(%arg) ({
  ^bb0(%arg1: tensor<f32>):
    %1 = mhlo.negate %arg1 : tensor<f32>
    mhlo.return %1 : tensor<f32>
  }) {fusion_kind = #mhlo<fusion_kind kLoop>} : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

func.func @main(%arg: tensor<f32>) -> tensor<f32> {
  %0 = func.call @sparsecore(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// Composite decompositions run on the decomposition function's thread.

// CHECK-LABEL: HloModule main
// CHECK-NOT:   {{^}$}}
// CHECK:       %sparsecore.{{[0-9]+}} (
// CHECK:         call({{.*}}), to_apply=%decomposition.{{[0-9]+}}, is_composite=true
// CHECK-NEXT:  }, execution_thread="sparsecore"
// CHECK:       ENTRY
// CHECK-NOT:   execution_thread=
func.func private @decomposition(%arg: tensor<f32>) -> tensor<f32> attributes {execution_thread = "sparsecore"} {
  %0 = stablehlo.negate %arg : tensor<f32>
  func.return %0 : tensor<f32>
}

func.func private @sparsecore(%arg: tensor<f32>) -> tensor<f32> attributes {execution_thread = "sparsecore"} {
  %0 = stablehlo.composite "foo.bar" %arg {decomposition = @decomposition} : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

func.func @main(%arg: tensor<f32>) -> tensor<f32> {
  %0 = func.call @sparsecore(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// An async-start runs on the same execution thread as the computation it wraps.

// CHECK-LABEL: HloModule main
// CHECK:       %[[ASYNC_BODY:region_[0-9.]+]] ({{.*}}) -> f32[4,2] {
// CHECK:         ROOT {{.*}} all-to-all(
// CHECK-NEXT:  }, execution_thread="sparsecore"
// CHECK:       %sparsecore.{{[0-9]+}} (
// CHECK:         async-start({{.*}}), async_execution_thread="sparsecore", calls=%[[ASYNC_BODY]]
// CHECK:       }, execution_thread="sparsecore"
// CHECK:       ENTRY
// CHECK-NOT:   execution_thread=
func.func private @sparsecore(%arg: tensor<4x2xf32>) -> tensor<4x2xf32> attributes {execution_thread = "sparsecore"} {
  %0 = "stablehlo.async_start"(%arg) ({
  ^bb0(%arg1: tensor<4x2xf32>):
    %1 = "stablehlo.all_to_all"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, concat_dimension = 1 : i64, replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>, split_count = 2 : i64, split_dimension = 1 : i64}> : (tensor<4x2xf32>) -> tensor<4x2xf32>
    stablehlo.return %1 : tensor<4x2xf32>
  }) : (tensor<4x2xf32>) -> !stablehlo.future<tensor<4x2xf32>>
  %2 = "stablehlo.async_done"(%0) : (!stablehlo.future<tensor<4x2xf32>>) -> tensor<4x2xf32>
  func.return %2 : tensor<4x2xf32>
}

func.func @main(%arg: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %0 = func.call @sparsecore(%arg) : (tensor<4x2xf32>) -> tensor<4x2xf32>
  func.return %0 : tensor<4x2xf32>
}

// -----

// An all-reduce-start's reducer runs on the caller's thread, not the callee's.

// CHECK-LABEL: HloModule main
// CHECK:       %all_reduce.{{[0-9]+}} (
// CHECK:       }, execution_thread="sparsecore"
// CHECK:       %[[REDUCER:region_[0-9.]+]] ({{.*}}) -> f32[] {
// CHECK:         ROOT {{.*}} maximum(
// CHECK-NEXT:  {{^}$}}
// CHECK:       ENTRY
// CHECK:         all-reduce-start({{.*}}), {{.*}}to_apply=%[[REDUCER]]
func.func private @all_reduce(%arg: tensor<10xf32>) -> tensor<10xf32> attributes {execution_thread = "sparsecore"} {
  %0 = "mhlo.all_reduce"(%arg) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %1 = mhlo.maximum %lhs, %rhs : tensor<f32>
    mhlo.return %1 : tensor<f32>
  }) {channel_handle = #mhlo.channel_handle<handle = 5, type = 2>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

func.func @main(%arg: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "mhlo.async_start"(%arg) {called_computation = @all_reduce, execution_thread = "sparsecore"} : (tensor<10xf32>) -> !mhlo.async_bundle<tensor<10xf32>, tensor<10xf32>>
  %1 = "mhlo.async_done"(%0) : (!mhlo.async_bundle<tensor<10xf32>, tensor<10xf32>>) -> tensor<10xf32>
  func.return %1 : tensor<10xf32>
}

// -----

// CHECK-LABEL: HloModule main
// CHECK:       %all_reduce.{{[0-9]+}} (
// CHECK:       {{^}$}}
// CHECK:       %[[REDUCER:region_[0-9.]+]] ({{.*}}) -> f32[] {
// CHECK:         ROOT {{.*}} maximum(
// CHECK-NEXT:  }, execution_thread="sparsecore"
// CHECK:       %sparsecore.{{[0-9]+}} (
// CHECK:         all-reduce-start({{.*}}), {{.*}}to_apply=%[[REDUCER]]
// CHECK:       }, execution_thread="sparsecore"
func.func private @all_reduce(%arg: tensor<10xf32>) -> tensor<10xf32> attributes {execution_thread = "main"} {
  %0 = "mhlo.all_reduce"(%arg) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %1 = mhlo.maximum %lhs, %rhs : tensor<f32>
    mhlo.return %1 : tensor<f32>
  }) {channel_handle = #mhlo.channel_handle<handle = 5, type = 2>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>} : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

func.func private @sparsecore(%arg: tensor<10xf32>) -> tensor<10xf32> attributes {execution_thread = "sparsecore"} {
  %0 = "mhlo.async_start"(%arg) {called_computation = @all_reduce, execution_thread = "main"} : (tensor<10xf32>) -> !mhlo.async_bundle<tensor<10xf32>, tensor<10xf32>>
  %1 = "mhlo.async_done"(%0) : (!mhlo.async_bundle<tensor<10xf32>, tensor<10xf32>>) -> tensor<10xf32>
  func.return %1 : tensor<10xf32>
}

func.func @main(%arg: tensor<10xf32>) -> tensor<10xf32> {
  %0 = func.call @sparsecore(%arg) : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

// Regions of main-thread functions stay on the main thread.

// CHECK-LABEL: HloModule main
// CHECK-NOT:   execution_thread=
// CHECK:       conditional(
// CHECK-NOT:   execution_thread=
func.func @main(%pred: tensor<i1>, %arg: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.if"(%pred) ({
    %1 = stablehlo.negate %arg : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }, {
    stablehlo.return %arg : tensor<f32>
  }) : (tensor<i1>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
