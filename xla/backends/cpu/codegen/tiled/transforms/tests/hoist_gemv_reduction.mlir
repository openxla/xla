// RUN: fusion_compiler_opt %s -xtile-cpu-hoist-gemv-reduction -split-input-file | FileCheck %s

#map_lhs = affine_map<(d0, d1) -> (d1, d0)>
#map_rhs = affine_map<(d0, d1) -> (d0)>
#map_acc = affine_map<(d0, d1) -> (d1)>

func.func @hoist_k_minor_gemv_f32_with_shape_casts(
    %mat: vector<8x8xf32>, %vec: vector<8xf32>,
    %lb: index, %ub: index, %step: index) -> vector<8x1xf32> {
  %zero = arith.constant dense<0.0> : vector<8x1xf32>
  %res = scf.for %k = %lb to %ub step %step iter_args(%acc = %zero) -> (vector<8x1xf32>) {
    %acc_1d = vector.shape_cast %acc : vector<8x1xf32> to vector<8xf32>
    %0 = vector.contract {
        indexing_maps = [#map_lhs, #map_rhs, #map_acc],
        iterator_types = ["reduction", "parallel"],
        kind = #vector.kind<add>}
      %mat, %vec, %acc_1d : vector<8x8xf32>, vector<8xf32> into vector<8xf32>
    %out = vector.shape_cast %0 : vector<8xf32> to vector<8x1xf32>
    scf.yield %out : vector<8x1xf32>
  }
  return %res : vector<8x1xf32>
}
// CHECK-LABEL: @hoist_k_minor_gemv_f32_with_shape_casts(
// CHECK-SAME: %[[MAT:.*]]: vector<8x8xf32>, %[[VEC:.*]]: vector<8xf32>,
// CHECK-DAG: %[[ZERO_1D:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK-DAG: %[[ZERO_2D:.*]] = arith.constant dense<0.000000e+00> : vector<8x8xf32>
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[ZERO_2D]]) -> (vector<8x8xf32>) {
// CHECK-NOT: vector.contract
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[VEC]] : vector<8xf32> to vector<8x8xf32>
// CHECK: %[[FMA:.*]] = vector.fma %[[MAT]], %[[BCAST]], %[[ACC]] : vector<8x8xf32>
// CHECK: scf.yield %[[FMA]] : vector<8x8xf32>
// CHECK: }
// CHECK: %[[RED:.*]] = vector.multi_reduction <add>, %[[LOOP]], %[[ZERO_1D]] [1] : vector<8x8xf32> to vector<8xf32>
// CHECK: %[[OUT:.*]] = vector.shape_cast %[[RED]] : vector<8xf32> to vector<8x1xf32>
// CHECK: return %[[OUT]] : vector<8x1xf32>

// -----

#map_rhs_mat_lhs = affine_map<(d0, d1) -> (d0)>
#map_rhs_mat_rhs = affine_map<(d0, d1) -> (d1, d0)>
#map_rhs_mat_acc = affine_map<(d0, d1) -> (d1)>

func.func @hoist_k_minor_gemv_rhs_matrix(
    %vec: vector<8xf32>, %mat: vector<4x8xf32>,
    %lb: index, %ub: index, %step: index) -> vector<1x4xf32> {
  %zero = arith.constant dense<0.0> : vector<1x4xf32>
  %res = scf.for %k = %lb to %ub step %step iter_args(%acc = %zero) -> (vector<1x4xf32>) {
    %acc_1d = vector.shape_cast %acc : vector<1x4xf32> to vector<4xf32>
    %0 = vector.contract {
        indexing_maps = [#map_rhs_mat_lhs, #map_rhs_mat_rhs, #map_rhs_mat_acc],
        iterator_types = ["reduction", "parallel"],
        kind = #vector.kind<add>}
      %vec, %mat, %acc_1d : vector<8xf32>, vector<4x8xf32> into vector<4xf32>
    %out = vector.shape_cast %0 : vector<4xf32> to vector<1x4xf32>
    scf.yield %out : vector<1x4xf32>
  }
  return %res : vector<1x4xf32>
}
// CHECK-LABEL: @hoist_k_minor_gemv_rhs_matrix(
// CHECK-SAME: %[[VEC:.*]]: vector<8xf32>, %[[MAT:.*]]: vector<4x8xf32>,
// CHECK-DAG: %[[ZERO_1D:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG: %[[ZERO_2D:.*]] = arith.constant dense<0.000000e+00> : vector<4x8xf32>
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[ZERO_2D]]) -> (vector<4x8xf32>) {
// CHECK-NOT: vector.contract
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[VEC]] : vector<8xf32> to vector<4x8xf32>
// CHECK: %[[FMA:.*]] = vector.fma %[[MAT]], %[[BCAST]], %[[ACC]] : vector<4x8xf32>
// CHECK: scf.yield %[[FMA]] : vector<4x8xf32>
// CHECK: }
// CHECK: %[[RED:.*]] = vector.multi_reduction <add>, %[[LOOP]], %[[ZERO_1D]] [1] : vector<4x8xf32> to vector<4xf32>
// CHECK: %[[OUT:.*]] = vector.shape_cast %[[RED]] : vector<4xf32> to vector<1x4xf32>
// CHECK: return %[[OUT]] : vector<1x4xf32>

// -----

#map_i32_lhs = affine_map<(d0, d1) -> (d1, d0)>
#map_i32_rhs = affine_map<(d0, d1) -> (d0)>
#map_i32_acc = affine_map<(d0, d1) -> (d1)>

func.func @hoist_k_minor_gemv_i32_nonzero_init(
    %mat: vector<4x4xi32>, %vec: vector<4xi32>, %init: vector<4x1xi32>,
    %lb: index, %ub: index, %step: index) -> vector<4x1xi32> {
  %res = scf.for %k = %lb to %ub step %step iter_args(%acc = %init) -> (vector<4x1xi32>) {
    %acc_1d = vector.shape_cast %acc : vector<4x1xi32> to vector<4xi32>
    %0 = vector.contract {
        indexing_maps = [#map_i32_lhs, #map_i32_rhs, #map_i32_acc],
        iterator_types = ["reduction", "parallel"],
        kind = #vector.kind<add>}
      %mat, %vec, %acc_1d : vector<4x4xi32>, vector<4xi32> into vector<4xi32>
    %out = vector.shape_cast %0 : vector<4xi32> to vector<4x1xi32>
    scf.yield %out : vector<4x1xi32>
  }
  return %res : vector<4x1xi32>
}
// CHECK-LABEL: @hoist_k_minor_gemv_i32_nonzero_init(
// CHECK-SAME: %[[MAT:.*]]: vector<4x4xi32>, %[[VEC:.*]]: vector<4xi32>,
// CHECK-SAME: %[[INIT:.*]]: vector<4x1xi32>,
// CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0> : vector<4x4xi32>
// CHECK-DAG: %[[INIT_1D:.*]] = vector.shape_cast %[[INIT]] : vector<4x1xi32> to vector<4xi32>
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[ZERO]]) -> (vector<4x4xi32>) {
// CHECK-NOT: vector.contract
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[VEC]] : vector<4xi32> to vector<4x4xi32>
// CHECK: %[[PROD:.*]] = arith.muli %[[MAT]], %[[BCAST]] : vector<4x4xi32>
// CHECK: %[[ADD:.*]] = arith.addi %[[PROD]], %[[ACC]] : vector<4x4xi32>
// CHECK: scf.yield %[[ADD]] : vector<4x4xi32>
// CHECK: }
// CHECK: %[[RED:.*]] = vector.multi_reduction <add>, %[[LOOP]], %[[INIT_1D]] [1] : vector<4x4xi32> to vector<4xi32>
// CHECK: %[[OUT:.*]] = vector.shape_cast %[[RED]] : vector<4xi32> to vector<4x1xi32>
// CHECK: return %[[OUT]] : vector<4x1xi32>

// -----

#map_kmajor_lhs = affine_map<(d0, d1) -> (d0, d1)>
#map_kmajor_rhs = affine_map<(d0, d1) -> (d0)>
#map_kmajor_acc = affine_map<(d0, d1) -> (d1)>

func.func @do_not_hoist_k_major_gemv(
    %mat: vector<8x8xf32>, %vec: vector<8xf32>,
    %lb: index, %ub: index, %step: index) -> vector<8xf32> {
  %zero = arith.constant dense<0.0> : vector<8xf32>
  %res = scf.for %k = %lb to %ub step %step iter_args(%acc = %zero) -> (vector<8xf32>) {
    %0 = vector.contract {
        indexing_maps = [#map_kmajor_lhs, #map_kmajor_rhs, #map_kmajor_acc],
        iterator_types = ["reduction", "parallel"],
        kind = #vector.kind<add>}
      %mat, %vec, %acc : vector<8x8xf32>, vector<8xf32> into vector<8xf32>
    scf.yield %0 : vector<8xf32>
  }
  return %res : vector<8xf32>
}
// CHECK-LABEL: @do_not_hoist_k_major_gemv
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (vector<8xf32>)
// CHECK: vector.contract
// CHECK: scf.yield %{{.*}} : vector<8xf32>

// -----

#map_multi_lhs = affine_map<(d0, d1) -> (d1, d0)>
#map_multi_rhs = affine_map<(d0, d1) -> (d0)>
#map_multi_acc = affine_map<(d0, d1) -> (d1)>

func.func @hoist_multiple_iter_args(
    %mat: vector<4x8xf32>, %vec: vector<8xf32>, %extra_init: f32,
    %lb: index, %ub: index, %step: index) -> (f32, vector<4xf32>) {
  %zero = arith.constant dense<0.0> : vector<4xf32>
  %res:2 = scf.for %k = %lb to %ub step %step
      iter_args(%extra = %extra_init, %acc = %zero) -> (f32, vector<4xf32>) {
    %next_extra = arith.addf %extra, %extra : f32
    %0 = vector.contract {
        indexing_maps = [#map_multi_lhs, #map_multi_rhs, #map_multi_acc],
        iterator_types = ["reduction", "parallel"],
        kind = #vector.kind<add>}
      %mat, %vec, %acc : vector<4x8xf32>, vector<8xf32> into vector<4xf32>
    scf.yield %next_extra, %0 : f32, vector<4xf32>
  }
  return %res#0, %res#1 : f32, vector<4xf32>
}
// CHECK-LABEL: @hoist_multiple_iter_args(
// CHECK-SAME: %[[MAT:.*]]: vector<4x8xf32>, %[[VEC:.*]]: vector<8xf32>,
// CHECK-SAME: %[[EXTRA_INIT:.*]]: f32,
// CHECK-DAG: %[[ZERO_1D:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG: %[[ZERO_2D:.*]] = arith.constant dense<0.000000e+00> : vector<4x8xf32>
// CHECK: %[[LOOP:.*]]:2 = scf.for {{.*}} iter_args(%[[EXTRA:.*]] = %[[EXTRA_INIT]], %[[ACC:.*]] = %[[ZERO_2D]]) -> (f32, vector<4x8xf32>) {
// CHECK: %[[NEXT_EXTRA:.*]] = arith.addf %[[EXTRA]], %[[EXTRA]] : f32
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[VEC]] : vector<8xf32> to vector<4x8xf32>
// CHECK: %[[FMA:.*]] = vector.fma %[[MAT]], %[[BCAST]], %[[ACC]] : vector<4x8xf32>
// CHECK: scf.yield %[[NEXT_EXTRA]], %[[FMA]] : f32, vector<4x8xf32>
// CHECK: }
// CHECK: %[[RED:.*]] = vector.multi_reduction <add>, %[[LOOP]]#1, %[[ZERO_1D]] [1] : vector<4x8xf32> to vector<4xf32>
// CHECK: return %[[LOOP]]#0, %[[RED]] : f32, vector<4xf32>
