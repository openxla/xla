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
// RUN: emitters_opt %s -split-input-file \
// RUN: -xla-gpu-lower-tensors="gpu_compute_capability='cuda_compute_capability {major: 9}'" \
// RUN: | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @cluster_dsmem(%idx: index, %val: f32) -> f32 {
    %shared = xla_gpu.allocate_shared : tensor<1024xf32>
    %inserted = tensor.insert %val into %shared[%idx] {xla_gpu.dsmem} : tensor<1024xf32>
    %synced = xla_gpu.sync_threads scope <cluster> %inserted : tensor<1024xf32>
    %extracted = tensor.extract %synced[%idx] {xla_gpu.dsmem} : tensor<1024xf32>
    return %extracted : f32
  }
}
// CHECK-LABEL: @cluster_dsmem(
// CHECK-SAME:      %[[IDX:.*]]: index, %[[VAL:.*]]: f32) -> f32 {
// CHECK-DAG:     %[[ADDR:.*]] = llvm.mlir.addressof @shared_{{.*}} : !llvm.ptr<3>
// CHECK-DAG:     %[[IDX_INT:.*]] = arith.index_castui %[[IDX]] : index to i32
// CHECK-DAG:     %[[C1024:.*]] = arith.constant 1024 : i32
// CHECK:         %[[CTA:.*]] = arith.divui %[[IDX_INT]], %[[C1024]] : i32
// CHECK:         %[[LOCAL_IDX:.*]] = arith.remui %[[IDX_INT]], %[[C1024]] : i32
// CHECK:         %[[CLUSTER_STORE:.*]] = nvvm.mapa %[[ADDR]], %[[CTA]] : !llvm.ptr<3> -> !llvm.ptr<7>
// CHECK:         %[[DSMEM_PTR:.*]] = llvm.addrspacecast %[[CLUSTER_STORE]] : !llvm.ptr<7> to !llvm.ptr
// CHECK:         %[[GEP_STORE:.*]] = llvm.getelementptr inbounds %[[DSMEM_PTR]][0, %[[LOCAL_IDX]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<1024 x f32>
// CHECK:         llvm.store %[[VAL]], %[[GEP_STORE]] : f32, !llvm.ptr
// CHECK:         gpu.barrier scope <cluster>
// CHECK:         %[[CLUSTER_LOAD:.*]] = nvvm.mapa %[[ADDR]], %{{.*}} : !llvm.ptr<3> -> !llvm.ptr<7>
// CHECK:         %[[DSMEM_PTR2:.*]] = llvm.addrspacecast %[[CLUSTER_LOAD]] : !llvm.ptr<7> to !llvm.ptr
// CHECK:         %[[GEP_LOAD:.*]] = llvm.getelementptr inbounds %[[DSMEM_PTR2]][0, %{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<1024 x f32>
// CHECK:         %[[LOAD:.*]] = llvm.load %[[GEP_LOAD]] : !llvm.ptr -> f32
// CHECK:         return %[[LOAD]] : f32

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @cluster_dsmem_sub_byte_extract(%idx: index) -> i4 {
    %shared = xla_gpu.allocate_shared : tensor<3xi4>
    %ext = tensor.extract %shared[%idx] {xla_gpu.dsmem} : tensor<3xi4>
    return %ext : i4
  }
}
// CHECK-LABEL: @cluster_dsmem_sub_byte_extract(
// CHECK-SAME:      %[[IDX:.*]]: index) -> i4 {
// CHECK-DAG:     %[[ADDR:.*]] = llvm.mlir.addressof @shared_{{.*}} : !llvm.ptr<3>
// CHECK-DAG:     %[[IDX_INT:.*]] = arith.index_castui %[[IDX]] : index to i32
// CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : i32
// CHECK:         %[[CTA:.*]] = arith.divui %[[IDX_INT]], %[[C3]] : i32
// CHECK:         %[[LOCAL_IDX:.*]] = arith.remui %[[IDX_INT]], %[[C3]] : i32
// CHECK:         %[[CLUSTER:.*]] = nvvm.mapa %[[ADDR]], %[[CTA]] : !llvm.ptr<3> -> !llvm.ptr<7>
// CHECK:         %[[DSMEM_PTR:.*]] = llvm.addrspacecast %[[CLUSTER]] : !llvm.ptr<7> to !llvm.ptr
// CHECK:         %[[BYTE_IDX:.*]] = arith.shrui %[[LOCAL_IDX]], %{{.*}} : i32
// CHECK:         %[[GEP:.*]] = llvm.getelementptr inbounds %[[DSMEM_PTR]][0, %[[BYTE_IDX]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<2 x i8>
// CHECK:         llvm.load %[[GEP]] : !llvm.ptr -> i8

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @cluster_dsmem_sub_byte_insert(%idx: index, %val: i4) -> tensor<3xi4> {
    %shared = xla_gpu.allocate_shared : tensor<3xi4>
    %ins = tensor.insert %val into %shared[%idx] {xla_gpu.dsmem} : tensor<3xi4>
    return %ins : tensor<3xi4>
  }
}
// CHECK-LABEL: @cluster_dsmem_sub_byte_insert(
// CHECK-SAME:      %[[IDX:.*]]: index, %[[VAL:.*]]: i4)
// CHECK-DAG:     %[[ADDR:.*]] = llvm.mlir.addressof @shared_{{.*}} : !llvm.ptr<3>
// CHECK-DAG:     %[[IDX_INT:.*]] = arith.index_castui %[[IDX]] : index to i32
// CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : i32
// CHECK:         arith.divui %[[IDX_INT]], %[[C3]] : i32
// CHECK:         arith.remui %[[IDX_INT]], %[[C3]] : i32
// CHECK:         %[[SMEM:.*]] = llvm.addrspacecast %{{.*}} : !llvm.ptr to !llvm.ptr<3>
// CHECK:         %[[CLUSTER:.*]] = nvvm.mapa %[[SMEM]], %{{.*}} : !llvm.ptr<3> -> !llvm.ptr<7>
// CHECK:         %[[DSMEM_PTR:.*]] = llvm.addrspacecast %[[CLUSTER]] : !llvm.ptr<7> to !llvm.ptr
// CHECK:         llvm.getelementptr inbounds %[[DSMEM_PTR]][0, %{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<2 x i8>
// CHECK:         llvm.cmpxchg

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @cluster_dsmem_sub_byte_atomic_rmw(%idx: index, %val: i4)
      -> tensor<3xi4> {
    %shared = xla_gpu.allocate_shared : tensor<3xi4>
    %rmw = xla.atomic_rmw %shared[%idx] : tensor<3xi4> {
      ^bb0(%cur: i4):
        %add = arith.addi %cur, %val : i4
        xla.yield %add : i4
    } {xla_gpu.dsmem}
    return %rmw : tensor<3xi4>
  }
}
// CHECK-LABEL: @cluster_dsmem_sub_byte_atomic_rmw(
// CHECK-SAME:      %[[IDX:.*]]: index, %[[VAL:.*]]: i4)
// CHECK-DAG:     %[[ADDR:.*]] = llvm.mlir.addressof @shared_{{.*}} : !llvm.ptr<3>
// CHECK-DAG:     %[[IDX_INT:.*]] = arith.index_castui %[[IDX]] : index to i32
// CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : i32
// CHECK:         %[[CTA:.*]] = arith.divui %[[IDX_INT]], %[[C3]] : i32
// CHECK:         %[[LOCAL_IDX:.*]] = arith.remui %[[IDX_INT]], %[[C3]] : i32
// CHECK:         %[[CLUSTER:.*]] = nvvm.mapa %[[ADDR]], %[[CTA]] : !llvm.ptr<3> -> !llvm.ptr<7>
// CHECK:         %[[DSMEM_PTR:.*]] = llvm.addrspacecast %[[CLUSTER]] : !llvm.ptr<7> to !llvm.ptr
// CHECK:         %[[BYTE_IDX:.*]] = arith.shrui %[[LOCAL_IDX]], %{{.*}} : i32
// CHECK:         %[[GEP:.*]] = llvm.getelementptr inbounds %[[DSMEM_PTR]][0, %[[BYTE_IDX]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<2 x i8>
// CHECK:         llvm.cmpxchg

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @cluster_dsmem_atomic_fadd(%idx: index, %val: f32) -> tensor<1024xf32> {
    %shared = xla_gpu.allocate_shared : tensor<1024xf32>
    %rmw = xla.atomic_rmw %shared[%idx] : tensor<1024xf32> {
      ^bb0(%cur: f32):
        %sum = arith.addf %cur, %val : f32
        xla.yield %sum : f32
    } {xla_gpu.dsmem}
    return %rmw : tensor<1024xf32>
  }
}
// CHECK-LABEL: @cluster_dsmem_atomic_fadd(
// CHECK:         %[[CLUSTER:.*]] = nvvm.mapa %{{.*}}, %{{.*}} : !llvm.ptr<3> -> !llvm.ptr<7>
// CHECK:         %[[RMW_PTR:.*]] = llvm.addrspacecast %[[CLUSTER]] : !llvm.ptr<7> to !llvm.ptr
// CHECK:         llvm.getelementptr inbounds %[[RMW_PTR]][0, %{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<1024 x f32>
// CHECK:         llvm.atomicrmw ignore_denormal_mode fadd

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @cluster_dsmem_atomic_store(%idx: index, %val: f32) -> tensor<1024xf32> {
    %shared = xla_gpu.allocate_shared : tensor<1024xf32>
    %st = xla.atomic_store %val into %shared[%idx] {xla_gpu.dsmem} : tensor<1024xf32>
    return %st : tensor<1024xf32>
  }
}
// CHECK-LABEL: @cluster_dsmem_atomic_store(
// CHECK:         %[[CLUSTER:.*]] = nvvm.mapa %{{.*}}, %{{.*}} : !llvm.ptr<3> -> !llvm.ptr<7>
// CHECK:         %[[ST_PTR:.*]] = llvm.addrspacecast %[[CLUSTER]] : !llvm.ptr<7> to !llvm.ptr
// CHECK:         llvm.getelementptr inbounds %[[ST_PTR]][0, %{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<1024 x f32>
// CHECK:         llvm.store %{{.*}}, %{{.*}} atomic
