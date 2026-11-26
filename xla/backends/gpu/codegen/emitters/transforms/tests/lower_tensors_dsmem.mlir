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
  func.func @cluster_dsmem(%cta: index, %idx: index, %val: f32) -> f32 {
    %shared = xla_gpu.allocate_shared : tensor<1024xf32>
    %inserted = xla_gpu.dsmem_insert %val into %shared[%cta, %idx] : tensor<1024xf32>
    %synced = xla_gpu.sync_threads scope <cluster> %inserted : tensor<1024xf32>
    %extracted = xla_gpu.dsmem_extract %synced[%cta, %idx] : tensor<1024xf32>
    return %extracted : f32
  }
}
// CHECK-LABEL: @cluster_dsmem(
// CHECK-SAME:      %[[CTA:.*]]: index, %[[IDX:.*]]: index, %[[VAL:.*]]: f32) -> f32 {
// CHECK-DAG:     %[[ADDR:.*]] = llvm.mlir.addressof @shared_{{.*}} : !llvm.ptr<3>
// CHECK-DAG:     %[[CTA_I32:.*]] = arith.index_castui %[[CTA]] : index to i32
// CHECK:         %[[CLUSTER_STORE:.*]] = nvvm.mapa %[[ADDR]], %[[CTA_I32]] : !llvm.ptr<3> -> !llvm.ptr<7>
// CHECK:         %[[DSMEM_PTR:.*]] = llvm.addrspacecast %[[CLUSTER_STORE]] : !llvm.ptr<7> to !llvm.ptr
// CHECK:         %[[IDX_I32:.*]] = arith.index_castui %[[IDX]] : index to i32
// CHECK:         %[[GEP_STORE:.*]] = llvm.getelementptr inbounds %[[DSMEM_PTR]][0, %[[IDX_I32]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<1024 x f32>
// CHECK:         llvm.store %[[VAL]], %[[GEP_STORE]] : f32, !llvm.ptr
// CHECK:         gpu.barrier scope <cluster>
// CHECK:         %[[CLUSTER_LOAD:.*]] = nvvm.mapa %[[ADDR]], %{{.*}} : !llvm.ptr<3> -> !llvm.ptr<7>
// CHECK:         %[[DSMEM_PTR2:.*]] = llvm.addrspacecast %[[CLUSTER_LOAD]] : !llvm.ptr<7> to !llvm.ptr
// CHECK:         %[[GEP_LOAD:.*]] = llvm.getelementptr inbounds %[[DSMEM_PTR2]][0, %{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<1024 x f32>
// CHECK:         %[[LOAD:.*]] = llvm.load %[[GEP_LOAD]] : !llvm.ptr -> f32
// CHECK:         return %[[LOAD]] : f32

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @cluster_dsmem_sub_byte_extract(%cta: index, %idx: index) -> i4 {
    %shared = xla_gpu.allocate_shared : tensor<3xi4>
    %ext = xla_gpu.dsmem_extract %shared[%cta, %idx] : tensor<3xi4>
    return %ext : i4
  }
}
// CHECK-LABEL: @cluster_dsmem_sub_byte_extract(
// CHECK-SAME:      %[[CTA:.*]]: index, %[[IDX:.*]]: index) -> i4 {
// CHECK-DAG:     %[[ADDR:.*]] = llvm.mlir.addressof @shared_{{.*}} : !llvm.ptr<3>
// CHECK-DAG:     %[[CTA_I32:.*]] = arith.index_castui %[[CTA]] : index to i32
// CHECK:         %[[CLUSTER:.*]] = nvvm.mapa %[[ADDR]], %[[CTA_I32]] : !llvm.ptr<3> -> !llvm.ptr<7>
// CHECK:         %[[DSMEM_PTR:.*]] = llvm.addrspacecast %[[CLUSTER]] : !llvm.ptr<7> to !llvm.ptr
// CHECK:         %[[IDX_I32:.*]] = arith.index_castui %[[IDX]] : index to i32
// CHECK:         %[[BYTE_IDX:.*]] = arith.shrui %[[IDX_I32]], %{{.*}} : i32
// CHECK:         %[[GEP:.*]] = llvm.getelementptr inbounds %[[DSMEM_PTR]][0, %[[BYTE_IDX]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<2 x i8>
// CHECK:         llvm.load %[[GEP]] : !llvm.ptr -> i8

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @cluster_dsmem_sub_byte_insert(%cta: index, %idx: index, %val: i4) -> tensor<3xi4> {
    %shared = xla_gpu.allocate_shared : tensor<3xi4>
    %ins = xla_gpu.dsmem_insert %val into %shared[%cta, %idx] : tensor<3xi4>
    return %ins : tensor<3xi4>
  }
}
// CHECK-LABEL: @cluster_dsmem_sub_byte_insert(
// CHECK-SAME:      %[[CTA:.*]]: index, %[[IDX:.*]]: index, %[[VAL:.*]]: i4)
// CHECK-DAG:     %[[ADDR:.*]] = llvm.mlir.addressof @shared_{{.*}} : !llvm.ptr<3>
// CHECK-DAG:     %[[CTA_I32:.*]] = arith.index_castui %[[CTA]] : index to i32
// CHECK:         %[[CLUSTER:.*]] = nvvm.mapa %[[ADDR]], %[[CTA_I32]] : !llvm.ptr<3> -> !llvm.ptr<7>
// CHECK:         %[[DSMEM_PTR:.*]] = llvm.addrspacecast %[[CLUSTER]] : !llvm.ptr<7> to !llvm.ptr
// CHECK:         %[[IDX_I32:.*]] = arith.index_castui %[[IDX]] : index to i32
// CHECK:         %[[BYTE_IDX:.*]] = arith.shrui %[[IDX_I32]], %{{.*}} : i32
// CHECK:         %[[GEP:.*]] = llvm.getelementptr inbounds %[[DSMEM_PTR]][0, %[[BYTE_IDX]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<2 x i8>
// CHECK:         llvm.intr.ptrmask %[[GEP]]
// CHECK:         llvm.cmpxchg

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @cluster_dsmem_reinterpreted_cast(%cta: index, %idx: index, %val: f64) -> f64 {
    %narrow = xla_gpu.allocate_shared : tensor<32xi4>
    %as_f64 = builtin.unrealized_conversion_cast %narrow : tensor<32xi4> to tensor<2xf64>
    %ins = xla_gpu.dsmem_insert %val into %as_f64[%cta, %idx] : tensor<2xf64>
    %ext = xla_gpu.dsmem_extract %ins[%cta, %idx] : tensor<2xf64>
    return %ext : f64
  }
}
// CHECK-LABEL: @cluster_dsmem_reinterpreted_cast(
// CHECK-SAME:      %[[CTA:.*]]: index, %[[IDX:.*]]: index, %[[VAL:.*]]: f64) -> f64 {
// CHECK-NOT:     llvm.cmpxchg
// CHECK-DAG:     %[[ADDR:.*]] = llvm.mlir.addressof @shared_{{.*}} : !llvm.ptr<3>
// CHECK-DAG:     %[[CTA_I32:.*]] = arith.index_castui %[[CTA]] : index to i32
// CHECK:         %[[CLUSTER_STORE:.*]] = nvvm.mapa %[[ADDR]], %[[CTA_I32]] : !llvm.ptr<3> -> !llvm.ptr<7>
// CHECK:         %[[DSMEM_STORE:.*]] = llvm.addrspacecast %[[CLUSTER_STORE]] : !llvm.ptr<7> to !llvm.ptr
// CHECK:         %[[IDX_I32:.*]] = arith.index_castui %[[IDX]] : index to i32
// CHECK:         %[[GEP_STORE:.*]] = llvm.getelementptr inbounds %[[DSMEM_STORE]][0, %[[IDX_I32]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<2 x f64>
// CHECK:         llvm.store %[[VAL]], %[[GEP_STORE]] : f64, !llvm.ptr
// CHECK:         %[[CLUSTER_LOAD:.*]] = nvvm.mapa %{{.*}}, %{{.*}} : !llvm.ptr<3> -> !llvm.ptr<7>
// CHECK:         %[[DSMEM_LOAD:.*]] = llvm.addrspacecast %[[CLUSTER_LOAD]] : !llvm.ptr<7> to !llvm.ptr
// CHECK:         %[[GEP_LOAD:.*]] = llvm.getelementptr inbounds %[[DSMEM_LOAD]][0, %{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<2 x f64>
// CHECK:         %[[LOAD:.*]] = llvm.load %[[GEP_LOAD]] : !llvm.ptr -> f64
// CHECK:         return %[[LOAD]] : f64
