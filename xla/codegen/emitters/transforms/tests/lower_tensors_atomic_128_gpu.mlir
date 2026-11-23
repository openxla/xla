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
// RUN: emitters_opt %s -split-input-file --verify-diagnostics \
// RUN: -xla-lower-tensors="gpu_device_info='cuda_compute_capability {major: 9}'" \
// RUN: | FileCheck %s

// GPU shared memory 128 bit atomic lowering tests (kept separate from
// lower_tensors_atomic_128.mlir so CPU runs do not lower xla_gpu ops).

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i128, dense<128> : vector<2xi64>>>} {
  func.func @atomic_store_and_rmw_i128_shared(
      %out: tensor<4xi128> {llvm.align = 16 : index}, %val: i128, %i: index)
      -> tensor<4xi128> {
    %shared = xla_gpu.allocate_shared : tensor<4xi128>
    %stored = xla.atomic_store %val into %shared[%i] : tensor<4xi128>
    %rmw = xla.atomic_rmw %stored[%i] : tensor<4xi128> {
      ^bb0(%current : i128):
        %a = arith.addi %current, %val : i128
        xla.yield %a : i128
    }
    %synced = xla_gpu.sync_threads %rmw : tensor<4xi128>
    %v = tensor.extract %synced[%i] : tensor<4xi128>
    %res = tensor.insert %v into %out[%i] : tensor<4xi128>
    return %res : tensor<4xi128>
  }
}
// CHECK: llvm.mlir.global private @[[SH128:shared_.*]]() {addr_space = 3 : i32} : !llvm.array<4 x i128>
// CHECK: @atomic_store_and_rmw_i128_shared
// CHECK:   %[[SH_ADDR:.*]] = llvm.mlir.addressof @[[SH128]] : !llvm.ptr<3>
// CHECK:   %[[SH_GEN:.*]] = llvm.addrspacecast %[[SH_ADDR]] : !llvm.ptr<3> to !llvm.ptr
// CHECK:   %[[ST_ADDR:.*]] = llvm.getelementptr inbounds %[[SH_GEN]]
// CHECK:   %[[ST_INIT:.*]] = llvm.load %[[ST_ADDR]] atomic unordered <alignment = 16> : !llvm.ptr -> i128
// CHECK:   scf.while (%[[ST_OLD:.*]] = %[[ST_INIT]]) : (i128) -> i128
// CHECK:     llvm.cmpxchg %[[ST_ADDR]], %[[ST_OLD]], %{{.*}} monotonic monotonic : !llvm.ptr, i128
// CHECK:   %[[RMW_ADDR:.*]] = llvm.getelementptr inbounds %[[SH_GEN]]
// CHECK:   %[[RMW_INIT:.*]] = llvm.load %[[RMW_ADDR]] atomic unordered <alignment = 16> : !llvm.ptr -> i128
// CHECK:   scf.while (%[[RMW_OLD:.*]] = %[[RMW_INIT]]) : (i128) -> i128
// CHECK:     llvm.cmpxchg %[[RMW_ADDR]], %[[RMW_OLD]], %{{.*}} monotonic monotonic : !llvm.ptr, i128

// -----

// Base alignment comes from the shared allocation element type (4 bytes for
// i8), not the reinterpreted f128 view.
func.func @atomic_store_f128_on_narrow_shared(%val: f128) -> tensor<1xf128> {
  %c0 = arith.constant 0 : index
  %shared = xla_gpu.allocate_shared : tensor<16xi8>
  %wide = builtin.unrealized_conversion_cast %shared : tensor<16xi8> to tensor<1xf128>
  // expected-error @+1 {{failed to lower to LLVM atomic operation}}
  %ret = xla.atomic_store %val into %wide[%c0] : tensor<1xf128>
  return %ret : tensor<1xf128>
}
