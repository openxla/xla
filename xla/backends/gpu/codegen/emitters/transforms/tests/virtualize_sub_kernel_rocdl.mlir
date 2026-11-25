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
// RUN: emitters_opt %s -split-input-file -verify-diagnostics \
// RUN:   -xla-gpu-virtualize-sub-kernel="entry_function_name=sub_kernel expected_num_tensor_args=2 block_dim_x=4 block_dim_y=2 block_dim_z=1 thread_dim_x=32 thread_dim_y=4 thread_dim_z=1 max_threads=256 warp_size=64 is_amd_gpu=true" \
// RUN: | FileCheck %s

module {
  llvm.func @sub_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %scratch0: !llvm.ptr, %scratch1: !llvm.ptr) attributes {rocdl.kernel, xla.entry} {
    %bx = rocdl.workgroup.id.x : i32
    %by = rocdl.workgroup.id.y : i32
    %tx = rocdl.workitem.id.x : i32
    %ty = rocdl.workitem.id.y : i32
    %sum0 = llvm.add %bx, %by : i32
    %sum1 = llvm.add %tx, %ty : i32
    %sum2 = llvm.add %sum0, %sum1 : i32
    llvm.store %sum2, %arg1 : i32, !llvm.ptr
    llvm.return
  }
}

// CHECK-LABEL: llvm.func @sub_kernel(
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr, %[[REL_BID:.*]]: i32)
// CHECK-NOT: rocdl.kernel
// CHECK-NOT: xla.entry
// CHECK-DAG: %[[RAW_TID:.*]] = rocdl.workitem.id.x : i32
// CHECK-DAG: %[[C32:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK-DAG: %[[TX:.*]] = llvm.urem %[[RAW_TID]], %[[C32]] : i32
// CHECK-DAG: %[[TY:.*]] = llvm.udiv %[[RAW_TID]], %[[C32]] : i32
// CHECK-DAG: %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK-DAG: %[[BX:.*]] = llvm.urem %[[REL_BID]], %[[C4]] : i32
// CHECK-DAG: %[[BY:.*]] = llvm.udiv %[[REL_BID]], %[[C4]] : i32
// CHECK-NOT: rocdl.workgroup.id
// CHECK: llvm.return

// -----

// Hardware IDs may not be read from helper functions of the sub-kernel.
module {
  llvm.func @helper_with_block_id() -> i32 {
    // expected-error @+1 {{Hardware ID and dimension ops in non-entry functions are not supported in horizontal fusion.}}
    %bx = rocdl.workgroup.id.x : i32
    llvm.return %bx : i32
  }

  llvm.func @sub_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {rocdl.kernel, xla.entry} {
    %0 = llvm.call @helper_with_block_id() : () -> i32
    llvm.return
  }
}
