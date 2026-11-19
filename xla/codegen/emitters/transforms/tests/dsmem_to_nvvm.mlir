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
// RUN:   -xla-lower-to-llvm-gpu="gpu_device_info='cuda_compute_capability {major: 9}'" \
// RUN:   | FileCheck %s

func.func @dsmem_pointer(%ptr: !llvm.ptr, %cta_rank: i32) -> !llvm.ptr {
  %0 = xla_gpu.dsmem_pointer %ptr, %cta_rank : !llvm.ptr -> !llvm.ptr
  return %0 : !llvm.ptr
}
// CHECK-LABEL: llvm.func @dsmem_pointer(
// CHECK-SAME:    %[[PTR:.*]]: !llvm.ptr, %[[CTA:.*]]: i32) -> !llvm.ptr
// CHECK:         %[[SHARED:.*]] = llvm.addrspacecast %[[PTR]] : !llvm.ptr to !llvm.ptr<3>
// CHECK:         %[[MAPA:.*]] = nvvm.mapa %[[SHARED]], %[[CTA]] : !llvm.ptr<3> -> !llvm.ptr<7>
// CHECK:         %[[RES:.*]] = llvm.addrspacecast %[[MAPA]] : !llvm.ptr<7> to !llvm.ptr
// CHECK:         llvm.return %[[RES]] : !llvm.ptr
