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
// RUN: xla-opt %s --triton-xla-pipeline='target=gfx942' \
// RUN:   | FileCheck %s --check-prefix=CHECK-AUTO
//
// RUN: xla-opt %s --triton-xla-pipeline='target=gfx942 kpack=2' \
// RUN:   | FileCheck %s --check-prefix=CHECK-KPACK2
//
// RUN: xla-opt %s --triton-xla-pipeline='target=gfx942' \
// RUN:   --dump-pass-pipeline -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PIPELINE-AUTO
//
// RUN: xla-opt %s --triton-xla-pipeline='target=gfx942 kpack=2' \
// RUN:   --dump-pass-pipeline -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PIPELINE-KPACK2
//
// RUN: xla-opt %s --triton-xla-pipeline='target=gfx950 kpack=2' \
// RUN:   --dump-pass-pipeline -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PIPELINE-GFX950

// kpack=2 widens the shared memory loads feeding the MFMA from 64 to 128 bits.

func.func @dot(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f32>) {
  %lhs = triton_xla.extract from %arg0
      as memref<64x64xf16, #xtile.layout<[1, 0]>>
      [0, 0] [64, 64] [1, 1] : tensor<64x64xf16>
  %rhs = triton_xla.extract from %arg1
      as memref<64x64xf16, #xtile.layout<[1, 0]>>
      [0, 0] [64, 64] [1, 1] : tensor<64x64xf16>
  %acc = arith.constant dense<0.0> : tensor<64x64xf32>
  %res = tt.dot %lhs, %rhs, %acc : tensor<64x64xf16> * tensor<64x64xf16>
      -> tensor<64x64xf32>
  triton_xla.insert %res into %arg2
      as memref<64x64xf32, #xtile.layout<[1, 0]>>
      [0, 0] [64, 64] [1, 1] : tensor<64x64xf32>
  func.return
}

// CHECK-AUTO-LABEL:     llvm.func @dot
// CHECK-AUTO-NOT:       !llvm.ptr<3> -> vector<8xf16>
// CHECK-AUTO:           rocdl.mfma.f32.32x32x8f16

// CHECK-KPACK2-LABEL:   llvm.func @dot
// CHECK-KPACK2:         !llvm.ptr<3> -> vector<8xf16>
// CHECK-KPACK2:         rocdl.mfma.f32.32x32x8f16

// PIPELINE-AUTO:        tritonamdgpu-accelerate-matmul{{.*}} kPack=1

// PIPELINE-KPACK2:      tritonamdgpu-accelerate-matmul{{.*}} kPack=2

// PIPELINE-GFX950:      tritonamdgpu-accelerate-matmul{{.*}} kPack=1
