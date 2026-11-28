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
// RUN: -verify-diagnostics

func.func @dsmem_extract_non_shared(%arg0: tensor<16xf32>, %cta: index, %idx: index) -> f32 {
  // expected-error @+1 {{DSMEM accesses require a shared memory tensor}}
  %ext = xla_gpu.dsmem_extract %arg0[%cta, %idx] : tensor<16xf32>
  return %ext : f32
}

// -----

func.func @dsmem_insert_non_shared(%arg0: tensor<16xf32>, %cta: index, %idx: index, %val: f32) -> tensor<16xf32> {
  // expected-error @+1 {{DSMEM accesses require a shared memory tensor}}
  %ins = xla_gpu.dsmem_insert %val into %arg0[%cta, %idx] : tensor<16xf32>
  return %ins : tensor<16xf32>
}
