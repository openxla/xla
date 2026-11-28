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
// RUN: -xla-cpu-lower-tensors

func.func @cluster_dsmem_insert_cpu(%in: tensor<4xi4>, %idx: index, %val: i4) -> tensor<4xi4> {
  // expected-error @+1 {{DSMEM accesses are only supported on CUDA GPUs}}
  %ins = tensor.insert %val into %in[%idx] {xla_gpu.dsmem} : tensor<4xi4>
  return %ins : tensor<4xi4>
}

// -----

func.func @cluster_dsmem_extract_cpu(%in: tensor<4xf32>, %idx: index) -> f32 {
  // expected-error @+1 {{DSMEM accesses are only supported on CUDA GPUs}}
  %ext = tensor.extract %in[%idx] {xla_gpu.dsmem} : tensor<4xf32>
  return %ext : f32
}

// -----

func.func @cluster_dsmem_atomic_store_cpu(%in: tensor<4xf32>, %idx: index, %val: f32) -> tensor<4xf32> {
  // expected-error @+1 {{DSMEM accesses are only supported on CUDA GPUs}}
  %st = xla.atomic_store %val into %in[%idx] {xla_gpu.dsmem} : tensor<4xf32>
  return %st : tensor<4xf32>
}
