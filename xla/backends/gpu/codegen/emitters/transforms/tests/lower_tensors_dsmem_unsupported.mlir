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
// RUN: -xla-gpu-lower-tensors="gpu_compute_capability='rocm_compute_capability {gcn_arch_name: \"gfx942:sramecc+:xnack\"}'" \
// RUN: -verify-diagnostics
// RUN: emitters_opt %s -split-input-file \
// RUN: -xla-gpu-lower-tensors="gpu_compute_capability='cuda_compute_capability {major: 8}'" \
// RUN: -verify-diagnostics
// RUN: emitters_opt %s -split-input-file \
// RUN: -xla-gpu-lower-tensors="gpu_compute_capability='oneapi_compute_capability {architecture: \"xe2hpg\"}'" \
// RUN: -verify-diagnostics

func.func @cluster_dsmem_insert_unsupported(%cta: index, %idx: index, %val: i4) -> tensor<4xi4> {
  %shared = xla_gpu.allocate_shared : tensor<4xi4>
  // expected-error @+1 {{DSMEM accesses are only supported on Hopper+ CUDA GPUs}}
  %ins = xla_gpu.dsmem_insert %val into %shared[%cta, %idx] : tensor<4xi4>
  return %ins : tensor<4xi4>
}

// -----

func.func @cluster_dsmem_extract_unsupported(%cta: index, %idx: index) -> f32 {
  %shared = xla_gpu.allocate_shared : tensor<4xf32>
  // expected-error @+1 {{DSMEM accesses are only supported on Hopper+ CUDA GPUs}}
  %ext = xla_gpu.dsmem_extract %shared[%cta, %idx] : tensor<4xf32>
  return %ext : f32
}

// -----

func.func @cluster_sync_unsupported() -> tensor<4xf32> {
  %shared = xla_gpu.allocate_shared : tensor<4xf32>
  // expected-error @+1 {{cluster-scoped sync_threads is only supported on Hopper+ CUDA GPUs}}
  %sync = xla_gpu.sync_threads scope <cluster> %shared : tensor<4xf32>
  return %sync : tensor<4xf32>
}
