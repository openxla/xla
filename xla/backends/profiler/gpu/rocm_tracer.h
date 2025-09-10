/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_FACADE_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_FACADE_H_

// Backend: 3=v3 (rocprofiler-sdk), 1=v1 (roctracer). Default to v3.
#ifndef XLA_GPU_ROCM_TRACER_BACKEND
#define XLA_GPU_ROCM_TRACER_BACKEND 3
#endif

#if XLA_GPU_ROCM_TRACER_BACKEND == 3
#include "xla/backends/profiler/gpu/rocm_profiler_sdk.h"
#else
#include "xla/backends/profiler/gpu/rocm_tracer_v1.h"
#endif

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_FACADE_H_
