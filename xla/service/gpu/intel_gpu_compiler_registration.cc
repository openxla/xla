/* Copyright (c) 2025 Intel Corporation

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

#include "xla/service/gpu/intel_gpu_compiler.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

static bool InitCompilerModule() {
  xla::Compiler::RegisterCompilerFactory(
      stream_executor::sycl::kSyclPlatformId,
      []() { return std::make_unique<xla::gpu::IntelGpuCompiler>(); });
  return true;
}
static bool compiler_module_initialized = InitCompilerModule();
