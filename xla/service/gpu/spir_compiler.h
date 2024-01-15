/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_SPIR_COMPILER_H_
#define XLA_SERVICE_GPU_SPIR_COMPILER_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {

// SPIRCompiler generates efficient GPU executables for SPIR target.
class SPIRCompiler : public GpuCompiler {
 public:
  SPIRCompiler();
  ~SPIRCompiler() override {}

  absl::Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module, se::GpuComputeCapability gpu_version,
      se::dnn::VersionInfo dnn_version,
      se::DeviceMemoryAllocator* device_allocator) override;

  absl::Status OptimizeHloPostLayoutAssignment(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      const CompileOptions& options, const TargetConfig& gpu_target_config,
      tsl::thread::ThreadPool* thread_pool) override;

  HloDataflowAnalysis::CanShareBuffer GetCanShareBuffer() const override;

  absl::StatusOr<BackendCompileResult> CompileTargetBinary(
      const HloModuleConfig& module_config, llvm::Module* llvm_module,
      se::GpuComputeCapability gpu_version, bool relocatable,
      const HloModule* debug_module, const CompileOptions& options) override;

 private:
  SPIRCompiler(const SPIRCompiler&) = delete;
  SPIRCompiler& operator=(const SPIRCompiler&) = delete;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_SPIR_COMPILER_H_
