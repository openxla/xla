/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/backends/gpu/codegen/prefetch.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/prefetch_kernel.h"

namespace xla {
namespace gpu {

using se::gpu::PrefetchKernel;

namespace {
se::KernelLoaderSpec::KernelArgsPacking CreatePrefetchArgsPacking(
    std::array<int, PrefetchKernel::kMaxNumBuffers> buffer_sizes) {
  return [=](const se::Kernel& kernel, const se::KernelArgs& args) {
    auto* mem_args = se::Cast<se::KernelArgsDeviceMemoryArray>(&args);
    const int num_buffers = mem_args->number_of_arguments();
    CHECK_LE(num_buffers, PrefetchKernel::kMaxNumBuffers);
    std::array<const void*, PrefetchKernel::kMaxNumBuffers> pointers;
    for (int i = 0; i < num_buffers; ++i) {
      CHECK_GT(buffer_sizes[i], 0);
      pointers[i] = mem_args->device_memory_ptr(i);
    }
    return se::PackKernelArgs(/*shmem_bytes=*/0, pointers, buffer_sizes);
  };
}
}  // namespace

absl::StatusOr<FusionEmissionResult> L2PrefetchFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  const HloInstruction& call =
      *fusion.fused_instructions_computation()->root_instruction();
  TF_RET_CHECK(call.operand_count() <= PrefetchKernel::kMaxNumBuffers);
  std::array<int, PrefetchKernel::kMaxNumBuffers> buffer_sizes{};
  for (int i = 0; i < call.operand_count(); ++i) {
    buffer_sizes[i] = ShapeUtil::ByteSizeOf(call.operand(i)->shape());
  }

  int num_blocks;
  if (!absl::SimpleAtoi(
          call.get_frontend_attribute(PrefetchKernel::kNumBlocksAttr)
              .value_or(""),
          &num_blocks)) {
    num_blocks = PrefetchKernel::kDefaultBlocks;
  }
  TF_RET_CHECK(num_blocks > 0);

  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      se::PlatformManager::PlatformWithName(
                          ir_emitter_context.platform_name()));
  TF_ASSIGN_OR_RETURN(se::KernelLoaderSpec spec,
                      se::gpu::GpuKernelRegistry::GetGlobalRegistry()
                          .FindKernel<se::gpu::PrefetchKernel>(platform->id()));
  spec.set_kernel_args_packing(CreatePrefetchArgsPacking(buffer_sizes));

  TF_ASSIGN_OR_RETURN(
      auto kernel_arguments,
      emitters::KernelArguments::Create(ir_emitter_context.buffer_assignment(),
                                        GetDefaultBufferAlignment(), &fusion));
  FusionEmissionResult result;
  result.thunks.emplace_back(std::make_unique<CustomKernelThunk>(
      &fusion,
      CustomKernel(std::string(kL2Prefetch), std::move(spec),
                   se::BlockDim(num_blocks), se::ThreadDim(1024),
                   /*shared_memory_bytes=*/0),
      std::move(kernel_arguments), ir_emitter_context.GetNextThunkId()));
  return result;
}

}  // namespace gpu
}  // namespace xla
