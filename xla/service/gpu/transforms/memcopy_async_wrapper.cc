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

#include "xla/service/gpu/transforms/memcopy_async_wrapper.h"

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/copy.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/transforms/async_wrapper.h"

namespace xla::gpu {

namespace {
bool is_memcopy(HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kFusion) {
    HloFusionInstruction* fusion =
        ::xla::Cast<HloFusionInstruction>(instruction);
    if (DynamicMemcpyFusion::GetMemcpyDescriptorForFusion(*fusion)) {
      return true;
    }
  }
  return false;
}
}  // namespace

absl::StatusOr<bool> MemcopyAsyncWrapper::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return AsyncWrapper(is_memcopy).Run(module, execution_threads);
}

}  // namespace xla::gpu
