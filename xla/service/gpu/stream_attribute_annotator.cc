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

#include "xla/service/gpu/stream_attribute_annotator.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {
absl::StatusOr<bool> AnnotateStreamAttributesForUsers(HloInstruction* instr) {
  auto instr_gpu_config = instr->backend_config<GpuBackendConfig>();
  if (!instr_gpu_config.ok()) {
    return false;
  }
  bool changed = false;
  int64_t stream_id = instr_gpu_config->operation_queue_id();
  if (stream_id == 0) {
    return changed;
  }
  std::vector<HloInstruction*> all_consumers;
  for (auto user : instr->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement) {
      user = user->users()[0];
    }
    all_consumers.push_back(user);
  }

  for (auto user : all_consumers) {
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        user->backend_config<GpuBackendConfig>());
    auto it = absl::c_find(gpu_config.wait_on_operation_queues(), stream_id);
    if (it == gpu_config.wait_on_operation_queues().end()) {
      gpu_config.mutable_wait_on_operation_queues()->Add(stream_id);
      TF_RETURN_IF_ERROR(user->set_backend_config(gpu_config));
      changed = true;
    }
  }

  return changed;
}
}  // namespace

absl::StatusOr<bool> StreamAttributeAnnotator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      2, "StreamAttributeAnnotator::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (const HloComputation* comp : module->computations(execution_threads)) {
    for (HloInstruction* instr : comp->instructions()) {
      if (!instr->has_backend_config()) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(bool result, AnnotateStreamAttributesForUsers(instr));
      changed |= result;
    }
  }
  XLA_VLOG_LINES(
      2, "StreamAttributeAnnotator::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace gpu
}  // namespace xla
