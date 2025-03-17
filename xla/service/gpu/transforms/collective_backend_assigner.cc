/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/collective_backend_assigner.h"

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/backends/gpu/collectives/nvshmem_collectives.h"

namespace xla {
namespace gpu {

// Assigns either NVSHMEM or NCCL as the backend for collective operations based
// on:
// 1. Communication pattern (intranode vs internode)
// 2. Message size (compared against threshold_in_bytes)
absl::StatusOr<bool> CollectiveBackendAssigner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  TF_ASSIGN_OR_RETURN(auto num_processes,
                      NvshmemCollectives::Default()->NumOfParticipantsInTeam(
                          NvshmemCollectives::TEAMSKIND::kNODE));
  VLOG(1) << "CollectiveBackendAssigner: device_count_per_process = "
          << NvshmemCollectives::Default()->device_count_per_process()
          << " num_processes = " << num_processes;

  for (HloComputation* comp : module->computations()) {
    for (HloInstruction* instr : comp->instructions()) {
      if (IsCollectiveOp(instr)) {
        TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                            instr->backend_config<GpuBackendConfig>());
        CollectiveBackendConfig& backend_config =
            *gpu_config.mutable_collective_backend_config();

        auto backend =
            !HasInternodeCommunication(*instr, num_processes) &&
                    GetShapeSize(instr->shape()) <= threshold_in_bytes_
                ? CollectiveBackendConfig::NVSHMEM
                : CollectiveBackendConfig::NCCL;

        backend_config.set_backend(backend);
        VLOG(1) << "CollectiveBackendAssigner: setting backend to "
                << CollectiveBackendConfig_CollectiveBackend_Name(backend);
        TF_RETURN_IF_ERROR(instr->set_backend_config(gpu_config));
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
