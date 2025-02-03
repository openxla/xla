/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/all_gather_combiner.h"

#include <optional>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/collectives/all_gather_combiner.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_collective_combiner_utils.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

namespace {

std::optional<AllGatherCombiner::GroupKey> PipelinedCombinerKey(
    const HloInstruction* instruction, const HloDomainMap& domain_map,
    bool combine_by_dim, bool combine_different_dtypes) {
  bool is_pipelined = false;
  auto backend_config = instruction->backend_config<GpuBackendConfig>();
  if (backend_config.ok()) {
    is_pipelined = backend_config->collective_backend_config().is_pipelined();
  }
  auto key = AllGatherCombiner::CombineKey(
      instruction, domain_map, combine_by_dim, combine_different_dtypes);
  if (!key.has_value()) {
    return std::nullopt;
  }
  return AllGatherCombiner::GroupKey{
      std::get<0>(key.value()),
      std::get<1>(key.value()),
      std::get<2>(key.value()),
      std::get<3>(key.value()),
      std::get<4>(key.value()),
      std::move(std::get<5>(key.value())),
      is_pipelined ? "pipelined" : "non-pipelined"};
}

}  // namespace

absl::StatusOr<bool> GpuAllGatherCombiner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // If there are no pipelined instructions in the IR, the optimizations below
  // do not kick in anyway.
  // Exit early so we do not perform expensive scheduling dry run below.
  if (!ContainsPipelinedInstruction(*module)) {
    return AllGatherCombiner::Run(module, execution_threads);
  }

  // Combine as much as possible for pipelined collectives.
  // Always respects the threshold users set if it doesn't increase
  // memory pressure.
  int64_t previous_combiner_threshold = combine_threshold_in_bytes_;
  combine_threshold_in_bytes_ =
      std::min(ComputeSuggestedCombinerThreshold(
                   *module, device_info_, HloOpcode::kAllGather, pointer_size_),
               previous_combiner_threshold);
  TF_ASSIGN_OR_RETURN(
      bool combined_pipelined_instructions,
      RunWithKeyCombiner(module, execution_threads, PipelinedCombinerKey));
  return combined_pipelined_instructions;
}

}  // namespace xla::gpu
