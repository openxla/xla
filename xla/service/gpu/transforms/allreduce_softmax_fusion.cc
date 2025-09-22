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

#include "xla/service/gpu/transforms/allreduce_softmax_fusion.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

namespace {

// Check if the instruction is a Triton softmax fusion
bool IsTritonSoftmaxFusion(const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kFusion) {
    return false;
  }

  if (!instruction->has_backend_config()) {
    return false;
  }

  auto backend_config = instruction->backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    return false;
  }

  return backend_config->has_fusion_backend_config() &&
         backend_config->fusion_backend_config().kind() == "__triton" &&
         backend_config->fusion_backend_config()
             .has_block_level_fusion_config();
}

// Check if AllReduce and Triton Softmax can be fused together
bool CanFuseAllReduceWithTritonSoftmax(const HloInstruction* all_reduce,
                                       const HloInstruction* triton_softmax) {
  VLOG(5) << "Checking fusion feasibility between " << all_reduce->name()
          << " and " << triton_softmax->name();

  // Verify this is an AllReduce instruction
  if (all_reduce->opcode() != HloOpcode::kAllReduce) {
    VLOG(5) << "First instruction is not all-reduce";
    return false;
  }

  // Verify this is a Triton softmax fusion
  if (!IsTritonSoftmaxFusion(triton_softmax)) {
    VLOG(5) << "Second instruction is not Triton softmax fusion";
    return false;
  }

  // Check direct data dependency: triton_softmax should use all_reduce as input
  if (triton_softmax->operand_count() != 1 ||
      triton_softmax->operand(0) != all_reduce) {
    VLOG(5) << "No direct dependency from all-reduce to triton softmax";
    return false;
  }

  // Check shape compatibility
  if (!ShapeUtil::Equal(all_reduce->shape(), triton_softmax->shape())) {
    VLOG(5) << "Shape mismatch between all-reduce and triton softmax";
    return false;
  }

  // Ensure all-reduce has only one user (the triton softmax)
  if (all_reduce->user_count() != 1) {
    VLOG(5) << "All-reduce has multiple users, cannot fuse";
    return false;
  }

  // Check tensor size constraints
  const Shape& shape = all_reduce->shape();
  if (shape.IsArray()) {
    int64_t total_elements = ShapeUtil::ElementsIn(shape);
    VLOG(5) << "Tensor has " << total_elements << " elements";

    // NVSHMEM has strict single-CTA requirement (checked later in
    // CreateSafeBackendConfig) For general fusion safety, use a more permissive
    // limit
    constexpr int64_t kMaxElementsForFusion = 100000;  // Increased from 10000
    if (total_elements > kMaxElementsForFusion) {
      VLOG(2) << "Tensor too large for fusion: " << total_elements
              << " elements (max " << kMaxElementsForFusion << ")";
      return false;
    }
  }

  VLOG(3) << "Fusion is feasible";
  return true;
}

// Create safe backend config for the fused operation
absl::StatusOr<GpuBackendConfig> CreateSafeBackendConfig(
    const HloInstruction* triton_softmax, const HloInstruction* all_reduce) {
  if (!triton_softmax->has_backend_config()) {
    return absl::InvalidArgumentError("No backend config available");
  }

  auto backend_config = triton_softmax->backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    return backend_config.status();
  }

  // Since GpuBackendConfig uses oneof for backend_config, we cannot have both
  // fusion_backend_config and collective_backend_config simultaneously.
  // Use a special fusion kind to indicate NVSHMEM fusion instead.
  GpuBackendConfig fused_config = backend_config.value();

  // Check if AllReduce uses NVSHMEM backend
  bool is_nvshmem_fusion = false;
  if (all_reduce->has_backend_config()) {
    auto ar_config = all_reduce->backend_config<GpuBackendConfig>();
    if (ar_config.ok() && ar_config->has_collective_backend_config()) {
      const auto& collective_config = ar_config->collective_backend_config();
      if (collective_config.backend() == CollectiveBackendConfig::NVSHMEM) {
        is_nvshmem_fusion = true;
        VLOG(3) << "Detected NVSHMEM AllReduce, marking fusion as NVSHMEM";
      }
    }
  }

  // Modify fusion kind to indicate NVSHMEM if needed
  if (is_nvshmem_fusion && fused_config.has_fusion_backend_config()) {
    auto* fusion_config = fused_config.mutable_fusion_backend_config();
    // Mark this as NVSHMEM fusion for runtime detection (idempotent operation)
    if (fusion_config->kind() == "__triton" ||
        fusion_config->kind() == "__triton_nvshmem") {
      fusion_config->set_kind("__triton_nvshmem");
      VLOG(3) << "Ensured fusion is marked as __triton_nvshmem for NVSHMEM "
                 "runtime detection";
    }
  }

  // Adjust configuration for small tensors if needed
  if (fused_config.has_fusion_backend_config() &&
      fused_config.fusion_backend_config().has_block_level_fusion_config()) {
    auto* block_config = fused_config.mutable_fusion_backend_config()
                             ->mutable_block_level_fusion_config();

    const Shape& output_shape = triton_softmax->shape();
    if (output_shape.IsArray()) {
      int64_t total_elements = ShapeUtil::ElementsIn(output_shape);

      VLOG(3) << "Configuring backend for NVSHMEM fusion with "
              << total_elements << " elements";

      // Configure optimal tile size for NVSHMEM fusion
      if (block_config->output_tiles_size() > 0) {
        auto* first_tile = block_config->mutable_output_tiles(0);
        if (first_tile->sizes_size() > 0) {
          for (int i = 0; i < first_tile->sizes_size(); ++i) {
            first_tile->set_sizes(i, total_elements);
          }
          VLOG(3) << "Set tile size to " << total_elements;
        }
      }

      // Configure warp count based on tensor size
      int64_t optimal_warps = (total_elements + 31) / 32;
      optimal_warps = std::min(optimal_warps, int64_t{32});
      optimal_warps = std::max(optimal_warps, int64_t{1});
      block_config->set_num_warps(optimal_warps);
      VLOG(3) << "Set num_warps to " << optimal_warps;
    }
  }

  return fused_config;
}

// Perform the actual fusion of AllReduce and Triton Softmax
absl::StatusOr<bool> FuseAllReduceWithTritonSoftmax(
    HloInstruction* all_reduce, HloInstruction* triton_softmax) {
  VLOG(1) << "AllReduceSoftmaxFusion: Attempting to fuse all-reduce with "
             "triton softmax";
  VLOG(3) << "AllReduce: " << all_reduce->ToString();
  VLOG(3) << "TritonSoftmax: " << triton_softmax->ToString();

  HloComputation* computation = all_reduce->parent();

  // Get the triton softmax's fused computation
  HloComputation* original_softmax_computation =
      triton_softmax->fused_instructions_computation();
  if (!original_softmax_computation) {
    return absl::InvalidArgumentError(
        "No fused computation found in triton softmax");
  }

  // Safety check: ensure the computation is not too complex
  if (original_softmax_computation->instruction_count() > 50) {
    VLOG(1) << "Skipping fusion: computation too complex ("
            << original_softmax_computation->instruction_count()
            << " instructions)";
    return false;
  }

  // Create a new fused computation that combines all-reduce and softmax
  HloComputation::Builder builder("nvshmem_fused_allreduce_softmax");

  // Add parameter for the all-reduce input
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, all_reduce->operand(0)->shape(), "input"));

  // Clone the all-reduce operation to work on the parameter
  std::vector<HloInstruction*> operands = {param};
  HloInstruction* cloned_all_reduce = builder.AddInstruction(
      all_reduce->CloneWithNewOperands(all_reduce->shape(), operands));

  // Clone all instructions from the original softmax computation
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      cloned_instructions;

  // Map the original parameter to our cloned all-reduce
  for (HloInstruction* original_param :
       original_softmax_computation->parameter_instructions()) {
    cloned_instructions[original_param] = cloned_all_reduce;
  }

  // Clone other instructions in topological order with safety checks
  for (HloInstruction* original_instr :
       original_softmax_computation->MakeInstructionPostOrder()) {
    if (original_instr->opcode() == HloOpcode::kParameter) {
      continue;  // Already handled above
    }

    // Update operands to use cloned instructions
    std::vector<HloInstruction*> new_operands;
    for (HloInstruction* operand : original_instr->operands()) {
      auto it = cloned_instructions.find(operand);
      if (it == cloned_instructions.end()) {
        return absl::InternalError(absl::StrCat(
            "Operand not found during cloning: ", operand->name()));
      }
      new_operands.push_back(it->second);
    }

    // Clone the instruction
    HloInstruction* cloned_instr =
        builder.AddInstruction(original_instr->CloneWithNewOperands(
            original_instr->shape(), new_operands));
    cloned_instructions[original_instr] = cloned_instr;
  }

  // Set the root to be the cloned version of the original root
  HloInstruction* original_root =
      original_softmax_computation->root_instruction();
  auto root_it = cloned_instructions.find(original_root);
  if (root_it == cloned_instructions.end()) {
    return absl::InternalError("Root instruction not found during cloning");
  }

  // Build the new computation
  std::unique_ptr<HloComputation> fused_computation =
      builder.Build(root_it->second);

  // Add the computation to the module
  HloComputation* new_computation =
      computation->parent()->AddEmbeddedComputation(
          std::move(fused_computation));

  // Create the fused fusion instruction
  std::vector<HloInstruction*> fusion_operands = {
      const_cast<HloInstruction*>(all_reduce->operand(0))};
  HloInstruction* fused_fusion =
      computation->AddInstruction(HloInstruction::CreateFusion(
          triton_softmax->shape(), HloInstruction::FusionKind::kCustom,
          fusion_operands, new_computation));

  // Set up backend config for the fused operation
  TF_ASSIGN_OR_RETURN(auto safe_config,
                      CreateSafeBackendConfig(triton_softmax, all_reduce));
  TF_RETURN_IF_ERROR(fused_fusion->set_backend_config(safe_config));

  // Replace all uses of the triton softmax with the fused fusion
  TF_RETURN_IF_ERROR(triton_softmax->ReplaceAllUsesWith(fused_fusion));

  // Remove the old instructions
  TF_RETURN_IF_ERROR(computation->RemoveInstruction(triton_softmax));
  TF_RETURN_IF_ERROR(computation->RemoveInstruction(all_reduce));

  VLOG(1) << "Successfully fused AllReduce with TritonSoftmax";
  return true;
}

}  // namespace

absl::StatusOr<bool> AllReduceSoftmaxFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "Running AllReduceSoftmaxFusion pass on module: "
          << module->name();

  bool changed = false;
  int fusion_count = 0;
  int fusion_attempts = 0;

  // Look for AllReduce → TritonSoftmax patterns in each computation
  for (HloComputation* computation : module->computations(execution_threads)) {
    // Collect all all-reduce instructions
    std::vector<HloInstruction*> all_reduces;
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kAllReduce) {
        all_reduces.push_back(instruction);
        VLOG(5) << "Found all-reduce: " << instruction->name();
      }
    }

    // Check each all-reduce for fusion opportunities
    for (HloInstruction* all_reduce : all_reduces) {
      // Look for directly connected Triton softmax fusions
      for (HloInstruction* user : all_reduce->users()) {
        if (IsTritonSoftmaxFusion(user) &&
            CanFuseAllReduceWithTritonSoftmax(all_reduce, user)) {
          fusion_attempts++;
          VLOG(2) << "Found fusion opportunity #" << fusion_attempts << ": "
                  << all_reduce->name() << " -> " << user->name();

          // Perform the fusion
          TF_ASSIGN_OR_RETURN(bool fused,
                              FuseAllReduceWithTritonSoftmax(all_reduce, user));

          if (fused) {
            fusion_count++;
            changed = true;
            VLOG(1) << "Successfully fused AllReduce with TritonSoftmax (#"
                    << fusion_count << ")";
            break;  // Move to next all-reduce since this one is now fused
          }
        }
      }
    }
  }

  VLOG(2) << "AllReduceSoftmaxFusion completed. Fusion attempts: "
          << fusion_attempts << ", Successful fusions: " << fusion_count;

  return changed;
}

}  // namespace gpu
}  // namespace xla