/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/hoist_fusion_boundary_bitcasts.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/mlir_context_pool.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

// Changes `instr`'s shape to `new_shape` and inserts a bitcast back to its old
// shape for all existing users (including the parent computation root).
absl::Status WrapWithBitcast(HloInstruction* instr, const Shape& new_shape) {
  Shape old_shape = instr->shape();
  if (instr->has_sharding() && !ShapeUtil::Compatible(old_shape, new_shape)) {
    instr->clear_sharding();
  }
  *instr->mutable_shape() = new_shape;
  HloInstruction* bitcast = instr->parent()->AddInstruction(
      HloInstruction::CreateBitcast(old_shape, instr));
  return instr->ReplaceAllUsesWithDifferentShape(bitcast);
}

bool ShouldProcessFusion(const HloFusionInstruction& fusion) {
  // Custom fusions other than generic Triton fusions (such as Triton GEMM or
  // cuDNN fusions) have specialized emitters and layouts that must not be
  // rewritten here; kLoop and kInput fusions in FusionPipeline never contain
  // dot operations. Note that this pass runs in FusionPipeline before
  // FusionBufferAliasAnnotator populates output_to_operand_aliasing.
  return fusion.fusion_kind() != HloInstruction::FusionKind::kCustom ||
         IsGenericTritonFusion(fusion);
}

absl::StatusOr<bool> HoistParameterBitcasts(
    HloFusionInstruction* fusion_instruction) {
  bool has_changed = false;
  HloComputation* fusion_computation =
      fusion_instruction->fused_instructions_computation();
  HloComputation* parent = fusion_instruction->parent();

  // This pass runs in FusionPipeline after LayoutAssignment (so all array
  // shapes have layouts) and AlgebraicSimplifier (which folds chained and
  // identity bitcasts, leaving at most one bitcast level per parameter).
  for (HloInstruction* param : fusion_computation->parameter_instructions()) {
    const HloInstruction* root = fusion_computation->root_instruction();
    if (param == root) {
      // Parameters directly connected to a root have no users.
      continue;
    }
    TF_RET_CHECK(!param->users().empty())
        << "Unused parameters should have been removed by DCE already";
    const Shape& target_shape = param->users().front()->shape();
    if (!absl::c_all_of(param->users(), [&](const HloInstruction* user) {
          return user->opcode() == HloOpcode::kBitcast &&
                 ShapeUtil::Equal(user->shape(), target_shape);
        })) {
      // The parameter has a non-bitcast user, or gets bitcasted to multiple
      // different shapes. Hoisting the bitcast outside of the fusion won't get
      // rid of the inner bitcast anyway.
      continue;
    }

    // Multi-output fusion tuple roots require distinct operands. Do not unwrap
    // if multiple bitcast users of `param` are direct operands of the root
    // tuple, as replacing them with `param` would duplicate `param` in `root`.
    if (root->opcode() == HloOpcode::kTuple &&
        absl::c_count_if(root->operands(), [&](const HloInstruction* op) {
          return op->opcode() == HloOpcode::kBitcast && op->operand(0) == param;
        }) > 1) {
      continue;
    }

    *param->mutable_shape() = target_shape;
    std::vector<HloInstruction*> bitcast_users(param->users().begin(),
                                               param->users().end());
    for (HloInstruction* bitcast_user : bitcast_users) {
      ABSL_RETURN_IF_ERROR(bitcast_user->ReplaceAllUsesWith(param));
      ABSL_RETURN_IF_ERROR(fusion_computation->RemoveInstruction(bitcast_user));
    }

    HloInstruction* original_operand =
        fusion_instruction->mutable_operand(param->parameter_number());
    // Reuse an existing identical bitcast of `original_operand` in `parent` if
    // available so we do not add redundant bitcast users on the producer.
    auto it = absl::c_find_if(
        original_operand->users(), [&](const HloInstruction* user) {
          return user->opcode() == HloOpcode::kBitcast &&
                 ShapeUtil::Equal(user->shape(), param->shape());
        });
    HloInstruction* outer_bitcast =
        it != original_operand->users().end()
            ? *it
            : parent->AddInstruction(HloInstruction::CreateBitcast(
                  param->shape(), original_operand));
    ABSL_RETURN_IF_ERROR(fusion_instruction->ReplaceOperandWithDifferentShape(
        param->parameter_number(), outer_bitcast));
    has_changed = true;
  }

  return has_changed;
}

absl::StatusOr<bool> HoistRootBitcasts(
    HloFusionInstruction* fusion_instruction) {
  HloComputation* fusion_computation =
      fusion_instruction->fused_instructions_computation();
  HloInstruction* root = fusion_computation->root_instruction();

  if (!fusion_instruction->IsMultiOutputFusion()) {
    if (root->opcode() != HloOpcode::kBitcast) {
      // No bitcast to hoist.
      return false;
    }
    HloInstruction* new_root = root->mutable_operand(0);
    fusion_computation->set_root_instruction(new_root,
                                             /*accept_different_shape=*/true);
    ABSL_RETURN_IF_ERROR(fusion_computation->RemoveInstruction(root));
    ABSL_RETURN_IF_ERROR(
        WrapWithBitcast(fusion_instruction, new_root->shape()));
    return true;
  }

  // Multi-output fusion case.

  // We can only hoist bitcasts for roots that have no users within the
  // computation, and only bitcasts that are not reused for other stuff.
  const int64_t num_outputs = root->operand_count();
  std::vector<HloInstruction*> candidate_roots(num_outputs);
  for (int64_t i = 0; i < num_outputs; ++i) {
    HloInstruction* op = root->mutable_operand(i);
    candidate_roots[i] =
        (op->opcode() == HloOpcode::kBitcast && op->user_count() == 1)
            ? op->mutable_operand(0)
            : op;
  }

  const bool was_same_shape_mof =
      absl::c_all_of(root->operands(), [&](const HloInstruction* operand) {
        return Shape::Equal().IgnoreElementType()(operand->shape(),
                                                  root->operand(0)->shape());
      });

  // Avoid hoisting:
  // * If roots have the form tuple(bitcast(x), x), where hoisting would
  //   duplicate x in the root. Other passes assume that multi-output fusion
  //   roots are separate HloInstructions.
  // * If resulting roots would change the fusion from same-shape to
  //   different-shape outputs. This can break tiling for Triton same-shape
  //   multi-output fusions and kInput transpose/reduction multi-output
  //   emitters.
  absl::flat_hash_set<const HloInstruction*> unique_candidates;
  for (const HloInstruction* candidate : candidate_roots) {
    const bool becomes_non_same_shape_mof =
        was_same_shape_mof &&
        !Shape::Equal().IgnoreElementType()(candidate->shape(),
                                            candidate_roots[0]->shape());

    if (becomes_non_same_shape_mof ||
        !unique_candidates.insert(candidate).second) {
      return false;
    }
  }

  // All looks good, hoist 'em.
  bool has_changed = false;
  for (int64_t i = 0; i < num_outputs; ++i) {
    HloInstruction* old_root = root->mutable_operand(i);
    HloInstruction* new_root = candidate_roots[i];
    if (new_root == old_root) {
      continue;
    }
    const Shape& new_shape = new_root->shape();
    if (fusion_instruction->has_sharding() &&
        !ShapeUtil::Compatible(old_root->shape(), new_shape)) {
      fusion_instruction->clear_sharding();
    }
    ABSL_RETURN_IF_ERROR(root->ReplaceOperandWithDifferentShape(i, new_root));
    *root->mutable_shape()->mutable_tuple_shapes(i) = new_shape;
    ABSL_RETURN_IF_ERROR(fusion_computation->RemoveInstruction(old_root));
    *fusion_instruction->mutable_shape()->mutable_tuple_shapes(i) = new_shape;
    // In XLA:GPU, multi-output fusion outputs are always accessed via
    // kGetTupleElement instructions in the parent computation.
    for (HloInstruction* user : fusion_instruction->users()) {
      if (user->tuple_index() == i) {
        ABSL_RETURN_IF_ERROR(WrapWithBitcast(user, new_shape));
      }
    }
    has_changed = true;
  }

  return has_changed;
}

absl::Status RecalculateOptimalTilingIfTiled(
    HloFusionInstruction* fusion_instruction,
    const se::DeviceDescription& device_info,
    HloCostAnalysis::ShapeSizeFunction shape_size,
    mlir::MLIRContext* mlir_context, tsl::thread::ThreadPool* thread_pool,
    MlirContextPool* mlir_context_pool) {
  if (!IsGenericTritonFusion(*fusion_instruction)) {
    return absl::OkStatus();
  }

  const DebugOptions& debug_options =
      fusion_instruction->GetModule()->config().debug_options();
  HloFusionAnalysisCache fusion_analysis_cache(device_info);
  GpuPerformanceModelWithIndexingAnalysis indexing_performance_model(
      &device_info, &fusion_analysis_cache, shape_size, mlir_context,
      /*use_experimental_tiling=*/true,
      debug_options
          .xla_gpu_experimental_enable_same_shape_multi_output_fusion(),
      mlir_context_pool);

  std::unique_ptr<HloFusionAdaptor> fusion_adaptor =
      HloFusionAdaptor::ForInstruction(fusion_instruction);
  tsl::Executor* tiling_search_executor =
      (thread_pool != nullptr && mlir_context_pool != nullptr)
          ? thread_pool->AsExecutor()
          : nullptr;
  ABSL_ASSIGN_OR_RETURN(TiledRunTimeDataOrError tiled_runtime_data_or_error,
                        indexing_performance_model
                            .TryFindBestTilingForFusionAsync(
                                *fusion_adaptor, tiling_search_executor)
                            .Await());

  // As of 2026-10-20, no pass that runs before this one can produce a
  // fusion that becomes untileable after hoisting bitcasts.
  //
  // Tiling propagation walks backwards from the root(s), so hoisting params
  // can never make a fusion untileable.
  //
  // The only earlier pass that can create bitcast root is
  // MultiOutputFusion, but that one doesn't create tiled fusions.
  TF_RET_CHECK(
      std::holds_alternative<TiledRunTimeData>(tiled_runtime_data_or_error))
      << "Hoisting bitcasts created an untileable fusion, recovery "
         "is not supported yet: "
      << fusion_instruction->ToString() << "; "
      << std::get<FusionDecision>(tiled_runtime_data_or_error).Explain();

  TiledRunTimeData tiled_runtime_data =
      std::get<TiledRunTimeData>(std::move(tiled_runtime_data_or_error));
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig backend_config,
                        fusion_instruction->backend_config<GpuBackendConfig>());
  *backend_config.mutable_fusion_backend_config()
       ->mutable_block_level_fusion_config() =
      tiled_runtime_data.block_level_parameters.ToBlockLevelFusionConfig();
  return fusion_instruction->set_backend_config(backend_config);
}

}  // namespace

absl::StatusOr<bool> HoistFusionBoundaryBitcasts::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool has_changed = false;

  for (HloComputation* computation :
       module->MakeComputationSorted(execution_threads)) {
    if (!computation->IsFusionComputation()) {
      continue;
    }
    HloFusionInstruction* fusion_instruction =
        Cast<HloFusionInstruction>(computation->FusionInstruction());
    if (!ShouldProcessFusion(*fusion_instruction)) {
      continue;
    }

    ABSL_ASSIGN_OR_RETURN(bool hoisted_params,
                          HoistParameterBitcasts(fusion_instruction));
    ABSL_ASSIGN_OR_RETURN(bool hoisted_roots,
                          HoistRootBitcasts(fusion_instruction));
    if (!hoisted_params && !hoisted_roots) {
      continue;
    }

    ABSL_RETURN_IF_ERROR(RecalculateOptimalTilingIfTiled(
        fusion_instruction, device_info_, shape_size_, mlir_context_,
        thread_pool_, mlir_context_pool_));
    has_changed = true;
  }

  if (has_changed && module->has_schedule()) {
    ABSL_RETURN_IF_ERROR(module->schedule().Update(execution_threads));
  }

  return has_changed;
}

}  // namespace xla::gpu
