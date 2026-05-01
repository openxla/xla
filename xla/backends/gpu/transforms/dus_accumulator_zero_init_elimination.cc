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

#include "xla/backends/gpu/transforms/dus_accumulator_zero_init_elimination.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/constant_value.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/value_range.h"
#include "xla/service/while_loop_unroller.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

bool IsBroadcastOfScalarConstant(const HloInstruction* h) {
  while (h->opcode() == HloOpcode::kBitcast ||
         h->opcode() == HloOpcode::kCopy ||
         h->opcode() == HloOpcode::kReshape) {
    h = h->operand(0);
  }
  if (h->opcode() != HloOpcode::kBroadcast) return false;
  const HloInstruction* src = hlo_query::StripCastLike(h->operand(0));
  return src->opcode() == HloOpcode::kConstant &&
         src->shape().dimensions().empty();
}

// Init value is irrelevant; soundness comes from the dead-input check.
std::optional<Shape> ClassifyReplaceableInit(const HloInstruction* init) {
  if (IsBroadcastOfScalarConstant(init)) {
    return init->shape();
  }
  if (init->opcode() == HloOpcode::kFusion && init->operand_count() == 0) {
    if (IsBroadcastOfScalarConstant(init->fused_expression_root())) {
      return init->shape();
    }
  }
  if (init->opcode() == HloOpcode::kGetTupleElement) {
    const HloInstruction* fusion = init->operand(0);
    if (fusion->opcode() == HloOpcode::kFusion &&
        fusion->operand_count() == 0) {
      const HloInstruction* root = fusion->fused_expression_root();
      if (root->opcode() == HloOpcode::kTuple &&
          IsBroadcastOfScalarConstant(root->operand(init->tuple_index()))) {
        return init->shape();
      }
    }
  }
  return std::nullopt;
}

struct WalkCtx {
  const WhileLoopConfig& cfg;
  int depth = 0;
  static constexpr int kMaxDepth = 8;
};

absl::StatusOr<bool> AllUsersKillBuffer(const HloInstruction* buffer,
                                        WalkCtx& ctx);
absl::StatusOr<bool> UserKillsBuffer(const HloInstruction* user,
                                     const HloInstruction* operand,
                                     WalkCtx& ctx);

// Read from the WhileLoopConfig annotation: upstream rewrites defeat
// structural IV detection.
// TODO: the annotation can outlive the IR shape it was derived from —
// upstream IV-rewriting passes should refresh or drop it.
Range LoopIterationRange(const WhileLoopConfig& cfg) {
  return Range{ConstantValue::GetSigned(0, /*bitwidth=*/64),
               ConstantValue::GetSigned(cfg.trip_count - 1, /*bitwidth=*/64),
               ConstantValue::GetSigned(1, /*bitwidth=*/64),
               /*is_linear=*/true};
}

// RecursivelyIdentifyRange handles the recursion into fused parameters, so
// this works inside or outside fusions.
std::optional<Range> ResolveIndexRangeAsFunctionOfIv(
    const HloInstruction* idx, const WhileLoopConfig& cfg) {
  Range loop_range = LoopIterationRange(cfg);
  absl::flat_hash_map<const HloInstruction*, Range> predefined;
  HloInstruction* body_param =
      cfg.while_instr->while_body()->parameter_instruction(0);
  for (HloInstruction* u : body_param->users()) {
    if (u->opcode() == HloOpcode::kGetTupleElement &&
        u->tuple_index() == cfg.induction_var_idx) {
      predefined[u] = loop_range;
    }
  }

  Range r = RecursivelyIdentifyRange(idx, predefined, nullptr);
  if (!r.IsBounded() || !r.IsStepKnown() || r.step()->GetSignedValue() == 0) {
    return std::nullopt;
  }
  return r;
}

// Generalizes AdvancedMatchShapeCoveringDynamicIndexInstruction to indices
// that are arithmetic expressions over fused parameters (e.g. descending DUS).
bool IsKillingDus(const HloInstruction* dus, const HloInstruction* operand,
                  const WhileLoopConfig& cfg) {
  if (dus->opcode() != HloOpcode::kDynamicUpdateSlice) return false;
  if (dus->operand(0) != operand) return false;

  const Shape& slice_shape = dus->operand(1)->shape();
  const Shape& input_shape = dus->operand(0)->shape();
  if (slice_shape.dimensions().size() != input_shape.dimensions().size()) {
    return false;
  }

  // Find the single dynamic dim; static dims must have start=0 and full size.
  std::optional<int64_t> dyn_dim_opt;
  std::optional<Range> dyn_dim_range;
  const int64_t num_dims = input_shape.dimensions().size();
  for (int64_t d = 0; d < num_dims; ++d) {
    const HloInstruction* idx = dus->operand(2 + d);
    std::optional<Range> r = ResolveIndexRangeAsFunctionOfIv(idx, cfg);
    if (r.has_value() && r->IsBounded() && r->IsSingleValue() &&
        r->min().GetSignedValue() == 0) {
      if (slice_shape.dimensions(d) != input_shape.dimensions(d)) {
        return false;
      }
      continue;
    }
    if (dyn_dim_opt.has_value()) return false;  // more than one dynamic dim
    if (!r.has_value()) return false;
    dyn_dim_opt = d;
    dyn_dim_range = r;
  }
  if (!dyn_dim_opt.has_value() || !dyn_dim_range.has_value()) return false;
  int64_t dyn_dim = *dyn_dim_opt;
  std::optional<Range> idx_range = dyn_dim_range;

  const int64_t dim_size = input_shape.dimensions(dyn_dim);
  const int64_t slice_size = slice_shape.dimensions(dyn_dim);
  if (slice_size <= 0 || slice_size > dim_size) return false;
  std::vector<bool> covered(dim_size, false);
  for (int64_t v = idx_range->min().GetSignedValue();
       v <= idx_range->max()->GetSignedValue();
       v += idx_range->step()->GetSignedValue()) {
    int64_t start =
        std::min<int64_t>(std::max<int64_t>(v, 0), dim_size - slice_size);
    for (int64_t i = start; i < start + slice_size; ++i) {
      if (i >= 0 && i < dim_size) covered[i] = true;
    }
  }
  for (bool c : covered) {
    if (!c) return false;
  }
  return true;
}

// TODO: scatter killing-write support; needs reasoning over scatter index
// expressions and update_window_dims.

absl::StatusOr<bool> FusionParamIsKilled(const HloInstruction* fusion,
                                         int64_t param_idx, WalkCtx& ctx) {
  if (++ctx.depth > WalkCtx::kMaxDepth) return false;
  HloComputation* fc = fusion->fused_instructions_computation();
  HloInstruction* fused_param = fc->parameter_instruction(param_idx);
  bool ok = false;
  TF_ASSIGN_OR_RETURN(ok, AllUsersKillBuffer(fused_param, ctx));
  --ctx.depth;
  return ok;
}

absl::StatusOr<bool> AllUsersKillBuffer(const HloInstruction* buffer,
                                        WalkCtx& ctx) {
  if (buffer->user_count() == 0) return true;  // trivially dead
  for (const HloInstruction* user : buffer->users()) {
    TF_ASSIGN_OR_RETURN(bool ok, UserKillsBuffer(user, buffer, ctx));
    if (!ok) return false;
  }
  return true;
}

absl::StatusOr<bool> UserKillsBuffer(const HloInstruction* user,
                                     const HloInstruction* operand,
                                     WalkCtx& ctx) {
  if (++ctx.depth > WalkCtx::kMaxDepth) {
    --ctx.depth;
    return false;
  }
  bool result = false;
  switch (user->opcode()) {
    case HloOpcode::kDynamicUpdateSlice:
      result = IsKillingDus(user, operand, ctx.cfg);
      break;

    case HloOpcode::kBitcast:
      // Bitcast is layout-only, so recurse to its users. kCopy intentionally
      // falls to default: it observes the init bytes we are eliding.
      result = AllUsersKillBuffer(user, ctx).value_or(false);
      break;

    case HloOpcode::kFusion: {
      int64_t op_idx = user->operand_index(operand);
      if (op_idx < 0) {
        result = false;
        break;
      }
      result = FusionParamIsKilled(user, op_idx, ctx).value_or(false);
      if (result) break;

      // DSF: in-place semantics live in output_to_operand_aliasing, not in
      // the fused body — walk the alias map instead.
      if (user->fusion_kind() != HloInstruction::FusionKind::kCustom) break;
      auto bcfg = user->backend_config<GpuBackendConfig>();
      if (!bcfg.ok()) break;
      const FusionBackendConfig& fc = bcfg->fusion_backend_config();
      if (fc.kind() != kCustomFusionKind || !fc.has_custom_fusion_config() ||
          fc.custom_fusion_config().name() !=
              kDynamicSliceFusionWithDynamicAddressComputationConfigName) {
        break;
      }
      for (const auto& alias : user->output_operand_aliasing()) {
        if (alias.second.first != op_idx) continue;
        if (!alias.second.second.empty()) continue;
        const ShapeIndex& output_idx = alias.first;
        const HloInstruction* fused_root = user->fused_expression_root();
        const HloInstruction* aliased;
        if (output_idx.empty()) {
          aliased = fused_root;
        } else if (fused_root->opcode() == HloOpcode::kTuple &&
                   output_idx.size() == 1 &&
                   output_idx[0] < fused_root->operand_count()) {
          aliased = fused_root->operand(output_idx[0]);
        } else {
          continue;
        }
        while (aliased->opcode() == HloOpcode::kBitcast) {
          aliased = aliased->operand(0);
        }
        if (aliased->opcode() != HloOpcode::kDynamicUpdateSlice) continue;
        // DUS's operand(0) must trace (through bitcasts) to
        // fused_param[op_idx].
        const HloInstruction* dus_target = aliased->operand(0);
        while (dus_target->opcode() == HloOpcode::kBitcast) {
          dus_target = dus_target->operand(0);
        }
        if (dus_target->opcode() != HloOpcode::kParameter ||
            dus_target->parameter_number() != op_idx) {
          continue;
        }
        if (IsKillingDus(aliased, dus_target, ctx.cfg)) {
          result = true;
          break;
        }
      }
      break;
    }

    case HloOpcode::kConditional: {
      // Scope: operand passed straight to a branch (not inside a tuple);
      // each receiving branch's parameter_0 must be killed.
      bool every_branch_kills = true;
      bool flows_into_some_branch = false;
      for (int b = 0; b < user->branch_count(); ++b) {
        const HloInstruction* branch_input = user->operand(b + 1);
        if (branch_input != operand) continue;
        flows_into_some_branch = true;
        HloComputation* branch = user->branch_computation(b);
        HloInstruction* branch_param = branch->parameter_instruction(0);
        TF_ASSIGN_OR_RETURN(bool branch_ok,
                            AllUsersKillBuffer(branch_param, ctx));
        if (!branch_ok) {
          every_branch_kills = false;
          break;
        }
      }
      result = flows_into_some_branch && every_branch_kills;
      break;
    }

    case HloOpcode::kTuple:
    case HloOpcode::kGetTupleElement:
      // Conservatively bail on tuple/GTE rather than tracing aliasing.
      result = false;
      break;

    default:
      result = false;  // anything else reads the buffer, so it's not killed
      break;
  }
  --ctx.depth;
  return result;
}

// Two checks: (1) body_param[slot] is overwritten before any read, and
// (2) body_root[slot] carries the writer's output, not a passthrough.
absl::StatusOr<bool> SlotIsDeadInput(int64_t slot, const WhileLoopConfig& cfg) {
  HloComputation* body = cfg.while_instr->while_body();
  HloInstruction* body_param = body->parameter_instruction(0);
  HloInstruction* body_root = body->root_instruction();
  if (body_root->opcode() != HloOpcode::kTuple) return false;

  HloInstruction* slot_gte = nullptr;
  for (HloInstruction* u : body_param->users()) {
    if (u->opcode() == HloOpcode::kGetTupleElement &&
        u->tuple_index() == slot) {
      if (slot_gte != nullptr) return false;  // multiple GTEs at slot
      slot_gte = u;
    }
  }
  if (slot_gte == nullptr) return true;  // never read; trivially dead

  WalkCtx ctx{cfg};
  TF_ASSIGN_OR_RETURN(bool all_kill, AllUsersKillBuffer(slot_gte, ctx));
  if (!all_kill) return false;

  // If the body passes slot through unchanged (root traces to slot_gte via
  // bitcast/copy), post-loop uses would see garbage if we elided the init.
  const HloInstruction* root_v = body_root->operand(slot);
  while (root_v->opcode() == HloOpcode::kBitcast ||
         root_v->opcode() == HloOpcode::kCopy) {
    root_v = root_v->operand(0);
  }
  if (root_v == slot_gte) return false;  // unmodified passthrough
  return true;
}

}  // namespace

absl::StatusOr<bool> DusAccumulatorZeroInitElimination::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!module->config()
           .debug_options()
           .xla_gpu_enable_dus_accumulator_zero_init_elimination()) {
    return false;
  }

  std::vector<HloInstruction*> whiles;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* ins : comp->instructions()) {
      if (ins->opcode() == HloOpcode::kWhile) whiles.push_back(ins);
    }
  }

  bool changed = false;
  for (HloInstruction* while_op : whiles) {
    if (while_op->has_sharding()) continue;
    auto loop_cfg = while_op->backend_config<WhileLoopBackendConfig>();
    if (!loop_cfg.ok() || !loop_cfg->has_known_trip_count()) continue;
    int64_t trip_count = loop_cfg->known_trip_count().n();
    if (trip_count <= 0) continue;

    int64_t iv_tuple_idx;
    if (loop_cfg->has_known_induction_variable()) {
      iv_tuple_idx = loop_cfg->known_induction_variable().tuple_index();
    } else {
      std::optional<int64_t> iv_idx_opt = GetLoopInductionVarTupleIdx(while_op);
      if (!iv_idx_opt.has_value()) continue;
      iv_tuple_idx = *iv_idx_opt;
    }

    if (!loop_cfg->has_known_init_step()) continue;
    if (loop_cfg->known_init_step().init() != 0 ||
        loop_cfg->known_init_step().step() != 1) {
      continue;
    }

    WhileLoopConfig cfg{/*while_instr=*/while_op,
                        /*init=*/0,
                        /*trip_count=*/trip_count,
                        /*induction_var_idx=*/iv_tuple_idx};

    HloInstruction* init_tuple = while_op->mutable_operand(0);
    if (init_tuple->opcode() != HloOpcode::kTuple) continue;

    int64_t n_slots = init_tuple->operand_count();
    for (int64_t slot = 0; slot < n_slots; ++slot) {
      if (slot == cfg.induction_var_idx) continue;

      HloInstruction* init = init_tuple->mutable_operand(slot);
      std::optional<Shape> alloc_shape_opt = ClassifyReplaceableInit(init);
      if (!alloc_shape_opt.has_value()) continue;
      if (init->has_sharding()) continue;
      if (init->user_count() != 1) continue;  // init must feed only this while
      const Shape& alloc_shape = *alloc_shape_opt;
      if (alloc_shape.dimensions_size() == 0) continue;
      if (alloc_shape.dimensions(0) != cfg.trip_count) continue;

      TF_ASSIGN_OR_RETURN(bool dead, SlotIsDeadInput(slot, cfg));
      if (!dead) continue;

      HloInstruction* alloc =
          while_op->parent()->AddInstruction(HloInstruction::CreateCustomCall(
              alloc_shape, /*operands=*/{}, "AllocateBuffer"));
      alloc->set_metadata(init->metadata());
      alloc->set_frontend_attributes(init->frontend_attributes());
      alloc->set_statistics_viz(init->statistics_viz());
      TF_RETURN_IF_ERROR(init_tuple->ReplaceOperandWith(slot, alloc));
      changed = true;
    }
  }

  // TODO: extend to multi-iter coverage analysis (a buffer may be killed
  // across iters 0..k-1 jointly even if iter 0 alone reads stale slots).
  return changed;
}

}  // namespace gpu
}  // namespace xla
