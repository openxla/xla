/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/rng_expander.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/builder/lib/prng.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/transforms/expanders/op_expander_pass.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

absl::StatusOr<int64_t> ValidateAndGet32bitUnits(const HloInstruction* rng) {
  const Shape& s = rng->shape();
  if (rng->opcode() != HloOpcode::kRng || !s.IsArray() ||
      rng->operand_count() < 2) {
    return absl::InvalidArgumentError("Invalid RNG instruction.");
  }
  if (s.is_dynamic()) {
    return absl::UnimplementedError("Dynamic RNG is not supported.");
  }
  const PrimitiveType ty = s.element_type();
  const bool fp = primitive_util::IsFloatingPointType(ty);
  const RandomDistribution dist = rng->random_distribution();
  const int64_t bw = primitive_util::BitWidth(ty);
  if ((!fp && !primitive_util::IsIntegralType(ty) && ty != PRED) ||
      (dist != RNG_UNIFORM && dist != RNG_NORMAL) ||
      (dist == RNG_NORMAL && !fp) || bw <= 0 || bw > 64 ||
      (bw >= 32 && bw != 32 && bw != 64)) {
    return absl::InvalidArgumentError("Unsupported RNG type or distribution.");
  }
  for (int i = 0; i < 2; ++i) {
    const Shape& os = rng->operand(i)->shape();
    const PrimitiveType oty = os.element_type();
    if (!os.IsArray() || !ShapeUtil::IsScalar(os) ||
        (oty != ty && !(fp && primitive_util::IsFloatingPointType(oty)))) {
      return absl::InvalidArgumentError("Invalid RNG operand.");
    }
  }
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  constexpr int64_t kMaxPhiloxUnits = (kMax / 16) * 4;
  const int64_t words = bw == 64 ? 2 : 1, max_elems = kMaxPhiloxUnits / words;
  int64_t n = 1, bytes = words * 4, max_d = 0;
  bool has_even = false;
  for (int64_t d : s.dimensions()) {
    if (d < 0) {
      return absl::InvalidArgumentError("Negative RNG dimension.");
    }
    if (d == 0) {
      n = bytes = 0;
    } else if (bytes > 0 && bytes > kMax / d) {
      return absl::InvalidArgumentError("RNG shape overflows 64-bit size.");
    } else if (bytes > 0) {
      bytes *= d;
      n *= d;
    }
    has_even |= (d % 2 == 0);
    max_d = std::max(max_d, d);
  }
  if (n > max_elems ||
      (dist == RNG_NORMAL && n > 0 && !s.dimensions().empty() && !has_even &&
       (max_d >= max_elems || (n / max_d) > max_elems / (max_d + 1)))) {
    return absl::InvalidArgumentError("RNG shape exceeds 64-bit Philox limit.");
  }
  return n * words;
}

HloInstruction* SetAttrs(HloInstruction* inst, const HloInstruction* rng,
                         bool same_shape = false) {
  if (same_shape) {
    rng->SetupDerivedInstruction(inst);
    if (rng->has_frontend_attributes()) {
      inst->set_frontend_attributes(rng->frontend_attributes());
    }
    return inst;
  }
  inst->set_metadata(rng->metadata());
  if (rng->has_sharding()) {
    const HloSharding& s = rng->sharding();
    if (!s.IsTuple() &&
        (s.IsReplicatedOrSingleDevice() || s.IsManual() || s.IsUnknown()) &&
        (!s.UseNamedShardingLeaf() ||
         s.named_sharding().dim_shardings().empty())) {
      inst->set_sharding(s);
    }
  }
  return inst;
}

Shape SanitizeShape(const Shape& orig, PrimitiveType ty,
                    bool keep_bits = false) {
  Shape s = ShapeUtil::ChangeElementType(orig, ty);
  if (s.has_layout()) {
    const int64_t bits = keep_bits ? orig.layout().element_size_in_bits() : 0;
    Layout* l = s.mutable_layout();
    l->clear_tiles();
    l->clear_split_configs();
    l->clear_physical_shape();
    l->set_tail_padding_alignment_in_elements(1).set_memory_space(0);
    if (!s.dimensions().empty() &&
        l->minor_to_major().size() != s.dimensions().size()) {
      LayoutUtil::SetToDefaultLayout(&s);
    }
    s.mutable_layout()->set_element_size_in_bits(bits);
  }
  return s;
}

HloInstruction* MaybeCopy(HloInstruction* op, const Shape& dst,
                          const HloInstruction* rng, bool same_shape = false) {
  return (dst.has_layout() && op->shape().layout() != dst.layout())
             ? SetAttrs(
                   op->parent()->AddInstruction(
                       HloInstruction::CreateUnary(dst, HloOpcode::kCopy, op)),
                   rng, same_shape)
             : op;
}

HloInstruction* ConvertOperand(HloInstruction* op, PrimitiveType dst,
                               const HloInstruction* rng, bool strip = false) {
  const bool same = op->shape().element_type() == dst;
  if (same && !strip) {
    return op;
  }
  const Shape clean = SanitizeShape(op->shape(), op->shape().element_type(),
                                    /*keep_bits=*/!strip);
  op = MaybeCopy(op, clean, rng);
  return same ? op
              : SetAttrs(
                    op->parent()->AddInstruction(HloInstruction::CreateConvert(
                        SanitizeShape(clean, dst), op)),
                    rng);
}

absl::StatusOr<HloInstruction*> ConvertNarrowRngToWideRng(HloInstruction* rng) {
  const Shape orig = rng->shape();
  const PrimitiveType ty = orig.element_type();
  const PrimitiveType wide =
      primitive_util::IsFloatingPointType(ty)
          ? F32
          : (primitive_util::IsSignedIntegralType(ty) ? S32 : U32);
  HloComputation* c = rng->parent();
  std::vector<HloInstruction*> ops;
  if (ty == PRED) {
    ops = {SetAttrs(MakeR0ConstantHlo<uint32_t>(c, 0), rng),
           SetAttrs(MakeR0ConstantHlo<uint32_t>(c, 2), rng)};
  } else {
    for (HloInstruction* op : rng->operands()) {
      ops.push_back(op->shape().IsArray() ? ConvertOperand(op, wide, rng) : op);
    }
  }
  HloCloneContext ctx(c->parent());
  HloInstruction* new_rng = c->AddInstruction(
      rng->CloneWithNewOperands(SanitizeShape(orig, wide), ops, &ctx));
  ABSL_RETURN_IF_ERROR(new_rng->CopyAllControlDepsFrom(rng));

  const Shape ns = SanitizeShape(orig, ty, /*keep_bits=*/true);
  auto add = [&](auto inst, bool same = true) {
    return SetAttrs(c->AddInstruction(std::move(inst)), rng, same);
  };
  HloInstruction* out = add(HloInstruction::CreateConvert(ns, new_rng));
  if (ty == PRED) {
    const Shape p0 = ShapeUtil::MakeShape(PRED, {});
    HloInstruction* lo =
        ConvertOperand(rng->mutable_operand(0), PRED, rng, true);
    HloInstruction* open = add(
        HloInstruction::CreateBinary(
            p0, HloOpcode::kAnd,
            add(HloInstruction::CreateUnary(p0, HloOpcode::kNot, lo), false),
            ConvertOperand(rng->mutable_operand(1), PRED, rng, true)),
        false);
    out = add(HloInstruction::CreateTernary(
        ns, HloOpcode::kSelect,
        add(HloInstruction::CreateBroadcast(ns, open, {})), out,
        add(HloInstruction::CreateBroadcast(ns, lo, {}))));
  }
  ABSL_RETURN_IF_ERROR(
      rng->ReplaceAllUsesWith(MaybeCopy(out, orig, rng, true)));
  ABSL_RETURN_IF_ERROR(rng->DropAllControlDeps());
  ABSL_RETURN_IF_ERROR(c->RemoveInstruction(rng));
  return new_rng;
}

// Builds a 64-bit key by mixing a per-op seed with the replica and partition
// IDs.
//
// The key is created as:
// (Seed ^ ((OpID * PRIME_CONSTANT) >> 32)) ^ ReplicaId ^ (PartitionId << 32)
//
// The rationale behind this: We want to generate a unique key for each RNG
// call that is stable as long as the HLO Graph does not change, and produces
// different keys on different devices (matching TPU behavior).
// With parallel compilation, we can no longer rely on a global counter, so we
// instead explicitly mix the op id with the replica and partition id.
//
// The multiplication and shift is a way to diffuse the individual components
// into the key and increase the entropy (as opposed to e.g. just XOR'ing all
// bits) - otherwise, ReplicaId and PartitionId would often clash.
//
// Params:
//   builder: The XlaBuilder to use for the key generation.
//   op_seed: The pre-calculated seed (OpID * PRIME_CONSTANT) >> 32) as
//     constant.
//   config: The HloModuleConfig to use for the key generation.
//
// Returns:
//   The 64-bit key to use for the RNG.
XlaOp BuildPhiloxKey(XlaBuilder* builder, XlaOp op_seed,
                     const HloModuleConfig& config) {
  FrontendAttributes attributes;
  attributes.mutable_map()->emplace(kXlaCseSafeZeroOperandAttr, "true");
  builder->SetFrontendAttributes(attributes);
  XlaOp seed_u64 =
      CustomCall(builder, "GetRngSeed", {}, ShapeUtil::MakeShape(U64, {}));
  builder->ClearFrontendAttributes();

  XlaOp replica_u64;
  bool has_replica = config.replica_count() > 1;
  if (has_replica) {
    XlaOp replica_id = ReplicaId(builder);
    replica_u64 = ConvertElementType(replica_id, U64);
  }

  XlaOp partition_u64;
  bool has_partition =
      !config.use_spmd_partitioning() && config.num_partitions() > 1;
  if (has_partition) {
    XlaOp partition_id = internal::XlaBuilderFriend::BuildPartitionId(
        builder, ShapeUtil::MakeShape(U32, {}));
    partition_u64 = ConvertElementType(partition_id, U64);
    XlaOp shift_32 = ConstantR0<uint64_t>(builder, 32);
    partition_u64 = ShiftLeft(partition_u64, shift_32);
  }

  XlaOp mixed_seed = Xor(seed_u64, op_seed);
  if (has_replica && has_partition) {
    XlaOp replica_and_partition = Xor(replica_u64, partition_u64);
    return Xor(mixed_seed, replica_and_partition);
  }
  if (has_replica) {
    return Xor(mixed_seed, replica_u64);
  }
  if (has_partition) {
    return Xor(mixed_seed, partition_u64);
  }
  return mixed_seed;
}

absl::StatusOr<HloComputation*> GetComputationForRng(HloInstruction* rng,
                                                     const Shape& op0_shape,
                                                     const Shape& op1_shape) {
  XlaBuilder builder("rng");
  const Shape u64_shape = ShapeUtil::MakeShape(xla::U64, {});
  const Shape u128_shape = ShapeUtil::MakeShape(xla::U64, {2});
  const Shape& result_shape = rng->shape();
  const PrimitiveType ty = result_shape.element_type();
  const Shape gen_shape = ShapeUtil::MakeShape(ty, result_shape.dimensions());
  const Shape p_shape = ShapeUtil::MakeShape(ty, {});

  XlaOp op_seed = Parameter(&builder, 0, u64_shape, "op_seed");
  XlaOp state = Parameter(&builder, 1, u128_shape, "state");
  XlaOp a_or_mean = Parameter(&builder, 2, p_shape, "a_or_mean");
  XlaOp b_or_sigma = Parameter(&builder, 3, p_shape, "b_or_sigma");

  XlaOp key = BuildPhiloxKey(&builder, op_seed, rng->GetModule()->config());

  auto generator = [](xla::XlaOp key, xla::XlaOp state,
                      const xla::Shape& shape) {
    return PhiloxBitGenerator(key, state, shape);
  };

  XlaOp result;
  if (rng->random_distribution() == RNG_NORMAL) {
    result =
        NormalFloatingPointDistribution(key, state, generator, gen_shape).value;
    // Transform standard normal distribution to normal distribution with the
    // given mean and standard deviation.
    result = a_or_mean + (b_or_sigma * result);
  } else if (rng->random_distribution() != RNG_UNIFORM) {
    return absl::InvalidArgumentError("Unsupported RNG distribution.");
  } else if (primitive_util::IsFloatingPointType(ty)) {
    result = UniformFloatingPointDistribution(key, state, generator, a_or_mean,
                                              b_or_sigma, gen_shape)
                 .value;
  } else {
    result = UniformIntDistribution(key, state, generator, a_or_mean,
                                    b_or_sigma, gen_shape)
                 .value;
  }

  ABSL_ASSIGN_OR_RETURN(XlaComputation xla_computation, builder.Build());
  ABSL_ASSIGN_OR_RETURN(
      HloComputation * comp,
      XlaComputationToHloComputation(xla_computation, rng->GetModule()));
  for (int i : {0, 1}) {
    const Shape& dst = i == 0 ? op0_shape : op1_shape;
    HloInstruction* p = comp->parameter_instruction(i + 2);
    if (dst.has_layout() && p->shape().layout() != dst.layout()) {
      const Shape clean = p->shape();
      std::vector<HloInstruction*> users(p->users().begin(), p->users().end());
      *p->mutable_shape() = dst;
      if (!users.empty()) {
        ABSL_RETURN_IF_ERROR(p->ReplaceUsesWith(
            users, comp->AddInstruction(HloInstruction::CreateUnary(
                       clean, HloOpcode::kCopy, p))));
      }
    }
  }
  if (HloInstruction* r = comp->root_instruction();
      result_shape.has_layout() &&
      r->shape().layout() != result_shape.layout()) {
    comp->set_root_instruction(comp->AddInstruction(HloInstruction::CreateUnary(
                                   result_shape, HloOpcode::kCopy, r)),
                               /*accept_different_shape=*/true);
  }
  if (absl::string_view th = rng->parent()->execution_thread();
      th != HloInstruction::kMainExecutionThread) {
    comp->SetExecutionThread(th);
  }
  return comp;
}

}  // namespace

absl::StatusOr<bool> RngExpander::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  expanded_rng_instructions_.clear();
  auto changed = OpExpanderPass::RunImpl(module, execution_threads);
  expanded_rng_instructions_.clear();
  return changed;
}

bool RngExpander::InstructionMatchesPattern(HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kRng;
}

absl::StatusOr<HloInstruction*> RngExpander::ExpandInstruction(
    HloInstruction* rng) {
  VLOG(2) << "Expand rng instruction " << rng->ToString();
  ABSL_ASSIGN_OR_RETURN(const int64_t num_units, ValidateAndGet32bitUnits(rng));
  if (primitive_util::BitWidth(rng->shape().element_type()) < 32) {
    ABSL_ASSIGN_OR_RETURN(rng, ConvertNarrowRngToWideRng(rng));
  }
  const PrimitiveType ty = rng->shape().element_type();
  HloInstruction* op0 = ConvertOperand(rng->mutable_operand(0), ty, rng);
  HloInstruction* op1 = ConvertOperand(rng->mutable_operand(1), ty, rng);
  HloComputation* computation = rng->parent();
  const HloModuleConfig& cfg = rng->GetModule()->config();
  RngCacheKey key(rng->random_distribution(), rng->shape(), op0->shape(),
                  op1->shape(), rng->GetModule(),
                  std::string(computation->execution_thread()),
                  cfg.replica_count() > 1,
                  !cfg.use_spmd_partitioning() && cfg.num_partitions() > 1);
  HloComputation* rng_computation = nullptr;
  if (auto it = expanded_rng_instructions_.find(key);
      it != expanded_rng_instructions_.end()) {
    rng_computation = it->second;
  } else {
    ABSL_ASSIGN_OR_RETURN(
        rng_computation, GetComputationForRng(rng, op0->shape(), op1->shape()));
    expanded_rng_instructions_.emplace(std::move(key), rng_computation);
  }
  uint64_t op_seed =
      (static_cast<uint64_t>(rng->unique_id()) * 0x9E3779B97F4A7C15ULL) >> 32;
  HloInstruction* op_seed_hlo =
      SetAttrs(MakeR0ConstantHlo<uint64_t>(computation, op_seed), rng);
  HloInstruction* state = SetAttrs(
      computation->AddInstruction(HloInstruction::CreateRngGetAndUpdateState(
          ShapeUtil::MakeShape(xla::U64, {2}), num_units)),
      rng);
  VLOG(2) << "Rng op_seed " << op_seed_hlo->ToString();
  VLOG(2) << "Rng state " << state->ToString();
  HloInstruction* new_rng = SetAttrs(
      computation->AddInstruction(HloInstruction::CreateCall(
          rng->shape(), {op_seed_hlo, state, op0, op1}, rng_computation)),
      rng, /*same_shape=*/true);
  ABSL_RETURN_IF_ERROR(new_rng->CopyAllControlDepsFrom(rng));
  ABSL_RETURN_IF_ERROR(rng->ReplaceAllUsesWith(new_rng));
  ABSL_RETURN_IF_ERROR(rng->DropAllControlDeps());
  ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(rng));
  return nullptr;
}

}  // namespace xla
