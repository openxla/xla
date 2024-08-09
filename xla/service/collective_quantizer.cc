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

#include "xla/service/collective_quantizer.h"

#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

namespace m = match;

// Describes the ops of a quantization, dequantization or plain type conversion
// subgraph.
struct ConversionSubgraph {
  HloInstruction* convert;
  HloInstruction* binary = nullptr;
  HloInstruction* clamp;
  HloInstruction* scale_bcast;
  std::vector<HloInstruction*> unaries;
};

// Matches a broadcast of a scalar operand.
template <typename... Args>
auto ScalarBroadcast(Args... args) {
  return m::Broadcast(args...).WithPredicate([](const HloInstruction* instr) {
    return ShapeUtil::IsScalar(instr->operand(0)->shape());
  });
}

// Matches a bitcast that preserves the element type of the operand.
auto BitcastPreservesElementType() {
  return m::Bitcast().WithPredicate([](const HloInstruction* instr) {
    return ShapeUtil::SameElementType(instr->shape(),
                                      instr->operand(0)->shape());
  });
}

// Matches a type conversion to a type with a smaller byte size than that of the
// operand.
auto ConvertToNarrowerType() {
  auto converts_to_narrower_type = [](const HloInstruction* instr) -> bool {
    return ShapeUtil::ByteSizeOfPrimitiveType(instr->shape().element_type()) <
           ShapeUtil::ByteSizeOfPrimitiveType(
               instr->operand(0)->shape().element_type());
  };
  return m::Convert().WithPredicate(converts_to_narrower_type);
}

// Matches a type conversion to a type with a larger byte size than that of the
// operand.
auto ConvertToWiderType() {
  auto converts_to_wider_type = [](const HloInstruction* instr) -> bool {
    return ShapeUtil::ByteSizeOfPrimitiveType(instr->shape().element_type()) >
           ShapeUtil::ByteSizeOfPrimitiveType(
               instr->operand(0)->shape().element_type());
  };
  return m::Convert().WithPredicate(converts_to_wider_type);
}

bool IsSupportedCollective(HloInstruction* instr) {
  return instr->operand_count() == 1 &&
         (instr->opcode() == HloOpcode::kAllGather ||
          instr->opcode() == HloOpcode::kAllToAll ||
          instr->opcode() == HloOpcode::kCollectiveBroadcast ||
          instr->opcode() == HloOpcode::kCollectivePermute);
}

// Sequentially applies the ops in unaries to the output of instr.
HloInstruction* ApplyUnary(HloInstruction* instr,
                           const std::vector<HloInstruction*>& unaries) {
  for (HloInstruction* unary_op : unaries) {
    instr = instr->AddInstruction(unary_op->CloneWithNewOperands(
        ShapeUtil::MakeShapeWithDenseLayout(
            instr->shape().element_type(), unary_op->shape().dimensions(),
            unary_op->shape().layout().minor_to_major()),
        {instr}));
  }
  return instr;
}

// Recursively collects unary, divide, or multiply operands of instr until a
// conversion to a wider type is reached. Returns an empty vector when no
// conversion is reached.
std::vector<HloInstruction*> FindDequantizationSubgraphRecursive(
    HloInstruction* instr, absl::flat_hash_set<int>& visited_instrs,
    std::vector<HloInstruction*> subgraph) {
  // Avoid visiting the same instruction more than once.
  if (!visited_instrs.emplace(instr->unique_id()).second) {
    return {};
  }

  subgraph.emplace_back(instr);
  if (Match(instr, ConvertToWiderType())) {
    return subgraph;
  }
  if (instr->operand_count() == 1 || instr->opcode() == HloOpcode::kDivide) {
    return FindDequantizationSubgraphRecursive(instr->mutable_operand(0),
                                               visited_instrs, subgraph);
  } else if (instr->opcode() == HloOpcode::kMultiply) {
    for (HloInstruction* operand : instr->unique_operands()) {
      auto binary_subgraph = FindDequantizationSubgraphRecursive(
          operand, visited_instrs, subgraph);
      if (!binary_subgraph.empty()) {
        return binary_subgraph;
      }
    }
  }
  return {};
}

// Returns true iff instr describes a dequantization, i.e. a multiplication or
// division by a broadcasted scalar operating on a type conversion, or a plain
// type conversion to a wider type. Unary bitcast, copy, reshape or slice ops
// may follow the dequantization or type conversion.
std::optional<ConversionSubgraph> IsSupportedDequantization(
    HloInstruction* instr) {
  ConversionSubgraph subgraph;
  absl::flat_hash_set<int> visited_instrs;
  std::vector<HloInstruction*> candidate_subgraph =
      FindDequantizationSubgraphRecursive(instr, visited_instrs,
                                          std::vector<HloInstruction*>{});
  std::reverse(candidate_subgraph.begin(), candidate_subgraph.end());

  // In the dequantization case, the type conversion is followed by a
  // multiplication or division by a broadcasted scalar.
  if (candidate_subgraph.size() > 1 &&
      (Match(
           candidate_subgraph[1],
           m::MultiplyAnyOrder(&subgraph.binary, m::Convert(&subgraph.convert),
                               ScalarBroadcast(&subgraph.scale_bcast))) ||
       Match(candidate_subgraph[1],
             m::Divide(&subgraph.binary, m::Convert(&subgraph.convert),
                       ScalarBroadcast(&subgraph.scale_bcast))))) {
    subgraph.unaries = {candidate_subgraph.begin() + 2,
                        candidate_subgraph.end()};
  } else if (candidate_subgraph.size() > 0 &&
             Match(candidate_subgraph[0], m::Convert(&subgraph.convert))) {
    subgraph.unaries = {candidate_subgraph.begin() + 1,
                        candidate_subgraph.end()};
  } else {
    VLOG(5) << "Did not find type conversion or dequantization pattern.";
    return std::nullopt;
  }

  // The collected unary ops between dequantization/type conversion and
  // collective may only include bitcast, copy, reshape and slice instructions.
  for (HloInstruction* unary : subgraph.unaries) {
    if (!Match(unary, m::AnyOf<HloInstruction>(m::Bitcast(), m::Copy(),
                                               m::Reshape(), m::Slice()))) {
      VLOG(5) << "Unexpected instruction in unary ops.";
      return std::nullopt;
    }
  }
  return std::make_optional<ConversionSubgraph>(subgraph);
}

// Returns true iff instr describes a quantization, i.e. a multiplication or
// division by a broadcasted scalar followed by a clamp and a type conversion,
// or a plain type conversion to a narrower type. Unary bitcast, copy, reshape
// or slice ops with one user may precede the quantization or type conversion.
std::optional<ConversionSubgraph> IsSupportedQuantization(
    HloInstruction* instr) {
  ConversionSubgraph subgraph;
  std::vector<HloInstruction*> ops;
  while (instr->user_count() <= 1) {
    if (Match(instr, m::AnyOf<HloInstruction>(
                         BitcastPreservesElementType(), m::Copy(), m::Reshape(),
                         m::Slice(), m::Multiply(), m::Divide(), m::Clamp()))) {
      if (instr->user_count() > 0) {
        ops.emplace_back(instr);
        instr = instr->users()[0];
        continue;
      }
      break;
    }

    if (Match(instr, ConvertToNarrowerType())) {
      ops.emplace_back(instr);
      break;
    }
    VLOG(5) << "Unsupported instruction.";
    return std::nullopt;
  }

  // In the quantization case, the type conversion is preceded by a
  // multiplication or division by a broadcasted scalar and a clamp instruction.
  if (ops.size() > 2 &&
      (Match(ops.back(),
             m::Convert(&subgraph.convert,
                        m::Clamp(&subgraph.clamp, ScalarBroadcast(),
                                 m::MultiplyAnyOrder(
                                     &subgraph.binary, m::Op(),
                                     ScalarBroadcast(&subgraph.scale_bcast)),
                                 ScalarBroadcast()))) ||
       Match(ops.back(),
             m::Convert(
                 &subgraph.convert,
                 m::Clamp(&subgraph.clamp, ScalarBroadcast(),
                          m::Divide(&subgraph.binary, m::Op(),
                                    ScalarBroadcast(&subgraph.scale_bcast)),
                          ScalarBroadcast()))))) {
    subgraph.unaries = {ops.begin(), ops.end() - 3};
  } else if (ops.size() > 0 &&
             Match(ops.back(), m::Convert(&subgraph.convert))) {
    subgraph.unaries = {ops.begin(), ops.end() - 1};
  } else {
    VLOG(5) << "Did not find type conversion or quantization pattern.";
    return std::nullopt;
  }

  // The collected unary ops between collective and quantization/type conversion
  // may only include bitcast, copy, reshape and slice instructions.
  for (HloInstruction* unary : subgraph.unaries) {
    if (!Match(unary, m::AnyOf<HloInstruction>(m::Bitcast(), m::Copy(),
                                               m::Reshape(), m::Slice()))) {
      VLOG(5) << "Unexpected instruction in unary ops.";
      return std::nullopt;
    }
  }
  return std::make_optional<ConversionSubgraph>(subgraph);
}

absl::Status MatchDequantization(HloInstruction* instr, bool* changed) {
  std::optional<ConversionSubgraph> subgraph =
      IsSupportedDequantization(instr->mutable_operand(0));

  if (subgraph.has_value()) {
    HloInstruction* new_coll_operand = subgraph->convert->mutable_operand(0);

    // Insert the collected unary ops ahead of the new collective.
    new_coll_operand = ApplyUnary(new_coll_operand, subgraph->unaries);

    // Move the collective before the conversion to the wider type.
    Shape new_coll_shape = ShapeUtil::ChangeElementType(
        instr->shape(), new_coll_operand->shape().element_type());
    HloInstruction* new_collective = instr->AddInstruction(
        instr->CloneWithNewOperands(new_coll_shape, {new_coll_operand}));
    Shape new_convert_shape = ShapeUtil::ChangeElementType(
        new_collective->shape(), subgraph->convert->shape().element_type());
    HloInstruction* new_convert =
        instr->AddInstruction(subgraph->convert->CloneWithNewOperands(
            new_convert_shape, {new_collective}));

    HloInstruction* new_binary;
    // When there is a dequantization, insert the scale ops.
    if (subgraph->binary) {
      HloInstruction* new_scale_bcast = instr->AddInstruction(
          subgraph->scale_bcast->CloneWithNewShape(new_convert->shape()));
      new_binary = instr->AddInstruction(subgraph->binary->CloneWithNewOperands(
          new_convert->shape(), {new_convert, new_scale_bcast}));
    }

    TF_RETURN_IF_ERROR(
        instr->ReplaceAllUsesWith(subgraph->binary ? new_binary : new_convert));

    *changed = true;
    VLOG(5) << "Quantized collective " << new_collective->ToShortString();
  }
  return absl::OkStatus();
}

absl::Status MatchQuantization(HloInstruction* instr, bool* changed) {
  std::optional<ConversionSubgraph> subgraph;
  if (instr->user_count() == 1) {
    subgraph = IsSupportedQuantization(instr->users()[0]);
  }

  if (subgraph.has_value()) {
    HloInstruction* coll_operand = instr->mutable_operand(0);

    HloInstruction *new_binary, *new_clamp;
    // When there is a quantization, insert the scale and clamp ops.
    if (subgraph->binary) {
      HloInstruction* new_scale_bcast = instr->AddInstruction(
          subgraph->scale_bcast->CloneWithNewShape(coll_operand->shape()));
      new_binary = instr->AddInstruction(subgraph->binary->CloneWithNewOperands(
          coll_operand->shape(), {coll_operand, new_scale_bcast}));
      HloInstruction* new_clamp_lower =
          instr->AddInstruction(subgraph->clamp->operand(0)->CloneWithNewShape(
              coll_operand->shape()));
      HloInstruction* new_clamp_upper =
          instr->AddInstruction(subgraph->clamp->operand(2)->CloneWithNewShape(
              coll_operand->shape()));
      new_clamp = instr->AddInstruction(subgraph->clamp->CloneWithNewOperands(
          coll_operand->shape(),
          {new_clamp_lower, new_binary, new_clamp_upper}));
    }

    // Move the collective past the conversion to the narrow type.
    Shape new_convert_shape = ShapeUtil::ChangeElementType(
        instr->operand(0)->shape(), subgraph->convert->shape().element_type());
    HloInstruction* new_convert =
        instr->AddInstruction(subgraph->convert->CloneWithNewOperands(
            new_convert_shape, {subgraph->binary ? new_clamp : coll_operand}));
    Shape new_collective_shape = ShapeUtil::ChangeElementType(
        instr->shape(), subgraph->convert->shape().element_type());
    HloInstruction* new_collective = instr->AddInstruction(
        instr->CloneWithNewOperands(new_collective_shape, {new_convert}));

    // Insert the collected unary ops after the new collective.
    new_collective = ApplyUnary(new_collective, subgraph->unaries);
    TF_RETURN_IF_ERROR(subgraph->convert->ReplaceAllUsesWith(new_collective));

    *changed = true;
    VLOG(5) << "Quantized collective " << new_collective->ToShortString();
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> CollectiveQuantizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (IsSupportedCollective(instr)) {
        TF_RETURN_IF_ERROR(MatchDequantization(instr, &changed));
        TF_RETURN_IF_ERROR(MatchQuantization(instr, &changed));
      }
    }
  }

  return changed;
}

}  // namespace xla
