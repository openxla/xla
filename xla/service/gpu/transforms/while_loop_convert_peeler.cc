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

#include "xla/service/gpu/transforms/while_loop_convert_peeler.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal_util.h"
#include "xla/service/value_range.h"

namespace xla::gpu {

std::string WhileLoopConvertPeeler::BufferInfo::ToString(int indent) const {
  std::stringstream ss;
  ss << std::string(indent, ' ') << "BufferInfo {\n";
  ss << std::string(indent + 2, ' ')
     << "body_buffer: " << (body_buffer ? body_buffer->ToString() : "nullptr")
     << "\n";
  ss << std::string(indent + 2, ' ') << "body_convert: "
     << (body_convert ? body_convert->ToString() : "nullptr") << "\n";
  ss << std::string(indent + 2, ' ')
     << "body_root: " << (body_root ? body_root->ToString() : "nullptr")
     << "\n";
  ss << std::string(indent + 2, ' ') << "dynamic_index_instruction: "
     << (dynamic_index_instruction ? dynamic_index_instruction->ToString()
                                   : "nullptr")
     << "\n";
  ss << std::string(indent + 2, ' ')
     << "while_tuple_buffer_idx: " << while_tuple_buffer_idx << "\n";
  ss << std::string(indent + 2, ' ') << "while_gte_user_after_loop: "
     << (while_gte_user_after_loop ? while_gte_user_after_loop->ToString()
                                   : "nullptr")
     << "\n";
  ss << std::string(indent, ' ') << "}";
  return ss.str();
}

std::string WhileLoopConvertPeeler::ConvertInfo::ToString(int indent) const {
  std::stringstream ss;
  ss << std::string(indent, ' ') << "ConvertInfo {\n";
  ss << std::string(indent + 2, ' ') << "buffer_infos: "
     << absl::StrJoin(buffer_infos, ",",
                      [indent](std::string* s, const BufferInfo& bufferInfo) {
                        s->append(bufferInfo.ToString(indent + 2));
                      })
     << "\n";
  ss << std::string(indent + 2, ' ')
     << "new_while_shape: " << new_while_shape.ToString() << "\n";
  ss << std::string(indent, ' ') << "}";
  return ss.str();
}

WhileLoopConvertPeeler::ConvertInfoMap
WhileLoopConvertPeeler::CollectConvertInfos(
    HloModule* module,
    absl::flat_hash_map<const HloInstruction*, Range>& known_ranges,
    absl::flat_hash_set<HloInstruction*>& while_ops) {
  ConvertInfoMap convert_infos;
  for (HloInstruction* while_op : while_ops) {
    // Find dynamic-slice instructions in the while body.
    for (HloInstruction* instr : while_op->while_body()->instructions()) {
      if (instr->opcode() == HloOpcode::kDynamicSlice) {
        VLOG(2) << "Found dynamic-slice instruction: " << instr->ToString();
        HloDynamicIndexInstruction* dynamic_slice =
            Cast<HloDynamicIndexInstruction>(instr);

        // The buffer for dynamic slice must be a get-tuple-element on the while
        // parameter.
        HloInstruction* buffer = dynamic_slice->mutable_operand(0);
        if (buffer->opcode() != HloOpcode::kGetTupleElement) {
          VLOG(2) << "Skipping dynamic-slice instruction. The buffer is not a "
                     "get-tuple-element: "
                  << instr->ToString();
          continue;
        }
        if (buffer->operand(0)->opcode() != HloOpcode::kParameter) {
          VLOG(2) << "Skipping dynamic-slice instruction. The buffer is not a "
                     "get-tuple-element of a parameter: "
                  << instr->ToString();
          continue;
        }

        // The dynamic-slice operation must have exactly one variable index
        // (lets call that index k).
        bool multiple_variable_indices = false;
        HloInstruction* dynamic_slice_index = nullptr;
        int64_t k = -1;
        VLOG(2) << "Buffer: " << buffer->ToString();
        for (int idx = 0; idx < dynamic_slice->index_operands().size(); idx++) {
          HloInstruction* idx_operand = dynamic_slice->index_operands()[idx];
          VLOG(2) << "Processing index " << idx << " with operand "
                  << idx_operand->ToString();
          if (dynamic_slice_index == nullptr &&
              idx_operand->opcode() != HloOpcode::kConstant) {
            VLOG(2) << "\tDynamic index found.";
            dynamic_slice_index = idx_operand;
            k = idx;
          } else if (dynamic_slice_index != nullptr &&
                     idx_operand->opcode() != HloOpcode::kConstant) {
            VLOG(2) << "\tMore than one variable index found.";
            // We found more than one variable index.
            multiple_variable_indices = true;
            break;
          } else if (dynamic_slice_index != nullptr &&
                     idx_operand->opcode() == HloOpcode::kConstant) {
            // all indices i>k, we should have that
            // dimension_of_buffer[i]=slice_size[i] and offset[i]=0.
            VLOG(2) << "\tConstant index i>k found.";
            VLOG(2) << "\tSlice size: " << dynamic_slice->slice_sizes(idx);
            VLOG(2) << "\tBuffer dim: " << buffer->shape().dimensions(idx);
            VLOG(2) << "\tOffset: "
                    << LiteralUtil::LiteralAsScalarInt64(idx_operand->literal())
                           .value();
            if (dynamic_slice->slice_sizes(/*dimension=*/idx) !=
                    buffer->shape().dimensions(idx) ||
                LiteralUtil::LiteralAsScalarInt64(idx_operand->literal()) !=
                    0) {
              VLOG(2) << "\tSlice size mismatch. Skipping dynamic-slice "
                         "instruction.";
              multiple_variable_indices = true;
              break;
            }
          } else if (dynamic_slice_index == nullptr &&
                     idx_operand->opcode() == HloOpcode::kConstant) {
            // For all indices i < k, we should have that
            // dimension[i]=slice_size[i]=1 and offset[i]=0
            VLOG(2) << "\tConstant index i<k found.";
            VLOG(2) << "\tSlice size: " << dynamic_slice->slice_sizes(idx);
            VLOG(2) << "\tBuffer dim: " << buffer->shape().dimensions(idx);
            VLOG(2) << "\tOffset: "
                    << LiteralUtil::LiteralAsScalarInt64(idx_operand->literal())
                           .value();
            if (dynamic_slice->slice_sizes(idx) !=
                    buffer->shape().dimensions(idx) ||
                dynamic_slice->slice_sizes(idx) != 1 ||
                LiteralUtil::LiteralAsScalarInt64(idx_operand->literal()) !=
                    0) {
              VLOG(2) << "\tSlice size mismatch. Skipping dynamic-slice "
                         "instruction.";
              multiple_variable_indices = true;
              break;
            }
          }
        }
        if (multiple_variable_indices || dynamic_slice_index == nullptr) {
          VLOG(2) << "Skipping dynamic-slice instruction with multiple "
                     "variable indices or no variable index: "
                  << instr->ToString();
          continue;
        }

        // The variable index k must have the monotonic range [0, dimension[k])
        // and each of those values must be taken once (step=1).
        std::optional<Range> k_range =
            RecursivelyIdentifyRange(dynamic_slice_index, known_ranges);
        if (!k_range.has_value() || k_range->IsEmpty() ||
            k_range->step()->GetSignedValue() != 1 || !k_range->IsLinear() ||
            k_range->min().GetSignedValue() != 0 ||
            k_range->max()->GetSignedValue() + 1 !=
                buffer->shape().dimensions(k)) {
          VLOG(2) << "Skipping dynamic-slice instruction. The range of the "
                     "variable index does not cover the full buffer: "
                  << instr->ToString();
          if (k_range.has_value()) {
            VLOG(2) << "Range: " << k_range->ToString();
            VLOG(2) << "Buffer dim: " << buffer->shape().dimensions(k);
          }
          continue;
        }

        // Pattern: dynamic-slice -> convert
        if (dynamic_slice->user_count() > 1) {
          VLOG(2) << "Skipping dynamic-slice instruction. The dynamic-slice "
                     "has more than 1 user: "
                  << instr->ToString();
          continue;
        }

        if (buffer->user_count() != 2) {
          VLOG(2) << "Skipping dynamic-slice instruction. The buffer must have "
                     "exactly 2 users: "
                  << buffer->ToString();
          continue;
        }

        HloInstruction* convert_user = dynamic_slice->users()[0];
        HloInstruction* root_user = nullptr;
        root_user = buffer->users()[0]->opcode() == HloOpcode::kTuple
                        ? buffer->users()[0]
                        : buffer->users()[1];
        VLOG(2) << "Root user: " << root_user->ToString();
        VLOG(2) << "Convert user: " << convert_user->ToString();

        if (root_user->opcode() != HloOpcode::kTuple ||
            convert_user->opcode() != HloOpcode::kConvert ||
            !root_user->IsRoot()) {
          VLOG(2) << "Skipping dynamic-slice instruction. Expected a tuple "
                     "(root with same index) and a convert user."
                  << instr->ToString();
          continue;
        }

        // Buffer index in input tuple should be same as index of buffer in root
        // instruction
        if (buffer->tuple_index() != root_user->operand_index(buffer)) {
          VLOG(2)
              << "Skipping dynamic-slice instruction. Buffer index in input "
                 "tuple should be same as index of buffer in root instruction.";
          VLOG(2) << "\tBuffer: " << buffer->ToString();
          VLOG(2) << "\tRoot user: " << root_user->ToString();
          continue;
        }
        convert_infos[while_op].buffer_infos.push_back(
            {dynamic_slice, buffer, convert_user, root_user,
             buffer->tuple_index(),
             hlo_query::GetUniqueGteInstruction(while_op,
                                                buffer->tuple_index())});
      }
    }
  }
  return convert_infos;
}

using Replacements =
    absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>;

absl::Status WhileLoopConvertPeeler::CreateTupleGetTupleElementForRoot(
    HloInstruction* while_op, ConvertInfo& convert_info) {
  TF_RET_CHECK(while_op->opcode() == HloOpcode::kWhile);
  TF_RET_CHECK(while_op->IsRoot());

  VLOG(2) << "Creating tuple(get-tuple-element(while_op)) for because while "
             "operation is root operation : "
          << while_op->ToString();

  absl::InlinedVector<HloInstruction*, 4> tuple_elements(
      while_op->shape().tuple_shapes_size());

  absl::flat_hash_set<int64_t> while_tuple_buffer_idxs;
  for (BufferInfo& buffer_info : convert_info.buffer_infos) {
    while_tuple_buffer_idxs.insert(buffer_info.while_tuple_buffer_idx);
  }
  for (int idx = 0; idx < while_op->shape().tuple_shapes_size(); ++idx) {
    HloInstruction* get_tuple_element =
        while_tuple_buffer_idxs.contains(idx)
            ? while_op->while_init()->mutable_operand(idx)
            : while_op->AddInstruction(HloInstruction::CreateGetTupleElement(
                  while_op->shape().tuple_shapes(idx), while_op, idx));
    tuple_elements[idx] = get_tuple_element;
  }
  HloInstruction* tuple =
      while_op->AddInstruction(HloInstruction::CreateTuple(tuple_elements));
  while_op->parent()->set_root_instruction(tuple);
  VLOG(2) << "After creating tuple(get-tuple-element(while_op)): "
          << while_op->parent()->ToString(HloPrintOptions::ShortParsable());
  return absl::OkStatus();
}

void WhileLoopConvertPeeler::DeduceNewWhileShape(HloInstruction* while_op,
                                                 ConvertInfo& convert_info) {
  VLOG(2) << "Deducing new while shape.";
  std::vector<Shape> new_tuple_shapes = while_op->shape().tuple_shapes();
  for (BufferInfo& buffer_info : convert_info.buffer_infos) {
    new_tuple_shapes[buffer_info.while_tuple_buffer_idx] = ShapeUtil::MakeShape(
        buffer_info.body_convert->shape().element_type(),
        new_tuple_shapes[buffer_info.while_tuple_buffer_idx].dimensions());
  }
  Shape new_shape = ShapeUtil::MakeTupleShape(new_tuple_shapes);
  convert_info.new_while_shape = new_shape;
  VLOG(2) << "New while shape: " << new_shape.ToString();
}

void WhileLoopConvertPeeler::FixWhileBody(HloInstruction* while_op,
                                          ConvertInfo& convert_info) {
  VLOG(2) << "Fixing while body. Before fix: "
          << while_op->while_body()->ToString(HloPrintOptions::ShortParsable());
  HloComputation* while_body = while_op->while_body();
  HloModule* module = while_body->parent();

  while_body->ReplaceParameter(
      0, HloInstruction::CreateParameter(0, convert_info.new_while_shape,
                                         "updated_while_param"));

  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;

  absl::InlinedVector<HloInstruction*, 2> root_tuple_elements =
      while_body->root_instruction()->operands();
  for (BufferInfo& buffer_info : convert_info.buffer_infos) {
    HloInstruction* body_buffer = buffer_info.body_buffer;
    replacements[body_buffer] = HloInstruction::CreateGetTupleElement(
        body_buffer->mutable_operand(0), body_buffer->tuple_index());
    replacements[body_buffer]->SetAndSanitizeName(body_buffer->name());

    HloDynamicIndexInstruction* dynamic_slice =
        buffer_info.dynamic_index_instruction;
    replacements[dynamic_slice] = HloInstruction::CreateDynamicSlice(
        buffer_info.body_convert->shape(),
        replacements[buffer_info.body_buffer].get(),
        /*start_indices=*/dynamic_slice->index_operands(),
        /*slice_sizes=*/dynamic_slice->dynamic_slice_sizes());
    replacements[dynamic_slice]->SetAndSanitizeName(dynamic_slice->name());

    root_tuple_elements[buffer_info.while_tuple_buffer_idx] =
        replacements[body_buffer].get();
  }
  replacements[while_body->root_instruction()] =
      HloInstruction::CreateTuple(root_tuple_elements);
  replacements[while_body->root_instruction()]->SetAndSanitizeName(
      while_body->root_instruction()->name());
  HloComputation* new_body = module->AddComputationAndUnifyNamesAndIds(
      while_body->CloneWithReplacements(&replacements), false);
  while_op->parent()->parent()->ReplaceComputations({{while_body, new_body}});
  VLOG(2) << "Fixing while body. After fix: "
          << new_body->ToString(HloPrintOptions::ShortParsable());
}

void WhileLoopConvertPeeler::FixWhileCond(HloInstruction* while_op,
                                          ConvertInfo& convert_info) {
  VLOG(2) << "Fixing while cond. Before fix: "
          << while_op->while_condition()->ToString(
                 HloPrintOptions::ShortParsable());
  HloComputation* while_cond = while_op->while_condition();
  while_cond->ReplaceParameter(
      0, HloInstruction::CreateParameter(0, convert_info.new_while_shape,
                                         "updated_while_param"));
  VLOG(2) << "Fixing while cond. After fix: "
          << while_cond->ToString(HloPrintOptions::ShortParsable());
}

absl::Status WhileLoopConvertPeeler::FixWhileInit(HloInstruction* while_op,
                                                  ConvertInfo& convert_info) {
  VLOG(2) << "Fixing while init. Before fix: "
          << while_op->parent()->ToString(HloPrintOptions::ShortParsable());
  HloInstruction* while_init = while_op->while_init();
  HloComputation* while_init_computation = while_init->parent();
  HloModule* module = while_init_computation->parent();
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;

  // Create the new init tuple.
  absl::InlinedVector<HloInstruction*, 2> tuple_elements =
      while_init->operands();
  for (BufferInfo& buffer_info : convert_info.buffer_infos) {
    HloInstruction* init_buffer =
        while_init->mutable_operand(buffer_info.while_tuple_buffer_idx);
    HloInstruction* convert =
        while_op->AddInstruction(HloInstruction::CreateConvert(
            ShapeUtil::MakeShape(
                buffer_info.body_convert->shape().element_type(),
                init_buffer->shape().dimensions()),
            init_buffer));
    tuple_elements[buffer_info.while_tuple_buffer_idx] = convert;
  }
  replacements[while_init] = HloInstruction::CreateTuple(tuple_elements);
  replacements[while_init]->SetAndSanitizeName(while_init->name());

  replacements[while_op] = HloInstruction::CreateWhile(
      convert_info.new_while_shape, while_op->while_condition(),
      while_op->while_body(), replacements[while_init].get());
  replacements[while_op]->SetAndSanitizeName(while_op->name());

  for (const BufferInfo& buffer_info : convert_info.buffer_infos) {
    HloInstruction* gte = buffer_info.while_gte_user_after_loop;
    if (gte != nullptr) {
      TF_RETURN_IF_ERROR(while_init_computation->ReplaceInstruction(
          gte,
          while_init->mutable_operand(buffer_info.while_tuple_buffer_idx)));
    }
  }

  HloComputation* new_init_computation =
      module->AddComputationAndUnifyNamesAndIds(
          while_init_computation->CloneWithReplacements(&replacements),
          /*is_entry=*/false);
  while_op->parent()->parent()->ReplaceComputations(
      {{while_init_computation, new_init_computation}});
  VLOG(2) << "Fixing while init. After fix: "
          << new_init_computation->ToString(HloPrintOptions::ShortParsable());
  return absl::OkStatus();
}

absl::StatusOr<bool> WhileLoopConvertPeeler::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Find while loops and get their induction variable ranges.
  absl::flat_hash_map<const HloInstruction*, Range> known_ranges;
  absl::flat_hash_set<HloInstruction*> while_ops;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        std::optional<Range> range = MatchTrivialLoopRange(instruction);
        std::optional<int64_t> tuple_idx =
            GetLoopInductionVarTupleIdx(instruction);
        if (range.has_value() && tuple_idx.has_value()) {
          HloInstruction* induction_var = hlo_query::GetUniqueGteInstruction(
              instruction->while_body()->parameter_instruction(0),
              tuple_idx.value());
          known_ranges[induction_var] = range.value();
          while_ops.insert(instruction);
        }
      }
    }
  }

  ConvertInfoMap convert_infos =
      CollectConvertInfos(module, known_ranges, while_ops);

  if (convert_infos.empty()) {
    return false;
  }

  // Preprocessing: adding the convert operation to while init, and removing
  // while_op as root if it is the root.
  for (auto& [while_op, convert_info] : convert_infos) {
    VLOG(2) << "Processing while loop: " << while_op->ToString();
    VLOG(2) << "Convert info: \n" << convert_info.ToString();
    if (while_op->IsRoot()) {
      TF_RETURN_IF_ERROR(
          CreateTupleGetTupleElementForRoot(while_op, convert_info));
    }
    DeduceNewWhileShape(while_op, convert_info);
    FixWhileBody(while_op, convert_info);
    FixWhileCond(while_op, convert_info);
    TF_RETURN_IF_ERROR(FixWhileInit(while_op, convert_info));
  }

  return true;
}

}  // namespace xla::gpu
