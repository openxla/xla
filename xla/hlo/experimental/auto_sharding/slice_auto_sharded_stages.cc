/*
 Copyright 2024 CNAEIT

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#include "xla/hlo/experimental/auto_sharding/slice_auto_sharded_stages.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_util.h"

namespace xla {
namespace spmd {

namespace nb = nanobind;

namespace {
std::vector<std::string> SplitString(const std::string &s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;

  while (std::getline(ss, item, delim)) {
      result.push_back(item);
  }

  return result;
}

std::pair<std::string, std::string> ParseNameAndType(const HloInstruction* ins) {
  auto tokens = SplitString(ins->metadata().op_name(), '$');
  CHECK(tokens.size() == 2);
  return std::make_pair(tokens[0], tokens[1]);
}
} // namespace


enum VisitState { kVisiting, kVisited };

std::unique_ptr<HloModule> CreateStageModule(
    HloModule* full_module, HloInstruction* stage_start_instruction,
    HloInstruction* stage_end_instruction, std::string stage_name_suffix) {
  CHECK(stage_start_instruction->IsCustomCall(kPipelineMarker));
  std::string start_stage_name, start_marker_type;
  std::tie(start_stage_name, start_marker_type) = ParseNameAndType(
    stage_start_instruction);
  CHECK(stage_end_instruction->IsCustomCall(kPipelineMarker));
  std::string end_stage_name, end_marker_type;
  std::tie(end_stage_name, end_marker_type) = ParseNameAndType(
    stage_end_instruction);
  CHECK_EQ(start_marker_type, kPipelineMarkerStartType);
  CHECK_EQ(end_marker_type, kPipelineMarkerEndType);
  CHECK_EQ(start_stage_name, end_stage_name);

  // DFS search to find all instructions in the stage.
  std::vector<const HloInstruction*> stage_instructions;
  absl::flat_hash_map<const HloInstruction*, VisitState> visited;
  std::vector<const HloInstruction*> dfs_stack;
  dfs_stack.push_back(stage_end_instruction->operand(0));

  while (!dfs_stack.empty()) {
    auto* cur = dfs_stack.back();
    auto it = visited.find(cur);
    if (it != visited.end()) {
      dfs_stack.pop_back();
      if (it->second == kVisited) {
        continue;
      }
      CHECK_EQ(it->second, kVisiting);
      stage_instructions.push_back(cur);
      it->second = kVisited;
      continue;
    }

    visited.insert({cur, kVisiting});
    if (!(cur->opcode() == HloOpcode::kGetTupleElement &&
          cur->operand(0) == stage_start_instruction)) {
      for (auto operand : cur->operands()) {
        dfs_stack.push_back(operand);
      }
    }
  }

  // Setup the new stage module.
  HloModuleConfig config = full_module->config();
  config.set_shardable_value_update_pairs({});
  config.mutable_fusion_config()->clear();
  config.mutable_dot_config()->clear();
  config.mutable_layout_config()->clear();

  auto module = absl::make_unique<HloModule>(
      absl::StrCat(full_module->name(), "-", stage_name_suffix), config);
  auto context_ptr =
      absl::make_unique<HloCloneContext>(module.get(), stage_name_suffix);
  HloCloneContext* context = context_ptr.get();

  std::vector<std::unique_ptr<HloInstruction>> instructions;

  // Create parameters for the new stage module.
  int n_parameters = stage_start_instruction->shape().tuple_shapes_size();
  std::vector<HloInstruction*> parameters(n_parameters);
  for (int i = 0; i < n_parameters; ++i) {
    auto new_param = HloInstruction::CreateParameter(
        i, stage_start_instruction->shape().tuple_shapes(i),
        absl::StrCat("param_", i));
    if (stage_start_instruction->has_sharding()) {
      CHECK(stage_start_instruction->sharding().IsTuple());
      new_param->set_sharding(
          stage_start_instruction->sharding().GetSubSharding(
              stage_start_instruction->shape(), {i}));
    }
    new_param->set_metadata(stage_start_instruction->metadata());
    parameters[i] = new_param.get();
    instructions.push_back(std::move(new_param));
  }

  // Process the instructions in the stage.
  for (auto ins : stage_instructions) {
    CHECK_NE(ins->opcode(), HloOpcode::kParameter)
        << "All the inputs to a pipeline stage should be from the start "
           "marker. " << ins->ToString();
    if (ins->opcode() == HloOpcode::kGetTupleElement &&
        ins->operand(0) == stage_start_instruction) {
      int64_t param_no = ins->tuple_index();
      context->MapInstruction(ins, parameters[param_no]);
    } else {
      std::vector<HloInstruction*> new_operands;
      for (HloInstruction* operand : ins->operands()) {
        new_operands.push_back(context->GetInstruction(operand));
      }
      instructions.push_back(
          ins->CloneWithNewOperands(ins->shape(), new_operands, context));
    }
  }

  // Build the HLO computation.
  HloComputation::Builder builder(absl::StrCat(
      full_module->entry_computation()->name(), "-", stage_name_suffix));
  for (auto& ins : instructions) {
    builder.AddInstruction(std::move(ins));
  }
  std::unique_ptr<HloComputation> new_computation = builder.Build(
      /*root_instruction=*/context->GetInstruction(
          stage_end_instruction->operand(0)));

  for (auto ins : stage_instructions) {
    HloInstruction* new_ins = context->GetInstruction(ins);
    for (auto successor : ins->control_successors()) {
      TF_CHECK_OK(
          new_ins->AddControlDependencyTo(context->GetInstruction(successor)));
    }
  }

  // NOTE: We assume the HLO graph only has one computation.
  module->AddEntryComputationWithLayouts(std::move(new_computation));

  return module;
}

std::vector<std::string> HookShardingProto(HloModule* module) {
  HloComputation* entry = module->entry_computation();
  std::vector<std::string> shardings;
  if (module->config().num_partitions() <= 1) {
    return shardings;
  }

  for (HloInstruction* inst : entry->instructions()) {
    if (inst->opcode() == HloOpcode::kOptimizationBarrier &&
        inst->metadata().op_name() == "hook$hook") {
      for (HloInstruction* operand : inst->operands()) {
        shardings.push_back(operand->sharding().ToProto().SerializeAsString());
      }
      return shardings;
    }
  }
  return shardings;
}


std::vector<std::unique_ptr<HloModule>> SliceAutoShardedStagesInternal(
    HloModule* module) {
  // ----- Slice the hlo module according to the pipeline marker -----
  HloComputation* entry = module->entry_computation();

  absl::flat_hash_map<std::string, std::pair<HloInstruction*, HloInstruction*>>
      stage_start_end_instructions;
  for (HloInstruction* ins : entry->instructions()) {
    if (ins->IsCustomCall(kPipelineMarker)) {
	  std::string pipeline_stage_name, marker_type;
	  std::tie(pipeline_stage_name, marker_type) = ParseNameAndType(ins);
      if (!stage_start_end_instructions.contains(pipeline_stage_name)) {
        stage_start_end_instructions[pipeline_stage_name] =
            std::make_pair(nullptr, nullptr);
      }
      if (marker_type == kPipelineMarkerStartType) {
        stage_start_end_instructions[pipeline_stage_name].first = ins;
      } else if (marker_type == kPipelineMarkerEndType) {
        stage_start_end_instructions[pipeline_stage_name].second = ins;
      } else {
		LOG(WARNING) << "Unexpected marker_type: " << marker_type;
      }
    }
  }

  std::vector<std::string> pipeline_stage_names;
  std::vector<std::unique_ptr<HloModule>> pipeline_stages;
  for (const auto& it : stage_start_end_instructions) {
    if (it.second.first != nullptr && it.second.second != nullptr) {
      pipeline_stage_names.push_back(it.first);
      pipeline_stages.push_back(CreateStageModule(module, it.second.first,
                                                  it.second.second, it.first));
    }
  }

  if (pipeline_stages.empty()) {
    return pipeline_stages;
  }

  // ----- Put the sharded HLO module back to Python -----
  // PyGILState_STATE gstate = PyGILState_Ensure();
  // {
  //   nb::object submodule =
  //       nb::module_::import("mesha.shard_parallel.auto_sharding");
  //   nb::list stage_names;
  //   nb::list stage_modules;
  //   for (const auto& name : pipeline_stage_names) {
  //     nb::str python_name(name);
  //     stage_names.append(name);
  //   }
  //   for (auto& stage_module : pipeline_stages) {
  //     std::shared_ptr<HloModule> module = std::move(stage_module);
  //     stage_modules.append(module);
  //   }
  //   nb::tuple stages = nb::make_tuple(stage_names, stage_modules);
  //   nb::object set_auto_sharded_hlo_stages =
  //       submodule.attr("set_auto_sharded_hlo_stages");
  //   nb::object ret = set_auto_sharded_hlo_stages(stages);
  //   if (!ret.is_none()) {
  //     PyGILState_Release(gstate);
  //     exit(-1);
  //   }
  //   // Hook sharding proto. TODO(yonghao): support more than one hook
  //   std::vector<std::string> hooked_sharding_protos = HookShardingProto(module);
  //   nb::list hooked_shardings;
  //   for (const std::string sharding_proto : hooked_sharding_protos) {
  //     hooked_shardings.append(nb::bytes(sharding_proto));
  //   }
  //   nb::object set_hooked_sharding_protos =
  //       submodule.attr("set_hooked_sharding_protos");
  //   ret = set_hooked_sharding_protos(hooked_shardings);
  //   if (!ret.is_none()) {
  //     PyGILState_Release(gstate);
  //     exit(-1);
  //   }
  // }
  // PyGILState_Release(gstate);

  return pipeline_stages;
}

StatusOr<bool> SliceAutoShardedStages::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  SliceAutoShardedStagesInternal(module);
  return false;
}

}  // namespace spmd
}  // namespace xla