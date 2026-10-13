/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/backends/cpu/buffer_allocation_info_util.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/buffer_allocation_info.h"
#include "xla/service/buffer_assignment.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"

namespace xla::cpu {

std::vector<BufferAllocationInfo> CreateBufferAllocationInfos(
    const HloModuleProto& module,
    const BufferAssignmentProto& buffer_assignment) {
  absl::flat_hash_map<int64_t, const HloComputationProto*> id_to_computation;
  absl::flat_hash_map<int64_t, const HloInstructionProto*> id_to_instruction;
  const HloComputationProto* entry_comp = nullptr;
  for (const HloComputationProto& comp : module.computations()) {
    id_to_computation[comp.id()] = &comp;
    if (comp.id() == module.entry_computation_id()) {
      entry_comp = &comp;
    }
    for (const HloInstructionProto& instr : comp.instructions()) {
      id_to_instruction[instr.id()] = &instr;
    }
  }
  if (entry_comp == nullptr && !module.computations().empty()) {
    entry_comp = &module.computations(module.computations_size() - 1);
  }

  using LocationKey = std::pair<int64_t, std::vector<int64_t>>;
  absl::flat_hash_map<LocationKey, int64_t> location_to_buffer_id;
  for (const LogicalBufferProto& lb : buffer_assignment.logical_buffers()) {
    const auto& loc = lb.defined_at();
    std::vector<int64_t> idx(loc.shape_index().begin(),
                             loc.shape_index().end());
    location_to_buffer_id[{loc.instruction_id(), std::move(idx)}] = lb.id();
  }
  for (const BufferAssignmentProto::BufferAlias& alias :
       buffer_assignment.buffer_aliases()) {
    const auto& loc = alias.location();
    std::vector<int64_t> idx(loc.shape_index().begin(),
                             loc.shape_index().end());
    location_to_buffer_id[{loc.instruction_id(), std::move(idx)}] =
        alias.source_buffer_id();
  }

  absl::flat_hash_map<int64_t, int64_t> buffer_id_to_alloc;
  for (const BufferAllocationProto& alloc :
       buffer_assignment.buffer_allocations()) {
    for (const auto& assigned : alloc.assigned()) {
      buffer_id_to_alloc[assigned.logical_buffer_id()] = alloc.index();
    }
  }

  auto resolve_allocation_index =
      [&](int64_t instr_id,
          std::vector<int64_t> index) -> std::optional<int64_t> {
    for (int depth = 0; depth < 64; ++depth) {
      auto buf_it = location_to_buffer_id.find({instr_id, index});
      if (buf_it != location_to_buffer_id.end()) {
        auto alloc_it = buffer_id_to_alloc.find(buf_it->second);
        if (alloc_it != buffer_id_to_alloc.end()) {
          return alloc_it->second;
        }
      }
      auto instr_it = id_to_instruction.find(instr_id);
      if (instr_it == id_to_instruction.end()) {
        break;
      }
      const HloInstructionProto* instr = instr_it->second;
      if (instr->opcode() == "tuple" && !index.empty() && index[0] >= 0 &&
          index[0] < instr->operand_ids_size()) {
        instr_id = instr->operand_ids(index[0]);
        index.erase(index.begin());
      } else if (instr->opcode() == "get-tuple-element" &&
                 instr->operand_ids_size() == 1) {
        instr_id = instr->operand_ids(0);
        index.insert(index.begin(), instr->tuple_index());
      } else if ((instr->opcode() == "bitcast" ||
                  instr->opcode() == "optimization-barrier" ||
                  instr->opcode() == "domain" ||
                  instr->opcode() == "add-dependency" ||
                  instr->opcode() == "copy" || instr->opcode() == "while") &&
                 instr->operand_ids_size() >= 1) {
        instr_id = instr->operand_ids(0);
      } else if (instr->opcode() == "call" &&
                 instr->called_computation_ids_size() == 1) {
        auto comp_it = id_to_computation.find(instr->called_computation_ids(0));
        if (comp_it == id_to_computation.end()) {
          break;
        }
        instr_id = comp_it->second->root_id();
      } else {
        break;
      }
    }
    return std::nullopt;
  };

  absl::flat_hash_map<int64_t, int64_t> result_allocations;
  if (entry_comp != nullptr) {
    auto root_it = id_to_instruction.find(entry_comp->root_id());
    if (root_it != id_to_instruction.end()) {
      Shape root_shape = Shape::FromProto(root_it->second->shape()).value();
      ShapeUtil::ForEachLeafShape(
          root_shape, [&](const Shape& subshape, const ShapeIndex& index) {
            std::vector<int64_t> idx(index.begin(), index.end());
            std::optional<int64_t> allocation_index =
                resolve_allocation_index(entry_comp->root_id(), std::move(idx));
            DCHECK(allocation_index.has_value());
            if (allocation_index.has_value()) {
              int64_t result_index = result_allocations.size();
              result_allocations[*allocation_index] = result_index;
            }
          });
    }
  }

  std::vector<BufferAllocationInfo> allocations;
  allocations.reserve(buffer_assignment.buffer_allocations_size());
  for (const BufferAllocationProto& allocation :
       buffer_assignment.buffer_allocations()) {
    DCHECK_EQ(allocation.index(), allocations.size());

    if (allocation.is_thread_local()) {
      allocations.push_back(
          BufferAllocationInfo::ThreadLocal(allocation.size()));
    } else if (allocation.is_constant()) {
      allocations.push_back(BufferAllocationInfo::Constant(allocation.size()));
    } else if (allocation.is_entry_computation_parameter() &&
               allocation.maybe_live_out()) {
      allocations.push_back(BufferAllocationInfo::InOutParameter(
          allocation.size(), allocation.parameter_number(),
          result_allocations.at(allocation.index())));
    } else if (allocation.is_entry_computation_parameter()) {
      allocations.push_back(BufferAllocationInfo::EntryParameter(
          allocation.size(), allocation.parameter_number()));
    } else if (allocation.maybe_live_out() &&
               result_allocations.contains(allocation.index())) {
      allocations.push_back(BufferAllocationInfo::Result(
          allocation.size(), result_allocations[allocation.index()]));
    } else if (allocation.maybe_live_out()) {
      allocations.push_back(
          BufferAllocationInfo::Result(allocation.size(), -1));
    } else {
      allocations.push_back(BufferAllocationInfo::Temp(allocation.size()));
    }
  }

  return allocations;
}

absl::StatusOr<ShapeTree<int64_t>> CreateResultAllocationIndexTree(
    const HloModuleProto& module,
    const BufferAssignmentProto& buffer_assignment) {
  absl::flat_hash_map<int64_t, const HloComputationProto*> id_to_computation;
  absl::flat_hash_map<int64_t, const HloInstructionProto*> id_to_instruction;
  const HloComputationProto* entry_comp = nullptr;
  for (const HloComputationProto& comp : module.computations()) {
    id_to_computation[comp.id()] = &comp;
    if (comp.id() == module.entry_computation_id()) {
      entry_comp = &comp;
    }
    for (const HloInstructionProto& instr : comp.instructions()) {
      id_to_instruction[instr.id()] = &instr;
    }
  }
  if (entry_comp == nullptr && !module.computations().empty()) {
    entry_comp = &module.computations(module.computations_size() - 1);
  }
  if (entry_comp == nullptr) {
    return absl::InternalError("HloModuleProto has no entry computation");
  }
  auto root_it = id_to_instruction.find(entry_comp->root_id());
  if (root_it == id_to_instruction.end()) {
    return absl::InternalError("HloModuleProto entry root not found");
  }

  using LocationKey = std::pair<int64_t, std::vector<int64_t>>;
  absl::flat_hash_map<LocationKey, int64_t> location_to_buffer_id;
  for (const LogicalBufferProto& lb : buffer_assignment.logical_buffers()) {
    const auto& loc = lb.defined_at();
    std::vector<int64_t> idx(loc.shape_index().begin(),
                             loc.shape_index().end());
    location_to_buffer_id[{loc.instruction_id(), std::move(idx)}] = lb.id();
  }
  for (const BufferAssignmentProto::BufferAlias& alias :
       buffer_assignment.buffer_aliases()) {
    const auto& loc = alias.location();
    std::vector<int64_t> idx(loc.shape_index().begin(),
                             loc.shape_index().end());
    location_to_buffer_id[{loc.instruction_id(), std::move(idx)}] =
        alias.source_buffer_id();
  }

  absl::flat_hash_map<int64_t, int64_t> buffer_id_to_alloc;
  for (const BufferAllocationProto& alloc :
       buffer_assignment.buffer_allocations()) {
    for (const auto& assigned : alloc.assigned()) {
      buffer_id_to_alloc[assigned.logical_buffer_id()] = alloc.index();
    }
  }

  auto resolve_allocation_index =
      [&](int64_t instr_id,
          std::vector<int64_t> index) -> std::optional<int64_t> {
    for (int depth = 0; depth < 64; ++depth) {
      auto buf_it = location_to_buffer_id.find({instr_id, index});
      if (buf_it != location_to_buffer_id.end()) {
        auto alloc_it = buffer_id_to_alloc.find(buf_it->second);
        if (alloc_it != buffer_id_to_alloc.end()) {
          return alloc_it->second;
        }
      }
      auto instr_it = id_to_instruction.find(instr_id);
      if (instr_it == id_to_instruction.end()) {
        break;
      }
      const HloInstructionProto* instr = instr_it->second;
      if (instr->opcode() == "tuple" && !index.empty() && index[0] >= 0 &&
          index[0] < instr->operand_ids_size()) {
        instr_id = instr->operand_ids(index[0]);
        index.erase(index.begin());
      } else if (instr->opcode() == "get-tuple-element" &&
                 instr->operand_ids_size() == 1) {
        instr_id = instr->operand_ids(0);
        index.insert(index.begin(), instr->tuple_index());
      } else if ((instr->opcode() == "bitcast" ||
                  instr->opcode() == "optimization-barrier" ||
                  instr->opcode() == "domain" ||
                  instr->opcode() == "add-dependency" ||
                  instr->opcode() == "copy" || instr->opcode() == "while") &&
                 instr->operand_ids_size() >= 1) {
        instr_id = instr->operand_ids(0);
      } else if (instr->opcode() == "call" &&
                 instr->called_computation_ids_size() == 1) {
        auto comp_it = id_to_computation.find(instr->called_computation_ids(0));
        if (comp_it == id_to_computation.end()) {
          break;
        }
        instr_id = comp_it->second->root_id();
      } else {
        break;
      }
    }
    return std::nullopt;
  };

  Shape root_shape = Shape::FromProto(root_it->second->shape()).value();
  ShapeTree<int64_t> result_tree(root_shape, -1);
  for (auto& [index, alloc_index] : result_tree) {
    std::vector<int64_t> idx(index.begin(), index.end());
    std::optional<int64_t> resolved =
        resolve_allocation_index(entry_comp->root_id(), std::move(idx));
    if (!resolved.has_value()) {
      return absl::InternalError(
          "Could not resolve result allocation index from proto");
    }
    alloc_index = *resolved;
  }
  return result_tree;
}

std::vector<int32_t> CreateArgIndexTable(
    absl::Span<const BufferAllocationInfo> allocations) {
  std::vector<int32_t> ret;
  for (int64_t i = 0; i < allocations.size(); i++) {
    if (allocations[i].is_entry_parameter()) {
      int32_t parameter_number = allocations[i].entry_parameter_number();
      if (parameter_number >= ret.size()) {
        ret.resize(parameter_number + 1);
      }
      ret[parameter_number] = i;
    }
  }
  return ret;
}

std::vector<int32_t> CreateResultIndexTable(
    absl::Span<const BufferAllocationInfo> allocations) {
  std::vector<int32_t> ret;
  for (int64_t i = 0; i < allocations.size(); i++) {
    if (allocations[i].is_result()) {
      int32_t result_number = allocations[i].result_number();
      if (result_number >= ret.size()) {
        ret.resize(result_number + 1);
      }
      ret[result_number] = i;
    }
  }
  return ret;
}

}  // namespace xla::cpu
