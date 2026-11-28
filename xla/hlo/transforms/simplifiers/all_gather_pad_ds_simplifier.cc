/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the apecific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/hlo/transforms/simplifiers/all_gather_pad_ds_simplifier.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/literal_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/collective_opt_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

bool IsStaticArray(const Shape& shape) {
  return shape.IsArray() && shape.is_static();
}

// Validates initial properties of the HloPadInstruction.
bool ValidateInitialPadProperties(const HloPadInstruction& pad,
                                  const Shape& ds_shape,
                                  const Shape& ag_shape) {
  // 1. Check if the padding value is a scalar zero constant with a literal.
  const HloInstruction* pad_value = pad.padding_value();
  if (pad.operand_count() != 2 || pad.operand(0) == nullptr ||
      pad_value == nullptr || pad_value->opcode() != HloOpcode::kConstant ||
      !Cast<HloConstantInstruction>(pad_value)->HasLiteral() ||
      !ShapeUtil::IsScalar(pad_value->shape()) ||
      !pad_value->literal().IsZero({})) {
    VLOG(2) << "Pad value is not a scalar zero constant: " << pad.ToString()
            << " value: "
            << (pad_value == nullptr ? "<null>" : pad_value->ToString());
    return false;
  }

  // 2. Check static array shapes, matching element types, and matching ranks.
  const Shape& pad_shape = pad.shape();
  const Shape& operand_shape = pad.operand(0)->shape();
  const PrimitiveType element_type = pad_shape.element_type();
  if (!IsStaticArray(pad_shape) || !IsStaticArray(operand_shape) ||
      !IsStaticArray(ds_shape) || !IsStaticArray(ag_shape) ||
      pad_value->shape().element_type() != element_type ||
      operand_shape.element_type() != element_type ||
      ag_shape.element_type() != element_type ||
      ds_shape.element_type() != element_type) {
    VLOG(2) << "Unsupported shape or element type in pad pattern: "
            << pad.ToString();
    return false;
  }

  const int64_t config_rank = pad.padding_config().dimensions_size();
  if (config_rank != pad_shape.dimensions().size() ||
      config_rank != operand_shape.dimensions().size() ||
      config_rank != ag_shape.dimensions().size() ||
      config_rank != ds_shape.dimensions().size()) {
    VLOG(2) << "pad dimension size mismatches the ag/ds dimension size: "
            << pad.ToString();
    return false;
  }
  return true;
}

// Processes the padding configuration for the split dimension.
std::optional<OffsetSpec> ProcessSplitDimensionPadding(
    const HloPadInstruction& pad,
    const PaddingConfig::PaddingConfigDimension& padding_dim, int64_t dim) {
  const int64_t low_pad = padding_dim.edge_padding_low();
  const int64_t high_pad = padding_dim.edge_padding_high();
  if (low_pad < 0 || high_pad < 0) {
    VLOG(2) << "Negative edge padding is not supported on split dim: " << dim
            << " pad:" << pad.ToString();
    return std::nullopt;
  }
  const bool has_low_padding = low_pad > 0;
  const bool has_high_padding = high_pad > 0;
  if (has_low_padding == has_high_padding) {
    VLOG(2) << "Split dim must have padding on exactly one edge: " << dim
            << " pad:" << pad.ToString();
    return std::nullopt;
  }

  const int64_t operand_dim = pad.operand(0)->shape().dimensions(dim);
  const int64_t pad_dim = pad.shape().dimensions(dim);
  const int64_t edge_pad = has_low_padding ? low_pad : high_pad;
  if (operand_dim < 0 || pad_dim < operand_dim ||
      pad_dim - operand_dim != edge_pad) {
    VLOG(2) << "Inconsistent split dim padding extent on dim " << dim
            << " pad:" << pad.ToString();
    return std::nullopt;
  }

  OffsetSpec valid_pad_spec;
  valid_pad_spec.split_dim = dim;
  if (has_low_padding) {
    valid_pad_spec.start_offset = 0;
    valid_pad_spec.end_offset = low_pad;
  } else {
    valid_pad_spec.start_offset = operand_dim;
    valid_pad_spec.end_offset = pad_dim;
  }
  return valid_pad_spec;
}

// Processes the padding configuration for non-split dimensions.
bool ProcessNonSplitDimensionPadding(
    const HloPadInstruction& pad,
    const PaddingConfig::PaddingConfigDimension& padding_dim, int64_t dim) {
  const bool has_low_padding = padding_dim.edge_padding_low() != 0;
  const bool has_high_padding = padding_dim.edge_padding_high() != 0;
  if (has_low_padding || has_high_padding ||
      pad.shape().dimensions(dim) != pad.operand(0)->shape().dimensions(dim)) {
    VLOG(2) << "Pad has edge padding on non-split dim " << dim << ": "
            << pad.ToString();
    return false;
  }
  return true;
}

}  // namespace

std::optional<OffsetSpec> ExtractValidPadSpec(const HloPadInstruction& pad,
                                              const Shape& ds_shape,
                                              const Shape& ag_shape,
                                              int64_t split_dim) {
  if (!ValidateInitialPadProperties(pad, ds_shape, ag_shape) || split_dim < 0 ||
      split_dim >= pad.padding_config().dimensions_size()) {
    return std::nullopt;
  }

  std::optional<OffsetSpec> valid_pad_spec;

  for (int dim = 0; dim < pad.padding_config().dimensions().size(); ++dim) {
    const PaddingConfig::PaddingConfigDimension& padding_dim =
        pad.padding_config().dimensions(dim);
    VLOG(2) << "padding_dim: " << padding_dim.DebugString();

    if (padding_dim.interior_padding() != 0) {
      VLOG(2) << "Interior padding is not allowed: " << pad.ToString();
      return std::nullopt;
    }

    if (dim == split_dim) {
      std::optional<OffsetSpec> spec =
          ProcessSplitDimensionPadding(pad, padding_dim, dim);
      if (!spec.has_value()) {
        return std::nullopt;
      }
      valid_pad_spec = spec;
    } else {
      if (!ProcessNonSplitDimensionPadding(pad, padding_dim, dim)) {
        return std::nullopt;
      }
    }
  }
  return valid_pad_spec;
}

std::optional<OffsetToIdMap::const_iterator> GetPartitionIdForOffset(
    const OffsetToIdMap& offset_to_partition_map, int64_t offset) {
  if (offset_to_partition_map.empty()) {
    return std::nullopt;  // Handle empty map case.
  }

  // 1. Find the upper_bound.
  auto it = offset_to_partition_map.upper_bound(offset);

  // 2. Check if it points to the beginning.
  if (it == offset_to_partition_map.begin()) {
    return std::nullopt;  // Provided offset is smaller than the first known
                          // offset in map.
  }
  // 3. Check if `it` points to the end, meaning the provided offset is larger
  // than the last known offset in the map.
  if (it == offset_to_partition_map.end()) {
    return std::prev(it);
  }
  // 4. Get the previous element and return the value.
  return std::prev(it);
}

HloInstruction* AddPredInstrBasedOnPartitionIdAndList(
    HloComputation* computation, absl::Span<const int64_t> select_list) {
  if (computation == nullptr || select_list.empty()) {
    return nullptr;
  }
  HloInstruction* const_list =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR1<int64_t>(select_list)));

  // Get the partition ID.
  HloInstruction* partition_id =
      computation->AddInstruction(HloInstruction::CreatePartitionId());

  // Dynamic slice the constant list based on the partition ID.
  Shape slice_shape = ShapeUtil::MakeShape(S64, {1});
  HloInstruction* sliced_value =
      computation->AddInstruction(HloInstruction::CreateDynamicSlice(
          slice_shape, const_list, {partition_id}, /*slice_sizes=*/{1}));

  // Reshape the sliced value to a scalar.
  HloInstruction* scalar_sliced_value =
      computation->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(S64, {}), sliced_value));

  // Create a constant 1 for comparison.
  HloInstruction* const_one = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(1)));

  // Compare the sliced value with 1.
  HloInstruction* pred =
      computation->AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::MakeShape(PRED, {}), scalar_sliced_value, const_one,
          ComparisonDirection::kEq));

  return pred;
}

namespace {

// Extracts the offset range of the original AllGather data within the padded
// tensor.
//
// Given the OffsetSpec of the padded tensor, this function calculates the start
// and end offsets that the AllGather tensor occupies within the larger tensor
// produced by the Pad instruction. It assumes the padding is only on one side
// (either low or high) along the split dimension.
std::optional<OffsetSpec> ExtractAllGatherOffsetSpec(
    const OffsetSpec& valid_pad_spec, const Shape& pad_shape,
    const Shape& ag_shape) {
  OffsetSpec ag_offset_spec;
  const int64_t split_dim = valid_pad_spec.split_dim;
  const int64_t pad_dim = pad_shape.dimensions(split_dim);
  if (valid_pad_spec.start_offset < 0 ||
      valid_pad_spec.end_offset < valid_pad_spec.start_offset ||
      valid_pad_spec.end_offset > pad_dim) {
    return std::nullopt;
  }
  int64_t ag_low_edge = valid_pad_spec.start_offset;
  int64_t ag_high_edge = pad_dim - valid_pad_spec.end_offset;
  if (ag_low_edge == 0) {
    if (ag_high_edge != ag_shape.dimensions(split_dim)) {
      VLOG(2) << "invalid shape between ag: " << ag_shape.ToString()
              << " and pad: " << pad_shape.ToString();
      return std::nullopt;
    }
    ag_offset_spec.start_offset = valid_pad_spec.end_offset;
    ag_offset_spec.end_offset = pad_dim;
    ag_offset_spec.split_dim = split_dim;
  } else if (ag_high_edge == 0) {
    if (ag_low_edge != ag_shape.dimensions(split_dim)) {
      VLOG(2) << "invalid shape between ag: " << ag_shape.ToString()
              << " and pad: " << pad_shape.ToString();
      return std::nullopt;
    }
    ag_offset_spec.start_offset = 0;
    ag_offset_spec.end_offset = valid_pad_spec.start_offset;
    ag_offset_spec.split_dim = split_dim;
  } else {
    return std::nullopt;
  }
  return ag_offset_spec;
}

// Adjusts the keys of the AllGather offset-to-ID map based on the AllGather's
// position within the padded tensor.
//
// The AllGather output is a slice within the result of the Pad instruction.
// This function shifts the offsets in the `ag_map` by the
// `ag_offset_spec.start_offset` to reflect their absolute positions
// in the padded tensor.
std::optional<OffsetToIdMap> GenerateOffsettedAgMap(
    const OffsetToIdMap& ag_map, const OffsetSpec& ag_offset_spec,
    int64_t ag_shard_size) {
  const int64_t ag_extent =
      ag_offset_spec.end_offset - ag_offset_spec.start_offset;
  if (ag_shard_size <= 0 || ag_extent < ag_shard_size) {
    return std::nullopt;
  }
  DCHECK_GT(ag_shard_size, 0);
  OffsetToIdMap ag_offsetted_map;
  for (const auto& [offset, partition_id] : ag_map) {
    if (offset < 0 || offset > ag_extent - ag_shard_size) {
      VLOG(2) << "ag offset: " << offset
              << " is out of range of ag_offset_spec: "
              << ag_offset_spec.ToString();
      return std::nullopt;
    }
    if (!ag_offsetted_map
             .try_emplace(ag_offset_spec.start_offset + offset, partition_id)
             .second) {
      return std::nullopt;
    }
  }
  return ag_offsetted_map;
}

enum class SlotKind { kPad, kLocalShard, kRemotePermute };

struct SliceSlot {
  SlotKind kind;
  int64_t split_size;
  std::pair<int64_t, int64_t> source_target;
};

// Generates a broadcast of `pad`'s scalar zero literal (preserving signed zero
// bits and `ds_instr`'s layout) with `split_dim` resized to `size`.
HloInstruction* CreateBroadcastConstant(HloComputation* computation,
                                        const HloInstruction& ds_instr,
                                        const HloPadInstruction& pad,
                                        int64_t split_dim, int64_t size) {
  if (size <= 0) {
    VLOG(2) << "Size of the split dimension must be positive, but got " << size;
    return nullptr;
  }
  Shape new_shape = ds_instr.shape();
  new_shape.set_dimensions(split_dim, size);

  HloInstruction* zero = computation->AddInstruction(
      HloInstruction::CreateConstant(pad.padding_value()->literal().Clone()));
  return computation->AddInstruction(
      HloInstruction::CreateBroadcast(new_shape, zero, {}));
}

// Checks if the total size of operands along the split dimension matches the
// target shape and creates a concatenate instruction.
HloInstruction* CheckSizeAndConcatenate(
    HloComputation* computation, absl::Span<HloInstruction* const> operands,
    const Shape& shape, int64_t split_dim) {
  int64_t total_split_dim_size = 0;
  for (HloInstruction* operand : operands) {
    if (operand == nullptr) {
      return nullptr;
    }
    VLOG(10) << "CheckSizeAndConcatenate operand: " << operand->ToString();
    total_split_dim_size += operand->shape().dimensions(split_dim);
  }

  if (total_split_dim_size != shape.dimensions(split_dim)) {
    VLOG(2) << "Concatenate size mismatch along dimension " << split_dim
            << ". Expected total size: " << shape.dimensions(split_dim)
            << ", but got: " << total_split_dim_size;
    return nullptr;
  }

  return computation->AddInstruction(
      HloInstruction::CreateConcatenate(shape, operands, split_dim));
}

// Validates a single replica group and produces a declarative slice plan
// without mutating the computation.
std::optional<std::vector<SliceSlot>> ProcessReplicaGroup(
    const OffsetToIdMap& ds_map, const OffsetToIdMap& ag_map,
    const OffsetSpec& ag_offset_spec, const HloDynamicSliceInstruction& ds,
    const HloPadInstruction& pad, const HloAllGatherInstruction& ag,
    int64_t split_dim, int64_t num_partitions,
    std::vector<bool>& seen_ag_partitions, size_t& total_ag_participants,
    int64_t& target_partition_id) {
  const int64_t pad_dim = pad.shape().dimensions(split_dim);
  const int64_t ds_result_size = ds.shape().dimensions(split_dim);
  const int64_t ag_result_size = ag.shape().dimensions(split_dim);
  const int64_t ag_shard_size = ag.operand(0)->shape().dimensions(split_dim);
  if (ds_map.empty() || ag_map.empty() || ds_map.size() != ag_map.size() ||
      ag_shard_size <= 0 || pad_dim < ds_result_size ||
      ds_result_size < ag_result_size || ag_result_size % ag_shard_size != 0 ||
      static_cast<int64_t>(ag_map.size()) != ag_result_size / ag_shard_size) {
    return std::nullopt;
  }

  std::vector<bool> in_group(num_partitions, false);
  for (const auto& [offset, pid] : ag_map) {
    if (pid < 0 || pid >= num_partitions || seen_ag_partitions[pid]) {
      return std::nullopt;
    }
    seen_ag_partitions[pid] = true;
    in_group[pid] = true;
    ++total_ag_participants;
  }

  std::optional<OffsetToIdMap> ag_offsetted_map =
      GenerateOffsettedAgMap(ag_map, ag_offset_spec, ag_shard_size);
  if (!ag_offsetted_map.has_value()) {
    return std::nullopt;
  }
  const int64_t ag_start = ag_offset_spec.start_offset;
  const int64_t ag_end = ag_offset_spec.end_offset;
  int64_t expected_ag_offset = ag_start;
  for (const auto& [ag_offset, pid] : *ag_offsetted_map) {
    if (ag_offset != expected_ag_offset) {
      return std::nullopt;
    }
    expected_ag_offset += ag_shard_size;
  }
  if (expected_ag_offset != ag_end) {
    return std::nullopt;
  }

  // Clamp every partition's DynamicSlice start to [0, pad_dim - ds_result_size]
  // and require a single target partition covering [ag_start, ag_end), with all
  // non-target partitions strictly disjoint from [ag_start, ag_end).
  const int64_t max_start = pad_dim - ds_result_size;
  target_partition_id = -1;
  int64_t start_offset = 0;
  std::vector<bool> seen_ds(num_partitions, false);
  for (const auto& [raw_offset, pid] : ds_map) {
    if (pid < 0 || pid >= num_partitions || !in_group[pid] || seen_ds[pid]) {
      return std::nullopt;
    }
    seen_ds[pid] = true;
    const int64_t clamped_start =
        std::clamp<int64_t>(raw_offset, int64_t{0}, max_start);
    const int64_t clamped_end = clamped_start + ds_result_size;
    if (clamped_start < ag_end && ag_start < clamped_end) {
      if (target_partition_id != -1 || clamped_start > ag_start ||
          clamped_end < ag_end) {
        return std::nullopt;
      }
      target_partition_id = pid;
      start_offset = clamped_start;
    }
  }
  if (target_partition_id == -1) {
    return std::nullopt;
  }

  const int64_t end_offset = start_offset + ds_result_size;
  std::vector<SliceSlot> slots;
  slots.reserve(ag_offsetted_map->size() + 2);
  if (ag_start > start_offset) {
    slots.push_back({SlotKind::kPad, ag_start - start_offset, {0, 0}});
  }
  for (const auto& [ag_offset, pid] : *ag_offsetted_map) {
    if (pid != target_partition_id) {
      slots.push_back({SlotKind::kRemotePermute,
                       ag_shard_size,
                       {pid, target_partition_id}});
    } else {
      slots.push_back({SlotKind::kLocalShard, ag_shard_size, {0, 0}});
    }
  }
  if (ag_end < end_offset) {
    slots.push_back({SlotKind::kPad, end_offset - ag_end, {0, 0}});
  }
  return slots;
}

// Validates that every replica group produced an identical (kind, split_size)
// slot layout and materializes the condensed HLO operands once.
std::optional<std::vector<HloInstruction*>> CondenseSliceSlots(
    absl::Span<const std::vector<SliceSlot>> group_slots,
    HloComputation* computation, const HloDynamicSliceInstruction& ds,
    const HloPadInstruction& pad, HloAllGatherInstruction& ag,
    int64_t split_dim) {
  if (group_slots.empty() || group_slots[0].empty()) {
    return std::nullopt;
  }
  const size_t num_groups = group_slots.size();
  const size_t num_slots = group_slots[0].size();
  for (size_t rg_idx = 1; rg_idx < num_groups; ++rg_idx) {
    if (group_slots[rg_idx].size() != num_slots) {
      return std::nullopt;
    }
    for (size_t i = 0; i < num_slots; ++i) {
      if (group_slots[rg_idx][i].kind != group_slots[0][i].kind ||
          group_slots[rg_idx][i].split_size != group_slots[0][i].split_size) {
        return std::nullopt;
      }
    }
  }

  std::vector<HloInstruction*> final_instrs;
  final_instrs.reserve(num_slots);
  for (size_t i = 0; i < num_slots; ++i) {
    const SliceSlot& spec = group_slots[0][i];
    switch (spec.kind) {
      case SlotKind::kPad: {
        HloInstruction* pad_bcast = CreateBroadcastConstant(
            computation, ds, pad, split_dim, spec.split_size);
        if (pad_bcast == nullptr) {
          return std::nullopt;
        }
        final_instrs.push_back(pad_bcast);
        break;
      }
      case SlotKind::kLocalShard:
        final_instrs.push_back(ag.mutable_operand(0));
        break;
      case SlotKind::kRemotePermute: {
        std::vector<std::pair<int64_t, int64_t>> pairs;
        pairs.reserve(num_groups);
        for (size_t rg_idx = num_groups; rg_idx-- > 0;) {
          pairs.push_back(group_slots[rg_idx][i].source_target);
        }
        HloInstruction* cp =
            computation->AddInstruction(HloInstruction::CreateCollectivePermute(
                ag.operand(0)->shape(), ag.mutable_operand(0), pairs,
                ag.channel_id()));
        CopyCollectiveGroupKey(ag, *cp);
        final_instrs.push_back(cp);
        break;
      }
    }
  }
  return final_instrs;
}

// Creates the final Concatenate and Select instructions.
std::optional<HloInstruction*> CreateFinalConcatAndSelect(
    HloComputation* computation, absl::Span<HloInstruction* const> final_instrs,
    HloDynamicSliceInstruction& ds, const HloPadInstruction& pad,
    int64_t split_dim, absl::Span<const int64_t> selection_list,
    int64_t ds_result_size) {
  HloInstruction* concat =
      CheckSizeAndConcatenate(computation, final_instrs, ds.shape(), split_dim);
  if (concat == nullptr) {
    VLOG(2) << "No valid concat instruction generated.";
    return std::nullopt;
  }

  HloInstruction* pred =
      AddPredInstrBasedOnPartitionIdAndList(computation, selection_list);
  if (pred == nullptr) {
    VLOG(2) << "No valid pred instruction generated.";
    return std::nullopt;
  }

  HloInstruction* zero_filled =
      CreateBroadcastConstant(computation, ds, pad, split_dim, ds_result_size);
  if (zero_filled == nullptr) {
    VLOG(2) << "No valid zero_filled instruction generated.";
    return std::nullopt;
  }

  return computation->AddInstruction(HloInstruction::CreateTernary(
      ds.shape(), HloOpcode::kSelect, pred, concat, zero_filled));
}

// Replaces a validated DynamicSlice(Pad(AllGather)) pattern with
// CollectivePermute, Concatenate, and Select instructions.
std::optional<HloInstruction*> InsertCollectivePermuteInstrSet(
    const PartitionOffsetSpec& ds_offset_partition_spec,
    const PartitionOffsetSpec& ag_offset_partition_spec,
    const OffsetSpec& ag_offset_spec, HloDynamicSliceInstruction& ds,
    const HloPadInstruction& pad, HloAllGatherInstruction& ag,
    int64_t split_dim, int64_t num_partitions) {
  const int64_t ds_result_size = ds.shape().dimensions(split_dim);
  const int64_t ag_result_size = ag.shape().dimensions(split_dim);
  HloComputation* computation = ds.parent();

  if (num_partitions <= 1 || ds_result_size <= 0 ||
      ds_result_size < ag_result_size ||
      ds_offset_partition_spec.per_replica_group_offsets.size() !=
          ag_offset_partition_spec.per_replica_group_offsets.size() ||
      ds_offset_partition_spec.per_replica_group_offsets.empty()) {
    return std::nullopt;
  }

  const size_t num_groups =
      ds_offset_partition_spec.per_replica_group_offsets.size();
  std::vector<int64_t> selection_list(num_partitions, 0);
  std::vector<bool> seen_ag_partitions(num_partitions, false);
  size_t total_ag_participants = 0;
  std::vector<std::vector<SliceSlot>> group_slots;
  group_slots.reserve(num_groups);

  for (size_t rg_idx = 0; rg_idx < num_groups; ++rg_idx) {
    const OffsetToIdMap& ds_map =
        ds_offset_partition_spec.per_replica_group_offsets[rg_idx];
    const OffsetToIdMap& ag_map =
        ag_offset_partition_spec.per_replica_group_offsets[rg_idx];

    int64_t target_partition_id = -1;
    std::optional<std::vector<SliceSlot>> slots = ProcessReplicaGroup(
        ds_map, ag_map, ag_offset_spec, ds, pad, ag, split_dim, num_partitions,
        seen_ag_partitions, total_ag_participants, target_partition_id);
    if (!slots.has_value() || target_partition_id < 0 ||
        target_partition_id >= num_partitions ||
        selection_list[target_partition_id] != 0) {
      VLOG(2) << "Failed to process replica group " << rg_idx;
      return std::nullopt;
    }
    selection_list[target_partition_id] = 1;
    group_slots.push_back(std::move(*slots));
  }

  // Non-participating partitions produce +0.0 from AllGather, which differs
  // from a negative-zero pad literal inside the AllGather slice window.
  if (total_ag_participants < static_cast<size_t>(num_partitions) &&
      !pad.padding_value()->literal().Equal(
          LiteralUtil::Zero(ds.shape().element_type()),
          /*layout_sensitive=*/false)) {
    return std::nullopt;
  }

  std::optional<std::vector<HloInstruction*>> final_instrs_optional =
      CondenseSliceSlots(group_slots, computation, ds, pad, ag, split_dim);
  if (!final_instrs_optional.has_value()) {
    VLOG(2) << "Failed to condense slice slots.";
    return std::nullopt;
  }

  return CreateFinalConcatAndSelect(computation, *final_instrs_optional, ds,
                                    pad, split_dim, selection_list,
                                    ds_result_size);
}

// Checks if the partition IDs within each replica group in the
// `PartitionOffsetSpec` are strictly ascending
// with respect to their corresponding offsets.
bool IsAscendingPartitionId(const PartitionOffsetSpec& offset_spec) {
  if (offset_spec.per_replica_group_offsets.empty()) {
    return false;
  }
  for (const OffsetToIdMap& rg_map : offset_spec.per_replica_group_offsets) {
    if (rg_map.empty()) {
      return false;
    }
    OffsetToIdMap::const_iterator iter = rg_map.begin();
    int64_t prev_id = iter->second;

    while (++iter != rg_map.end()) {
      if (iter->second <= prev_id) {
        VLOG(2) << "iter->second: " << iter->second
                << " <= prev_id: " << prev_id;
        return false;
      }
      prev_id = iter->second;
    }
  }
  return true;
}

bool HasMatchingLayoutAttrs(const Shape& a, const Shape& b) {
  const int64_t a_space = a.has_layout() ? a.layout().memory_space() : 0;
  const int64_t b_space = b.has_layout() ? b.layout().memory_space() : 0;
  const int64_t a_bits = a.has_layout() ? a.layout().element_size_in_bits() : 0;
  const int64_t b_bits = b.has_layout() ? b.layout().element_size_in_bits() : 0;
  return a_space == b_space && a_bits == b_bits;
}

const HloInstruction* StripSameTypeIndexWrappers(const HloInstruction* hlo) {
  while (hlo->opcode() == HloOpcode::kBitcast ||
         hlo->opcode() == HloOpcode::kReshape ||
         hlo->opcode() == HloOpcode::kCopy) {
    if (hlo->shape().element_type() !=
        hlo->operand(0)->shape().element_type()) {
      return nullptr;
    }
    hlo = hlo->operand(0);
  }
  return hlo;
}

std::optional<int64_t> ResolveValidatedIndex(const HloInstruction* hlo,
                                             int64_t partition_id) {
  hlo = StripSameTypeIndexWrappers(hlo);
  if (hlo == nullptr) {
    return std::nullopt;
  }
  if (hlo->opcode() == HloOpcode::kPartitionId) {
    return partition_id;
  }
  if (hlo->opcode() != HloOpcode::kDynamicSlice || hlo->operand_count() != 2 ||
      hlo->dynamic_slice_sizes().size() != 1 ||
      hlo->dynamic_slice_sizes()[0] != 1) {
    return std::nullopt;
  }
  const HloInstruction* table = hlo->operand(0);
  const PrimitiveType ty = table->shape().element_type();
  if (!IsStaticArray(table->shape()) ||
      table->shape().dimensions().size() != 1 || (ty != S32 && ty != U32)) {
    return std::nullopt;
  }
  std::optional<int64_t> idx =
      ResolveValidatedIndex(hlo->operand(1), partition_id);
  if (!idx.has_value() || *idx < 0 || *idx >= table->shape().dimensions(0)) {
    return std::nullopt;
  }
  if (table->opcode() == HloOpcode::kIota) {
    return *idx;
  }
  if (table->opcode() == HloOpcode::kConstant &&
      Cast<HloConstantInstruction>(table)->HasLiteral()) {
    return table->literal().GetIntegralAsS64({*idx});
  }
  return std::nullopt;
}

// Rejects zero-multiplier kMultiply, type-changing kBitcast, out-of-bounds
// (clamped) kIota/kConstant table lookups, and S32/U32 modular multiply wrap.
bool ValidateDsOffsetIndexExpr(const HloAllGatherInstruction& all_gather,
                               const HloInstruction* offset_hlo,
                               int64_t split_dim_size, int64_t num_partitions) {
  if (split_dim_size <= 0 || all_gather.replica_groups().empty()) {
    return false;
  }
  offset_hlo = StripSameTypeIndexWrappers(offset_hlo);
  if (offset_hlo == nullptr) {
    return false;
  }
  const PrimitiveType idx_ty = offset_hlo->shape().element_type();
  if (idx_ty != S32 && idx_ty != U32) {
    return false;
  }
  const int64_t idx_min =
      idx_ty == U32 ? int64_t{0}
                    : static_cast<int64_t>(std::numeric_limits<int32_t>::min());
  const int64_t idx_max =
      idx_ty == U32 ? static_cast<int64_t>(std::numeric_limits<uint32_t>::max())
                    : static_cast<int64_t>(std::numeric_limits<int32_t>::max());

  int64_t multiplier = 1;
  const HloInstruction* lookup_hlo = offset_hlo;
  if (offset_hlo->opcode() == HloOpcode::kMultiply) {
    if (!ShapeUtil::IsEffectiveScalar(offset_hlo->shape())) {
      return false;
    }
    const int const_idx = offset_hlo->operand(0)->IsConstant()   ? 0
                          : offset_hlo->operand(1)->IsConstant() ? 1
                                                                 : -1;
    if (const_idx < 0) {
      return false;
    }
    const HloInstruction* c = offset_hlo->operand(const_idx);
    if (!Cast<HloConstantInstruction>(c)->HasLiteral() ||
        !ShapeUtil::IsEffectiveScalar(c->shape()) ||
        c->shape().element_type() != idx_ty) {
      return false;
    }
    std::vector<int64_t> zero_index(c->shape().dimensions().size(), 0);
    std::optional<int64_t> mult = c->literal().GetIntegralAsS64(zero_index);
    if (!mult.has_value() || *mult == 0 || *mult < idx_min || *mult > idx_max ||
        split_dim_size % *mult != 0) {
      return false;
    }
    multiplier = *mult;
    lookup_hlo = offset_hlo->operand(1 - const_idx);
  }

  const int64_t split_dim = all_gather.all_gather_dimension();
  const int64_t ag_shard_size =
      all_gather.operand(0)->shape().dimensions(split_dim);
  const int64_t ag_result_size = all_gather.shape().dimensions(split_dim);
  if (ag_shard_size <= 0 || ag_result_size % ag_shard_size != 0) {
    return false;
  }
  const int64_t expected_group_size = ag_result_size / ag_shard_size;
  for (const auto& group : all_gather.replica_groups()) {
    if (group.replica_ids().empty() ||
        group.replica_ids_size() != expected_group_size) {
      return false;
    }
    for (int64_t pid : group.replica_ids()) {
      if (pid < 0 || pid >= num_partitions) {
        return false;
      }
      std::optional<int64_t> raw = ResolveValidatedIndex(lookup_hlo, pid);
      if (!raw.has_value() || *raw < idx_min || *raw > idx_max) {
        return false;
      }
      if ((*raw > 0 && multiplier > 0 && *raw > idx_max / multiplier) ||
          (*raw > 0 && multiplier < 0 && multiplier < idx_min / *raw) ||
          (*raw < 0 && multiplier > 0 && *raw < idx_min / multiplier) ||
          (*raw < 0 && multiplier < 0 && *raw < idx_max / multiplier)) {
        return false;
      }
    }
  }
  return true;
}

struct PatternMatchResult {
  HloDynamicSliceInstruction* ds;
  HloPadInstruction* pad;
  HloAllGatherInstruction* ag;
};

// Matches the pattern: dynamic-slice(pad(all-gather)).
std::optional<PatternMatchResult> MatchDynamicSlicePadAllGather(
    HloInstruction* dynamic_slice_hlo) {
  HloDynamicSliceInstruction* dynamic_slice =
      Cast<HloDynamicSliceInstruction>(dynamic_slice_hlo);

  const HloModuleConfig& config = dynamic_slice->GetModule()->config();
  if (config.num_partitions() <= 1 || config.replica_count() != 1) {
    VLOG(2) << "Do not match module with num_partitions: "
            << config.num_partitions()
            << " and replica_count: " << config.replica_count();
    return std::nullopt;
  }

  HloInstruction* pad_hlo = nullptr;
  HloInstruction* ag_hlo = nullptr;
  if (!MatchDsPadAllGather(dynamic_slice, &pad_hlo, &ag_hlo)) {
    VLOG(2) << "Do not find pattern for dynamic-slice(pad(ag)) "
            << dynamic_slice->ToString();
    return std::nullopt;
  }
  auto* pad = Cast<HloPadInstruction>(pad_hlo);
  auto* ag = Cast<HloAllGatherInstruction>(ag_hlo);
  if (ag->operand_count() != 1 || ag->constrain_layout() ||
      pad->HasControlDependencies() || ag->HasControlDependencies()) {
    return std::nullopt;
  }
  const Shape& ds_shape = dynamic_slice->shape();
  const Shape& op_shape = ag->operand(0)->shape();
  if (!IsStaticArray(ds_shape) || !IsStaticArray(pad->shape()) ||
      !IsStaticArray(ag->shape()) || !IsStaticArray(op_shape) ||
      op_shape.element_type() != ds_shape.element_type() ||
      !HasMatchingLayoutAttrs(op_shape, ds_shape)) {
    return std::nullopt;
  }

  // Match all-gather for kFlattenedID collective mode.
  absl::StatusOr<CollectiveOpGroupMode> mode = GetCollectiveOpGroupMode(ag_hlo);
  if (!mode.ok() ||
      *mode != CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID) {
    VLOG(2) << "AG does not use global device ids or channel id "
            << ag_hlo->ToString();
    return std::nullopt;
  }

  return PatternMatchResult{dynamic_slice, pad, ag};
}

std::optional<SplitDimSpec> ExtractAndValidateSplitDim(
    const HloDynamicSliceInstruction& dynamic_slice,
    const HloPadInstruction& pad, const HloAllGatherInstruction& all_gather) {
  const Shape& ds_shape = dynamic_slice.shape();
  const Shape& pad_shape = pad.shape();
  const Shape& ag_shape = all_gather.shape();
  const Shape& op_shape = all_gather.operand(0)->shape();
  const int64_t rank = ds_shape.dimensions().size();
  if (pad_shape.dimensions().size() != rank ||
      ag_shape.dimensions().size() != rank ||
      op_shape.dimensions().size() != rank ||
      dynamic_slice.operand_count() != rank + 1 ||
      dynamic_slice.dynamic_slice_sizes().size() != rank) {
    return std::nullopt;
  }
  std::optional<SplitDimSpec> split_dim_spec =
      ExtractSplitDimSpec(dynamic_slice, /*allow_multiple_split_dims=*/false);
  if (!split_dim_spec.has_value() || split_dim_spec->split_dims.size() != 1 ||
      split_dim_spec->split_dim < 0 || split_dim_spec->split_dim >= rank ||
      split_dim_spec->split_dim != all_gather.all_gather_dimension() ||
      split_dim_spec->split_dim_size <= 0) {
    VLOG(2) << "In dynamic-slice(pad(ag)), ds split dim mismatch "
            << dynamic_slice.ToString()
            << " or all-gather: " << all_gather.ToString();
    return std::nullopt;
  }
  const int64_t split_dim = split_dim_spec->split_dim;
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (dim == split_dim) {
      if (pad_shape.dimensions(dim) < ds_shape.dimensions(dim) ||
          ds_shape.dimensions(dim) < ag_shape.dimensions(dim)) {
        return std::nullopt;
      }
      continue;
    }
    if (ds_shape.dimensions(dim) != pad_shape.dimensions(dim) ||
        pad_shape.dimensions(dim) != ag_shape.dimensions(dim) ||
        ag_shape.dimensions(dim) != op_shape.dimensions(dim)) {
      return std::nullopt;
    }
  }
  return split_dim_spec;
}

struct PadOffsetSpecs {
  OffsetSpec pad_spec;
  OffsetSpec ag_spec;
};

std::optional<PadOffsetSpecs> ValidatePadAndExtractOffsetSpecs(
    const HloPadInstruction& pad, const Shape& ds_shape, const Shape& ag_shape,
    int64_t split_dim) {
  // Match the pad pattern:
  //  1. pad only one edge padding.
  //  2. pad value == 0.
  //  3. pad size is large: ag result size is smaller than dynamic slice result
  //  size.
  std::optional<OffsetSpec> valid_pad_split_offset_spec =
      ExtractValidPadSpec(pad, ds_shape, ag_shape, split_dim);
  if (!valid_pad_split_offset_spec.has_value()) {
    VLOG(2) << "No valid pad spec found from pad: " << pad.ToString();
    return std::nullopt;
  }

  VLOG(10) << "Got valid_pad_split_offset_spec: "
           << valid_pad_split_offset_spec->ToString();

  // Extracts the all-gather offset spec on dynamic-slice result
  // based on pad split offset spec.
  std::optional<OffsetSpec> valid_ag_split_offset_spec =
      ExtractAllGatherOffsetSpec(*valid_pad_split_offset_spec, pad.shape(),
                                 ag_shape);
  if (!valid_ag_split_offset_spec.has_value()) {
    VLOG(2) << "No valid ag offset spec found from pad: " << pad.ToString()
            << ", ag: " << ag_shape.ToString() << ", pad_spec"
            << valid_pad_split_offset_spec->ToString();
    return std::nullopt;
  }

  VLOG(10) << "Got valid_ag_offset_spec: "
           << valid_ag_split_offset_spec->ToString();
  return PadOffsetSpecs{*valid_pad_split_offset_spec,
                        *valid_ag_split_offset_spec};
}

struct OffsetToIdMaps {
  PartitionOffsetSpec ds_spec;
  PartitionOffsetSpec ag_spec;
};

std::optional<OffsetToIdMaps> ExtractAndValidateOffsetToIdMaps(
    const HloDynamicSliceInstruction& dynamic_slice,
    const HloAllGatherInstruction& all_gather, const HloModuleConfig& config,
    int64_t split_dim, int64_t split_dim_size) {
  if (dynamic_slice.shape().dimensions(split_dim) <
          all_gather.shape().dimensions(split_dim) ||
      !ValidateDsOffsetIndexExpr(all_gather,
                                 dynamic_slice.operand(split_dim + 1),
                                 split_dim_size, config.num_partitions())) {
    return std::nullopt;
  }

  MapIdToTableOffset map_partition_id = [&](const HloInstruction* hlo,
                                            int64_t id) {
    return HloPredicateIsOp<HloOpcode::kPartitionId>(hlo) ? id : -1;
  };

  // Extracts the dynamic slice offset to partition id map spec on the split
  // dimension.
  std::optional<PartitionOffsetSpec> ds_offset_spec =
      GetIndicesSpecForDynamicSliceWithMultiply(
          &all_gather, dynamic_slice.operand(split_dim + 1), map_partition_id,
          split_dim_size);

  if (!ds_offset_spec.has_value()) {
    VLOG(2) << "No valid ds offset spec on dim" << split_dim
            << "found from dynamic_slice: " << dynamic_slice.ToString()
            << " and ag: " << all_gather.ToString();
    return std::nullopt;
  }

  // Validates the dynamic slice offset list is based on ascending order.
  if (!IsAscendingPartitionId(*ds_offset_spec)) {
    VLOG(2) << "Partition id is not ascending on dim" << split_dim
            << "found from dynamic_slice: " << dynamic_slice.ToString()
            << " ds offset: "
            << dynamic_slice.operand(split_dim + 1)->ToString();
    return std::nullopt;
  }

  // Extracts the all-gather offset to partition id map spec on the split
  // dimension.
  std::optional<PartitionOffsetSpec> ag_offset_partition_spec =
      ExtractPartitionOffsetSpec(&all_gather, config.num_partitions());
  if (!ag_offset_partition_spec.has_value()) {
    VLOG(2) << "No valid ag offset spec " << all_gather.ToString()
            << " with num_partitions " << config.num_partitions();
    return std::nullopt;
  }
  return OffsetToIdMaps{*ds_offset_spec, *ag_offset_partition_spec};
}

}  // namespace

absl::Status AllGatherPadDsSimplifierVisitor::HandleDynamicSlice(
    HloInstruction* dynamic_slice_hlo) {
  // Match and Validate Pattern.
  std::optional<PatternMatchResult> pattern_match =
      MatchDynamicSlicePadAllGather(dynamic_slice_hlo);

  if (!pattern_match.has_value()) {
    return absl::OkStatus();
  }
  HloDynamicSliceInstruction* dynamic_slice = pattern_match->ds;
  HloPadInstruction* pad = pattern_match->pad;
  HloAllGatherInstruction* all_gather = pattern_match->ag;

  // Extract and Validate Split Dimension.
  std::optional<SplitDimSpec> split_dim_spec =
      ExtractAndValidateSplitDim(*dynamic_slice, *pad, *all_gather);
  if (!split_dim_spec.has_value()) {
    return absl::OkStatus();
  }
  const int64_t split_dim = split_dim_spec->split_dim;
  const int64_t split_dim_size = split_dim_spec->split_dim_size;

  // Validate Pad and Extract Offset Specs.
  std::optional<PadOffsetSpecs> offset_specs = ValidatePadAndExtractOffsetSpecs(
      *pad, dynamic_slice->shape(), all_gather->shape(), split_dim);
  if (!offset_specs.has_value()) {
    return absl::OkStatus();
  }
  const OffsetSpec& ag_spec = offset_specs->ag_spec;

  // Extract and Validate OffsetToId Maps.
  const HloModuleConfig& config = dynamic_slice->GetModule()->config();
  std::optional<OffsetToIdMaps> offset_maps = ExtractAndValidateOffsetToIdMaps(
      *dynamic_slice, *all_gather, config, split_dim, split_dim_size);
  if (!offset_maps.has_value()) {
    return absl::OkStatus();
  }
  const PartitionOffsetSpec& ds_offset_spec = offset_maps->ds_spec;
  const PartitionOffsetSpec& ag_offset_spec = offset_maps->ag_spec;

  // HLO Generation.
  std::optional<HloInstruction*> selected = InsertCollectivePermuteInstrSet(
      ds_offset_spec, ag_offset_spec, ag_spec, *dynamic_slice, *pad,
      *all_gather, split_dim, config.num_partitions());

  if (!selected.has_value() || *selected == nullptr) {
    VLOG(2) << "No valid selected found from dynamic_slice: "
            << dynamic_slice->ToString()
            << " and ag: " << all_gather->ToString();
    return absl::OkStatus();
  }

  // Replacement.
  dynamic_slice->SetupDerivedInstruction(*selected);
  ClearCollectiveGroupKey(**selected);
  return ReplaceInstruction(dynamic_slice, *selected,
                            /*preserve_frontend_attributes=*/false);
}

absl::StatusOr<bool> AllGatherPadDsSimplifier::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    AllGatherPadDsSimplifierVisitor visitor;
    ABSL_RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }
  return changed;
}

}  // namespace xla
