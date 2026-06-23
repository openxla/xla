/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/codegen/tiling/experimental/tiled_hlo.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/codegen/tiling/experimental/tile.h"
#include "xla/codegen/tiling/experimental/tile_propagation.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/name_uniquer.h"
#include "xla/util.h"

namespace xla::gpu::experimental {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;

std::string TiledHloInstruction::ToString(
    absl::string_view field_separator) const {
  std::stringstream ss;
  ss << "hlo: " << hlo_->ToString() << field_separator;
  ss << "tile: " << tile().ToString();
  for (const auto& [index, region] : llvm::enumerate(regions_)) {
    ss << field_separator << "region #" << index << " {";
    for (const auto& instruction : region) {
      ss << field_separator << instruction->ToString(field_separator);
    }
    ss << field_separator << "}";
  }
  return ss.str();
}

llvm::SmallVector<const TiledHloInstruction*, 2>
TiledHloInstruction::runtime_variables() const {
  llvm::SmallVector<const TiledHloInstruction*, 2> runtime_variables;
  if (auto dyn_slice = DynCast<HloDynamicSliceInstruction>(hlo_)) {
    // `operands_` might be empty and inconsistent with `hlo_->operand_count()`
    // if the instruction lies outside the fusion boundary (we skip populating
    // its operands during traversal).
    for (int i = dyn_slice->first_index_operand_number(); i < operands_.size();
         ++i) {
      runtime_variables.push_back(operands_[i]);
    }
  }
  return runtime_variables;
}
class TiledHloScopeCache {
 public:
  TiledHloScopeCache() { PushScope(); }

  // Finds a cached TiledHloInstruction matching the given HloInstruction
  // and Tile shape within the active scoped cache stack, traversing from the
  // innermost scope outwards.
  TiledHloInstruction* Find(const HloInstruction* hlo, const Tile& tile) const {
    auto key = std::make_pair(hlo, tile);
    for (auto it = active_scopes_.rbegin(); it != active_scopes_.rend(); ++it) {
      if (auto map_it = it->find(key); map_it != it->end()) {
        return map_it->second;
      }
    }
    return nullptr;
  }

  // Inserts a tiled instruction into the innermost (current active) scope.
  void Insert(const HloInstruction* hlo, const Tile& tile,
              TiledHloInstruction* tiled_hlo) {
    CHECK(!active_scopes_.empty());
    active_scopes_.back()[std::make_pair(hlo, tile)] = tiled_hlo;
  }

  void PushScope() { active_scopes_.emplace_back(); }

  void PopScope() {
    CHECK(!active_scopes_.empty());
    active_scopes_.pop_back();
  }

 private:
  using CacheMap = absl::flat_hash_map<std::pair<const HloInstruction*, Tile>,
                                       TiledHloInstruction*>;
  std::vector<CacheMap> active_scopes_;
};

static void SimplifyTiles(TiledHloInstruction* instruction) {
  class Tile tile = instruction->tile();
  tile.Simplify();
  instruction->set_tile(std::move(tile));
  for (const auto& region : instruction->hlo_regions()) {
    for (const auto& nested : region) {
      SimplifyTiles(nested.get());
    }
  }
}

// If the operand represents a dynamic/runtime variable (e.g. index offset in
// dynamic slice), maps it to its tiled instruction representation and valid
// value bounds.
static void MaybeRegisterRTVariable(
    const HloInstruction* hlo, int operand_id,
    const TiledHloInstruction* operand_tiled_hlo, TilingSpace& tiling_space,
    absl::flat_hash_map<int64_t,
                        std::pair<const TiledHloInstruction*, Interval>>&
        rt_symbol_to_tiled_hlo) {
  std::optional<const TilingSpace::RTVarInfo*> rt_var_info =
      tiling_space.GetRTVarInfo(*hlo, operand_id);
  if (rt_var_info.has_value()) {
    rt_symbol_to_tiled_hlo.insert(std::make_pair(
        rt_var_info.value()->id + tiling_space.num_dimensions(),
        std::make_pair(operand_tiled_hlo, rt_var_info.value()->bounds)));
  }
}

// Forward declared to allow mutual recursion.
static absl::StatusOr<TiledHloRegion> TileInstruction(
    const HloInstruction* hlo, const Tile& tile, const HloFusionAdaptor& fusion,
    TilingSpace& tiling_space, TiledHloScopeCache& cache,
    absl::flat_hash_map<int64_t,
                        std::pair<const TiledHloInstruction*, Interval>>&
        rt_symbol_to_tiled_hlo);

// Helper function that recursively tiles a list of operand instructions and
// links them to the parent tiled instruction. Also registers any
// dynamic/runtime variables.
static absl::StatusOr<TiledHloRegion> TileOperandsAndLink(
    TiledHloInstruction* tiled_instr, llvm::ArrayRef<Tile> operands_tiles,
    llvm::ArrayRef<HloInstructionAdaptor> operands_adaptors,
    llvm::ArrayRef<int> operand_indices, const HloFusionAdaptor& fusion,
    TilingSpace& tiling_space, TiledHloScopeCache& cache,
    absl::flat_hash_map<int64_t,
                        std::pair<const TiledHloInstruction*, Interval>>&
        rt_symbol_to_tiled_hlo) {
  TiledHloRegion tiled_operand_instructions;
  const HloInstruction* hlo = tiled_instr->hlo();

  for (int operand_id : operand_indices) {
    const Tile& operand_tile = operands_tiles[operand_id];
    const HloInstruction* operand_hlo =
        &operands_adaptors[operand_id].instruction();
    ASSIGN_OR_RETURN(
        TiledHloRegion op_region,
        TileInstruction(operand_hlo, operand_tile, fusion, tiling_space, cache,
                        rt_symbol_to_tiled_hlo));
    absl::c_move(op_region, std::back_inserter(tiled_operand_instructions));
  }

  for (int operand_id = 0; operand_id < operands_adaptors.size();
       ++operand_id) {
    const Tile& operand_tile = operands_tiles[operand_id];
    const HloInstruction* operand_hlo =
        &operands_adaptors[operand_id].instruction();

    TiledHloInstruction* operand_tiled_hlo =
        cache.Find(operand_hlo, operand_tile);
    CHECK(operand_tiled_hlo != nullptr);
    tiled_instr->AddOperand(operand_tiled_hlo);

    MaybeRegisterRTVariable(hlo, operand_id, operand_tiled_hlo, tiling_space,
                            rt_symbol_to_tiled_hlo);
  }

  return tiled_operand_instructions;
}

/// Helper function that tiles operands inside a loop body region for dot and
/// scaled dot instructions.
static absl::StatusOr<TiledHloRegion> TileDotInstruction(
    std::unique_ptr<TiledHloInstruction> tiled_instr,
    llvm::ArrayRef<Tile> operands_tiles,
    llvm::ArrayRef<HloInstructionAdaptor> operands_adaptors,
    llvm::ArrayRef<int> operand_indices, const HloFusionAdaptor& fusion,
    TilingSpace& tiling_space, TiledHloScopeCache& cache,
    absl::flat_hash_map<int64_t,
                        std::pair<const TiledHloInstruction*, Interval>>&
        rt_symbol_to_tiled_hlo) {
  // Push new scope for the loop body region.
  cache.PushScope();

  ASSIGN_OR_RETURN(
      TiledHloRegion loop_body,
      TileOperandsAndLink(tiled_instr.get(), operands_tiles, operands_adaptors,
                          operand_indices, fusion, tiling_space, cache,
                          rt_symbol_to_tiled_hlo));

  // Pop the loop body scope.
  cache.PopScope();

  tiled_instr->AddHloRegion(std::move(loop_body));

  TiledHloRegion result;
  result.push_back(std::move(tiled_instr));
  return result;
}

// Helper function that tiles each operand of a concatenate instruction inside
// its own separate region.
static absl::StatusOr<TiledHloRegion> TileConcatenateInstruction(
    std::unique_ptr<TiledHloInstruction> tiled_instr,
    llvm::ArrayRef<Tile> operands_tiles,
    llvm::ArrayRef<HloInstructionAdaptor> operands_adaptors,
    const HloFusionAdaptor& fusion, TilingSpace& tiling_space,
    TiledHloScopeCache& cache,
    absl::flat_hash_map<int64_t,
                        std::pair<const TiledHloInstruction*, Interval>>&
        rt_symbol_to_tiled_hlo) {
  const HloInstruction* hlo = tiled_instr->hlo();
  for (const auto& [operand_id, tile_and_operand] :
       llvm::enumerate(llvm::zip(operands_tiles, operands_adaptors))) {
    auto& [operand_tile, operand_adaptor] = tile_and_operand;
    const HloInstruction* operand_hlo = &operand_adaptor.instruction();

    // Push operand region cache.
    cache.PushScope();

    ASSIGN_OR_RETURN(
        TiledHloRegion op_region,
        TileInstruction(operand_hlo, operand_tile, fusion, tiling_space, cache,
                        rt_symbol_to_tiled_hlo));

    TiledHloInstruction* operand_tiled_hlo =
        cache.Find(operand_hlo, operand_tile);
    CHECK(operand_tiled_hlo != nullptr);

    tiled_instr->AddOperand(operand_tiled_hlo);
    tiled_instr->AddHloRegion(std::move(op_region));

    MaybeRegisterRTVariable(hlo, operand_id, operand_tiled_hlo, tiling_space,
                            rt_symbol_to_tiled_hlo);

    // Pop operand region cache.
    cache.PopScope();
  }

  TiledHloRegion result;
  result.push_back(std::move(tiled_instr));
  return result;
}

// Recursively traverses and tiles the HLO dataflow subgraph starting at the
// given HloInstruction. It packages subgraphs into nested regions for
// loop-introducing (kDot, kScaledDot) or branch-introducing (kConcatenate)
// instructions, maintaining a scoped cache stack to avoid duplication and
// infinite loops.
static absl::StatusOr<TiledHloRegion> TileInstruction(
    const HloInstruction* hlo, const Tile& tile, const HloFusionAdaptor& fusion,
    TilingSpace& tiling_space, TiledHloScopeCache& cache,
    absl::flat_hash_map<int64_t,
                        std::pair<const TiledHloInstruction*, Interval>>&
        rt_symbol_to_tiled_hlo) {
  // Step 1: Check active cache to prevent cycles and duplicate node generation.
  if (TiledHloInstruction* cached = cache.Find(hlo, tile)) {
    return TiledHloRegion{};
  }

  // Step 2: Leaf cases (no operands or outside fusion boundaries). Stop
  // traversal and create a leaf node.
  if (!fusion.ContainsInstruction(hlo) || hlo->operand_count() == 0) {
    auto tiled_instr = std::make_unique<TiledHloInstruction>(hlo, tile);
    cache.Insert(hlo, tile, tiled_instr.get());
    TiledHloRegion result;
    result.push_back(std::move(tiled_instr));
    return result;
  }

  // Step 3: Propagate the current tile shape backwards to determine operand
  // tile shapes.
  ASSIGN_OR_RETURN(auto operands_tiles,
                   PropagateTileToInput(tiling_space, *hlo, tile, 0));

  HloInstructionAdaptor instruction_adaptor(*hlo, &fusion);
  auto operands_adaptors = instruction_adaptor.GetOperands();
  const HloOpcode opcode = hlo->opcode();

  // Step 4: Order dependency traversal so that runtime variable/dynamic index
  // operands are processed first (stable partition).
  std::vector<int> operand_indices(operands_adaptors.size());
  for (int i = 0; i < operand_indices.size(); ++i) {
    operand_indices[i] = i;
  }
  absl::c_stable_partition(operand_indices, [&](int operand_id) {
    return tiling_space.GetRTVarInfo(*hlo, operand_id).has_value();
  });

  // Step 5: Instantiate the tiled HLO node and register it in the current scope
  // cache.
  auto tiled_instr = std::make_unique<TiledHloInstruction>(hlo, tile);
  cache.Insert(hlo, tile, tiled_instr.get());

  // Step 6: Dispatch to opcode-specific helpers to construct regions and link
  // operands.
  if (opcode == HloOpcode::kDot || opcode == HloOpcode::kScaledDot) {
    return TileDotInstruction(std::move(tiled_instr), operands_tiles,
                              operands_adaptors, operand_indices, fusion,
                              tiling_space, cache, rt_symbol_to_tiled_hlo);
  }
  if (opcode == HloOpcode::kConcatenate) {
    return TileConcatenateInstruction(std::move(tiled_instr), operands_tiles,
                                      operands_adaptors, fusion, tiling_space,
                                      cache, rt_symbol_to_tiled_hlo);
  }
  // Flat case: recursively tile and link operands, merging them directly into
  // the parent region.
  ASSIGN_OR_RETURN(
      TiledHloRegion result,
      TileOperandsAndLink(tiled_instr.get(), operands_tiles, operands_adaptors,
                          operand_indices, fusion, tiling_space, cache,
                          rt_symbol_to_tiled_hlo));
  result.push_back(std::move(tiled_instr));
  return result;
}

// Constructs a TiledHloComputation by recursively tiling the fusion roots.
// Returns a computation containing the tiled instructions in def-before-use
// order, along with root pointers and runtime variable maps.
/*static*/ absl::StatusOr<TiledHloComputation> TiledHloComputation::Tile(
    const HloFusionAdaptor& fusion, std::unique_ptr<TilingSpace> tiling_space) {
  SmallVector<const TiledHloInstruction*> roots;
  TiledHloRegion tiled_hlo_instructions;

  absl::flat_hash_map<int64_t, std::pair<const TiledHloInstruction*, Interval>>
      rt_symbol_to_tiled_hlo;
  TiledHloScopeCache cache;

  for (const auto& [root, tile] :
       llvm::zip(fusion.GetRoots(), tiling_space->tiled_roots())) {
    ASSIGN_OR_RETURN(
        TiledHloRegion root_region,
        TileInstruction(&root.instruction(), tile, fusion, *tiling_space, cache,
                        rt_symbol_to_tiled_hlo));
    absl::c_move(root_region, std::back_inserter(tiled_hlo_instructions));

    TiledHloInstruction* root_tiled_hlo = cache.Find(&root.instruction(), tile);
    CHECK(root_tiled_hlo != nullptr);
    roots.push_back(root_tiled_hlo);
  }

  for (auto& instr : tiled_hlo_instructions) {
    SimplifyTiles(instr.get());
  }

  return TiledHloComputation(std::move(tiling_space),
                             TiledHloRegion{std::move(tiled_hlo_instructions)},
                             std::move(roots),
                             std::move(rt_symbol_to_tiled_hlo));
}

// Recursively populates `tile_names` with unique names for `tiled_hlo` and
// all instructions within its regions.
void PrepopulateTileNames(
    const TiledHloInstruction* tiled_hlo, NameUniquer& name_uniquer,
    absl::flat_hash_map<const TiledHloInstruction*, std::string>& tile_names) {
  auto [_, inserted] = tile_names.try_emplace(
      tiled_hlo, name_uniquer.GetUniqueName(
                     absl::StrCat(tiled_hlo->hlo()->name(), ".tile_0")));
  if (!inserted) {
    return;
  }
  for (const auto& region : tiled_hlo->hlo_regions()) {
    for (const auto& region_instruction : region) {
      PrepopulateTileNames(region_instruction.get(), name_uniquer, tile_names);
    }
  }
}

std::string TiledHloOperandsToString(
    const TiledHloInstruction* tiled_hlo,
    const absl::flat_hash_map<const TiledHloInstruction*, std::string>&
        tile_names) {
  const HloInstruction* hlo = tiled_hlo->hlo();
  if (auto parameter = DynCast<HloParameterInstruction>(hlo)) {
    return std::to_string(parameter->parameter_number());
  }
  absl::InlinedVector<std::string, 4> operand_names;
  for (const auto& operand : tiled_hlo->operands()) {
    CHECK(tile_names.contains(operand)) << operand->hlo()->name();
    operand_names.push_back(tile_names.at(operand));
  }
  return absl::StrJoin(operand_names, ", ");
}

// Recursively prints `tiled_hlo` and all instructions within its regions.
void PrintTiledHloInstruction(
    const TiledHloInstruction* tiled_hlo,
    const absl::flat_hash_map<const TiledHloInstruction*, std::string>&
        tile_names,
    std::stringstream& ss, int indent) {
  std::string indentation(indent, ' ');
  ss << indentation << tile_names.at(tiled_hlo) << " = "
     << HloOpcodeString(tiled_hlo->hlo()->opcode()) << "("
     << TiledHloOperandsToString(tiled_hlo, tile_names) << ") "
     << tiled_hlo->tile().ToString(false) << "\n";

  for (auto const& [i, region] : llvm::enumerate(tiled_hlo->hlo_regions())) {
    ss << indentation << "region #" << i << " {\n";
    for (const auto& instruction : region) {
      PrintTiledHloInstruction(instruction.get(), tile_names, ss, indent + 2);
    }
    ss << indentation << "}\n";
  }
}

// Extracts `HloInstruction`s from a span of `HloInstructionAdaptor`s.
absl::InlinedVector<const HloInstruction*, 2> ToInstructions(
    absl::Span<const HloInstructionAdaptor> instruction_adaptors) {
  absl::InlinedVector<const HloInstruction*, 2> hlo_instructions;
  hlo_instructions.reserve(instruction_adaptors.size());
  absl::c_transform(
      instruction_adaptors, std::back_inserter(hlo_instructions),
      [&](const HloInstructionAdaptor& instr) { return &instr.instruction(); });
  return hlo_instructions;
}

std::string TiledHloComputation::ToString() const {
  std::stringstream ss;

  ss << tiling_space_->ToString() << "\n";

  NameUniquer name_uniquer("_");
  absl::flat_hash_map<const TiledHloInstruction*, std::string> tile_names;
  for (const auto& tiled_hlo : tiled_hlo_instructions_) {
    PrepopulateTileNames(tiled_hlo.get(), name_uniquer, tile_names);
  }

  ss << "Tiled HLO:\n";
  for (const auto& tiled_hlo : tiled_hlo_instructions_) {
    PrintTiledHloInstruction(tiled_hlo.get(), tile_names, ss, /*indent=*/2);
  }
  return ss.str();
}

}  // namespace xla::gpu::experimental
