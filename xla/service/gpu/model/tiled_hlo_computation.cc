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

#include "xla/service/gpu/model/tiled_hlo_computation.h"

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/service/name_uniquer.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

std::string TiledHloComputation::ToString() const {
  std::stringstream ss;
  NameUniquer name_uniquer("_");
  absl::flat_hash_map<const TiledHloInstruction*, std::string> tile_names;

  for (const auto* tiled_hlo : instructions()) {
    std::string tile_name = name_uniquer.GetUniqueName(
        absl::StrCat(tiled_hlo->hlo()->name(), ".tile_0"));
    tile_names[tiled_hlo] = tile_name;

    absl::InlinedVector<std::string, 4> operand_names;
    for (const auto& operand : tiled_hlo->operands()) {
      operand_names.push_back(tile_names.at(operand));
    }

    ss << tile_name << " = " << HloOpcodeString(tiled_hlo->hlo()->opcode())
       << "(" << absl::StrJoin(operand_names, ", ") << ")\n";

    ss << tiled_hlo->ToString() << "\n";
  }
  return ss.str();
}

void TiledHloComputation::InitializeRoots(
    const std::vector<const HloInstruction*>& roots) {
  absl::flat_hash_map<const HloInstruction*, int64_t> roots_to_output_index;
  roots_to_output_index.reserve(roots.size());
  int64_t output_index = 0;
  for (auto* root : roots) {
    roots_to_output_index[root] = output_index;
    ++output_index;
  }

  // Collect a tiled hlo instruction for each root. The roots which are extra
  // outputs can reference "internal" tiled hlo instructions and may appear
  // multiple times in `instructions_`.
  roots_.assign(roots.size(), nullptr);
  for (const auto& tiled_hlo_instr : instructions_) {
    auto it = roots_to_output_index.find(tiled_hlo_instr->hlo());
    if (it != roots_to_output_index.end()) {
      // We may overwrite a previous value, but in case there are multiple
      // tiled hlo instructions for the root, we prefer the last one in
      // def-before-use order.
      roots_[it->second] = tiled_hlo_instr.get();
    }
  }
  // We expect that we found at least one tiled hlo instruction for each root.
  for (auto& root : roots_) {
    CHECK_NE(root, nullptr);
  }
}

}  // namespace gpu
}  // namespace xla
