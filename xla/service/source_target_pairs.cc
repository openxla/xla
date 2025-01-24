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

#include "xla/service/source_target_pairs.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/service/graphcycles/graphcycles.h"

namespace xla {

std::string SourceTargetPairs::ToString() const {
  auto formatter = [](std::string* out, const SourceTargetPair& pair) {
    absl::StrAppend(out, "{", pair.source, ",", pair.target, "}");
  };
  const std::string pairs_str = absl::StrJoin(pairs_, ",", formatter);
  return absl::StrCat("{", pairs_str, "}");
}

namespace {
int32_t GetNodeId(int64_t replica, GraphCycles& graph,
                  absl::flat_hash_map<int64_t, int32_t>& map) {
  if (!map.contains(replica)) {
    map.emplace(replica, graph.NewNode());
  }
  return map.at(replica);
}
}  // namespace

bool SourceTargetPairs::HasCycles() {
  GraphCycles graph;
  absl::flat_hash_map<int64_t, int32_t> replica_to_node_id;
  for (const SourceTargetPair& pair : pairs_) {
    const int source = GetNodeId(pair.source, graph, replica_to_node_id);
    const int target = GetNodeId(pair.target, graph, replica_to_node_id);
    if (!graph.InsertEdge(source, target)) {
      return true;
    }
  }
  return false;
}

// TODO: b/388623407 - remove assumptions that pairs are ordered and 0 based.
bool SourceTargetPairs::IsForwardCycle(const SourceTargetPairs& backedge,
                                       const SourceTargetPairs& others) {
  if (backedge.size() != 1) {
    return false;
  }
  const int64_t num_pairs = others.size() + 1;
  if (backedge[0].source != num_pairs - 1 || backedge[0].target != 0) {
    return false;
  }
  for (int64_t i = 0; i < num_pairs - 1; ++i) {
    const SourceTargetPair& pair = others[i];
    if (pair.source != i || pair.target != i + 1) {
      return false;
    }
  }
  return true;
}

bool SourceTargetPairs::IsBackwardCycle(const SourceTargetPairs& backedge,
                                        const SourceTargetPairs& others) {
  if (backedge.size() != 1) {
    return false;
  }
  const int64_t num_pairs = others.size() + 1;
  if (backedge[0].source != 0 || backedge[0].target != num_pairs - 1) {
    return false;
  }
  for (int64_t i = 0; i < num_pairs - 1; ++i) {
    const SourceTargetPair& pair = others[i];
    if (pair.source != i + 1 || pair.target != i) {
      return false;
    }
  }
  return true;
}

}  // namespace xla
