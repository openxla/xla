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

#include "xla/backends/gpu/runtime/async_thunk_passes.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/async_execution.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/resource_use.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/xla.pb.h"
#include "tsl/platform/casts.h"

namespace xla::gpu {

// Returns mutable pointers to nested ThunkSequences inside a control flow
// thunk (WhileThunk, ConditionalThunk, SequentialThunk). Returns an empty
// vector for non-control-flow thunks.
static std::vector<ThunkSequence*> GetNestedThunkSequences(Thunk* thunk) {
  switch (thunk->kind()) {
    case Thunk::kWhile: {
      auto* while_thunk = tsl::down_cast<WhileThunk*>(thunk);
      return {&while_thunk->condition_executor().thunks(),
              &while_thunk->body_executor().thunks()};
    }
    case Thunk::kConditional: {
      auto* cond_thunk = tsl::down_cast<ConditionalThunk*>(thunk);
      std::vector<ThunkSequence*> nested;
      for (auto& executor : cond_thunk->branch_executors()) {
        nested.push_back(&executor.thunks());
      }
      return nested;
    }
    case Thunk::kSequential: {
      auto* seq_thunk = tsl::down_cast<SequentialThunk*>(thunk);
      return {&seq_thunk->executor().thunks()};
    }
    default:
      return {};
  }
}

//===----------------------------------------------------------------------===//
// RemoveRedundantAsyncThunkPass.
//===----------------------------------------------------------------------===//

absl::StatusOr<bool> RemoveRedundantAsyncThunkPass::Run(
    ThunkSequence* thunk_sequence, const DebugOptions& debug_options,
    const HloModule* absl_nullable hlo_module,
    const se::DeviceDescription& device_info,
    ThunkPassBufferAllocator& allocator) {
  bool changed = false;

  // Build a new thunk sequence, replacing adjacent start/done pairs with the
  // inlined nested thunks from the start thunk.
  ThunkSequence result;
  result.reserve(thunk_sequence->size());

  for (size_t i = 0; i < thunk_sequence->size(); ++i) {
    std::unique_ptr<Thunk>& thunk = (*thunk_sequence)[i];

    // Check for the pattern: AsyncStartThunk immediately followed by its
    // matching AsyncDoneThunk.
    if (thunk->kind() == Thunk::kAsyncStart && i + 1 < thunk_sequence->size()) {
      if (std::unique_ptr<Thunk>& next = (*thunk_sequence)[i + 1];
          next->kind() == Thunk::kAsyncDone) {
        auto* start = tsl::down_cast<AsyncStartThunk*>(thunk.get());
        auto* done = tsl::down_cast<AsyncDoneThunk*>(next.get());

        // If async start and done thunks share execution id, inline them into
        // the parent thunk sequence.
        if (start->async_execution_id() == done->async_execution_id()) {
          result.Append(std::move(start->thunks()));
          changed = true;
          ++i;
          continue;
        }
      }
    }

    // Otherwise move thunk into the new sequence.
    result.push_back(std::move(thunk));
  }

  // Recurse into nested control flow thunks. We must iterate over `result`
  // because thunks have been moved out of `*thunk_sequence`.
  for (auto& thunk : result) {
    for (ThunkSequence* nested : GetNestedThunkSequences(thunk.get())) {
      ASSIGN_OR_RETURN(
          bool nested_changed,
          Run(nested, debug_options, hlo_module, device_info, allocator));
      changed |= nested_changed;
    }
  }

  *thunk_sequence = std::move(result);
  return changed;
}

//===----------------------------------------------------------------------===//
// ExpandAsyncScopeThunkPass.
//===----------------------------------------------------------------------===//

// Returns true if thunks at positions `a` and `b` have buffer or resource
// conflicts that prevent reordering.
static bool HasConflicts(
    std::vector<BufferUse::ReadWriteSet>& buffer_rw_sets,
    std::vector<ResourceUse::ReadWriteSet>& resource_rw_sets, size_t a,
    size_t b) {
  return buffer_rw_sets[a].HasConflicts(buffer_rw_sets[b]) ||
         resource_rw_sets[a].HasConflicts(resource_rw_sets[b]);
}

absl::StatusOr<bool> ExpandAsyncScopeThunkPass::Run(
    ThunkSequence* thunk_sequence, const DebugOptions& debug_options,
    const HloModule* absl_nullable hlo_module,
    const se::DeviceDescription& device_info,
    ThunkPassBufferAllocator& allocator) {
  size_t n = thunk_sequence->size();

  // Start thunks found in the thunk sequence grouped by async execution id.
  absl::flat_hash_map<AsyncExecutionId, std::vector<const AsyncStartThunk*>>
      start_thunks;

  // Pre-compute buffer and resource read-write sets for all thunks using the
  // transitive Walk API to collect uses from nested thunks. For
  // AsyncStartThunks we also record them in start_thunks so that we can look up
  // the matching start when processing AsyncDoneThunks. AsyncDoneThunk doesn't
  // report buffer or resource uses of its own, so we copy the matching start
  // thunk's uses.
  std::vector<BufferUse::ReadWriteSet> buffer_rw_sets(n);
  std::vector<ResourceUse::ReadWriteSet> resource_rw_sets(n);
  for (size_t i = 0; i < n; ++i) {
    Thunk* current = (*thunk_sequence)[i].get();

    current->Walk([&](auto* thunk) {
      buffer_rw_sets[i].AddAll(thunk->buffer_uses());
      resource_rw_sets[i].AddAll(thunk->resource_uses());
    });

    if (current->kind() == Thunk::kAsyncStart) {
      auto* start = tsl::down_cast<AsyncStartThunk*>(current);
      start_thunks[start->async_execution_id()].push_back(start);
    }

    if (current->kind() == Thunk::kAsyncDone) {
      auto* done = tsl::down_cast<AsyncDoneThunk*>(current);
      // Walk all matching start thunks because pipelined async chains may have
      // multiple start thunks sharing the same execution id. They should all
      // have the same buffer and resource uses, but we traverse them all to be
      // safe.
      for (auto* start : start_thunks.at(done->async_execution_id())) {
        start->Walk([&](auto* thunk) {
          buffer_rw_sets[i].AddAll(thunk->buffer_uses());
          resource_rw_sets[i].AddAll(thunk->resource_uses());
        });
      }
    }
  }

  bool changed = false;

  // Move AsyncStartThunks as far up the sequence as possible.
  for (int64_t i = 1; i < n; ++i) {
    if ((*thunk_sequence)[i]->kind() != Thunk::kAsyncStart) {
      continue;
    }

    int64_t j = i;
    while (j > 0 && !HasConflicts(buffer_rw_sets, resource_rw_sets, j - 1, j)) {
      std::swap((*thunk_sequence)[j], (*thunk_sequence)[j - 1]);
      std::swap(buffer_rw_sets[j], buffer_rw_sets[j - 1]);
      std::swap(resource_rw_sets[j], resource_rw_sets[j - 1]);
      --j;
      changed = true;
    }
  }

  // Move AsyncDoneThunks as far down the sequence as possible.
  for (int64_t i = n - 2; i >= 0; --i) {
    if ((*thunk_sequence)[i]->kind() != Thunk::kAsyncDone) {
      continue;
    }

    int64_t j = i;
    while (j < n - 1 &&
           !HasConflicts(buffer_rw_sets, resource_rw_sets, j, j + 1)) {
      std::swap((*thunk_sequence)[j], (*thunk_sequence)[j + 1]);
      std::swap(buffer_rw_sets[j], buffer_rw_sets[j + 1]);
      std::swap(resource_rw_sets[j], resource_rw_sets[j + 1]);
      ++j;
      changed = true;
    }
  }

  // Recurse into nested control flow thunks.
  for (auto& thunk : *thunk_sequence) {
    for (ThunkSequence* nested : GetNestedThunkSequences(thunk.get())) {
      ASSIGN_OR_RETURN(
          bool nested_changed,
          Run(nested, debug_options, hlo_module, device_info, allocator));
      changed |= nested_changed;
    }
  }

  return changed;
}

}  // namespace xla::gpu
