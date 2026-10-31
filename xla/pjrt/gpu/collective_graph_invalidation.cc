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

#include "xla/pjrt/gpu/collective_graph_invalidation.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/process_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/status_macros.h"
#include "tsl/platform/fingerprint.h"

namespace xla::gpu {
namespace {

struct InvalidationState {
  struct Sequence {
    uint64_t next = 0;
    bool can_delete = true;
  };

  InvalidationState(
      ProcessId process_id,
      absl::flat_hash_map<GlobalDeviceId, ProcessId> device_to_process,
      std::shared_ptr<KeyValueStoreInterface> kv_store)
      : process_id(process_id),
        device_to_process(std::move(device_to_process)),
        kv_store(std::move(kv_store)) {}

  ProcessId process_id;
  absl::flat_hash_map<GlobalDeviceId, ProcessId> device_to_process;
  std::shared_ptr<KeyValueStoreInterface> kv_store;
  absl::Mutex mutex;
  absl::flat_hash_map<std::string, Sequence> sequences ABSL_GUARDED_BY(mutex);

  absl::StatusOr<bool> Synchronize(
      absl::string_view signature,
      absl::Span<const GlobalDeviceId> participants, bool local_dirty) {
    TF_RET_CHECK(kv_store != nullptr);
    absl::btree_set<ProcessId> processes;
    for (GlobalDeviceId device : participants) {
      auto it = device_to_process.find(device);
      TF_RET_CHECK(it != device_to_process.end())
          << "Unknown collective participant " << device;
      processes.insert(it->second);
    }
    TF_RET_CHECK(processes.contains(process_id))
        << "Current process does not participate in the collective";
    if (processes.size() == 1) {
      return local_dirty;
    }

    std::vector<GlobalDeviceId> devices(participants.begin(),
                                        participants.end());
    absl::c_sort(devices);
    std::string identity = absl::StrCat(signature.size(), ":", signature, ":",
                                        absl::StrJoin(devices, ","));
    uint64_t sequence;
    {
      absl::MutexLock lock(mutex);
      // RunId can differ across processes after unrelated local executions.
      // Clique locks serialize executions sharing this collective signature.
      Sequence& state = sequences[identity];
      if (!state.can_delete) {
        return true;
      }
      sequence = state.next++;
    }
    tsl::Fprint128 fingerprint = tsl::Fingerprint128(identity);
    std::string prefix =
        absl::StrFormat("gpu/collective_graph_invalidation/%016x%016x/%d/",
                        fingerprint.high64, fingerprint.low64, sequence);
    auto get = [&](absl::string_view key) {
      return kv_store->Get(key, absl::Minutes(10));
    };
    bool is_coordinator = process_id == *processes.begin();
    if (sequence == 0) {
      std::string capability_key = absl::StrCat(prefix, "can_delete");
      if (is_coordinator) {
        absl::Status status = kv_store->Delete(absl::StrCat(prefix, "probe"));
        if (!status.ok() && !absl::IsUnimplemented(status)) {
          return status;
        }
        absl::Status published =
            kv_store->Set(capability_key, status.ok() ? "1" : "0");
        if (!published.ok()) {
          if (!absl::IsAlreadyExists(published)) {
            return published;
          }
          // A previous client using this store may have left an undeletable
          // fallback decision. Every reader must keep honoring it.
          ABSL_ASSIGN_OR_RETURN(std::string previous, get(capability_key));
          if (previous != "0") {
            return published;
          }
        }
      }
      ABSL_ASSIGN_OR_RETURN(std::string can_delete, get(capability_key));
      TF_RET_CHECK(can_delete == "0" || can_delete == "1");
      if (can_delete == "0") {
        // Older PJRT C API hosts cannot delete or overwrite keys. Agree once
        // to rebuild every execution instead of accumulating per-run keys.
        absl::MutexLock lock(mutex);
        sequences[identity].can_delete = false;
        return true;
      }
    }

    auto key = [&](absl::string_view kind, ProcessId process) {
      return absl::StrCat(prefix, kind, "/", process.value());
    };

    ABSL_RETURN_IF_ERROR(
        kv_store->Set(key("dirty", process_id), local_dirty ? "1" : "0"));
    if (is_coordinator) {
      bool dirty = false;
      for (ProcessId process : processes) {
        ABSL_ASSIGN_OR_RETURN(std::string value, get(key("dirty", process)));
        TF_RET_CHECK(value == "0" || value == "1");
        dirty |= value == "1";
      }
      ABSL_RETURN_IF_ERROR(
          kv_store->Set(absl::StrCat(prefix, "result"), dirty ? "1" : "0"));
    }

    ABSL_ASSIGN_OR_RETURN(std::string result,
                          get(absl::StrCat(prefix, "result")));
    TF_RET_CHECK(result == "0" || result == "1");
    ABSL_RETURN_IF_ERROR(kv_store->Set(key("ack", process_id), ""));
    if (is_coordinator) {
      // Every reader must consume the result before its keys can be deleted.
      for (ProcessId process : processes) {
        ABSL_RETURN_IF_ERROR(get(key("ack", process)).status());
      }
      ABSL_RETURN_IF_ERROR(kv_store->Delete(prefix));
    }
    return result == "1";
  }
};

}  // namespace

CollectiveGraphInvalidationCallback CreateCollectiveGraphInvalidationCallback(
    ProcessId process_id,
    absl::flat_hash_map<GlobalDeviceId, ProcessId> device_to_process,
    std::shared_ptr<KeyValueStoreInterface> kv_store) {
  auto state = std::make_shared<InvalidationState>(
      process_id, std::move(device_to_process), std::move(kv_store));
  return [state = std::move(state)](
             absl::string_view signature,
             absl::Span<const GlobalDeviceId> participants,
             bool local_dirty) -> absl::StatusOr<bool> {
    absl::StatusOr<bool> result =
        state->Synchronize(signature, participants, local_dirty);
    if (!result.ok()) {
      return absl::Status(result.status().code(),
                          absl::StrCat("Collective graph invalidation failed: ",
                                       result.status().message()));
    }
    return result;
  };
}

}  // namespace xla::gpu
