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

#ifndef XLA_HLO_IR_BACKEND_CONFIG_POOL_H_
#define XLA_HLO_IR_BACKEND_CONFIG_POOL_H_

#include <cstddef>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/backend_config.h"

namespace xla {

// A process-wide pool for deduplicating BackendConfigWrapper objects.
// It allows multiple instructions to share the same backend config in memory.
class BackendConfigPool {
 public:
  // Returns the singleton instance of the pool.
  static BackendConfigPool* Get();

  // Returns a shared pointer to a BackendConfigWrapper corresponding to the
  // provided JSON string. If the string is already in the pool and valid,
  // returns the existing instance. Otherwise, creates a new one.
  std::shared_ptr<const BackendConfigWrapper> Intern(absl::string_view json);

  // Runs garbage collection to remove expired weak pointers from the pool.
  // Returns the number of entries removed.
  size_t GarbageCollect();

 private:
  absl::Mutex mutex_;
  absl::flat_hash_map<std::string, std::weak_ptr<const BackendConfigWrapper>>
      registry_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla

#endif  // XLA_HLO_IR_BACKEND_CONFIG_POOL_H_
