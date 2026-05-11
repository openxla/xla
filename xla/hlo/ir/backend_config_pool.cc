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

#include "xla/hlo/ir/backend_config_pool.h"

#include <cstddef>
#include <memory>
#include <string>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/backend_config.h"

namespace xla {

BackendConfigPool* BackendConfigPool::Get() {
  static absl::NoDestructor<BackendConfigPool> pool;
  return &*pool;
}

std::shared_ptr<const BackendConfigWrapper> BackendConfigPool::Intern(
    absl::string_view json) {
  absl::MutexLock lock(mutex_);

  // Periodically garbage collect (e.g., every 1000 calls).
  static int call_count = 0;
  if (++call_count % 1000 == 0) {
    size_t num_erased = absl::erase_if(
        registry_, [](auto& entry) { return entry.second.expired(); });
    VLOG(3) << "Garbage collected " << num_erased << " expired backend configs";
  }

  auto it = registry_.find(json);
  if (it != registry_.end()) {
    auto shared = it->second.lock();
    if (shared) {
      return shared;
    }
  }

  // Create as NON-CONST object on the heap to make const_cast safe for COW.
  auto non_const_shared =
      std::make_shared<BackendConfigWrapper>(std::string(json));

  // Implicitly converts to shared_ptr<const BackendConfigWrapper>
  std::shared_ptr<const BackendConfigWrapper> shared = non_const_shared;

  registry_[json] = non_const_shared;
  return shared;
}

size_t BackendConfigPool::GarbageCollect() {
  absl::MutexLock lock(mutex_);
  size_t num_erased = absl::erase_if(
      registry_, [](auto& entry) { return entry.second.expired(); });
  VLOG(3) << "Garbage collected " << num_erased << " expired backend configs";
  return num_erased;
}

}  // namespace xla
