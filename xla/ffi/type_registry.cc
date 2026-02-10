/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/ffi/type_registry.h"

#include <atomic>
#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/util.h"

namespace xla::ffi {

static internal::TypeRegistrationMap& StaticTypeRegistrationMap() {
  static absl::NoDestructor<internal::TypeRegistrationMap> registry;
  return *registry;
}

TypeRegistry::TypeId TypeRegistry::GetNextTypeId() {
  static absl::NoDestructor<std::atomic<int64_t>> counter(1);
  return TypeId(counter->fetch_add(1));
}

absl::StatusOr<TypeRegistry::TypeId> TypeRegistry::AssignExternalTypeId(
    absl::string_view name, TypeInfo type_info) {
  auto& registry = StaticTypeRegistrationMap();
  absl::MutexLock lock(registry.mu);

  VLOG(3) << absl::StreamFormat("Assign external type id: name=%s registry=%p",
                                name, &registry);

  // Try to emplace with unknown type id and fill it with real type id only if
  // we successfully acquired an entry for a given name.
  auto emplaced =
      registry.map.emplace(name, TypeRegistration{kUnknownTypeId, type_info});
  if (!emplaced.second) {
    return Internal("Type name %s already registered with type id %d", name,
                    emplaced.first->second.type_id.value());
  }

  // Returns true if the registry contains an entry with a given type id.
  auto type_id_is_in_use = [&registry](TypeId type_id) {
    return absl::c_any_of(registry.map, [&](const auto& e) {
      return e.second.type_id == type_id;
    });
  };

  // Create a new type id that is not already in use.
  TypeId type_id = GetNextTypeId();
  while (type_id_is_in_use(type_id)) {
    type_id = GetNextTypeId();
  }

  VLOG(3) << absl::StreamFormat(
      "Assigned external type id: name=%s type_id=%v registry=%p", name,
      type_id, &registry);
  return emplaced.first->second.type_id = type_id;
}

absl::Status TypeRegistry::RegisterExternalTypeId(absl::string_view name,
                                                  TypeId type_id,
                                                  TypeInfo type_info) {
  auto& registry = StaticTypeRegistrationMap();
  absl::MutexLock lock(registry.mu);

  VLOG(3) << absl::StreamFormat(
      "Register external type id: name=%s type_id=%v registry=%p", name,
      type_id, &registry);

  auto emplaced =
      registry.map.emplace(name, TypeRegistration{type_id, type_info});
  if (!emplaced.second && emplaced.first->second.type_id != type_id) {
    return Internal("Type name %s already registered with type id %d vs %d)",
                    name, emplaced.first->second.type_id.value(),
                    type_id.value());
  }

  return absl::OkStatus();
}

absl::StatusOr<absl::string_view> TypeRegistry::GetTypeName(TypeId type_id) {
  auto& registry = StaticTypeRegistrationMap();
  absl::MutexLock lock(registry.mu);

  auto it = absl::c_find_if(registry.map, [&](const auto& kv) {
    return kv.second.type_id == type_id;
  });

  if (it == registry.map.end()) {
    return Internal("Type id %d is not registered with a static registry",
                    type_id.value());
  }

  return it->first;
}

absl::StatusOr<TypeRegistry::TypeId> TypeRegistry::GetTypeId(
    absl::string_view name) {
  auto& registry = StaticTypeRegistrationMap();
  absl::MutexLock lock(registry.mu);

  auto it = registry.map.find(name);
  if (it == registry.map.end()) {
    return Internal("Type name %s is not registered", name);
  }
  return it->second.type_id;
}

absl::StatusOr<TypeRegistry::TypeInfo> TypeRegistry::GetTypeInfo(
    TypeId type_id) {
  auto& registry = StaticTypeRegistrationMap();
  absl::MutexLock lock(registry.mu);

  auto it = absl::c_find_if(registry.map, [&](const auto& kv) {
    auto& [name, registration] = kv;
    return registration.type_id == type_id;
  });

  if (it == registry.map.end()) {
    return Internal("Type id %d is not registered with a static registry",
                    type_id.value());
  }

  return it->second.type_info;
}

}  // namespace xla::ffi
