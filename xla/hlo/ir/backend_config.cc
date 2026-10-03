/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/ir/backend_config.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "google/protobuf/message.h"
#include "re2/re2.h"
#include "tsl/platform/human_readable_json.h"
#include "tsl/platform/protobuf.h"
#include "xla/util.h"

// TODO(dasenov): Remove this after 2026-07-15.
namespace {
std::string RemoveWaitOnOperationQueues(std::string&& s) {
  static constexpr LazyRE2 kReWaitOnOperationQueues = {
      R"("wait_on_operation_queues"\s*:\s*\[\s*\]\s*,)"};
  RE2::GlobalReplace(&s, *kReWaitOnOperationQueues, "");
  return std::move(s);
}
}  // namespace

namespace xla {

std::unique_ptr<tsl::protobuf::Message> CloneBackendConfigProto(
    const tsl::protobuf::Message* proto) {
  if (proto == nullptr) {
    return nullptr;
  }
  std::unique_ptr<tsl::protobuf::Message> result(proto->New());
  result->CopyFrom(*proto);
  return result;
}

absl::StatusOr<std::string> BackendConfigToRawString(
    const tsl::protobuf::Message& proto) {
  // Pass ignore_accuracy_loss = true because estimated_cycles field can be
  // INT64_MAX. If ignore_accuracy_loss = false and estimated_cycles =
  // INT64_MAX, JsonFormat will return an error status, although there is no
  // accuracy loss for int64_t.
  return tsl::ProtoToHumanReadableJson(proto, /*ignore_accuracy_loss=*/true);
}

namespace {

// True when some message type reachable from descriptor has a map field, is
// google.protobuf.Any or has an extension range. The JSON of such a message
// depends on more than its serialized bytes: map entries print in the field's
// iteration order, Any hides its type behind a URL, extensions admit both.
bool ReachesUncacheableField(
    const tsl::protobuf::Descriptor* descriptor,
    absl::flat_hash_set<const tsl::protobuf::Descriptor*>& visited) {
  if (!visited.insert(descriptor).second) {
    return false;  // Walked already, or an ancestor still being walked.
  }
  if (descriptor->well_known_type() ==
          tsl::protobuf::Descriptor::WELLKNOWNTYPE_ANY ||
      descriptor->extension_range_count() > 0) {
    return true;
  }
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const tsl::protobuf::FieldDescriptor* field = descriptor->field(i);
    if (field->is_map() ||
        (field->message_type() != nullptr &&
         ReachesUncacheableField(field->message_type(), visited))) {
      return true;
    }
  }
  return false;
}

// Whether the JSON of a message of this type is a function of its serialized
// bytes. Memoized per root type for the process: descriptors are immutable
// and outlive the compiles, which run on many threads.
bool IsCacheable(const tsl::protobuf::Descriptor* descriptor) {
  struct Memo {
    absl::Mutex mutex;
    absl::flat_hash_map<const tsl::protobuf::Descriptor*, bool> cacheable
        ABSL_GUARDED_BY(mutex);
  };
  static absl::NoDestructor<Memo> memo;
  {
    absl::ReaderMutexLock lock(&memo->mutex);
    auto it = memo->cacheable.find(descriptor);
    if (it != memo->cacheable.end()) {
      return it->second;
    }
  }
  absl::flat_hash_set<const tsl::protobuf::Descriptor*> visited;
  const bool cacheable = !ReachesUncacheableField(descriptor, visited);
  absl::WriterMutexLock lock(&memo->mutex);
  memo->cacheable.emplace(descriptor, cacheable);
  return cacheable;
}

}  // namespace

BackendConfigWrapper::BackendConfigWrapper(std::string raw_string)
    : raw_string_(RemoveWaitOnOperationQueues(std::move(raw_string))) {}

const std::string& BackendConfigWrapper::GetRawStringWithoutMutex() const {
  if (proto_ && raw_string_.empty()) {
    // Cache the raw string.
    raw_string_ = BackendConfigToRawString(*proto_).value();
  }
  static const std::string* const kEmptyString = new std::string();
  return raw_string_.empty() ? *kEmptyString : raw_string_;
}

const std::string& BackendConfigWrapper::GetRawString(
    BackendConfigRawStringCache* cache) const {
  absl::WriterMutexLock lock{mutex_};
  if (cache != nullptr && proto_ != nullptr && raw_string_.empty() &&
      IsCacheable(proto_->GetDescriptor())) {
    // Equal protos of one type serialize to the same bytes, so the first one
    // pays for the JSON printer. A failed encoding leaves the entry empty and
    // GetRawStringWithoutMutex fails on it below, as without a cache.
    std::string& shared = (*cache)[std::make_pair(
        proto_->GetDescriptor(), proto_->SerializePartialAsString())];
    if (shared.empty()) {
      absl::StatusOr<std::string> raw_string =
          BackendConfigToRawString(*proto_);
      if (raw_string.ok()) {
        shared = *std::move(raw_string);
      }
    }
    raw_string_ = shared;
  }
  return GetRawStringWithoutMutex();
}

absl::Status BackendConfigWrapper::GetProto(
    tsl::protobuf::Message* output_proto) const {
  output_proto->Clear();

  auto copy_from_cache =
      [&]() ABSL_SHARED_LOCKS_REQUIRED(mutex_) -> absl::Status {
    if (proto_->GetDescriptor() != output_proto->GetDescriptor()) {
      return Internal("Mismatched backend config descriptors.");
    }
    output_proto->CopyFrom(*proto_);
    return absl::OkStatus();
  };

  // Fast path: check with reader lock if proto is already cached.
  {
    absl::ReaderMutexLock lock{mutex_};
    if (proto_ != nullptr) {
      return copy_from_cache();
    }
    // Empty string does not parse as valid JSON, but it's a valid backend
    // config, corresponding to the empty proto.
    if (raw_string_.empty()) {
      return absl::OkStatus();
    }
  }

  absl::WriterMutexLock lock{mutex_};
  // Check again if another thread parsed and cached the proto while we were
  // waiting for the writer lock.
  if (proto_ != nullptr) {
    return copy_from_cache();
  }

  ABSL_RETURN_IF_ERROR(
      tsl::HumanReadableJsonToProto(raw_string_, output_proto));
  // Cache the proto into the empty proto_.
  proto_ = CloneBackendConfigProto(output_proto);
  return absl::OkStatus();
}

BackendConfigWrapper& BackendConfigWrapper::operator=(
    BackendConfigWrapper&& other) {
  std::unique_ptr<tsl::protobuf::Message> temp_proto;
  std::string temp_string;

  // Do not hold two mutexes at the same time to avoid deadlocks.
  {
    absl::MutexLock other_lock{other.mutex_};
    temp_proto = std::move(other.proto_);
    temp_string = std::move(other.raw_string_);
  }

  absl::MutexLock this_lock{mutex_};

  proto_ = std::move(temp_proto);
  raw_string_ = std::move(temp_string);
  return *this;
}

bool BackendConfigWrapper::operator==(const BackendConfigWrapper& other) const {
  const std::string* other_raw_string = nullptr;
  {
    // Make sure to drop the lock on this mutex before calling GetRawString()
    // to avoid deadlock.
    absl::MutexLock other_lock{other.mutex_};
    other_raw_string = &other.GetRawStringWithoutMutex();
  }

  return GetRawString() == *other_raw_string;
}

namespace {
CoreAssignmentHandler* g_core_assignment_handler = nullptr;
}  // namespace

void RegisterCoreAssignmentHandler(CoreAssignmentHandler* handler) {
  g_core_assignment_handler = handler;
}

absl::Status SetCoreAssignment(HloInstruction* inst,
                               absl::Span<const int64_t> core_ids) {
  if (g_core_assignment_handler != nullptr) {
    return g_core_assignment_handler->SetCoreAssignment(inst, core_ids);
  }
  return absl::UnimplementedError(
      "Core assignment is not implemented for this target.");
}

absl::StatusOr<std::vector<int64_t>> GetCoreAssignment(
    const HloInstruction* inst) {
  if (g_core_assignment_handler != nullptr) {
    return g_core_assignment_handler->GetCoreAssignment(inst);
  }
  return absl::UnimplementedError(
      "Core assignment is not implemented for this target.");
}

}  // namespace xla
