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

#include "xla/backends/gpu/libraries/cub/cub_scratch_size_deviceless_lookup.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_utils.h"
#include "xla/stream_executor/semantic_version.h"

namespace xla::gpu {

absl::StatusOr<CubScratchSizeDevicelessLookup>
CubScratchSizeDevicelessLookup::Create(CubScratchSizeLookupTable proto) {
  for (const auto& entry : proto.entries()) {
    for (int i = 1; i < entry.scratch_size_recordings_size(); ++i) {
      if (entry.scratch_size_recordings(i).num_items() <=
          entry.scratch_size_recordings(i - 1).num_items()) {
        return absl::InvalidArgumentError(
            "scratch_size_recordings must be sorted by num_items");
      }
    }
  }
  return CubScratchSizeDevicelessLookup(std::move(proto));
}

CubScratchSizeDevicelessLookup::CubScratchSizeDevicelessLookup(
    CubScratchSizeLookupTable proto)
    : proto_(std::move(proto)) {}

const CubScratchSizeEntry* CubScratchSizeDevicelessLookup::FindEntry(
    stream_executor::SemanticVersion cub_version, absl::string_view device_name,
    int32_t key_type_size, std::optional<int32_t> value_type_size,
    bool is_segmented) const {
  for (const CubScratchSizeEntry& entry : proto_.entries()) {
    bool version_matched =
        std::find(entry.cub_version().begin(), entry.cub_version().end(),
                  cub_version.ToString()) != entry.cub_version().end();

    if (version_matched && entry.device_name() == device_name &&
        entry.key_type_size() == key_type_size &&
        entry.value_type_size() == value_type_size.value_or(0) &&
        entry.is_segmented() == is_segmented) {
      return &entry;
    }
  }
  return nullptr;
}

std::optional<int64_t> CubScratchSizeDevicelessLookup::Lookup(
    stream_executor::SemanticVersion cub_version, absl::string_view device_name,
    int32_t key_type_size, std::optional<int32_t> value_type_size,
    int64_t num_items, int64_t batch_size) const {
  const CubScratchSizeEntry* entry =
      FindEntry(cub_version, device_name, key_type_size, value_type_size,
                /*is_segmented=*/batch_size > 1);
  if (entry == nullptr) {
    return std::nullopt;
  }

  auto it = std::lower_bound(
      entry->scratch_size_recordings().begin(),
      entry->scratch_size_recordings().end(), num_items,
      [](const CubScratchSizeEntry::ScratchSizeRecord& record,
         int64_t num_items) { return record.num_items() < num_items; });

  if (it == entry->scratch_size_recordings().end()) {
    return std::nullopt;
  }

  return AddSegmentedSortOffsetsToScratchSize(it->scratch_space_bytes(),
                                              batch_size);
}

bool CubScratchSizeDevicelessLookup::CanLookup(
    stream_executor::SemanticVersion cub_version, absl::string_view device_name,
    int32_t key_type_size, std::optional<int32_t> value_type_size,
    int64_t num_items, int64_t batch_size) const {
  const CubScratchSizeEntry* entry = FindEntry(
      cub_version, device_name, key_type_size, value_type_size, batch_size > 1);
  if (entry == nullptr || entry->scratch_size_recordings().empty()) {
    return false;
  }

  return entry->scratch_size_recordings()
             .Get(entry->scratch_size_recordings().size() - 1)
             .num_items() >= num_items;
}

}  // namespace xla::gpu
