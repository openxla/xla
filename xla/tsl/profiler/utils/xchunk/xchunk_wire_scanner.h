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

#ifndef XLA_TSL_PROFILER_UTILS_XCHUNK_XCHUNK_WIRE_SCANNER_H_
#define XLA_TSL_PROFILER_UTILS_XCHUNK_XCHUNK_WIRE_SCANNER_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/string_reader.h"

namespace tsl::profiler::xchunk {

// Standard Protobuf wire type constants per the public encoding specification.
inline constexpr uint32_t kWireTypeBits = 3;
inline constexpr uint32_t kWireTypeMask = (1u << kWireTypeBits) - 1;
inline constexpr uint32_t kWireTypeVarint = 0;
inline constexpr uint32_t kWireTypeFixed64 = 1;
inline constexpr uint32_t kWireTypeLengthDelimited = 2;
inline constexpr uint32_t kWireTypeStartGroup = 3;
inline constexpr uint32_t kWireTypeEndGroup = 4;
inline constexpr uint32_t kWireTypeFixed32 = 5;
inline constexpr int kMaxProtoGroupNestingDepth = 64;

constexpr uint32_t MakeLengthDelimitedTag(uint32_t field_number) {
  return (field_number << kWireTypeBits) | kWireTypeLengthDelimited;
}

// Skips a single protobuf wire field on a 64 bit `riegeli::StringReader<>`.
bool SkipProtoWireField(riegeli::StringReader<>& reader, uint32_t tag,
                        int depth = 0);

// Extracts a zero copy `absl::string_view` over a length delimited protobuf
// field from a 64 bit `riegeli::StringReader<>`.
std::optional<absl::string_view> ReadLengthDelimitedSlice(
    riegeli::StringReader<>& reader);

struct ScannedXPlaneWire {
  absl::string_view name;
  std::string metadata_wire;
  std::vector<absl::string_view> line_slices;
};

struct ScannedXLineBatches {
  std::string header_wire;
  std::vector<absl::string_view> event_batch_wires;
};

// Scans a serialized `XSpace` message at the wire level, extracting the plane
// name, non line metadata wire bytes (`id`, `name`, `event_metadata`,
// `stat_metadata`), and zero copy `XLine` wire slices for each `XPlane`.
absl::StatusOr<std::vector<ScannedXPlaneWire>> ScanXSpaceWirePlanes(
    absl::string_view space_bytes, absl::string_view context_label = "");

// Scans a serialized `XLine` message at the wire level, separating non event
// header fields (`header_wire`) from contiguous wire batches of `XEvent`
// submessages (`event_batch_wires`). The first batch contains at most
// `first_batch_max_events` events, and subsequent batches contain at most
// `max_events_per_batch` events (or 32 MB of wire bytes).
absl::StatusOr<ScannedXLineBatches> SplitXLineWireBatches(
    absl::string_view line_bytes, size_t first_batch_max_events,
    size_t max_events_per_batch, absl::string_view context_label = "");

}  // namespace tsl::profiler::xchunk

#endif  // XLA_TSL_PROFILER_UTILS_XCHUNK_XCHUNK_WIRE_SCANNER_H_
