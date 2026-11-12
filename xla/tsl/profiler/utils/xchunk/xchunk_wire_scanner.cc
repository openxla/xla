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

#include "xla/tsl/profiler/utils/xchunk/xchunk_wire_scanner.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/varint/varint_reading.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl::profiler::xchunk {
namespace {

using ::tensorflow::profiler::XLine;
using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XSpace;

constexpr uint32_t kXSpacePlanesTag =
    MakeLengthDelimitedTag(XSpace::kPlanesFieldNumber);
constexpr uint32_t kXPlaneNameTag =
    MakeLengthDelimitedTag(XPlane::kNameFieldNumber);
constexpr uint32_t kXPlaneLinesTag =
    MakeLengthDelimitedTag(XPlane::kLinesFieldNumber);
constexpr uint32_t kXLineEventsTag =
    MakeLengthDelimitedTag(XLine::kEventsFieldNumber);

constexpr size_t kMaxEventBatchWireBytes = 32 << 20;  // 32 MiB

}  // namespace

bool SkipProtoWireField(riegeli::StringReader<>& reader, uint32_t tag,
                        int depth) {
  if ((tag >> kWireTypeBits) == 0 || depth > kMaxProtoGroupNestingDepth) {
    return false;
  }
  switch (tag & kWireTypeMask) {
    case kWireTypeVarint: {
      uint64_t varint_value = 0;
      return riegeli::ReadVarint64(reader, varint_value);
    }
    case kWireTypeFixed64:
      return reader.available() >= 8 && reader.Skip(8);
    case kWireTypeLengthDelimited: {
      uint64_t len = 0;
      return riegeli::ReadVarint64(reader, len) && len <= reader.available() &&
             reader.Skip(len);
    }
    case kWireTypeStartGroup: {
      const uint32_t expected_end_tag =
          (tag & ~kWireTypeMask) | kWireTypeEndGroup;
      while (reader.available() > 0) {
        uint32_t inner_tag = 0;
        if (!riegeli::ReadVarint32(reader, inner_tag)) {
          return false;
        }
        if (inner_tag == expected_end_tag) {
          return true;
        }
        if (!SkipProtoWireField(reader, inner_tag, depth + 1)) {
          return false;
        }
      }
      return false;
    }
    case kWireTypeFixed32:
      return reader.available() >= 4 && reader.Skip(4);
    default:
      return false;
  }
}

std::optional<absl::string_view> ReadLengthDelimitedSlice(
    riegeli::StringReader<>& reader) {
  uint64_t len = 0;
  if (!riegeli::ReadVarint64(reader, len) || len > reader.available()) {
    return std::nullopt;
  }
  const absl::string_view slice(reader.cursor(), static_cast<size_t>(len));
  if (!reader.Skip(len)) {
    return std::nullopt;
  }
  return slice;
}

absl::StatusOr<std::vector<ScannedXPlaneWire>> ScanXSpaceWirePlanes(
    absl::string_view space_bytes, absl::string_view context_label) {
  std::vector<ScannedXPlaneWire> planes;
  riegeli::StringReader<> space_input(space_bytes);
  while (space_input.available() > 0) {
    uint32_t stag = 0;
    if (!riegeli::ReadVarint32(space_input, stag) ||
        (stag >> kWireTypeBits) == 0) {
      return absl::InternalError(
          absl::StrCat("Invalid XSpace tag in ", context_label));
    }
    if (stag != kXSpacePlanesTag) {
      if (!SkipProtoWireField(space_input, stag)) {
        return absl::InternalError(
            absl::StrCat("Invalid XSpace field in ", context_label));
      }
      continue;
    }
    std::optional<absl::string_view> plane_bytes =
        ReadLengthDelimitedSlice(space_input);
    if (!plane_bytes.has_value()) {
      return absl::InternalError(
          absl::StrCat("Truncated XPlane in ", context_label));
    }

    ScannedXPlaneWire plane_wire;
    riegeli::StringReader<> plane_input(*plane_bytes);
    while (plane_input.available() > 0) {
      const size_t field_start_pos = static_cast<size_t>(plane_input.pos());
      uint32_t ptag = 0;
      if (!riegeli::ReadVarint32(plane_input, ptag) ||
          (ptag >> kWireTypeBits) == 0) {
        return absl::InternalError(
            absl::StrCat("Invalid XPlane tag in ", context_label));
      }
      if (ptag == kXPlaneLinesTag) {
        std::optional<absl::string_view> line_slice =
            ReadLengthDelimitedSlice(plane_input);
        if (!line_slice.has_value()) {
          return absl::InternalError(
              absl::StrCat("Truncated XLine in ", context_label));
        }
        plane_wire.line_slices.push_back(*line_slice);
      } else if (ptag == kXPlaneNameTag) {
        std::optional<absl::string_view> name_slice =
            ReadLengthDelimitedSlice(plane_input);
        if (!name_slice.has_value()) {
          return absl::InternalError(
              absl::StrCat("Truncated XPlane.name in ", context_label));
        }
        plane_wire.name = *name_slice;
        const size_t field_end_pos = static_cast<size_t>(plane_input.pos());
        plane_wire.metadata_wire.append(plane_bytes->data() + field_start_pos,
                                        field_end_pos - field_start_pos);
      } else {
        if (!SkipProtoWireField(plane_input, ptag)) {
          return absl::InternalError(
              absl::StrCat("Invalid XPlane field in ", context_label));
        }
        const size_t field_end_pos = static_cast<size_t>(plane_input.pos());
        plane_wire.metadata_wire.append(plane_bytes->data() + field_start_pos,
                                        field_end_pos - field_start_pos);
      }
    }
    planes.push_back(std::move(plane_wire));
  }
  return planes;
}

absl::StatusOr<ScannedXLineBatches> SplitXLineWireBatches(
    absl::string_view line_bytes, size_t first_batch_max_events,
    size_t max_events_per_batch, absl::string_view context_label) {
  ScannedXLineBatches out;
  const size_t effective_first_max =
      first_batch_max_events > 0 ? first_batch_max_events : 65536;
  const size_t effective_batch_max =
      max_events_per_batch > 0 ? max_events_per_batch : 524288;

  riegeli::StringReader<> line_input(line_bytes);
  std::optional<size_t> batch_start_pos;
  size_t batch_end_pos = 0;
  size_t events_in_batch = 0;
  while (line_input.available() > 0) {
    const size_t field_start_pos = static_cast<size_t>(line_input.pos());
    uint32_t ltag = 0;
    if (!riegeli::ReadVarint32(line_input, ltag) ||
        (ltag >> kWireTypeBits) == 0) {
      return absl::InternalError(
          absl::StrCat("Invalid XLine tag in ", context_label));
    }
    if (ltag == kXLineEventsTag) {
      if (!ReadLengthDelimitedSlice(line_input).has_value()) {
        return absl::InternalError(
            absl::StrCat("Truncated XEvent in ", context_label));
      }
      if (!batch_start_pos.has_value()) {
        batch_start_pos = field_start_pos;
      }
      batch_end_pos = static_cast<size_t>(line_input.pos());
      ++events_in_batch;
      const size_t batch_limit = out.event_batch_wires.empty()
                                     ? effective_first_max
                                     : effective_batch_max;
      if (events_in_batch >= batch_limit ||
          (batch_end_pos - *batch_start_pos) >= kMaxEventBatchWireBytes) {
        out.event_batch_wires.emplace_back(line_bytes.data() + *batch_start_pos,
                                           batch_end_pos - *batch_start_pos);
        batch_start_pos.reset();
        batch_end_pos = 0;
        events_in_batch = 0;
      }
    } else {
      if (batch_start_pos.has_value()) {
        out.event_batch_wires.emplace_back(line_bytes.data() + *batch_start_pos,
                                           batch_end_pos - *batch_start_pos);
        batch_start_pos.reset();
        batch_end_pos = 0;
        events_in_batch = 0;
      }
      if (!SkipProtoWireField(line_input, ltag)) {
        return absl::InternalError(
            absl::StrCat("Invalid XLine field in ", context_label));
      }
      const size_t field_end_pos = static_cast<size_t>(line_input.pos());
      out.header_wire.append(
          line_bytes.substr(field_start_pos, field_end_pos - field_start_pos));
    }
  }
  if (batch_start_pos.has_value()) {
    out.event_batch_wires.emplace_back(line_bytes.data() + *batch_start_pos,
                                       batch_end_pos - *batch_start_pos);
  }
  return out;
}

}  // namespace tsl::profiler::xchunk
