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

#ifndef XLA_TSL_PROFILER_UTILS_XCHUNK_XCHUNK_H_
#define XLA_TSL_PROFILER_UTILS_XCHUNK_XCHUNK_H_

// XChunk: a columnar, chunk streamable in process substitute for XPlane.
//
// XPlane is a hierarchical proto (plane -> lines -> events -> stats) which
// forces trace and analysis backends to materialize the whole plane before any
// event can be processed or rendered. XChunk carries the same information as a
// stream of native C++ messages:
//
//   XChunkStreamHeader                    (once)
//   XChunkPlane                           (once per plane, before its events)
//   XChunkEvents { plane_index, line_id } (one or more per line, in time order)
//   ...
//   XChunkEnd                             (once)
//
// All columns are event indexed so the consumer decodes them in a single pass,
// and sparse columns (flows, group ids, counters, name overrides) cost nothing
// for events that do not carry them. Chunks of different planes can be produced
// in parallel and interleaved.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/strings/string_view.h"

namespace tsl::profiler::xchunk {

// Bit flags encoded in the upper bits of uint64_t duration_ps for streaming
// multi chunk spans (e.g., long running HLO modules or while/call loops that
// remain open across chunk boundaries during progressive trace streaming):
// - Bits 0..60 (`kDurationValueMask`): actual span duration in picoseconds
//   (supports spans up to ~26.7 days).
// - Bit 61 (`kDurationPhaseBeginBit`): marks an open span segment flushed at an
//   intermediate chunk boundary before its closing trace entry has arrived.
// - Bit 62 (`kDurationPhaseEndBit`): marks the closing segment of a previously
//   split span (also used by trace event sorting to order closing segments
//   deterministically).
inline constexpr uint64_t kDurationPhaseBeginBit = 1ULL << 61;
inline constexpr uint64_t kDurationPhaseEndBit = 1ULL << 62;
inline constexpr uint64_t kDurationPhaseMask =
    kDurationPhaseBeginBit | kDurationPhaseEndBit;
inline constexpr uint64_t kDurationValueMask = ~kDurationPhaseMask;

// Helper adapter for vector columns that provides .Reserve() in addition
// to .reserve() for source compatibility.
template <typename T>
class ColumnVector : public std::vector<T> {
 public:
  using std::vector<T>::vector;
  void Reserve(size_t n) { this->reserve(n); }
};

// Emitted once at the start of a stream.
struct XChunkStreamHeader {
  enum DeviceFamily {
    UNKNOWN = 0,
    TPU = 1,
    GPU = 2,
  };

  std::string hostname_str;
  uint32_t host_id_val = 0;
  DeviceFamily device_family_val = UNKNOWN;

  absl::string_view hostname() const { return hostname_str; }
  void set_hostname(std::string h) { hostname_str = std::move(h); }
  void set_hostname(absl::string_view h) { hostname_str = std::string(h); }
  void set_hostname(const char* h) { hostname_str = h; }

  uint32_t host_id() const { return host_id_val; }
  void set_host_id(uint32_t id) { host_id_val = id; }

  DeviceFamily device_family() const { return device_family_val; }
  void set_device_family(DeviceFamily df) { device_family_val = df; }
};

// Describes one XLine. Mirrors the XLine header (without the events).
struct XChunkLine {
  int64_t line_id_val = 0;
  // Zero means "same as line_id" (XLineVisitor::DisplayId semantics).
  int64_t display_id_val = 0;
  std::string name_str;
  // Empty means "same as name" (XLineVisitor::DisplayName semantics).
  std::string display_name_str;
  int64_t timestamp_ns_val = 0;
  int64_t duration_ps_val = 0;
  // Optional hint of the number of events in this line when known upfront
  // (e.g., when converting a materialized XPlane via ChunkXPlane). Purely
  // streaming producers that emit line slices incrementally may leave this as 0
  // (or set it to the initial slice count); consumers use
  // XChunkEvents::last_chunk() rather than num_events() to detect end of line.
  int64_t num_events_val = 0;

  int64_t line_id() const { return line_id_val; }
  void set_line_id(int64_t v) { line_id_val = v; }

  int64_t display_id() const { return display_id_val; }
  void set_display_id(int64_t v) { display_id_val = v; }

  absl::string_view name() const { return name_str; }
  void set_name(std::string v) { name_str = std::move(v); }
  void set_name(absl::string_view v) { name_str = std::string(v); }
  void set_name(const char* v) { name_str = v; }

  absl::string_view display_name() const { return display_name_str; }
  void set_display_name(std::string v) { display_name_str = std::move(v); }
  void set_display_name(absl::string_view v) {
    display_name_str = std::string(v);
  }
  void set_display_name(const char* v) { display_name_str = v; }

  int64_t timestamp_ns() const { return timestamp_ns_val; }
  void set_timestamp_ns(int64_t v) { timestamp_ns_val = v; }

  int64_t duration_ps() const { return duration_ps_val; }
  void set_duration_ps(int64_t v) { duration_ps_val = v; }

  int64_t num_events() const { return num_events_val; }
  void set_num_events(int64_t v) { num_events_val = v; }
};

// Describes one XPlane: identity, the trace viewer process it maps to, its
// lines and the dense event/stat metadata tables referenced by the chunks.
struct XChunkPlane {
  uint32_t plane_index_val = 0;
  int64_t plane_id_val = 0;
  std::string plane_name_str;
  uint32_t process_id_val = 0;
  std::string process_name_str;

  // Parallel arrays indexed by dense event metadata index
  // [0, event_names_size()). `event_metadata_ids_vec` and `event_names_vec`
  // MUST have the same length. `event_display_names_vec` is either the same
  // length (with empty strings for events without a distinct display name) or
  // shorter/empty when trailing events have no display name. Prefer
  // `AddEventMetadata(...)` to populate all three vectors in lockstep.
  std::vector<int64_t> event_metadata_ids_vec;
  std::vector<std::string> event_names_vec;
  std::vector<std::string> event_display_names_vec;

  std::vector<int64_t> stat_metadata_ids_vec;
  std::vector<std::string> stat_names_vec;

  std::vector<XChunkLine> lines_vec;

  uint32_t plane_index() const { return plane_index_val; }
  void set_plane_index(uint32_t v) { plane_index_val = v; }

  int64_t plane_id() const { return plane_id_val; }
  void set_plane_id(int64_t v) { plane_id_val = v; }

  absl::string_view plane_name() const { return plane_name_str; }
  void set_plane_name(std::string v) { plane_name_str = std::move(v); }
  void set_plane_name(absl::string_view v) { plane_name_str = std::string(v); }
  void set_plane_name(const char* v) { plane_name_str = v; }

  uint32_t process_id() const { return process_id_val; }
  void set_process_id(uint32_t v) { process_id_val = v; }

  absl::string_view process_name() const { return process_name_str; }
  void set_process_name(std::string v) { process_name_str = std::move(v); }
  void set_process_name(absl::string_view v) {
    process_name_str = std::string(v);
  }
  void set_process_name(const char* v) { process_name_str = v; }

  // Event metadata table.
  void AddEventMetadata(int64_t metadata_id, absl::string_view name,
                        absl::string_view display_name = "") {
    event_metadata_ids_vec.push_back(metadata_id);
    event_names_vec.emplace_back(name);
    event_display_names_vec.emplace_back(display_name);
  }

  const std::vector<int64_t>& event_metadata_ids() const {
    return event_metadata_ids_vec;
  }
  int64_t event_metadata_ids(int index) const {
    return event_metadata_ids_vec[index];
  }
  void add_event_metadata_ids(int64_t v) {
    event_metadata_ids_vec.push_back(v);
  }

  const std::vector<std::string>& event_names() const {
    return event_names_vec;
  }
  absl::string_view event_names(int index) const {
    return event_names_vec[index];
  }
  size_t event_names_size() const { return event_names_vec.size(); }
  void add_event_names(std::string v) {
    event_names_vec.push_back(std::move(v));
  }
  void add_event_names(absl::string_view v) { event_names_vec.emplace_back(v); }
  void add_event_names(const char* v) { event_names_vec.emplace_back(v); }

  const std::vector<std::string>& event_display_names() const {
    return event_display_names_vec;
  }
  absl::string_view event_display_names(int index) const {
    return event_display_names_vec[index];
  }
  size_t event_display_names_size() const {
    return event_display_names_vec.size();
  }
  void add_event_display_names(std::string v) {
    event_display_names_vec.push_back(std::move(v));
  }
  void add_event_display_names(absl::string_view v) {
    event_display_names_vec.emplace_back(v);
  }
  void add_event_display_names(const char* v) {
    event_display_names_vec.emplace_back(v);
  }

  // Stat metadata table.
  void AddStatMetadata(int64_t metadata_id, absl::string_view name) {
    stat_metadata_ids_vec.push_back(metadata_id);
    stat_names_vec.emplace_back(name);
  }

  const std::vector<int64_t>& stat_metadata_ids() const {
    return stat_metadata_ids_vec;
  }
  int64_t stat_metadata_ids(int index) const {
    return stat_metadata_ids_vec[index];
  }
  void add_stat_metadata_ids(int64_t v) { stat_metadata_ids_vec.push_back(v); }

  const std::vector<std::string>& stat_names() const { return stat_names_vec; }
  absl::string_view stat_names(int index) const {
    return stat_names_vec[index];
  }
  size_t stat_names_size() const { return stat_names_vec.size(); }
  void add_stat_names(std::string v) { stat_names_vec.push_back(std::move(v)); }
  void add_stat_names(absl::string_view v) { stat_names_vec.emplace_back(v); }
  void add_stat_names(const char* v) { stat_names_vec.emplace_back(v); }

  // Lines.
  const std::vector<XChunkLine>& lines() const { return lines_vec; }
  std::vector<XChunkLine>& lines() { return lines_vec; }
  const XChunkLine& lines(int index) const { return lines_vec[index]; }
  XChunkLine& lines(int index) { return lines_vec[index]; }
  size_t lines_size() const { return lines_vec.size(); }
  XChunkLine* add_lines() {
    lines_vec.emplace_back();
    return &lines_vec.back();
  }
};

// Sparse columns: values[i] belongs to event at event_index[i] within the
// chunk. `event_indices` and `value_items` always have the same length and are
// populated together via `Add(event_index, value)`.
template <typename T>
struct SparseColumn {
  std::vector<uint32_t> event_indices;
  std::vector<T> value_items;
  bool needs_sort = false;

  const std::vector<uint32_t>& event_index() const { return event_indices; }
  uint32_t event_index(int i) const { return event_indices[i]; }
  int event_index_size() const {
    return static_cast<int>(event_indices.size());
  }

  const std::vector<T>& values() const { return value_items; }
  const T& values(int i) const { return value_items[i]; }
  int values_size() const { return static_cast<int>(value_items.size()); }

  void Add(uint32_t idx, T val) {
    if (ABSL_PREDICT_FALSE(!event_indices.empty() &&
                           idx <= event_indices.back())) {
      if (idx == event_indices.back()) {
        value_items.back() = std::move(val);
        return;
      }
      needs_sort = true;
    }
    event_indices.push_back(idx);
    value_items.push_back(std::move(val));
  }

  template <typename U = T>
  std::enable_if_t<std::is_same_v<U, std::string>> Add(uint32_t idx,
                                                       absl::string_view val) {
    if (ABSL_PREDICT_FALSE(!event_indices.empty() &&
                           idx <= event_indices.back())) {
      if (idx == event_indices.back()) {
        value_items.back().assign(val.begin(), val.end());
        return;
      }
      needs_sort = true;
    }
    event_indices.push_back(idx);
    value_items.emplace_back(val);
  }

  template <typename U = T>
  std::enable_if_t<std::is_same_v<U, std::string>> Add(uint32_t idx,
                                                       const char* val) {
    Add(idx, absl::string_view(val));
  }

  // Sorts entries by `event_index` ascending and deduplicates (keeping the
  // last value added for any duplicate `event_index`) when out of order
  // additions occurred. No op on the common monotonic fast path.
  void SortAndDeduplicate() {
    if (ABSL_PREDICT_TRUE(!needs_sort)) {
      return;
    }
    needs_sort = false;
    const size_t n = event_indices.size();
    if (n <= 1) {
      return;
    }
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) {
      order[i] = i;
    }
    absl::c_stable_sort(order, [this](size_t a, size_t b) {
      return event_indices[a] < event_indices[b];
    });
    std::vector<uint32_t> sorted_indices;
    std::vector<T> sorted_values;
    sorted_indices.reserve(n);
    sorted_values.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      if (i + 1 < n && event_indices[order[i]] == event_indices[order[i + 1]]) {
        continue;
      }
      sorted_indices.push_back(event_indices[order[i]]);
      sorted_values.push_back(std::move(value_items[order[i]]));
    }
    event_indices = std::move(sorted_indices);
    value_items = std::move(sorted_values);
  }

  bool empty() const { return event_indices.empty(); }
  void clear() {
    event_indices.clear();
    value_items.clear();
    needs_sort = false;
  }
  size_t size() const { return event_indices.size(); }
};

using SparseInt64Column = SparseColumn<int64_t>;
using SparseUint64Column = SparseColumn<uint64_t>;
using SparseDoubleColumn = SparseColumn<double>;
using SparseStringColumn = SparseColumn<std::string>;

// Generic XStat column (used when include_stats is enabled), backed by a
// typed SparseColumn variant.
struct XStatColumn {
  using ValueColumn = std::variant<SparseInt64Column, SparseUint64Column,
                                   SparseDoubleColumn, SparseStringColumn>;

  static constexpr int64_t kStrValueTag = -1;
  static constexpr int64_t kBytesValueTag = -2;

  uint32_t stat_metadata_index_val = 0;
  ValueColumn values;
  // Optional parallel vector to `str_values()` when the stat column contains
  // `XStat::kRefValue` (`ref_ids[i] >= 0`, holding the referenced
  // `stat_metadata` ID, including ID 0) or `XStat::kBytesValue`
  // (`ref_ids[i] == kBytesValueTag`). Entries with `ref_ids[i] == kStrValueTag`
  // (or when `ref_ids` is empty) are regular `XStat::kStrValue` strings.
  std::vector<int64_t> ref_ids;

  uint32_t stat_metadata_index() const { return stat_metadata_index_val; }
  void set_stat_metadata_index(uint32_t v) { stat_metadata_index_val = v; }

  bool has_ref_id(size_t i) const {
    return i < ref_ids.size() && ref_ids[i] >= 0;
  }
  uint64_t ref_id(size_t i) const { return static_cast<uint64_t>(ref_ids[i]); }
  bool is_bytes(size_t i) const {
    return i < ref_ids.size() && ref_ids[i] == kBytesValueTag;
  }

  const std::vector<uint32_t>& event_index() const {
    return std::visit(
        [](const auto& col) -> const std::vector<uint32_t>& {
          return col.event_index();
        },
        values);
  }
  uint32_t event_index(int i) const { return event_index()[i]; }

  const std::vector<int64_t>& int64_values() const {
    return std::get<SparseInt64Column>(values).values();
  }
  void AddInt64(uint32_t idx, int64_t v) {
    if (!std::holds_alternative<SparseInt64Column>(values)) {
      values.emplace<SparseInt64Column>();
      ref_ids.clear();
    }
    std::get<SparseInt64Column>(values).Add(idx, v);
  }

  const std::vector<uint64_t>& uint64_values() const {
    return std::get<SparseUint64Column>(values).values();
  }
  void AddUint64(uint32_t idx, uint64_t v) {
    if (!std::holds_alternative<SparseUint64Column>(values)) {
      values.emplace<SparseUint64Column>();
      ref_ids.clear();
    }
    std::get<SparseUint64Column>(values).Add(idx, v);
  }

  const std::vector<double>& double_values() const {
    return std::get<SparseDoubleColumn>(values).values();
  }
  void AddDouble(uint32_t idx, double v) {
    if (!std::holds_alternative<SparseDoubleColumn>(values)) {
      values.emplace<SparseDoubleColumn>();
      ref_ids.clear();
    }
    std::get<SparseDoubleColumn>(values).Add(idx, v);
  }

  const std::vector<std::string>& str_values() const {
    return std::get<SparseStringColumn>(values).values();
  }
  void AddStr(uint32_t idx, std::string v) {
    if (!std::holds_alternative<SparseStringColumn>(values)) {
      values.emplace<SparseStringColumn>();
      ref_ids.clear();
    }
    auto& col = std::get<SparseStringColumn>(values);
    const size_t prev_size = col.size();
    col.Add(idx, std::move(v));
    if (!ref_ids.empty()) {
      if (col.size() == prev_size) {
        ref_ids.back() = kStrValueTag;
      } else {
        ref_ids.push_back(kStrValueTag);
      }
    }
  }
  void AddStr(uint32_t idx, absl::string_view v) {
    if (!std::holds_alternative<SparseStringColumn>(values)) {
      values.emplace<SparseStringColumn>();
      ref_ids.clear();
    }
    auto& col = std::get<SparseStringColumn>(values);
    const size_t prev_size = col.size();
    col.Add(idx, v);
    if (!ref_ids.empty()) {
      if (col.size() == prev_size) {
        ref_ids.back() = kStrValueTag;
      } else {
        ref_ids.push_back(kStrValueTag);
      }
    }
  }
  void AddStr(uint32_t idx, const char* v) {
    AddStr(idx, absl::string_view(v));
  }
  void AddRef(uint32_t idx, uint64_t ref_id, absl::string_view resolved_str) {
    if (ABSL_PREDICT_FALSE(ref_id > static_cast<uint64_t>(
                                        std::numeric_limits<int64_t>::max()))) {
      AddStr(idx, resolved_str);
      return;
    }
    if (!std::holds_alternative<SparseStringColumn>(values)) {
      values.emplace<SparseStringColumn>();
      ref_ids.clear();
    }
    auto& col = std::get<SparseStringColumn>(values);
    const size_t prev_size = col.size();
    if (ref_ids.size() < prev_size) {
      ref_ids.resize(prev_size, kStrValueTag);
    }
    col.Add(idx, resolved_str);
    if (col.size() == prev_size) {
      ref_ids.back() = static_cast<int64_t>(ref_id);
    } else {
      ref_ids.push_back(static_cast<int64_t>(ref_id));
    }
  }
  void AddBytes(uint32_t idx, std::string v) {
    if (!std::holds_alternative<SparseStringColumn>(values)) {
      values.emplace<SparseStringColumn>();
      ref_ids.clear();
    }
    auto& col = std::get<SparseStringColumn>(values);
    const size_t prev_size = col.size();
    if (ref_ids.size() < prev_size) {
      ref_ids.resize(prev_size, kStrValueTag);
    }
    col.Add(idx, std::move(v));
    if (col.size() == prev_size) {
      ref_ids.back() = kBytesValueTag;
    } else {
      ref_ids.push_back(kBytesValueTag);
    }
  }
  void AddBytes(uint32_t idx, absl::string_view v) {
    if (!std::holds_alternative<SparseStringColumn>(values)) {
      values.emplace<SparseStringColumn>();
      ref_ids.clear();
    }
    auto& col = std::get<SparseStringColumn>(values);
    const size_t prev_size = col.size();
    if (ref_ids.size() < prev_size) {
      ref_ids.resize(prev_size, kStrValueTag);
    }
    col.Add(idx, v);
    if (col.size() == prev_size) {
      ref_ids.back() = kBytesValueTag;
    } else {
      ref_ids.push_back(kBytesValueTag);
    }
  }
  void AddBytes(uint32_t idx, const char* v) {
    AddBytes(idx, absl::string_view(v));
  }

  void SortAndDeduplicate() {
    if (std::holds_alternative<SparseStringColumn>(values) &&
        !ref_ids.empty()) {
      auto& col = std::get<SparseStringColumn>(values);
      if (ABSL_PREDICT_TRUE(!col.needs_sort)) {
        return;
      }
      col.needs_sort = false;
      const size_t n = col.event_indices.size();
      if (n <= 1) {
        return;
      }
      if (ref_ids.size() < n) {
        ref_ids.resize(n, kStrValueTag);
      }
      std::vector<size_t> order(n);
      for (size_t i = 0; i < n; ++i) {
        order[i] = i;
      }
      absl::c_stable_sort(order, [&col](size_t a, size_t b) {
        return col.event_indices[a] < col.event_indices[b];
      });
      std::vector<uint32_t> sorted_indices;
      std::vector<std::string> sorted_values;
      std::vector<int64_t> sorted_refs;
      sorted_indices.reserve(n);
      sorted_values.reserve(n);
      sorted_refs.reserve(n);
      for (size_t i = 0; i < n; ++i) {
        if (i + 1 < n &&
            col.event_indices[order[i]] == col.event_indices[order[i + 1]]) {
          continue;
        }
        sorted_indices.push_back(col.event_indices[order[i]]);
        sorted_values.push_back(std::move(col.value_items[order[i]]));
        sorted_refs.push_back(ref_ids[order[i]]);
      }
      col.event_indices = std::move(sorted_indices);
      col.value_items = std::move(sorted_values);
      ref_ids = std::move(sorted_refs);
      return;
    }
    std::visit([](auto& col) { col.SortAndDeduplicate(); }, values);
  }
};

// A slice of the events of one line.
// Note on event ordering:
// - `ChunkXPlane` / `StreamingPlaneChunker` sort events in
//   `(offset asc, duration desc)` order, so `offset_ps_deltas` are
//   non negative within each chunk.
// - Direct streaming producers may emit events in completion order without
//   sorting at chunk flush time; `offset_ps_deltas` are signed `int64_t` values
//   and may be negative when adjacent events are out of start timestamp order.
//   Downstream consumers reconstruct `offset_ps += offset_ps_deltas[i]` using
//   signed arithmetic and sort per track when required by the output format.
struct XChunkEvents {
  uint32_t plane_index_val = 0;
  int64_t line_id_val = 0;
  uint32_t chunk_index_val = 0;
  bool last_chunk_val = false;
  bool flush_frame_val = false;
  uint32_t num_events_val = 0;

  ColumnVector<int64_t> offset_ps_deltas_vec;
  ColumnVector<int64_t> duration_ps_vec;
  ColumnVector<uint32_t> event_metadata_index_vec;

  SparseUint64Column flow_col;
  SparseInt64Column group_id_col;
  ColumnVector<uint32_t> async_event_index_vec;
  SparseStringColumn name_override_col;
  SparseUint64Column counter_uint64_col;
  SparseDoubleColumn counter_double_col;

  std::vector<XStatColumn> stats_vec;
  std::vector<int64_t> new_event_metadata_ids_vec;
  std::vector<std::string> new_event_names_vec;
  std::vector<std::string> new_event_display_names_vec;
  std::vector<int64_t> new_stat_metadata_ids_vec;
  std::vector<std::string> new_stat_names_vec;
  std::optional<XChunkLine> line_info;

  uint32_t plane_index() const { return plane_index_val; }
  void set_plane_index(uint32_t v) { plane_index_val = v; }

  int64_t line_id() const { return line_id_val; }
  void set_line_id(int64_t v) { line_id_val = v; }

  uint32_t chunk_index() const { return chunk_index_val; }
  void set_chunk_index(uint32_t v) { chunk_index_val = v; }

  bool last_chunk() const { return last_chunk_val; }
  void set_last_chunk(bool v) { last_chunk_val = v; }

  bool flush_frame() const { return flush_frame_val; }
  void set_flush_frame(bool v) { flush_frame_val = v; }

  uint32_t num_events() const { return num_events_val; }
  void set_num_events(uint32_t v) { num_events_val = v; }

  // Dense per event columns.
  const ColumnVector<int64_t>& offset_ps_deltas() const {
    return offset_ps_deltas_vec;
  }
  int64_t offset_ps_deltas(int i) const { return offset_ps_deltas_vec[i]; }
  size_t offset_ps_deltas_size() const { return offset_ps_deltas_vec.size(); }
  void add_offset_ps_deltas(int64_t v) { offset_ps_deltas_vec.push_back(v); }
  ColumnVector<int64_t>* mutable_offset_ps_deltas() {
    return &offset_ps_deltas_vec;
  }

  const ColumnVector<int64_t>& duration_ps() const { return duration_ps_vec; }
  int64_t duration_ps(int i) const { return duration_ps_vec[i]; }
  size_t duration_ps_size() const { return duration_ps_vec.size(); }
  void add_duration_ps(int64_t v) { duration_ps_vec.push_back(v); }
  ColumnVector<int64_t>* mutable_duration_ps() { return &duration_ps_vec; }

  const ColumnVector<uint32_t>& event_metadata_index() const {
    return event_metadata_index_vec;
  }
  uint32_t event_metadata_index(int i) const {
    return event_metadata_index_vec[i];
  }
  size_t event_metadata_index_size() const {
    return event_metadata_index_vec.size();
  }
  void add_event_metadata_index(uint32_t v) {
    event_metadata_index_vec.push_back(v);
  }
  ColumnVector<uint32_t>* mutable_event_metadata_index() {
    return &event_metadata_index_vec;
  }

  // Sparse per event columns.
  const SparseUint64Column& flow() const { return flow_col; }
  SparseUint64Column* mutable_flow() { return &flow_col; }

  const SparseInt64Column& group_id() const { return group_id_col; }
  SparseInt64Column* mutable_group_id() { return &group_id_col; }

  const ColumnVector<uint32_t>& async_event_index() const {
    return async_event_index_vec;
  }
  uint32_t async_event_index(int i) const { return async_event_index_vec[i]; }
  int async_event_index_size() const {
    return static_cast<int>(async_event_index_vec.size());
  }
  void add_async_event_index(uint32_t v) { async_event_index_vec.push_back(v); }

  const SparseStringColumn& name_override() const { return name_override_col; }
  SparseStringColumn* mutable_name_override() { return &name_override_col; }

  const SparseUint64Column& counter_uint64() const {
    return counter_uint64_col;
  }
  SparseUint64Column* mutable_counter_uint64() { return &counter_uint64_col; }

  const SparseDoubleColumn& counter_double() const {
    return counter_double_col;
  }
  SparseDoubleColumn* mutable_counter_double() { return &counter_double_col; }

  // Generic stats.
  int stats_size() const { return static_cast<int>(stats_vec.size()); }
  const std::vector<XStatColumn>& stats() const { return stats_vec; }
  const XStatColumn& stats(int i) const { return stats_vec[i]; }
  XStatColumn* mutable_stats(int i) { return &stats_vec[i]; }
  XStatColumn* add_stats() {
    stats_vec.emplace_back();
    return &stats_vec.back();
  }

  void Clear() {
    plane_index_val = 0;
    line_id_val = 0;
    chunk_index_val = 0;
    last_chunk_val = false;
    flush_frame_val = false;
    num_events_val = 0;
    offset_ps_deltas_vec.clear();
    duration_ps_vec.clear();
    event_metadata_index_vec.clear();
    flow_col.clear();
    group_id_col.clear();
    async_event_index_vec.clear();
    name_override_col.clear();
    counter_uint64_col.clear();
    counter_double_col.clear();
    stats_vec.clear();
    new_event_metadata_ids_vec.clear();
    new_event_names_vec.clear();
    new_event_display_names_vec.clear();
    new_stat_metadata_ids_vec.clear();
    new_stat_names_vec.clear();
    line_info.reset();
  }
};

// Emitted once at the end of a stream.
struct XChunkEnd {
  uint64_t num_events_val = 0;
  uint32_t num_planes_val = 0;

  uint64_t num_events() const { return num_events_val; }
  void set_num_events(uint64_t v) { num_events_val = v; }

  uint32_t num_planes() const { return num_planes_val; }
  void set_num_planes(uint32_t v) { num_planes_val = v; }
};

// XChunk variant representing one message in the stream.
class XChunk {
 public:
  enum KindCase {
    KIND_NOT_SET = 0,
    kStreamHeader = 1,
    kPlane = 2,
    kEvents = 3,
    kEnd = 4,
  };

  XChunk() = default;
  explicit XChunk(XChunkStreamHeader header) : data_(std::move(header)) {}
  explicit XChunk(XChunkPlane plane) : data_(std::move(plane)) {}
  explicit XChunk(XChunkEvents events) : data_(std::move(events)) {}
  explicit XChunk(XChunkEnd end) : data_(std::move(end)) {}

  KindCase kind_case() const {
    switch (data_.index()) {
      case 1:
        return kStreamHeader;
      case 2:
        return kPlane;
      case 3:
        return kEvents;
      case 4:
        return kEnd;
      default:
        return KIND_NOT_SET;
    }
  }

  bool has_stream_header() const {
    return std::holds_alternative<XChunkStreamHeader>(data_);
  }
  const XChunkStreamHeader& stream_header() const {
    return std::get<XChunkStreamHeader>(data_);
  }
  XChunkStreamHeader* mutable_stream_header() {
    if (!has_stream_header()) {
      data_.emplace<XChunkStreamHeader>();
    }
    return &std::get<XChunkStreamHeader>(data_);
  }

  bool has_plane() const { return std::holds_alternative<XChunkPlane>(data_); }
  const XChunkPlane& plane() const { return std::get<XChunkPlane>(data_); }
  XChunkPlane* mutable_plane() {
    if (!has_plane()) {
      data_.emplace<XChunkPlane>();
    }
    return &std::get<XChunkPlane>(data_);
  }

  bool has_events() const {
    return std::holds_alternative<XChunkEvents>(data_);
  }
  const XChunkEvents& events() const { return std::get<XChunkEvents>(data_); }
  XChunkEvents* mutable_events() {
    if (!has_events()) {
      data_.emplace<XChunkEvents>();
    }
    return &std::get<XChunkEvents>(data_);
  }

  bool has_end() const { return std::holds_alternative<XChunkEnd>(data_); }
  const XChunkEnd& end() const { return std::get<XChunkEnd>(data_); }
  XChunkEnd* mutable_end() {
    if (!has_end()) {
      data_.emplace<XChunkEnd>();
    }
    return &std::get<XChunkEnd>(data_);
  }

  // Rough memory size estimate for testing and metrics.
  size_t ByteSizeLong() const {
    switch (kind_case()) {
      case kStreamHeader: {
        const auto& h = stream_header();
        return sizeof(XChunkStreamHeader) + h.hostname_str.size();
      }
      case kPlane: {
        const auto& p = plane();
        size_t sz = sizeof(XChunkPlane) + p.plane_name_str.size() +
                    p.process_name_str.size();
        sz += p.event_metadata_ids_vec.size() * sizeof(int64_t);
        for (const auto& s : p.event_names_vec) {
          sz += s.size() + sizeof(s);
        }
        for (const auto& s : p.event_display_names_vec) {
          sz += s.size() + sizeof(s);
        }
        sz += p.stat_metadata_ids_vec.size() * sizeof(int64_t);
        for (const auto& s : p.stat_names_vec) {
          sz += s.size() + sizeof(s);
        }
        for (const auto& l : p.lines_vec) {
          sz += sizeof(XChunkLine) + l.name_str.size() +
                l.display_name_str.size();
        }
        return sz;
      }
      case kEvents: {
        const auto& e = events();
        size_t sz = sizeof(XChunkEvents);
        sz += e.offset_ps_deltas_vec.size() * sizeof(int64_t);
        sz += e.duration_ps_vec.size() * sizeof(int64_t);
        sz += e.event_metadata_index_vec.size() * sizeof(uint32_t);
        sz += e.flow_col.size() * (sizeof(uint32_t) + sizeof(uint64_t));
        sz += e.group_id_col.size() * (sizeof(uint32_t) + sizeof(int64_t));
        sz += e.async_event_index_vec.size() * sizeof(uint32_t);
        sz += e.name_override_col.size() * (sizeof(uint32_t) + 32);
        sz +=
            e.counter_uint64_col.size() * (sizeof(uint32_t) + sizeof(uint64_t));
        sz += e.counter_double_col.size() * (sizeof(uint32_t) + sizeof(double));
        return sz;
      }
      case kEnd:
        return sizeof(XChunkEnd);
      default:
        return 0;
    }
  }

 private:
  std::variant<std::monostate, XChunkStreamHeader, XChunkPlane, XChunkEvents,
               XChunkEnd>
      data_;
};

}  // namespace tsl::profiler::xchunk

#endif  // XLA_TSL_PROFILER_UTILS_XCHUNK_XCHUNK_H_
