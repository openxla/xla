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

#include "xla/tsl/profiler/utils/xchunk/xplane_to_xchunk.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xchunk/xchunk.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"

namespace tsl::profiler::xchunk {
namespace {

using ::tensorflow::profiler::XEvent;
using ::tensorflow::profiler::XEventMetadata;
using ::tensorflow::profiler::XLine;
using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XSpace;
using ::tensorflow::profiler::XStat;
using ::tensorflow::profiler::XStatMetadata;

// The special stats the trace viewer decodes from an event. Also counts the
// trace viewer arguments the event would have, which decides whether a counter
// event carries a value (exactly one numeric argument).
struct SpecialStats {
  std::optional<uint64_t> flow;
  std::optional<int64_t> group_id;
  bool is_async = false;
  std::optional<absl::string_view> step_name;
  int num_args = 0;
  std::optional<uint64_t> arg_uint64;
  std::optional<double> arg_double;
  std::optional<uint64_t> correlation_id;
  std::optional<uint64_t> producer_type;
  std::optional<uint64_t> consumer_type;
  std::optional<uint64_t> producer_id;
  std::optional<uint64_t> consumer_id;

  void AddNumericArg(std::optional<uint64_t> u, std::optional<double> d) {
    if (num_args++ == 0) {
      arg_uint64 = u;
      arg_double = d;
    } else {
      arg_uint64.reset();
      arg_double.reset();
    }
  }
  void AddOtherArg() {
    ++num_args;
    arg_uint64.reset();
    arg_double.reset();
  }
  bool HasCounterValue() const {
    return num_args == 1 && (arg_uint64.has_value() || arg_double.has_value());
  }
  void MaybeSynthesizeHostFlow(int32_t host_id) {
    if (host_id < 0 || flow.has_value()) {
      return;
    }
    if (correlation_id.has_value()) {
      XFlow xflow(XFlow::GetFlowId(host_id, *correlation_id),
                  XFlow::FlowDirection::kFlowOut, ContextType::kGpuLaunch);
      flow = xflow.ToStatValue();
    }
    if (producer_type.has_value() && producer_id.has_value()) {
      ContextType context_type = GetSafeContextType(*producer_type);
      XFlow xflow(XFlow::GetFlowId(host_id, *producer_id, context_type),
                  XFlow::FlowDirection::kFlowOut, context_type);
      flow = xflow.ToStatValue();
    }
    if (consumer_type.has_value() && consumer_id.has_value()) {
      ContextType context_type = GetSafeContextType(*consumer_type);
      XFlow xflow(XFlow::GetFlowId(host_id, *consumer_id, context_type),
                  XFlow::FlowDirection::kFlowIn, context_type);
      flow = xflow.ToStatValue();
    }
  }
};

struct StatInfo {
  std::optional<int64_t> type;
  bool internal = false;
  uint32_t dense_index = 0;
};

struct EventMetadataInfo {
  uint32_t dense_index = 0;
  // Rank of the metadata name among the plane's event names (equal names
  // share a rank), so events can be ordered by name without string compares.
  uint32_t name_rank = 0;
  bool internal = false;
  // Special stats declared on the XEventMetadata itself (shared by all the
  // occurrences of the event); per event stats override them.
  SpecialStats specials;
  // Non special generic stats declared on XEventMetadata (populated when
  // options_.include_stats is enabled).
  std::vector<std::pair<uint32_t /*stat_dense_index*/, const XStat*>>
      generic_stats;
};

// Maps XPlane metadata ids to dense indices. Ids are usually small and dense,
// so most lookups are a table access; the rest fall back to a hash map.
class IdIndex {
 public:
  static constexpr int64_t kMaxDirectId = 1 << 16;

  // `ids` must be sorted and unique; ids[i] maps to dense index i.
  void Build(absl::Span<const int64_t> ids) {
    int64_t direct_size = 0;
    if (!ids.empty() && ids.back() >= 0) {
      direct_size = ids.back() >= kMaxDirectId ? kMaxDirectId : ids.back() + 1;
    }
    direct_.assign(direct_size, -1);
    for (size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] >= 0 && ids[i] < static_cast<int64_t>(direct_.size())) {
        direct_[ids[i]] = static_cast<int32_t>(i);
      } else {
        map_[ids[i]] = static_cast<int32_t>(i);
      }
    }
  }

  // Returns the dense index of `id`, or -1 if unknown.
  int32_t Find(int64_t id) const {
    if (id >= 0 && id < static_cast<int64_t>(direct_.size())) {
      return direct_[id];
    }
    auto it = map_.find(id);
    return it == map_.end() ? -1 : it->second;
  }

 private:
  std::vector<int32_t> direct_;
  absl::flat_hash_map<int64_t, int32_t> map_;
};

// Sort key of one event: sorting these is far cheaper than sorting event
// indices with a comparator that chases XEvent pointers.
struct EventKey {
  int64_t offset_ps;
  int64_t duration_ps;
  uint32_t name_rank;
  uint32_t dense_index;  // into PlaneChunker::event_infos_
  uint32_t index;        // into XLine::events

  // (offset asc, duration desc, name asc): the trace viewer's event order,
  // with the original position as a tie breaker.
  friend bool operator<(const EventKey& a, const EventKey& b) {
    if (a.offset_ps != b.offset_ps) {
      return a.offset_ps < b.offset_ps;
    }
    if (a.duration_ps != b.duration_ps) {
      return a.duration_ps > b.duration_ps;
    }
    if (a.name_rank != b.name_rank) {
      return a.name_rank < b.name_rank;
    }
    return a.index < b.index;
  }
};

class PlaneChunker {
 public:
  PlaneChunker(const XPlane& plane, uint32_t plane_index, uint32_t process_id,
               absl::string_view process_name, const XChunkOptions& options,
               XChunkSink sink, int32_t host_id = -1)
      : plane_(plane),
        visitor_(CreateTfXPlaneVisitor(&plane)),
        plane_index_(plane_index),
        process_id_(process_id),
        process_name_(process_name),
        host_id_(host_id),
        options_(options),
        sink_(sink) {}

  void EmitPlaneHeader(bool include_initial_lines) {
    XChunk header;
    BuildPlane(header.mutable_plane(), include_initial_lines);
    sink_(std::move(header));
  }

  ChunkXPlaneStats Run() {
    EmitPlaneHeader(/*include_initial_lines=*/true);
    for (const XLine& line : plane_.lines()) {
      if (line.events_size() == 0) {
        continue;
      }
      ChunkLine(line, /*start_chunk_index=*/0, /*line_info=*/std::nullopt,
                /*is_last_slice=*/true);
      ++stats_.num_lines;
    }
    return stats_;
  }

  const XChunkLine& RegisterLineHeader(const XLine& line,
                                       uint64_t start_time_ns) {
    auto [it, inserted] = line_headers_.try_emplace(line.id());
    if (inserted) {
      ++stats_.num_lines;
      it->second.set_line_id(line.id());
      it->second.set_display_id(line.display_id() != 0 ? line.display_id()
                                                       : line.id());
    }
    XChunkLine& hdr = it->second;
    if (line.display_id() != 0) {
      hdr.set_display_id(line.display_id());
    }
    if (!line.name().empty()) {
      hdr.set_name(absl::string_view(line.name()));
    }
    if (!line.display_name().empty()) {
      hdr.set_display_name(absl::string_view(line.display_name()));
    }
    if (line.timestamp_ns() != 0) {
      int64_t ts_ns = line.timestamp_ns();
      if (start_time_ns > 0 && ts_ns >= 0 &&
          static_cast<uint64_t>(ts_ns) >= start_time_ns) {
        const uint64_t diff_ns = static_cast<uint64_t>(ts_ns) - start_time_ns;
        ts_ns = diff_ns <= static_cast<uint64_t>(
                               std::numeric_limits<int64_t>::max())
                    ? static_cast<int64_t>(diff_ns)
                    : std::numeric_limits<int64_t>::max();
      }
      hdr.set_timestamp_ns(ts_ns);
    }
    if (line.duration_ps() != 0) {
      hdr.set_duration_ps(line.duration_ps());
    }
    return hdr;
  }

  void ChunkLineStreaming(const XLine& line, uint64_t start_time_ns,
                          XChunkSink sink, bool is_last_slice) {
    sink_ = sink;
    const XChunkLine& hdr = RegisterLineHeader(line, start_time_ns);
    if (line.events_size() == 0) {
      return;
    }
    uint32_t& next_chunk_index = line_next_chunk_index_[line.id()];
    const size_t chunks_before = stats_.num_chunks;
    ChunkLine(line, next_chunk_index, hdr, is_last_slice);
    next_chunk_index +=
        static_cast<uint32_t>(stats_.num_chunks - chunks_before);
  }

  const ChunkXPlaneStats& stats() const { return stats_; }

 private:
  void BuildPlane(XChunkPlane* out, bool include_initial_lines) {
    out->set_plane_index(plane_index_);
    out->set_plane_id(plane_.id());
    out->set_plane_name(plane_.name());
    out->set_process_id(process_id_);
    out->set_process_name(process_name_);

    // Stat metadata table, sorted by id for determinism.
    std::vector<const XStatMetadata*> stat_metadata;
    stat_metadata.reserve(plane_.stat_metadata_size());
    // NOLINTNEXTLINE
    for (const auto& [id, metadata] : plane_.stat_metadata()) {
      stat_metadata.push_back(&metadata);
    }
    absl::c_sort(stat_metadata,
                 [](const XStatMetadata* a, const XStatMetadata* b) {
                   return a->id() < b->id();
                 });
    std::vector<int64_t> stat_ids;
    stat_ids.reserve(stat_metadata.size());
    stat_infos_.resize(stat_metadata.size());
    for (size_t i = 0; i < stat_metadata.size(); ++i) {
      const XStatMetadata* metadata = stat_metadata[i];
      stat_ids.push_back(metadata->id());
      StatInfo& info = stat_infos_[i];
      info.type = visitor_.GetStatType(metadata->id());
      info.internal = IsInternalStat(info.type);
      info.dense_index = i;
      if (options_.include_stats) {
        out->add_stat_metadata_ids(metadata->id());
        out->add_stat_names(metadata->name());
      }
    }
    stat_ids_.Build(stat_ids);

    // Event metadata table, sorted by id for determinism.
    std::vector<const XEventMetadata*> event_metadata;
    event_metadata.reserve(plane_.event_metadata_size());
    // NOLINTNEXTLINE
    for (const auto& [id, metadata] : plane_.event_metadata()) {
      event_metadata.push_back(&metadata);
    }
    absl::c_sort(event_metadata,
                 [](const XEventMetadata* a, const XEventMetadata* b) {
                   return a->id() < b->id();
                 });
    std::vector<int64_t> event_ids;
    event_ids.reserve(event_metadata.size());
    event_infos_.resize(event_metadata.size());
    for (size_t i = 0; i < event_metadata.size(); ++i) {
      const XEventMetadata* metadata = event_metadata[i];
      event_ids.push_back(metadata->id());
      out->AddEventMetadata(metadata->id(), metadata->name(),
                            metadata->display_name());
      EventMetadataInfo& info = event_infos_[i];
      info.dense_index = i;
      info.internal = IsInternalEvent(visitor_.GetEventType(metadata->id()));
      if (!metadata->display_name().empty()) {
        // The legacy converter adds a "long_name" argument for events with a
        // display name.
        info.specials.AddOtherArg();
      }
      for (const XStat& stat : metadata->stats()) {
        DecodeStat(stat, &info.specials, /*chunk_index=*/std::nullopt,
                   &info.generic_stats);
      }
    }
    event_ids_.Build(event_ids);
    // Name ranks: position of each distinct name in name order.
    std::vector<uint32_t> by_name(event_metadata.size());
    for (uint32_t i = 0; i < by_name.size(); ++i) {
      by_name[i] = i;
    }
    absl::c_sort(by_name, [&](uint32_t a, uint32_t b) {
      return event_metadata[a]->name() < event_metadata[b]->name();
    });
    uint32_t rank = 0;
    for (size_t n = 0; n < by_name.size(); ++n) {
      if (n > 0 && event_metadata[by_name[n - 1]]->name() !=
                       event_metadata[by_name[n]]->name()) {
        ++rank;
      }
      event_infos_[by_name[n]].name_rank = rank;
    }

    if (include_initial_lines) {
      for (const XLine& line : plane_.lines()) {
        if (line.events_size() == 0) {
          continue;
        }
        XChunkLine* out_line = out->add_lines();
        out_line->set_line_id(line.id());
        out_line->set_display_id(line.display_id());
        out_line->set_name(line.name());
        out_line->set_display_name(line.display_name());
        out_line->set_timestamp_ns(line.timestamp_ns());
        out_line->set_duration_ps(line.duration_ps());
        out_line->set_num_events(line.events_size());
      }
    }
  }

  // Decodes one XStat into `specials`. If `chunk_index` is set and generic
  // stats are requested, non special stats with valid metadata are also
  // appended to the current chunk's XStatColumn for the event at `chunk_index`.
  // When `chunk_index` is unset and `metadata_generic_stats` is non null,
  // non special metadata level stats are recorded for emission on each event.
  void DecodeStat(const XStat& stat, SpecialStats* specials,
                  std::optional<uint32_t> chunk_index,
                  std::vector<std::pair<uint32_t, const XStat*>>*
                      metadata_generic_stats = nullptr) {
    const int32_t dense = stat_ids_.Find(stat.metadata_id());
    // Stats without metadata behave like untyped, non internal stats.
    static const StatInfo kUnknownStat;
    const StatInfo& info = dense < 0 ? kUnknownStat : stat_infos_[dense];
    if (host_id_ >= 0 && info.type.has_value() &&
        (stat.value_case() == XStat::kInt64Value ||
         stat.value_case() == XStat::kUint64Value)) {
      const uint64_t val = stat.value_case() == XStat::kUint64Value
                               ? stat.uint64_value()
                               : static_cast<uint64_t>(stat.int64_value());
      switch (*info.type) {
        case StatType::kCorrelationId:
          specials->correlation_id = val;
          break;
        case StatType::kProducerType:
          specials->producer_type = val;
          break;
        case StatType::kConsumerType:
          specials->consumer_type = val;
          break;
        case StatType::kProducerId:
          specials->producer_id = val;
          break;
        case StatType::kConsumerId:
          specials->consumer_id = val;
          break;
        default:
          break;
      }
    }
    if (info.internal) {
      return;
    }
    const bool has_type = info.type.has_value();
    auto record_generic_stat = [&]() {
      if (dense < 0) {
        return;
      }
      if (chunk_index) {
        MaybeAddStat(info, stat, *chunk_index);
      } else if (options_.include_stats && metadata_generic_stats != nullptr) {
        metadata_generic_stats->emplace_back(info.dense_index, &stat);
      }
    };
    switch (stat.value_case()) {
      case XStat::kInt64Value:
        if (has_type && *info.type == StatType::kFlow) {
          specials->flow = static_cast<uint64_t>(stat.int64_value());
        } else if (has_type && *info.type == StatType::kGroupId) {
          specials->group_id = stat.int64_value();
        } else if (has_type && *info.type == StatType::kIsAsync) {
          specials->is_async = true;
        } else {
          // int64 arguments are not accepted as counter values.
          specials->AddOtherArg();
          record_generic_stat();
        }
        break;
      case XStat::kUint64Value:
        if (has_type && *info.type == StatType::kFlow) {
          specials->flow = stat.uint64_value();
        } else if (has_type && *info.type == StatType::kGroupId) {
          specials->group_id = static_cast<int64_t>(stat.uint64_value());
        } else if (has_type && *info.type == StatType::kIsAsync) {
          specials->is_async = true;
        } else {
          specials->AddNumericArg(stat.uint64_value(), std::nullopt);
          record_generic_stat();
        }
        break;
      case XStat::kDoubleValue:
        specials->AddNumericArg(std::nullopt, stat.double_value());
        record_generic_stat();
        break;
      case XStat::kStrValue:
      case XStat::kRefValue: {
        if (has_type && *info.type == StatType::kStepName) {
          if (stat.value_case() == XStat::kStrValue) {
            specials->step_name = absl::string_view(stat.str_value());
          } else if (const XStatMetadata* ref_meta =
                         visitor_.GetStatMetadata(stat.ref_value());
                     ref_meta != nullptr) {
            specials->step_name = absl::string_view(ref_meta->name());
          }
        }
        specials->AddOtherArg();
        record_generic_stat();
        break;
      }
      case XStat::kBytesValue:
        specials->AddOtherArg();
        record_generic_stat();
        break;
      case XStat::VALUE_NOT_SET:
        break;
    }
  }

  static bool StatColumnMatchesValueCase(const XStatColumn& column,
                                         XStat::ValueCase value_case) {
    if (column.event_index().empty()) {
      return true;
    }
    switch (value_case) {
      case XStat::kInt64Value:
        return std::holds_alternative<SparseInt64Column>(column.values);
      case XStat::kUint64Value:
        return std::holds_alternative<SparseUint64Column>(column.values);
      case XStat::kDoubleValue:
        return std::holds_alternative<SparseDoubleColumn>(column.values);
      case XStat::kStrValue:
      case XStat::kRefValue:
      case XStat::kBytesValue:
      default:
        return std::holds_alternative<SparseStringColumn>(column.values);
    }
  }

  void MaybeAddStat(const StatInfo& info, const XStat& stat,
                    uint32_t chunk_index) {
    if (!options_.include_stats) {
      return;
    }
    if (stat_column_index_.size() != stat_infos_.size()) {
      stat_column_index_.assign(stat_infos_.size(), -1);
    }
    int& column_index = stat_column_index_[info.dense_index];
    if (column_index < 0 ||
        !StatColumnMatchesValueCase(current_.stats(column_index),
                                    stat.value_case())) {
      column_index = -1;
      for (int idx = 0; idx < current_.stats_size(); ++idx) {
        if (current_.stats(idx).stat_metadata_index() == info.dense_index &&
            StatColumnMatchesValueCase(current_.stats(idx),
                                       stat.value_case())) {
          column_index = idx;
          break;
        }
      }
      if (column_index < 0) {
        column_index = current_.stats_size();
        current_.add_stats()->set_stat_metadata_index(info.dense_index);
      }
    }
    XStatColumn* column = current_.mutable_stats(column_index);
    switch (stat.value_case()) {
      case XStat::kInt64Value:
        column->AddInt64(chunk_index, stat.int64_value());
        break;
      case XStat::kUint64Value:
        column->AddUint64(chunk_index, stat.uint64_value());
        break;
      case XStat::kDoubleValue:
        column->AddDouble(chunk_index, stat.double_value());
        break;
      case XStat::kStrValue:
        column->AddStr(chunk_index, absl::string_view(stat.str_value()));
        break;
      case XStat::kRefValue: {
        const XStatMetadata* ref_meta =
            visitor_.GetStatMetadata(stat.ref_value());
        column->AddRef(
            chunk_index, stat.ref_value(),
            ref_meta != nullptr ? ref_meta->name() : absl::string_view());
        break;
      }
      case XStat::kBytesValue:
        column->AddBytes(chunk_index, absl::string_view(stat.bytes_value()));
        break;
      default:
        column->AddStr(chunk_index, "");
        break;
    }
  }

  void ChunkLine(const XLine& line, uint32_t start_chunk_index,
                 const std::optional<XChunkLine>& line_info,
                 bool is_last_slice) {
    const absl::string_view line_name = line_info.has_value()
                                            ? line_info->name()
                                            : absl::string_view(line.name());
    const bool counter_line = line_name == kCounterEventsLineName;
    const auto& events = line.events();
    const absl::Time line_start = absl::Now();
    // One pass to collect the compact sort keys of the visible events.
    std::vector<EventKey> keys;
    keys.reserve(events.size());
    for (uint32_t index = 0; index < static_cast<uint32_t>(events.size());
         ++index) {
      const XEvent& event = events[index];
      const int32_t dense = event_ids_.Find(event.metadata_id());
      if (dense < 0 || event_infos_[dense].internal) {
        continue;
      }
      keys.push_back({event.offset_ps(), event.duration_ps(),
                      event_infos_[dense].name_rank,
                      static_cast<uint32_t>(dense), index});
    }
    if (keys.empty() && line_info.has_value()) {
      return;
    }
    const absl::Time keys_done = absl::Now();
    // Converter output is normally already in order: check before sorting.
    const bool sorted = absl::c_is_sorted(keys);
    if (!sorted) {
      absl::c_sort(keys);
    }
    const absl::Time sort_done = absl::Now();
    const size_t chunks_before = stats_.num_chunks;
    EmitLine(line, counter_line, events, keys, start_chunk_index, line_info,
             is_last_slice);
    VLOG(2) << "XChunk: line " << line_name << " (" << keys.size() << "/"
            << events.size() << " events, "
            << (sorted ? "already sorted" : "sorted") << "): keys "
            << keys_done - line_start << ", sort " << sort_done - keys_done
            << ", emit " << absl::Now() - sort_done << " in "
            << stats_.num_chunks - chunks_before << " chunks";
  }

  void EmitLine(const XLine& line, bool counter_line,
                const tsl::protobuf::RepeatedPtrField<XEvent>& events,
                const std::vector<EventKey>& keys, uint32_t start_chunk_index,
                const std::optional<XChunkLine>& line_info,
                bool is_last_slice) {
    StartChunk(line, /*chunk_index=*/start_chunk_index, keys.size(), line_info);
    for (size_t n = 0; n < keys.size(); ++n) {
      const EventKey& key = keys[n];
      const XEvent& event = events[key.index];
      const EventMetadataInfo& info = event_infos_[key.dense_index];
      if (num_in_chunk_ == options_.max_events_per_chunk) {
        EmitChunk(/*last_chunk=*/false);
        StartChunk(line, chunk_index_ + 1, keys.size() - n, line_info);
      }
      const uint32_t i = num_in_chunk_++;
      current_.add_offset_ps_deltas(key.offset_ps - last_offset_ps_);
      last_offset_ps_ = key.offset_ps;
      current_.add_duration_ps(key.duration_ps);
      current_.add_event_metadata_index(info.dense_index);

      SpecialStats specials = info.specials;
      if (!info.generic_stats.empty()) {
        for (const auto& [stat_dense, meta_stat] : info.generic_stats) {
          const int64_t meta_stat_id = meta_stat->metadata_id();
          const bool overridden = absl::c_any_of(
              event.stats(), [meta_stat_id](const XStat& ev_stat) {
                return ev_stat.metadata_id() == meta_stat_id;
              });
          if (!overridden) {
            MaybeAddStat(stat_infos_[stat_dense], *meta_stat, i);
          }
        }
      }
      for (const XStat& stat : event.stats()) {
        DecodeStat(stat, &specials, i);
      }
      specials.MaybeSynthesizeHostFlow(host_id_);
      if (specials.flow) {
        current_.mutable_flow()->Add(i, *specials.flow);
      }
      if (specials.group_id) {
        current_.mutable_group_id()->Add(i, *specials.group_id);
      }
      if (specials.is_async) {
        current_.add_async_event_index(i);
      }
      if (specials.step_name) {
        current_.mutable_name_override()->Add(i, *specials.step_name);
      }
      if (counter_line && specials.HasCounterValue()) {
        if (specials.arg_uint64) {
          current_.mutable_counter_uint64()->Add(i, *specials.arg_uint64);
        } else {
          current_.mutable_counter_double()->Add(i, *specials.arg_double);
        }
      }
      ++stats_.num_events;
    }
    EmitChunk(/*last_chunk=*/is_last_slice);
  }

  void StartChunk(const XLine& line, uint32_t chunk_index, size_t remaining,
                  const std::optional<XChunkLine>& line_info) {
    current_.Clear();
    current_.set_plane_index(plane_index_);
    current_.set_line_id(line.id());
    current_.set_chunk_index(chunk_index);
    if (line_info.has_value()) {
      current_.line_info = *line_info;
    }
    const int reserve = static_cast<int>(
        std::min<size_t>(remaining, options_.max_events_per_chunk));
    current_.mutable_offset_ps_deltas()->Reserve(reserve);
    current_.mutable_duration_ps()->Reserve(reserve);
    current_.mutable_event_metadata_index()->Reserve(reserve);
    chunk_index_ = chunk_index;
    num_in_chunk_ = 0;
    last_offset_ps_ = 0;
    stat_column_index_.clear();
  }

  void EmitChunk(bool last_chunk) {
    current_.set_num_events(num_in_chunk_);
    current_.set_last_chunk(last_chunk);
    XChunk chunk;
    *chunk.mutable_events() = std::move(current_);
    current_.Clear();
    sink_(std::move(chunk));
    ++stats_.num_chunks;
  }

  const XPlane& plane_;
  XPlaneVisitor visitor_;
  uint32_t plane_index_;
  uint32_t process_id_;
  std::string process_name_;
  int32_t host_id_;
  XChunkOptions options_;
  XChunkSink sink_;

  IdIndex stat_ids_;
  std::vector<StatInfo> stat_infos_;  // by dense index
  IdIndex event_ids_;
  std::vector<EventMetadataInfo> event_infos_;  // by dense index

  absl::flat_hash_map<int64_t, XChunkLine> line_headers_;
  absl::flat_hash_map<int64_t, uint32_t> line_next_chunk_index_;

  XChunkEvents current_;
  uint32_t chunk_index_ = 0;
  uint32_t num_in_chunk_ = 0;
  int64_t last_offset_ps_ = 0;
  std::vector<int> stat_column_index_;
  ChunkXPlaneStats stats_;
};

XPlane CopyPlaneMetadataOnly(const XPlane& plane) {
  XPlane copy;
  copy.set_id(plane.id());
  copy.set_name(plane.name());
  *copy.mutable_event_metadata() = plane.event_metadata();
  *copy.mutable_stat_metadata() = plane.stat_metadata();
  return copy;
}

}  // namespace

class StreamingPlaneChunker::Impl {
 public:
  Impl(const XPlane& metadata_plane, uint32_t plane_index, uint32_t process_id,
       absl::string_view process_name, int32_t host_id,
       const XChunkOptions& options, XChunkSink sink)
      : metadata_plane_(CopyPlaneMetadataOnly(metadata_plane)),
        chunker_(metadata_plane_, plane_index, process_id, process_name,
                 options, sink, host_id) {
    chunker_.EmitPlaneHeader(/*include_initial_lines=*/false);
  }

  void RegisterLineHeader(const XLine& line, uint64_t start_time_ns) {
    chunker_.RegisterLineHeader(line, start_time_ns);
  }

  void ChunkLine(const XLine& line, uint64_t start_time_ns, XChunkSink sink,
                 bool is_last_slice) {
    chunker_.ChunkLineStreaming(line, start_time_ns, sink, is_last_slice);
  }

  const ChunkXPlaneStats& stats() const { return chunker_.stats(); }

 private:
  XPlane metadata_plane_;
  PlaneChunker chunker_;
};

StreamingPlaneChunker::StreamingPlaneChunker(
    const XPlane& metadata_plane, uint32_t plane_index, uint32_t process_id,
    absl::string_view process_name, int32_t host_id,
    const XChunkOptions& options, XChunkSink sink)
    : impl_(std::make_unique<Impl>(metadata_plane, plane_index, process_id,
                                   process_name, host_id, options, sink)) {}

StreamingPlaneChunker::~StreamingPlaneChunker() = default;
StreamingPlaneChunker::StreamingPlaneChunker(StreamingPlaneChunker&&) noexcept =
    default;
StreamingPlaneChunker& StreamingPlaneChunker::operator=(
    StreamingPlaneChunker&&) noexcept = default;

void StreamingPlaneChunker::RegisterLineHeader(const XLine& line,
                                               uint64_t start_time_ns) {
  impl_->RegisterLineHeader(line, start_time_ns);
}

void StreamingPlaneChunker::ChunkLine(const XLine& line, uint64_t start_time_ns,
                                      XChunkSink sink, bool is_last_slice) {
  impl_->ChunkLine(line, start_time_ns, sink, is_last_slice);
}

const ChunkXPlaneStats& StreamingPlaneChunker::stats() const {
  return impl_->stats();
}

ChunkXPlaneStats ChunkXPlane(const XPlane& plane, uint32_t plane_index,
                             uint32_t process_id,
                             absl::string_view process_name,
                             const XChunkOptions& options, XChunkSink sink) {
  PlaneChunker chunker(plane, plane_index, process_id, process_name, options,
                       sink);
  return chunker.Run();
}

uint32_t TraceViewerProcessId(const XPlane& plane, int host_id,
                              int* next_custom_plane_index,
                              int* next_host_threads_plane_index) {
  const uint32_t base_pid =
      static_cast<uint32_t>(std::max(0, host_id)) * kMaxDevicesPerHost;
  if (plane.name() == kHostThreadsPlaneName) {
    return base_pid + kHostThreadsDeviceId;
  }
  if (absl::StartsWith(plane.name(), kHostThreadsPlaneName)) {
    const int sub_index = next_host_threads_plane_index != nullptr
                              ? (*next_host_threads_plane_index)++
                              : 0;
    DCHECK_GE(sub_index, 0);
    const uint32_t sub_pid =
        std::min<uint32_t>(kHostThreadsDeviceId + 1 +
                               static_cast<uint32_t>(std::max(0, sub_index)),
                           kMaxDevicesPerHost - 2);
    return base_pid + sub_pid;
  }
  if (plane.name() == kHostCpusPlaneName) {
    return base_pid + (kMaxDevicesPerHost - 1);
  }
  if (plane.name() == kMetadataPlaneName) {
    return base_pid;
  }
  if (absl::StartsWith(plane.name(), kCustomPlanePrefix)) {
    const int custom_index =
        next_custom_plane_index != nullptr ? (*next_custom_plane_index)++ : 0;
    DCHECK_GE(custom_index, 0);
    const uint32_t custom_pid =
        std::min<uint32_t>(kFirstCustomPlaneDeviceId +
                               static_cast<uint32_t>(std::max(0, custom_index)),
                           kLastCustomPlaneDeviceId);
    return base_pid + custom_pid;
  }
  DCHECK_GE(plane.id(), 0);
  const uint32_t device_pid = static_cast<uint32_t>(std::min<int64_t>(
      kFirstDeviceId + std::max<int64_t>(0, plane.id()), kLastDeviceId));
  return base_pid + device_pid;
}

void StreamXSpaceToXChunks(const XSpace& space, absl::string_view hostname,
                           int host_id, const XChunkOptions& options,
                           XChunkSink sink) {
  XChunk header;
  XChunkStreamHeader* stream_header = header.mutable_stream_header();
  stream_header->set_hostname(std::string(hostname));
  stream_header->set_host_id(host_id);
  for (const XPlane& plane : space.planes()) {
    if (absl::StartsWith(plane.name(), kTpuPlanePrefix)) {
      stream_header->set_device_family(XChunkStreamHeader::TPU);
      break;
    }
    if (absl::StartsWith(plane.name(), kGpuPlanePrefix)) {
      stream_header->set_device_family(XChunkStreamHeader::GPU);
      break;
    }
  }
  sink(std::move(header));

  // Precompute subprocess host threads PIDs in deterministic name order so
  // `/host:CPU [pid]` planes map consistently regardless of XSpace plane order.
  std::vector<const XPlane*> subprocess_host_planes;
  for (const XPlane& plane : space.planes()) {
    if (plane.lines_size() == 0 || plane.name() == kMetadataPlaneName) {
      continue;
    }
    if (plane.name() != kHostThreadsPlaneName &&
        absl::StartsWith(plane.name(), kHostThreadsPlaneName)) {
      subprocess_host_planes.push_back(&plane);
    }
  }
  absl::c_sort(subprocess_host_planes, [](const XPlane* a, const XPlane* b) {
    return a->name() < b->name();
  });
  absl::flat_hash_map<const XPlane*, uint32_t> subprocess_pids;
  subprocess_pids.reserve(subprocess_host_planes.size());
  int next_host_threads_plane_index = 0;
  for (const XPlane* plane : subprocess_host_planes) {
    subprocess_pids.emplace(
        plane, TraceViewerProcessId(*plane, host_id, nullptr,
                                    &next_host_threads_plane_index));
  }

  uint64_t num_events = 0;
  uint32_t num_planes = 0;
  int next_custom_plane_index = 0;
  for (const XPlane& plane : space.planes()) {
    if (plane.lines_size() == 0 || plane.name() == kMetadataPlaneName) {
      continue;
    }
    uint32_t pid = 0;
    if (auto it = subprocess_pids.find(&plane); it != subprocess_pids.end()) {
      pid = it->second;
    } else {
      pid = TraceViewerProcessId(plane, host_id, &next_custom_plane_index);
    }
    ChunkXPlaneStats stats =
        ChunkXPlane(plane, num_planes, pid,
                    absl::StrCat(hostname, " ", plane.name()), options, sink);
    num_events += stats.num_events;
    ++num_planes;
  }

  XChunk end;
  end.mutable_end()->set_num_events(num_events);
  end.mutable_end()->set_num_planes(num_planes);
  sink(std::move(end));
}

namespace {

int64_t GetOrCreateSpecialStatId(XPlane* out_plane,
                                 absl::string_view stat_name) {
  int64_t max_id = 0;
  std::optional<int64_t> found_id;
  // NOLINTNEXTLINE
  for (const auto& [id, meta] : out_plane->stat_metadata()) {
    if (!stat_name.empty() && meta.name() == stat_name &&
        (!found_id.has_value() || id < *found_id)) {
      found_id = id;
    }
    if (id >= max_id) {
      max_id = id < std::numeric_limits<int64_t>::max() ? id + 1 : id;
    }
  }
  if (found_id.has_value()) {
    return *found_id;
  }
  if (max_id == 0) {
    max_id = 1;
  }
  const auto* stat_metas = &out_plane->stat_metadata();
  if (stat_metas->find(max_id) != stat_metas->end()) {
    int64_t candidate = 1;
    while (stat_metas->find(candidate) != stat_metas->end() &&
           candidate < std::numeric_limits<int64_t>::max()) {
      ++candidate;
    }
    max_id = candidate;
  }
  XStatMetadata& meta = (*out_plane->mutable_stat_metadata())[max_id];
  meta.set_id(max_id);
  meta.set_name(std::string(stat_name));
  return max_id;
}

XLine* FindOrAddLine(XPlane* out_plane, int64_t line_id) {
  for (XLine& line : *out_plane->mutable_lines()) {
    if (line.id() == line_id) {
      return &line;
    }
  }
  XLine* line = out_plane->add_lines();
  line->set_id(line_id);
  return line;
}

void ApplyLineInfo(const XChunkLine& info, XLine* line) {
  line->set_id(info.line_id());
  if (info.display_id() != 0) {
    line->set_display_id(info.display_id());
  }
  if (!info.name().empty()) {
    line->set_name(std::string(info.name()));
  }
  if (!info.display_name().empty()) {
    line->set_display_name(std::string(info.display_name()));
  }
  if (info.timestamp_ns() != 0) {
    line->set_timestamp_ns(info.timestamp_ns());
  }
  if (info.duration_ps() != 0) {
    line->set_duration_ps(info.duration_ps());
  }
}

template <typename ColT, typename SetFn>
void AppendSparseStats(const ColT& col, int base_idx, uint32_t num_events,
                       int64_t stat_id,
                       tsl::protobuf::RepeatedPtrField<XEvent>* proto_events,
                       SetFn&& set_fn) {
  const size_t n = std::min(col.event_indices.size(), col.value_items.size());
  for (size_t k = 0; k < n; ++k) {
    const uint32_t ev_idx = col.event_index(k);
    if (ABSL_PREDICT_FALSE(ev_idx >= num_events)) {
      continue;
    }
    XStat* st = (*proto_events)[base_idx + ev_idx].add_stats();
    st->set_metadata_id(stat_id);
    set_fn(st, col.values(k));
  }
}

template <typename FindOrAddLineFn, typename GetOrCreateStatIdFn,
          typename RegisterStatMetaFn, typename RegisterEventMetaFn>
void AppendXChunkToXPlaneImpl(const XChunk& chunk,
                              const XChunkPlane* current_plane,
                              XPlane* out_plane,
                              FindOrAddLineFn&& find_or_add_line,
                              GetOrCreateStatIdFn&& get_or_create_stat_id,
                              RegisterStatMetaFn&& register_stat_meta,
                              RegisterEventMetaFn&& register_event_meta) {
  if (out_plane == nullptr) {
    return;
  }
  if (chunk.has_plane()) {
    const XChunkPlane& p = chunk.plane();
    out_plane->set_id(p.plane_id());
    if (!p.plane_name().empty()) {
      out_plane->set_name(std::string(p.plane_name()));
    }
    auto* ev_metas = out_plane->mutable_event_metadata();
    const size_t num_ev_metas =
        std::min(p.event_metadata_ids().size(), p.event_names_size());
    for (size_t i = 0; i < num_ev_metas; ++i) {
      const int64_t id = p.event_metadata_ids(i);
      XEventMetadata& m = (*ev_metas)[id];
      m.set_id(id);
      if (m.name().empty() && !p.event_names(i).empty()) {
        m.set_name(std::string(p.event_names(i)));
      }
      if (i < p.event_display_names_size() && m.display_name().empty() &&
          !p.event_display_names(i).empty()) {
        m.set_display_name(std::string(p.event_display_names(i)));
      }
      register_event_meta(id);
    }
    auto* stat_metas = out_plane->mutable_stat_metadata();
    const size_t num_stat_metas =
        std::min(p.stat_metadata_ids().size(), p.stat_names_size());
    for (size_t i = 0; i < num_stat_metas; ++i) {
      const int64_t id = p.stat_metadata_ids(i);
      XStatMetadata& m = (*stat_metas)[id];
      m.set_id(id);
      if (m.name().empty() && !p.stat_names(i).empty()) {
        m.set_name(std::string(p.stat_names(i)));
      }
      register_stat_meta(id, m.name());
    }
    for (const XChunkLine& l : p.lines()) {
      XLine* line = find_or_add_line(l.line_id());
      ApplyLineInfo(l, line);
    }
    return;
  }

  if (!chunk.has_events()) {
    return;
  }
  const XChunkEvents& evs = chunk.events();
  if (evs.num_events() == 0 && !evs.line_info.has_value()) {
    return;
  }

  XLine* line = find_or_add_line(evs.line_id());
  if (evs.line_info.has_value()) {
    ApplyLineInfo(*evs.line_info, line);
  }
  const uint32_t num_events = static_cast<uint32_t>(std::min<size_t>(
      {static_cast<size_t>(evs.num_events()), evs.offset_ps_deltas_size(),
       evs.duration_ps_size(), evs.event_metadata_index_size()}));
  if (num_events == 0) {
    return;
  }

  auto* proto_events = line->mutable_events();
  const int base_idx = proto_events->size();
  proto_events->Reserve(base_idx + static_cast<int>(num_events));

  const auto* ev_meta_ids =
      current_plane != nullptr ? &current_plane->event_metadata_ids() : nullptr;
  int64_t offset_ps = 0;
  for (uint32_t i = 0; i < num_events; ++i) {
    offset_ps += evs.offset_ps_deltas(i);
    XEvent* ev = proto_events->Add();
    ev->set_offset_ps(offset_ps);
    const uint64_t raw_dur = static_cast<uint64_t>(evs.duration_ps(i));
    ev->set_duration_ps(static_cast<int64_t>(raw_dur & kDurationValueMask));
    const uint32_t dense_meta_idx = evs.event_metadata_index(i);
    const int64_t meta_id =
        (ev_meta_ids != nullptr && dense_meta_idx < ev_meta_ids->size())
            ? (*ev_meta_ids)[dense_meta_idx]
            : static_cast<int64_t>(dense_meta_idx);
    ev->set_metadata_id(meta_id);
  }

  // Materialize generic stats first so their stat_metadata IDs (and any
  // referenced `kRefValue` target IDs) are registered in `out_plane` before
  // special stats allocate new IDs.
  const auto* stat_meta_ids =
      current_plane != nullptr ? &current_plane->stat_metadata_ids() : nullptr;
  for (const XStatColumn& scol : evs.stats()) {
    const uint32_t dense_stat_idx = scol.stat_metadata_index();
    const int64_t stat_id =
        (stat_meta_ids != nullptr && dense_stat_idx < stat_meta_ids->size())
            ? (*stat_meta_ids)[dense_stat_idx]
            : static_cast<int64_t>(dense_stat_idx);
    if (std::holds_alternative<SparseInt64Column>(scol.values)) {
      AppendSparseStats(std::get<SparseInt64Column>(scol.values), base_idx,
                        num_events, stat_id, proto_events,
                        [](XStat* st, int64_t v) { st->set_int64_value(v); });
    } else if (std::holds_alternative<SparseUint64Column>(scol.values)) {
      AppendSparseStats(std::get<SparseUint64Column>(scol.values), base_idx,
                        num_events, stat_id, proto_events,
                        [](XStat* st, uint64_t v) { st->set_uint64_value(v); });
    } else if (std::holds_alternative<SparseDoubleColumn>(scol.values)) {
      AppendSparseStats(std::get<SparseDoubleColumn>(scol.values), base_idx,
                        num_events, stat_id, proto_events,
                        [](XStat* st, double v) { st->set_double_value(v); });
    } else if (std::holds_alternative<SparseStringColumn>(scol.values)) {
      const auto& col = std::get<SparseStringColumn>(scol.values);
      const size_t n =
          std::min(col.event_indices.size(), col.value_items.size());
      for (size_t k = 0; k < n; ++k) {
        const uint32_t ev_idx = col.event_index(k);
        if (ABSL_PREDICT_FALSE(ev_idx >= num_events)) {
          continue;
        }
        XStat* st = (*proto_events)[base_idx + ev_idx].add_stats();
        st->set_metadata_id(stat_id);
        if (scol.has_ref_id(k)) {
          int64_t ref_id = static_cast<int64_t>(scol.ref_id(k));
          const std::string& resolved_str = col.values(k);
          auto* stat_metas = out_plane->mutable_stat_metadata();
          auto [it, inserted] =
              stat_metas->try_emplace(ref_id, XStatMetadata());
          if (inserted) {
            it->second.set_id(ref_id);
            it->second.set_name(resolved_str);
            register_stat_meta(ref_id, resolved_str);
          } else if (it->second.name().empty() && !resolved_str.empty()) {
            it->second.set_name(resolved_str);
            register_stat_meta(ref_id, resolved_str);
          } else if (it->second.name() != resolved_str) {
            ref_id = get_or_create_stat_id(resolved_str);
          }
          st->set_ref_value(static_cast<uint64_t>(ref_id));
        } else if (scol.is_bytes(k)) {
          st->set_bytes_value(col.values(k));
        } else {
          st->set_str_value(col.values(k));
        }
      }
    }
  }

  if (!evs.flow().empty()) {
    const int64_t stat_id =
        get_or_create_stat_id(GetStatTypeStr(StatType::kFlow));
    AppendSparseStats(evs.flow(), base_idx, num_events, stat_id, proto_events,
                      [](XStat* st, uint64_t v) { st->set_uint64_value(v); });
  }
  if (!evs.group_id().empty()) {
    const int64_t stat_id =
        get_or_create_stat_id(GetStatTypeStr(StatType::kGroupId));
    AppendSparseStats(evs.group_id(), base_idx, num_events, stat_id,
                      proto_events,
                      [](XStat* st, int64_t v) { st->set_int64_value(v); });
  }
  if (evs.async_event_index_size() > 0) {
    const int64_t stat_id =
        get_or_create_stat_id(GetStatTypeStr(StatType::kIsAsync));
    for (int k = 0; k < evs.async_event_index_size(); ++k) {
      const uint32_t ev_idx = evs.async_event_index(k);
      if (ABSL_PREDICT_FALSE(ev_idx >= num_events)) {
        continue;
      }
      XStat* st = (*proto_events)[base_idx + ev_idx].add_stats();
      st->set_metadata_id(stat_id);
      st->set_int64_value(1);
    }
  }
  if (!evs.name_override().empty()) {
    const int64_t stat_id =
        get_or_create_stat_id(GetStatTypeStr(StatType::kStepName));
    AppendSparseStats(
        evs.name_override(), base_idx, num_events, stat_id, proto_events,
        [](XStat* st, absl::string_view v) { st->set_str_value(v); });
  }

  if (evs.stats().empty()) {
    if (!evs.counter_uint64().empty()) {
      const int64_t stat_id = get_or_create_stat_id("value");
      AppendSparseStats(evs.counter_uint64(), base_idx, num_events, stat_id,
                        proto_events,
                        [](XStat* st, uint64_t v) { st->set_uint64_value(v); });
    }
    if (!evs.counter_double().empty()) {
      const int64_t stat_id = get_or_create_stat_id("value");
      AppendSparseStats(evs.counter_double(), base_idx, num_events, stat_id,
                        proto_events,
                        [](XStat* st, double v) { st->set_double_value(v); });
    }
  }
}

}  // namespace

void AppendXChunkToXPlane(const XChunk& chunk,
                          const XChunkPlane* absl_nullable current_plane,
                          XPlane* absl_nullable out_plane) {
  AppendXChunkToXPlaneImpl(
      chunk, current_plane, out_plane,
      [out_plane](int64_t line_id) {
        return FindOrAddLine(out_plane, line_id);
      },
      [out_plane](absl::string_view stat_name) {
        return GetOrCreateSpecialStatId(out_plane, stat_name);
      },
      [](int64_t, absl::string_view) {}, [](int64_t) {});
}

XChunkToXPlaneConverter::XChunkToXPlaneConverter(
    XPlane* absl_nullable out_plane)
    : out_plane_(out_plane) {
  if (out_plane_ == nullptr) {
    return;
  }
  XPlane& out_plane_ref = *out_plane_;
  for (XLine& line : *out_plane_ref.mutable_lines()) {
    lines_by_id_[line.id()] = &line;
  }
  // NOLINTNEXTLINE
  for (const auto& [id, meta] : out_plane_ref.stat_metadata()) {
    RegisterStatMetadata(id, meta.name());
  }
  // NOLINTNEXTLINE
  for (const auto& [id, _] : out_plane_ref.event_metadata()) {
    RegisterEventMetadataId(id);
  }
}

void XChunkToXPlaneConverter::RegisterStatMetadata(int64_t id,
                                                   absl::string_view name) {
  if (id >= next_stat_id_) {
    next_stat_id_ = id < std::numeric_limits<int64_t>::max() ? id + 1 : id;
  }
  if (!name.empty()) {
    auto [it, inserted] = stat_ids_by_name_.try_emplace(name, id);
    if (!inserted && id < it->second) {
      it->second = id;
    }
  }
}

void XChunkToXPlaneConverter::RegisterEventMetadataId(int64_t id) {
  if (id >= next_event_id_) {
    next_event_id_ = id < std::numeric_limits<int64_t>::max() ? id + 1 : id;
  }
}

int64_t XChunkToXPlaneConverter::GetOrCreateSpecialStatId(
    absl::string_view stat_name) {
  if (!stat_name.empty()) {
    auto it = stat_ids_by_name_.find(stat_name);
    if (it != stat_ids_by_name_.end()) {
      return it->second;
    }
  }
  auto* stat_metas = out_plane_->mutable_stat_metadata();
  while (stat_metas->find(next_stat_id_) != stat_metas->end() &&
         next_stat_id_ < std::numeric_limits<int64_t>::max()) {
    ++next_stat_id_;
  }
  if (stat_metas->find(next_stat_id_) != stat_metas->end()) {
    next_stat_id_ = 1;
    while (stat_metas->find(next_stat_id_) != stat_metas->end() &&
           next_stat_id_ < std::numeric_limits<int64_t>::max()) {
      ++next_stat_id_;
    }
  }
  const int64_t id = next_stat_id_ < std::numeric_limits<int64_t>::max()
                         ? next_stat_id_++
                         : next_stat_id_;
  XStatMetadata& meta = (*stat_metas)[id];
  meta.set_id(id);
  meta.set_name(std::string(stat_name));
  if (!stat_name.empty()) {
    stat_ids_by_name_.emplace(meta.name(), id);
  }
  return id;
}

XLine* XChunkToXPlaneConverter::FindOrAddLine(int64_t line_id) {
  auto [it, inserted] = lines_by_id_.try_emplace(line_id, nullptr);
  if (!inserted) {
    return it->second;
  }
  XLine* line = out_plane_->add_lines();
  line->set_id(line_id);
  it->second = line;
  return line;
}

void XChunkToXPlaneConverter::AppendChunkInternal(const XChunk& chunk) {
  AppendXChunkToXPlaneImpl(
      chunk, &current_plane_, out_plane_,
      [this](int64_t line_id) { return FindOrAddLine(line_id); },
      [this](absl::string_view stat_name) {
        return GetOrCreateSpecialStatId(stat_name);
      },
      [this](int64_t id, absl::string_view name) {
        RegisterStatMetadata(id, name);
      },
      [this](int64_t id) { RegisterEventMetadataId(id); });
}

void XChunkToXPlaneConverter::Consume(const XChunk& chunk) {
  if (chunk.has_plane()) {
    current_plane_ = chunk.plane();
    for (int64_t id : current_plane_.event_metadata_ids()) {
      RegisterEventMetadataId(id);
    }
    const size_t num_stat_metas =
        std::min(current_plane_.stat_metadata_ids().size(),
                 current_plane_.stat_names_size());
    for (size_t i = 0; i < num_stat_metas; ++i) {
      RegisterStatMetadata(current_plane_.stat_metadata_ids(i),
                           current_plane_.stat_names(i));
    }
  } else if (chunk.has_events()) {
    const XChunkEvents& evs = chunk.events();
    for (size_t i = 0; i < evs.new_event_names_vec.size(); ++i) {
      int64_t next_id = 0;
      if (i < evs.new_event_metadata_ids_vec.size()) {
        next_id = evs.new_event_metadata_ids_vec[i];
      } else {
        if (out_plane_ != nullptr) {
          while (out_plane_->event_metadata().find(next_event_id_) !=
                     out_plane_->event_metadata().end() &&
                 next_event_id_ < std::numeric_limits<int64_t>::max()) {
            ++next_event_id_;
          }
        }
        next_id = next_event_id_;
      }
      RegisterEventMetadataId(next_id);
      const absl::string_view disp =
          i < evs.new_event_display_names_vec.size()
              ? absl::string_view(evs.new_event_display_names_vec[i])
              : absl::string_view();
      current_plane_.AddEventMetadata(next_id, evs.new_event_names_vec[i],
                                      disp);
      if (out_plane_ != nullptr) {
        XEventMetadata& m = (*out_plane_->mutable_event_metadata())[next_id];
        m.set_id(next_id);
        if (m.name().empty() && !evs.new_event_names_vec[i].empty()) {
          m.set_name(evs.new_event_names_vec[i]);
        }
        if (m.display_name().empty() && !disp.empty()) {
          m.set_display_name(std::string(disp));
        }
      }
    }
    for (size_t i = 0; i < evs.new_stat_names_vec.size(); ++i) {
      int64_t stat_id = 0;
      if (i < evs.new_stat_metadata_ids_vec.size()) {
        stat_id = evs.new_stat_metadata_ids_vec[i];
      } else {
        if (out_plane_ != nullptr) {
          while (out_plane_->stat_metadata().find(next_stat_id_) !=
                     out_plane_->stat_metadata().end() &&
                 next_stat_id_ < std::numeric_limits<int64_t>::max()) {
            ++next_stat_id_;
          }
        }
        stat_id = next_stat_id_;
      }
      RegisterStatMetadata(stat_id, evs.new_stat_names_vec[i]);
      current_plane_.AddStatMetadata(stat_id, evs.new_stat_names_vec[i]);
      if (out_plane_ != nullptr) {
        XStatMetadata& m = (*out_plane_->mutable_stat_metadata())[stat_id];
        m.set_id(stat_id);
        if (m.name().empty() && !evs.new_stat_names_vec[i].empty()) {
          m.set_name(evs.new_stat_names_vec[i]);
        }
      }
    }
  }
  AppendChunkInternal(chunk);
}

void XChunkToXPlaneConverter::Consume(XChunk&& chunk) {
  if (chunk.has_plane()) {
    AppendChunkInternal(chunk);
    current_plane_ = std::move(*chunk.mutable_plane());
    for (int64_t id : current_plane_.event_metadata_ids()) {
      RegisterEventMetadataId(id);
    }
    const size_t num_stat_metas =
        std::min(current_plane_.stat_metadata_ids().size(),
                 current_plane_.stat_names_size());
    for (size_t i = 0; i < num_stat_metas; ++i) {
      RegisterStatMetadata(current_plane_.stat_metadata_ids(i),
                           current_plane_.stat_names(i));
    }
    return;
  }
  Consume(static_cast<const XChunk&>(chunk));
}

}  // namespace tsl::profiler::xchunk
