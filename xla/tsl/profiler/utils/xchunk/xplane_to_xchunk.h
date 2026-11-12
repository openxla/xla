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

#ifndef XLA_TSL_PROFILER_UTILS_XCHUNK_XPLANE_TO_XCHUNK_H_
#define XLA_TSL_PROFILER_UTILS_XCHUNK_XPLANE_TO_XCHUNK_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xla/tsl/profiler/utils/xchunk/xchunk.h"

namespace tsl::profiler::xchunk {

struct XChunkOptions {
  // Maximum number of events per XChunkEvents message.
  size_t max_events_per_chunk = 32768;
  // Whether to emit generic XStat columns. The trace viewer first load does
  // not need them (only the special stats: flow, group id, async, step name,
  // and counter values, which are always emitted).
  bool include_stats = false;
};

// Receives the XChunk messages of a stream, in order.
using XChunkSink = absl::FunctionRef<void(XChunk&&)>;

struct ChunkXPlaneStats {
  size_t num_lines = 0;
  size_t num_events = 0;  // non internal events emitted
  size_t num_chunks = 0;
};

// Converts one XPlane into an XChunkPlane message followed by the XChunkEvents
// of every non empty line, in line order. Within a line, events are emitted in
// (offset ascending, duration descending) order, which is the order the trace
// viewer consumes them in. Internal events (tsl::profiler::IsInternalEvent)
// are dropped.
// `plane_index` must be unique within the stream. `process_id` /
// `process_name` are the trace viewer process ("pid") the plane maps to (see
// TraceViewerProcessId).
ChunkXPlaneStats ChunkXPlane(const tensorflow::profiler::XPlane& plane,
                             uint32_t plane_index, uint32_t process_id,
                             absl::string_view process_name,
                             const XChunkOptions& options, XChunkSink sink);

// Stateful chunker for streaming an XPlane whose XLines (and XLine event
// slices) arrive incrementally across multiple records without merging them
// into a monolithic XSpace or XPlane first.
class StreamingPlaneChunker {
 public:
  // Builds the metadata tables from `metadata_plane` (which contains
  // event_metadata and stat_metadata) and immediately emits the XChunkPlane
  // header to `sink`. If `host_id >= 0`, host correlation, producer, and
  // consumer stats are converted into XFlow stats on the fly.
  StreamingPlaneChunker(const tensorflow::profiler::XPlane& metadata_plane,
                        uint32_t plane_index, uint32_t process_id,
                        absl::string_view process_name, int32_t host_id,
                        const XChunkOptions& options, XChunkSink sink);
  ~StreamingPlaneChunker();

  StreamingPlaneChunker(StreamingPlaneChunker&&) noexcept;
  StreamingPlaneChunker& operator=(StreamingPlaneChunker&&) noexcept;

  // Registers or updates the line header (id, display_id, name, display_name,
  // timestamp_ns normalized against `start_time_ns`, duration_ps) for a line
  // that may be split across multiple records.
  void RegisterLineHeader(const tensorflow::profiler::XLine& line,
                          uint64_t start_time_ns);

  // Chunks and emits all non internal events in `line` (using any previously
  // registered line header for `line.id()`), normalizing line timestamps by
  // `start_time_ns`. Set `is_last_slice` to false when `line` is a partial
  // slice of a line split across multiple records or batches and more slices
  // for `line.id()` will follow; only the final chunk of a slice with
  // `is_last_slice == true` sets `XChunkEvents::last_chunk() == true`.
  void ChunkLine(const tensorflow::profiler::XLine& line,
                 uint64_t start_time_ns, XChunkSink sink,
                 bool is_last_slice = true);

  const ChunkXPlaneStats& stats() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Maps an XPlane to the trace viewer process id: device planes get
// `host_id * kMaxDevicesPerHost + kFirstDeviceId + plane.id()`, the main host
// threads plane gets `host_id * kMaxDevicesPerHost + kHostThreadsDeviceId`,
// `/host:CPU` prefixed subprocess planes get consecutive host thread PIDs
// starting after `kHostThreadsDeviceId`, `/host:metadata` gets
// `host_id * kMaxDevicesPerHost`, `Host CPUs` gets
// `host_id * kMaxDevicesPerHost + (kMaxDevicesPerHost - 1)`, and custom planes
// get consecutive custom plane PIDs. `next_custom_plane_index` is incremented
// for every custom plane, and `next_host_threads_plane_index` (when non null)
// is incremented for every `/host:CPU` prefixed subprocess plane.
uint32_t TraceViewerProcessId(const tensorflow::profiler::XPlane& plane,
                              int host_id, int* next_custom_plane_index,
                              int* next_host_threads_plane_index = nullptr);

// Converts an XSpace (e.g. an `.xplane.pb` profile or an in memory XSpace)
// into a complete XChunk stream: header, planes with their events, and end
// marker. Skips empty planes and the metadata plane (`/host:metadata`).
void StreamXSpaceToXChunks(const tensorflow::profiler::XSpace& space,
                           absl::string_view hostname, int host_id,
                           const XChunkOptions& options, XChunkSink sink);

// Appends a single XChunk (either XChunkPlane or XChunkEvents) into
// `out_plane`. When `chunk.has_events()`, `current_plane` (if non null)
// provides the dense `event_metadata_ids` and `stat_metadata_ids` tables for
// translating dense metadata indices back to XPlane metadata IDs.
void AppendXChunkToXPlane(
    const XChunk& chunk, const XChunkPlane* absl_nullable current_plane,
    tensorflow::profiler::XPlane* absl_nullable out_plane);

// Stateful helper that accumulates a stream of XChunks for a single plane into
// an `XPlane` proto.
class XChunkToXPlaneConverter {
 public:
  explicit XChunkToXPlaneConverter(
      tensorflow::profiler::XPlane* absl_nullable out_plane);

  void Consume(const XChunk& chunk);
  void Consume(XChunk&& chunk);

  const XChunkPlane& current_plane() const { return current_plane_; }

 private:
  int64_t GetOrCreateSpecialStatId(absl::string_view stat_name);
  tensorflow::profiler::XLine* FindOrAddLine(int64_t line_id);
  void RegisterStatMetadata(int64_t id, absl::string_view name);
  void RegisterEventMetadataId(int64_t id);
  void AppendChunkInternal(const XChunk& chunk);

  tensorflow::profiler::XPlane* out_plane_;
  XChunkPlane current_plane_;
  absl::flat_hash_map<int64_t, tensorflow::profiler::XLine*> lines_by_id_;
  absl::flat_hash_map<std::string, int64_t> stat_ids_by_name_;
  int64_t next_stat_id_ = 1;
  int64_t next_event_id_ = 1;
};

}  // namespace tsl::profiler::xchunk

#endif  // XLA_TSL_PROFILER_UTILS_XCHUNK_XPLANE_TO_XCHUNK_H_
