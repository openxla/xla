/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/cpu/collectives/in_process_communicator.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/collectives/cpu_collective_reduction.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/debug_options_flags.h"
#include "xla/future.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_rendezvous.h"
#include "xla/service/rendezvous.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/lib/math/math_util.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

absl::Duration WarnStuckTimeout() {
  static const absl::Duration warn_stuck_timeout = []() {
    int64_t timeout = xla::GetDebugOptionsFromFlags()
                          .xla_cpu_collective_call_warn_stuck_seconds();
    return timeout >= 0 ? absl::Seconds(timeout) : absl::InfiniteDuration();
  }();
  return warn_stuck_timeout;
}

absl::Duration TerminateTimeout() {
  static const absl::Duration terminate_timeout = []() {
    int64_t timeout = xla::GetDebugOptionsFromFlags()
                          .xla_cpu_collective_call_terminate_timeout_seconds();
    return timeout >= 0 ? absl::Seconds(timeout) : absl::InfiniteDuration();
  }();
  return terminate_timeout;
}

class PhaseBarrier {
 public:
  explicit PhaseBarrier(size_t count) : counter_(static_cast<int64_t>(count)) {}

  void Wait() {
    int64_t cnt = counter_.fetch_sub(1, std::memory_order_acq_rel);
    DCHECK_GE(cnt, 1) << "Counter should never drop below 0";
    if (cnt == 1) {
      notification_.Notify();
    }
    notification_.WaitForNotification();
  }

  absl::Mutex& write_mutex() { return write_mu_; }

 private:
  std::atomic<int64_t> counter_;
  absl::Notification notification_;
  absl::Mutex write_mu_;
};

template <typename Participant>
class OpParticipants {
 public:
  explicit OpParticipants(std::vector<Participant> participants)
      : stage_barrier_(participants.size()),
        done_barrier_(participants.size()),
        participants_(std::move(participants)) {}

  template <typename Fn, typename... Args>
  absl::Status Invoke(Fn fn, Args... args) {
    absl::Status status =
        fn(participants_, stage_barrier_, std::forward<Args>(args)...);
    done_barrier_.Wait();
    return status;
  }

 private:
  PhaseBarrier stage_barrier_;
  PhaseBarrier done_barrier_;
  std::vector<Participant> participants_;
};

template <typename Participant>
bool ByRank(const Participant* a, const Participant* b) {
  return a->rank < b->rank;
}

template <typename Participant>
std::vector<Participant> CollectParticipants(
    absl::Span<Participant*> participants) {
  absl::c_sort(participants, ByRank<Participant>);

  std::vector<Participant> ret;
  ret.reserve(participants.size());
  for (const Participant* participant : participants) {
    ret.push_back(*participant);
  }
  return ret;
}

absl::Status ValidateReductionKind(PrimitiveType dtype,
                                   ReductionKind reduction_kind) {
  if (primitive_util::IsComplexType(dtype) &&
      (reduction_kind == ReductionKind::MIN ||
       reduction_kind == ReductionKind::MAX)) {
    return absl::InvalidArgumentError(
        "MIN/MAX reductions are not supported for complex types");
  }
  return absl::OkStatus();
}

void* OffsetBytes(se::DeviceAddressBase mem, size_t offset) {
  return offset == 0 ? mem.opaque()
                     : static_cast<std::byte*>(mem.opaque()) + offset;
}

template <ReductionKind>
constexpr bool always_false_v = false;

template <ReductionKind kKind, typename T>
void ReduceHelper(absl::Span<T> acc, absl::Span<T const* const> inputs) {
  if constexpr (kKind == ReductionKind::SUM) {
    for (const T* input : inputs) {
      for (size_t i = 0; i < acc.size(); ++i) {
        acc[i] += input[i];
      }
    }
  } else if constexpr (kKind == ReductionKind::PRODUCT) {
    for (const T* input : inputs) {
      for (size_t i = 0; i < acc.size(); ++i) {
        acc[i] *= input[i];
      }
    }
  } else if constexpr (kKind == ReductionKind::MIN) {
    if (inputs.empty()) {
      if constexpr (is_specialized_floating_point_v<T> &&
                    std::numeric_limits<T>::has_signaling_NaN) {
        for (T& v : acc) {
          v = IeeeMinMax<T, /*kIsMax=*/false>(v, v);
        }
      }
      return;
    }
    for (const T* input : inputs) {
      for (size_t i = 0; i < acc.size(); ++i) {
        acc[i] = IeeeMinMax<T, /*kIsMax=*/false>(acc[i], input[i]);
      }
    }
  } else if constexpr (kKind == ReductionKind::MAX) {
    if (inputs.empty()) {
      if constexpr (is_specialized_floating_point_v<T> &&
                    std::numeric_limits<T>::has_signaling_NaN) {
        for (T& v : acc) {
          v = IeeeMinMax<T, /*kIsMax=*/true>(v, v);
        }
      }
      return;
    }
    for (const T* input : inputs) {
      for (size_t i = 0; i < acc.size(); ++i) {
        acc[i] = IeeeMinMax<T, /*kIsMax=*/true>(acc[i], input[i]);
      }
    }
  } else {
    static_assert(always_false_v<kKind>, "Unsupported reduction kind");
  }
}

template <PrimitiveType PT>
absl::Status ReduceChunk(ReductionKind reduction_kind,
                         absl::Span<const void* const> inputs, void* output,
                         size_t num_elems) {
  using T = primitive_util::NativeTypeOf<PT>;
  TF_RET_CHECK(!inputs.empty());
  if (num_elems == 0) {
    return absl::OkStatus();
  }
  CopyBytesIfNeeded(output, inputs[0], num_elems * sizeof(T));
  absl::Span<T> out_chunk(static_cast<T*>(output), num_elems);
  absl::Span<T const* const> rest(
      reinterpret_cast<T const* const*>(inputs.data() + 1), inputs.size() - 1);
  switch (reduction_kind) {
    case ReductionKind::SUM:
      ReduceHelper<ReductionKind::SUM, T>(out_chunk, rest);
      return absl::OkStatus();
    case ReductionKind::PRODUCT:
      ReduceHelper<ReductionKind::PRODUCT, T>(out_chunk, rest);
      return absl::OkStatus();
    case ReductionKind::MIN:
      if constexpr (!is_complex_v<T>) {
        ReduceHelper<ReductionKind::MIN, T>(out_chunk, rest);
        return absl::OkStatus();
      }
      return absl::InvalidArgumentError(
          "MIN reductions are not supported for complex types");
    case ReductionKind::MAX:
      if constexpr (!is_complex_v<T>) {
        ReduceHelper<ReductionKind::MAX, T>(out_chunk, rest);
        return absl::OkStatus();
      }
      return absl::InvalidArgumentError(
          "MAX reductions are not supported for complex types");
  }
  return absl::InvalidArgumentError("Unsupported reduction kind");
}

template <PrimitiveType PT, typename WriteFn>
absl::Status ReduceChunkStaged(ReductionKind reduction_kind,
                               absl::Span<const void* const> inputs,
                               size_t num_elems, PhaseBarrier& stage_barrier,
                               WriteFn write_fn) {
  using T = primitive_util::NativeTypeOf<PT>;
  std::unique_ptr<T[]> scratch;
  absl::Status status = absl::OkStatus();
  if (num_elems > 0) {
    scratch = std::make_unique<T[]>(num_elems);
    status = ReduceChunk<PT>(reduction_kind, inputs, scratch.get(), num_elems);
  }
  stage_barrier.Wait();
  if (status.ok() && num_elems > 0) {
    write_fn(scratch.get(), num_elems * sizeof(T));
  }
  return status;
}

//===----------------------------------------------------------------------===//
// AllReduce
//===----------------------------------------------------------------------===//

struct AllReduceParticipant {
  size_t rank;
  se::DeviceAddressBase src;
  se::DeviceAddressBase dest;
};

absl::Status AllReduceOp(absl::Span<const AllReduceParticipant> participants,
                         PhaseBarrier& stage_barrier, size_t rank,
                         PrimitiveType primitive_type, size_t count,
                         size_t total_bytes, ReductionKind reduction_kind) {
  if (count == 0 || participants.empty()) {
    return absl::OkStatus();
  }
  const size_t n = participants.size();
  const size_t byte_width = primitive_util::ByteWidth(primitive_type);
  bool needs_staging = false;
  bool dests_overlap = false;
  for (size_t a = 0; a < n; ++a) {
    for (size_t b = 0; b < n; ++b) {
      if (a == b &&
          participants[a].dest.opaque() == participants[b].src.opaque()) {
        continue;
      }
      if (ByteRangesOverlap(participants[a].dest.opaque(), total_bytes,
                            participants[b].src.opaque(), total_bytes)) {
        needs_staging = true;
      }
    }
    for (size_t b = a + 1; b < n; ++b) {
      if (ByteRangesOverlap(participants[a].dest.opaque(), total_bytes,
                            participants[b].dest.opaque(), total_bytes)) {
        dests_overlap = true;
        needs_staging = true;
      }
    }
  }

  const size_t chunk_size = tsl::MathUtil::CeilOfRatio(count, n);
  const size_t start_elem = std::min(chunk_size * rank, count);
  const size_t end_elem =
      rank + 1 < n ? std::min(chunk_size * (rank + 1), count) : count;
  const size_t chunk_count = end_elem - start_elem;
  const size_t chunk_offset = start_elem * byte_width;

  std::vector<const void*> inputs(n, nullptr);
  if (chunk_count > 0) {
    for (size_t j = 0; j < n; ++j) {
      inputs[j] = OffsetBytes(participants[j].src, chunk_offset);
    }
  }

  if (needs_staging) {
    return primitive_util::ArrayTypeSwitch(
        [&](const auto type_tag) {
          return ReduceChunkStaged<type_tag>(
              reduction_kind, inputs, chunk_count, stage_barrier,
              [&](const void* data, size_t chunk_bytes) {
                auto scatter = [&]() {
                  for (size_t i = 0; i < n; ++i) {
                    CopyBytesIfNeeded(
                        OffsetBytes(participants[i].dest, chunk_offset), data,
                        chunk_bytes);
                  }
                };
                if (dests_overlap) {
                  absl::MutexLock lock(stage_barrier.write_mutex());
                  scatter();
                } else {
                  scatter();
                }
              });
        },
        primitive_type);
  }

  if (chunk_count == 0) {
    return absl::OkStatus();
  }
  void* output = OffsetBytes(participants[0].dest, chunk_offset);
  ABSL_RETURN_IF_ERROR(primitive_util::ArrayTypeSwitch(
      [&](const auto type_tag) {
        return ReduceChunk<type_tag>(reduction_kind, inputs, output,
                                     chunk_count);
      },
      primitive_type));

  const size_t chunk_bytes = chunk_count * byte_width;
  for (size_t i = 1; i < n; ++i) {
    CopyBytesIfNeeded(OffsetBytes(participants[i].dest, chunk_offset), output,
                      chunk_bytes);
  }
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// ReduceScatter
//===----------------------------------------------------------------------===//

struct ReduceScatterParticipant {
  size_t rank;
  se::DeviceAddressBase src;
  se::DeviceAddressBase dest;
};

absl::Status ReduceScatterOp(
    absl::Span<const ReduceScatterParticipant> participants,
    PhaseBarrier& stage_barrier, size_t rank, PrimitiveType primitive_type,
    size_t count, size_t num_bytes, size_t total_src_bytes,
    ReductionKind reduction_kind) {
  if (count == 0 || participants.empty()) {
    return absl::OkStatus();
  }
  const size_t n = participants.size();
  const size_t offset = rank * num_bytes;

  bool needs_staging = false;
  bool dests_overlap = false;
  for (size_t a = 0; a < n; ++a) {
    for (size_t b = 0; b < n; ++b) {
      // Rank 0's output aliasing its own shard is seeded in place without
      // overwriting any unconsumed input; higher ranks stage to preserve
      // rank-0..n-1 reduction order.
      if (a == 0 && b == 0 &&
          participants[0].dest.opaque() == participants[0].src.opaque()) {
        continue;
      }
      if (ByteRangesOverlap(participants[a].dest.opaque(), num_bytes,
                            participants[b].src.opaque(), total_src_bytes)) {
        needs_staging = true;
      }
    }
    for (size_t b = a + 1; b < n; ++b) {
      if (ByteRangesOverlap(participants[a].dest.opaque(), num_bytes,
                            participants[b].dest.opaque(), num_bytes)) {
        dests_overlap = true;
        needs_staging = true;
      }
    }
  }

  std::vector<const void*> inputs(n);
  for (size_t j = 0; j < n; ++j) {
    inputs[j] = OffsetBytes(participants[j].src, offset);
  }
  void* output = participants[rank].dest.opaque();
  if (needs_staging) {
    return primitive_util::ArrayTypeSwitch(
        [&](const auto type_tag) {
          return ReduceChunkStaged<type_tag>(
              reduction_kind, inputs, count, stage_barrier,
              [&](const void* data, size_t bytes) {
                if (dests_overlap) {
                  absl::MutexLock lock(stage_barrier.write_mutex());
                  CopyBytesIfNeeded(output, data, bytes);
                } else {
                  CopyBytesIfNeeded(output, data, bytes);
                }
              });
        },
        primitive_type);
  }

  return primitive_util::ArrayTypeSwitch(
      [&](const auto type_tag) {
        return ReduceChunk<type_tag>(reduction_kind, inputs, output, count);
      },
      primitive_type);
}

//===----------------------------------------------------------------------===//
// AllGather
//===----------------------------------------------------------------------===//

struct AllGatherParticipant {
  size_t rank;
  se::DeviceAddressBase src;
  se::DeviceAddressBase dest;
};

absl::Status AllGatherOp(absl::Span<const AllGatherParticipant> participants,
                         PhaseBarrier& stage_barrier, size_t rank,
                         size_t num_bytes, size_t total_dest_bytes) {
  if (num_bytes == 0 || participants.empty()) {
    return absl::OkStatus();
  }
  const size_t n = participants.size();
  bool needs_staging = false;
  bool dests_overlap = false;
  for (size_t a = 0; a < n; ++a) {
    for (size_t b = 0; b < n; ++b) {
      if (a == b && participants[b].src.opaque() ==
                        OffsetBytes(participants[a].dest, a * num_bytes)) {
        continue;
      }
      if (ByteRangesOverlap(participants[a].dest.opaque(), total_dest_bytes,
                            participants[b].src.opaque(), num_bytes)) {
        needs_staging = true;
      }
    }
    for (size_t b = a + 1; b < n; ++b) {
      if (ByteRangesOverlap(participants[a].dest.opaque(), total_dest_bytes,
                            participants[b].dest.opaque(), total_dest_bytes)) {
        dests_overlap = true;
        needs_staging = true;
      }
    }
  }

  se::DeviceAddressBase dest = participants[rank].dest;
  if (needs_staging) {
    std::vector<std::byte> scratch(total_dest_bytes);
    for (size_t i = 0; i < n; ++i) {
      std::memcpy(scratch.data() + i * num_bytes, participants[i].src.opaque(),
                  num_bytes);
    }
    stage_barrier.Wait();
    if (dests_overlap) {
      absl::MutexLock lock(stage_barrier.write_mutex());
      CopyBytesIfNeeded(dest.opaque(), scratch.data(), total_dest_bytes);
    } else {
      CopyBytesIfNeeded(dest.opaque(), scratch.data(), total_dest_bytes);
    }
    return absl::OkStatus();
  }

  for (size_t i = 0; i < n; ++i) {
    CopyBytesIfNeeded(OffsetBytes(dest, i * num_bytes),
                      participants[i].src.opaque(), num_bytes);
  }
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// AllToAll
//===----------------------------------------------------------------------===//

struct AllToAllParticipant {
  size_t rank;

  std::vector<se::DeviceAddressBase> src;
  std::vector<se::DeviceAddressBase> dest;
};

absl::Status AllToAllOp(absl::Span<const AllToAllParticipant> participants,
                        PhaseBarrier& stage_barrier, size_t rank,
                        size_t num_bytes, size_t total_bytes) {
  const size_t n = participants.size();
  for (const AllToAllParticipant& p : participants) {
    if (p.src.size() != n || p.dest.size() != n) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "AllToAll expected %d send and recv buffers per participant, got %d "
          "and %d",
          n, p.src.size(), p.dest.size()));
    }
  }
  if (num_bytes == 0 || n == 0) {
    return absl::OkStatus();
  }

  bool needs_staging = false;
  bool dests_overlap = false;
  for (size_t a = 0; a < n; ++a) {
    for (size_t k = 0; k < n; ++k) {
      const void* da = participants[a].dest[k].opaque();
      for (size_t b = 0; b < n; ++b) {
        for (size_t m = 0; m < n; ++m) {
          const void* sb = participants[b].src[m].opaque();
          if (a == b && k == a && m == a && da == sb) {
            continue;
          }
          if (ByteRangesOverlap(da, num_bytes, sb, num_bytes)) {
            needs_staging = true;
          }
        }
      }
      for (size_t b = a; b < n; ++b) {
        for (size_t m = (b == a ? k + 1 : 0); m < n; ++m) {
          if (ByteRangesOverlap(da, num_bytes, participants[b].dest[m].opaque(),
                                num_bytes)) {
            dests_overlap = true;
            needs_staging = true;
          }
        }
      }
    }
  }

  if (needs_staging) {
    std::vector<std::byte> scratch(total_bytes);
    for (size_t i = 0; i < n; ++i) {
      std::memcpy(scratch.data() + i * num_bytes,
                  participants[i].src[rank].opaque(), num_bytes);
    }
    stage_barrier.Wait();
    auto write_dests = [&]() {
      for (size_t i = 0; i < n; ++i) {
        CopyBytesIfNeeded(participants[rank].dest[i].opaque(),
                          scratch.data() + i * num_bytes, num_bytes);
      }
    };
    if (dests_overlap) {
      absl::MutexLock lock(stage_barrier.write_mutex());
      write_dests();
    } else {
      write_dests();
    }
    return absl::OkStatus();
  }

  for (size_t i = 0; i < n; ++i) {
    CopyBytesIfNeeded(participants[rank].dest[i].opaque(),
                      participants[i].src[rank].opaque(), num_bytes);
  }
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// CollectivePermute
//===----------------------------------------------------------------------===//

struct CollectivePermuteParticipant {
  size_t rank;
  std::optional<RankId> src_rank;

  se::DeviceAddressBase src;
  se::DeviceAddressBase dest;
};

absl::Status CollectivePermuteOp(
    absl::Span<const CollectivePermuteParticipant> participants,
    PhaseBarrier& stage_barrier, size_t rank, size_t num_bytes) {
  const size_t n = participants.size();
  std::vector<size_t> reader_count(n, 0);
  for (const CollectivePermuteParticipant& p : participants) {
    if (p.src_rank.has_value()) {
      const int64_t sr = p.src_rank->value();
      if (sr < 0 || static_cast<size_t>(sr) >= n) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Invalid CollectivePermute source rank %d for %d participants", sr,
            n));
      }
      ++reader_count[static_cast<size_t>(sr)];
    }
  }
  if (num_bytes == 0 || n == 0) {
    return absl::OkStatus();
  }

  bool needs_staging = false;
  bool dests_overlap = false;
  for (size_t a = 0; a < n; ++a) {
    const void* da = participants[a].dest.opaque();
    for (size_t b = 0; b < n; ++b) {
      if (reader_count[b] == 0) {
        continue;
      }
      const void* sb = participants[b].src.opaque();
      if (a == b && reader_count[b] == 1 &&
          participants[b].src_rank == RankId(static_cast<int64_t>(b)) &&
          da == sb) {
        continue;
      }
      if (ByteRangesOverlap(da, num_bytes, sb, num_bytes)) {
        needs_staging = true;
      }
    }
    for (size_t b = a + 1; b < n; ++b) {
      if (ByteRangesOverlap(da, num_bytes, participants[b].dest.opaque(),
                            num_bytes)) {
        dests_overlap = true;
        needs_staging = true;
      }
    }
  }

  const CollectivePermuteParticipant& participant = participants[rank];
  void* dest = participant.dest.opaque();
  if (needs_staging) {
    std::vector<std::byte> scratch;
    if (participant.src_rank.has_value()) {
      scratch.resize(num_bytes);
      const size_t sr = static_cast<size_t>(participant.src_rank->value());
      std::memcpy(scratch.data(), participants[sr].src.opaque(), num_bytes);
    }
    stage_barrier.Wait();
    auto finish = [&]() {
      if (participant.src_rank.has_value()) {
        CopyBytesIfNeeded(dest, scratch.data(), num_bytes);
      } else {
        std::memset(dest, 0, num_bytes);
      }
    };
    if (dests_overlap) {
      absl::MutexLock lock(stage_barrier.write_mutex());
      finish();
    } else {
      finish();
    }
    return absl::OkStatus();
  }

  if (participant.src_rank.has_value()) {
    const size_t sr = static_cast<size_t>(participant.src_rank->value());
    CopyBytesIfNeeded(dest, participants[sr].src.opaque(), num_bytes);
  } else {
    std::memset(dest, 0, num_bytes);
  }
  return absl::OkStatus();
}

}  // namespace

//===----------------------------------------------------------------------===//

InProcessCommunicator::InProcessCommunicator(size_t rank, size_t num_ranks)
    : rank_(rank), num_ranks_(num_ranks) {}

Future<> InProcessCommunicator::AllReduce(se::DeviceAddressBase send_buffer,
                                          se::DeviceAddressBase recv_buffer,
                                          PrimitiveType dtype, size_t count,
                                          ReductionKind reduction_kind,
                                          const Executor& executor) {
  ABSL_RETURN_IF_ERROR(ValidateByteArrayDataType(dtype));
  ABSL_RETURN_IF_ERROR(ValidateReductionKind(dtype, reduction_kind));
  ABSL_ASSIGN_OR_RETURN(
      const size_t total_bytes,
      CheckedCollectiveByteSize(count, primitive_util::ByteWidth(dtype)));
  ABSL_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));
  const RendezvousKey& key = cpu_executor->rendezvous_key();

  std::string name = absl::StrCat("all reduce ", key.ToString());
  AllReduceParticipant participant{rank_, send_buffer, recv_buffer};

  ABSL_ASSIGN_OR_RETURN(auto op,
                        Rendezvous<OpParticipants<AllReduceParticipant>>(
                            name, key, participant, key.num_local_participants,
                            CollectParticipants<AllReduceParticipant>,
                            WarnStuckTimeout(), TerminateTimeout()));

  ABSL_RETURN_IF_ERROR(op->Invoke(AllReduceOp, rank_, dtype, count, total_bytes,
                                  reduction_kind));
  return Future<>(absl::OkStatus());
}

Future<> InProcessCommunicator::ReduceScatter(se::DeviceAddressBase send_buffer,
                                              se::DeviceAddressBase recv_buffer,
                                              PrimitiveType dtype, size_t count,
                                              ReductionKind reduction_kind,
                                              const Executor& executor) {
  ABSL_RETURN_IF_ERROR(ValidateByteArrayDataType(dtype));
  ABSL_RETURN_IF_ERROR(ValidateReductionKind(dtype, reduction_kind));
  ABSL_ASSIGN_OR_RETURN(
      const size_t num_bytes,
      CheckedCollectiveByteSize(count, primitive_util::ByteWidth(dtype)));
  ABSL_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));
  const RendezvousKey& key = cpu_executor->rendezvous_key();
  ABSL_ASSIGN_OR_RETURN(
      const size_t total_src_bytes,
      CheckedCollectiveByteSize(num_bytes, key.num_local_participants));

  std::string name = absl::StrCat("reduce scatter ", key.ToString());
  ReduceScatterParticipant participant{rank_, send_buffer, recv_buffer};

  ABSL_ASSIGN_OR_RETURN(auto op,
                        Rendezvous<OpParticipants<ReduceScatterParticipant>>(
                            name, key, participant, key.num_local_participants,
                            CollectParticipants<ReduceScatterParticipant>,
                            WarnStuckTimeout(), TerminateTimeout()));

  ABSL_RETURN_IF_ERROR(op->Invoke(ReduceScatterOp, rank_, dtype, count,
                                  num_bytes, total_src_bytes, reduction_kind));
  return Future<>(absl::OkStatus());
}

Future<> InProcessCommunicator::CollectivePermute(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  ABSL_RETURN_IF_ERROR(ValidateByteArrayDataType(dtype));
  ABSL_ASSIGN_OR_RETURN(
      const size_t num_bytes,
      CheckedCollectiveByteSize(count, primitive_util::ByteWidth(dtype)));
  ABSL_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));
  const RendezvousKey& key = cpu_executor->rendezvous_key();

  std::string name = absl::StrCat("collective permute ", key.ToString());
  CollectivePermuteParticipant participant{rank_, source_rank, send_buffer,
                                           recv_buffer};

  ABSL_ASSIGN_OR_RETURN(
      auto op, Rendezvous<OpParticipants<CollectivePermuteParticipant>>(
                   name, key, participant, key.num_local_participants,
                   CollectParticipants<CollectivePermuteParticipant>,
                   WarnStuckTimeout(), TerminateTimeout()));

  ABSL_RETURN_IF_ERROR(op->Invoke(CollectivePermuteOp, rank_, num_bytes));
  return Future<>(absl::OkStatus());
}

Future<> InProcessCommunicator::AllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  ABSL_RETURN_IF_ERROR(ValidateByteArrayDataType(dtype));
  ABSL_ASSIGN_OR_RETURN(
      const size_t num_bytes,
      CheckedCollectiveByteSize(count, primitive_util::ByteWidth(dtype)));
  ABSL_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));
  const RendezvousKey& key = cpu_executor->rendezvous_key();
  ABSL_ASSIGN_OR_RETURN(
      const size_t total_bytes,
      CheckedCollectiveByteSize(num_bytes, key.num_local_participants));

  std::string name = absl::StrCat("all to all ", key.ToString());
  AllToAllParticipant participant{rank_,
                                  {send_buffers.begin(), send_buffers.end()},
                                  {recv_buffers.begin(), recv_buffers.end()}};

  ABSL_ASSIGN_OR_RETURN(
      auto op, Rendezvous<OpParticipants<AllToAllParticipant>>(
                   name, key, participant, key.num_local_participants,
                   CollectParticipants<AllToAllParticipant>, WarnStuckTimeout(),
                   TerminateTimeout()));

  ABSL_RETURN_IF_ERROR(op->Invoke(AllToAllOp, rank_, num_bytes, total_bytes));
  return Future<>(absl::OkStatus());
}

Future<> InProcessCommunicator::AllGather(se::DeviceAddressBase send_buffer,
                                          se::DeviceAddressBase recv_buffer,
                                          PrimitiveType dtype, size_t count,
                                          const Executor& executor) {
  ABSL_RETURN_IF_ERROR(ValidateByteArrayDataType(dtype));
  ABSL_ASSIGN_OR_RETURN(
      const size_t num_bytes,
      CheckedCollectiveByteSize(count, primitive_util::ByteWidth(dtype)));
  ABSL_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));
  const RendezvousKey& key = cpu_executor->rendezvous_key();
  ABSL_ASSIGN_OR_RETURN(
      const size_t total_dest_bytes,
      CheckedCollectiveByteSize(num_bytes, key.num_local_participants));

  std::string name = absl::StrCat("all gather ", key.ToString());
  AllGatherParticipant participant{rank_, send_buffer, recv_buffer};

  ABSL_ASSIGN_OR_RETURN(auto op,
                        Rendezvous<OpParticipants<AllGatherParticipant>>(
                            name, key, participant, key.num_local_participants,
                            CollectParticipants<AllGatherParticipant>,
                            WarnStuckTimeout(), TerminateTimeout()));

  ABSL_RETURN_IF_ERROR(
      op->Invoke(AllGatherOp, rank_, num_bytes, total_dest_bytes));
  return Future<>(absl::OkStatus());
}

}  // namespace xla::cpu
