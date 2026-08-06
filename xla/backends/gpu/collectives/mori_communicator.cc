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

#include "xla/backends/gpu/collectives/mori_communicator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/mori_collectives.h"
#include "xla/backends/gpu/collectives/mori_kernels.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace shmem = ::mori::shmem;
namespace xla::gpu {

using ::mori::collective::CollectivesFacade;

static auto AsRocmStream(se::Stream* stream) {
  return reinterpret_cast<std::intptr_t>(
      stream->platform_specific_handle().stream);
}

static hipStream_t AsHipStream(se::Stream* stream) {
  return reinterpret_cast<hipStream_t>(
      stream->platform_specific_handle().stream);
}

static size_t ToMoriByteCount(PrimitiveType dtype, size_t count) {
  if (primitive_util::IsComplexType(dtype)) {
    count *= 2;
  }
  return count * primitive_util::BitWidth(dtype) / 8;
}

absl::StatusOr<std::unique_ptr<MoriCommunicator>> MoriCommunicator::Create(
    MoriCollectives* coll, std::shared_ptr<CancellationToken> cancel, int rank,
    absl::Span<const int> rank_to_pe) {
  auto comm = absl::WrapUnique(new MoriCommunicator(coll, cancel));

  const int num_ranks = static_cast<int>(rank_to_pe.size());
  if (num_ranks <= 0 || num_ranks > kMaxRanks) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "MoriCommunicator: unsupported participant count %d (max %d)",
        num_ranks, kMaxRanks));
  }
  comm->rank_ = rank;
  comm->num_ranks_ = num_ranks;

  // The per-device CollectivesFacade owns the symmetric-heap staging buffer and
  // the push reduce-scatter group counters. Init records this rank's identity
  // (rank/num_ranks) and, on first call for the device, allocates the ~2GB
  // staging (shared across communicators on the same device). Init is
  // idempotent for the buffers; subsequent communicators just refresh
  // myPe/nPes.
  const size_t buffer_size = 2UL << 30;  // 2GB
  RETURN_IF_ERROR(se::gpu::ToStatus(
      CollectivesFacade::Get().Init(rank, num_ranks, buffer_size),
      "MORI CollectivesFacade::Init failed"));

  VLOG(1) << "Created " << *comm << " with participants: " << num_ranks;
  return comm;
}

MoriCommunicator::~MoriCommunicator() {
  // The staging buffer + counters are owned by the per-device CollectivesFacade
  // singleton (shared across communicators), so they are released once at
  // MoriCollectives teardown, not here.
}

#define CHECK_CANCELLED()                                               \
  if (cancel_->IsCancelled()) {                                         \
    return absl::FailedPreconditionError("MoriCommunicator cancelled"); \
  }

absl::Status MoriCommunicator::Abort() {
  // By setting the cancellation token all pending collectives scheduled on
  // executor_ will cancel. This will allow the aborting lambda below to run.
  cancel_->Cancel();

  VLOG(1) << "Abort MORI communicator: " << ToString();
  if (aborted_) {
    return FailedPrecondition("MoriCommunicator already aborted");
  }
  aborted_ = true;
  // Call rocm_mori_global_exit with a non-zero return code to abort the
  // program. rocm_mori_global_exit(1);
  return absl::OkStatus();
}

absl::Status MoriCommunicator::Barrier(const Communicator::Executor& executor) {
  VLOG(1) << "Barrier: " << ToString();
  CHECK_CANCELLED()
  ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  CollectivesFacade::Get().RunBarrier<>(AsHipStream(stream));
  return absl::OkStatus();
}

absl::StatusOr<size_t> MoriCommunicator::NumRanks() const {
  VLOG(5) << "Get the number of ranks in MORI communicator: " << ToString();
  CHECK_CANCELLED()

  return static_cast<size_t>(num_ranks_);
}

absl::StatusOr<size_t> MoriCommunicator::CurrentRank() {
  VLOG(5) << "Get current rank in MORI communicator: " << ToString();
  CHECK_CANCELLED()

  return static_cast<size_t>(rank_);
}

std::string MoriCommunicator::ToString() const {
  return absl::StrFormat("MoriCommunicator(rank=%d, num_ranks=%d, my_pe=%d)",
                         rank_, num_ranks_, shmem::ShmemMyPe());
}

absl::StatusOr<se::Stream*> MoriCommunicator::ToStream(
    const Executor& executor) {
  if (auto* gpu_executor =
          tsl::down_cast<const GpuCollectives::Executor*>(&executor)) {
    return gpu_executor->stream();
  }
  return InvalidArgument("Communicator executor is not a GPU executor");
}

Future<> MoriCommunicator::AllReduce(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     ReductionKind reduction_kind,
                                     const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, reduction_kind,
                  &executor, this]() -> absl::Status {
    return LaunchAllReduce(send_buffer, recv_buffer, dtype, count,
                           reduction_kind, executor);
  });
}

Future<> MoriCommunicator::Broadcast(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     RankId root, const Executor& executor) {
  return Execute(
      [send_buffer, recv_buffer, dtype, count, root, &executor, this]() {
        return LaunchBroadcast(send_buffer, recv_buffer, dtype, count, root,
                               executor);
      });
}

Future<> MoriCommunicator::ReduceScatter(se::DeviceAddressBase send_buffer,
                                         se::DeviceAddressBase recv_buffer,
                                         PrimitiveType dtype, size_t count,
                                         ReductionKind reduction_kind,
                                         const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, reduction_kind,
                  &executor, this]() {
    return LaunchReduceScatter(send_buffer, recv_buffer, dtype, count,
                               reduction_kind, executor);
  });
}

Future<> MoriCommunicator::AllGather(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, &executor, this]() {
    return LaunchAllGather(send_buffer, recv_buffer, dtype, count, executor);
  });
}

Future<> MoriCommunicator::AllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  return Execute([send_buffers, recv_buffers, dtype, count, &executor, this]() {
    return LaunchAllToAll(send_buffers, recv_buffers, dtype, count, executor);
  });
}

Future<> MoriCommunicator::CollectivePermute(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  std::vector<RankId> owned_target_ranks(target_ranks.begin(),
                                         target_ranks.end());
  return Execute([send_buffer, recv_buffer, dtype, count, source_rank,
                  owned_target_ranks = std::move(owned_target_ranks), &executor,
                  this]() {
    return LaunchCollectivePermute(send_buffer, recv_buffer, dtype, count,
                                   source_rank, owned_target_ranks, executor);
  });
}

Future<> MoriCommunicator::Send(se::DeviceAddressBase recv_buffer,
                                se::DeviceAddressBase send_buffer,
                                PrimitiveType dtype, size_t count, RankId peer,
                                const Executor& executor) {
  return P2P(P2PType::Send, dtype, recv_buffer, send_buffer, count, peer,
             executor);
}

Future<> MoriCommunicator::Recv(se::DeviceAddressBase recv_buffer,
                                se::DeviceAddressBase send_buffer,
                                PrimitiveType dtype, size_t count, RankId peer,
                                const Executor& executor) {
  return P2P(P2PType::Recv, dtype, recv_buffer, send_buffer, count, peer,
             executor);
}

absl::Status MoriCommunicator::LaunchAllGather(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  CHECK_CANCELLED()
  ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  VLOG(3) << "LaunchAllGather: send_buffer=" << send_buffer.opaque()
          << " recv_buffer=" << recv_buffer.opaque() << " count=" << count
          << " dtype=" << primitive_util::LowercasePrimitiveTypeName(dtype)
          << " stream=" << AsRocmStream(stream);
  if (dtype != PrimitiveType::F32) {
    return absl::UnimplementedError(
        "MoriCommunicator::AllGather only supports F32");
  }
  // XLA `count` is the per-rank shard; the facade wants the TOTAL gathered
  // element count N = count * num_ranks (chunk = N / num_ranks).
  CollectivesFacade::Get().RunAllGather<float>(
      static_cast<const float*>(send_buffer.opaque()),
      static_cast<float*>(recv_buffer.opaque()), count * num_ranks_,
      AsHipStream(stream));
  return absl::OkStatus();
}

absl::Status MoriCommunicator::LaunchAllReduce(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  CHECK_CANCELLED()

  ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << absl::StreamFormat(
      "Launch MORI AllReduce operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%v; stream=%p",
      send_buffer.opaque(), recv_buffer.opaque(),
      primitive_util::LowercasePrimitiveTypeName(dtype), count, reduction_kind,
      stream);

  if (dtype != PrimitiveType::F32 || reduction_kind != ReductionKind::SUM) {
    return absl::UnimplementedError(
        "MoriCommunicator::AllReduce only supports F32 + Sum");
  }
  // XLA `count` is the full vector length N; the facade splits it into
  // num_ranks chunks internally (chunk = N / num_ranks).
  CollectivesFacade::Get().RunAllReduce<float, ::SumOp<float>>(
      static_cast<const float*>(send_buffer.opaque()),
      static_cast<float*>(recv_buffer.opaque()), count, AsHipStream(stream));
  return absl::OkStatus();
}

absl::Status MoriCommunicator::LaunchReduceScatter(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind kind,
    const Executor& executor) {
  CHECK_CANCELLED()
  ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << "LaunchReduceScatter: send_buffer=" << send_buffer.opaque()
          << " recv_buffer=" << recv_buffer.opaque() << " count=" << count
          << " dtype=" << primitive_util::LowercasePrimitiveTypeName(dtype)
          << " stream=" << AsRocmStream(stream);
  if (dtype != PrimitiveType::F32 || kind != ReductionKind::SUM) {
    return absl::UnimplementedError(
        "MoriCommunicator::ReduceScatter only supports F32 + Sum");
  }
  // XLA `count` is this rank's per-rank output block; the facade wants the
  // TOTAL input element count N = count * num_ranks (chunk = N / num_ranks).
  CollectivesFacade::Get().RunReduceScatter<float, ::SumOp<float>>(
      static_cast<const float*>(send_buffer.opaque()),
      static_cast<float*>(recv_buffer.opaque()), count * num_ranks_,
      AsHipStream(stream));
  return absl::OkStatus();
}

absl::Status MoriCommunicator::LaunchCollectivePermute(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  CHECK_CANCELLED()
  ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  size_t bytes = ToMoriByteCount(dtype, count);
  (void)bytes;
  auto rank_formatter = [](std::string* out, RankId rank) {
    absl::StrAppendFormat(out, "%d", rank.value());
  };
  VLOG(3) << absl::StreamFormat(
      "[%d] Launch MORI CollectivePermute operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; source_rank=%s; target_[ranks=%s]; count=%d; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      source_rank ? absl::StrCat(source_rank->value()) : "<empty>",
      absl::StrJoin(target_ranks, ", ", rank_formatter), count, stream);

  // CollectivePermute is not wired through the CollectivesFacade yet.
  return absl::UnimplementedError("Not implemented");
}

// Performs point-to-point communication between two ranks using MORI.
// Send: launches a single GPU kernel that copies data to the peer via P2P
//       and sets a completion flag on the peer.
// Recv: launches a single-thread GPU kernel that waits for the flag.
absl::Status MoriCommunicator::P2P(P2PType p2p_type, PrimitiveType dtype,
                                   se::DeviceAddressBase recv_buffer,
                                   se::DeviceAddressBase send_buffer,
                                   size_t count, RankId peer,
                                   const Executor& executor) {
  const char* stype = (p2p_type == P2PType::Send ? " Send" : " Recv");
  VLOG(1) << CurrentRank().value() << stype << " to " << peer.value()
          << " count " << count << " MORI communicator: " << ToString();
  CHECK_CANCELLED()

  void* source_ptr = send_buffer.opaque();
  void* dest_ptr = recv_buffer.opaque();

  ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  auto gpu_stream = AsRocmStream(stream);
  size_t bytes = ToMoriByteCount(dtype, count);
  int res = 0;
  (void)bytes;
  (void)res;
  (void)gpu_stream;
  (void)source_ptr;
  (void)dest_ptr;
  (void)peer;
  (void)stream;
  (void)dtype;
  (void)count;
  (void)p2p_type;
  // if (p2p_type == P2PType::Send) {
  //   res = xla_mori::Send(dest_ptr, source_ptr, bytes, peer.value(),
  //                        signal_flags_, gpu_stream);
  // } else {
  //   res = xla_mori::Recv(dest_ptr, source_ptr, bytes, peer.value(),
  //                        signal_flags_, gpu_stream);
  // }
  // if (res == 0) return absl::OkStatus();
  return absl::InternalError(absl::StrFormat("MORI %s failed", stype));
}

Future<> MoriCommunicator::GroupExecute(
    absl::AnyInvocable<absl::Status() &&> group) {
  return Execute([group = std::move(group), this]() mutable {
    return GroupLaunch([&] { return std::move(group)(); });
  });
}

absl::Status MoriCommunicator::GroupLaunch(
    absl::FunctionRef<absl::Status()> group) {
  return group();
}

absl::Status MoriCommunicator::Quiet(const Executor& executor) {
  VLOG(1) << "Quiet MORI communicator: " << ToString();
  CHECK_CANCELLED()
  ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  auto gpu_stream = AsRocmStream(stream);
  (void)gpu_stream;
  return absl::UnimplementedError("Not implemented");
  // rocm_mori_quiet_on_stream(AsRocmStream(stream));
  // return absl::OkStatus();
}

absl::Status MoriCommunicator::Fence() {
  VLOG(1) << "Fence MORI communicator: " << ToString();
  CHECK_CANCELLED()
  // rocm_mori_fence();
  return absl::UnimplementedError("Not implemented");
}

absl::Status MoriCommunicator::PollUntilDone() const {
  CHECK_CANCELLED()
  return absl::UnimplementedError("Not implemented");
}

Future<> MoriCommunicator::Execute(
    absl::AnyInvocable<absl::Status() &&> f) const {
  return Future<>(std::move(f)());
}

}  // namespace xla::gpu
