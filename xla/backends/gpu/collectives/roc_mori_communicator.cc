/* Copyright 2025 The OpenXLA Authors.
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

#include "xla/backends/gpu/collectives/roc_mori_communicator.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/roc_mori_collectives.h"
#include "xla/backends/gpu/collectives/roc_mori_kernels.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/future.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

using namespace mori;
namespace xla::gpu {

// A NCCL window registration handle that makes local buffers accessible from
// remote peers via symmetric memory registration process.
class MoriSymmetricMemory final : public SymmetricMemory {
 public:
  ~MoriSymmetricMemory() final {
    // NOTE NOTE: this might be called from a wrong thread!!
    // hence we cannpot deregister it here


    // Remove the object from our global map
    // roc_mori::DeregisterMemObjPtr(addr_.opaque());
    // shmem::ShmemSymmetricDeregister(addr_.opaque(), addr_.size());
  }

  static absl::StatusOr<std::unique_ptr<MoriSymmetricMemory>> Create(
    se::DeviceAddressBase addr) {
      auto obj = shmem::ShmemSymmetricRegister(addr.opaque(), addr.size());
      if (obj.cpu == nullptr || obj.gpu == nullptr) {
        return absl::InternalError("Failed to register symmetric memory");
      }
      // Add the object to our global map
      roc_mori::RegisterMemObjPtr(addr.opaque(), obj);
      VLOG(1) << "Registered symmetric memory: "
              << obj.cpu->localPtr << " size: " << obj.cpu->size;
      return absl::WrapUnique(new MoriSymmetricMemory(addr));
  }

  se::DeviceAddressBase addr() const final { 
    VLOG(1) << "MoriSymmetricMemory::addr: " << addr_.opaque();
    return addr_; 
  }

  absl::StatusOr<se::DeviceAddressBase> peer_addr(
    RankId rank) const final { 
        return absl::UnimplementedError("Peer address not supported"); 
  }

  std::string ToString() const final {
    return absl::StrFormat(
      "MoriSymmetricMemory(ptr=%p, size=%ld)", addr_.opaque(), addr_.size());
  }

  PackedKernelArg PackKernelArg() const final { 
    return addr_.opaque();
  }

 private:
  MoriSymmetricMemory(
    se::DeviceAddressBase addr) : addr_(addr) {}

  se::DeviceAddressBase addr_;
};


static auto AsRocmStream(se::Stream* stream) {
  return reinterpret_cast< std::intptr_t >(
                                  stream->platform_specific_handle().stream);
}

static size_t ToMoriByteCount(PrimitiveType dtype, size_t count) {
  if (primitive_util::IsComplexType(dtype)) count *= 2;
  return count * primitive_util::BitWidth(dtype) / 8;
}

absl::StatusOr<std::unique_ptr<MoriCommunicator>> MoriCommunicator::Create(
    MoriCollectives* coll) {
  auto comm = absl::WrapUnique(new MoriCommunicator(coll));
  //TF_RETURN_IF_ERROR(comm->InitSignals());
  VLOG(1) << "Created " << *comm << " with npes: " << shmem::ShmemNPes();
  return comm;
}

MoriCommunicator::~MoriCommunicator() {
  if (signal_flags_ != nullptr) {
    collectives_->Deallocate(signal_flags_).IgnoreError();
    signal_flags_ = nullptr;
  }
  // if (teams_ != nullptr) {
  //   for (uint32_t i = 0; i < kMaxTeams; i++) {
  //     rocm_mori_team_destroy(teams_[i]);
  //   }
  // }
  // collectives_->Deallocate(teams_).IgnoreError();
}

#define CHECK_ABORTED() \
  if (aborted_) return FailedPrecondition("MoriCommunicator aborted");

absl::Status MoriCommunicator::Abort() {
  VLOG(1) << "Abort MORI communicator: " << ToString();
  CHECK_ABORTED()
  aborted_ = true;
  // Call rocm_mori_global_exit with a non-zero return code to abort the program.
  // rocm_mori_global_exit(1);
  return absl::OkStatus();
}

absl::Status MoriCommunicator::Barrier(
    const Communicator::Executor& executor) {
  VLOG(1) << "Barrier: " << ToString();
  CHECK_ABORTED()
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  return roc_mori::BarrierOnStream(AsRocmStream(stream));
}

absl::StatusOr<size_t> MoriCommunicator::NumRanks() const {
  VLOG(5) << "Get the number of ranks in MORI communicator: " << ToString();
  CHECK_ABORTED()

  int32_t count = shmem::ShmemNPes();
  if (count < 0) {
    return absl::InvalidArgumentError("MoriCommunicator::NumRanks failed.");
  }
  return count;
}

absl::StatusOr<size_t> MoriCommunicator::CurrentRank() {
  VLOG(5) << "Get current rank in MORI communicator: " << ToString();
  CHECK_ABORTED()

  int32_t rank = shmem::ShmemMyPe();
  if (rank < 0) {
    return absl::InvalidArgumentError("MoriCommunicator::CurrentRank failed.");
  }
  return rank;
}

std::string MoriCommunicator::ToString() const {
  return absl::StrFormat("MoriCommunicator(my_pe=%d, npes=%d)", 
        shmem::ShmemMyPe(), shmem::ShmemNPes());
}

absl::StatusOr<se::Stream*> MoriCommunicator::ToStream(
    const Executor& executor) {
  if (auto* gpu_executor =
          tsl::down_cast<const GpuCollectives::Executor*>(&executor)) {
    return gpu_executor->stream();
  }
  return InvalidArgument("Communicator executor is not a GPU executor");
}

absl::StatusOr<std::unique_ptr<SymmetricMemory>>
MoriCommunicator::CreateSymmetricMemory(se::DeviceAddressBase addr) {
  return MoriSymmetricMemory::Create(addr);
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

Future<> MoriCommunicator::Send(se::DeviceMemoryBase recv_buffer,
                                se::DeviceAddressBase send_buffer,
                                PrimitiveType dtype, size_t count, RankId peer,
                                const Executor& executor) {
  return P2P(P2PType::Send, dtype, recv_buffer, send_buffer, count, peer, executor);
}

Future<> MoriCommunicator::Recv(se::DeviceAddressBase recv_buffer,
                                se::DeviceAddressBase send_buffer,
                                PrimitiveType dtype, size_t count, RankId peer,
                                const Executor& executor) {
  return P2P(P2PType::Recv, dtype, recv_buffer, send_buffer, count, peer, executor);
}

absl::Status MoriCommunicator::LaunchAllGather(se::DeviceAddressBase send_buffer,
  se::DeviceAddressBase recv_buffer, PrimitiveType dtype, size_t count,
  const Executor& executor) {
  CHECK_ABORTED()
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  VLOG(3) << "LaunchAllGather: send_buffer=" << send_buffer.opaque() 
          << " recv_buffer=" << recv_buffer.opaque() 
          << " count=" << count 
          << " dtype=" << primitive_util::LowercasePrimitiveTypeName(dtype) 
          << " stream=" << AsRocmStream(stream);
  return roc_mori::AllGather(send_buffer.opaque(), recv_buffer.opaque(), 
                      ToMoriByteCount(dtype, count), 
                      AsRocmStream(stream), stream->parent()->device_ordinal());
}

absl::Status MoriCommunicator::LaunchAllReduce(
        se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
        PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
        const Executor& executor) {
  CHECK_ABORTED()

  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  auto gpu_stream = AsRocmStream(stream);
  (void)gpu_stream;
  void* source_ptr = send_buffer.opaque();
  void* dest_ptr = recv_buffer.opaque();
  (void)source_ptr;
  (void)dest_ptr;
  if (primitive_util::IsComplexType(dtype)) count *= 2;

  VLOG(3) << absl::StreamFormat(
      "Launch MORI AllReduce operation on device #%d; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%v; comm=node; "
      "team=%p;"
      "stream=%p",
      -1,//rocm_mori_team_my_pe(host_team_), 
      send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, reduction_kind, nullptr, stream);

  // auto call = [&](auto T) -> absl::Status {
  //   using Type = decltype(T);
  //   auto *dest = static_cast< Type *>(dest_ptr);
  //   const auto *source = static_cast< const Type *>(source_ptr);
  //   return allreduce_on_stream< Type >(
  //         teams_, kMaxTeams, dest, source, count, reduction_kind, gpu_stream);
  // };

  // switch(dtype) {
  // case PrimitiveType::F64: return call(double{});
  // case PrimitiveType::F32: return call(float{});
  // case PrimitiveType::S64: return call(longlong{});
  // case PrimitiveType::S32: return call(int{});
  // case PrimitiveType::S16: return call(short{});
  // }
  return absl::InternalError("Invalid MORI reduction type.");
}

absl::Status MoriCommunicator::LaunchCollectivePermute(
     se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer, 
     PrimitiveType dtype, size_t count, std::optional<RankId> source_rank, 
     absl::Span<const RankId> target_ranks, const Executor& executor) {
  CHECK_ABORTED()
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  size_t bytes = ToMoriByteCount(dtype, count);

  if (target_ranks.empty()) {
    VLOG(3) << "No target ranks, skipping CollectivePermute";
    return absl::OkStatus();
  }

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

  // NOTE normally we could merge these to a single kernel
  for (auto target_rank : target_ranks) {
    TF_RETURN_IF_ERROR(roc_mori::SendSDMA(recv_buffer.opaque(), send_buffer.opaque(), 
                bytes, target_rank.value(), AsRocmStream(stream), 
                stream->parent()->device_ordinal()));
  }
  return absl::OkStatus();
}

// Lazily allocate per-peer signal flags and block retirement counter.
absl::Status MoriCommunicator::InitSignals() {
  TF_ASSIGN_OR_RETURN(auto npes, NumRanks());
  size_t alloc_size = (npes + 1) * sizeof(uint32_t);
  TF_ASSIGN_OR_RETURN(auto *ptr, collectives_->Allocate(alloc_size));
  signal_flags_ = static_cast< uint32_t *>(ptr);
  // Zero-initialise everything (signals must start at 0).
  roc_mori::InitSignalMemory(signal_flags_, alloc_size);
  return absl::OkStatus();
}

// Performs point-to-point communication between two ranks using MORI.
// Send: launches a single GPU kernel that copies data to the peer via P2P
//       and sets a completion flag on the peer.
// Recv: launches a single-thread GPU kernel that waits for the flag.
absl::Status MoriCommunicator::P2P(P2PType p2p_type,
                                      PrimitiveType dtype,
                                      se::DeviceMemoryBase recv_buffer,
                                      se::DeviceMemoryBase send_buffer,
                                      size_t count, RankId peer,
                                      const Executor& executor) {
  
  const char *stype = (p2p_type == P2PType::Send ? " Send" : " Recv");
  VLOG(1) << CurrentRank().value() << stype << " to " << peer.value() 
          << " count " << count
          << " MORI communicator: " << ToString() ;
  CHECK_ABORTED()

  void* source_ptr = send_buffer.opaque();
  void* dest_ptr = recv_buffer.opaque();

  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  auto gpu_stream = AsRocmStream(stream); 
  size_t bytes = ToMoriByteCount(dtype, count);
  int res = 0;
  if (p2p_type == P2PType::Send) {
    res = roc_mori::Send(dest_ptr, source_ptr, bytes, peer.value(),
                         signal_flags_, gpu_stream);
  } else {
    res = roc_mori::Recv(dest_ptr, source_ptr, bytes, peer.value(),
                         signal_flags_, gpu_stream);
  }
  if (res == 0) return absl::OkStatus();
  return absl::InternalError(absl::StrFormat("MORI %s failed", stype));
}

Future<> MoriCommunicator::GroupExecute(
  absl::AnyInvocable<absl::Status(GpuCommunicator*)> f) {
  return Execute([f = std::move(f), this]() mutable -> absl::Status {
    return f(this);
  });
}

absl::Status MoriCommunicator::Quiet(const Executor& executor) {
  VLOG(1) << "Quiet MORI communicator: " << ToString();
  CHECK_ABORTED()
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  auto gpu_stream = AsRocmStream(stream);
  (void)gpu_stream;
  return absl::UnimplementedError("Not implementedA");
  // rocm_mori_quiet_on_stream(AsRocmStream(stream));
  // return absl::OkStatus();
}

absl::Status MoriCommunicator::Fence() {
  VLOG(1) << "Fence MORI communicator: " << ToString();
  CHECK_ABORTED()
  // rocm_mori_fence();
  return absl::UnimplementedError("Not implementedB");
}

Future<> MoriCommunicator::Execute(
     absl::AnyInvocable<absl::Status() &&> f) const {
  return Future<>(std::move(f)());
}

}  // namespace xla::gpu
