/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_NCCL_ALL_TO_ALL_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_NCCL_ALL_TO_ALL_THUNK_H_

#include <cstdint>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {
namespace gpu {

struct NcclAllToAllConfig {
  NcclCollectiveConfig config;
  bool has_split_dimension;
};

// Thunk that performs a NCCL-based All-to-All among CUDA GPU-based replicas.
class NcclAllToAllStartThunk : public NcclCollectiveThunk {
 public:
  class RecvPtrMap {
   public:
    bool IsInitialized(int64_t send_id, int64_t receive_id) {
      absl::MutexLock lock(&mutex_);
      return recv_ptrs_.find(send_id) != recv_ptrs_.end() &&
             recv_ptrs_.at(send_id).find(receive_id) !=
                 recv_ptrs_.at(send_id).end();
    }

    absl::Status InitializeId(int64_t send_id, int64_t receive_id) {
      absl::MutexLock lock(&mutex_);
      if (recv_ptrs_.find(send_id) == recv_ptrs_.end()) {
        recv_ptrs_[send_id] =
            absl::node_hash_map<int64_t, tsl::AsyncValueRef<void*>>();
      }
      if (recv_ptrs_.at(send_id).find(receive_id) ==
          recv_ptrs_.at(send_id).end()) {
        recv_ptrs_.at(send_id)[receive_id] =
            tsl::MakeUnconstructedAsyncValueRef<void*>();
      }
      return absl::OkStatus();
    }

    absl::Status PutRecvPtr(int64_t send_id, int64_t receive_id, void* ptr) {
      if (!IsInitialized(send_id, receive_id)) {
        return absl::InternalError(absl::StrCat("Send-receive pair ", send_id,
                                                ", ", receive_id,
                                                " has not been initialized!"));
      }
      absl::MutexLock lock(&mutex_);
      if (recv_ptrs_.at(send_id).at(receive_id).IsUnavailable()) {
        VLOG(3) << "Putting pointer: " << ptr << " for send_id " << send_id
                << ", and receive_id " << receive_id;
        recv_ptrs_.at(send_id).at(receive_id).emplace(ptr);
      }
      return absl::OkStatus();
    }

    absl::StatusOr<tsl::AsyncValueRef<void*>> GetRecvPtr(int64_t send_id,
                                                         int64_t receive_id) {
      if (!IsInitialized(send_id, receive_id)) {
        return absl::InternalError(absl::StrCat("Send-receive pair ", send_id,
                                                ", ", receive_id,
                                                " has not been initialized!"));
      }
      absl::MutexLock lock(&mutex_);
      return recv_ptrs_.at(send_id).at(receive_id);
    }

   private:
    absl::Mutex mutex_;
    absl::node_hash_map<int64_t,
                        absl::node_hash_map<int64_t, tsl::AsyncValueRef<void*>>>
        recv_ptrs_ ABSL_GUARDED_BY(mutex_);
  };

  NcclAllToAllStartThunk(ThunkInfo thunk_info, NcclApi* nccl_api,
                         const HloAllToAllInstruction* instr,
                         std::vector<Buffer> buffers, bool p2p_memcpy_enabled);

  // Returns whether the given instruction can be lowered to a nccl all-to-all
  // call.
  static absl::Status CheckImplementable(const HloAllToAllInstruction* instr,
                                         int64_t replica_count,
                                         int64_t partition_count);

  absl::Status Initialize(const InitializeParams& params) override;

  static const char* GetHloOpName() { return "all-to-all-start"; }

  static CollectiveOpGroupMode GetGroupMode(
      const HloAllToAllInstruction* instr);

 protected:
  const NcclCollectiveConfig& config() const override { return config_.config; }
  absl::Status RunNcclCollective(const ExecuteParams& params,
                                 se::Stream& stream,
                                 NcclCommHandleWrapper comm_wrapper) override;

  AsyncStreamKind GetAsyncStreamKind() const override;

  bool is_local() const;

 private:
  const NcclAllToAllConfig config_;
  const std::vector<Buffer> buffers_;
  RecvPtrMap recv_ptr_map_;
  int64_t device_count_ = 1;
  bool p2p_memcpy_enabled_ = false;
};

absl::Status RunAllToAll(NcclApi* nccl_api, bool has_split_dimension,
                         std::vector<DeviceBufferPair>& buffers,
                         se::Stream& stream, NcclApi::NcclCommHandle comm,
                         int64_t current_id, bool use_memcpy,
                         NcclAllToAllStartThunk::RecvPtrMap& recv_ptr_map);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_NCCL_ALL_TO_ALL_THUNK_H_
