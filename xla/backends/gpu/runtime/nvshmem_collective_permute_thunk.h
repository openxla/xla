#ifndef XLA_BACKENDS_GPU_RUNTIME_NVSHMEM_COLLECTIVE_PERMUTE_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_NVSHMEM_COLLECTIVE_PERMUTE_THUNK_H_

#include <cstdint>
#include <memory>
#include <unordered_map>

#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"
#include "xla/backends/gpu/runtime/nvshmem_p2p_thunk_common.h"
#include "xla/backends/gpu/collectives/nvshmem_collectives.h"
#include "xla/backends/gpu/runtime/collective_permute_thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {
namespace gpu {

using tsl::AsyncValueRef;

// Thunk that performs a NVSHMEM-based collective permute.
class NvshmemCollectivePermuteStartThunk : public NvshmemCollectiveThunk {
 public:
  NvshmemCollectivePermuteStartThunk(
      ThunkInfo thunk_info, const HloCollectivePermuteInstruction* instr,
      int64_t replica_count, int64_t partition_count,
      const std::vector<CollectiveThunk::Buffer>& buffers,
      bool p2p_memcpy_enabled = false,
      AsyncStreamKind stream_kind = AsyncStreamKind::kCollective);

  static const char* GetHloOpName() { return "collective-permute-start"; }

  static absl::Status CheckImplementable(
      const HloCollectivePermuteInstruction* inst, int64_t replica_count,
      int64_t partition_count);

  static CollectiveOpGroupMode GetGroupMode(
      const HloCollectivePermuteInstruction* instr);

  static NvshmemP2PConfig GetNvshmemP2PConfig(
      const HloCollectivePermuteInstruction* instr, int64_t replica_count,
      int64_t partition_count);

  absl::Status Initialize(const InitializeParams& params) override;

 protected:
  const CollectiveConfig& config() const override { return config_.config; }
  absl::Status RunNvshmemCollective(const ExecuteParams& params,
                                    se::Stream& stream) override;

 private:
  const NvshmemP2PConfig config_;
  const std::vector<CollectiveThunk::Buffer> buffers_;
  const bool p2p_memcpy_enabled_ = false;
  absl::Mutex barrier_mutex_;
  std::map<int64_t, std::unique_ptr<se::Event>> receiver_barrier_events_;
};

// Thunk that performs a NVSHMEM-based collective permute done operation.
class NvshmemCollectivePermuteDoneThunk : public NvshmemCollectiveDoneThunk {
 public:
  NvshmemCollectivePermuteDoneThunk(
      ThunkInfo thunk_info,
      std::shared_ptr<CollectiveThunk::AsyncEvents> async_events,
      AsyncStreamKind stream_kind);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
};

absl::Status RunCollectivePermute(
    NvshmemP2PConfig::SourceTargetMapEntry source_target,
    std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
    absl::string_view device_string, int64_t current_id);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_NVSHMEM_COLLECTIVE_PERMUTE_THUNK_H_