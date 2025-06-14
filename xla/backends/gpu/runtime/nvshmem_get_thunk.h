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

#ifndef XLA_SERVICE_GPU_NVSHMEM_GET_THUNK_H_
#define XLA_SERVICE_GPU_NVSHMEM_GET_THUNK_H_

#include <vector>

#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/stream_executor/stream.h"
#include "xla/backends/gpu/runtime/nvshmem_p2p_thunk_common.h"

namespace xla {
namespace gpu {

// Thunk to perform NVSHMEM get operations
class NvshmemGetThunk : public NvshmemCollectiveThunk {
 public:
  NvshmemGetThunk(ThunkInfo thunk_info, const HloRecvInstruction* inst,
                  int64_t replica_count, int64_t partition_count,
                  const CollectiveThunk::Buffer& buffer);
  absl::Status Initialize(const InitializeParams& params) override;

  // Returns the group mode (cross-replica or cross-partition) for the operation
  static CollectiveOpGroupMode GetGroupMode(const HloRecvInstruction* inst);

 protected:
  const CollectiveConfig& config() const override { return config_.config; }
  absl::Status RunNvshmemCollective(const ExecuteParams& params,
                                    se::Stream& stream) override;

 private:
  const NvshmemP2PConfig config_;
  const CollectiveThunk::Buffer buffer_;
  std::shared_ptr<NvshmemP2PExecutionCounters> execution_counters_;
  std::string hlo_name_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_NVSHMEM_GET_THUNK_H_