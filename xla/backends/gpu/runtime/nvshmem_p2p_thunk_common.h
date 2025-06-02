/* Copyright 2025 The OpenXLA Authors. */

#ifndef XLA_BACKENDS_GPU_RUNTIME_NVSHMEM_P2P_THUNK_COMMON_H_
#define XLA_BACKENDS_GPU_RUNTIME_NVSHMEM_P2P_THUNK_COMMON_H_

#include "xla/backends/gpu/runtime/thunk.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/backends/gpu/runtime/nvshmem_collective_thunk.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/global_device_id.h"

namespace xla {
namespace gpu {

// Count the number of times a Send or Recv instruction executed on a device.
class NvshmemP2PExecutionCounters {
 public:
  absl::Status Initialize(se::StreamExecutor* executor, RunId run_id);
  absl::StatusOr<int64_t*> GetCounter(se::StreamExecutor* executor,
                                      RunId run_id);

 private:
  using CounterKey = std::pair<se::StreamExecutor*, RunId>;
  absl::Mutex mu_;
  // TODO(b/338288906): may need to clean up the counters for finished runs.
  absl::flat_hash_map<CounterKey, int64_t> counters_ ABSL_GUARDED_BY(mu_);
};

// Structure to store P2P configuration for NVSHMEM operations
struct NvshmemP2PConfig {
  // For put: target is populated, source is nullopt
  // For get: source is populated, target is nullopt
  struct SourceTargetMapEntry {
    std::optional<int64_t> source;
    std::optional<int64_t> target;
  };

  using IdToSourceTargetMap =
      absl::flat_hash_map<int64_t, SourceTargetMapEntry>;

  enum class ValidationKind { kValid = 0, kInvalid = 1, kConditional = 2 };

  using SourceTargetToBounds = absl::flat_hash_map<std::pair<int64_t, int64_t>,
                                                   std::pair<int64_t, int64_t>>;

  static SourceTargetMapEntry GetSourceTarget(
      const IdToSourceTargetMap& id_to_source_target, int64_t id) {
    auto it = id_to_source_target.find(id);
    if (it != id_to_source_target.end()) return it->second;
    return SourceTargetMapEntry{};
  }

  CollectiveConfig config;
  IdToSourceTargetMap id_to_source_target;
  ValidationKind validation_kind = ValidationKind::kValid;
  // When a Send or Recv has validation_kind = ValidationKind::kConditional,
  // record the valid execution numbers as a pair of [lower-bound, upper-bound]
  // for each source and target pair.
  SourceTargetToBounds source_target_to_bounds;
};

// Extracts source/target pairs for send/recv from frontend attributes.
absl::StatusOr<std::vector<std::pair<int64_t, int64_t>>>
NvshmemP2PGetSourceTargetPairs(mlir::DictionaryAttr frontend_attributes);

// Get P2P config for put/get operations
NvshmemP2PConfig GetNvshmemP2PConfigForPutGet(
    const HloSendRecvInstruction* instr, const Shape& shape,
    int64_t replica_count, int64_t partition_count);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_NVSHMEM_P2P_THUNK_COMMON_H_
