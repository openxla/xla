#ifndef XLA_SERVICE_GPU_CUSTOM_CALL_PARTITIONERS_H_
#define XLA_SERVICE_GPU_CUSTOM_CALL_PARTITIONERS_H_

#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/custom_call_sharding_helper.h"
#include "xla/service/spmd/spmd_partitioner.h"

namespace xla::gpu::spmd {

// Partitions custom calls as element-wise ops.
// Custom calls using this partitioner must follow element-wise op semantics,
// otherwise there will be undefined behaviors.
class PassThroughPartitioner : public CustomCallPartitioner {
 public:
  bool IsCustomCallShardable(const HloInstruction* instruction) const override {
    return true;
  }

  absl::Status Partition(xla::spmd::SpmdPartitioningVisitor* partitioner,
                         HloInstruction* hlo) const override {
    // Partitions the custom call just as its operands.
    return partitioner->HandleElementwise(hlo);
  }

  // This allows replicated sharding on custom-call op to pass checks at spmd
  // partitioner preprocess stage.
  bool CanSideEffectingHaveReplicatedSharding() const override { return true; }

  // Run through this custom call for both forward and backwnard propagation.
  std::optional<HloSharding> InferShardingFromOperands(
      const HloInstruction* instruction) const override {
    if (instruction->operand(0)->has_sharding()) {
      return instruction->operand(0)->sharding();
    }
    return std::nullopt;
  }
};

class AllocateBufferPartitioner : public CustomCallPartitioner {
 public:
  bool IsCustomCallShardable(const HloInstruction* instruction) const override {
    return true;
  }

  absl::Status Partition(xla::spmd::SpmdPartitioningVisitor* partitioner,
                         HloInstruction* hlo) const override {
    // Partitions the custom call just as its operands.
    return partitioner->HandleElementwise(hlo);
  }

  // Always let backward propagation run through with ShardBarrierFrom.
  HloSharding PropagateUserSharding(
      const HloInstruction* instruction, const HloInstruction* user,
      const HloSharding& sharding) const override {
    return sharding;
  }
};

}  // namespace xla::gpu::spmd

#endif  // XLA_SERVICE_GPU_CUSTOM_CALL_PARTITIONERS_H_
