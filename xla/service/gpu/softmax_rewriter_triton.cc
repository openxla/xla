/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

#include "xla/service/gpu/softmax_rewriter_triton.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout_util.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/bitcast_dtypes_expander.h"
#include "xla/service/compilation_environments.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/float_normalization.h"
#include "xla/service/gpu/autotuner_compile_util.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/fusion_pipeline.h"
#include "xla/service/gpu/gemm_rewriter_triton.h"
#include "xla/service/gpu/gpu_device_info.h"
#include "xla/service/gpu/gpu_float_support.h"
#include "xla/service/gpu/gpu_types.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/prepare_hlo_for_ir_emitting_pipeline.h"
#include "xla/service/gpu/reduction_dimension_grouper.h"
#include "xla/service/gpu/reduction_splitter.h"
#include "xla/service/gpu/tree_reduction_rewriter.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/simplify_fp_conversions.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_pimpl.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

namespace xla::gpu {
namespace {

bool HasDefaultLayout(const Shape& shape) {
  return shape.has_layout() &&
         LayoutUtil::IsMonotonicWithDim0Major(shape.layout());
}

bool IsTritonSupportedInstruction(const HloInstruction* instr,
                                  const GpuVersion& gpu_version) {
  if (!instr->shape().IsArray()) {
    return false;
  }

  if (!IsTritonSupportedDataType(instr->shape().element_type(), gpu_version)) {
    return false;
  }

  for (const HloInstruction* operand : instr->operands()) {
    if (!IsTritonSupportedDataType(operand->shape().element_type(),
                                   gpu_version)) {
      return false;
    }
  }

  // TODO(bchetioui): expand with non-trivial instructions.
  if (instr->IsElementwise()) {
    return IsTritonSupportedElementwise(instr->opcode(),
                                        instr->shape().element_type());
  }

  switch (instr->opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kParameter:
      return true;
    default:
      return false;
  }
}

// Returns true if a trivially connected producer of 'consumer' with opcode
// 'opcode' exists. If such an instruction is found, the value of 'producer' is
// set to it. The definition of "trivial" operations is as given in
// 'IsTriviallyFusible'.
bool TrivialEdge(HloInstruction** producer, HloInstruction* consumer,
                 HloOpcode opcode, const GpuVersion& gpu_version);

bool BitcastIsTilingNoop(HloInstruction* bitcast,
                         const GpuVersion& gpu_version) {
  CHECK_EQ(bitcast->opcode(), HloOpcode::kBitcast);

  if (ShapeUtil::IsEffectiveScalar(bitcast->shape())) {
    return true;
  }

  // In the Softmax rewriter for now, tiling is derived from a hero reduction
  // operation, which should be reducing its input on the last axis. Therefore,
  // a bitcast is always a no-op with regards to a tile if
  //   (1) it does not change the size of the reduction dimension of its input
  //       (the last one); if its input is already reduced, then (1) is true
  //       by default
  //   (2) the layout of its output is ordered in the same way as the layout of
  //       its input. This is a fuzzy definition, but since we assume fusible
  //       ops to always have a default layout, we can just check if both the
  //       bitcast and its input have a default layout
  auto last_dimension = [](const HloInstruction* instr) {
    return instr->shape().dimensions().back();
  };

  HloInstruction* reduce = nullptr;
  TrivialEdge(&reduce, bitcast->mutable_operand(0), HloOpcode::kReduce,
              gpu_version);

  return (HasDefaultLayout(bitcast->shape()) &&
          HasDefaultLayout(bitcast->operand(0)->shape()) &&
          (reduce != nullptr ||
           last_dimension(bitcast->operand(0)) == last_dimension(bitcast)));
}

bool IsTriviallyFusible(HloInstruction* instr, const GpuVersion& gpu_version,
                        int num_allowed_users = 1) {
  // Checks whether an op is trivially fusible. An op is said to be trivially
  // fusible if it does not increase the amount of memory read/written by the
  // resulting fusion, is compatible with any chosen tiling, and can be
  // codegen'd using Triton. The op is allowed to have up to num_allowed_users
  // users.
  if (instr->user_count() > num_allowed_users ||
      !HasDefaultLayout(instr->shape())) {
    return false;
  }

  if (instr->opcode() == HloOpcode::kBitcast &&
      BitcastIsTilingNoop(instr, gpu_version)) {
    return true;
  }

  if (instr->IsElementwise() && instr->operand_count() == 1) {
    return IsTritonSupportedInstruction(instr, gpu_version);
  }

  if (instr->IsElementwiseBinary() && instr->operand(0) == instr->operand(1)) {
    return IsTritonSupportedInstruction(instr, gpu_version);
  }

  return false;
}

bool TrivialEdge(HloInstruction** producer, HloInstruction* consumer,
                 HloOpcode opcode, const GpuVersion& gpu_version) {
  while (consumer->opcode() != opcode) {
    if (IsTriviallyFusible(consumer, gpu_version)) {
      consumer = consumer->mutable_operand(0);
    } else {
      return false;
    }
  }

  *producer = consumer;
  return true;
}

bool IsTriviallyConnectedProducerOf(HloInstruction* producer,
                                    HloInstruction* consumer,
                                    const GpuVersion& gpu_version) {
  if (producer == consumer) {
    return true;
  }

  HloInstruction* found_producer = consumer;
  while (
      TrivialEdge(&found_producer, consumer, producer->opcode(), gpu_version)) {
    if (found_producer == producer) {
      return true;
    }

    if (!IsTriviallyFusible(found_producer, gpu_version)) {
      return false;
    }

    consumer = found_producer->mutable_operand(0);
  }

  return false;
}

inline bool HasOneUse(const HloInstruction* instr) {
  return instr->user_count() == 1;
}

bool IsTritonSupportedComputation(const HloComputation* computation,
                                  const GpuVersion& gpu_version) {
  for (const HloInstruction* instr : computation->instructions()) {
    if (!IsTritonSupportedInstruction(instr, gpu_version)) {
      return false;
    }
  }
  return true;
}

std::optional<HloInstruction*> MatchesTritonCompatibleClosedReductionDiamond(
    HloInstruction* instr, const GpuVersion& gpu_version) {
  // Return the producer of the following pattern:
  //
  // producer
  // |    \
  // |  reduce_{max,sum,...}
  // |     |
  // |  broadcast
  // |   /
  // binop (elementwise)
  //
  // where each edge is allowed to contain also trivial operations that can be
  // generated by Triton. We mean by "trivial" here those operations that do not
  // increase the amount of memory read/written by the fusion, and that are
  // compatible with any chosen tiling.
  //
  // We also assume that the reduction is done on the last axis of the producer
  // array.
  std::optional<HloInstruction*> match_failure = std::nullopt;

  if (!instr->IsElementwiseBinary() ||
      !IsTritonSupportedInstruction(instr, gpu_version)) {
    return match_failure;
  }

  HloInstruction* producer;
  HloInstruction* broadcast;
  HloInstruction* reduce;

  if (!(TrivialEdge(&broadcast, instr->mutable_operand(1),
                    HloOpcode::kBroadcast, gpu_version) &&
        TrivialEdge(&reduce, broadcast->mutable_operand(0), HloOpcode::kReduce,
                    gpu_version) &&
        HasDefaultLayout(broadcast->shape()) &&
        HasDefaultLayout(reduce->shape()) && reduce->operand_count() == 2 &&
        reduce->operand(1)->opcode() == HloOpcode::kConstant &&
        IsTritonSupportedComputation(reduce->to_apply(), gpu_version))) {
    return match_failure;
  }

  if (!HasOneUse(broadcast) || !HasOneUse(reduce)) {
    return match_failure;
  }

  producer = reduce->mutable_operand(0);

  if (!(reduce->dimensions().size() == 1 &&
        reduce->dimensions(0) == producer->shape().rank() - 1 &&
        !absl::c_linear_search(broadcast->dimensions(),
                               broadcast->shape().rank() - 1))) {
    return match_failure;
  }

  // TODO(b/291204753): remove this filter. This heuristic enables flipping the
  // default flag while filtering out cases that could result in regressions.
  if (reduce->operand(0)->shape().dimensions().back() < 64) {
    return match_failure;
  }

  while (IsTriviallyFusible(producer, gpu_version)) {
    producer = producer->mutable_operand(0);
  }

  if (!HasDefaultLayout(producer->shape()) ||
      !IsTriviallyConnectedProducerOf(producer, instr->mutable_operand(0),
                                      gpu_version) ||
      !(producer == instr->operand(0) ||
        instr->operand(0)->user_count() == 1)) {
    return match_failure;
  }

  return producer;
}

// Finds the first non-fusible producer of a diamond. This instruction is either
//   1. the direct producer of the diamond, if that producer is used more than
//      twice and/or is not otherwise trivially fusible
//   2. the first parent instruction of the producer of the diamond such that
//      that instruction is used more than once, and/or is not trivially
//      fusible.
HloInstruction* FindFirstNonFusibleDiamondProducer(
    HloInstruction* diamond_producer, const GpuVersion& gpu_version) {
  if (IsTriviallyFusible(diamond_producer, gpu_version,
                         /*num_allowed_users=*/2)) {
    diamond_producer = diamond_producer->mutable_operand(0);
    while (IsTriviallyFusible(diamond_producer, gpu_version)) {
      diamond_producer = diamond_producer->mutable_operand(0);
    }
  }

  return diamond_producer;
}

using InstructionSet = absl::flat_hash_set<const HloInstruction*>;

struct ModuleDescriptor {
  InstructionSet inputs;
  InstructionSet outputs;
};

// A filter that returns true for a given opcode if the corresponding operation
// is likely to create a fusion boundary with regards to a diamond chain
// TODO(bchetioui): explain what it means to create a fusion boundary
bool CreatesFusionBoundary(const HloOpcode& opcode) {
  switch (opcode) {
    // Expensive instructions or unusual instructions for which fusion is
    // nonsensical.
    case HloOpcode::kAddDependency:
    case HloOpcode::kAfterAll:
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kCall:
    case HloOpcode::kCholesky:
    case HloOpcode::kConditional:
    case HloOpcode::kConvolution:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDomain:
    case HloOpcode::kDot:
    case HloOpcode::kFft:
    case HloOpcode::kFusion:
    case HloOpcode::kGather:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kParameter:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kRng:
    case HloOpcode::kRngGetAndUpdateState:
    case HloOpcode::kRngBitGenerator:
    case HloOpcode::kScatter:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kSort:
    case HloOpcode::kTopK:
    case HloOpcode::kTriangularSolve:
    // Stop at tuple and tuple elements for now to avoid memory problems
    // TODO(bchetioui): fix this by simplifying tuples outside of this.
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kTuple:
    case HloOpcode::kWhile:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kSetDimensionSize:
      return true;
    // The non-exhaustive enumeration means that we may be too conservative
    // here. That's OK, since the goal is to avoid introducing new
    // regressions, and the analysis is anyway approximative.
    // TODO(bchetioui): adjust comment above and see if better to make the
    // opposite list or not. (probably should just make an exhaustive list)
    default:
      return false;
  }
}

// Sort an instruction set topologically, i.e. such that for two instructions
// i1 and i2 such that i1 is an operand to i2 (noted i1 -> i2), then i1 appears
// before i1 in the resulting vector. For three instructions i1 -> i2 -> i3,
// this property holds transitively for i1 and i3 only if i2 is also part of
// the instruction set; otherwise, i1 and i3 may be ordered one way or the
// other.
std::vector<const HloInstruction*> SortInstructionSetTopologically(
    InstructionSet instructions_to_sort) {
  if (instructions_to_sort.empty()) return {};

  std::vector<const HloInstruction*> ordered_instructions;
  ordered_instructions.reserve(instructions_to_sort.size());

  std::queue<const HloInstruction*> processable_instructions;

  // An instruction is processable/ready to be processed iff all its operands
  // have already been processed. Operands that have not been tagged as fusible
  // during the first traversal in this function are considered to have been
  // processed. First, we populate `processable_instructions` with all the
  // initially processable instructions.
  for (const HloInstruction* instr : instructions_to_sort) {
    bool is_processable = true;
    for (const HloInstruction* operand : instr->operands()) {
      if (instructions_to_sort.contains(operand)) {
        is_processable = false;
        break;
      }
    }

    if (is_processable) {
      processable_instructions.push(instr);
    }
  }

  // If there is no processable instruction, the graph contains a cycle, which
  // should be impossible.
  CHECK(!processable_instructions.empty());

  InstructionSet processed;
  processed.reserve(instructions_to_sort.size());

  while (!processable_instructions.empty()) {
    const HloInstruction* current_instr = processable_instructions.front();
    processable_instructions.pop();

    if (!instructions_to_sort.contains(current_instr)) {
      continue;
    }

    ordered_instructions.push_back(current_instr);
    processed.insert(current_instr);

    for (const HloInstruction* user : current_instr->users()) {
      if (!instructions_to_sort.contains(user)) {
        continue;
      }

      // By construction, there should not be any cycle, so `user` can not have
      // been previously visited.
      CHECK(!processed.contains(user)) << user->ToString();
      bool all_user_operands_were_processed = true;
      for (const HloInstruction* operand : user->operands()) {
        if (!processed.contains(operand) &&
            instructions_to_sort.contains(operand)) {
          all_user_operands_were_processed = false;
        }
      }

      if (all_user_operands_were_processed) {
        processable_instructions.push(user);
      }
    }
  }

  CHECK_EQ(instructions_to_sort.size(), ordered_instructions.size());
  return ordered_instructions;
}

// Returns all the producer instructions judged to be "easy to fuse" for XLA
// according to `CreatesFusionBoundary`. It is possible to set
// `max_number_instructions` to set a bound on the maximum number of
// instructions this function can return. Instructions are processed using a
// breadth-first search and the resulting vector is guaranteed to be
// topologically sorted, i.e., if an instruction i1 is a producer of an
// instruction i2 and both are contained in the result, then i1 appears before
// i2.
std::vector<const HloInstruction*> GatherEasyToFuseProducerInstructions(
    const HloInstruction* consumer,
    size_t max_number_instructions = UINT32_MAX) {
  constexpr size_t reserve_bound = 100;

  std::queue<const HloInstruction*> worklist(std::deque<const HloInstruction*>(
      consumer->operands().begin(), consumer->operands().end()));
  InstructionSet visited;
  InstructionSet fusible;
  visited.reserve(reserve_bound);
  fusible.reserve(reserve_bound);

  while (!worklist.empty() && fusible.size() < max_number_instructions) {
    const HloInstruction* current_instr = worklist.front();
    worklist.pop();

    if (!visited.insert(current_instr).second ||
        CreatesFusionBoundary(current_instr->opcode())) {
      continue;
    }

    fusible.insert(current_instr);

    for (const HloInstruction* operand : current_instr->operands()) {
      worklist.push(operand);
    }
  }

  return SortInstructionSetTopologically(fusible);
}

// Like `GatherEasyToFuseProducerInstructions` but from a producer towards
// users. Here too, the resulting vector is guaranteed to be topologically
// sorted.
std::vector<const HloInstruction*> GatherEasyToFuseUserInstructions(
    const HloInstruction* producer,
    size_t max_number_instructions = UINT32_MAX) {
  // TODO(bchetioui): not sure if the max_number_instructions bound is
  // meaningful at all.
  constexpr size_t reserve_bound = 100;

  // First, traverse the whole graph using a BFS to find out what the set of
  // instructions we want to return is.
  std::queue<const HloInstruction*> worklist(std::deque<const HloInstruction*>(
      producer->users().begin(), producer->users().end()));
  InstructionSet visited;
  InstructionSet fusible;
  visited.reserve(reserve_bound);
  fusible.reserve(reserve_bound);

  while (!worklist.empty() && fusible.size() < max_number_instructions) {
    const HloInstruction* current_instr = worklist.front();
    worklist.pop();

    if (!visited.insert(current_instr).second ||
        CreatesFusionBoundary(current_instr->opcode())) {
      continue;
    }

    fusible.insert(current_instr);

    for (const HloInstruction* user : current_instr->users()) {
      worklist.push(user);
    }
  }

  // Second, create a topologically sorted vector of instructions in order to
  // satisfy the post-ordering property.
  return SortInstructionSetTopologically(fusible);
}

Status InlineFusion(HloFusionInstruction* fusion) {
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      old_to_new_mapping;

  for (const HloInstruction* instr :
       fusion->called_computation()->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kParameter) {
      old_to_new_mapping[instr] =
          fusion->mutable_operand(instr->parameter_number());
    } else {
      std::vector<HloInstruction*> new_operands;
      for (const HloInstruction* operand : instr->operands()) {
        new_operands.push_back(old_to_new_mapping[operand]);
      }
      old_to_new_mapping[instr] = fusion->AddInstruction(
          instr->CloneWithNewOperands(instr->shape(), new_operands));
    }
  }

  TF_RETURN_IF_ERROR(fusion->ReplaceAllUsesWith(
      old_to_new_mapping[fusion->called_computation()->root_instruction()]));

  return OkStatus();
}

Status FuseDiamondChainImpl(const DiamondChainDescriptor& diamond_chain) {
  auto [root, producer] = diamond_chain;

  std::string suggested_name = "triton_softmax";
  HloComputation::Builder builder(absl::StrCat(suggested_name, "_computation"));
  // Original instruction -> fused one.
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      old_to_new_mapping;

  old_to_new_mapping[producer] = builder.AddInstruction(
      HloInstruction::CreateParameter(0, producer->shape(), "parameter_0"));

  std::function<void(const HloInstruction*)> create_computation =
      [&](const HloInstruction* instr) -> void {
    if (old_to_new_mapping.contains(instr)) {
      return;
    }
    std::vector<HloInstruction*> new_operands;
    for (const HloInstruction* operand : instr->operands()) {
      create_computation(operand);
      new_operands.push_back(old_to_new_mapping[operand]);
    }
    old_to_new_mapping[instr] = builder.AddInstruction(
        instr->CloneWithNewOperands(instr->shape(), new_operands));
  };
  create_computation(root);

  HloComputation* computation =
      root->GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                           /*is_entry=*/false);

  HloInstruction* softmax_fusion =
      root->parent()->AddInstruction(HloInstruction::CreateFusion(
          root->shape(), HloInstruction::FusionKind::kCustom,
          std::vector<HloInstruction*>({producer}), computation));

  softmax_fusion->GetModule()->SetAndUniquifyInstrName(softmax_fusion,
                                                       suggested_name);

  TF_ASSIGN_OR_RETURN(auto backend_config,
                      softmax_fusion->backend_config<FusionBackendConfig>());
  backend_config.set_kind(std::string(kTritonSoftmaxFusionKind));
  TF_RETURN_IF_ERROR(softmax_fusion->set_backend_config(backend_config));

  if (root->IsRoot()) {
    root->parent()->set_root_instruction(softmax_fusion);
    TF_RETURN_IF_ERROR(
        root->parent()->RemoveInstructionAndUnusedOperands(root));
  } else {
    TF_RETURN_IF_ERROR(
        root->parent()->ReplaceInstruction(root, softmax_fusion));
  }

  VLOG(5) << softmax_fusion->ToString();
  return OkStatus();
}

StatusOr<HloComputation*> ExtractLocalSurroundingGraphWithFusedDiamondChain(
    const DiamondChainDescriptor& diamond_chain) {
  auto [root, producer] = diamond_chain;

  // We attempt to extract a set of prefix and a set of suffix instructions
  // in order to construct a local module for autotuning.
  std::vector<const HloInstruction*> fusion_prefix =
      GatherEasyToFuseProducerInstructions(producer);

  std::vector<const HloInstruction*> fusion_suffix =
      GatherEasyToFuseUserInstructions(root);

  // Gather producer and all fusion instructions in topological order.
  InstructionSet unsorted_fusion_instructions;
  std::queue<const HloInstruction*> worklist({root});

  while (!worklist.empty()) {
    const HloInstruction* current_instr = worklist.front();
    CHECK_NE(current_instr, producer);
    worklist.pop();

    unsorted_fusion_instructions.insert(current_instr);
    for (const HloInstruction* operand : current_instr->operands()) {
      if (operand != producer) {
        worklist.push(operand);
      }
    }
  }
  unsorted_fusion_instructions.insert(producer);

  std::vector<const HloInstruction*> fusion_instructions =
      SortInstructionSetTopologically(unsorted_fusion_instructions);

  // Create a computation wrapping the desired ops.
  HloComputation::Builder builder(
      absl::StrCat(root->name(), "softmax_autotuning"));

  // Original instruction -> fused one.
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      old_to_new_mapping;

  int parameter_number = 0;

  auto add_to_computation =
      [&](std::vector<const HloInstruction*> instructions) {
        for (const HloInstruction* instr : instructions) {
          std::vector<HloInstruction*> new_operands;
          for (const HloInstruction* operand : instr->operands()) {
            // If an op has not been previously traversed, then it is a
            // parameter to the new computation. This is given by the ordering
            // guarantees of `GatherEasyToFuseUserInstructions` and of
            // `GatherEasyToFuseProducerInstructions`.
            if (!old_to_new_mapping.contains(operand)) {
              old_to_new_mapping[operand] =
                  builder.AddInstruction(HloInstruction::CreateParameter(
                      parameter_number, operand->shape(),
                      absl::StrCat("parameter_", parameter_number)));
              ++parameter_number;
            }

            new_operands.push_back(old_to_new_mapping[operand]);
          }

          // It could be that some of the instructions in the suffix rely on
          // instructions in the prefix, or that the constants used by the
          // fusion's reductions have users in the prefix.
          if (old_to_new_mapping.contains(instr)) {
            continue;
          }

          if (instr->opcode() == HloOpcode::kParameter) {
            old_to_new_mapping[instr] =
                builder.AddInstruction(HloInstruction::CreateParameter(
                    parameter_number, instr->shape(),
                    absl::StrCat("parameter_", parameter_number)));
            ++parameter_number;
          } else {
            old_to_new_mapping[instr] = builder.AddInstruction(
                instr->CloneWithNewOperands(instr->shape(), new_operands));
          }
        }
      };

  // Construct a computation containing instructions as follows:
  // computation {
  //   fusion_prefix...
  //   fusion_instructions ...
  //   fusion_suffix...
  // }
  add_to_computation(fusion_prefix);
  add_to_computation(fusion_instructions);
  add_to_computation(fusion_suffix);

  HloComputation* computation =
      root->GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                           /*is_entry=*/false);

  DiamondChainDescriptor new_diamond_chain = {
      /*root=*/old_to_new_mapping[root],
      /*producer=*/old_to_new_mapping[producer]};

  TF_RET_CHECK(FuseDiamondChainImpl(new_diamond_chain).ok());

  return computation;
}

// Runs the relevant post-layout assignment GPU compiler passes that occur
// after `SoftmaxRewriterTriton`. Because XLA GPU is not able to seamlessly
// process programs that have gone through layout assignment directly from
// `GpuCompiler::RunHloPasses`, we need to reproduce the passes necessary and
// relevant for autotuning-based comparison to be meaningful. The necessary
// passes are those that fix the correctness of the compiled program, allowing
// it to be emitted: this is mostly the float normalization pipelien. For the
// performance to be representative, we mainly need to reproduce here
// XLA GPU's fusion pipeline. The passes separating `SoftmaxRewriterTriton`
// from the rest of the fusion pipeline are also relevant, since they may
// affect how the fusion pipeline affects the computation graph.
//
// Lastly, we also run the relevant correctness-related passes from
// `GpuCompiler::PrepareHloModuleForIrEmitting`.
// TODO(bchetioui): should I reconstruct the whole pipeline?
// TODO(bchetioui): can we expose this pipeline statically?
Status RunPartialOptimizationPipeline(HloModule* hlo_module,
                                      const AutotuneConfig& autotune_config,
                                      se::StreamExecutor* stream_executor) {
  DebugOptions debug_options = hlo_module->config().debug_options();
  debug_options.set_xla_gpu_enable_xla_runtime_executable(false);
  hlo_module->config().set_debug_options(debug_options);

  se::CudaComputeCapability compute_capability =
      stream_executor->GetDeviceDescription().cuda_compute_capability();

  {
    HloPassPipeline pipeline("pre-xla-general-fusion-pipeline");
    pipeline.AddPass<ReductionDimensionGrouper>();
    pipeline.AddPass<HloPassFix<ReductionSplitter>>();
    pipeline.AddPass<HloPassFix<GpuTreeReductionRewriter>>(compute_capability);

    // TODO(bchetioui): make sure this works without a sub-pipeline
    // TODO(bchetioui): add other float norms to this
    // Triton compilation needs normalized operations on bf16 (i.e. converted
    // to f32).
    GpuFloatSupport bf16_support(BF16);
    pipeline.AddPass<FloatNormalization>(&bf16_support);
    if (debug_options.xla_gpu_simplify_all_fp_conversions()) {
      pipeline.AddPass<SimplifyFPConversions>();
    }

    pipeline.AddPass<TupleSimplifier>();
    {
      // The LayoutAssignment pass may leave behind kCopy instructions which
      // are duplicate or NOPs, so remove them with algebraic simplification
      // and CSE.
      AlgebraicSimplifierOptions options;
      options.set_supports_non_canonical_dots(false);
      options.set_is_layout_sensitive(true);
      options.set_enable_conv_operand_swap(false);
      // "slow" minmax means we propagate nan.
      options.set_minmax_propagate_nan(
          !debug_options.xla_gpu_enable_fast_min_max());
      options.set_enable_unconditional_reduce_of_concat_replacement(false);
      pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(options);
    }

    // Since this CSE runs after collective schedule linearizer which inserts
    // control dependencies, ignore these control deps when replacing
    // instructions with equivalent ones here.
    pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true,
                             /*only_fusion_computations=*/false,
                             /*ignore_control_dependencies=*/true);
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  GpuDeviceInfo gpu_device_info =
      GetGpuDeviceInfo(autotune_config.GetExecutor());

  TF_ASSIGN_OR_RETURN(Compiler * compiler,
                      Compiler::GetForPlatform(stream_executor->platform()));

  TF_RETURN_IF_ERROR(FusionPipeline(debug_options,
                                    compiler->ShapeSizeBytesFunction(),
                                    gpu_device_info)
                         .Run(hlo_module)
                         .status());

  TF_RETURN_IF_ERROR(
      HorizontalFusionPipeline(gpu_device_info).Run(hlo_module).status());

  // TODO(bchetioui): perhaps we need to set this to exactly what it should
  // be?
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);

  TF_RETURN_IF_ERROR(AlgebraicSimplifier(options).Run(hlo_module).status());

  // Custom implementation of buffer sharing to avoid a cyclic dependency
  // between this file and gpu_compiler.h.
  auto can_share_buffer = [](const HloInstruction*, const HloInstruction*,
                             const ShapeIndex&) { return false; };

  // Passes necessary to ensure the correctness of the module before
  // emission.
  TF_RETURN_IF_ERROR(
      PrepareHloModuleForIrEmittingPipeline(*hlo_module, can_share_buffer)
          .Run(hlo_module)
          .status());

  return OkStatus();
}

using ProfilingOutput = AutotunerCompileUtil::ProfilingOutput;

// Wrappers around Triton and non-Triton values for type safety.
template <typename ValueType>
struct TritonValueWrapper {
  ValueType value;
};

template <typename ValueType>
struct NoTritonValueWrapper {
  ValueType value;
};

template <typename ValueType>
struct TritonAndNoTriton {
  ValueType triton;
  ValueType no_triton;
};

enum Label : bool { kTriton = 0, kNoTriton = 1 };

template <typename ValueType>
struct Labeled {
  Label label;
  ValueType value;
};

struct ParameterizedExecutable {
  std::vector<Shape> parameter_shapes;
  std::unique_ptr<Executable> executable;
};

using LabeledExecutablesByDiamondChainRoot =
    absl::flat_hash_map<const HloInstruction*,
                        TritonAndNoTriton<ParameterizedExecutable>>;

// TODO(bchetioui): change to DCD here
StatusOr<LabeledExecutablesByDiamondChainRoot>
CompileManyDiamondChainExecutables(
    const std::vector<DiamondChainDescriptor>& diamond_chains,
    AutotunerCompileUtil& autotuner_compile_util,
    const AutotuneConfig& autotune_config,
    tsl::thread::ThreadPool* thread_pool) {
  // TODO(bchetioui): can already be made parallel if necessary

  auto extract_modules = [&](const DiamondChainDescriptor& diamond_chain)
      -> StatusOr<TritonAndNoTriton<std::unique_ptr<HloModule>>> {
    TF_ASSIGN_OR_RETURN(
        HloComputation * computation_with_fusion,
        ExtractLocalSurroundingGraphWithFusedDiamondChain(diamond_chain));

    const HloModule* initial_module = computation_with_fusion->parent();

    // Now, computation contains the result of our Triton Softmax fusion. We
    // create two modules for autotuning, one that contains the fusion, and
    // one from which we will remove it.
    // TODO(bchetioui): setting the config is probably not necessary
    const HloModuleConfig& initial_config = initial_module->config();

    DebugOptions debug_opts(initial_module->config().debug_options());
    debug_opts.set_xla_gpu_enable_triton_softmax_fusion(false);
    HloModuleConfig new_config(initial_config);
    new_config.set_debug_options(debug_opts);

    auto triton_softmax_module = std::make_unique<HloModule>(
        "extracted", new_config,
        std::make_unique<CompilationEnvironments>(
            computation_with_fusion->parent()->comp_envs()));
    HloCloneContext clone_context(triton_softmax_module.get());
    triton_softmax_module->AddEntryComputationWithLayouts(
        computation_with_fusion->CloneInContext(clone_context));

    std::unique_ptr<HloModule> no_triton_softmax_module =
        triton_softmax_module->Clone(/*suffix=*/"");

    // Remove generated computation from the original module.
    TF_RETURN_IF_ERROR(
        computation_with_fusion->parent()->RemoveEmbeddedComputation(
            computation_with_fusion));

    HloInstruction* no_triton_softmax_module_fusion =
        hlo_query::GetFirstInstructionWithOpcode(
            *no_triton_softmax_module->entry_computation(), HloOpcode::kFusion);

    // There should be a single fusion instruction within the module.
    CHECK_NOTNULL(no_triton_softmax_module_fusion);
    CHECK_EQ(
        no_triton_softmax_module_fusion->backend_config<FusionBackendConfig>()
            .value()
            .kind(),
        kTritonSoftmaxFusionKind);

    TF_CHECK_OK(InlineFusion(
        Cast<HloFusionInstruction>(no_triton_softmax_module_fusion)));

    return TritonAndNoTriton<std::unique_ptr<HloModule>>{
        /*triton=*/{std::move(triton_softmax_module)},
        /*no_triton=*/{std::move(no_triton_softmax_module)}};
  };

  // First, generate all the unoptimized modules we want to compile.
  std::vector<
      std::pair<const HloInstruction*, Labeled<std::unique_ptr<HloModule>>>>
      instruction_module_pairs;

  instruction_module_pairs.reserve(diamond_chains.size() * 2);

  using LabeledModule = Labeled<std::unique_ptr<HloModule>>;

  for (const DiamondChainDescriptor& diamond_chain : diamond_chains) {
    TF_ASSIGN_OR_RETURN(TritonAndNoTriton<std::unique_ptr<HloModule>> modules,
                        extract_modules(diamond_chain));
    LabeledModule triton_module{kTriton, std::move(modules.triton)};
    LabeledModule no_triton_module{kNoTriton, std::move(modules.no_triton)};

    instruction_module_pairs.push_back(
        std::make_pair(diamond_chain.root, std::move(triton_module)));
    instruction_module_pairs.push_back(
        std::make_pair(diamond_chain.root, std::move(no_triton_module)));
  }

  // Check that whatever is needed for compilation is available.
  if (autotune_config.IsDeviceless()) {
    // TODO(bchetioui): can not autotune here.
    return InternalError(
        "Expect autotune result cache hit for deviceless compilation.");
  }

  se::StreamExecutor* stream_exec = autotune_config.GetExecutor();
  if (!stream_exec->SynchronizeAllActivity()) {
    return InternalError("Failed to synchronize GPU for autotuning.");
  }

  GpuDeviceInfo gpu_device_info = GetGpuDeviceInfo(stream_exec);

  // Do compilation on parallel if it is possible.
  // TODO(bchetioui): add single-thread special path.

  absl::Mutex executable_map_mutex;
  LabeledExecutablesByDiamondChainRoot executable_map;

  auto compile_module = [&](std::unique_ptr<HloModule> module)
      -> StatusOr<std::unique_ptr<Executable>> {
    return autotuner_compile_util.Compile(
        [&](const DebugOptions&) -> StatusOr<std::unique_ptr<HloModule>> {
          // Reduce memory usage during compilation by disabling GPU runtime.
          TF_CHECK_OK(RunPartialOptimizationPipeline(
              module.get(), autotune_config, stream_exec));
          return std::move(module);
        });
  };

  LOG(INFO) << "Compiling " << instruction_module_pairs.size() << " modules "
            << "on " << thread_pool->NumThreads() << " threads for diamond "
            << "chain autotuning.";
  // Use a thread pool to compile in parallel.
  tsl::BlockingCounter counter(instruction_module_pairs.size());
  for (auto& instruction_module : instruction_module_pairs) {
    thread_pool->Schedule([&]() -> void {
      const HloInstruction* root_instruction = instruction_module.first;
      LabeledModule& labeled_module = instruction_module.second;
      Label label = labeled_module.label;

      absl::Span<const HloInstruction* const> parameters =
          labeled_module.value->entry_computation()->parameter_instructions();

      std::vector<Shape> parameter_shapes;
      parameter_shapes.reserve(parameters.size());

      for (const HloInstruction* parameter : parameters) {
        parameter_shapes.push_back(parameter->shape());
      }

      // TF_ASSERT_OK_AND_ASSIGN is only accessible by importing gunit, which
      // we do not want to do.
      StatusOr<std::unique_ptr<Executable>> status_or_executable =
          compile_module(std::move(labeled_module.value));
      CHECK_OK(status_or_executable.status());

      std::unique_ptr<Executable> executable =
          std::move(status_or_executable).value();

      // TODO(bchetioui): check if actually this is OK in some cases
      CHECK(executable != nullptr);

      executable_map_mutex.Lock();
      if (label == kTriton) {
        executable_map[root_instruction].triton =
            ParameterizedExecutable{parameter_shapes, std::move(executable)};
      } else {
        executable_map[root_instruction].no_triton =
            ParameterizedExecutable{parameter_shapes, std::move(executable)};
      }
      executable_map_mutex.Unlock();

      counter.DecrementCount();
    });
  }
  counter.Wait();

  LOG(INFO) << "Done compiling.";

  return executable_map;
}

StatusOr<std::vector<se::DeviceMemoryBase>> AllocateBuffers(
    const AutotuneConfig& autotune_config, se::RedzoneAllocator& rz_allocator,
    absl::Span<const Shape> shapes) {
  std::vector<se::DeviceMemoryBase> buffers;
  buffers.reserve(shapes.size());

  int64_t rng_state = 0;

  for (const Shape& shape : shapes) {
    TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase buffer,
                        AutotunerUtil::CreateBuffer(
                            rz_allocator, shape, autotune_config, rng_state));
    CHECK_NE(buffer.opaque(), nullptr);
    buffers.push_back(std::move(buffer));
  }

  return buffers;
}

StatusOr<std::optional<ProfilingOutput>> ProfileModuleWithInputs(
    const ParameterizedExecutable& parameterized_executable,
    absl::Span<se::DeviceMemoryBase const> inputs,
    const AutotuneConfig& autotune_config,
    AutotunerCompileUtil& autotuner_compile_util, uint32_t repeat = 10) {
  se::StreamExecutor* stream_exec = autotune_config.GetExecutor();
  // if (!stream_exec->SynchronizeAllActivity()) {
  //   return InternalError("Failed to synchronize GPU for autotuning.");
  // }
  CHECK_NE(stream_exec, nullptr);

  TF_ASSIGN_OR_RETURN(
      se::Stream* const stream,
      stream_exec->GetAllocator()->GetStream(stream_exec->device_ordinal()));
  CHECK_NE(stream, nullptr);

  std::optional<ProfilingOutput> best_profiling_output = std::nullopt;

  for (uint32_t run = 0; run < repeat; ++run) {
    TF_ASSIGN_OR_RETURN(std::optional<ProfilingOutput> profiling_output,
                        autotuner_compile_util.ProfileExecutable(
                            parameterized_executable.executable.get(), stream,
                            inputs, parameterized_executable.parameter_shapes));

    if (profiling_output.has_value()) {
      if (!best_profiling_output.has_value()) {
        best_profiling_output = std::move(profiling_output);
      } else if (profiling_output->duration < best_profiling_output->duration) {
        best_profiling_output = std::move(profiling_output);
      }

      // Deflake runs are only really useful for runs where varying by a few μs
      // could significantly affect runtime. We pick a (somewhat) arbitrary
      // upper bound for deflaking at 500 μs, since even a discrepancy as high
      // as 10 μs can only ever affect total runtime by 2%, which is usually
      // within the range of acceptable noise.
      // TODO(bchetioui): we comment this out due to some bug after syncing
      // that seems to cause a lot of noise. Have to investigate.
      // if (absl::Microseconds(500) <= best_profiling_output.value().duration
      // &&
      //     run > 1) {
      //   break;
      // }
    }
  }

  return best_profiling_output;
}

using DiamondDescriptor = DiamondChainDescriptor;

}  // anonymous namespace

std::vector<DiamondChainDescriptor>
SoftmaxRewriterTriton::FindAllFusibleDiamondChains(
    HloModule& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) const {
  std::vector<DiamondDescriptor> matched_diamonds;

  for (HloComputation* comp :
       module.MakeNonfusionComputations(execution_threads)) {
    if (comp->IsCustomCallComputation()) {
      continue;
    }
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      PrimitiveType element_ty = instr->shape().element_type();
      // TODO(b/281980675): ensure that code generation also works well for FP8
      // and BF16. This fails for the moment due to these data types requiring
      // float normalization.
      if (element_ty != F16 && element_ty != F32 && element_ty != BF16) {
        continue;
      }

      if (auto producer = MatchesTritonCompatibleClosedReductionDiamond(
              instr, gpu_version_)) {
        matched_diamonds.push_back(DiamondDescriptor{instr, producer.value()});
      }
    }
  }

  if (matched_diamonds.empty()) {
    return {};
  }

  auto reduction_dimension_size_from_diamond_root =
      [](HloInstruction* diamond_root) {
        HloInstruction* instr = diamond_root->mutable_operand(1);
        while (instr->opcode() != HloOpcode::kReduce) {
          instr = instr->mutable_operand(0);
        }

        int operand_rank = instr->operand(0)->shape().rank();
        CHECK_EQ(instr->dimensions().size(), 1);
        CHECK_EQ(instr->dimensions(0), operand_rank - 1);
        return instr->operand(0)->shape().dimensions(operand_rank - 1);
      };

  auto last_trivially_fusible_user = [&](HloInstruction* instr) {
    while (HasOneUse(instr) && !instr->IsRoot() &&
           IsTriviallyFusible(instr->users().front(), gpu_version_)) {
      instr = instr->users().front();
    }

    // We do not care about the number of users for the last instruction of the
    // fusion, so attempt to fuse one more instruction with this relaxed
    // restriction.
    if (HasOneUse(instr) && !instr->IsRoot() &&
        IsTriviallyFusible(
            instr->users().front(), gpu_version_,
            /*num_allowed_users=*/instr->users().front()->user_count())) {
      instr = instr->users().front();
    }
    return instr;
  };

  // If we matched several diamonds, it may be possible for some of them to be
  // fused together. This is the case if the following conditions hold:
  //   1. The path between the root of diamond n towards the producer of
  //      diamond n+1 is composed only of trivially fusible operations. In that
  //      case, the first non-trivially fusible producer of diamond n+1 must be
  //      exactly the root of diamond n.
  //   2. The root of diamond n/first non-fusible producer of diamond n+1 must
  //      have
  //        a. exactly one user if it is not exactly the producer of diamond
  //           n+1;
  //        b/ exactly two users otherwise.
  //   3. The axis being reduced must have the same length in all the diamonds
  //      being fused together.
  //
  // Crucially, this approach relies on a diamond root never being considered a
  // trivially fusible operation.
  std::vector<DiamondChainDescriptor> diamond_chains;
  diamond_chains.reserve(matched_diamonds.size());

  HloInstruction* current_fusion_producer = FindFirstNonFusibleDiamondProducer(
      matched_diamonds.front().producer, gpu_version_);
  int current_reduce_dimension_size =
      reduction_dimension_size_from_diamond_root(matched_diamonds.front().root);

  for (int diamond_idx = 1; diamond_idx < matched_diamonds.size();
       ++diamond_idx) {
    auto [diamond_root, diamond_producer] = matched_diamonds[diamond_idx];
    HloInstruction* previous_diamond_root =
        matched_diamonds[diamond_idx - 1].root;

    HloInstruction* first_non_fusible_diamond_producer =
        FindFirstNonFusibleDiamondProducer(diamond_producer, gpu_version_);

    int diamond_reduce_dimension_size =
        reduction_dimension_size_from_diamond_root(diamond_root);

    if (first_non_fusible_diamond_producer == previous_diamond_root &&  // 1
        ((first_non_fusible_diamond_producer != diamond_producer &&
          HasOneUse(first_non_fusible_diamond_producer)) ||  // 2.a
         (first_non_fusible_diamond_producer == diamond_producer &&
          first_non_fusible_diamond_producer->user_count() == 2)) &&  // 2.b
        diamond_reduce_dimension_size == current_reduce_dimension_size) {  // 3
      continue;
    }

    // The "last trivially fusible user" chain of diamond chain n should never
    // intersect with the "first non fusible diamond producer" chain of diamond
    // chain n+1: if these chains intersected, then all the intermediate ops
    // between the diamond chains could be trivially fused, and both diamond
    // chains could be fused into a single diamond chain. Note that this only
    // holds insofar as we do not allow fusing in bitcasts that modify the last
    // dimension of the input array. It is however possible for the last
    // trivially fusible user of diamond chain n to be the first non fusible
    // diamond producer of diamond chain n+1.
    diamond_chains.push_back(DiamondChainDescriptor{
        last_trivially_fusible_user(previous_diamond_root),
        current_fusion_producer});

    current_fusion_producer = first_non_fusible_diamond_producer;
    current_reduce_dimension_size = diamond_reduce_dimension_size;
  }

  // The last diamond chain is still open; close it.
  diamond_chains.push_back(DiamondChainDescriptor{
      last_trivially_fusible_user(matched_diamonds.back().root),
      current_fusion_producer});

  return diamond_chains;
}

Status SoftmaxRewriterTriton::FuseDiamondChain(
    const DiamondChainDescriptor& diamond_chain) {
  return FuseDiamondChainImpl(diamond_chain);
}

StatusOr<bool> SoftmaxRewriterTriton::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // SoftmaxRewriterTriton is conservative, and does not perform softmax fusions
  // when autotuning can't be done.
  if (!autotune_config_.has_value() ||
      autotune_config_.value().IsDeviceless()) {
    return false;
  }

  // Construct a new autotune config that sets the autotune level back to 1 if
  // it is higher. We do not implement different autotuning levels here at this
  // point, this is thus useful to save compilation time.
  const DebugOptions& initial_debug_options = module->config().debug_options();

  DebugOptions debug_opts(initial_debug_options);
  debug_opts.set_xla_gpu_autotune_level(
      initial_debug_options.xla_gpu_autotune_level() > 1
          ? 1
          : initial_debug_options.xla_gpu_autotune_level());
  debug_opts.set_xla_gpu_enable_xla_runtime_executable(false);

  se::StreamExecutor* stream_exec = autotune_config_.value().GetExecutor();
  AutotuneConfig autotune_config{
      DeviceConfig{stream_exec, stream_exec->GetAllocator()}, debug_opts};

  std::vector<DiamondChainDescriptor> diamond_chains =
      FindAllFusibleDiamondChains(*module, execution_threads);

  if (diamond_chains.empty()) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(
      std::optional<AutotunerCompileUtil> autotuner_compile_util,
      AutotunerCompileUtil::Create(autotune_config, debug_opts));

  // TODO(bchetioui): is that right?
  CHECK(autotuner_compile_util.has_value());

  TF_ASSIGN_OR_RETURN(
      LabeledExecutablesByDiamondChainRoot executables_by_diamond_chain_root,
      CompileManyDiamondChainExecutables(diamond_chains,
                                         autotuner_compile_util.value(),
                                         autotune_config, thread_pool_));

  LOG(INFO) << "Profiling " << diamond_chains.size() << " diamond chains";

  bool has_changed = false;

  if (!stream_exec->SynchronizeAllActivity()) {
    return InternalError("Failed to synchronize GPU for autotuning.");
  }

  se::DeviceMemoryAllocator* allocator = stream_exec->GetAllocator();
  CHECK_NE(allocator, nullptr);

  TF_ASSIGN_OR_RETURN(se::Stream* const stream,
                      allocator->GetStream(stream_exec->device_ordinal()));
  CHECK_NE(stream, nullptr);

  // The diamond chains must be emitted in reverse order, to make sure that
  // producer instructions are emitted correctly when the root of
  // diamond chain n is exactly the producer of diamond chain n+1.
  for (auto diamond_chain = diamond_chains.rbegin();
       diamond_chain != diamond_chains.rend(); ++diamond_chain) {
    CHECK(executables_by_diamond_chain_root.contains(diamond_chain->root));

    // Profile both executables, and fuse if profitable.
    TritonAndNoTriton<ParameterizedExecutable> executables =
        std::move(executables_by_diamond_chain_root[diamond_chain->root]);

    // Create allocator in the loop in order for the buffers to be automatically
    // freed after each loop iteration.
    TF_ASSIGN_OR_RETURN(
        se::RedzoneAllocator rz_allocator,
        AutotunerUtil::CreateRedzoneAllocator(autotune_config, debug_opts,
                                              /*force_stream=*/stream));

    TF_ASSIGN_OR_RETURN(std::vector<se::DeviceMemoryBase> inputs,
                        AllocateBuffers(autotune_config, rz_allocator,
                                        executables.triton.parameter_shapes));

    TF_ASSIGN_OR_RETURN(
        std::optional<ProfilingOutput> triton_profile,
        ProfileModuleWithInputs(executables.triton, inputs, autotune_config,
                                autotuner_compile_util.value()));

    TF_ASSIGN_OR_RETURN(
        std::optional<ProfilingOutput> no_triton_profile,
        ProfileModuleWithInputs(executables.no_triton, inputs, autotune_config,
                                autotuner_compile_util.value()));

    LOG(INFO) << "Finished diamond profiling round";
    bool profiling_succeeded =
        triton_profile.has_value() && no_triton_profile.has_value();

    if (!profiling_succeeded) {
      // If profiling failed one way or the other, we're not able to make sure
      // that Softmax fusion is worthwhile. In that case, let other XLA fusion
      // passes take the wheel.
      VLOG(2) << "Profiling failed for " << diamond_chain->root;
    } else {
      LOG(INFO) << "Runtime was " << no_triton_profile->duration << " against "
                << triton_profile->duration;

      // Fusion seems to be worthwhile, let's do it!
      if (triton_profile->duration < no_triton_profile->duration) {
        TF_RET_CHECK(FuseDiamondChain(*diamond_chain).ok());
        has_changed = true;
      }
    }
  }

  return has_changed;
}
}  // namespace xla::gpu
