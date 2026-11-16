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

#include "xla/backends/gpu/transforms/horizontal_fusion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/codegen/thread_block_specialized.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

// Maximum total kernel arguments (deduplicated inputs + outputs).
//
// Every argument is a 64-bit buffer pointer in the kernel parameter space,
// which CUDA limits to 4 KiB (32 KiB on Volta+ with CUDA >= 12.1), so 64
// arguments (512 bytes) are far from the hard limit. The cap instead bounds the
// per-launch argument marshalling cost and, more importantly, the number of
// buffers whose live ranges are merged: buffer assignment keeps every operand
// and output of the outer fusion live for the whole kernel, so very wide
// horizontal fusions increase peak memory.
constexpr int64_t kMaxKernelArgs = 64;

// Conservative per-thread register budgets used to estimate how many threads of
// a horizontally fused kernel can be resident on one SM at the same time.
//
// The fused kernel's register allocation is the maximum over all nested leaf
// kernels, and every Ampere (sm_8x), Hopper (sm_90) and Blackwell
// (sm_10x/sm_12x) SM has a 64K-entry 32-bit register file.
//
// `kEstimatedRegistersPerThread`: Triton (`kCustom`) and `kInput`
// reduction/transpose kernels with 4-8 warps use up to ~64 registers per thread
// (row accumulators, shared-memory tile addressing), allowing
// 65536 / 64 = 1024 resident threads per SM, half of the 2048-thread hardware
// limit on A100/H100/B200.
//
// `kEstimatedLoopRegistersPerThread`: pure `kLoop` fusions emitted by XLA's
// loop emitter have no barriers, shared-memory tiles or cross-thread
// accumulators; each thread computes a few independent elements, which
// typically compiles within 32 registers per thread and allows the full
// 65536 / 32 = 2048 resident threads per SM. This is the less conservative
// budget: heavily unrolled or transcendental-heavy loop kernels can exceed it,
// in which case part of the grid runs as a second, partial wave. The cost of
// that is bounded because every leaf is launch-bound
// (`Options::max_leaf_exec_time`).
//
// Overestimating register usage only makes the single-wave guard stricter
// (fewer, smaller horizontal fusions), whereas underestimating it could spill
// into a second wave whose tail latency exceeds the saved launches.
constexpr int64_t kEstimatedRegistersPerThread = 64;
constexpr int64_t kEstimatedLoopRegistersPerThread = 32;

// Maximum number of cross-barrier small-output batches kept open at once (see
// `Options::max_cross_barrier_output_bytes`). Every isolated tiny fusion is
// tried against each open batch, so this bounds the extra compile time to
// O(kMaxOpenSmallBatches) `TryAddToBatch` calls per fusion. It also bounds how
// far back in the schedule a tiny fusion can be moved, because the oldest batch
// is closed when a new one is opened. 8 matches the default
// `Options::max_nested_kernels`, so in the common case of one isolated tiny
// fusion per layer, a batch can fill up before it is evicted.
constexpr int64_t kMaxOpenSmallBatches = 8;

struct FusionBatch {
  std::vector<HloInstruction*> instructions;
  absl::flat_hash_set<const HloInstruction*> unique_operands;
  int64_t total_blocks = 0;
  int64_t max_threads_per_block = 0;
  int64_t total_outputs = 0;
  // Whether any member is not a `kLoop` fusion, which selects the stricter
  // register budget for the residency check.
  bool has_non_loop_fusion = false;
};

// A batch of tiny fusions that may be separated by non-fusible instructions.
// The batch is emitted at the schedule position of its first member,
// `anchor_pos`.
struct SmallFusionBatch {
  FusionBatch batch;
  int64_t anchor_pos = 0;
};

// Returns a conservative estimate of the number of threads of a horizontally
// fused kernel that can be resident on one SM at the same time.
//
// This is the hardware thread limit (`threads_per_core_limit()`: 2048 on
// A100/H100/B200, 1536 on GA10x/AD10x), capped by the register file at
// `estimated_registers_per_thread` registers per thread. The result is never
// below `threads_per_block_limit()` (1024 on all CUDA GPUs): a kernel launched
// with maximum-sized blocks must fit at least one such block on an SM to be
// launchable at all. The same value is used as a fallback when the device
// description does not provide `threads_per_core_limit()`.
int64_t ResidentThreadsPerSm(const se::DeviceDescription& device_info,
                             int64_t estimated_registers_per_thread) {
  const int64_t min_resident_threads = device_info.threads_per_block_limit();
  int64_t resident_threads = device_info.threads_per_core_limit() > 0
                                 ? device_info.threads_per_core_limit()
                                 : min_resident_threads;
  if (device_info.registers_per_core_limit() > 0) {
    const int64_t reg_limited_threads =
        device_info.registers_per_core_limit() / estimated_registers_per_thread;
    resident_threads = std::min<int64_t>(resident_threads, reg_limited_threads);
  }
  return std::max<int64_t>(resident_threads, min_resident_threads);
}

// Returns true if `fusion` runs on the default compute stream. Fusions that are
// explicitly assigned to another stream (via the `_xla_stream_annotation`
// frontend attribute or a non-zero `operation_queue_id`) must not be merged
// with default-stream fusions, as the outer fusion would run on a single
// stream and lose the requested concurrency.
bool UsesDefaultExecutionStream(const HloFusionInstruction& fusion) {
  auto stream_it =
      fusion.frontend_attributes().map().find(kXlaStreamAnnotationAttr);
  if (stream_it != fusion.frontend_attributes().map().end() &&
      !stream_it->second.empty() && stream_it->second != "0") {
    return false;
  }
  if (absl::StatusOr<GpuBackendConfig> gpu_config =
          fusion.backend_config<GpuBackendConfig>();
      gpu_config.ok() && gpu_config->operation_queue_id() != 0) {
    return false;
  }
  return true;
}

bool IsZeroCostOp(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kConstant:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
      return true;
    default:
      return false;
  }
}

std::optional<LaunchDimensions> GetLaunchDimensions(
    const HloInstruction* instr, const se::DeviceDescription& device_info) {
  const auto* fusion = DynCast<HloFusionInstruction>(instr);
  if (fusion == nullptr) {
    return std::nullopt;
  }

  // Dead, side-effecting, aliased, dynamic, tuple-operand, or non-default
  // stream fusions are not supported.
  if (fusion->IsDead() || fusion->HasSideEffect() ||
      !fusion->output_to_operand_aliasing().empty() ||
      !UsesDefaultExecutionStream(*fusion) ||
      absl::c_any_of(
          fusion->operands(),
          [](const HloInstruction* op) { return op->shape().IsTuple(); }) ||
      absl::c_any_of(fusion->fused_instructions(),
                     [](const HloInstruction* hlo) {
                       return hlo->shape().is_dynamic();
                     })) {
    return std::nullopt;
  }

  // Nested tuple outputs (e.g. a variadic reduce that is one root of a
  // multi-output fusion) are not supported: the outer fusion returns a flat
  // tuple with one element per nested fusion output.
  if (ShapeUtil::IsNestedTuple(fusion->shape())) {
    return std::nullopt;
  }

  // Custom fusions are only supported if they are generic Triton fusions that
  // can be emitted as nested leaf kernels. Triton GEMMs (dot, scaled-dot) and
  // transposes are excluded: GEMMs are compute-bound rather than launch-bound,
  // and both stage tiles through large shared-memory buffers, which would be
  // allocated for the whole outer kernel and reduce its occupancy.
  if (fusion->fusion_kind() == HloInstruction::FusionKind::kCustom) {
    if (!IsGenericTritonFusion(*fusion) ||
        absl::c_any_of(fusion->fused_instructions(),
                       HloPredicateIsOp<HloOpcode::kDot, HloOpcode::kScaledDot,
                                        HloOpcode::kTranspose>)) {
      return std::nullopt;
    }
    return ThreadBlockSpecializedFusion::GetNestedTritonLaunchDimensions(
        HloFusionAnalysis::Create(*fusion, device_info));
  }

  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*fusion, device_info);

  PreBufferAssignmentFusionInfo fusion_info(analysis);
  // Trivial copy and dynamic update slice fusions are not supported.
  if (fusion_info.GetCopyFusion().has_value() ||
      (IsDynamicUpdateSliceFusion(analysis.fusion_spec()) &&
       fusion_info.CanEmitDynamicUpdateSliceInPlace())) {
    return std::nullopt;
  }

  // TODO(b/480040242): Consider unrolling small fusions in grid-bound
  // horizontal fusions as well.
  std::unique_ptr<MlirKernelEmitter> emitter =
      ThreadBlockSpecializedFusion::CreateNativeNestedEmitter(
          analysis, /*unroll_small_fusions=*/false);
  if (emitter == nullptr) {
    return std::nullopt;
  }
  LaunchDimensions launch_dims = emitter->launch_dimensions();

  // If a nested kernel has less than max_threads and uses a barrier, we need to
  // use `bar.sync 0, T_i` in PTX. The instruction has a requirement that T_i
  // must be a multiple of the warp size.
  //
  // Only loop emitter can have a count of threads that is not a multiple of the
  // warp size, but we also know that loop emitter never uses barriers.
  if (launch_dims.num_threads_per_block() % WarpSize(device_info) != 0 &&
      analysis.emitter_fusion_kind() !=
          HloFusionAnalysis::EmitterFusionKind::kLoop) {
    return std::nullopt;
  }

  return launch_dims;
}

bool TryAddToBatch(
    FusionBatch& batch, HloInstruction* candidate,
    const HloReachabilityMap& reachability,
    const se::DeviceDescription& device_info,
    const HorizontalFusion::Options& options,
    absl::FunctionRef<std::optional<LaunchDimensions>(const HloInstruction*)>
        get_launch_dims,
    absl::FunctionRef<absl::Duration(const HloInstruction*)> estimate_runtime) {
  if (batch.instructions.size() >= options.max_nested_kernels) {
    return false;
  }

  if (absl::c_any_of(batch.instructions, [&](const HloInstruction* batch_inst) {
        return reachability.IsReachable(batch_inst, candidate);
      })) {
    return false;
  }

  std::optional<LaunchDimensions> candidate_dims = get_launch_dims(candidate);
  if (!candidate_dims.has_value()) {
    return false;
  }

  const int64_t total_blocks =
      batch.total_blocks + candidate_dims->num_blocks();
  const int64_t max_threads_per_block = std::max<int64_t>(
      batch.max_threads_per_block, candidate_dims->num_threads_per_block());
  const bool has_non_loop_fusion =
      batch.has_non_loop_fusion ||
      candidate->fusion_kind() != HloInstruction::FusionKind::kLoop;
  // The outer kernel launches `total_blocks` blocks of `max_threads_per_block`
  // threads each. Require that they all fit in `max_waves` resident waves so
  // that nested kernels do not serialize behind each other.
  if (options.max_waves > 0 && device_info.core_count() > 0) {
    const int64_t estimated_registers = has_non_loop_fusion
                                            ? kEstimatedRegistersPerThread
                                            : kEstimatedLoopRegistersPerThread;
    const int64_t max_resident_threads =
        options.max_waves * device_info.core_count() *
        ResidentThreadsPerSm(device_info, estimated_registers);
    if (total_blocks * max_threads_per_block > max_resident_threads) {
      return false;
    }
  }

  // Count deduplicated outer operands + outputs.
  absl::flat_hash_set<const HloInstruction*> unique_operands =
      batch.unique_operands;
  unique_operands.insert(candidate->operands().begin(),
                         candidate->operands().end());

  const int64_t num_outputs =
      batch.total_outputs + ShapeUtil::GetLeafCount(candidate->shape());
  if (unique_operands.size() + num_outputs > kMaxKernelArgs) {
    return false;
  }

  // Checked last because it is the most expensive check.
  if (options.max_leaf_exec_time > absl::ZeroDuration() &&
      estimate_runtime(candidate) >= options.max_leaf_exec_time) {
    return false;
  }

  batch.instructions.push_back(candidate);
  batch.unique_operands = std::move(unique_operands);
  batch.total_blocks = total_blocks;
  batch.max_threads_per_block = max_threads_per_block;
  batch.total_outputs = num_outputs;
  batch.has_non_loop_fusion = has_non_loop_fusion;
  return true;
}

absl::StatusOr<std::vector<HloInstruction*>> FuseCandidates(
    HloModule* module, HloComputation* comp,
    absl::Span<HloInstruction* const> candidates) {
  HloComputation::Builder builder("horizontal_fusion_computation");
  std::vector<HloInstruction*> outer_operands;
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> operand_to_param;
  std::vector<HloInstruction*> tuple_elements;

  for (HloInstruction* nested_fusion : candidates) {
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(nested_fusion->operand_count());
    for (HloInstruction* operand : nested_fusion->mutable_operands()) {
      auto [it, inserted] = operand_to_param.try_emplace(operand, nullptr);
      if (inserted) {
        it->second = builder.AddInstruction(HloInstruction::CreateParameter(
            outer_operands.size(), operand->shape(),
            absl::StrCat("param_", outer_operands.size())));
        outer_operands.push_back(operand);
      }
      new_operands.push_back(it->second);
    }

    HloInstruction* cloned_fusion =
        builder.AddInstruction(nested_fusion->CloneWithNewOperands(
            nested_fusion->shape(), new_operands, /*suffix=*/""));
    if (nested_fusion->shape().IsTuple()) {
      for (int64_t i = 0; i < nested_fusion->shape().tuple_shapes_size(); ++i) {
        tuple_elements.push_back(builder.AddInstruction(
            HloInstruction::CreateGetTupleElement(cloned_fusion, i)));
      }
    } else {
      tuple_elements.push_back(cloned_fusion);
    }
  }

  HloInstruction* root_tuple =
      builder.AddInstruction(HloInstruction::CreateTuple(tuple_elements));
  HloComputation* outer_computation =
      module->AddEmbeddedComputation(builder.Build(root_tuple));
  HloInstruction* outer_fusion = comp->AddInstruction(
      HloInstruction::CreateFusion(root_tuple->shape(),
                                   HloInstruction::FusionKind::kCustom,
                                   outer_operands, outer_computation),
      "horizontal_fusion");
  outer_fusion->set_metadata(candidates.front()->metadata());
  std::vector<HloInstruction*> emitted_instructions = {outer_fusion};

  GpuBackendConfig gpu_backend_config;
  gpu_backend_config.mutable_fusion_backend_config()->set_kind(
      kThreadBlockSpecializedFusionKind);
  ABSL_RETURN_IF_ERROR(outer_fusion->set_backend_config(gpu_backend_config));

  int64_t output_idx = 0;
  for (HloInstruction* nested_fusion : candidates) {
    HloComputation* comp_i = nested_fusion->fused_instructions_computation();
    ABSL_RETURN_IF_ERROR(outer_fusion->CopyAllControlDepsFrom(nested_fusion));
    ABSL_RETURN_IF_ERROR(nested_fusion->DropAllControlDeps());
    if (nested_fusion->shape().IsTuple()) {
      // Redirect GetTupleElement users to the new outer fusion.
      for (HloInstruction* user : std::vector<HloInstruction*>(
               nested_fusion->users().begin(), nested_fusion->users().end())) {
        if (user->opcode() != HloOpcode::kGetTupleElement) {
          continue;
        }
        user->set_tuple_index(output_idx + user->tuple_index());
        ABSL_RETURN_IF_ERROR(
            nested_fusion->ReplaceUseWithDifferentShape(user, outer_fusion));
      }

      // If there are still users left, then the tuple output was used as a
      // whole. Replace it with a new tuple create with gtes of the outer
      // fusion.
      if (nested_fusion->user_count() > 0 || nested_fusion->IsRoot()) {
        int64_t num_outputs = nested_fusion->shape().tuple_shapes().size();
        std::vector<HloInstruction*> new_gtes;
        new_gtes.reserve(num_outputs);
        // It's intentional that we create new gtes here instead of reusing the
        // one created above. Potentially, gte can have scheduling predecessor
        // annotation that we don't want to inherit.
        for (int64_t i = 0; i < num_outputs; ++i) {
          HloInstruction* new_gte =
              comp->AddInstruction(HloInstruction::CreateGetTupleElement(
                  outer_fusion, output_idx + i));
          new_gtes.push_back(new_gte);
          emitted_instructions.push_back(new_gte);
        }
        HloInstruction* new_tuple =
            comp->AddInstruction(HloInstruction::CreateTuple(new_gtes));
        emitted_instructions.push_back(new_tuple);
        ABSL_RETURN_IF_ERROR(nested_fusion->ReplaceAllUsesWith(new_tuple));
      }
    } else {
      HloInstruction* new_gte = comp->AddInstruction(
          HloInstruction::CreateGetTupleElement(outer_fusion, output_idx));
      emitted_instructions.push_back(new_gte);
      ABSL_RETURN_IF_ERROR(nested_fusion->ReplaceAllUsesWith(new_gte));
    }
    output_idx += ShapeUtil::GetLeafCount(nested_fusion->shape());

    ABSL_RETURN_IF_ERROR(comp->RemoveInstruction(nested_fusion));
    ABSL_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(comp_i));
  }

  return emitted_instructions;
}

// Lazily estimates the standalone execution time (excluding launch overhead)
// of candidate leaf fusions in one computation. Nested Triton fusions use the
// tiled indexing performance model; all other fusions use the analytical
// `GpuPerformanceModel`. The models are only built on first use, so
// computations without candidates do not pay for them.
class LeafRuntimeEstimator {
 public:
  LeafRuntimeEstimator(const HloComputation* absl_nonnull comp,
                       const se::DeviceDescription* absl_nonnull device_info,
                       bool enable_same_shape_mof)
      : comp_(comp),
        device_info_(device_info),
        enable_same_shape_mof_(enable_same_shape_mof) {}

  // Returns `absl::InfiniteDuration()` if the estimate is unavailable, so that
  // the candidate is rejected.
  absl::Duration EstimateExecTime(const HloInstruction* instr) {
    if (auto it = exec_time_cache_.find(instr); it != exec_time_cache_.end()) {
      return it->second;
    }
    absl::Duration exec_time = ComputeExecTime(instr);
    exec_time_cache_.emplace(instr, exec_time);
    return exec_time;
  }

  // Returns the status of the cost analysis of the computation, if it ran.
  const absl::Status& status() const { return status_; }

 private:
  absl::Duration ComputeExecTime(const HloInstruction* instr) {
    if (instr->opcode() == HloOpcode::kFusion &&
        instr->fusion_kind() == HloInstruction::FusionKind::kCustom) {
      return EstimateTritonExecTime(instr);
    }
    if (cost_analysis_ == nullptr) {
      GpuHloCostAnalysis::Options cost_opts{
          HloCostAnalysis::DefaultShapeSize,
          /*per_second_rates=*/{},
          /*min_latencies_seconds=*/{},
          /*count_multiple_input_accesses=*/true};
      cost_analysis_ =
          std::make_unique<GpuHloCostAnalysis>(cost_opts, *device_info_);
      status_ = comp_->Accept(cost_analysis_.get());
      perf_model_ = std::make_unique<GpuPerformanceModelOwning>(*device_info_);
    }
    if (!status_.ok()) {
      return absl::InfiniteDuration();
    }
    return perf_model_->Get()
        .EstimateRunTimeForInstruction(instr, cost_analysis_.get())
        .exec_time;
  }

  absl::Duration EstimateTritonExecTime(const HloInstruction* instr) {
    if (indexing_perf_model_ == nullptr) {
      mlir_context_ = std::make_unique<mlir::MLIRContext>(
          mlir::MLIRContext::Threading::DISABLED);
      RegisterSymbolicExprStorage(mlir_context_.get());
      fusion_analysis_cache_ =
          std::make_unique<HloFusionAnalysisCache>(*device_info_);
      indexing_perf_model_ =
          std::make_unique<GpuPerformanceModelWithIndexingAnalysis>(
              device_info_, fusion_analysis_cache_.get(),
              HloCostAnalysis::DefaultShapeSize, mlir_context_.get(),
              enable_same_shape_mof_);
    }
    absl::StatusOr<EstimateRunTimeData> runtime =
        indexing_perf_model_->EstimateRunTimeForTriton(instr);
    return runtime.ok() ? runtime->exec_time : absl::InfiniteDuration();
  }

  const HloComputation* absl_nonnull comp_;
  const se::DeviceDescription* absl_nonnull device_info_;
  bool enable_same_shape_mof_;

  absl::flat_hash_map<const HloInstruction*, absl::Duration> exec_time_cache_;
  std::unique_ptr<mlir::MLIRContext> mlir_context_;
  std::unique_ptr<HloFusionAnalysisCache> fusion_analysis_cache_;
  std::unique_ptr<GpuPerformanceModelWithIndexingAnalysis> indexing_perf_model_;
  std::unique_ptr<GpuHloCostAnalysis> cost_analysis_;
  std::unique_ptr<GpuPerformanceModelOwning> perf_model_;
  absl::Status status_;
};

using InstructionPosMap = absl::flat_hash_map<const HloInstruction*, int64_t>;
using CompInstructionPosMap =
    absl::flat_hash_map<const HloComputation*, InstructionPosMap>;
using WarPredecessorMap =
    absl::flat_hash_map<const HloInstruction*,
                        std::vector<const HloInstruction*>>;
using CompWarPredecessorMap =
    absl::flat_hash_map<const HloComputation*, WarPredecessorMap>;

struct BufferWriterEntry {
  int64_t pos = 0;
  const HloInstruction* instr = nullptr;
  absl::InlinedVector<HloValue::Id, 1> value_ids;
};

struct BufferReaderEntry {
  int64_t pos = 0;
  const HloInstruction* instr = nullptr;
  HloValue::Id value_id = 0;
};

struct CompBufferAccesses {
  absl::flat_hash_map<const HloInstruction*,
                      absl::InlinedVector<HloValue::Id, 1>>
      writer_values;
  std::vector<const HloInstruction*> writer_order;
  std::vector<BufferReaderEntry> readers;
  absl::flat_hash_set<HloValue::Id> touched_values;
};

// Invokes `fn(comp, ancestor)` for each non-zero-cost scheduled instruction in
// `pos_maps` that executes `instr` (either `instr` itself or a non-fusion
// caller such as `kWhile`, `kConditional`, or `kCall`).
void ForEachScheduledAncestor(
    const HloInstruction* instr, const CompInstructionPosMap& pos_maps,
    absl::FunctionRef<void(const HloComputation*, const HloInstruction*)> fn) {
  absl::InlinedVector<const HloInstruction*, 4> stack = {instr};
  absl::flat_hash_set<const HloInstruction*> visited;
  while (!stack.empty()) {
    const HloInstruction* curr = stack.back();
    stack.pop_back();
    if (!visited.insert(curr).second) {
      continue;
    }
    const HloComputation* parent = curr->parent();
    if (parent == nullptr) {
      continue;
    }
    if (!IsZeroCostOp(curr) && pos_maps.contains(parent)) {
      fn(parent, curr);
    }
    for (const HloInstruction* caller : parent->caller_instructions()) {
      if (caller->opcode() != HloOpcode::kFusion) {
        stack.push_back(caller);
      }
    }
  }
}

// Collects all scheduled writer and reader instructions per candidate
// computation for the `HloValue`s sharing `buffer`.
absl::flat_hash_map<const HloComputation*, CompBufferAccesses>
CollectBufferAccessesByComputation(const HloBuffer& buffer,
                                   const CompInstructionPosMap& pos_maps) {
  absl::flat_hash_map<const HloComputation*, CompBufferAccesses>
      accesses_by_comp;
  for (const HloValue* value : buffer.values()) {
    const HloValue::Id val_id = value->id();
    ForEachScheduledAncestor(
        value->defining_instruction(), pos_maps,
        [&](const HloComputation* comp, const HloInstruction* writer) {
          CompBufferAccesses& acc = accesses_by_comp[comp];
          acc.touched_values.insert(val_id);
          auto [it, inserted] = acc.writer_values.try_emplace(writer);
          if (inserted) {
            acc.writer_order.push_back(writer);
          }
          if (!absl::c_linear_search(it->second, val_id)) {
            it->second.push_back(val_id);
          }
        });
    absl::flat_hash_map<const HloComputation*,
                        absl::flat_hash_set<const HloInstruction*>>
        seen_readers;
    for (const HloUse& use : value->GetUses()) {
      ForEachScheduledAncestor(
          use.instruction, pos_maps,
          [&](const HloComputation* comp, const HloInstruction* reader) {
            if (!seen_readers[comp].insert(reader).second) {
              return;
            }
            CompBufferAccesses& acc = accesses_by_comp[comp];
            acc.touched_values.insert(val_id);
            acc.readers.push_back(BufferReaderEntry{
                pos_maps.at(comp).at(reader), reader, val_id});
          });
    }
  }
  return accesses_by_comp;
}

// Records schedule-preserving ordering edges in `comp_war` between writers and
// readers of distinct `HloValue`s within the same `HloBuffer`.
void AddBufferOrderingEdges(CompBufferAccesses& acc,
                            const InstructionPosMap& pos_map,
                            WarPredecessorMap& comp_war) {
  std::vector<BufferWriterEntry> writers;
  writers.reserve(acc.writer_order.size());
  for (const HloInstruction* instr : acc.writer_order) {
    writers.push_back(BufferWriterEntry{
        pos_map.at(instr), instr, std::move(acc.writer_values.at(instr))});
  }
  absl::c_sort(writers,
               [](const BufferWriterEntry& a, const BufferWriterEntry& b) {
                 return a.pos < b.pos;
               });

  auto has_other_value = [](const BufferWriterEntry& w, HloValue::Id v_id) {
    return w.value_ids.size() > 1 || w.value_ids[0] != v_id;
  };

  // Chain all distinct writer instructions of the shared buffer in schedule
  // order so transitivity orders every writer pair (including repeated calls to
  // a subcomputation defining the same `HloValue`) in O(W) edges.
  for (size_t i = 0; i + 1 < writers.size(); ++i) {
    comp_war[writers[i + 1].instr].push_back(writers[i].instr);
  }

  // Connect each reader to the latest preceding writer and earliest succeeding
  // writer of a different value.
  for (const BufferReaderEntry& r : acc.readers) {
    auto it = absl::c_lower_bound(
        writers, r.pos,
        [](const BufferWriterEntry& w, int64_t pos) { return w.pos < pos; });
    for (auto prev = it; prev != writers.begin();) {
      --prev;
      if (has_other_value(*prev, r.value_id)) {
        comp_war[r.instr].push_back(prev->instr);
        break;
      }
    }
    for (auto next = it; next != writers.end(); ++next) {
      if (next->pos == r.pos) {
        continue;
      }
      if (has_other_value(*next, r.value_id)) {
        comp_war[next->instr].push_back(r.instr);
        break;
      }
    }
  }
}

void SortAndDeduplicatePredecessors(
    const HloModule& module, absl::Span<HloComputation* const> candidate_comps,
    const CompInstructionPosMap& pos_maps, CompWarPredecessorMap& war_preds) {
  for (HloComputation* comp : candidate_comps) {
    auto comp_it = war_preds.find(comp);
    if (comp_it == war_preds.end()) {
      continue;
    }
    WarPredecessorMap& comp_war = comp_it->second;
    const InstructionPosMap& pos_map = pos_maps.at(comp);
    for (const HloInstruction* instr :
         module.schedule().sequence(comp).instructions()) {
      auto preds_it = comp_war.find(instr);
      if (preds_it == comp_war.end()) {
        continue;
      }
      std::vector<const HloInstruction*>& preds = preds_it->second;
      absl::c_sort(preds,
                   [&](const HloInstruction* a, const HloInstruction* b) {
                     return pos_map.at(a) < pos_map.at(b);
                   });
      preds.erase(std::unique(preds.begin(), preds.end()), preds.end());
    }
  }
}

// Computes per-computation ordering dependencies induced by mandatory buffer
// sharing (`HloBuffer`s containing multiple `HloValue`s, including while-loop
// carry buffers, conditional cross-branch root aliases, entry input-output
// aliases, and in-place operations).
//
// `HorizontalFusion` runs after `RunPostSchedulingCopyInsertion`, which relies
// on the sequential HLO schedule to keep live ranges of distinct `HloValue`s
// in the same `HloBuffer` non-overlapping. Whenever two instructions in a
// scheduled computation access distinct `HloValue`s of the same `HloBuffer` and
// at least one of them writes its `HloValue`, their relative schedule order
// must be preserved and they must not be horizontally fused into one concurrent
// kernel.
absl::StatusOr<CompWarPredecessorMap> ComputeSharedBufferOrderingPredecessors(
    const HloModule& module, const se::DeviceDescription& device_info,
    absl::Span<HloComputation* const> candidate_comps) {
  CompWarPredecessorMap result;
  if (candidate_comps.empty()) {
    return result;
  }

  GpuAliasInfo alias_info(device_info);
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                        HloAliasAnalysis::Run(&module, &alias_info));

  CompInstructionPosMap pos_maps;
  pos_maps.reserve(candidate_comps.size());
  for (HloComputation* comp : candidate_comps) {
    const std::vector<HloInstruction*>& seq =
        module.schedule().sequence(comp).instructions();
    InstructionPosMap& pos_map = pos_maps[comp];
    pos_map.reserve(seq.size());
    for (int64_t i = 0; i < static_cast<int64_t>(seq.size()); ++i) {
      pos_map[seq[i]] = i;
    }
  }

  for (const HloBuffer& buffer : alias_analysis->buffers()) {
    if (buffer.values().size() < 2) {
      continue;
    }
    absl::flat_hash_map<const HloComputation*, CompBufferAccesses>
        accesses_by_comp = CollectBufferAccessesByComputation(buffer, pos_maps);
    for (HloComputation* comp : candidate_comps) {
      auto acc_it = accesses_by_comp.find(comp);
      if (acc_it == accesses_by_comp.end()) {
        continue;
      }
      CompBufferAccesses& acc = acc_it->second;
      if (acc.touched_values.size() < 2 || acc.writer_order.empty()) {
        continue;
      }
      AddBufferOrderingEdges(acc, pos_maps.at(comp), result[comp]);
    }
  }

  SortAndDeduplicatePredecessors(module, candidate_comps, pos_maps, result);
  return result;
}

// Schedule positions of a computation's instructions together with the
// earliest position each instruction could be moved to.
//
// The effective ready position of a zero-cost instruction (bitcast, GTE, ...)
// is the latest effective ready position of its data and control predecessors,
// because zero-cost instructions can be freely rescheduled behind their
// operands. For all other instructions it is their own schedule position.
class ScheduleReadiness {
 public:
  ScheduleReadiness(absl::Span<HloInstruction* const> sequence,
                    const WarPredecessorMap* absl_nullable war_preds)
      : war_preds_(war_preds), effective_ready_(sequence.size(), -1) {
    pos_.reserve(sequence.size());
    for (int64_t i = 0; i < static_cast<int64_t>(sequence.size()); ++i) {
      const HloInstruction* instr = sequence[i];
      pos_[instr] = i;
      effective_ready_[i] =
          IsZeroCostOp(instr) ? PredecessorsReadyPos(instr) : i;
    }
  }

  int64_t SchedulePos(const HloInstruction* instr) const {
    return pos_.at(instr);
  }

  // Returns the latest effective ready position over the data, control, and WAR
  // predecessors of `instr`, or -1 if it has none: `instr` could be moved to
  // any schedule position after it. Predecessors outside the schedule
  // conservatively count as `instr` itself.
  int64_t PredecessorsReadyPos(const HloInstruction* instr) const {
    const int64_t self_pos = pos_.at(instr);
    int64_t ready = -1;
    auto visit = [&](const HloInstruction* pred) {
      auto it = pos_.find(pred);
      ready = std::max(
          ready, it != pos_.end() ? effective_ready_[it->second] : self_pos);
    };
    absl::c_for_each(instr->operands(), visit);
    absl::c_for_each(instr->control_predecessors(), visit);
    if (war_preds_ != nullptr) {
      if (auto war_it = war_preds_->find(instr); war_it != war_preds_->end()) {
        absl::c_for_each(war_it->second, visit);
      }
    }
    return ready;
  }

 private:
  const WarPredecessorMap* absl_nullable war_preds_;
  absl::flat_hash_map<const HloInstruction*, int64_t> pos_;
  std::vector<int64_t> effective_ready_;
};

// Each batch is replaced by one fusion that is emitted at the position of its
// first instruction (the anchor). Moves the zero-cost transitive predecessors
// of the remaining batch members that are scheduled after the anchor to just
// before it, preserving their relative order. Batch formation guarantees that
// all non-zero-cost predecessors are already scheduled before the anchor (for
// consecutive batches by construction, for cross-barrier batches via
// `ScheduleReadiness::PredecessorsReadyPos`), so the result is a valid
// schedule. Returns `std::nullopt` if no instruction needs to move.
std::optional<std::vector<HloInstruction*>> HoistZeroCostPredecessors(
    absl::Span<HloInstruction* const> sequence,
    const ScheduleReadiness& readiness,
    absl::Span<const std::vector<HloInstruction*>> batches) {
  std::vector<int64_t> hoist_to(sequence.size());
  absl::c_iota(hoist_to, 0);
  bool needs_reorder = false;
  std::vector<const HloInstruction*> worklist;
  for (const std::vector<HloInstruction*>& batch : batches) {
    const int64_t anchor_pos = readiness.SchedulePos(batch.front());
    worklist.assign(batch.begin() + 1, batch.end());
    while (!worklist.empty()) {
      const HloInstruction* curr = worklist.back();
      worklist.pop_back();
      auto visit_pred = [&](const HloInstruction* pred) {
        const int64_t pred_pos = readiness.SchedulePos(pred);
        if (pred_pos > anchor_pos && IsZeroCostOp(pred) &&
            anchor_pos < hoist_to[pred_pos]) {
          hoist_to[pred_pos] = anchor_pos;
          needs_reorder = true;
          worklist.push_back(pred);
        }
      };
      absl::c_for_each(curr->operands(), visit_pred);
      absl::c_for_each(curr->control_predecessors(), visit_pred);
    }
  }
  if (!needs_reorder) {
    return std::nullopt;
  }

  std::vector<std::vector<HloInstruction*>> hoisted_before(sequence.size());
  for (int64_t i = 0; i < static_cast<int64_t>(sequence.size()); ++i) {
    if (hoist_to[i] < i) {
      hoisted_before[hoist_to[i]].push_back(sequence[i]);
    }
  }
  std::vector<HloInstruction*> reordered;
  reordered.reserve(sequence.size());
  for (int64_t i = 0; i < static_cast<int64_t>(sequence.size()); ++i) {
    absl::c_copy(hoisted_before[i], std::back_inserter(reordered));
    if (hoist_to[i] == i) {
      reordered.push_back(sequence[i]);
    }
  }
  return reordered;
}

}  // namespace

absl::StatusOr<bool> HorizontalFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!module->has_schedule()) {
    return false;
  }
  // Nested Triton leaf kernels require Triton, which XLA:GPU only enables on
  // CUDA Ampere (sm_80) and newer, and the thread-block-specialized emitter
  // emits NVVM-specific code. `GpuCompiler` applies the same guard; it is
  // repeated here so that the pass is safe to run standalone.
  const se::CudaComputeCapability* cuda_cc =
      device_info_.gpu_compute_capability().cuda_compute_capability();
  if (cuda_cc == nullptr || !cuda_cc->IsAtLeastAmpere()) {
    return false;
  }

  std::vector<HloComputation*> candidate_comps;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    if (!module->schedule().is_computation_scheduled(comp)) {
      continue;
    }
    int64_t fusion_count = 0;
    for (const HloInstruction* instr :
         module->schedule().sequence(comp).instructions()) {
      if (instr->opcode() == HloOpcode::kFusion && !IsZeroCostOp(instr)) {
        if (++fusion_count >= 2) {
          candidate_comps.push_back(comp);
          break;
        }
      }
    }
  }
  if (candidate_comps.empty()) {
    return false;
  }

  ABSL_ASSIGN_OR_RETURN(const CompWarPredecessorMap war_preds_by_comp,
                        ComputeSharedBufferOrderingPredecessors(
                            *module, device_info_, candidate_comps));
  const bool enable_same_shape_mof =
      module->config()
          .debug_options()
          .xla_gpu_experimental_enable_same_shape_multi_output_fusion();

  bool changed = false;
  for (HloComputation* comp : candidate_comps) {
    const std::vector<HloInstruction*> sequence =
        module->schedule().sequence(comp).instructions();
    const auto comp_war_it = war_preds_by_comp.find(comp);
    const WarPredecessorMap* war_preds =
        comp_war_it != war_preds_by_comp.end() ? &comp_war_it->second : nullptr;
    const ScheduleReadiness readiness(sequence, war_preds);

    std::unique_ptr<HloReachabilityMap> reachability =
        HloReachabilityMap::Build(comp);
    if (war_preds != nullptr && !war_preds->empty()) {
      std::vector<const HloInstruction*> inputs;
      for (const HloInstruction* instr : sequence) {
        inputs.clear();
        inputs.push_back(instr);
        inputs.insert(inputs.end(), instr->operands().begin(),
                      instr->operands().end());
        inputs.insert(inputs.end(), instr->control_predecessors().begin(),
                      instr->control_predecessors().end());
        if (auto war_it = war_preds->find(instr); war_it != war_preds->end()) {
          inputs.insert(inputs.end(), war_it->second.begin(),
                        war_it->second.end());
        }
        reachability->FastSetReachabilityToUnion(inputs, instr);
      }
    }

    absl::flat_hash_map<const HloInstruction*, std::optional<LaunchDimensions>>
        launch_dims_cache;
    auto get_launch_dims =
        [&](const HloInstruction* instr) -> std::optional<LaunchDimensions> {
      if (auto it = launch_dims_cache.find(instr);
          it != launch_dims_cache.end()) {
        return it->second;
      }
      std::optional<LaunchDimensions> dims =
          GetLaunchDimensions(instr, device_info_);
      launch_dims_cache.emplace(instr, dims);
      return dims;
    };

    LeafRuntimeEstimator runtime_estimator(comp, &device_info_,
                                           enable_same_shape_mof);
    auto estimate_runtime = [&](const HloInstruction* instr) {
      return runtime_estimator.EstimateExecTime(instr);
    };

    std::vector<std::vector<HloInstruction*>> batches_to_fuse;
    // Batches of tiny fusions that are separated by non-fusible instructions
    // in the schedule. Each batch will be emitted at the position of its first
    // instruction (`anchor_pos`), so a later fusion may only join if all of its
    // non-zero-cost predecessors are scheduled before the anchor.
    std::vector<SmallFusionBatch> open_small_batches;
    FusionBatch current_batch;

    auto try_add_orphan_to_small_batches = [&](FusionBatch single_batch) {
      HloInstruction* orphan = single_batch.instructions.front();
      if (options_.max_cross_barrier_output_bytes <= 0 ||
          ShapeUtil::ByteSizeOfElementsRecursive(orphan->shape()) >
              options_.max_cross_barrier_output_bytes) {
        return;
      }
      const int64_t ready_pos = readiness.PredecessorsReadyPos(orphan);
      for (auto it = open_small_batches.begin(); it != open_small_batches.end();
           ++it) {
        if (ready_pos < it->anchor_pos &&
            TryAddToBatch(it->batch, orphan, *reachability, device_info_,
                          options_, get_launch_dims, estimate_runtime)) {
          if (static_cast<int64_t>(it->batch.instructions.size()) >=
              options_.max_nested_kernels) {
            batches_to_fuse.push_back(std::move(it->batch.instructions));
            open_small_batches.erase(it);
          }
          return;
        }
      }
      // Close the oldest open batch to bound the number of candidate batches.
      if (static_cast<int64_t>(open_small_batches.size()) >=
          kMaxOpenSmallBatches) {
        if (open_small_batches.front().batch.instructions.size() >= 2) {
          batches_to_fuse.push_back(
              std::move(open_small_batches.front().batch.instructions));
        }
        open_small_batches.erase(open_small_batches.begin());
      }
      open_small_batches.push_back(SmallFusionBatch{
          std::move(single_batch), readiness.SchedulePos(orphan)});
    };

    auto flush_batch = [&]() {
      if (current_batch.instructions.size() >= 2) {
        batches_to_fuse.push_back(std::move(current_batch.instructions));
      } else if (current_batch.instructions.size() == 1) {
        try_add_orphan_to_small_batches(std::move(current_batch));
      }
      current_batch = FusionBatch{};
    };

    for (HloInstruction* instr : sequence) {
      if (IsZeroCostOp(instr)) {
        continue;
      }

      if (!TryAddToBatch(current_batch, instr, *reachability, device_info_,
                         options_, get_launch_dims, estimate_runtime) &&
          !current_batch.instructions.empty()) {
        flush_batch();
        TryAddToBatch(current_batch, instr, *reachability, device_info_,
                      options_, get_launch_dims, estimate_runtime);
      }
    }
    ABSL_RETURN_IF_ERROR(runtime_estimator.status());
    flush_batch();
    for (SmallFusionBatch& small_batch : open_small_batches) {
      if (small_batch.batch.instructions.size() >= 2) {
        batches_to_fuse.push_back(std::move(small_batch.batch.instructions));
      }
    }

    if (batches_to_fuse.empty()) {
      continue;
    }

    // Place each fused batch explicitly at the schedule position of its anchor
    // (`batch.front()`) rather than leaving `outer_fusion` unscheduled for
    // `HloSchedule::Update()`, which would hoist it to its earliest HLO operand
    // and potentially violate shared-buffer WAR ordering.
    const std::vector<HloInstruction*> base_sequence =
        HoistZeroCostPredecessors(sequence, readiness, batches_to_fuse)
            .value_or(sequence);
    InstructionPosMap base_pos;
    base_pos.reserve(base_sequence.size());
    for (int64_t i = 0; i < static_cast<int64_t>(base_sequence.size()); ++i) {
      base_pos[base_sequence[i]] = i;
    }

    std::vector<bool> skip_slot(base_sequence.size(), false);
    std::vector<std::vector<HloInstruction*>> replace_at_slot(
        base_sequence.size());
    for (const std::vector<HloInstruction*>& batch : batches_to_fuse) {
      const int64_t anchor_slot = base_pos.at(batch.front());
      for (size_t i = 1; i < batch.size(); ++i) {
        skip_slot[base_pos.at(batch[i])] = true;
      }
      ABSL_ASSIGN_OR_RETURN(replace_at_slot[anchor_slot],
                            FuseCandidates(module, comp, batch));
      changed = true;
    }

    std::vector<HloInstruction*> updated_sequence;
    updated_sequence.reserve(comp->instruction_count());
    for (size_t i = 0; i < base_sequence.size(); ++i) {
      if (!replace_at_slot[i].empty()) {
        absl::c_copy(replace_at_slot[i], std::back_inserter(updated_sequence));
      } else if (!skip_slot[i]) {
        updated_sequence.push_back(base_sequence[i]);
      }
    }
    module->schedule().set_sequence(comp, updated_sequence);
  }

  if (changed) {
    ABSL_RETURN_IF_ERROR(module->schedule().Update());
  }
  return changed;
}

}  // namespace xla::gpu
