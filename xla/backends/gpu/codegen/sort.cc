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
#include "xla/backends/gpu/codegen/sort.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialectDecl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/ptx_custom_kernel.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/future.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

namespace arith = ::mlir::arith;
namespace scf = ::mlir::scf;
namespace tensor = ::mlir::tensor;

using ::llvm::SmallVector;
using ::mlir::ImplicitLocOpBuilder;
using ::mlir::MLIRContext;
using ::mlir::OpBuilder;
using ::mlir::RankedTensorType;
using ::mlir::Type;
using ::mlir::TypeRange;
using ::mlir::Value;
using ::mlir::ValueRange;
using ::mlir::func::FuncOp;

constexpr uint64_t kBitonicSortUnrollFactor = 4;

SmallVector<Value> EmitIfElse(
    ImplicitLocOpBuilder& b, Value condition, TypeRange result_types,
    absl::FunctionRef<SmallVector<Value>(ImplicitLocOpBuilder&)> then_fn,
    absl::FunctionRef<SmallVector<Value>(ImplicitLocOpBuilder&)> else_fn) {
  auto if_op =
      scf::IfOp::create(b, result_types, condition, /*withElseRegion=*/true);

  OpBuilder then_b = if_op.getThenBodyBuilder();
  ImplicitLocOpBuilder implicit_then_b(b.getLoc(), then_b);
  scf::YieldOp::create(implicit_then_b, then_fn(implicit_then_b));

  OpBuilder else_b = if_op.getElseBodyBuilder();
  ImplicitLocOpBuilder implicit_else_b(b.getLoc(), else_b);
  scf::YieldOp::create(implicit_else_b, else_fn(implicit_else_b));

  return SmallVector<Value>(if_op.getResults().begin(),
                            if_op.getResults().end());
}

absl::StatusOr<SmallVector<Value>> EmitIfElseWithStatus(
    ImplicitLocOpBuilder& b, Value condition, TypeRange result_types,
    absl::FunctionRef<absl::StatusOr<SmallVector<Value>>(ImplicitLocOpBuilder&)>
        then_fn,
    absl::FunctionRef<absl::StatusOr<SmallVector<Value>>(ImplicitLocOpBuilder&)>
        else_fn) {
  auto if_op =
      scf::IfOp::create(b, result_types, condition, /*withElseRegion=*/true);

  OpBuilder then_b = if_op.getThenBodyBuilder();
  ImplicitLocOpBuilder implicit_then_b(b.getLoc(), then_b);
  ABSL_ASSIGN_OR_RETURN(SmallVector<Value> then_results,
                        then_fn(implicit_then_b));
  scf::YieldOp::create(implicit_then_b, then_results);

  OpBuilder else_b = if_op.getElseBodyBuilder();
  ImplicitLocOpBuilder implicit_else_b(b.getLoc(), else_b);
  ABSL_ASSIGN_OR_RETURN(SmallVector<Value> else_results,
                        else_fn(implicit_else_b));
  scf::YieldOp::create(implicit_else_b, else_results);

  return SmallVector<Value>(if_op.getResults().begin(),
                            if_op.getResults().end());
}

SmallVector<Value> EmitUpdateIf(
    ImplicitLocOpBuilder& b, Value condition, ValueRange values,
    absl::FunctionRef<SmallVector<Value>(ImplicitLocOpBuilder&)>
        updated_values_fn) {
  return EmitIfElse(b, condition, TypeRange(values), updated_values_fn,
                    [&](ImplicitLocOpBuilder&) {
                      return SmallVector<Value>(values.begin(), values.end());
                    });
}

absl::StatusOr<SmallVector<Value>> EmitUpdateIfWithStatus(
    ImplicitLocOpBuilder& b, Value condition, ValueRange values,
    absl::FunctionRef<absl::StatusOr<SmallVector<Value>>(ImplicitLocOpBuilder&)>
        updated_values_fn) {
  return EmitIfElseWithStatus(
      b, condition, TypeRange(values), updated_values_fn,
      [&](ImplicitLocOpBuilder&) -> absl::StatusOr<SmallVector<Value>> {
        return SmallVector<Value>(values.begin(), values.end());
      });
}

absl::StatusOr<SmallVector<Value>> EmitForWithStatus(
    ImplicitLocOpBuilder& b, Value lower_bound, Value upper_bound, Value step,
    ValueRange init_args,
    absl::FunctionRef<absl::StatusOr<SmallVector<Value>>(
        ImplicitLocOpBuilder&, Value /*iv*/, ValueRange /*iter_args*/)>
        body_fn) {
  auto for_op =
      scf::ForOp::create(b, lower_bound, upper_bound, step, init_args,
                         [](OpBuilder&, mlir::Location, Value, ValueRange) {});
  ImplicitLocOpBuilder body_b(b.getLoc(), for_op.getBody(),
                              for_op.getBody()->begin());
  ABSL_ASSIGN_OR_RETURN(
      SmallVector<Value> results,
      body_fn(body_b, for_op.getInductionVar(), for_op.getRegionIterArgs()));
  scf::YieldOp::create(body_b, results);
  return SmallVector<Value>(for_op.getResults().begin(),
                            for_op.getResults().end());
}

struct BitonicSortKernelSpec {
  enum class Kind {
    // Empty sort (dimension_to_sort_bound == 1) that only copies iota
    // operands to the output.
    kIotaCopy,
    // Single compare step without shared memory (global steps or 1-step sort).
    kSingleStep,
    // Trailing intra-tile (and optional cluster) merge for outer stages: steps
    // mask = num_stages - 1 down to 0 with xor_mask = block_size = 1 << mask.
    kIntraTileMerge,
    // Initial triangular intra-tile (and optional cluster) sort:
    // stages 0 .. num_stages - 1.
    kIntraTileSort,
  };

  Kind kind;
  // For kSingleStep, the xor mask of the step. For kIntraTileMerge and
  // kIntraTileSort, the number of stages (intra-tile + cluster) covered by the
  // kernel.
  int64_t xor_mask_or_num_stages = 0;
  // For kSingleStep, the block size of the step.
  int64_t block_size = 0;

  bool is_tiled() const {
    return kind == Kind::kIntraTileMerge || kind == Kind::kIntraTileSort;
  }

  friend bool operator==(const BitonicSortKernelSpec& lhs,
                         const BitonicSortKernelSpec& rhs) {
    return lhs.kind == rhs.kind &&
           lhs.xor_mask_or_num_stages == rhs.xor_mask_or_num_stages &&
           lhs.block_size == rhs.block_size;
  }

  template <typename H>
  friend H AbslHashValue(H h, const BitonicSortKernelSpec& s) {
    return H::combine(std::move(h), s.kind, s.xor_mask_or_num_stages,
                      s.block_size);
  }
};

using ReadElementFn = absl::FunctionRef<absl::StatusOr<Value>(
    ImplicitLocOpBuilder&, ValueRange /*tensors*/, int64_t /*operand*/,
    Value /*index*/)>;
using WriteElementFn = absl::FunctionRef<Value(
    ImplicitLocOpBuilder&, Value /*tensor*/, int64_t /*operand*/,
    Value /*index*/, Value /*value*/)>;

struct ClusterStepInfo {
  Value cta_base;
  int64_t tile_size;
  Value block_size;
  ReadElementFn read_partner;
  WriteElementFn write_partner;
};

enum class Pair1Stride {
  kDefault,
  // block_size <= 32 and xor_mask < 64 with num_threads % 32 == 0 and
  // unroll_factor == 2: pair 1 is offset by +64 from pair 0.
  kWithinDoubleWarp64,
  // Merge step (xor_mask == block_size) with block_size >= 64,
  // num_threads % 32 == 0, and unroll_factor == 2:
  // idx_1 = idx_0 + 32 and cmp_1 = cmp_0 + 32.
  kMerge32,
  // Flip step (xor_mask == 2 * block_size - 1) with block_size >= 64,
  // num_threads % 32 == 0, and unroll_factor == 2:
  // idx_1 = idx_0 + 32 and cmp_1 = cmp_0 - 32.
  kFlip32,
};

absl::StatusOr<SmallVector<Value>> EmitCompareLoopBody(
    ImplicitLocOpBuilder& b, Value iteration_bound, int64_t num_threads,
    int64_t unroll_factor, int64_t num_values, Value element_pair_index,
    Value xor_mask, Value mask_low, ValueRange tensors,
    ReadElementFn read_element, WriteElementFn write_element, FuncOp comparator,
    bool force_write, bool needs_bounds_checks, bool is_merge_step,
    Pair1Stride pair1_stride = Pair1Stride::kDefault,
    std::optional<ClusterStepInfo> cluster_step = std::nullopt) {
  auto index_const = [&](int64_t value) -> Value {
    return arith::ConstantIndexOp::create(b, value);
  };
  if (pair1_stride != Pair1Stride::kDefault) {
    TF_RET_CHECK(num_threads % kNumShmemBanks == 0 && unroll_factor == 2 &&
                 !cluster_step.has_value());
  }

  if (num_threads % kNumShmemBanks == 0 && unroll_factor > 1) {
    Value bank_row_id = arith::DivUIOp::create(b, element_pair_index,
                                               index_const(kNumShmemBanks));
    Value offset = arith::RemUIOp::create(b, element_pair_index,
                                          index_const(kNumShmemBanks));
    Value first_in_bank_row = arith::MulIOp::create(
        b, bank_row_id, index_const(unroll_factor * kNumShmemBanks));
    element_pair_index = arith::AddIOp::create(b, first_in_bank_row, offset);
  } else {
    element_pair_index = arith::MulIOp::create(b, element_pair_index,
                                               index_const(unroll_factor));
  }

  std::optional<int64_t> const_mask_low;
  if (auto const_op = mask_low.getDefiningOp<arith::ConstantIndexOp>()) {
    const_mask_low = const_op.value();
  }

  Value cta_is_high;
  Value partner_cta_base;
  Value idx_high_cta_base;
  Value half_tile_offset;
  Value intra_tile_xor;
  if (cluster_step.has_value()) {
    cta_is_high =
        arith::CmpIOp::create(b, arith::CmpIPredicate::ne,
                              arith::AndIOp::create(b, cluster_step->cta_base,
                                                    cluster_step->block_size),
                              index_const(0));
    Value cta_xor_mask = arith::AndIOp::create(
        b, xor_mask, index_const(-cluster_step->tile_size));
    partner_cta_base =
        arith::XOrIOp::create(b, cluster_step->cta_base, cta_xor_mask);
    idx_high_cta_base = arith::SelectOp::create(
        b, cta_is_high, cluster_step->cta_base, partner_cta_base);
    half_tile_offset = arith::SelectOp::create(
        b, cta_is_high, index_const(cluster_step->tile_size / 2),
        index_const(0));
    intra_tile_xor = arith::AndIOp::create(
        b, xor_mask, index_const(cluster_step->tile_size - 1));
  }

  ReadElementFn read_partner =
      cluster_step.has_value() ? cluster_step->read_partner : read_element;
  WriteElementFn write_partner =
      cluster_step.has_value() ? cluster_step->write_partner : write_element;

  SmallVector<Value> current_tensors(tensors.begin(), tensors.end());
  Value pair0_keys_index;
  Value pair0_compare_index;
  for (int64_t i = 0; i < unroll_factor; ++i) {
    Value current_keys_index;
    Value compare_keys_index;
    Value upper_keys_index;
    if (i == 1 && pair1_stride != Pair1Stride::kDefault) {
      int64_t idx_stride = pair1_stride == Pair1Stride::kWithinDoubleWarp64
                               ? 2 * kNumShmemBanks
                               : kNumShmemBanks;
      current_keys_index =
          arith::AddIOp::create(b, pair0_keys_index, index_const(idx_stride));
      if (pair1_stride == Pair1Stride::kFlip32) {
        compare_keys_index = arith::SubIOp::create(b, pair0_compare_index,
                                                   index_const(kNumShmemBanks));
      } else {
        compare_keys_index = arith::AddIOp::create(b, pair0_compare_index,
                                                   index_const(idx_stride));
      }
      upper_keys_index = compare_keys_index;
    } else if (cluster_step.has_value()) {
      Value pair_offset = element_pair_index;
      if (i > 0) {
        int64_t offset_stride =
            num_threads % kNumShmemBanks == 0 ? i * kNumShmemBanks : i;
        pair_offset = arith::AddIOp::create(b, element_pair_index,
                                            index_const(offset_stride));
      }
      Value local_idx_low =
          arith::OrIOp::create(b, half_tile_offset, pair_offset);
      Value local_idx_high =
          arith::XOrIOp::create(b, local_idx_low, intra_tile_xor);
      current_keys_index = arith::SelectOp::create(
          b, cta_is_high, local_idx_high, local_idx_low);
      Value partner_local_idx = arith::AndIOp::create(
          b,
          arith::SelectOp::create(b, cta_is_high, local_idx_low,
                                  local_idx_high),
          index_const(cluster_step->tile_size - 1));
      compare_keys_index =
          arith::OrIOp::create(b, partner_cta_base, partner_local_idx);
      upper_keys_index =
          arith::OrIOp::create(b, idx_high_cta_base, local_idx_high);
    } else {
      Value pair_offset = element_pair_index;
      if (i > 0) {
        int64_t offset_stride =
            num_threads % kNumShmemBanks == 0 ? i * kNumShmemBanks : i;
        pair_offset = arith::AddIOp::create(b, element_pair_index,
                                            index_const(offset_stride));
      }
      Value low_bits = arith::AndIOp::create(b, pair_offset, mask_low);
      if (const_mask_low.has_value()) {
        // Preserve KnownBits on compile-time constant masks so LLVM folds
        // compare_keys_index and pair-1 offsets into shared-memory immediate
        // displacements.
        Value mask_high = index_const(~(*const_mask_low));
        Value high_bits = arith::ShLIOp::create(
            b, arith::AndIOp::create(b, pair_offset, mask_high),
            index_const(1));
        current_keys_index = arith::OrIOp::create(b, high_bits, low_bits);
      } else {
        Value double_pair_offset =
            arith::ShLIOp::create(b, pair_offset, index_const(1));
        current_keys_index =
            arith::SubIOp::create(b, double_pair_offset, low_bits);
      }
      // On merge steps (xor_mask == block_size), bit log2(block_size) of
      // current_keys_index is 0, so XOR equals addition.
      compare_keys_index =
          is_merge_step ? arith::AddIOp::create(b, current_keys_index, xor_mask)
                              .getResult()
                        : arith::XOrIOp::create(b, current_keys_index, xor_mask)
                              .getResult();
      upper_keys_index = compare_keys_index;
      if (i == 0) {
        pair0_keys_index = current_keys_index;
        pair0_compare_index = compare_keys_index;
      }
    }

    auto emit_body = [&](ImplicitLocOpBuilder& body_b)
        -> absl::StatusOr<SmallVector<Value>> {
      SmallVector<Value> first_values;
      SmallVector<Value> second_values;
      first_values.reserve(num_values);
      second_values.reserve(num_values);
      if (cluster_step.has_value()) {
        for (int64_t param = 0; param < num_values; ++param) {
          ABSL_ASSIGN_OR_RETURN(
              Value val_partner,
              read_partner(body_b, current_tensors, param, compare_keys_index));
          first_values.push_back(val_partner);
        }
        for (int64_t param = 0; param < num_values; ++param) {
          ABSL_ASSIGN_OR_RETURN(
              Value val_self,
              read_element(body_b, current_tensors, param, current_keys_index));
          second_values.push_back(val_self);
        }
      } else {
        for (int64_t param = 0; param < num_values; ++param) {
          ABSL_ASSIGN_OR_RETURN(
              Value val_compare,
              read_element(body_b, current_tensors, param, compare_keys_index));
          first_values.push_back(val_compare);
          ABSL_ASSIGN_OR_RETURN(
              Value val_current,
              read_element(body_b, current_tensors, param, current_keys_index));
          second_values.push_back(val_current);
        }
      }

      SmallVector<Value> values_to_compare;
      values_to_compare.reserve(num_values * 2);
      for (int64_t param = 0; param < num_values; ++param) {
        if (cluster_step.has_value()) {
          Value val_low = arith::SelectOp::create(
              body_b, cta_is_high, first_values[param], second_values[param]);
          Value val_high = arith::SelectOp::create(
              body_b, cta_is_high, second_values[param], first_values[param]);
          values_to_compare.push_back(val_high);
          values_to_compare.push_back(val_low);
        } else {
          values_to_compare.push_back(first_values[param]);
          values_to_compare.push_back(second_values[param]);
        }
      }
      Value pred = PureCallOp::create(body_b, comparator, values_to_compare)
                       .getResult(0);
      Value zero = arith::ConstantIntOp::create(body_b, body_b.getI8Type(), 0);
      Value is_smaller_than =
          arith::CmpIOp::create(body_b, arith::CmpIPredicate::ne, pred, zero);

      auto write_swapped_order =
          [&](ImplicitLocOpBuilder& swap_b) -> SmallVector<Value> {
        SmallVector<Value> updated(current_tensors.begin(),
                                   current_tensors.end());
        for (int64_t param = 0; param < num_values; ++param) {
          updated[param] =
              write_element(swap_b, updated[param], param, current_keys_index,
                            first_values[param]);
          updated[param] =
              write_partner(swap_b, updated[param], param, compare_keys_index,
                            second_values[param]);
        }
        return updated;
      };

      if (force_write) {
        auto write_original_order =
            [&](ImplicitLocOpBuilder& orig_b) -> SmallVector<Value> {
          SmallVector<Value> updated(current_tensors.begin(),
                                     current_tensors.end());
          for (int64_t param = 0; param < num_values; ++param) {
            updated[param] =
                write_element(orig_b, updated[param], param, current_keys_index,
                              second_values[param]);
            updated[param] =
                write_partner(orig_b, updated[param], param, compare_keys_index,
                              first_values[param]);
          }
          return updated;
        };
        return EmitIfElse(body_b, is_smaller_than, TypeRange(current_tensors),
                          write_swapped_order, write_original_order);
      }
      return EmitUpdateIf(body_b, is_smaller_than, current_tensors,
                          write_swapped_order);
    };

    if (needs_bounds_checks) {
      Value index_is_inbounds =
          arith::CmpIOp::create(b, arith::CmpIPredicate::slt, upper_keys_index,
                                iteration_bound)
              .getResult();
      ABSL_ASSIGN_OR_RETURN(current_tensors,
                            EmitUpdateIfWithStatus(b, index_is_inbounds,
                                                   current_tensors, emit_body));
    } else {
      ABSL_ASSIGN_OR_RETURN(current_tensors, emit_body(b));
    }
  }
  return current_tensors;
}

absl::StatusOr<SmallVector<Value>> EmitTiledCompareLoop(
    ImplicitLocOpBuilder& b, ValueRange tiled_keys_index, Value thread_id_x,
    const HloSortInstruction* sort, int64_t dimension_to_sort_bound,
    int64_t num_threads, const BitonicSortKernelSpec& kernel_spec,
    ValueRange output_tensors, bool emit_iota_operands, int64_t tile_size,
    int64_t cluster_size, int64_t unroll_factor, FuncOp comparator) {
  int64_t dimension_to_sort = sort->sort_dimension();
  int64_t num_values = sort->operand_count();
  auto index_const = [&](int64_t value) -> Value {
    return arith::ConstantIndexOp::create(b, value);
  };

  SmallVector<Value> shmem_tensors;
  shmem_tensors.reserve(num_values);
  int64_t shmem_tile_size = std::max(tile_size, int64_t{64});
  for (int64_t param = 0; param < num_values; ++param) {
    Type elem_type = emitters::PrimitiveTypeToMlirType(
        sort->operand(param)->shape().element_type(), b);
    shmem_tensors.push_back(AllocateSharedOp::create(
        b, RankedTensorType::get({shmem_tile_size}, elem_type)));
  }

  Value unroll = index_const(unroll_factor);
  Value base_keys_index =
      arith::MulIOp::create(b, tiled_keys_index[dimension_to_sort], unroll);
  Value base_cache_index = arith::MulIOp::create(b, thread_id_x, unroll);

  // Copy `unroll_factor` adjacent elements from global memory (or kIota) to
  // shared memory.
  for (int64_t i = 0; i < unroll_factor; ++i) {
    Value offset = index_const(i);
    Value current_keys_index =
        arith::AddIOp::create(b, base_keys_index, offset);
    Value in_bounds =
        arith::CmpIOp::create(b, arith::CmpIPredicate::slt, current_keys_index,
                              index_const(dimension_to_sort_bound));
    ABSL_ASSIGN_OR_RETURN(
        shmem_tensors,
        EmitUpdateIfWithStatus(
            b, in_bounds, shmem_tensors,
            [&](ImplicitLocOpBuilder& if_b)
                -> absl::StatusOr<SmallVector<Value>> {
              Value cache_index =
                  arith::AddIOp::create(if_b, base_cache_index, offset);
              SmallVector<Value> elem_index(tiled_keys_index.begin(),
                                            tiled_keys_index.end());
              elem_index[dimension_to_sort] = current_keys_index;
              SmallVector<Value> updated(shmem_tensors.begin(),
                                         shmem_tensors.end());
              for (int64_t param = 0; param < num_values; ++param) {
                Value value;
                if (emit_iota_operands &&
                    HloPredicateIsOp<HloOpcode::kIota>(sort->operand(param))) {
                  ABSL_ASSIGN_OR_RETURN(auto iota_vals,
                                        emitters::EmitIota(sort->operand(param),
                                                           elem_index, if_b));
                  value = iota_vals.front();
                } else {
                  value = tensor::ExtractOp::create(if_b, output_tensors[param],
                                                    elem_index);
                }
                updated[param] = tensor::InsertOp::create(
                    if_b, value, updated[param], ValueRange{cache_index});
              }
              return updated;
            }));
  }

  int64_t cluster_tile_size = tile_size * cluster_size;
  int64_t full_tiles_bound = RoundDownTo(dimension_to_sort_bound, tile_size);
  int64_t full_clusters_bound =
      RoundDownTo(dimension_to_sort_bound, cluster_tile_size);
  int64_t active_tiles_bound = RoundUpTo(dimension_to_sort_bound, tile_size);
  int64_t rounded_cluster_bound =
      RoundUpTo(dimension_to_sort_bound, cluster_tile_size);
  int64_t partial_tile_elems = dimension_to_sort_bound % tile_size;

  Value cta_base;
  if (cluster_size > 1) {
    auto cluster_block_id_x = mlir::gpu::ClusterBlockIdOp::create(
        b, mlir::gpu::Dimension::x, b.getIndexAttr(cluster_size));
    cluster_block_id_x->setAttr("xla.range",
                                b.getIndexArrayAttr({0, cluster_size - 1}));
    cta_base =
        arith::MulIOp::create(b, cluster_block_id_x, index_const(tile_size));
  }

  auto read_shmem = [](ImplicitLocOpBuilder& body_b, ValueRange tensors,
                       int64_t operand, Value index) -> absl::StatusOr<Value> {
    return tensor::ExtractOp::create(body_b, tensors[operand],
                                     ValueRange{index})
        .getResult();
  };
  auto write_shmem = [](ImplicitLocOpBuilder& body_b, Value tensor,
                        int64_t operand, Value index, Value value) -> Value {
    return tensor::InsertOp::create(body_b, value, tensor, ValueRange{index});
  };
  auto read_cluster_shmem = [](ImplicitLocOpBuilder& body_b, ValueRange tensors,
                               int64_t operand,
                               Value index) -> absl::StatusOr<Value> {
    auto op =
        tensor::ExtractOp::create(body_b, tensors[operand], ValueRange{index});
    op->setAttr("xla_gpu.dsmem", body_b.getUnitAttr());
    return op.getResult();
  };
  auto write_cluster_shmem = [](ImplicitLocOpBuilder& body_b, Value tensor,
                                int64_t operand, Value index,
                                Value value) -> Value {
    auto op =
        tensor::InsertOp::create(body_b, value, tensor, ValueRange{index});
    op->setAttr("xla_gpu.dsmem", body_b.getUnitAttr());
    return op.getResult();
  };

  Value c0 = index_const(0);
  Value c1 = index_const(1);
  bool use_warp_pair1_stride =
      num_threads % kNumShmemBanks == 0 && (unroll_factor / 2) == 2;
  int64_t intra_stages = Log2Floor<uint64_t>(tile_size);
  int64_t tail_tile_elems = dimension_to_sort_bound < tile_size
                                ? dimension_to_sort_bound
                                : partial_tile_elems;
  int64_t tail_tile_stages =
      tail_tile_elems == 0
          ? 0
          : std::min<int64_t>(
                intra_stages,
                Log2Ceiling(static_cast<uint64_t>(tail_tile_elems)));

  TF_RET_CHECK(cluster_size == 1 ||
               (cluster_size == 2 && tile_size >= shmem_tile_size &&
                dimension_to_sort_bound > tile_size));
  bool has_cluster_step = kernel_spec.xor_mask_or_num_stages > intra_stages;
  TF_RET_CHECK(!has_cluster_step || cluster_size == 2);

  auto sync_workgroup = [&](ImplicitLocOpBuilder& sync_b,
                            ValueRange tensors) -> SmallVector<Value> {
    auto synced = SyncThreadsOp::create(sync_b, TypeRange(tensors), tensors,
                                        mlir::gpu::BarrierScope::Workgroup)
                      .getResults();
    return SmallVector<Value>(synced.begin(), synced.end());
  };

  auto emit_intra_step =
      [&](ImplicitLocOpBuilder& step_b, Value tile_bound, Value xor_mask,
          Value mask_low, ValueRange step_tensors, bool is_merge_step,
          Pair1Stride pair1_stride, bool check_bounds,
          bool post_sync) -> absl::StatusOr<SmallVector<Value>> {
    if (post_sync) {
      ABSL_ASSIGN_OR_RETURN(
          SmallVector<Value> updated,
          EmitCompareLoopBody(
              step_b, tile_bound, num_threads, unroll_factor / 2, num_values,
              thread_id_x, xor_mask, mask_low, step_tensors, read_shmem,
              write_shmem, comparator, /*force_write=*/false, check_bounds,
              is_merge_step,
              use_warp_pair1_stride ? pair1_stride : Pair1Stride::kDefault));
      return sync_workgroup(step_b, updated);
    }
    SmallVector<Value> synced = sync_workgroup(step_b, step_tensors);
    return EmitCompareLoopBody(
        step_b, tile_bound, num_threads, unroll_factor / 2, num_values,
        thread_id_x, xor_mask, mask_low, synced, read_shmem, write_shmem,
        comparator, /*force_write=*/false, check_bounds, is_merge_step,
        use_warp_pair1_stride ? pair1_stride : Pair1Stride::kDefault);
  };

  auto emit_rolled_merge_steps =
      [&](ImplicitLocOpBuilder& merge_b, Value tile_bound, Value num_steps,
          Value initial_merge_mask_low, ValueRange merge_tensors,
          Pair1Stride pair1_stride, bool check_bounds,
          bool post_sync) -> absl::StatusOr<SmallVector<Value>> {
    SmallVector<Value> inner_init_args;
    inner_init_args.reserve(merge_tensors.size() + 1);
    inner_init_args.push_back(initial_merge_mask_low);
    inner_init_args.append(merge_tensors.begin(), merge_tensors.end());
    ABSL_ASSIGN_OR_RETURN(
        SmallVector<Value> inner_results,
        EmitForWithStatus(
            merge_b, c0, num_steps, c1, inner_init_args,
            [&](ImplicitLocOpBuilder& step_b, Value /*step*/,
                ValueRange step_args) -> absl::StatusOr<SmallVector<Value>> {
              Value merge_mask_low = step_args.front();
              Value merge_xor_mask =
                  arith::AddIOp::create(step_b, merge_mask_low, c1);
              ABSL_ASSIGN_OR_RETURN(
                  SmallVector<Value> next_tensors,
                  emit_intra_step(step_b, tile_bound, merge_xor_mask,
                                  merge_mask_low, step_args.drop_front(),
                                  /*is_merge_step=*/true, pair1_stride,
                                  check_bounds, post_sync));
              Value next_mask_low =
                  arith::ShRUIOp::create(step_b, merge_mask_low, c1);
              SmallVector<Value> yield_args;
              yield_args.reserve(next_tensors.size() + 1);
              yield_args.push_back(next_mask_low);
              yield_args.append(next_tensors.begin(), next_tensors.end());
              return yield_args;
            }));
    return SmallVector<Value>(inner_results.begin() + 1, inner_results.end());
  };

  auto compute_tail_tile_bound = [&](ImplicitLocOpBuilder& bound_b) -> Value {
    if (dimension_to_sort_bound < tile_size) {
      return arith::ConstantIndexOp::create(bound_b, dimension_to_sort_bound);
    }
    Value bound = arith::ConstantIndexOp::create(bound_b, partial_tile_elems);
    if (partial_tile_elems != 0 &&
        active_tiles_bound != rounded_cluster_bound) {
      Value is_oob_tile = arith::CmpIOp::create(
          bound_b, arith::CmpIPredicate::sge, base_keys_index,
          arith::ConstantIndexOp::create(bound_b, active_tiles_bound));
      bound = arith::SelectOp::create(
          bound_b, is_oob_tile, arith::ConstantIndexOp::create(bound_b, 0),
          bound);
    }
    return bound;
  };

  auto emit_per_tile_intra =
      [&](ImplicitLocOpBuilder& phase_b, ValueRange initial_tensors,
          absl::FunctionRef<absl::StatusOr<SmallVector<Value>>(
              ImplicitLocOpBuilder&, ValueRange, Value /*tile_bound*/,
              int64_t /*stage_limit*/, bool /*check_bounds*/)>
              emit_tile_fn) -> absl::StatusOr<SmallVector<Value>> {
    if (full_tiles_bound == 0) {
      return emit_tile_fn(phase_b, initial_tensors,
                          compute_tail_tile_bound(phase_b), tail_tile_stages,
                          /*check_bounds=*/true);
    }
    if (full_tiles_bound == rounded_cluster_bound) {
      return emit_tile_fn(phase_b, initial_tensors,
                          arith::ConstantIndexOp::create(phase_b, tile_size),
                          intra_stages, /*check_bounds=*/false);
    }
    // Split per-tile so every full tile within a partial cluster still runs
    // the bounds-check-free unrolled intra-tile fast path.
    Value is_partial_or_oob_tile = arith::CmpIOp::create(
        phase_b, arith::CmpIPredicate::sge, base_keys_index,
        arith::ConstantIndexOp::create(phase_b, full_tiles_bound));
    return EmitIfElseWithStatus(
        phase_b, is_partial_or_oob_tile, TypeRange(initial_tensors),
        [&](ImplicitLocOpBuilder& then_b) {
          return emit_tile_fn(then_b, initial_tensors,
                              compute_tail_tile_bound(then_b), tail_tile_stages,
                              /*check_bounds=*/true);
        },
        [&](ImplicitLocOpBuilder& else_b) {
          return emit_tile_fn(else_b, initial_tensors,
                              arith::ConstantIndexOp::create(else_b, tile_size),
                              intra_stages, /*check_bounds=*/false);
        });
  };

  constexpr int64_t kUnrolledTailMasks = 6;
  static_assert(kUnrolledTailMasks == 6);

  auto emit_intra_sort_tile =
      [&](ImplicitLocOpBuilder& tile_b, ValueRange tile_tensors,
          Value tile_bound, int64_t stage_limit,
          bool check_bounds) -> absl::StatusOr<SmallVector<Value>> {
    SmallVector<Value> tensors(tile_tensors.begin(), tile_tensors.end());
    if (stage_limit <= 0) {
      return tensors;
    }
    if (check_bounds) {
      // Keep bounds-checked loops rolled with default pair-1 stride so LLVM
      // LICM does not hoist unrolled tail indices and bounds predicates across
      // stages, preserving high SM occupancy when combined with the full-tile
      // fast path.
      return EmitForWithStatus(
          tile_b, c0, arith::ConstantIndexOp::create(tile_b, stage_limit), c1,
          tensors,
          [&](ImplicitLocOpBuilder& stage_b, Value stage,
              ValueRange stage_tensors) -> absl::StatusOr<SmallVector<Value>> {
            Value flip_block_size = arith::ShLIOp::create(stage_b, c1, stage);
            Value flip_mask_low =
                arith::SubIOp::create(stage_b, flip_block_size, c1);
            Value flip_xor_mask = arith::SubIOp::create(
                stage_b, arith::ShLIOp::create(stage_b, flip_block_size, c1),
                c1);
            ABSL_ASSIGN_OR_RETURN(
                SmallVector<Value> rolled_tensors,
                emit_intra_step(stage_b, tile_bound, flip_xor_mask,
                                flip_mask_low, stage_tensors,
                                /*is_merge_step=*/false, Pair1Stride::kDefault,
                                /*check_bounds=*/true, /*post_sync=*/false));
            Value initial_merge_mask_low =
                arith::ShRUIOp::create(stage_b, flip_mask_low, c1);
            return emit_rolled_merge_steps(
                stage_b, tile_bound, stage, initial_merge_mask_low,
                rolled_tensors, Pair1Stride::kDefault, /*check_bounds=*/true,
                /*post_sync=*/false);
          });
    }

    // Unroll intra-warp mask steps (mask in [0, 6)) with compile-time
    // constants so bit-scatter masks, shared-memory offsets, and vectorized
    // loads/stores fold into constants, while ensuring every remaining rolled
    // step has block_size >= 64 (and thus a constant +32 pair-1 stride).
    int64_t unrolled_stages = std::min(stage_limit, kUnrolledTailMasks);
    for (int64_t stage = 0; stage < unrolled_stages; ++stage) {
      for (int64_t step = 0; step <= stage; ++step) {
        int64_t mask = stage - step;
        int64_t block_size = int64_t{1} << mask;
        int64_t xor_mask = step == 0 ? (block_size << 1) - 1 : block_size;
        ABSL_ASSIGN_OR_RETURN(
            tensors,
            emit_intra_step(
                tile_b, tile_bound,
                arith::ConstantIndexOp::create(tile_b, xor_mask),
                arith::ConstantIndexOp::create(tile_b, block_size - 1), tensors,
                /*is_merge_step=*/step > 0, Pair1Stride::kWithinDoubleWarp64,
                /*check_bounds=*/false, /*post_sync=*/false));
      }
    }

    if (stage_limit > kUnrolledTailMasks) {
      ABSL_ASSIGN_OR_RETURN(
          tensors,
          EmitForWithStatus(
              tile_b,
              arith::ConstantIndexOp::create(tile_b, kUnrolledTailMasks),
              arith::ConstantIndexOp::create(tile_b, stage_limit), c1, tensors,
              [&](ImplicitLocOpBuilder& stage_b, Value stage,
                  ValueRange stage_tensors)
                  -> absl::StatusOr<SmallVector<Value>> {
                // Emit step 0 (flip step) separately so the inner merge loop
                // has xor_mask == block_size on every iteration and avoids
                // loop peeling.
                Value flip_block_size =
                    arith::ShLIOp::create(stage_b, c1, stage);
                Value flip_mask_low =
                    arith::SubIOp::create(stage_b, flip_block_size, c1);
                Value flip_xor_mask = arith::SubIOp::create(
                    stage_b,
                    arith::ShLIOp::create(stage_b, flip_block_size, c1), c1);
                ABSL_ASSIGN_OR_RETURN(
                    SmallVector<Value> rolled_tensors,
                    emit_intra_step(stage_b, tile_bound, flip_xor_mask,
                                    flip_mask_low, stage_tensors,
                                    /*is_merge_step=*/false,
                                    Pair1Stride::kFlip32,
                                    /*check_bounds=*/false,
                                    /*post_sync=*/false));

                Value initial_merge_mask_low =
                    arith::ShRUIOp::create(stage_b, flip_mask_low, c1);
                Value num_merge_steps =
                    arith::SubIOp::create(stage_b, stage,
                                          arith::ConstantIndexOp::create(
                                              stage_b, kUnrolledTailMasks));
                ABSL_ASSIGN_OR_RETURN(
                    rolled_tensors,
                    emit_rolled_merge_steps(
                        stage_b, tile_bound, num_merge_steps,
                        initial_merge_mask_low, rolled_tensors,
                        Pair1Stride::kMerge32, /*check_bounds=*/false,
                        /*post_sync=*/false));

                for (int64_t mask = kUnrolledTailMasks - 1; mask >= 0; --mask) {
                  int64_t block_size = int64_t{1} << mask;
                  ABSL_ASSIGN_OR_RETURN(
                      rolled_tensors,
                      emit_intra_step(
                          stage_b, tile_bound,
                          arith::ConstantIndexOp::create(stage_b, block_size),
                          arith::ConstantIndexOp::create(stage_b,
                                                         block_size - 1),
                          rolled_tensors, /*is_merge_step=*/true,
                          Pair1Stride::kWithinDoubleWarp64,
                          /*check_bounds=*/false, /*post_sync=*/false));
                }
                return rolled_tensors;
              }));
    }
    return tensors;
  };

  auto emit_intra_merge_tile =
      [&](ImplicitLocOpBuilder& tile_b, ValueRange tile_tensors,
          Value tile_bound, int64_t stage_limit, bool check_bounds,
          bool post_sync) -> absl::StatusOr<SmallVector<Value>> {
    SmallVector<Value> tensors(tile_tensors.begin(), tile_tensors.end());
    if (stage_limit <= 0) {
      return tensors;
    }
    if (check_bounds ||
        (post_sync &&
         kernel_spec.kind == BitonicSortKernelSpec::Kind::kIntraTileSort)) {
      // Keep bounds-checked loops and post-cluster merges inside
      // kIntraTileSort rolled with default pair-1 stride so LLVM does not
      // hoist bounds predicates or keep stage 0..5 shared-memory pointers live
      // from Phase 1 across cluster barriers (which causes register spills).
      Value initial_merge_mask_low = arith::ConstantIndexOp::create(
          tile_b, (int64_t{1} << (stage_limit - 1)) - 1);
      return emit_rolled_merge_steps(
          tile_b, tile_bound,
          arith::ConstantIndexOp::create(tile_b, stage_limit),
          initial_merge_mask_low, tensors, Pair1Stride::kDefault, check_bounds,
          post_sync);
    }
    for (int64_t mask = stage_limit - 1; mask >= 0; --mask) {
      int64_t block_size = int64_t{1} << mask;
      Pair1Stride pair1_stride = block_size >= 64
                                     ? Pair1Stride::kMerge32
                                     : Pair1Stride::kWithinDoubleWarp64;
      ABSL_ASSIGN_OR_RETURN(
          tensors, emit_intra_step(
                       tile_b, tile_bound,
                       arith::ConstantIndexOp::create(tile_b, block_size),
                       arith::ConstantIndexOp::create(tile_b, block_size - 1),
                       tensors, /*is_merge_step=*/true, pair1_stride,
                       /*check_bounds=*/false, post_sync));
    }
    return tensors;
  };

  if (kernel_spec.kind == BitonicSortKernelSpec::Kind::kIntraTileSort) {
    ABSL_ASSIGN_OR_RETURN(
        shmem_tensors,
        emit_per_tile_intra(b, shmem_tensors, emit_intra_sort_tile));
  }

  if (has_cluster_step) {
    int64_t block_size = int64_t{1} << intra_stages;
    int64_t cluster_xor_mask =
        kernel_spec.kind == BitonicSortKernelSpec::Kind::kIntraTileSort
            ? (block_size << 1) - 1
            : block_size;
    Value xor_mask_val = index_const(cluster_xor_mask);
    Value block_size_val = index_const(block_size);

    auto emit_cluster_phase =
        [&](ImplicitLocOpBuilder& phase_b, ValueRange phase_tensors,
            Value cluster_bound,
            bool check_bounds) -> absl::StatusOr<SmallVector<Value>> {
      auto pre_synced =
          SyncThreadsOp::create(phase_b, TypeRange(phase_tensors),
                                phase_tensors, mlir::gpu::BarrierScope::Cluster)
              .getResults();
      ClusterStepInfo cluster_info{cta_base, tile_size, block_size_val,
                                   read_cluster_shmem, write_cluster_shmem};
      ABSL_ASSIGN_OR_RETURN(
          SmallVector<Value> updated,
          EmitCompareLoopBody(
              phase_b, cluster_bound, num_threads, unroll_factor / 2,
              num_values, thread_id_x, xor_mask_val, block_size_val, pre_synced,
              read_shmem, write_shmem, comparator, /*force_write=*/false,
              check_bounds, /*is_merge_step=*/false, Pair1Stride::kDefault,
              cluster_info));
      auto post_synced =
          SyncThreadsOp::create(phase_b, TypeRange(updated), updated,
                                mlir::gpu::BarrierScope::Cluster)
              .getResults();
      return SmallVector<Value>(post_synced.begin(), post_synced.end());
    };

    if (full_clusters_bound == 0) {
      ABSL_ASSIGN_OR_RETURN(
          shmem_tensors,
          emit_cluster_phase(b, shmem_tensors,
                             index_const(dimension_to_sort_bound),
                             /*check_bounds=*/true));
    } else if (full_clusters_bound == rounded_cluster_bound) {
      ABSL_ASSIGN_OR_RETURN(
          shmem_tensors,
          emit_cluster_phase(b, shmem_tensors, index_const(cluster_tile_size),
                             /*check_bounds=*/false));
    } else {
      // Split once on is_last_cluster so full clusters run without bounds
      // checks and cluster synchronization remains cluster-uniform.
      Value is_last_cluster =
          arith::CmpIOp::create(b, arith::CmpIPredicate::sge, base_keys_index,
                                index_const(full_clusters_bound));
      int64_t tail_elem_bound = dimension_to_sort_bound % cluster_tile_size;
      ABSL_ASSIGN_OR_RETURN(
          shmem_tensors,
          EmitIfElseWithStatus(
              b, is_last_cluster, TypeRange(shmem_tensors),
              [&](ImplicitLocOpBuilder& then_b)
                  -> absl::StatusOr<SmallVector<Value>> {
                if (tail_elem_bound <= tile_size) {
                  // Only CTA 0 in the last cluster has in-bounds elements;
                  // skip cross-CTA DSMEM traffic and sync locally instead.
                  return sync_workgroup(then_b, shmem_tensors);
                }
                return emit_cluster_phase(
                    then_b, shmem_tensors,
                    arith::ConstantIndexOp::create(then_b, tail_elem_bound),
                    /*check_bounds=*/true);
              },
              [&](ImplicitLocOpBuilder& else_b) {
                return emit_cluster_phase(
                    else_b, shmem_tensors,
                    arith::ConstantIndexOp::create(else_b, cluster_tile_size),
                    /*check_bounds=*/false);
              }));
    }
  }

  bool needs_post_cluster_merge =
      kernel_spec.kind == BitonicSortKernelSpec::Kind::kIntraTileMerge ||
      has_cluster_step;
  bool post_sync = has_cluster_step;
  if (needs_post_cluster_merge) {
    ABSL_ASSIGN_OR_RETURN(
        shmem_tensors,
        emit_per_tile_intra(
            b, shmem_tensors,
            [&](ImplicitLocOpBuilder& tile_b, ValueRange tile_tensors,
                Value tile_bound, int64_t stage_limit, bool check_bounds) {
              return emit_intra_merge_tile(tile_b, tile_tensors, tile_bound,
                                           stage_limit, check_bounds,
                                           post_sync);
            }));
  }
  if (!post_sync) {
    shmem_tensors = sync_workgroup(b, shmem_tensors);
  }

  // Copy sorted tile from shared memory back to global memory.
  SmallVector<Value> updated_outputs(output_tensors.begin(),
                                     output_tensors.end());
  for (int64_t i = 0; i < unroll_factor; ++i) {
    Value offset = index_const(i);
    Value current_keys_index =
        arith::AddIOp::create(b, base_keys_index, offset);
    Value in_bounds =
        arith::CmpIOp::create(b, arith::CmpIPredicate::slt, current_keys_index,
                              index_const(dimension_to_sort_bound));
    updated_outputs = EmitUpdateIf(
        b, in_bounds, updated_outputs,
        [&](ImplicitLocOpBuilder& if_b) -> SmallVector<Value> {
          Value cache_index =
              arith::AddIOp::create(if_b, base_cache_index, offset);
          SmallVector<Value> elem_index(tiled_keys_index.begin(),
                                        tiled_keys_index.end());
          elem_index[dimension_to_sort] = current_keys_index;
          SmallVector<Value> updated(updated_outputs.begin(),
                                     updated_outputs.end());
          for (int64_t param = 0; param < num_values; ++param) {
            Value value = tensor::ExtractOp::create(if_b, shmem_tensors[param],
                                                    ValueRange{cache_index});
            updated[param] = tensor::InsertOp::create(
                if_b, value, updated[param], elem_index);
          }
          return updated;
        });
  }
  return updated_outputs;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateBitonicSortMlirModule(
    MLIRContext& mlir_context, const HloSortInstruction* sort,
    const HloInstruction* hlo_with_buffers,
    const BufferAssignment* buffer_assignment,
    absl::string_view entry_function_name,
    const BitonicSortKernelSpec& kernel_spec,
    const LaunchDimensions& launch_dimensions,
    int64_t num_iterations_in_sort_dim, int64_t tile_size,
    int64_t unroll_factor, bool emit_iota_operands) {
  mlir::OpBuilder builder(&mlir_context);
  auto loc =
      mlir::NameLoc::get(builder.getStringAttr(hlo_with_buffers->name()));
  mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(loc);

  bool force_64_bit = launch_dimensions.launch_bound() * unroll_factor >
                      std::numeric_limits<int32_t>::max();
  emitters::SetIndexDataLayout(*module, *hlo_with_buffers, force_64_bit);

  ABSL_ASSIGN_OR_RETURN(FuncOp entry_func,
                        emitters::EmitKernelApi(
                            *module, *hlo_with_buffers, buffer_assignment,
                            GetDefaultBufferAlignment(), entry_function_name));
  SetBackendKind(&mlir_context, entry_func, BackendKind::kGpu);

  emitters::PartitionedComputations computations(sort->to_apply(),
                                                 &mlir_context);
  ABSL_ASSIGN_OR_RETURN(
      auto call_targets,
      emitters::EmitPartitionedComputations(*module, computations));
  FuncOp comparator = call_targets(sort->to_apply()->root_instruction());

  ImplicitLocOpBuilder b(loc, entry_func);
  b.setInsertionPointToStart(entry_func.addEntryBlock());

  ValueRange output_args =
      entry_func.getArguments().drop_front(hlo_with_buffers->operand_count());
  SmallVector<Value> output_tensors(output_args.begin(), output_args.end());

  const Shape& keys_shape = sort->operand(0)->shape();
  int64_t rank = keys_shape.dimensions().size();
  int64_t dimension_to_sort = sort->sort_dimension();
  int64_t dimension_to_sort_bound = keys_shape.dimensions(dimension_to_sort);

  Shape iteration_shape = keys_shape;
  iteration_shape.set_dimensions(dimension_to_sort, num_iterations_in_sort_dim);
  std::vector<int64_t> minor_to_major;
  minor_to_major.reserve(rank);
  minor_to_major.push_back(dimension_to_sort);
  for (int64_t dim : LayoutUtil::MinorToMajor(keys_shape)) {
    if (dim != dimension_to_sort) {
      minor_to_major.push_back(dim);
    }
  }
  *iteration_shape.mutable_layout() = LayoutUtil::MakeLayout(minor_to_major);

  if (ShapeUtil::ElementsIn(iteration_shape) == 0) {
    mlir::func::ReturnOp::create(b, output_tensors);
    return module;
  }

  SmallVector<Value> thread_and_block_ids =
      MlirKernelEmitter::EmitThreadAndBlockIds(b, launch_dimensions);
  Value thread_id_x = thread_and_block_ids[0];

  IndexingMap indexing_map = emitters::GetDefaultWorkItemIndexingMap(
      launch_dimensions.AsWorkDimensions(), iteration_shape, &mlir_context);

  int64_t num_threads = launch_dimensions.thread_counts_per_block().x;
  int64_t num_values = sort->operand_count();

  if (kernel_spec.kind == BitonicSortKernelSpec::Kind::kIotaCopy) {
    // A sort dimension of size 1 needs no comparisons; only iota operands have
    // to be written to their output buffers.
    Value in_bounds =
        emitters::CheckConstraints(indexing_map, thread_and_block_ids, {}, b);
    ABSL_ASSIGN_OR_RETURN(
        output_tensors,
        EmitUpdateIfWithStatus(
            b, in_bounds, output_tensors,
            [&](ImplicitLocOpBuilder& if_b)
                -> absl::StatusOr<SmallVector<Value>> {
              SmallVector<Value, 3> keys_index = emitters::ApplyIndexing(
                  indexing_map, thread_and_block_ids, {}, if_b);
              SmallVector<Value> updated(output_tensors.begin(),
                                         output_tensors.end());
              for (int64_t operand = 0; operand < num_values; ++operand) {
                if (!HloPredicateIsOp<HloOpcode::kIota>(
                        sort->operand(operand))) {
                  continue;
                }
                ABSL_ASSIGN_OR_RETURN(auto iota_vals,
                                      emitters::EmitIota(sort->operand(operand),
                                                         keys_index, if_b));
                updated[operand] = tensor::InsertOp::create(
                    if_b, iota_vals.front(), updated[operand], keys_index);
              }
              return updated;
            }));
  } else if (kernel_spec.is_tiled()) {
    SmallVector<Value, 3> keys_index =
        emitters::ApplyIndexing(indexing_map, thread_and_block_ids, {}, b);
    int64_t cluster_size = launch_dimensions.cluster_dims().has_value()
                               ? launch_dimensions.cluster_dims()->x
                               : 1;
    ABSL_ASSIGN_OR_RETURN(
        output_tensors,
        EmitTiledCompareLoop(b, keys_index, thread_id_x, sort,
                             dimension_to_sort_bound, num_threads, kernel_spec,
                             output_tensors, emit_iota_operands, tile_size,
                             cluster_size, unroll_factor, comparator));
  } else {
    Value in_bounds =
        emitters::CheckConstraints(indexing_map, thread_and_block_ids, {}, b);
    ABSL_ASSIGN_OR_RETURN(
        output_tensors,
        EmitUpdateIfWithStatus(
            b, in_bounds, output_tensors,
            [&](ImplicitLocOpBuilder& if_b)
                -> absl::StatusOr<SmallVector<Value>> {
              SmallVector<Value, 3> keys_index = emitters::ApplyIndexing(
                  indexing_map, thread_and_block_ids, {}, if_b);
              auto read_global = [&](ImplicitLocOpBuilder& body_b,
                                     ValueRange tensors, int64_t operand,
                                     Value index) -> absl::StatusOr<Value> {
                SmallVector<Value> elem_index(keys_index.begin(),
                                              keys_index.end());
                elem_index[dimension_to_sort] = index;
                if (emit_iota_operands && HloPredicateIsOp<HloOpcode::kIota>(
                                              sort->operand(operand))) {
                  ABSL_ASSIGN_OR_RETURN(
                      auto iota_vals, emitters::EmitIota(sort->operand(operand),
                                                         elem_index, body_b));
                  return iota_vals.front();
                }
                return tensor::ExtractOp::create(body_b, tensors[operand],
                                                 elem_index)
                    .getResult();
              };
              auto write_global = [&](ImplicitLocOpBuilder& body_b,
                                      Value tensor, int64_t operand,
                                      Value index, Value value) -> Value {
                SmallVector<Value> elem_index(keys_index.begin(),
                                              keys_index.end());
                elem_index[dimension_to_sort] = index;
                return tensor::InsertOp::create(body_b, value, tensor,
                                                elem_index);
              };
              return EmitCompareLoopBody(
                  if_b,
                  arith::ConstantIndexOp::create(if_b, dimension_to_sort_bound),
                  /*num_threads=*/1, unroll_factor / 2, num_values,
                  keys_index[dimension_to_sort],
                  arith::ConstantIndexOp::create(
                      if_b, kernel_spec.xor_mask_or_num_stages),
                  arith::ConstantIndexOp::create(if_b,
                                                 kernel_spec.block_size - 1),
                  output_tensors, read_global, write_global, comparator,
                  /*force_write=*/emit_iota_operands,
                  /*needs_bounds_checks=*/dimension_to_sort_bound <
                          unroll_factor ||
                      !llvm::isPowerOf2_64(dimension_to_sort_bound),
                  /*is_merge_step=*/kernel_spec.xor_mask_or_num_stages ==
                      kernel_spec.block_size);
            }));
  }

  mlir::func::ReturnOp::create(b, output_tensors);
  return module;
}

}  // namespace

AsyncThunkSequence EmitBitonicSort(const HloSortInstruction* sort,
                                   IrEmitterContext* ir_emitter_context) {
  std::string op_name(sort->name());

  int64_t dimension_to_sort = sort->sort_dimension();
  const Shape& keys_shape = sort->operand(0)->shape();
  uint64_t dimension_to_sort_bound = keys_shape.dimensions(dimension_to_sort);
  if (ShapeUtil::IsZeroElementArray(keys_shape)) {
    return ThunkSequence{};
  }
  int64_t num_stages = Log2Ceiling(dimension_to_sort_bound);
  VLOG(2) << op_name << " requires " << num_stages << " stages.";
  TF_RET_CHECK((uint64_t{1} << num_stages) >= dimension_to_sort_bound);
  TF_RET_CHECK(num_stages == 0 ||
               (uint64_t{1} << (num_stages - 1)) < dimension_to_sort_bound);

  uint64_t total_element_size = 0;
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    total_element_size += ShapeUtil::ByteSizeOfPrimitiveType(
        sort->operand(i)->shape().element_type());
  }
  const uint64_t kMaxSharedMemoryPerBlock =
      ir_emitter_context->gpu_device_info().shared_memory_per_block();
  uint64_t max_tile_size_fitting_into_shared_memory =
      kMaxSharedMemoryPerBlock / total_element_size;
  const uint64_t kMaxThreadsPerBlock =
      ir_emitter_context->gpu_device_info().threads_per_block_limit();
  uint64_t tile_size =
      std::min(std::min(kMaxThreadsPerBlock * kBitonicSortUnrollFactor,
                        max_tile_size_fitting_into_shared_memory),
               uint64_t{1} << num_stages);
  tile_size = uint64_t{1} << Log2Floor(tile_size);

  constexpr uint64_t kMaxClusterSize = 2;
  const auto* cuda_cc = ir_emitter_context->gpu_device_info()
                            .gpu_compute_capability()
                            .cuda_compute_capability();
  bool supports_clusters = cuda_cc != nullptr && cuda_cc->IsAtLeastHopper();
  uint64_t cluster_size = 1;
  if (supports_clusters && tile_size >= 64 &&
      (uint64_t{1} << num_stages) > tile_size) {
    uint64_t active_tiles = CeilOfRatio(dimension_to_sort_bound, tile_size);
    uint64_t batch_size =
        ShapeUtil::ElementsIn(keys_shape) / dimension_to_sort_bound;
    uint64_t unclustered_blocks = batch_size * active_tiles;
    int core_count = ir_emitter_context->gpu_device_info().core_count();
    uint64_t candidate_cluster_bound =
        RoundUpTo(dimension_to_sort_bound, tile_size * kMaxClusterSize);
    // Cap clusters at 2 CTAs (one TPC on Hopper/Blackwell) to avoid GPC
    // cluster-slot fragmentation, cross-TPC DSMEM crossbar contention, and
    // elevated register pressure. Require > 1 CTA/SM to amortize cluster
    // synchronization and scheduling over multi-wave execution, and skip
    // clustering when OOB cluster padding exceeds 12.5% (either idle CTAs or
    // heavily asymmetric partial tiles).
    if (core_count > 0 &&
        unclustered_blocks > static_cast<uint64_t>(core_count) &&
        (candidate_cluster_bound - dimension_to_sort_bound) * 8 <=
            candidate_cluster_bound) {
      cluster_size = kMaxClusterSize;
    }
  }
  uint64_t cluster_tile_size = tile_size * cluster_size;

  Shape standard_iteration_shape = keys_shape;
  uint64_t standard_num_iterations_in_sort_dim =
      num_stages == 0 ? 1
                      : CeilOfRatio(uint64_t{1} << (num_stages - 1),
                                    kBitonicSortUnrollFactor / 2);
  standard_iteration_shape.set_dimensions(dimension_to_sort,
                                          standard_num_iterations_in_sort_dim);

  LaunchDimensions standard_launch_dimensions = CalculateLaunchDimensions(
      standard_iteration_shape, ir_emitter_context->gpu_device_info());

  uint64_t rounded_bound =
      RoundUpTo(dimension_to_sort_bound, cluster_tile_size);
  Shape iteration_shape = keys_shape;
  uint64_t num_iterations_in_sort_dim =
      CeilOfRatio(rounded_bound, kBitonicSortUnrollFactor);
  iteration_shape.set_dimensions(dimension_to_sort, num_iterations_in_sort_dim);
  uint64_t num_iterations = ShapeUtil::ElementsIn(iteration_shape);

  const uint64_t kThreadsPerBlock =
      std::max(uint64_t{1}, tile_size / kBitonicSortUnrollFactor);

  uint64_t num_blocks = CeilOfRatio(num_iterations, kThreadsPerBlock);
  std::optional<se::ClusterDim> cluster_dim =
      cluster_size > 1 ? std::make_optional(se::ClusterDim(cluster_size, 1, 1))
                       : std::nullopt;
  LaunchDimensions tiled_launch_dimensions(
      se::BlockDim(num_blocks, 1, 1), se::ThreadDim(kThreadsPerBlock, 1, 1),
      cluster_dim);
  VLOG(2) << absl::StreamFormat("%s launch dims: %d blocks, %d threads/block",
                                op_name, num_blocks, kThreadsPerBlock);

  bool is_fusion = sort->parent()->IsFusionComputation();
  const HloInstruction* hlo_with_buffers =
      is_fusion ? sort->parent()->FusionInstruction() : sort;
  ABSL_ASSIGN_OR_RETURN(auto kernel_arguments,
                        emitters::KernelArguments::Create(
                            ir_emitter_context->buffer_assignment(),
                            GetDefaultBufferAlignment(), hlo_with_buffers));

  struct CompiledSortKernel {
    GpuKernelSpec spec;
    std::shared_ptr<const std::vector<uint8_t>> cubin;
  };
  absl::flat_hash_map<std::pair<BitonicSortKernelSpec, bool>,
                      Future<CompiledSortKernel>>
      compiled_kernels;
  bool use_pdl = IsPdlEnabled(
      sort->GetModule()->config().debug_options(),
      ir_emitter_context->gpu_device_info().gpu_compute_capability());
  int devices_in_process =
      ir_emitter_context->gpu_topology().num_devices_per_process();

  auto get_or_compile_kernel =
      [&](const BitonicSortKernelSpec& kernel_spec,
          bool emit_iota) -> Future<CompiledSortKernel> {
    std::pair<BitonicSortKernelSpec, bool> cache_key{kernel_spec, emit_iota};
    if (auto it = compiled_kernels.find(cache_key);
        it != compiled_kernels.end()) {
      VLOG(2) << op_name << " reuses previously compiled kernel";
      return it->second;
    }

    LaunchDimensions launch_dimensions = kernel_spec.is_tiled()
                                             ? tiled_launch_dimensions
                                             : standard_launch_dimensions;
    std::string sanitized_kernel_name =
        ir_emitter_context->GetSanitizedUniqueName(op_name);

    BorrowedMlirContext borrowed_context =
        ir_emitter_context->BorrowMlirContext();
    MLIRContext* mlir_context = borrowed_context->get();
    mlir_context->appendDialectRegistry(
        MlirKernelEmitter::GetDialectRegistry());
    mlir_context->loadAllAvailableDialects();

    ABSL_ASSIGN_OR_RETURN(
        mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
        CreateBitonicSortMlirModule(
            *mlir_context, sort, hlo_with_buffers,
            &ir_emitter_context->buffer_assignment(), sanitized_kernel_name,
            kernel_spec, launch_dimensions,
            kernel_spec.is_tiled() ? num_iterations_in_sort_dim
                                   : standard_num_iterations_in_sort_dim,
            tile_size, kBitonicSortUnrollFactor, emit_iota));

    // KernelCompiler::CompileToTargetBinary inserts a PDL wait at the kernel
    // entry when PDL is enabled for this module and device, so launch with PDL
    // in that case.
    GpuKernelSpec spec(std::move(sanitized_kernel_name), launch_dimensions,
                       kernel_arguments, std::nullopt, use_pdl,
                       /*tma_metadata=*/{}, launch_dimensions.cluster_dims());
    MlirKernelDefinition kernel_def(
        spec, MlirKernelSource(nullptr, std::move(mlir_module)));
    Future<CompiledSortKernel> compiled_future =
        ir_emitter_context->kernel_compiler()
            ->CompileToTargetBinary(std::move(kernel_def),
                                    std::move(borrowed_context),
                                    *sort->GetModule(), /*unroll_factor=*/0)
            .Map([spec = std::move(spec)](
                     std::vector<uint8_t> cubin) mutable -> CompiledSortKernel {
              return CompiledSortKernel{
                  std::move(spec),
                  std::make_shared<const std::vector<uint8_t>>(
                      std::move(cubin)),
              };
            });
    compiled_kernels.emplace(std::move(cache_key), compiled_future);
    return compiled_future;
  };

  std::vector<tsl::Future<std::unique_ptr<Thunk>>> thunks;
  bool emit_iota_operands =
      absl::c_any_of(sort->operands(), HloPredicateIsOp<HloOpcode::kIota>);
  auto emit_thunk = [&](const BitonicSortKernelSpec& kernel_spec)
      -> tsl::Future<std::unique_ptr<Thunk>> {
    bool current_emit_iota = emit_iota_operands;
    emit_iota_operands = false;
    Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
        hlo_with_buffers, ir_emitter_context->GetNextThunkId());

    return get_or_compile_kernel(kernel_spec, current_emit_iota)
        .Map([thunk_info = std::move(thunk_info),
              devices_in_process](const CompiledSortKernel& compiled) mutable
                 -> absl::StatusOr<std::unique_ptr<Thunk>> {
          ABSL_ASSIGN_OR_RETURN(CustomKernel custom_kernel,
                                kernel::CreateSharedCubinCustomKernel(
                                    compiled.spec, compiled.cubin));
          return std::make_unique<CustomKernelThunk>(
              std::move(thunk_info), std::move(custom_kernel),
              compiled.spec.kernel_arguments(), devices_in_process,
              compiled.spec.use_pdl());
        });
  };

  if (num_stages == 0) {
    if (emit_iota_operands) {
      thunks.push_back(emit_thunk({BitonicSortKernelSpec::Kind::kIotaCopy}));
    }
  } else {
    int64_t tiled_stages = Log2Floor(cluster_tile_size);
    int64_t local_stages = std::min(num_stages, tiled_stages);
    if (local_stages >= 2) {
      thunks.push_back(emit_thunk(
          {BitonicSortKernelSpec::Kind::kIntraTileSort, local_stages}));
    } else {
      thunks.push_back(
          emit_thunk({BitonicSortKernelSpec::Kind::kSingleStep, 1, 1}));
    }
    for (int64_t stage = tiled_stages; stage < num_stages; ++stage) {
      for (int64_t mask = stage; mask >= tiled_stages; --mask) {
        int64_t block_size = int64_t{1} << mask;
        int64_t xor_mask = mask == stage ? (block_size << 1) - 1 : block_size;
        thunks.push_back(emit_thunk(
            {BitonicSortKernelSpec::Kind::kSingleStep, xor_mask, block_size}));
      }
      if (tiled_stages >= 2) {
        thunks.push_back(emit_thunk(
            {BitonicSortKernelSpec::Kind::kIntraTileMerge, tiled_stages}));
      } else {
        thunks.push_back(
            emit_thunk({BitonicSortKernelSpec::Kind::kSingleStep, 1, 1}));
      }
    }
  }
  return tsl::JoinFutures(absl::MakeSpan(thunks))
      .Map([](std::vector<std::unique_ptr<Thunk>> thunks) {
        return ThunkSequence(std::move(thunks));
      });
}

AsyncThunkSequence SortFusion::Emit(IrEmitterContext& ir_emitter_context,
                                    const HloFusionInstruction& fusion) const {
  std::vector<BufferAllocation::Slice> src_buffers;
  std::vector<BufferAllocation::Slice> dst_buffers;
  std::vector<Shape> src_shapes;
  src_buffers.reserve(fusion.operand_count());
  dst_buffers.reserve(fusion.operand_count());
  src_shapes.reserve(fusion.operand_count());
  const HloSortInstruction* sort =
      Cast<HloSortInstruction>(fusion.fused_expression_root());
  Shape keys_shape = sort->operand(0)->shape();
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    // We assume that the layout of all involved operands and
    // outputs is the same.
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        keys_shape, sort->operand(i)->shape(),
        Layout::Equal().IgnoreMemorySpace().IgnoreElementSize()));
    ShapeIndex shape_index =
        sort->operand_count() > 1 ? ShapeIndex({i}) : ShapeIndex({});
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        keys_shape, ShapeUtil::GetSubshape(sort->shape(), shape_index),
        Layout::Equal().IgnoreMemorySpace().IgnoreElementSize()));
    // We expect only parameters or iotas as operand of sort.
    if (HloPredicateIsOp<HloOpcode::kParameter>(sort->operand(i))) {
      const HloInstruction* src_instr =
          fusion.operand(sort->operand(i)->parameter_number());
      ABSL_ASSIGN_OR_RETURN(
          BufferAllocation::Slice slice,
          ir_emitter_context.buffer_assignment().GetUniqueSlice(src_instr, {}));
      src_buffers.push_back(slice);
      src_shapes.push_back(sort->operand(i)->shape());
      ABSL_ASSIGN_OR_RETURN(
          slice, ir_emitter_context.buffer_assignment().GetUniqueSlice(
                     &fusion, shape_index));
      dst_buffers.push_back(slice);
    } else {
      TF_RET_CHECK(HloPredicateIsOp<HloOpcode::kIota>(sort->operand(i)));
    }
  }

  ThunkSequence thunks;
  for (int i = 0; i < src_buffers.size(); ++i) {
    if (src_buffers[i] != dst_buffers[i]) {
      thunks.emplace_back(std::make_unique<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              &fusion, ir_emitter_context.GetNextThunkId()),
          /*source_buffer=*/ShapedSlice{src_buffers[i], src_shapes[i]},
          /*destination_buffer=*/ShapedSlice{dst_buffers[i], src_shapes[i]},
          /*mem_size=*/src_buffers[i].size()));
    }
  }
  return EmitBitonicSort(sort, &ir_emitter_context)
      .Map([thunks = std::move(thunks)](ThunkSequence sort_thunks) mutable {
        thunks.insert(thunks.end(),
                      std::make_move_iterator(sort_thunks.begin()),
                      std::make_move_iterator(sort_thunks.end()));
        return std::move(thunks);
      });
}

}  // namespace gpu
}  // namespace xla
