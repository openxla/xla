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

#include "xla/backends/gpu/libraries/cub/cub_sort_planner.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_policy.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

absl::Status ValidateKeyAndValueTypes(PrimitiveType key_type,
                                      std::optional<PrimitiveType> value_type) {
  if (!value_type.has_value()) {
    switch (key_type) {
      case BF16:
      case F16:
      case F32:
      case F64:
      case S8:
      case S16:
      case S32:
      case S64:
      case U8:
      case U16:
      case U32:
      case U64:
        return absl::OkStatus();
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported key type for CUDA CUB keys-only sort: ",
                         PrimitiveType_Name(key_type)));
    }
  }

  switch (key_type) {
    case U8:
    case U16:
    case S32:
    case U32:
    case U64:
    case F32:
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported key type for CUDA CUB key-value sort: ",
                       PrimitiveType_Name(key_type)));
  }

  if (!primitive_util::IsArrayType(*value_type)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported non-array value type for CUDA CUB sort: ",
                     PrimitiveType_Name(*value_type)));
  }
  const int val_bits = primitive_util::BitWidth(*value_type);
  if (val_bits != 16 && val_bits != 32 && val_bits != 64) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported value bit width (", val_bits,
                     ") for CUDA CUB sort: ", PrimitiveType_Name(*value_type)));
  }
  return absl::OkStatus();
}

absl::StatusOr<int64_t> CheckedAdd(int64_t a, int64_t b) {
  if (a < 0 || b < 0 || a > std::numeric_limits<int64_t>::max() - b) {
    return absl::InvalidArgumentError(
        absl::StrCat("Integer overflow adding ", a, " + ", b));
  }
  return a + b;
}

absl::StatusOr<int64_t> CheckedMultiply(int64_t a, int64_t b) {
  if (a < 0 || b < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Negative operand in multiplication: ", a, " * ", b));
  }
  if (a > 0 && b > std::numeric_limits<int64_t>::max() / a) {
    return absl::InvalidArgumentError(
        absl::StrCat("Integer overflow multiplying ", a, " * ", b));
  }
  return a * b;
}

absl::StatusOr<int64_t> CheckedRoundUpTo(int64_t value, int64_t align) {
  if (value < 0 || align <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid alignment rounding: ", value, ", ", align));
  }
  const int64_t rem = value % align;
  if (rem == 0) {
    return value;
  }
  return CheckedAdd(value, align - rem);
}

// Bump allocator placing independent scratch sub-buffers on 256-byte
// boundaries.
class ScratchBumpAllocator {
 public:
  absl::StatusOr<CubSortScratchRange> Allocate(int64_t size_bytes) {
    if (size_bytes < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Scratch allocation size must be non-negative, got ", size_bytes));
    }
    if (size_bytes == 0) {
      return CubSortScratchRange{};
    }
    ABSL_ASSIGN_OR_RETURN(
        const int64_t offset,
        CheckedRoundUpTo(current_offset_, kCubSortScratchAlignmentBytes));
    ABSL_ASSIGN_OR_RETURN(current_offset_, CheckedAdd(offset, size_bytes));
    return CubSortScratchRange{offset, size_bytes};
  }

  absl::StatusOr<int64_t> total_bytes() const {
    return CheckedRoundUpTo(current_offset_, kCubSortScratchAlignmentBytes);
  }

 private:
  int64_t current_offset_ = 0;
};

// Returns (src_role, dst_role) for `pass` in [0, num_passes) such that:
// - kIn is read only on pass 0 and never overwritten.
// - The final pass (pass == num_passes - 1) always writes kOut.
// - Every pass has src_role != dst_role.
std::pair<CubSortBufferRole, CubSortBufferRole> PingPongRoles(int pass,
                                                              int num_passes) {
  const int remaining = num_passes - 1 - pass;
  const CubSortBufferRole dst =
      (remaining % 2 == 0) ? CubSortBufferRole::kOut : CubSortBufferRole::kTmp;
  const CubSortBufferRole src =
      (pass == 0) ? CubSortBufferRole::kIn
                  : ((remaining % 2 == 0) ? CubSortBufferRole::kTmp
                                          : CubSortBufferRole::kOut);
  return {src, dst};
}

}  // namespace

absl::StatusOr<CubSortExecutionPlan> PlanCubSortExecution(
    const stream_executor::CudaComputeCapability& cc, PrimitiveType key_type,
    std::optional<PrimitiveType> value_type, int64_t num_items,
    int64_t batch_size, std::optional<int32_t> portion_size_override,
    std::optional<int32_t> max_grid_x_override) {
  ABSL_RETURN_IF_ERROR(ValidateKeyAndValueTypes(key_type, value_type));
  if (num_items < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("num_items must be non-negative, got ", num_items));
  }
  if (batch_size < 0 || (num_items > 0 && batch_size == 0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("batch_size must be positive when num_items > 0 "
                     "(and non-negative when num_items == 0), got ",
                     batch_size));
  }
  if (num_items > 0 && num_items % batch_size != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("num_items (", num_items,
                     ") must be divisible by batch_size (", batch_size, ")"));
  }

  const int32_t max_grid_x =
      max_grid_x_override.value_or(std::numeric_limits<int32_t>::max());
  if (max_grid_x <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("max_grid_x_override must be positive, got ", max_grid_x));
  }

  const int key_byte_size = primitive_util::ByteWidth(key_type);
  const int value_byte_size =
      value_type.has_value() ? primitive_util::ByteWidth(*value_type) : 0;
  const int sm_version = CanonicalCubSortPolicySmVersion(cc.major);

  CubSortExecutionPlan plan;
  plan.policy =
      GetCubRadixSortPolicySpec(sm_version, key_byte_size, value_byte_size,
                                /*is_float32_key=*/key_type == F32);
  if (portion_size_override.has_value()) {
    const int32_t tile_items = plan.policy.onesweep.tile_items();
    const int32_t p = *portion_size_override;
    if (p < tile_items || p > kCubSortMaxPortionLimit || p % tile_items != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "portion_size_override (", p,
          ") must be a positive multiple of onesweep.tile_items() (",
          tile_items, ") and <= ", kCubSortMaxPortionLimit));
    }
  }
  plan.key_type = key_type;
  plan.value_type = value_type;
  plan.num_items = num_items;
  plan.batch_size = batch_size;
  plan.segment_size = num_items == 0 ? 0 : num_items / batch_size;

  if (num_items == 0) {
    plan.mode = CubSortMode::kEmpty;
    return plan;
  }

  if (plan.segment_size == 1) {
    plan.mode = CubSortMode::kTrivialCopy;
    return plan;
  }

  if (plan.segment_size <= plan.policy.single_tile.tile_items()) {
    plan.mode = CubSortMode::kSingleTile;
    for (int64_t offset = 0; offset < batch_size; offset += max_grid_x) {
      const int32_t chunk = static_cast<int32_t>(
          std::min<int64_t>(batch_size - offset, max_grid_x));
      plan.single_tile_launches.push_back(
          CubSortSingleTileLaunch{offset, chunk});
    }
    return plan;
  }

  if (batch_size == 1) {
    plan.mode = CubSortMode::kOnesweep;
    const int total_bits = key_byte_size * 8;
    const int radix_bits = kCubSortOnesweepRadixBits;
    plan.num_passes = CeilOfRatio(total_bits, radix_bits);
    const int32_t tile_items = plan.policy.onesweep.tile_items();
    const int32_t default_portion_size =
        (kCubSortMaxPortionLimit / tile_items) * tile_items;
    plan.portion_size = portion_size_override.value_or(default_portion_size);

    plan.num_portions = CeilOfRatio(num_items, int64_t{plan.portion_size});
    if (plan.num_portions > std::numeric_limits<int32_t>::max()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Onesweep num_portions (", plan.num_portions,
                       ") exceeds int32_t limit (",
                       std::numeric_limits<int32_t>::max(), ")"));
    }
    const int32_t max_blocks_per_portion = static_cast<int32_t>(CeilOfRatio(
        std::min<int64_t>(num_items, plan.portion_size), int64_t{tile_items}));

    const int64_t bins_per_pass_bytes =
        static_cast<int64_t>(kCubSortOnesweepRadixDigits) *
        sizeof(CubGlobalOffset);
    const int64_t bins_all_passes_bytes =
        static_cast<int64_t>(plan.num_passes) * bins_per_pass_bytes;
    const int64_t max_lookback_bytes =
        static_cast<int64_t>(max_blocks_per_portion) *
        kCubSortOnesweepRadixDigits * sizeof(int32_t);

    ScratchBumpAllocator alloc;
    ABSL_ASSIGN_OR_RETURN(plan.bins_raw, alloc.Allocate(bins_all_passes_bytes));

    std::vector<CubSortScratchRange> bins_portions;
    bins_portions.reserve(plan.num_portions);
    for (int64_t p = 0; p < plan.num_portions; ++p) {
      ABSL_ASSIGN_OR_RETURN(CubSortScratchRange range,
                            alloc.Allocate(bins_all_passes_bytes));
      bins_portions.push_back(range);
    }
    plan.bins_portion0 = bins_portions.front();
    ABSL_ASSIGN_OR_RETURN(plan.lookback_max,
                          alloc.Allocate(max_lookback_bytes));
    // One 256-byte-strided counter slot per (portion, pass) so ExclusiveSum can
    // zero all dynamic tile counters in a single launch without per-pass
    // MemZero nodes, while each Onesweep launch owns a disjoint slice.
    ABSL_ASSIGN_OR_RETURN(const int64_t total_ctr_slots,
                          CheckedMultiply(plan.num_passes, plan.num_portions));
    ABSL_ASSIGN_OR_RETURN(
        const int64_t ctrs_bytes,
        CheckedMultiply(total_ctr_slots, kCubSortScratchAlignmentBytes));
    ABSL_ASSIGN_OR_RETURN(plan.ctrs_all, alloc.Allocate(ctrs_bytes));
    if (plan.num_passes > 1) {
      ABSL_ASSIGN_OR_RETURN(const int64_t keys_tmp_bytes,
                            CheckedMultiply(num_items, key_byte_size));
      ABSL_ASSIGN_OR_RETURN(plan.keys_tmp, alloc.Allocate(keys_tmp_bytes));
      if (value_type.has_value()) {
        ABSL_ASSIGN_OR_RETURN(const int64_t vals_tmp_bytes,
                              CheckedMultiply(num_items, value_byte_size));
        ABSL_ASSIGN_OR_RETURN(plan.vals_tmp, alloc.Allocate(vals_tmp_bytes));
      }
    }
    ABSL_ASSIGN_OR_RETURN(plan.scratch_bytes, alloc.total_bytes());

    plan.onesweep_launches.reserve(static_cast<size_t>(plan.num_passes) *
                                   static_cast<size_t>(plan.num_portions));
    for (int pass = 0; pass < plan.num_passes; ++pass) {
      const auto [src_role, dst_role] = PingPongRoles(pass, plan.num_passes);
      const int32_t current_bit = pass * radix_bits;
      const int32_t num_bits = std::min(radix_bits, total_bits - current_bit);
      const int64_t pass_bin_offset =
          static_cast<int64_t>(pass) * bins_per_pass_bytes;

      for (int64_t portion = 0; portion < plan.num_portions; ++portion) {
        const int64_t item_offset = portion * plan.portion_size;
        const int32_t portion_items = static_cast<int32_t>(
            std::min<int64_t>(num_items - item_offset, plan.portion_size));
        const int32_t num_blocks = static_cast<int32_t>(
            CeilOfRatio<int64_t>(portion_items, tile_items));

        CubSortOnesweepPassLaunch step;
        step.pass = pass;
        step.portion = portion;
        step.src_role = src_role;
        step.dst_role = dst_role;
        step.bins_in =
            CubSortScratchRange{bins_portions[portion].offset + pass_bin_offset,
                                bins_per_pass_bytes};
        if (portion + 1 < plan.num_portions) {
          step.bins_out = CubSortScratchRange{
              bins_portions[portion + 1].offset + pass_bin_offset,
              bins_per_pass_bytes};
        }
        step.lookback = CubSortScratchRange{
            plan.lookback_max.offset,
            static_cast<int64_t>(num_blocks) * kCubSortOnesweepRadixDigits *
                static_cast<int64_t>(sizeof(int32_t))};
        const int64_t ctr_slot = portion * plan.num_passes + pass;
        step.ctr = CubSortScratchRange{
            plan.ctrs_all.offset + ctr_slot * kCubSortScratchAlignmentBytes,
            sizeof(int32_t)};
        step.portion_item_offset = item_offset;
        step.portion_num_items = portion_items;
        step.grid_x = num_blocks;
        step.current_bit = current_bit;
        step.num_bits = num_bits;
        plan.onesweep_launches.push_back(step);
      }
    }
    return plan;
  }

  // Batched multi-pass segmented sort (batch_size > 1 && segment_size >
  // single_tile.tile_items()).
  if (plan.segment_size > std::numeric_limits<int32_t>::max()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Batched CUDA CUB sort segment_size (", plan.segment_size,
        ") exceeds int32_t limit (", std::numeric_limits<int32_t>::max(), ")"));
  }

  plan.mode = CubSortMode::kSegmentedMultiPass;
  const int total_bits = key_byte_size * 8;
  const int primary_bits = plan.policy.segmented.radix_bits;
  const int alt_bits = plan.policy.alt_segmented.radix_bits;
  plan.num_passes = CeilOfRatio(total_bits, primary_bits);
  const int max_alt_passes = plan.num_passes * primary_bits - total_bits;
  const int alt_end_bit = std::min(total_bits, max_alt_passes * alt_bits);

  ScratchBumpAllocator alloc;
  if (plan.num_passes > 1) {
    ABSL_ASSIGN_OR_RETURN(const int64_t keys_tmp_bytes,
                          CheckedMultiply(num_items, key_byte_size));
    ABSL_ASSIGN_OR_RETURN(plan.keys_tmp, alloc.Allocate(keys_tmp_bytes));
    if (value_type.has_value()) {
      ABSL_ASSIGN_OR_RETURN(const int64_t vals_tmp_bytes,
                            CheckedMultiply(num_items, value_byte_size));
      ABSL_ASSIGN_OR_RETURN(plan.vals_tmp, alloc.Allocate(vals_tmp_bytes));
    }
  }
  ABSL_ASSIGN_OR_RETURN(plan.scratch_bytes, alloc.total_bytes());

  int current_bit = 0;
  for (int pass = 0; pass < plan.num_passes; ++pass) {
    const bool use_alt = current_bit < alt_end_bit;
    const int radix_bits = use_alt ? alt_bits : primary_bits;
    const int pass_bits = std::min(radix_bits, total_bits - current_bit);
    const auto [src_role, dst_role] = PingPongRoles(pass, plan.num_passes);

    for (int64_t offset = 0; offset < batch_size; offset += max_grid_x) {
      const int32_t chunk = static_cast<int32_t>(
          std::min<int64_t>(batch_size - offset, max_grid_x));
      CubSortSegmentedPassLaunch step;
      step.pass = pass;
      step.use_alt_policy = use_alt;
      step.src_role = src_role;
      step.dst_role = dst_role;
      step.segment_offset = offset;
      step.grid_x = chunk;
      step.current_bit = current_bit;
      step.pass_bits = pass_bits;
      plan.segmented_launches.push_back(step);
    }
    current_bit += pass_bits;
  }

  return plan;
}

absl::StatusOr<int64_t> ComputeCudaCubSortScratchSize(
    const stream_executor::CudaComputeCapability& cc, PrimitiveType key_type,
    std::optional<PrimitiveType> value_type, int64_t num_items,
    int64_t batch_size) {
  ABSL_ASSIGN_OR_RETURN(
      CubSortExecutionPlan plan,
      PlanCubSortExecution(cc, key_type, value_type, num_items, batch_size));
  return plan.scratch_bytes;
}

}  // namespace xla::gpu
