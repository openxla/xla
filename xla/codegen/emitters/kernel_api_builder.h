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

#ifndef XLA_CODEGEN_EMITTERS_KERNEL_API_BUILDER_H_
#define XLA_CODEGEN_EMITTERS_KERNEL_API_BUILDER_H_

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"

namespace xla::emitters {

inline constexpr absl::string_view kXlaInvariantAttr = "xla.invariant";
inline constexpr absl::string_view kXlaAliasedAttr = "xla.aliased";

// Emits and return the kernel entry function into the provided module it for
// the given HLO instruction.
// The function will be given the provided name and the arguments will be
// annotated with the buffer indices.
absl::StatusOr<mlir::func::FuncOp> EmitKernelApi(
    mlir::ModuleOp module, const HloInstruction& hlo_instruction,
    const BufferAssignment* buffer_assignment,
    const KernelArguments::BufferAlignment& buffer_alignment,
    absl::string_view entry_function_name);

void SetIndexDataLayout(mlir::ModuleOp module,
                        const HloInstruction& hlo_instruction,
                        bool force_64_bit = false);

// Changes the order in which consecutive work items visit the elements of the
// output. By default work items walk the output in memory order. The
// physically linearized output is viewed as the row-major array
//
//   [outer, blocked_size, middle_size, minor_size]
//
// and is instead traversed as
//
//   [outer, blocked_size / block_size, middle_size, block_size, minor_size]
//
// i.e. the `block_size` consecutive indices of the blocked dimension are moved
// right above the minor dimension. For a transpose that swaps the blocked and
// the middle dimensions this makes the work items that run at the same time
// read `block_size * minor_size` contiguous elements of the operand instead of
// `minor_size` elements that are `blocked_size * minor_size` elements apart.
//
// The traversal order is a bijection of the element indices, so it does not
// change which elements are computed.
struct BlockedDimensionOrder {
  // Size of the dimension that is split into blocks.
  int64_t blocked_size;
  // Number of consecutive indices of the blocked dimension that are visited
  // together. Must divide `blocked_size`.
  int64_t block_size;
  // Size of the dimension between the blocked and the minor dimension.
  int64_t middle_size;
  // Number of elements in all dimensions that are more minor than the middle
  // dimension.
  int64_t minor_size;

  bool operator==(const BlockedDimensionOrder& other) const {
    return blocked_size == other.blocked_size &&
           block_size == other.block_size && middle_size == other.middle_size &&
           minor_size == other.minor_size;
  }
};

// Get the default indexing map for the given work dimensions, unroll factor,
// and output shape. If `order` is set, the work items traverse the output in
// the given order instead of the memory order.
IndexingMap GetDefaultWorkItemIndexingMap(
    const WorkDimensions& work_dimensions, const Shape& shape,
    mlir::MLIRContext* mlir_context,
    const std::optional<BlockedDimensionOrder>& order = std::nullopt);

// Emits the work group id ops annotated with the range of each dimension.
llvm::SmallVector<mlir::Value> EmitWorkGroupIds(
    mlir::ImplicitLocOpBuilder& builder, const NumWorkGroups& num_work_groups);

absl::StatusOr<CallTargetProvider> EmitPartitionedComputations(
    mlir::ModuleOp module, const PartitionedComputations& computations);

absl::StatusOr<KernelSpec> GetKernelSpec(
    absl::string_view entry_function_name,
    const HloInstruction& hlo_instruction,
    const BufferAssignment* buffer_assignment,
    const WorkDimensions& work_dimensions);

}  // namespace xla::emitters

#endif  // XLA_CODEGEN_EMITTERS_KERNEL_API_BUILDER_H_
