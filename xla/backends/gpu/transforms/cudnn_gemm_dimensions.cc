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

#include "xla/backends/gpu/transforms/cudnn_gemm_dimensions.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/transforms/cudnn_fusion_utils.h"
#include "xla/hlo/analysis/shape_tracker.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/cudnn_support_utils.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

// The ports of the dot: its operands (by operand number) and its output. Every
// instruction is described relative to the port that it contributes to.
enum class DotPort : int8_t { kLhs, kRhs, kLhsScale, kRhsScale, kOutput };

// Instructions that change the logical shape and are tracked by ShapeTracker.
bool IsShapeOp(const HloInstruction& hlo) {
  return HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kReshape,
                          HloOpcode::kTranspose, HloOpcode::kBroadcast>(&hlo);
}

// Instructions that keep the logical shape.
bool IsElementwiseOrCopy(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCopy || hlo.IsElementwise();
}

bool HasUnitStrides(const HloInstruction& slice) {
  return absl::c_all_of(slice.slice_strides(),
                        [](int64_t stride) { return stride == 1; });
}

// Instructions of the computation that are not part of the cuDNN graph.
bool IsOutsideCuDnnGraph(const HloInstruction& hlo) {
  return hlo.IsCustomCall(kWorkspaceAllocationCustomCallTarget) ||
         (hlo.opcode() == HloOpcode::kTuple && hlo.IsRoot());
}

// Returns the single output of the computation, ignoring workspace allocation.
absl::StatusOr<const HloInstruction*> GetOutput(
    const HloComputation& computation) {
  const HloInstruction* root = computation.root_instruction();
  if (root->opcode() != HloOpcode::kTuple) {
    return root;
  }
  std::vector<const HloInstruction*> outputs;
  for (const HloInstruction* operand : root->operands()) {
    if (!operand->IsCustomCall(kWorkspaceAllocationCustomCallTarget)) {
      outputs.push_back(operand);
    }
  }
  if (outputs.size() != 1) {
    return absl::UnimplementedError(
        "Only GEMM fusions with a single output are supported.");
  }
  return outputs.front();
}

// For each of the three canonical cuDNN dimensions, the dimensions of a port of
// the dot that it consists of.
using CanonicalDims = std::array<std::vector<int64_t>, 3>;
using PortToCanonicalDims = absl::flat_hash_map<DotPort, CanonicalDims>;

absl::StatusOr<PortToCanonicalDims> GetCanonicalDimensions(
    const HloInstruction& dot) {
  std::vector<DotOperandDims> operands;
  if (dot.opcode() == HloOpcode::kScaledDot) {
    ABSL_ASSIGN_OR_RETURN((std::array<DotOperandDims, 4> dims),
                          DotOperandDims::FromScaledDot(&dot));
    operands.assign(dims.begin(), dims.end());
  } else {
    ABSL_ASSIGN_OR_RETURN((std::array<DotOperandDims, 2> dims),
                          DotOperandDims::FromDot(&dot));
    operands.assign(dims.begin(), dims.end());
  }
  auto indices = [](const DotOperandDims& dims,
                    DotOperandDims::Category category) {
    absl::Span<const int64_t> span = dims.Indices(category);
    return std::vector<int64_t>(span.begin(), span.end());
  };

  PortToCanonicalDims result;
  for (int64_t i = 0; i < operands.size(); ++i) {
    const DotPort port = static_cast<DotPort>(i);
    const std::vector<int64_t> batch =
        indices(operands[i], DotOperandDims::kBatch);
    const std::vector<int64_t> nc =
        indices(operands[i], DotOperandDims::kNonContracting);
    const std::vector<int64_t> k =
        indices(operands[i], DotOperandDims::kContracting);
    // cuDNN multiplies [batch, M, K] by [batch, K, N].
    const bool is_lhs = port == DotPort::kLhs || port == DotPort::kLhsScale;
    result[port] =
        is_lhs ? CanonicalDims{batch, nc, k} : CanonicalDims{batch, k, nc};
  }
  // The dot is [batch..., M..., N...].
  const int64_t num_batch = operands[0].Rank(DotOperandDims::kBatch);
  const int64_t num_m = operands[0].Rank(DotOperandDims::kNonContracting);
  CanonicalDims& dot_dims = result[DotPort::kOutput];
  for (int64_t dim = 0; dim < dot.shape().dimensions().size(); ++dim) {
    const int64_t position =
        dim < num_batch ? 0 : (dim < num_batch + num_m ? 1 : 2);
    dot_dims[position].push_back(dim);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Step 1: relate every instruction to a port of the dot.
// ---------------------------------------------------------------------------

// Ranges of the dimensions of a port of the dot that a parameter is sliced to.
struct SliceRanges {
  std::vector<int64_t> starts;
  std::vector<int64_t> sizes;

  bool operator==(const SliceRanges& other) const {
    return starts == other.starts && sizes == other.sizes;
  }
};

// How the shape of an instruction gets to a port of the dot.
struct PortPath {
  DotPort port;
  // Maps the logical shape of the instruction to the shape of `port` before
  // `slices` are applied.
  ShapeTracker to_port;
  // Slices between the instruction and `port`, if any.
  std::optional<SliceRanges> slices;

  bool operator==(const PortPath& other) const {
    return port == other.port && to_port == other.to_port &&
           slices == other.slices;
  }
};

// All the ways in which each instruction relates to the dot: diamonds and
// instructions contributing to several dot ports have several paths.
using PortPaths =
    absl::flat_hash_map<const HloInstruction*, std::vector<PortPath>>;

// Returns the shape transformations from the dot to `output`.
absl::StatusOr<ShapeTracker> TrackFromDot(const HloInstruction& dot,
                                          const HloInstruction& output) {
  ShapeTracker tracker(dot.shape());
  for (const HloInstruction* hlo = &dot; hlo != &output;) {
    if (hlo->users().empty()) {
      return absl::UnimplementedError(
          absl::StrCat("Not computed from the dot: ", output.ToString()));
    }
    // Every instruction has to contribute to the output or it's invalid.
    hlo = hlo->users().front();
    if (IsShapeOp(*hlo)) {
      ABSL_RETURN_IF_ERROR(tracker.AppendInstruction(hlo));
    } else if (!IsElementwiseOrCopy(*hlo)) {
      return absl::UnimplementedError(absl::StrCat(
          "Unsupported instruction in a cuDNN GEMM fusion: ", hlo->ToString()));
    }
    tracker.SetElementType(hlo->shape().element_type());
  }
  return tracker;
}

// Applies a reshape to `target` to the dimensions of a sliced shape, which are
// described by their sizes before slicing (`unsliced`) and by the `starts` and
// the `sizes` of the slice. Dimensions that the reshape merges or splits can
// only be sliced if the slice remains a range of a single dimension.
absl::Status ReshapeSlicedDimensions(absl::Span<const int64_t> target,
                                     std::vector<int64_t>& unsliced,
                                     std::vector<int64_t>& starts,
                                     std::vector<int64_t>& sizes) {
  std::vector<int64_t> new_unsliced(target.begin(), target.end());
  std::vector<int64_t> new_starts(target.size(), 0);
  // Groups of source and target dimensions with the same number of elements.
  const auto groups = CommonFactors(sizes, target);
  for (int64_t group = 0; group + 1 < groups.size(); ++group) {
    const auto [src_begin, dst_begin] = groups[group];
    const auto [src_end, dst_end] = groups[group + 1];
    absl::Span<const int64_t> src_sizes =
        absl::MakeConstSpan(sizes).subspan(src_begin, src_end - src_begin);
    absl::Span<const int64_t> dst_sizes =
        target.subspan(dst_begin, dst_end - dst_begin);
    for (int64_t i = 0; i < src_sizes.size(); ++i) {
      const int64_t src = src_begin + i;
      if (unsliced[src] == sizes[src]) {
        continue;
      }
      // Only the major-most dimension of the group can be sliced, and the
      // slice has to be a range of the major-most target dimension.
      const absl::Status error = absl::UnimplementedError(
          "The slice is not a range of a dimension after a reshape.");
      if (Product(src_sizes.first(i)) != 1 || dst_sizes.empty()) {
        return error;
      }
      const auto it =
          absl::c_find_if(dst_sizes, [](int64_t size) { return size != 1; });
      const int64_t j = it == dst_sizes.end() ? 0 : it - dst_sizes.begin();
      const int64_t dst = dst_begin + j;
      const int64_t minor_src = Product(src_sizes.subspan(i + 1));
      const int64_t minor_dst = Product(dst_sizes.subspan(j + 1));
      if (new_unsliced[dst] != target[dst] ||
          (starts[src] * minor_src) % minor_dst != 0 ||
          (unsliced[src] * minor_src) % minor_dst != 0) {
        return error;
      }
      new_unsliced[dst] = unsliced[src] * minor_src / minor_dst;
      new_starts[dst] = starts[src] * minor_src / minor_dst;
    }
  }
  unsliced = std::move(new_unsliced);
  starts = std::move(new_starts);
  sizes.assign(target.begin(), target.end());
  return absl::OkStatus();
}

// Moves `slice` from the start of `to_port` to its end: makes `to_port` start
// at the shape of the operand of `slice` instead of the shape of `slice`, and
// returns the ranges of the dimensions of its (now larger) output shape that
// `slice` selects.
absl::StatusOr<SliceRanges> PropagateSlice(const HloInstruction& slice,
                                           ShapeTracker& to_port) {
  const Shape& operand_shape = slice.operand(0)->shape();
  absl::Span<const int64_t> slice_dims = slice.shape().dimensions();
  // The shape that the steps of `to_port` are replayed on.
  std::vector<int64_t> unsliced(operand_shape.dimensions().begin(),
                                operand_shape.dimensions().end());
  std::vector<int64_t> starts = slice.slice_starts();
  std::vector<int64_t> sizes(slice_dims.begin(), slice_dims.end());
  ShapeTracker unsliced_to_port(operand_shape);
  for (const ShapeTracker::Step& step : to_port.GetSteps()) {
    switch (step.type) {
      case ShapeTracker::Step::Type::kTranspose:
        unsliced = Permute(unsliced, step.dimensions);
        starts = Permute(starts, step.dimensions);
        sizes = Permute(sizes, step.dimensions);
        ABSL_RETURN_IF_ERROR(unsliced_to_port.AppendTranspose(step.dimensions));
        break;
      case ShapeTracker::Step::Type::kReshape:
        ABSL_RETURN_IF_ERROR(
            ReshapeSlicedDimensions(step.dimensions, unsliced, starts, sizes));
        ABSL_RETURN_IF_ERROR(unsliced_to_port.AppendReshape(unsliced));
        break;
      case ShapeTracker::Step::Type::kBroadcast: {
        // Broadcasted dimensions are not sliced.
        std::vector<int64_t> new_unsliced = step.dimensions;
        std::vector<int64_t> new_starts(step.dimensions.size(), 0);
        for (int64_t i = 0; i < step.broadcast_dimensions.size(); ++i) {
          new_unsliced[step.broadcast_dimensions[i]] = unsliced[i];
          new_starts[step.broadcast_dimensions[i]] = starts[i];
        }
        unsliced = std::move(new_unsliced);
        starts = std::move(new_starts);
        sizes = step.dimensions;
        ABSL_RETURN_IF_ERROR(unsliced_to_port.AppendBroadcast(
            unsliced, step.broadcast_dimensions));
        break;
      }
      case ShapeTracker::Step::Type::kReduce:
        return absl::UnimplementedError("Cannot slice a reduced tensor.");
    }
  }
  to_port = std::move(unsliced_to_port);
  return SliceRanges{std::move(starts), std::move(sizes)};
}

// Records `path` for `hlo` and walks towards the parameters, stopping at
// the dot. Instructions that cannot be walked through have to be parameters or
// constants.
absl::Status TraceToPort(const HloInstruction& hlo, PortPath path,
                         const HloInstruction& dot, PortPaths& paths) {
  path.to_port.SetElementType(hlo.shape().element_type());
  std::vector<PortPath>& hlo_paths = paths[&hlo];
  if (absl::c_linear_search(hlo_paths, path)) {
    return absl::OkStatus();
  }
  hlo_paths.push_back(path);
  if (&hlo == &dot) {
    return absl::OkStatus();
  }
  if (IsShapeOp(hlo)) {
    ABSL_RETURN_IF_ERROR(path.to_port.PrependInstruction(&hlo));
    return TraceToPort(*hlo.operand(0), std::move(path), dot, paths);
  }
  if (hlo.opcode() == HloOpcode::kSlice) {
    if (!HasUnitStrides(hlo)) {
      return absl::UnimplementedError(absl::StrCat(
          "Only unit-stride slices are supported: ", hlo.ToString()));
    }
    if (path.port == DotPort::kOutput) {
      // cuDNN builds plans for graphs with sliced operands after the dot, but
      // fails to execute them.
      return absl::UnimplementedError(absl::StrCat(
          "Slices are not supported after the dot: ", hlo.ToString()));
    }
    // Slices are tracked alongside the ShapeTracker as ranges of the
    // dimensions of the port, so they are moved to the port side of the
    // ShapeTracker.
    absl::StatusOr<SliceRanges> ranges = PropagateSlice(hlo, path.to_port);
    if (!ranges.ok()) {
      return absl::UnimplementedError(
          absl::StrCat("Unsupported slice: ", hlo.ToString(), ": ",
                       ranges.status().message()));
    }
    if (path.slices.has_value()) {
      // Slicing of an already sliced port means adding offsets.
      for (int64_t dim = 0; dim < ranges->starts.size(); ++dim) {
        ranges->starts[dim] += path.slices->starts[dim];
      }
      ranges->sizes = std::move(path.slices->sizes);
    }
    path.slices = *std::move(ranges);
    return TraceToPort(*hlo.operand(0), std::move(path), dot, paths);
  }
  if (IsElementwiseOrCopy(hlo)) {
    for (const HloInstruction* operand : hlo.operands()) {
      PortPath operand_path = path;
      // Scalar operands of clamp are implicitly broadcasted.
      if (ShapeUtil::IsScalar(operand->shape()) &&
          !ShapeUtil::IsScalar(hlo.shape())) {
        ABSL_RETURN_IF_ERROR(operand_path.to_port.PrependBroadcast({}));
      }
      ABSL_RETURN_IF_ERROR(
          TraceToPort(*operand, std::move(operand_path), dot, paths));
    }
    return absl::OkStatus();
  }
  if (!HloPredicateIsOp<HloOpcode::kParameter, HloOpcode::kConstant>(&hlo)) {
    return absl::UnimplementedError(absl::StrCat(
        "Unsupported instruction in a cuDNN GEMM fusion: ", hlo.ToString()));
  }
  return absl::OkStatus();
}

absl::StatusOr<PortPaths> TraceDotPortsToFusionEdges(
    const HloInstruction& dot, const HloInstruction& output) {
  ABSL_ASSIGN_OR_RETURN(ShapeTracker dot_to_output, TrackFromDot(dot, output));
  ABSL_ASSIGN_OR_RETURN(ShapeTracker output_to_dot,
                        dot_to_output.GetInverted());

  PortPaths paths;
  // The output and the instructions computed between the dot and the output.
  ABSL_RETURN_IF_ERROR(TraceToPort(
      output, {DotPort::kOutput, std::move(output_to_dot)}, dot, paths));
  // The instructions computing the operands of the dot.
  for (int64_t i = 0; i < dot.operand_count(); ++i) {
    const HloInstruction& operand = *dot.operand(i);
    ABSL_RETURN_IF_ERROR(TraceToPort(
        operand, {static_cast<DotPort>(i), ShapeTracker(operand.shape())}, dot,
        paths));
  }
  return paths;
}

// ---------------------------------------------------------------------------
// Step 2: check the rules of cuDNN GEMM graphs.
// ---------------------------------------------------------------------------

absl::Status ValidatePaths(const HloComputation& computation,
                           const PortPaths& paths) {
  for (const HloInstruction* hlo : computation.instructions()) {
    if (IsOutsideCuDnnGraph(*hlo)) {
      continue;
    }
    auto it = paths.find(hlo);
    if (it == paths.end()) {
      return absl::UnimplementedError(absl::StrCat(
          "Instruction does not contribute to the output: ", hlo->ToString()));
    }
    // Scalars have no layout to agree on.
    if (ShapeUtil::ElementsIn(hlo->shape()) == 1) {
      continue;
    }
    // The cuDNN graph aliases shape transformations away and has a single
    // tensor per instruction, so all paths must relate an instruction to the
    // same port in the same way. This rejects e.g. `d + transpose(d)`.
    // Paths are deduplicated when they are traced.
    if (it->second.size() > 1) {
      return absl::UnimplementedError(absl::StrCat(
          "Instruction relates to the dot in several ways: ", hlo->ToString()));
    }
  }
  return absl::OkStatus();
}

// ---------------------------------------------------------------------------
// Step 3: compute sizes and strides.
// ---------------------------------------------------------------------------

// Makes `to_port`, which starts at the logical shape of a tensor, start at the
// physical order of the tensor instead.
absl::Status PrependPhysicalLayout(const Shape& shape, ShapeTracker& to_port) {
  if (!shape.has_layout()) {
    return absl::OkStatus();
  }
  absl::Span<const int64_t> minor_to_major = shape.layout().minor_to_major();
  std::vector<int64_t> major_to_minor(minor_to_major.rbegin(),
                                      minor_to_major.rend());
  return to_port.PrependTranspose(InversePermutation(major_to_minor));
}

// Adjusts the strides of degenerate dimensions of `dims`, which do not affect
// the elements that are read, to the values that cuDNN expects.
void NormalizeStrides(bool has_batch_dimensions, CudnnTensorLayout& dims) {
  // 0 is always the batch dimension; 1 and 2 are the non-batch ones. cuDNN
  // relies on strides to determine layouts and gets confused when both
  // strides of non-batch dimensions are equal to 1 - this is the case for
  // tensors with 1-sized dimension like [A,1]. The stride of the 1-sized
  // dimension does not matter for correctness because there is no iteration
  // along this dimension, but setting it to A and representing the tensor as
  // its equivalent [1,A] helps cuDNN.
  if (dims.strides[1] == 1 && dims.strides[2] == 1) {
    const int one_sized_dim_idx = (dims.sizes[1] == 1) ? 1 : 2;
    dims.strides[one_sized_dim_idx] = dims.sizes[1] * dims.sizes[2];
  }
  // For tensors with an implicit batch dimension, set the batch stride to the
  // total packed size of the non-batch dimensions as required by cuDNN for
  // operations like block-scale dequantization.
  if (!has_batch_dimensions && dims.sizes[0] == 1) {
    dims.strides[0] = std::max(dims.sizes[1] * dims.strides[1],
                               dims.sizes[2] * dims.strides[2]);
  }
}

// Returns the sizes of the three canonical dimensions of a port with dimensions
// `port_dims`.
std::vector<int64_t> CanonicalPortSizes(absl::Span<const int64_t> port_dims,
                                        const CanonicalDims& canonical) {
  std::vector<int64_t> sizes(canonical.size(), 1);
  for (int position = 0; position < canonical.size(); ++position) {
    for (int64_t dim : canonical[position]) {
      sizes[position] *= port_dims[dim];
    }
  }
  return sizes;
}

// Returns how each of the three canonical dimensions of the port that `to_port`
// leads to iterates over the buffer of a tensor of shape `shape`.
absl::StatusOr<std::vector<ShapeTracker::BufferView>> CanonicalViews(
    const Shape& shape, ShapeTracker to_port, const CanonicalDims& canonical) {
  ABSL_RETURN_IF_ERROR(PrependPhysicalLayout(shape, to_port));
  // Transpose and reshape the port into its canonical rank-3 shape.
  const std::vector<int64_t> sizes =
      CanonicalPortSizes(to_port.output_shape().dimensions(), canonical);
  std::vector<int64_t> permutation;
  for (const std::vector<int64_t>& dims : canonical) {
    permutation.insert(permutation.end(), dims.begin(), dims.end());
  }
  ABSL_RETURN_IF_ERROR(to_port.AppendTranspose(permutation));
  ABSL_RETURN_IF_ERROR(to_port.AppendReshape(sizes));
  std::optional<std::vector<ShapeTracker::BufferView>> views =
      to_port.TryGetStridedView();
  if (!views.has_value()) {
    return absl::UnimplementedError(
        absl::StrCat("Not a strided view: ", to_port.DebugString()));
  }
  return *std::move(views);
}

// Computes the layout of a tensor of shape `shape` that relates to a port of
// the dot via `path`.
absl::StatusOr<CudnnTensorLayout> ComputeTensorLayout(
    const Shape& shape, const PortPath& path, const CanonicalDims& canonical) {
  ABSL_ASSIGN_OR_RETURN(std::vector<ShapeTracker::BufferView> views,
                        CanonicalViews(shape, path.to_port, canonical));
  CudnnTensorLayout layout;
  for (const ShapeTracker::BufferView& view : views) {
    if (absl::c_all_of(view.strides(), [](int64_t s) { return s == 0; })) {
      // Broadcasted and degenerate dimensions have size 1 and are broadcasted
      // by cuDNN.
      layout.strides.push_back(layout.strides.empty() ? 1
                                                      : layout.strides.back());
      layout.sizes.push_back(1);
    } else if (view.strides().size() == 1) {
      layout.strides.push_back(view.strides()[0]);
      layout.sizes.push_back(view.extents()[0]);
    } else {
      return absl::UnimplementedError(absl::StrCat(
          "Canonical dimension ", layout.sizes.size(),
          " is not a single strided dimension: ", path.to_port.DebugString()));
    }
  }
  if (path.slices.has_value()) {
    absl::Span<const int64_t> unsliced =
        path.to_port.output_shape().dimensions();
    std::vector<std::pair<int64_t, int64_t>> ranges;
    bool is_sliced = false;
    for (int position = 0; position < canonical.size(); ++position) {
      ranges.push_back({0, layout.sizes[position]});
      for (int64_t dim : canonical[position]) {
        const int64_t start = path.slices->starts[dim];
        const int64_t size = path.slices->sizes[dim];
        // Slices of broadcasted dimensions have no effect on the tensor.
        if (layout.sizes[position] == 1 ||
            (start == 0 && size == unsliced[dim])) {
          continue;
        }
        if (unsliced[dim] != layout.sizes[position]) {
          return absl::UnimplementedError(
              "Cannot slice a dimension that is not a whole canonical "
              "dimension.");
        }
        ranges.back() = {start, start + size};
        is_sliced = true;
      }
    }
    if (is_sliced) {
      layout.slices = std::move(ranges);
    }
  }
  NormalizeStrides(/*has_batch_dimensions=*/!canonical[0].empty(), layout);
  return layout;
}

}  // namespace

absl::StatusOr<CuDnnGemmDimensions> CuDnnGemmDimensions::Create(
    const HloInstruction& dot) {
  const HloComputation& computation = *dot.parent();
  ABSL_ASSIGN_OR_RETURN(PortToCanonicalDims canonical,
                        GetCanonicalDimensions(dot));
  ABSL_ASSIGN_OR_RETURN(const HloInstruction* output, GetOutput(computation));

  // Step 1: relate every instruction to a port of the dot.
  ABSL_ASSIGN_OR_RETURN(PortPaths paths,
                        TraceDotPortsToFusionEdges(dot, *output));

  // Step 2: check the rules of cuDNN GEMM graphs.
  ABSL_RETURN_IF_ERROR(ValidatePaths(computation, paths));

  // Step 3: compute sizes and strides.
  CuDnnGemmDimensions result;
  for (const HloInstruction* hlo : computation.instructions()) {
    auto it = paths.find(hlo);
    if (it == paths.end()) {
      continue;  // Outside of the cuDNN graph.
    }
    const PortPath& path = it->second.front();
    const CanonicalDims& port_canonical = canonical.at(path.port);
    // cuDNN cannot infer the output dimensions of a unary op whose operand is a
    // broadcast, e.g. `convert(broadcast(constant))`: broadcasts are not part
    // of the cuDNN graph, so the operand is a [1,1,1] tensor. The compiler sets
    // them to the canonical sizes of the port that the op contributes to.
    const Shape& port_shape =
        path.port == DotPort::kOutput
            ? dot.shape()
            : dot.operand(static_cast<int>(path.port))->shape();
    result.canonical_sizes_[hlo] =
        CanonicalPortSizes(port_shape.dimensions(), port_canonical);
    if (hlo == output || hlo->opcode() == HloOpcode::kParameter) {
      ABSL_ASSIGN_OR_RETURN(
          result.tensor_layouts_[hlo],
          ComputeTensorLayout(hlo->shape(), path, port_canonical));
    }
  }
  return result;
}

const CudnnTensorLayout* absl_nullable CuDnnGemmDimensions::TensorLayout(
    const HloInstruction& hlo) const {
  auto it = tensor_layouts_.find(&hlo);
  return it == tensor_layouts_.end() ? nullptr : &it->second;
}

const std::vector<int64_t>* absl_nullable CuDnnGemmDimensions::CanonicalSizes(
    const HloInstruction& hlo) const {
  auto it = canonical_sizes_.find(&hlo);
  return it == canonical_sizes_.end() ? nullptr : &it->second;
}

}  // namespace xla::gpu
