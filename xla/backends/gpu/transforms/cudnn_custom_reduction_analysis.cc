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

#include "xla/backends/gpu/transforms/cudnn_custom_reduction_analysis.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

// The types that the lowering maps to cuDNN frontend types with an identity
// type (Custom_reduction_identity). The frontend also has UINT8 and UINT32
// identities, but the cuDNN fusion compiler does not map U8 or U32 tensors.
bool HasIdentityTypeMapping(PrimitiveType type) {
  switch (type) {
    case PRED:
    case S8:
    case S32:
    case S64:
    case F16:
    case BF16:
    case F32:
    case F64:
    case F8E4M3FN:
    case F8E5M2:
      return true;
    default:
      return false;
  }
}

// The types that the cuDNN frontend accepts as compile-time constants.
bool HasFrontendConstantType(PrimitiveType type) {
  return type == F16 || type == BF16 || type == F32 || type == F64 ||
         type == S32 || type == S64;
}

// Whether the lowering maps `op` to cuDNN frontend pointwise modes.
bool IsTranslatableReducerOp(const HloInstruction& op) {
  switch (op.opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kMultiply:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kClamp:
    case HloOpcode::kSelect:
      return true;
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
      // The frontend has logical modes, but no bitwise modes.
      return op.shape().element_type() == PRED;
    case HloOpcode::kCompare:
      // The frontend compares floating-point values in IEEE order only.
      return !(Cast<HloCompareInstruction>(&op)->comparison().IsTotalOrder() &&
               primitive_util::IsFloatingPointType(
                   op.operand(0)->shape().element_type()));
    default:
      return false;
  }
}

absl::Status CheckReducerBody(const HloInstruction& reduce, int64_t count) {
  const HloComputation& body = *reduce.to_apply();
  if (body.num_parameters() != 2 * count) {
    return absl::UnimplementedError(
        "Reducer parameter count does not match its streams.");
  }
  const HloInstruction* root = body.root_instruction();
  if ((count == 1 && root->shape().IsTuple()) ||
      (count > 1 && (root->opcode() != HloOpcode::kTuple ||
                     root->operand_count() != count))) {
    return absl::UnimplementedError("Unsupported reducer result structure.");
  }
  for (int64_t i = 0; i < count; ++i) {
    const PrimitiveType type = reduce.operand(i)->shape().element_type();
    const HloInstruction* output = count == 1 ? root : root->operand(i);
    for (int64_t number : {i, count + i}) {
      const Shape& parameter = body.parameter_instruction(number)->shape();
      if (!ShapeUtil::IsScalar(parameter) || parameter.element_type() != type) {
        return absl::UnimplementedError("Unsupported reducer parameter type.");
      }
    }
    if (!ShapeUtil::IsScalar(output->shape()) ||
        output->shape().element_type() != type) {
      return absl::UnimplementedError("Unsupported reducer output type.");
    }
  }
  for (const HloInstruction* op : body.instructions()) {
    if (op == root && op->opcode() == HloOpcode::kTuple) {
      continue;
    }
    const PrimitiveType type = op->shape().element_type();
    if (!ShapeUtil::IsScalar(op->shape()) || !HasIdentityTypeMapping(type) ||
        (op->opcode() == HloOpcode::kConstant &&
         !HasFrontendConstantType(type)) ||
        (op->opcode() != HloOpcode::kParameter &&
         op->opcode() != HloOpcode::kConstant &&
         !IsTranslatableReducerOp(*op))) {
      return absl::UnimplementedError(absl::StrCat(
          "Unsupported scalar custom reduction body: ", op->ToString()));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status AnalyzeCudnnCustomReduction(const HloInstruction& reduce) {
  if (reduce.opcode() != HloOpcode::kReduce || reduce.operand_count() == 0 ||
      reduce.operand_count() % 2 != 0) {
    return absl::UnimplementedError(
        "Expected an HLO reduction with input and identity streams.");
  }
  const int64_t count = reduce.operand_count() / 2;
  if (reduce.dimensions().empty()) {
    return absl::UnimplementedError("Custom reduction requires nonempty axes.");
  }
  const Shape& input_shape = reduce.operand(0)->shape();
  const int64_t rank = input_shape.dimensions_size();
  std::vector<int64_t> axes(reduce.dimensions().begin(),
                            reduce.dimensions().end());
  absl::c_sort(axes);
  for (size_t i = 0; i < axes.size(); ++i) {
    if (axes[i] < 0 || axes[i] >= rank || (i != 0 && axes[i] == axes[i - 1])) {
      return absl::UnimplementedError(
          "Custom reduction axes must be distinct input dimensions.");
    }
  }
  if ((count == 1 && !reduce.shape().IsArray()) ||
      (count > 1 && (!reduce.shape().IsTuple() ||
                     reduce.shape().tuple_shapes_size() != count))) {
    return absl::UnimplementedError(
        "Unsupported custom reduction result structure.");
  }
  std::vector<int64_t> result_dimensions;
  for (int64_t axis = 0; axis < rank; ++axis) {
    if (!absl::c_linear_search(axes, axis)) {
      result_dimensions.push_back(input_shape.dimensions(axis));
    }
  }
  for (int64_t i = 0; i < count; ++i) {
    const Shape& shape = reduce.operand(i)->shape();
    const HloInstruction* init = reduce.operand(count + i);
    if (!shape.IsArray() || shape.is_dynamic() ||
        ShapeUtil::IsZeroElementArray(shape) ||
        !ShapeUtil::SameDimensions(shape, input_shape)) {
      return absl::UnimplementedError(
          "Unsupported custom reduction input shape.");
    }
    if (!HasIdentityTypeMapping(shape.element_type())) {
      return absl::UnimplementedError(absl::StrCat(
          "Custom reduction stream type ",
          primitive_util::LowercasePrimitiveTypeName(shape.element_type()),
          " has no cuDNN identity type mapping."));
    }
    if (!init->IsConstant() || !ShapeUtil::IsScalar(init->shape()) ||
        init->shape().element_type() != shape.element_type()) {
      return absl::UnimplementedError(
          "Custom reduction requires a typed constant scalar identity.");
    }
    const Shape& result =
        count == 1 ? reduce.shape() : reduce.shape().tuple_shapes(i);
    if (!result.IsArray() || result.is_dynamic() ||
        result.element_type() != shape.element_type() ||
        !absl::c_equal(result.dimensions(), result_dimensions)) {
      return absl::UnimplementedError(
          "Unsupported custom reduction result shape or type.");
    }
  }
  // The reducer must be associative and commutative, and the inits must be an
  // identity of the reducer on either side: cuDNN combines elements in any
  // grouping and order (see the reducer subgraph contract in cudnn_graph.h).
  // This is the caller's contract and is not checked.
  return CheckReducerBody(reduce, count);
}

std::string CudnnCustomReductionSemanticKey(const HloComputation& body) {
  std::string key = absl::StrCat(
      "xla.reduce.v1:",
      body.ToString(
          HloPrintOptions::Fingerprint().set_print_only_essential_constants(
              false)));
  // HLO text need not distinguish NaN payloads. Include exact typed constant
  // bytes as printable hex so the FE cache key preserves those differences.
  for (const HloInstruction* op : body.MakeInstructionPostOrder()) {
    if (op->opcode() == HloOpcode::kConstant) {
      absl::StrAppend(
          &key, ":constant:",
          absl::BytesToHexString(op->literal().ToProto().SerializeAsString()));
    }
  }
  return key;
}

}  // namespace xla::gpu
