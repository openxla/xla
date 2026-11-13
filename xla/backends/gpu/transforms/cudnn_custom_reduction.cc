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

#include "xla/backends/gpu/transforms/cudnn_custom_reduction.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend.h"
#include "xla/backends/gpu/transforms/cudnn_custom_reduction_analysis.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/cuda/cudnn_custom_reduction_api.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace fe = cudnn_frontend;
namespace graph = fe::graph;
using Tensor = std::shared_ptr<graph::Tensor_attributes>;

// Compile-required: these frontend symbols do not exist before the custom
// reduction API.
#if XLA_HAS_CUDNN_CUSTOM_REDUCTION_API
using Attributes = graph::Custom_reduction_attributes;
using Identity = graph::Custom_reduction_identity;

// The cuDNN frontend type of a stream or a reducer value. The analysis admits
// only these types, so the fatal default is not reachable after a successful
// analysis.
fe::DataType_t ReductionType(PrimitiveType type) {
  switch (type) {
    case PRED:
      return fe::DataType_t::BYTE_BOOLEAN;
    case S8:
      return fe::DataType_t::INT8;
    case S32:
      return fe::DataType_t::INT32;
    case S64:
      return fe::DataType_t::INT64;
    case F16:
      return fe::DataType_t::HALF;
    case BF16:
      return fe::DataType_t::BFLOAT16;
    case F32:
      return fe::DataType_t::FLOAT;
    case F64:
      return fe::DataType_t::DOUBLE;
    case F8E4M3FN:
      return fe::DataType_t::FP8_E4M3;
    case F8E5M2:
      return fe::DataType_t::FP8_E5M2;
    default:
      LOG(FATAL) << "Unexpected custom reduction type "
                 << primitive_util::LowercasePrimitiveTypeName(type);
  }
}

fe::PointwiseMode_t CompareMode(ComparisonDirection direction) {
  switch (direction) {
    case ComparisonDirection::kEq:
      return fe::PointwiseMode_t::CMP_EQ;
    case ComparisonDirection::kNe:
      return fe::PointwiseMode_t::CMP_NEQ;
    case ComparisonDirection::kLt:
      return fe::PointwiseMode_t::CMP_LT;
    case ComparisonDirection::kLe:
      return fe::PointwiseMode_t::CMP_LE;
    case ComparisonDirection::kGt:
      return fe::PointwiseMode_t::CMP_GT;
    case ComparisonDirection::kGe:
      return fe::PointwiseMode_t::CMP_GE;
  }
  LOG(FATAL) << "Unexpected comparison direction";
}

fe::PointwiseMode_t ReducerMode(const HloInstruction& op) {
  switch (op.opcode()) {
    case HloOpcode::kAdd:
      return fe::PointwiseMode_t::ADD;
    case HloOpcode::kMultiply:
      return fe::PointwiseMode_t::MUL;
    case HloOpcode::kMaximum:
      return fe::PointwiseMode_t::MAX;
    case HloOpcode::kMinimum:
      return fe::PointwiseMode_t::MIN;
    case HloOpcode::kSelect:
      return fe::PointwiseMode_t::BINARY_SELECT;
    case HloOpcode::kAnd:
      return fe::PointwiseMode_t::LOGICAL_AND;
    case HloOpcode::kOr:
      return fe::PointwiseMode_t::LOGICAL_OR;
    case HloOpcode::kXor:
      // PRED only: a xor b is a != b.
      return fe::PointwiseMode_t::CMP_NEQ;
    case HloOpcode::kCompare:
      return CompareMode(op.comparison_direction());
    default:
      LOG(FATAL) << "Unexpected custom reduction op " << op.ToString();
  }
}

template <PrimitiveType Type, typename Storage>
Storage ScalarBits(const HloInstruction& constant) {
  using Native = primitive_util::NativeTypeOf<Type>;
  const Native native = constant.literal().GetFirstElement<Native>();
  static_assert(sizeof(Native) == sizeof(Storage));
  Storage storage;
  std::memcpy(&storage, &native, sizeof(storage));
  return storage;
}

Identity GetIdentity(const HloInstruction& constant) {
  switch (constant.shape().element_type()) {
    case PRED:
      return Identity(graph::Custom_reduction_byte_boolean{
          ScalarBits<PRED, uint8_t>(constant)});
    case S8:
      return Identity(ScalarBits<S8, int8_t>(constant));
    case S32:
      return Identity(ScalarBits<S32, int32_t>(constant));
    case S64:
      return Identity(ScalarBits<S64, int64_t>(constant));
    case F16:
      return Identity(ScalarBits<F16, ::half>(constant));
    case BF16:
      return Identity(ScalarBits<BF16, ::nv_bfloat16>(constant));
    case F32:
      return Identity(ScalarBits<F32, float>(constant));
    case F64:
      return Identity(ScalarBits<F64, double>(constant));
    case F8E4M3FN:
      return Identity(graph::Custom_reduction_fp8_e4m3{
          ScalarBits<F8E4M3FN, uint8_t>(constant)});
    case F8E5M2:
      return Identity(graph::Custom_reduction_fp8_e5m2{
          ScalarBits<F8E5M2, uint8_t>(constant)});
    default:
      LOG(FATAL) << "Unexpected custom reduction identity "
                 << constant.ToString();
  }
}

Tensor ScalarConstant(const HloInstruction& constant, graph::Graph& body) {
  switch (constant.shape().element_type()) {
    case F32:
      return body.tensor(ScalarBits<F32, float>(constant),
                         graph::ScalarType::COMPILE_TIME_CONST);
    case F16:
      return body.tensor(ScalarBits<F16, ::half>(constant),
                         graph::ScalarType::COMPILE_TIME_CONST);
    case BF16:
      return body.tensor(ScalarBits<BF16, ::nv_bfloat16>(constant),
                         graph::ScalarType::COMPILE_TIME_CONST);
    case F64:
      return body.tensor(ScalarBits<F64, double>(constant),
                         graph::ScalarType::COMPILE_TIME_CONST);
    case S32:
      return body.tensor(ScalarBits<S32, int32_t>(constant),
                         graph::ScalarType::COMPILE_TIME_CONST);
    case S64:
      return body.tensor(ScalarBits<S64, int64_t>(constant),
                         graph::ScalarType::COMPILE_TIME_CONST);
    default:
      LOG(FATAL) << "Unexpected custom reduction constant "
                 << constant.ToString();
  }
}

std::vector<Tensor> BuildReducer(const HloComputation& computation,
                                 Attributes::Graph_t body_graph,
                                 const std::vector<Tensor>& lhs,
                                 const std::vector<Tensor>& rhs) {
  const int64_t count = lhs.size();
  absl::flat_hash_map<const HloInstruction*, Tensor> values;
  for (const HloInstruction* op : computation.MakeInstructionPostOrder()) {
    if (op->opcode() == HloOpcode::kTuple) {
      continue;
    }
    if (op->opcode() == HloOpcode::kParameter) {
      const int64_t number = op->parameter_number();
      values[op] = number < count ? lhs[number] : rhs[number - count];
      continue;
    }
    if (op->opcode() == HloOpcode::kConstant) {
      values[op] = ScalarConstant(*op, *body_graph);
      continue;
    }
    const PrimitiveType result_type = op->shape().element_type();
    const PrimitiveType compute_type =
        op->opcode() == HloOpcode::kCompare
            ? op->operand(0)->shape().element_type()
            : result_type;
    const fe::DataType_t compute_data_type = compute_type == PRED
                                                 ? fe::DataType_t::INT32
                                                 : ReductionType(compute_type);
    const auto attributes = [&](fe::PointwiseMode_t mode) {
      return graph::Pointwise_attributes().set_mode(mode).set_compute_data_type(
          compute_data_type);
    };
    const auto set_scalar_type = [&](const Tensor& tensor) {
      tensor
          ->set_data_type(result_type == PRED ? fe::DataType_t::BOOLEAN
                                              : ReductionType(result_type))
          .set_dim({1})
          .set_stride({1});
    };
    Tensor result;
    if (op->opcode() == HloOpcode::kClamp) {
      // HLO clamp(min, operand, max) is min(max(operand, min), max).
      Tensor lower_bounded = body_graph->pointwise(
          values.at(op->operand(1)), values.at(op->operand(0)),
          attributes(fe::PointwiseMode_t::MAX));
      set_scalar_type(lower_bounded);
      result = body_graph->pointwise(lower_bounded, values.at(op->operand(2)),
                                     attributes(fe::PointwiseMode_t::MIN));
    } else if (op->operand_count() == 3) {
      // HLO select(predicate, true_value, false_value) versus cuDNN's
      // pointwise(true_value, false_value, predicate).
      result = body_graph->pointwise(
          values.at(op->operand(1)), values.at(op->operand(2)),
          values.at(op->operand(0)), attributes(ReducerMode(*op)));
    } else {
      result = body_graph->pointwise(values.at(op->operand(0)),
                                     values.at(op->operand(1)),
                                     attributes(ReducerMode(*op)));
    }
    set_scalar_type(result);
    values[op] = std::move(result);
  }
  const HloInstruction* root = computation.root_instruction();
  if (count == 1) {
    return {values.at(root)};
  }
  std::vector<Tensor> results;
  for (const HloInstruction* output : root->operands()) {
    results.push_back(values.at(output));
  }
  return results;
}

// Converts `tensor` to `type` with the same dimensions and strides.
Tensor Convert(graph::Graph& graph, const Tensor& tensor, fe::DataType_t type,
               fe::DataType_t compute_type) {
  Tensor result =
      graph.pointwise(tensor, graph::Pointwise_attributes()
                                  .set_mode(fe::PointwiseMode_t::IDENTITY)
                                  .set_compute_data_type(compute_type));
  result->set_data_type(type)
      .set_dim(tensor->get_dim())
      .set_stride(tensor->get_stride());
  return result;
}
#endif  // XLA_HAS_CUDNN_CUSTOM_REDUCTION_API

}  // namespace

absl::StatusOr<std::vector<Tensor>> LowerCudnnCustomReduction(
    const HloInstruction& reduce, graph::Graph& graph,
    const std::vector<Tensor>& inputs) {
  ABSL_RETURN_IF_ERROR(AnalyzeCudnnCustomReduction(reduce));
  if (static_cast<int64_t>(inputs.size()) != reduce.operand_count() / 2 ||
      absl::c_any_of(inputs,
                     [](const Tensor& input) { return input == nullptr; })) {
    return absl::InvalidArgumentError(
        "Custom reduction input tensors do not match HLO streams.");
  }
  // Compile-required: these frontend symbols do not exist before the custom
  // reduction API.
#if XLA_HAS_CUDNN_CUSTOM_REDUCTION_API
  const int64_t count = inputs.size();
  std::vector<fe::DataType_t> types;
  std::vector<Identity> identities;
  std::vector<Tensor> streams;
  for (int64_t i = 0; i < count; ++i) {
    const PrimitiveType type = reduce.operand(i)->shape().element_type();
    types.push_back(ReductionType(type));
    identities.push_back(GetIdentity(*reduce.operand(count + i)));
    // The fusion compiler stores PRED as INT8. The stream type must be the
    // identity type, so convert such an input, and below the result back.
    const fe::DataType_t storage = inputs[i]->get_data_type();
    if (storage == types[i]) {
      streams.push_back(inputs[i]);
    } else if (type == PRED && storage == fe::DataType_t::INT8) {
      streams.push_back(Convert(graph, inputs[i], types[i], storage));
    } else {
      return absl::InvalidArgumentError(
          "Custom reduction input tensor types do not match HLO stream "
          "types.");
    }
  }
  std::vector<int64_t> axes(reduce.dimensions().begin(),
                            reduce.dimensions().end());
  absl::c_sort(axes);
  Attributes attributes;
  attributes.set_name(std::string(reduce.name()))
      .set_reduction_dims(axes)
      .set_identity_values(std::move(identities))
      .set_compute_data_types(types);
  // The callback runs when the graph expands, possibly after the HLO module
  // is destroyed, so it owns a clone of the reducer body.
  std::shared_ptr<const HloComputation> owned_body(reduce.to_apply()->Clone());
  attributes.set_reducer(
      [owned_body](Attributes::Graph_t body_graph, std::vector<Tensor> lhs,
                   std::vector<Tensor> rhs) {
        return BuildReducer(*owned_body, std::move(body_graph), lhs, rhs);
      },
      CudnnCustomReductionSemanticKey(*reduce.to_apply()));
  std::vector<Tensor> results =
      graph.custom_reduction(streams, std::move(attributes));
  std::vector<int64_t> keep_dims(
      reduce.operand(0)->shape().dimensions().begin(),
      reduce.operand(0)->shape().dimensions().end());
  for (int64_t axis : axes) {
    keep_dims[axis] = 1;
  }
  std::vector<int64_t> strides(keep_dims.size(), 1);
  for (int64_t i = keep_dims.size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * keep_dims[i];
  }
  for (int64_t i = 0; i < count; ++i) {
    results[i]->set_dim(keep_dims).set_stride(strides).set_data_type(types[i]);
    if (streams[i] != inputs[i]) {
      results[i] = Convert(graph, results[i], fe::DataType_t::INT8,
                           fe::DataType_t::INT8);
    }
  }
  return results;
#else
  static_cast<void>(graph);
  return absl::UnimplementedError(
      "Custom reduction requires a cuDNN frontend with "
      "cudnn_frontend/node/custom_reduction.h and cuDNN 9.29+ headers.");
#endif  // XLA_HAS_CUDNN_CUSTOM_REDUCTION_API
}

}  // namespace xla::gpu
