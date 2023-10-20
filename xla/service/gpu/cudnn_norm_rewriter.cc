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

#include "xla/service/gpu/cudnn_norm_rewriter.h"

#include <numeric>

#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cudnn/cudnn.h"
#endif

namespace xla {
namespace gpu {

namespace {

namespace m = match;

bool IsReduceAdd(const HloInstruction* instr) {
  HloComputation* reduce_comp = instr->to_apply();
  HloInstruction* reduce_comp_root = reduce_comp->root_instruction();
  return instr->operand_count() == 2 &&
         instr->operand(1)->opcode() == HloOpcode::kConstant &&
         ShapeUtil::IsScalar(instr->operand(1)->shape()) &&
         instr->operand(1)->literal().GetAsDouble({}) == 0. &&
         reduce_comp_root->opcode() == HloOpcode::kAdd &&
         reduce_comp_root->operand(0)->opcode() == HloOpcode::kParameter &&
         reduce_comp_root->operand(1)->opcode() == HloOpcode::kParameter;
}

bool CompatibleElementType(HloInstruction* instr) {
  PrimitiveType element_type = instr->shape().element_type();
  if (element_type == BF16 || element_type == F16 || element_type == F32) {
    return true;
  }
  return false;
}

StatusOr<int64_t> CConstant(se::CudaComputeCapability cuda_compute_capability) {
  if (cuda_compute_capability.major == se::CudaComputeCapability::AMPERE) {
    return 32 * 128;
  } else if (cuda_compute_capability.major ==
             se::CudaComputeCapability::HOPPER) {
    return 32 * 144;
  }
  return xla::InternalError(
      "Norm kernels require Ampere or newer architecture.");
}

// Matches pattern, convert(pattern), reshape(pattern),
// convert(reshape(pattern)) and reshape(convert(pattern)).
template <typename Pattern>
auto OptionalConvertAndOrReshape(Pattern pattern) {
  auto shared_subpattern = m::SharedSubpattern(pattern);
  return m::AnyOf<HloInstruction>(
      m::Convert(m::Reshape(shared_subpattern)),
      m::Reshape(m::Convert(shared_subpattern)), m::Convert(shared_subpattern),
      m::Reshape(shared_subpattern), shared_subpattern);
}

// Rsqrt with optional convert and/or reshape.
template <typename Pattern>
auto Rsqrt(Pattern pattern) {
  return OptionalConvertAndOrReshape(m::Rsqrt(pattern));
}

// AddAnyOrder with optional convert and/or reshape.
template <typename Pattern0, typename Pattern1>
auto AddAnyOrder(Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalConvertAndOrReshape(m::AddAnyOrder(pattern0, pattern1));
}

// Subtract with optional convert and/or reshape.
template <typename Pattern0, typename Pattern1>
auto Subtract(Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalConvertAndOrReshape(m::Subtract(pattern0, pattern1));
}

// Capturing subtract with optional convert and/or reshape.
template <typename Pattern0, typename Pattern1>
auto Subtract(HloInstruction** subtract, Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalConvertAndOrReshape(m::Subtract(subtract, pattern0, pattern1));
}

// Multiply with optional convert and/or reshape.
template <typename Pattern0, typename Pattern1>
auto MultiplyAnyOrder(Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalConvertAndOrReshape(m::MultiplyAnyOrder(pattern0, pattern1));
}

// Capturing multiply with optional convert and/or reshape.
template <typename Pattern0, typename Pattern1>
auto MultiplyAnyOrder(HloInstruction** multiply, Pattern0 pattern0,
                      Pattern1 pattern1) {
  return OptionalConvertAndOrReshape(
      m::MultiplyAnyOrder(multiply, pattern0, pattern1));
}

// Multiplication of pattern by itself with optional convert and/or reshape.
template <typename Pattern>
auto Square(Pattern pattern) {
  return MultiplyAnyOrder(pattern, pattern)
      .WithPredicate([](const HloInstruction* instr) {
        return instr->unique_operands().size() == 1;
      });
}

// Reduction-addition of pattern with optional convert and/or reshape and
// constant 0 scalar.
template <typename Pattern>
auto ReduceAdd(Pattern pattern) {
  return OptionalConvertAndOrReshape(
      m::Reduce(pattern, m::ConstantScalar(0))
          .WithPredicate(
              [](const HloInstruction* instr) { return IsReduceAdd(instr); }));
}

// Capturing reduction-addition of pattern with optional convert and/or reshape
// and constant 0 scalar.
template <typename Pattern>
auto ReduceAdd(HloInstruction** reduction, Pattern pattern) {
  return OptionalConvertAndOrReshape(
      m::Reduce(reduction, pattern, m::ConstantScalar(0))
          .WithPredicate(
              [](const HloInstruction* instr) { return IsReduceAdd(instr); }));
}

// Expected value, or mean, with optional broadcast.
template <typename Pattern>
auto Expectation(Pattern pattern) {
  auto shared_subpattern =
      MultiplyAnyOrder(m::Broadcast(m::ConstantScalar()), ReduceAdd(pattern));
  return m::AnyOf<HloInstruction>(m::Broadcast(shared_subpattern),
                                  shared_subpattern);
}

// Expected value, or mean, with optional broadcast.
template <typename Pattern>
auto Expectation(HloInstruction** expectation, Pattern pattern) {
  auto shared_subpattern = MultiplyAnyOrder(
      expectation, m::Broadcast(m::ConstantScalar()), ReduceAdd(pattern));
  return m::AnyOf<HloInstruction>(m::Broadcast(shared_subpattern),
                                  shared_subpattern);
}

// Expected value, or mean, with optional broadcast.
template <typename Pattern>
auto Expectation(HloInstruction** expectation, HloInstruction** reduction,
                 Pattern pattern) {
  auto shared_subpattern =
      MultiplyAnyOrder(expectation, m::Broadcast(m::ConstantScalar()),
                       ReduceAdd(reduction, pattern));
  return m::AnyOf<HloInstruction>(m::Broadcast(shared_subpattern),
                                  shared_subpattern);
}

// Variance, expressed as E[(X - E[X])^2] or E[X^2] - E[X]^2 where E is the
// expecation.
auto Variance(HloInstruction** input0, HloInstruction** input1) {
  return m::AnyOf<HloInstruction>(
      Subtract(Expectation(Square(m::Op(input0))),
               Square(Expectation(m::Op(input1)))),
      Expectation(Square(Subtract(m::Op(input0), Expectation(m::Op(input1))))));
}

// Variance, expressed as E[(X - E[X])^2] or E[X^2] - E[X]^2 where E is the
// expecation.
auto Variance(HloInstruction** variance, HloInstruction** expectation,
              HloInstruction** input0, HloInstruction** input1) {
  return m::AnyOf<HloInstruction>(
      Subtract(variance, Expectation(Square(m::Op(input0))),
               Square(Expectation(expectation, m::Op(input1)))),
      Expectation(variance,
                  Square(Subtract(m::Op(input0),
                                  Expectation(expectation, m::Op(input1))))));
}

// Reciprocal of the square root of variance + epsilon with optional broadcast.
auto NormFactor(HloInstruction** input0, HloInstruction** input1,
                HloInstruction** epsilon) {
  auto shared_subpattern = Rsqrt(AddAnyOrder(
      Variance(input0, input1), m::Broadcast(m::ConstantScalar(epsilon))));
  return m::AnyOf<HloInstruction>(m::Broadcast(shared_subpattern),
                                  shared_subpattern);
}

// Reciprocal of the square root of variance + epsilon with optional broadcast.
auto NormFactor(HloInstruction** input0, HloInstruction** input1,
                HloInstruction** variance, HloInstruction** expectation,
                HloInstruction** epsilon) {
  auto shared_subpattern = m::SharedSubpattern(
      Rsqrt(AddAnyOrder(Variance(variance, expectation, input0, input1),
                        m::Broadcast(m::ConstantScalar(epsilon)))));
  return m::AnyOf<HloInstruction>(m::Broadcast(shared_subpattern),
                                  shared_subpattern);
}

// Any order of p0 * p1 * p2.
template <typename P0, typename P1, typename P2>
auto MultiplyMultiplyAnyOrder(P0 p0, P1 p1, P2 p2) {
  return m::AnyOf<HloInstruction>(
      MultiplyAnyOrder(p0, MultiplyAnyOrder(p1, p2)),
      MultiplyAnyOrder(p1, MultiplyAnyOrder(p0, p2)),
      MultiplyAnyOrder(p2, MultiplyAnyOrder(p0, p1)));
}

// Any order of p0 - p1 + p2.
template <typename P0, typename P1, typename P2>
auto SubtractAddAnyOrder(P0 p0, P1 p1, P2 p2) {
  return m::AnyOf<HloInstruction>(m::AddAnyOrder(m::Subtract(p0, p1), p2),
                                  m::AddAnyOrder(m::Subtract(p2, p1), p0),
                                  m::Subtract(m::AddAnyOrder(p0, p2), p1));
}

// Any order of (p0 - p1) * p2 * p3 + p4.
template <typename P0, typename P1, typename P2, typename P3, typename P4>
auto SubtractMultiplyAddAnyOrder(P0 p0, P1 p1, P2 p2, P3 p3, P4 p4) {
  return m::AnyOf<HloInstruction>(
      SubtractAddAnyOrder(MultiplyMultiplyAnyOrder(p0, p2, p3),
                          MultiplyMultiplyAnyOrder(p1, p2, p3), p4),
      m::AddAnyOrder(MultiplyMultiplyAnyOrder(Subtract(p0, p1), p2, p3), p4));
}

class CudnnNormRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit CudnnNormRewriterVisitor(
      const se::CudaComputeCapability cuda_compute_capability)
      : cuda_compute_capability_(cuda_compute_capability) {}

  Status HandleAdd(HloInstruction* instr) override {
    return MatchLayerNorm(instr);
  }

  Status HandleSubtract(HloInstruction* instr) override {
    return MatchLayerNorm(instr);
  }

  Status HandleDivide(HloInstruction* instr) override {
    return MatchNormfactor(instr);
  }

  // Matches and rewrites a layer norm pattern with scale multiplication and
  // bias addition into a Custom Call to cuDNN. The pattern matching employs
  // modularity to cover the scope of realizations.
  Status MatchLayerNorm(HloInstruction* instr) {
    HloInstruction *input, *input0, *input1, *input2, *epsilon, *scale, *bias,
        *expectation, *reduce;
    if (Match(instr,
              SubtractMultiplyAddAnyOrder(
                  m::Op(&input),
                  Expectation(&expectation, &reduce, m::Op(&input0)),
                  NormFactor(&input1, &input2, &epsilon),
                  m::Broadcast(m::Op(&scale)), m::Broadcast(m::Op(&bias))))) {
#if CUDNN_VERSION < 8905
      // Layer Norm kernels are available with cuDNN 8.9.5 and above.
      VLOG(1) << "Layer norm Custom Calls require cuDNN 8.9.5.";
      return OkStatus();
#endif  // CUDNN_VERSION < 8905

      // Layer norm kernels require Ampere or newer architectures.
      if (!cuda_compute_capability_.IsAtLeast(
              se::CudaComputeCapability::AMPERE)) {
        VLOG(1)
            << "Layer norm Custom Calls require Ampere or newer architecture.";
        return OkStatus();
      }

      // Verify the uniqueness of the inputs.
      auto is_input = [input](HloInstruction* inputx) -> bool {
        return inputx->unique_id() == input->unique_id() ||
               inputx->operand_count() == 1 &&
                   inputx->operand(0)->unique_id() == input->unique_id();
      };
      if (!is_input(input0) || !is_input(input1) || !is_input(input2)) {
        VLOG(1) << "Layer norm operands not unique.";
        return OkStatus();
      }

      // Skip initial convert, if present.
      if (input->opcode() == HloOpcode::kConvert) {
        input = input->mutable_operand(0);
      }

      // Verify the element types. The types and shapes of the scale and bias
      // must match.
      if (!CompatibleElementType(input) || !CompatibleElementType(instr) ||
          !CompatibleElementType(scale) || !CompatibleElementType(bias) ||
          !ShapeUtil::Equal(scale->shape(), bias->shape())) {
        VLOG(1) << "Layer norm input types not supported.";
        return OkStatus();
      }

      // Verify that the shapes of scale and bias are compatible with the
      // operation.
      std::vector<int64_t> norm_dims(reduce->dimensions().begin(),
                                     reduce->dimensions().end());
      if (norm_dims.size() != scale->shape().dimensions_size()) {
        VLOG(1) << "Layer norm input dimensions not supported.";
        return OkStatus();
      }
      for (int i = 0; i < norm_dims.size(); ++i) {
        if (input->shape().dimensions(norm_dims[i]) !=
            scale->shape().dimensions(i)) {
          VLOG(1) << "Layer norm input dimensions not supported.";
          return OkStatus();
        }
      }

      // If necessary, transpose the input so that the dimensions not being
      // normalized are the leading dimensions.
      std::vector<int64_t> non_norm_dims;
      for (int64_t dim = 0; dim < input->shape().rank(); ++dim) {
        if (std::find(norm_dims.begin(), norm_dims.end(), dim) ==
            norm_dims.end()) {
          non_norm_dims.emplace_back(dim);
        }
      }
      std::vector<int64_t> transpose_order = non_norm_dims;
      transpose_order.insert(transpose_order.end(), norm_dims.begin(),
                             norm_dims.end());

      bool apply_transpose = false;
      for (int i = 0; i < transpose_order.size(); ++i) {
        if (transpose_order[i] != i) {
          apply_transpose = true;
          break;
        }
      }

      std::optional<HloInstruction*> transpose;
      std::vector<int64_t> inverse_transpose_order(transpose_order.size());
      if (apply_transpose) {
        for (int k = 0; k < transpose_order.size(); ++k) {
          inverse_transpose_order[transpose_order[k]] = k;
        }

        std::vector<int64_t> transposed_dims;
        for (int64_t non_norm_dim : non_norm_dims) {
          transposed_dims.emplace_back(input->shape().dimensions(non_norm_dim));
        }
        for (int64_t norm_dim : norm_dims) {
          transposed_dims.emplace_back(input->shape().dimensions(norm_dim));
        }
        TF_ASSIGN_OR_RETURN(transpose,
                            MakeTransposeHlo(input, transpose_order));
      }

      // Combine the dimensions not normalized into the first dimension of the
      // input as required by cuDNN.
      std::vector<int64_t> reshaped_dims = {1};
      for (auto non_norm_dim : non_norm_dims) {
        reshaped_dims[0] *= input->shape().dimensions(non_norm_dim);
      }
      for (auto norm_dim : norm_dims) {
        reshaped_dims.emplace_back(input->shape().dimensions(norm_dim));
      }
      // cuDNN requires tensors to have at least four dimensions.
      while (reshaped_dims.size() < 4) {
        reshaped_dims.emplace_back(1);
      }

      Shape reshaped_shape =
          ShapeUtil::MakeShape(input->shape().element_type(), reshaped_dims);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * reshape,
          MakeReshapeHlo(reshaped_shape, transpose.value_or(input)));

      // Reshape the scale and bias.
      std::vector<int64_t> reshaped_scale_dims(reshaped_dims.begin() + 1,
                                               reshaped_dims.end());
      // cuDNN requires tensors to have at least four dimensions.
      while (reshaped_scale_dims.size() < 4) {
        reshaped_scale_dims.emplace_back(1);
      }
      Shape scale_bias_shape = ShapeUtil::MakeShape(
          scale->shape().element_type(), reshaped_scale_dims);
      TF_ASSIGN_OR_RETURN(HloInstruction * reshaped_scale,
                          MakeReshapeHlo(scale_bias_shape, scale));
      TF_ASSIGN_OR_RETURN(HloInstruction * reshaped_bias,
                          MakeReshapeHlo(scale_bias_shape, bias));

      CudnnNormBackendConfig backend_config;
      backend_config.set_epsilon(epsilon->literal().GetAsDouble({}).value());
      auto* algorithm = backend_config.mutable_algorithm();
      algorithm->set_algo_id(0);
      algorithm->set_math_type(se::dnn::AlgorithmProto::TENSOR_OP_MATH);
      algorithm->set_is_cudnn_frontend(true);

      // Set the workspace size to its upper bound.
      // TODO(philipphack): Consider autotuning the norm kernels.
      TF_ASSIGN_OR_RETURN(const int64_t c_constant,
                          CConstant(cuda_compute_capability_));
      const int64_t workspace_size =
          (2 * c_constant * (4 + 256)) + (2 * reshaped_dims[0] * 4) + 64;
      algorithm->mutable_workspace_size()->set_value(workspace_size);

      // The output of the Custom Call is a tuple, the second element of which
      // describes the scratch space.
      Shape custom_call_shape = ShapeUtil::MakeTupleShape(
          {reshape->shape(), ShapeUtil::MakeShape(U8, {workspace_size})});

      HloInstruction* custom_call =
          instr->AddInstruction(HloInstruction::CreateCustomCall(
              custom_call_shape, {reshape, reshaped_scale, reshaped_bias},
              kCudnnNormCallTarget));
      TF_RETURN_IF_ERROR(custom_call->set_backend_config(backend_config));

      TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                          MakeGetTupleElementHlo(custom_call, 0));
      TF_ASSIGN_OR_RETURN(
          HloInstruction * inverse_reshape,
          MakeReshapeHlo(transpose.value_or(instr)->shape(), gte));

      if (!apply_transpose) {
        TF_RETURN_IF_ERROR(ReplaceInstruction(instr, inverse_reshape));
      } else {
        TF_ASSIGN_OR_RETURN(
            HloInstruction * inverse_transpose,
            MakeTransposeHlo(inverse_reshape, inverse_transpose_order));
        TF_RETURN_IF_ERROR(ReplaceInstruction(instr, inverse_transpose));
      }

      VLOG(1) << "Layer norm rewritten into Custom Call.";
    }

    return OkStatus();
  }

  // The layer norm training graph separately contains the expectation as well
  // as the norm factor and its cube, (variance + epsilon)^-1/2 and (variance +
  // epsilon)^-3/2. When identified in the graph, these quantities are fused
  // into the layer norm Custom Call.
  Status MatchNormfactor(HloInstruction* instr) {
    HloInstruction *custom_call = nullptr, *input, *input0, *expectation,
                   *variance, *vairance0, *epsilon, *epsilon0, *gte;
    if (Match(
            instr,
            m::Divide(
                NormFactor(&input, &input0, &variance, &expectation, &epsilon),
                AddAnyOrder(m::Op(&vairance0),
                            m::Broadcast(m::ConstantScalar(&epsilon0)))))) {
      // Verify the uniqueness of the operands.
      if (variance->unique_id() != vairance0->unique_id() ||
          epsilon->unique_id() != epsilon0->unique_id() ||
          input->unique_id() != input0->unique_id()) {
        VLOG(1) << "Layer norm operands not unique.";
        return OkStatus();
      }
      if (!CompatibleElementType(instr) ||
          !CompatibleElementType(expectation)) {
        VLOG(1) << "Layer norm input types not compatible.";
        return OkStatus();
      }

      // A layer norm Custom Call must be another user of input, possibly
      // separated by a type conversion.
      if (input->opcode() == HloOpcode::kConvert) {
        input = input->mutable_operand(0);
      }
      for (HloInstruction* user : input->users()) {
        if ((custom_call = FindLayerNormRecursive(user))) {
          break;
        }
      }
      if (!custom_call) {
        VLOG(1) << "Unable to identify layer norm Custom Call.";
        return OkStatus();
      }

      // The single user of the Custom Call must be a Get-Tuple-Element
      // accessing the output.
      if (custom_call->user_count() == 1 &&
          custom_call->users()[0]->opcode() == HloOpcode::kGetTupleElement &&
          custom_call->users()[0]->tuple_index() == 0) {
        gte = custom_call->users()[0];
      } else {
        VLOG(1) << "Incompatible users of layer norm Custom Call.";
        return OkStatus();
      }

      auto make_compatible_shape = [](Shape shape) -> Shape {
        // Eliminate any leading degenerate dimensions.
        Shape compatible_shape = ShapeUtil::DropDegenerateDimensions(shape);
        // cuDNN requires tensors to have at least four dimensions.
        while (compatible_shape.rank() < 4) {
          ShapeUtil::AppendMinorDimension(1, &compatible_shape);
        }
        return compatible_shape;
      };

      Shape expectation_shape = make_compatible_shape(expectation->shape());
      Shape norm_factor_shape = make_compatible_shape(instr->shape());

      // The augmented Custom Call additionally returns the expecation and the
      // norm factor.
      std::vector<Shape> tuple_shapes = custom_call->shape().tuple_shapes();
      tuple_shapes.insert(tuple_shapes.begin() + 1,
                          {expectation_shape, norm_factor_shape});

      Shape custom_call_shape = ShapeUtil::MakeTupleShape(tuple_shapes);

      HloInstruction* new_custom_call = instr->AddInstruction(
          custom_call->CloneWithNewShape(custom_call_shape));

      // Update the workspace size.
      TF_ASSIGN_OR_RETURN(const int64_t c_constant,
                          CConstant(cuda_compute_capability_));
      const int64_t workspace_size = (2 * c_constant * (4 + 256)) + 32;
      TF_ASSIGN_OR_RETURN(
          CudnnNormBackendConfig backend_config,
          custom_call->backend_config<xla::gpu::CudnnNormBackendConfig>());
      backend_config.mutable_algorithm()->mutable_workspace_size()->set_value(
          workspace_size);
      TF_RETURN_IF_ERROR(custom_call->set_backend_config(backend_config));

      auto replace_with_new_cc = [new_custom_call, this](
                                     HloInstruction* old_instr,
                                     int tuple_index) -> Status {
        TF_ASSIGN_OR_RETURN(
            HloInstruction * new_gte,
            MakeGetTupleElementHlo(new_custom_call, tuple_index));
        HloInstruction* new_instr = new_gte;
        if (!ShapeUtil::Equal(new_gte->shape(), old_instr->shape())) {
          TF_ASSIGN_OR_RETURN(new_instr,
                              MakeReshapeHlo(old_instr->shape(), new_gte));
        }
        if (tuple_index != 2) {
          // Replace the result of the layer norm or the expectation.
          TF_RETURN_IF_ERROR(ReplaceInstruction(old_instr, new_instr));
        } else {
          // Replace the norm factor, (variance + epsilon)^-1/2.
          TF_RETURN_IF_ERROR(
              ReplaceInstruction(old_instr->mutable_operand(0), new_instr));
          // Also replace the norm factor to the power of 3, (variance +
          // epsilon)^-1/2 / (variance + epsilon) = ((variance +
          // epsilon)^-1/2)^3.
          TF_ASSIGN_OR_RETURN(
              HloInstruction * new_multiply0,
              MakeBinaryHlo(HloOpcode::kMultiply, new_instr, new_instr));
          TF_ASSIGN_OR_RETURN(
              HloInstruction * new_multiply1,
              MakeBinaryHlo(HloOpcode::kMultiply, new_multiply0, new_instr));
          TF_RETURN_IF_ERROR(ReplaceInstruction(old_instr, new_multiply1));
        }
        return OkStatus();
      };

      // Replace the result of the original Custom Call as well as the
      // expectation and the norm factor with the augmented Custom Call.
      TF_RETURN_IF_ERROR(replace_with_new_cc(gte, 0));
      TF_RETURN_IF_ERROR(replace_with_new_cc(expectation, 1));
      TF_RETURN_IF_ERROR(replace_with_new_cc(instr, 2));

      VLOG(1)
          << "Expectation and norm factor fused into layer norm Custom Call.";
    }
    return OkStatus();
  }

  // Recursively traverses the graph downward across reshapes and transposes,
  // starting from instr, and returns a pointer to the layer norm Custom Call,
  // if found. Returns nullptr otherwise.
  HloInstruction* FindLayerNormRecursive(HloInstruction* instr) {
    if (Match(instr, m::CustomCall({kCudnnNormCallTarget}))) {
      return instr;
    }
    if (Match(instr, m::AnyOf<HloInstruction>(m::Reshape(), m::Transpose()))) {
      for (HloInstruction* user : instr->users()) {
        HloInstruction* custom_call = FindLayerNormRecursive(user);
        if (custom_call) {
          return custom_call;
        }
      }
    }
    return nullptr;
  }

 private:
  se::CudaComputeCapability cuda_compute_capability_;
};

StatusOr<bool> RunOnComputation(
    HloComputation* computation,
    se::CudaComputeCapability cuda_compute_capability) {
  CudnnNormRewriterVisitor visitor(cuda_compute_capability);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

}  // anonymous namespace

CudnnNormRewriter::CudnnNormRewriter(
    se::CudaComputeCapability cuda_compute_capability)
    : cuda_compute_capability_(cuda_compute_capability) {}

StatusOr<bool> CudnnNormRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(
        bool result, RunOnComputation(computation, cuda_compute_capability_));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
