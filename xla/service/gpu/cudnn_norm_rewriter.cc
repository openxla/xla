/* Copyright 2023 The OpenXLA Authors.

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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <vector>

#include "google/protobuf/wrappers.pb.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/protobuf/dnn.pb.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"  // IWYU pragma: keep
#include "third_party/gpus/cudnn/cudnn.h"        // IWYU pragma: keep
#include "third_party/gpus/cudnn/cudnn_version.h"
#endif

namespace xla {
namespace gpu {

namespace {

namespace m = match;

// Returns an architecture-specific constant for the calculation of an upper
// bound for the size of the scratch space for layer norm kernels.
absl::StatusOr<int64_t> CConstant(
    se::CudaComputeCapability cuda_compute_capability) {
  if (cuda_compute_capability.major == se::CudaComputeCapability::AMPERE) {
    return 32 * 128;
  } else if (cuda_compute_capability.major ==
             se::CudaComputeCapability::HOPPER) {
    return 32 * 144;
  }
  return xla::Internal("Norm kernels require Ampere or Hopper architecture.");
}

// Returns whether the element type of instr is compatible with layer norm
// kernels.
bool CompatibleElementType(const HloInstruction* instr) {
  PrimitiveType element_type = instr->shape().element_type();
  return element_type == BF16 || element_type == F16 || element_type == F32;
}

// Traverses the graph upward or downward starting at instr and returns the
// first instruction that is not a convert, bitcast or reshape. When traversing
// the graph downward, returns nullptr if a convert, bitcast or reshape does not
// have a single user.
const HloInstruction* SkipUnaryOps(const HloInstruction* instr,
                                   bool top_down = false) {
  while (instr->opcode() == HloOpcode::kConvert ||
         instr->opcode() == HloOpcode::kBitcast ||
         instr->opcode() == HloOpcode::kReshape) {
    if (top_down) {
      if (instr->user_count() != 1) {
        return nullptr;
      }
      instr = instr->users()[0];
    } else {
      instr = instr->operand(0);
    }
  }
  return instr;
}

// Returns whether the HLO Computation applied by instr calculates the sum of
// the elements.
bool AppliesAddReduce(const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kReduce) {
    return false;
  }
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

// Returns whether instr multiplies the result of a reduction by one over the
// number of reduced elements.
bool CalculatesExpectation(const HloInstruction* instr) {
  instr = SkipUnaryOps(instr);
  if (instr->opcode() != HloOpcode::kMultiply) {
    return false;
  }
  bool bcast_operand = instr->operand(0)->opcode() != HloOpcode::kBroadcast;
  const HloInstruction *broadcast = instr->operand(bcast_operand),
                       *reduce = SkipUnaryOps(instr->operand(!bcast_operand));
  reduce = SkipUnaryOps(reduce);
  if (reduce->opcode() != HloOpcode::kReduce ||
      broadcast->opcode() != HloOpcode::kBroadcast ||
      broadcast->operand(0)->opcode() != HloOpcode::kConstant) {
    return false;
  }

  float actual_r_nelems =
      broadcast->operand(0)->literal().GetAsDouble({}).value();
  int64_t nelems = 1;
  for (int64_t norm_dim : reduce->dimensions()) {
    nelems *= reduce->operand(0)->shape().dimensions()[norm_dim];
  }
  // The absolute of the difference between the actual scaling factor and the
  // reference value must not exceed a prescribed threshold.
  float r_nelems = 1. / static_cast<float>(nelems);
  float numerical_epsilon = std::numeric_limits<bfloat16>::epsilon();
  return abs(actual_r_nelems - r_nelems) <
         ((actual_r_nelems + r_nelems) * numerical_epsilon);
}

// Recursively traverses the graph across converts, bitcasts, reshapes and
// transposes, starting from instr, and returns the operand of custom_call with
// index operand_idx. Returns nullptr if the prescribed operand of custom_call
// is not found.
HloInstruction* FindOperandRecursive(HloInstruction* instr,
                                     HloInstruction* custom_call,
                                     absl::flat_hash_set<int>& visited_instrs,
                                     int operand_idx) {
  visited_instrs.emplace(instr->unique_id());
  const absl::flat_hash_set<HloOpcode> supported_ops = {
      HloOpcode::kConvert, HloOpcode::kBitcast, HloOpcode::kReshape,
      HloOpcode::kTranspose};
  // Look for the Custom Call among the users of instr.
  for (HloInstruction* user : instr->users()) {
    if (user->unique_id() == custom_call->unique_id() &&
        user->operand_index(instr) == operand_idx) {
      return instr;
    }
    if (supported_ops.contains(user->opcode()) &&
        !visited_instrs.contains(user->unique_id())) {
      return FindOperandRecursive(user, custom_call, visited_instrs,
                                  operand_idx);
    }
  }
  // Ascend the graph if the Custom Call is not found and instr is a
  // convert, reshape or transpose.
  if (supported_ops.contains(instr->opcode())) {
    return FindOperandRecursive(instr->mutable_operand(0), custom_call,
                                visited_instrs, operand_idx);
  }
  return nullptr;
}

// Recursively traverses the graph downward across converts, bitcasts, reshapes
// and transposes, starting from instr, and returns the first addition-reduction
// identified. Returns nullptr if no addition-reduction is found.
HloInstruction* FindAddReduceRecursive(
    HloInstruction* instr, absl::flat_hash_set<int>& visited_instrs) {
  visited_instrs.emplace(instr->unique_id());
  const absl::flat_hash_set<HloOpcode> supported_ops = {
      HloOpcode::kConvert, HloOpcode::kBitcast, HloOpcode::kReshape,
      HloOpcode::kTranspose};
  // Look for the reduction among the users of instr.
  for (HloInstruction* user : instr->users()) {
    if (AppliesAddReduce(user)) {
      return user;
    }
    if (supported_ops.contains(user->opcode()) &&
        !visited_instrs.contains(user->unique_id())) {
      return FindAddReduceRecursive(user, visited_instrs);
    }
  }
  return nullptr;
}

// Type conversion from and to any of BF16, FP16 and FP32.
template <typename Pattern>
auto SupportedConvert(Pattern pattern) {
  auto supported_convert = [](const HloInstruction* instr) -> bool {
    return CompatibleElementType(instr) &&
           CompatibleElementType(instr->operand(0));
  };
  return m::Convert(pattern).WithPredicate(supported_convert);
}

// Bitcast or reshape adding or removing degenerate dimensions.
template <typename Pattern>
auto SupportedBitcastOrReshape(Pattern pattern) {
  auto supported_bitcast_or_reshape = [](const HloInstruction* instr) -> bool {
    return ShapeUtil::Equal(
        ShapeUtil::DropDegenerateDimensions(instr->shape()),
        ShapeUtil::DropDegenerateDimensions(instr->operand(0)->shape()));
  };
  return m::AnyOf<HloInstruction>(
      m::Bitcast(pattern).WithPredicate(supported_bitcast_or_reshape),
      m::Reshape(pattern).WithPredicate(supported_bitcast_or_reshape));
}

// Matches pattern, SupportedConvert(pattern),
// SupportedBitcastOrReshape(pattern),
// SupportedConvert(SupportedBitcastOrReshape(pattern)) and
// SupportedBitcastOrReshape(SupportedConvert(pattern)).
template <typename Pattern>
auto OptionalSupportedTransform(Pattern pattern) {
  auto shared_subpattern = m::SharedSubpattern(pattern);
  return m::AnyOf<HloInstruction>(
      SupportedConvert(SupportedBitcastOrReshape(shared_subpattern)),
      SupportedBitcastOrReshape(SupportedConvert(shared_subpattern)),
      SupportedConvert(shared_subpattern),
      SupportedBitcastOrReshape(shared_subpattern), shared_subpattern);
}

// Bitcast or reshape with optional supported type conversion and/or addition or
// removal of degenerate dimensions.
template <typename Pattern>
auto BitcastOrReshape(Pattern pattern) {
  return OptionalSupportedTransform(
      m::AnyOf<HloInstruction>(m::Bitcast(pattern), m::Reshape(pattern)));
}

// Transpose with optional supported type conversion and/or addition or removal
// of degenerate dimensions.
template <typename Pattern>
auto Transpose(Pattern pattern) {
  return OptionalSupportedTransform(m::Transpose(pattern));
}

// Rsqrt with optional supported type conversion and/or addition or removal of
// degenerate dimensions.
template <typename Pattern>
auto Rsqrt(HloInstruction** rsqrt, Pattern pattern) {
  return OptionalSupportedTransform(m::Rsqrt(rsqrt, pattern));
}

// AddAnyOrder with optional supported type conversion and/or addition or
// removal of degenerate dimensions.
template <typename Pattern0, typename Pattern1>
auto AddAnyOrder(Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalSupportedTransform(m::AddAnyOrder(pattern0, pattern1));
}

// Subtract with optional supported type conversion and/or addition or removal
// of degenerate dimensions.
template <typename Pattern0, typename Pattern1>
auto Subtract(Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalSupportedTransform(m::Subtract(pattern0, pattern1));
}

// Capturing subtract with optional supported type conversion and/or addition or
// removal of degenerate dimensions.
template <typename Pattern0, typename Pattern1>
auto Subtract(HloInstruction** subtract, Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalSupportedTransform(m::Subtract(subtract, pattern0, pattern1));
}

// Multiply with optional supported type conversion and/or addition or removal
// of degenerate dimensions.
template <typename Pattern0, typename Pattern1>
auto MultiplyAnyOrder(Pattern0 pattern0, Pattern1 pattern1) {
  return OptionalSupportedTransform(m::MultiplyAnyOrder(pattern0, pattern1));
}

// Capturing multiply with optional supported type conversion and/or addition or
// removal of degenerate dimensions.
template <typename Pattern0, typename Pattern1>
auto MultiplyAnyOrder(HloInstruction** multiply, Pattern0 pattern0,
                      Pattern1 pattern1) {
  return OptionalSupportedTransform(
      m::MultiplyAnyOrder(multiply, pattern0, pattern1));
}

// Multiplication of pattern by itself with optional supported type conversion
// and/or addition or removal of degenerate dimensions.
template <typename Pattern>
auto Square(Pattern pattern) {
  return MultiplyAnyOrder(pattern, pattern)
      .WithPredicate([](const HloInstruction* instr) {
        return instr->unique_operands().size() == 1;
      });
}

// Multiplication of the square of pattern by pattern with optional supported
// type conversion and/or addition or removal of degenerate dimensions. The root
// instruction of pattern cannot be a multiplication.
template <typename Pattern>
auto Cube(Pattern pattern) {
  auto unique_cube = [](const HloInstruction* instr) -> bool {
    bool square_operand = instr->operand(0)->opcode() != HloOpcode::kMultiply;
    return instr->operand(!square_operand)->opcode() != HloOpcode::kMultiply &&
           instr->operand(square_operand)->operand(0)->unique_id() ==
               instr->operand(!square_operand)->unique_id();
  };
  return MultiplyAnyOrder(Square(pattern), pattern).WithPredicate(unique_cube);
}

// Addition-reduction of pattern with optional supported type conversion and/or
// addition or removal of degenerate dimensions and constant 0 scalar.
template <typename Pattern>
auto AddReduce(Pattern pattern) {
  return OptionalSupportedTransform(
      m::Reduce(pattern, m::Op())
          .WithPredicate([](const HloInstruction* instr) {
            return AppliesAddReduce(instr);
          }));
}

// Capturing addition-reduction of pattern with optional supported type
// conversion and/or addition or removal of degenerate dimensions and constant 0
// scalar.
template <typename Pattern>
auto AddReduce(HloInstruction** reduction, Pattern pattern) {
  return OptionalSupportedTransform(
      m::Reduce(reduction, pattern, m::Op())
          .WithPredicate([](const HloInstruction* instr) {
            return AppliesAddReduce(instr);
          }));
}

// Negated addition-reduction.
template <typename Pattern>
auto NegateAddReduce(Pattern pattern) {
  return m::AnyOf<HloInstruction>(AddReduce(m::Negate(pattern)),
                                  m::Negate(AddReduce(pattern)));
}

// Expected value, or mean, with optional broadcast.
template <typename Pattern>
auto Expectation(Pattern pattern) {
  auto shared_subpattern =
      MultiplyAnyOrder(m::Broadcast(m::ConstantScalar()), AddReduce(pattern))
          .WithPredicate([](const HloInstruction* instr) {
            return CalculatesExpectation(instr);
          });
  return m::AnyOf<HloInstruction>(m::Broadcast(shared_subpattern),
                                  shared_subpattern);
}

// Expected value, or mean, with optional broadcast.
template <typename Pattern>
auto Expectation(HloInstruction** expectation, Pattern pattern) {
  auto shared_subpattern =
      MultiplyAnyOrder(expectation, m::Broadcast(m::ConstantScalar()),
                       AddReduce(pattern))
          .WithPredicate([](const HloInstruction* instr) {
            return CalculatesExpectation(instr);
          });
  return m::AnyOf<HloInstruction>(m::Broadcast(shared_subpattern),
                                  shared_subpattern);
}

// Expected value, or mean, with optional broadcast.
template <typename Pattern>
auto Expectation(HloInstruction** expectation, HloInstruction** reduce,
                 Pattern pattern) {
  auto shared_subpattern =
      MultiplyAnyOrder(expectation, m::Broadcast(m::ConstantScalar()),
                       AddReduce(reduce, pattern))
          .WithPredicate([](const HloInstruction* instr) {
            return CalculatesExpectation(instr);
          });
  return m::AnyOf<HloInstruction>(m::Broadcast(shared_subpattern),
                                  shared_subpattern);
}

// Variance, expressed as expectation(X^2) - expectation(X)^2 or
// expectation((X - expectation(X))^2). The simultaneous capture of x0 and
// x1 allows the caller to verify that they are identical.
auto Variance(HloInstruction** expectation, HloInstruction** x0,
              HloInstruction** x1) {
  return m::AnyOf<HloInstruction>(
      Subtract(Expectation(Square(m::Op(x0))),
               Square(Expectation(expectation, m::Op(x1)))),
      Expectation(
          Square(Subtract(m::Op(x0), Expectation(expectation, m::Op(x1))))));
}

// Variance, expressed as expectation(X^2) - expectation(X)^2 or
// expectation((X - expectation(X))^2). The simultaneous capture of x0 and
// x1 allows the caller to verify that they are identical.
auto Variance(HloInstruction** variance, HloInstruction** expectation,
              HloInstruction** x0, HloInstruction** x1) {
  return m::AnyOf<HloInstruction>(
      Subtract(variance, Expectation(Square(m::Op(x0))),
               Square(Expectation(expectation, m::Op(x1)))),
      Expectation(
          variance,
          Square(Subtract(m::Op(x0), Expectation(expectation, m::Op(x1))))));
}

// Reciprocal of the square root of variance + epsilon with optional broadcast.
// The simultaneous capture of x0 and x1 allows the caller to verify
// that they are identical.
auto NormFactor(HloInstruction** norm_factor, HloInstruction** x0,
                HloInstruction** x1, HloInstruction** variance,
                HloInstruction** expectation, HloInstruction** epsilon) {
  auto shared_subpattern = m::SharedSubpattern(Rsqrt(
      norm_factor, AddAnyOrder(Variance(variance, expectation, x0, x1),
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

// Any order of p0 + p1 + p2.
template <typename P0, typename P1, typename P2>
auto AddAddAnyOrder(P0 p0, P1 p1, P2 p2) {
  return m::AnyOf<HloInstruction>(AddAnyOrder(p0, AddAnyOrder(p1, p2)),
                                  AddAnyOrder(p1, AddAnyOrder(p0, p2)),
                                  AddAnyOrder(p2, AddAnyOrder(p0, p1)));
}

// Any order of p0 * (p1 + p2).
template <typename P0, typename P1, typename P2>
auto MultiplyAddAnyOrder(P0 p0, P1 p1, P2 p2) {
  return m::AnyOf<HloInstruction>(
      MultiplyAnyOrder(p0, AddAnyOrder(p1, p2)),
      AddAnyOrder(MultiplyAnyOrder(p0, p1), MultiplyAnyOrder(p0, p2)));
}

// Any order of p0 - p1 + p2.
template <typename P0, typename P1, typename P2>
auto SubtractAddAnyOrder(P0 p0, P1 p1, P2 p2) {
  return m::AnyOf<HloInstruction>(AddAnyOrder(Subtract(p0, p1), p2),
                                  AddAnyOrder(Subtract(p2, p1), p0),
                                  Subtract(AddAnyOrder(p0, p2), p1));
}

// Any order of (p0 - p1) * p2 * p3 + p4.
template <typename P0, typename P1, typename P2, typename P3, typename P4>
auto SubtractMultiplyAddAnyOrder(P0 p0, P1 p1, P2 p2, P3 p3, P4 p4) {
  return m::AnyOf<HloInstruction>(
      SubtractAddAnyOrder(MultiplyMultiplyAnyOrder(p0, p2, p3),
                          MultiplyMultiplyAnyOrder(p1, p2, p3), p4),
      AddAnyOrder(MultiplyMultiplyAnyOrder(Subtract(p0, p1), p2, p3), p4));
}

// Expectation fused into a layer norm Custom Call. Compares the Custom Call to
// *custom_call.
auto FusedExpectation(HloInstruction** custom_call) {
  // Compare *custom_call to instr and set *custom_call to nullptr if the
  // comparison is unsuccessful. Because of the separation of matching and
  // capturing in the pattern matcher, the return value cannot convey the result
  // of the comparison.
  auto verify_cc = [custom_call](const HloInstruction* instr) -> bool {
    if (*custom_call && (*custom_call)->unique_id() != instr->unique_id()) {
      *custom_call = nullptr;
    }
    return true;
  };
  auto shared_subpattern = m::SharedSubpattern(m::GetTupleElement(
      m::CustomCall({kCudnnNormCallTarget}).WithPredicate(verify_cc), 1));
  return m::AnyOf<HloInstruction>(
      shared_subpattern, BitcastOrReshape(shared_subpattern),
      Transpose(BitcastOrReshape(shared_subpattern)));
}

// Expectation fused into a layer norm Custom Call. Captures the Custom Call in
// *custom_call if cc_capture is true, and compares the Custom Call to
// *custom_call otherwise.
auto FusedExpectation(HloInstruction** fused_expectation,
                      HloInstruction** custom_call, bool cc_capture) {
  if (cc_capture) {
    auto shared_subpattern = m::SharedSubpattern(m::GetTupleElement(
        fused_expectation, m::CustomCall(custom_call, {kCudnnNormCallTarget}),
        1));
    return m::AnyOf<HloInstruction>(
        shared_subpattern, BitcastOrReshape(shared_subpattern),
        Transpose(BitcastOrReshape(shared_subpattern)));
  } else {
    // Compare *custom_call to instr and set *custom_call to nullptr if the
    // comparison is unsuccessful.
    auto verify_cc = [custom_call](const HloInstruction* instr) -> bool {
      if (*custom_call && (*custom_call)->unique_id() != instr->unique_id()) {
        *custom_call = nullptr;
      }
      return true;
    };
    auto shared_subpattern = m::SharedSubpattern(m::GetTupleElement(
        fused_expectation,
        m::CustomCall({kCudnnNormCallTarget}).WithPredicate(verify_cc), 1));
    return m::AnyOf<HloInstruction>(
        shared_subpattern, BitcastOrReshape(shared_subpattern),
        Transpose(BitcastOrReshape(shared_subpattern)));
  }
}

// Norm factor fused into a layer norm Custom Call. Compares the Custom Call to
// *custom_call.
auto FusedNormFactor(HloInstruction** custom_call) {
  // Compare *custom_call to instr and set *custom_call to nullptr if the
  // comparison is unsuccessful.
  auto verify_cc = [custom_call](const HloInstruction* instr) -> bool {
    if (*custom_call && (*custom_call)->unique_id() != instr->unique_id()) {
      *custom_call = nullptr;
    }
    return true;
  };
  auto shared_subpattern = m::SharedSubpattern(m::GetTupleElement(
      m::CustomCall({kCudnnNormCallTarget}).WithPredicate(verify_cc), 2));
  return m::AnyOf<HloInstruction>(
      shared_subpattern, BitcastOrReshape(shared_subpattern),
      Transpose(BitcastOrReshape(shared_subpattern)));
}

// Norm factor fused into a layer norm Custom Call. Compares the Custom Call to
// *custom_call.
auto FusedNormFactor(HloInstruction** fused_norm_factor,
                     HloInstruction** custom_call) {
  // Compare *custom_call to instr and set *custom_call to nullptr if the
  // comparison is unsuccessful.
  auto verify_cc = [custom_call](const HloInstruction* instr) -> bool {
    if (*custom_call && (*custom_call)->unique_id() != instr->unique_id()) {
      *custom_call = nullptr;
    }
    return true;
  };
  auto shared_subpattern = m::SharedSubpattern(m::GetTupleElement(
      fused_norm_factor,
      m::CustomCall({kCudnnNormCallTarget}).WithPredicate(verify_cc), 2));
  return m::AnyOf<HloInstruction>(
      shared_subpattern, BitcastOrReshape(shared_subpattern),
      Transpose(BitcastOrReshape(shared_subpattern)));
}

// Derivative of the norm factor w.r.t. variance + epsilon,
// d(norm_factor)/d(variance + epsilon)
// = d((variance + epsilon)^-1/2)/d(variance + epsilon)
// = -1/2 * norm_factor^3.
// Forwards custom_call to FusedNormFactor for comparison.
auto DNormFactor(HloInstruction** custom_call) {
  return MultiplyAnyOrder(m::Broadcast(m::ConstantScalar(-0.5)),
                          Cube(FusedNormFactor(custom_call)));
}

//  Zero-centered input of the layer norm, X - expectation(X). Verifies that
//  *custom_call is a forward layer norm fusing X. Forwards custom_call to
//  FusedExpectation for comparison.
auto XCenter(HloInstruction** custom_call) {
  // Set *custom_call to nullptr if *custom_call is not a forward layer norm
  // operating on X, i.e. the first operand of instr.
  auto verify_x = [custom_call](const HloInstruction* instr) -> bool {
    if (*custom_call) {
      absl::flat_hash_set<int> visited_instrs;
      if (!FindOperandRecursive(const_cast<HloInstruction*>(instr->operand(0)),
                                *custom_call, visited_instrs, 0)) {
        *custom_call = nullptr;
      };
    }
    return true;
  };
  return Subtract(m::Op(), m::Broadcast(FusedExpectation(custom_call)))
      .WithPredicate(verify_x);
}

// Zero-centered input of the layer norm, X - expectation(X). Captures X in *x
// *custom_call is a forward layer norm fusing X. Forwards custom_call to
// FusedExpectation for capture or comparison.
auto XCenter(HloInstruction** x, HloInstruction** x_center,
             HloInstruction** fused_expectation, HloInstruction** custom_call,
             bool cc_capture) {
  // Capture X if *custom_call is a forward layer norm operating on X, i.e. the
  // first operand of instr.
  auto capture_x = [x, custom_call](const HloInstruction* instr) -> bool {
    absl::flat_hash_set<int> visited_instrs;
    if (*custom_call) {
      *x = FindOperandRecursive(const_cast<HloInstruction*>(instr->operand(0)),
                                *custom_call, visited_instrs, 0);
    }
    return true;
  };
  return Subtract(x_center, m::Op(),
                  m::Broadcast(FusedExpectation(fused_expectation, custom_call,
                                                cc_capture)))
      .WithPredicate(capture_x);
}

// Addition-reduction of the product of XCenter, the broadcasted scale and DY.
// Captures the scale in *scale if *custom_call is a forward layer norm fusing
// the scale. Forwards custom_call to XCenter for comparison.
auto F0(HloInstruction** custom_call, HloInstruction** scale,
        HloInstruction** dy) {
  // Capture the scale if *custom_call is a forward layer norm operating on the
  // scale, i.e. the first operand of instr.
  auto capture_scale = [scale,
                        custom_call](const HloInstruction* instr) -> bool {
    absl::flat_hash_set<int> visited_instrs;
    if (*custom_call) {
      *scale =
          FindOperandRecursive(const_cast<HloInstruction*>(instr->operand(0)),
                               *custom_call, visited_instrs, 1);
    }
    return true;
  };
  return AddReduce(MultiplyMultiplyAnyOrder(
      XCenter(custom_call), m::Broadcast().WithPredicate(capture_scale),
      m::Op(dy)));
}

// Product of XCenter and the scaled and broadcasted product of F0 and
// d(norm_factor)/d(variance + epsilon). Forwards custom_call to XCenter, F0 and
// DNormFactor for capture or comparison.
auto F1(HloInstruction** x, HloInstruction** x_center,
        HloInstruction** fused_expectation, HloInstruction** custom_call,
        HloInstruction** scale, HloInstruction** dy, bool cc_capture) {
  auto broadcasts_two_over_nelems = [](const HloInstruction* instr) -> bool {
    const HloInstruction* multiply = SkipUnaryOps(instr->operand(0));
    bool bcast_operand =
        multiply->operand(0)->opcode() != HloOpcode::kBroadcast;

    // The captured scalar must be two over the number of elements in the
    // broadcasted dimensions.
    float actual_two_over_nelems = multiply->operand(bcast_operand)
                                       ->operand(0)
                                       ->literal()
                                       .GetAsDouble({})
                                       .value();
    int64_t nelems = 1;
    for (int i = 0; i < instr->shape().dimensions_size(); ++i) {
      if (!c_linear_search(instr->dimensions(), i)) {
        nelems *= instr->shape().dimensions()[i];
      }
    }
    // The absolute of the difference between the actual scaling factor and the
    // reference value must not exceed a prescribed threshold.
    float two_over_nelems = 2. / static_cast<float>(nelems);
    float numerical_epsilon = std::numeric_limits<bfloat16>::epsilon();
    return abs(actual_two_over_nelems - two_over_nelems) <
           ((actual_two_over_nelems + two_over_nelems) * numerical_epsilon);
  };
  auto shared_subpattern = m::SharedSubpattern(MultiplyAnyOrder(
      XCenter(x, x_center, fused_expectation, custom_call, cc_capture),
      m::Broadcast(
          MultiplyAnyOrder(m::Broadcast(m::ConstantScalar()),
                           MultiplyAnyOrder(DNormFactor(custom_call),
                                            F0(custom_call, scale, dy))))
          .WithPredicate(broadcasts_two_over_nelems)));
  return m::AnyOf<HloInstruction>(AddAnyOrder(shared_subpattern, m::Constant()),
                                  shared_subpattern);
}

// Product of the norm factor, scale and DY. Captures the scale in *scale if
// *custom_call is a forward layer norm fusing the scale. Forwards custom_call
// to FusedNormFactor for comparison.
auto F2(HloInstruction** fused_norm_factor, HloInstruction** scale,
        HloInstruction** dy, HloInstruction** custom_call) {
  // Capture the scale if *custom_call is a forward layer norm operating on the
  // scale, i.e. the first operand of instr.
  auto capture_scale = [scale,
                        custom_call](const HloInstruction* instr) -> bool {
    absl::flat_hash_set<int> visited_instrs;
    if (*custom_call) {
      *scale =
          FindOperandRecursive(const_cast<HloInstruction*>(instr->operand(0)),
                               *custom_call, visited_instrs, 1);
    }
    return true;
  };
  return MultiplyAnyOrder(
      m::Broadcast(
          BitcastOrReshape(FusedNormFactor(fused_norm_factor, custom_call))),
      MultiplyAnyOrder(m::Broadcast().WithPredicate(capture_scale), m::Op(dy)));
}

class CudnnNormRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit CudnnNormRewriterVisitor(
      const se::CudaComputeCapability cuda_compute_capability)
      : cuda_compute_capability_(cuda_compute_capability) {}

  absl::Status HandleAdd(HloInstruction* instr) override {
    TF_RETURN_IF_ERROR(MatchLayerNorm(instr));
    TF_RETURN_IF_ERROR(MatchLayerNormBackward(instr));
    return OkStatus();
  }

  absl::Status HandleSubtract(HloInstruction* instr) override {
    return MatchLayerNorm(instr);
  }

  // Matches and rewrites layer norm patterns,
  // Y = (X - expectation(X))/(variance(X) + epsilon)^1/2 * scale + bias,
  // into Custom Calls to cuDNN.
  absl::Status MatchLayerNorm(HloInstruction* instr) {
    HloInstruction *x, *x0, *x1, *x2, *scale, *bias, *epsilon, *expectation,
        *expectation0, *reduce, *norm_factor, *variance, *broadcast_scale,
        *broadcast_bias;
    if (Match(instr,
              SubtractMultiplyAddAnyOrder(
                  m::Op(&x), Expectation(&expectation, &reduce, m::Op(&x0)),
                  NormFactor(&norm_factor, &x1, &x2, &variance, &expectation0,
                             &epsilon),
                  m::Broadcast(&broadcast_scale, m::Op(&scale)),
                  m::Broadcast(&broadcast_bias, m::Op(&bias))))) {
#if CUDNN_VERSION < 8905
      // Layer norm kernels are available with cuDNN 8.9.5 and above.
      VLOG(1) << "Layer norm Custom Calls require cuDNN 8.9.5.";
      return absl::OkStatus();
#endif  // CUDNN_VERSION < 8905

      if (!instr->GetModule()
               ->config()
               .debug_options()
               .xla_gpu_enable_cudnn_layer_norm()) {
        VLOG(1) << "Layer norm Custom Calls disabled.";
        return absl::OkStatus();
      }

      // Layer norm kernels require Ampere or Hopper architectures.
      if (cuda_compute_capability_.major != se::CudaComputeCapability::AMPERE &&
          cuda_compute_capability_.major != se::CudaComputeCapability::HOPPER) {
        VLOG(1) << "Layer norm Custom Calls require Ampere or Hopper "
                   "architectures.";
        return absl::OkStatus();
      }

      // Verify the uniqueness of the inputs.
      auto is_x = [x](HloInstruction* xk) -> bool {
        return xk->unique_id() == x->unique_id() ||
               (xk->opcode() == HloOpcode::kConvert &&
                xk->operand(0)->unique_id() == x->unique_id());
      };
      if (!is_x(x0) || !is_x(x1) || !is_x(x2) ||
          expectation->unique_id() != expectation0->unique_id()) {
        VLOG(1) << "Layer norm operands not unique.";
        return absl::OkStatus();
      }

      // Skip initial convert, if present.
      if (x->opcode() == HloOpcode::kConvert) {
        x = x->mutable_operand(0);
      }

      // Verify the input and output layouts.
      // TODO(philipphack): Consider supporting more general cases.
      if (!LayoutUtil::IsMonotonicWithDim0Major(x->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(scale->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(bias->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(instr->shape().layout())) {
        VLOG(1) << "Layer norm input and/or output layouts nor supported.";
        return absl::OkStatus();
      }

      // Verify the element types. The types and shapes of the scale and bias
      // must match.
      if (!CompatibleElementType(x) || !CompatibleElementType(instr) ||
          !CompatibleElementType(scale) || !CompatibleElementType(bias) ||
          !ShapeUtil::Equal(scale->shape(), bias->shape())) {
        VLOG(1) << "Layer norm input types or shapes not supported.";
        return absl::OkStatus();
      }

      // Verify that the shapes of scale and bias are compatible with the
      // operation.
      std::vector<int64_t> norm_dims(reduce->dimensions().begin(),
                                     reduce->dimensions().end());
      if (norm_dims.size() != scale->shape().dimensions_size()) {
        VLOG(1) << "Layer norm input dimensions not supported.";
        return absl::OkStatus();
      }
      for (int i = 0; i < norm_dims.size(); ++i) {
        if (x->shape().dimensions(norm_dims[i]) !=
            scale->shape().dimensions(i)) {
          VLOG(1) << "Layer norm input dimensions not supported.";
          return absl::OkStatus();
        }
      }

      // Verify the broadcasts of scale and bias.
      if (!ShapeUtil::EqualIgnoringElementType(reduce->operand(0)->shape(),
                                               broadcast_scale->shape()) ||
          !ShapeUtil::EqualIgnoringElementType(reduce->operand(0)->shape(),
                                               broadcast_bias->shape()) ||
          reduce->dimensions() != broadcast_scale->dimensions() ||
          reduce->dimensions() != broadcast_bias->dimensions()) {
        VLOG(1) << "Layer norm operand broadcast not supported.";
        return absl::OkStatus();
      }

      // If necessary, transpose the input so that the dimensions not being
      // normalized are the leading dimensions.
      std::vector<int64_t> non_norm_dims;
      for (int64_t x_dim = 0; x_dim < x->shape().rank(); ++x_dim) {
        if (std::find(norm_dims.begin(), norm_dims.end(), x_dim) ==
            norm_dims.end()) {
          non_norm_dims.emplace_back(x_dim);
        }
      }
      std::vector<int64_t> x_transpose_order = non_norm_dims;
      x_transpose_order.insert(x_transpose_order.end(), norm_dims.begin(),
                               norm_dims.end());

      bool apply_transpose = false;
      for (int i = 0; i < x_transpose_order.size(); ++i) {
        if (x_transpose_order[i] != i) {
          apply_transpose = true;
          break;
        }
      }

      std::optional<HloInstruction*> transpose;
      std::vector<int64_t> y_transpose_order(x_transpose_order.size());
      if (apply_transpose) {
        for (int k = 0; k < x_transpose_order.size(); ++k) {
          y_transpose_order[x_transpose_order[k]] = k;
        }
        TF_ASSIGN_OR_RETURN(transpose, MakeTransposeHlo(x, x_transpose_order));
      }

      // Combine the dimensions not normalized into the first dimension of the
      // input as required by cuDNN.
      std::vector<int64_t> reshaped_dims = {1};
      for (auto non_norm_dim : non_norm_dims) {
        reshaped_dims[0] *= x->shape().dimensions(non_norm_dim);
      }
      for (auto norm_dim : norm_dims) {
        reshaped_dims.emplace_back(x->shape().dimensions(norm_dim));
      }
      // cuDNN requires tensors to have at least four dimensions.
      while (reshaped_dims.size() < 4) {
        reshaped_dims.emplace_back(1);
      }

      Shape reshaped_shape =
          ShapeUtil::MakeShape(x->shape().element_type(), reshaped_dims);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * x_reshape,
          MakeReshapeHlo(reshaped_shape, transpose.value_or(x)));

      // Reshape the scale and bias.
      std::vector<int64_t> reshaped_scale_dims(reshaped_dims.begin() + 1,
                                               reshaped_dims.end());
      // cuDNN requires tensors to have at least four dimensions.
      while (reshaped_scale_dims.size() < 4) {
        reshaped_scale_dims.emplace_back(1);
      }
      Shape scale_bias_shape = ShapeUtil::MakeShape(
          scale->shape().element_type(), reshaped_scale_dims);
      TF_ASSIGN_OR_RETURN(HloInstruction * scale_reshape,
                          MakeReshapeHlo(scale_bias_shape, scale));
      TF_ASSIGN_OR_RETURN(HloInstruction * bias_reshape,
                          MakeReshapeHlo(scale_bias_shape, bias));
      GpuBackendConfig gpu_backend_config;
      CudnnNormBackendConfig& backend_config =
          *gpu_backend_config.mutable_cudnn_norm_backend_config();
      backend_config.set_epsilon(epsilon->literal().GetAsDouble({}).value());
      backend_config.set_kind(CudnnNormBackendConfig::LAYER_FWD_INFER);
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
          {x_reshape->shape(), ShapeUtil::MakeShape(U8, {workspace_size})});

      HloInstruction* custom_call =
          instr->AddInstruction(HloInstruction::CreateCustomCall(
              custom_call_shape, {x_reshape, scale_reshape, bias_reshape},
              kCudnnNormCallTarget));
      TF_RETURN_IF_ERROR(custom_call->set_backend_config(gpu_backend_config));

      TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                          MakeGetTupleElementHlo(custom_call, 0));
      TF_ASSIGN_OR_RETURN(
          HloInstruction * y_reshape,
          MakeReshapeHlo(transpose.value_or(instr)->shape(), gte));

      if (!apply_transpose) {
        TF_RETURN_IF_ERROR(ReplaceInstruction(instr, y_reshape));
      } else {
        TF_ASSIGN_OR_RETURN(HloInstruction * y_transpose,
                            MakeTransposeHlo(y_reshape, y_transpose_order));
        TF_RETURN_IF_ERROR(ReplaceInstruction(instr, y_transpose));
      }

      // Store the transpose orders applied to X and Y for potential use in the
      // backward graph.
      transpose_orders_.insert(
          {custom_call->unique_id(),
           apply_transpose
               ? TransposeOrders({x_transpose_order, y_transpose_order})
               : TransposeOrders({{}, {}})});

      VLOG(1) << "Layer norm rewritten into Custom Call.";

      // The layer norm training graph separately contains the norm factor
      // divided by the sum of variance and epsilon.
      for (HloInstruction* user : norm_factor->users()) {
        if (user->opcode() == HloOpcode::kDivide &&
            user->operand_index(norm_factor) == 0) {
          TF_RETURN_IF_ERROR(MatchNormFactor(user, custom_call, variance,
                                             expectation, epsilon));
        }
      }
    }

    return absl::OkStatus();
  }

  // The layer norm training graph separately contains the expectation as well
  // as the norm factor and its cube, (variance + epsilon)^-1/2 and (variance +
  // epsilon)^-3/2. When identified in the graph, these quantities are fused
  // into the layer norm Custom Call.
  absl::Status MatchNormFactor(HloInstruction* instr,
                               HloInstruction* custom_call,
                               HloInstruction* variance,
                               HloInstruction* expectation,
                               HloInstruction* epsilon) {
    HloInstruction *variance0, *epsilon0, *gte = custom_call->users()[0];
    if (Match(instr,
              m::Divide(m::Op(), AddAnyOrder(m::Op(&variance0),
                                             m::Broadcast(m::ConstantScalar(
                                                 &epsilon0)))))) {
      // Verify the uniqueness of the operands.
      if (variance->unique_id() != variance0->unique_id() ||
          epsilon->unique_id() != epsilon0->unique_id()) {
        VLOG(1) << "Layer norm operands not unique.";
        return absl::OkStatus();
      }

      // Verify the element types.
      if (!CompatibleElementType(instr) ||
          !CompatibleElementType(expectation)) {
        VLOG(1) << "Layer norm input types not compatible.";
        return absl::OkStatus();
      }

      // Retrieve the transpose orders applied in the forward layer norm.
      auto transpose_orders =
          transpose_orders_.extract(custom_call->unique_id());
      if (!transpose_orders) {
        VLOG(1)
            << "Unable to retrieve transpose orders for forward Custom Call.";
        return OkStatus();
      }

      // The shape of the expectation and norm factor return values of the
      // Custom Call is [nelems, 1, 1, 1], where nelems is the
      // number of elements in the expectation and norm factor shapes.
      auto make_compatible_shape = [](Shape shape) -> Shape {
        return ShapeUtil::MakeShape(shape.element_type(),
                                    {ShapeUtil::ElementsIn(shape), 1, 1, 1});
      };

      Shape expectation_shape = make_compatible_shape(expectation->shape());
      Shape norm_factor_shape = make_compatible_shape(instr->shape());

      // The augmented Custom Call additionally returns the expectation and the
      // norm factor.
      std::vector<Shape> tuple_shapes = custom_call->shape().tuple_shapes();
      tuple_shapes.insert(tuple_shapes.begin() + 1,
                          {expectation_shape, norm_factor_shape});

      Shape custom_call_shape = ShapeUtil::MakeTupleShape(tuple_shapes);

      HloInstruction* new_custom_call = instr->AddInstruction(
          custom_call->CloneWithNewShape(custom_call_shape));

      TF_ASSIGN_OR_RETURN(
          GpuBackendConfig gpu_backend_config,
          custom_call->backend_config<xla::gpu::GpuBackendConfig>());
      CudnnNormBackendConfig& backend_config =
          *gpu_backend_config.mutable_cudnn_norm_backend_config();
      backend_config.set_kind(CudnnNormBackendConfig::LAYER_FWD_TRAIN);

      // Update the workspace size.
      TF_ASSIGN_OR_RETURN(const int64_t c_constant,
                          CConstant(cuda_compute_capability_));
      const int64_t workspace_size = (2 * c_constant * (4 + 256)) + 32;
      backend_config.mutable_algorithm()->mutable_workspace_size()->set_value(
          workspace_size);
      TF_RETURN_IF_ERROR(
          new_custom_call->set_backend_config(gpu_backend_config));

      auto replace_with_new_cc = [new_custom_call, this](
                                     HloInstruction* old_instr,
                                     int tuple_index) -> absl::Status {
        TF_ASSIGN_OR_RETURN(
            HloInstruction * new_gte,
            MakeGetTupleElementHlo(new_custom_call, tuple_index));
        HloInstruction* new_instr = new_gte;
        if (!ShapeUtil::Equal(new_gte->shape(), old_instr->shape())) {
          TF_ASSIGN_OR_RETURN(new_instr,
                              MakeReshapeHlo(old_instr->shape(), new_gte));
        }
        if (old_instr->opcode() != HloOpcode::kDivide) {
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
        return absl::OkStatus();
      };

      // Replace the result of the original Custom Call as well as the
      // expectation and the norm factor with the augmented Custom Call.
      TF_RETURN_IF_ERROR(replace_with_new_cc(gte, 0));
      TF_RETURN_IF_ERROR(replace_with_new_cc(expectation, 1));
      TF_RETURN_IF_ERROR(replace_with_new_cc(instr, 2));

      // Update the Custom Call associated with the input transpose orders.
      transpose_orders.key() = new_custom_call->unique_id();
      transpose_orders_.insert(std::move(transpose_orders));

      VLOG(1)
          << "Expectation and norm factor fused into layer norm Custom Call.";
    }
    return absl::OkStatus();
  }

  // Matches and rewrites the backward graph of layer norm patterns into Custom
  // Calls to cuDNN when the associated forward graph has been rewritten into a
  // cuDNN Custom Call.
  Status MatchLayerNormBackward(HloInstruction* instr) {
    HloInstruction *custom_call = nullptr, *x = nullptr, *x0 = nullptr,
                   *dy = nullptr, *scale = nullptr, *fused_expectation,
                   *fused_expectation0, *fused_norm_factor, *fused_norm_factor0,
                   *broadcast, *scalar, *x_center, *x_center0, *dscale, *dbias;
    std::array<HloInstruction*, 3> dyk, scalek;
    if (Match(instr,
              AddAddAnyOrder(
                  m::Broadcast(
                      &broadcast,
                      MultiplyAddAnyOrder(
                          m::Broadcast(m::ConstantScalar(&scalar)),
                          NegateAddReduce(F1(&x, &x_center, &fused_expectation,
                                             &custom_call, &scale, &dy,
                                             /*cc_capture=*/true)),
                          NegateAddReduce(F2(&fused_norm_factor, &scalek[0],
                                             &dyk[0], &custom_call)))),
                  F2(&fused_norm_factor0, &scalek[1], &dyk[1], &custom_call),
                  F1(&x0, &x_center0, &fused_expectation0, &custom_call,
                     &scalek[2], &dyk[2],
                     /*cc_capture=*/false)))) {
      // Skip initial convert, if present.
      if (instr->user_count() == 1 &&
          instr->users()[0]->opcode() == HloOpcode::kConvert &&
          CompatibleElementType(instr->users()[0])) {
        instr = instr->users()[0];
      }

      // Verify the uniqueness of the captured Custom Call and inputs.
      if (!custom_call || !x || !x0 || !dy || !scale ||
          std::count(dyk.begin(), dyk.end(), dy) != dyk.size() ||
          std::count(scalek.begin(), scalek.end(), scale) != scalek.size() ||
          fused_expectation->unique_id() != fused_expectation0->unique_id() ||
          fused_norm_factor->unique_id() != fused_norm_factor0->unique_id() ||
          x->unique_id() != x0->unique_id() ||
          x_center->unique_id() != x_center0->unique_id()) {
        VLOG(1) << "Layer norm backward inputs not unique.";
        return OkStatus();
      }

      // The captured scalar must be one over the number of elements in the
      // broadcasted dimensions.
      float actual_r_nelems = scalar->literal().GetAsDouble({}).value();
      int64_t nelems = 1;
      for (int i = 0; i < broadcast->shape().dimensions_size(); ++i) {
        if (!c_linear_search(broadcast->dimensions(), i)) {
          nelems *= broadcast->shape().dimensions()[i];
        }
      }
      // The absolute of the difference between the actual scaling factor and
      // the reference value must not exceed a prescribed threshold.
      float r_nelems = 1. / static_cast<float>(nelems);
      float numerical_epsilon = std::numeric_limits<bfloat16>::epsilon();
      if (!(abs(actual_r_nelems - r_nelems) <
            ((actual_r_nelems + r_nelems) * numerical_epsilon))) {
        VLOG(1)
            << "Layer norm backward broadcast operand outside expected range.";
        return OkStatus();
      }

      // Identify Dscale = AddReduce(XCenter * NormFactor * DY)
      // through a top-down search starting from XCenter.
      auto find_dscale = [x_center, fused_norm_factor,
                          dy]() -> HloInstruction* {
        for (const HloInstruction* multiply_norm_factor : x_center->users()) {
          multiply_norm_factor =
              SkipUnaryOps(multiply_norm_factor, /*top-down=*/true);
          // Verify that the user of XCenter is a multiply with a single user.
          if (!multiply_norm_factor ||
              multiply_norm_factor->opcode() != HloOpcode::kMultiply ||
              multiply_norm_factor->user_count() != 1) {
            continue;
          }
          // Verify that one of the factors is a broadcast operating on the
          // fused norm factor.
          bool bcast_operand = multiply_norm_factor->operand(0)->opcode() !=
                               HloOpcode::kBroadcast;
          if (multiply_norm_factor->operand(bcast_operand)->opcode() !=
                  HloOpcode::kBroadcast ||
              SkipUnaryOps(
                  SkipUnaryOps(multiply_norm_factor->operand(bcast_operand))
                      ->operand(0))
                      ->unique_id() != fused_norm_factor->unique_id()) {
            continue;
          }
          const HloInstruction* multiply_dy =
              SkipUnaryOps(multiply_norm_factor->users()[0], /*top-down=*/true);
          // Verify that the user of the multiply is another multiply that is
          // also a user of DY.
          if (!multiply_dy || multiply_dy->opcode() != HloOpcode::kMultiply ||
              !multiply_dy->IsUserOf(dy)) {
            continue;
          }
          // Dscale applies an addition-reduction to multiply_dy.
          for (HloInstruction* multiply_dy_user : multiply_dy->users()) {
            if (AppliesAddReduce(multiply_dy_user)) {
              return multiply_dy_user;
            }
          }
        }
        return nullptr;
      };
      if (!(dscale = find_dscale())) {
        VLOG(1) << "Unable to identify Dscale in graph.";
        return OkStatus();
      }

      // Find Dbias, i.e. an addition-reduction of DY, starting from DY.
      // Rewriting proceeds without fusing Dbias if unsuccessful.
      absl::flat_hash_set<int> visited_instrs;
      dbias = FindAddReduceRecursive(dy, visited_instrs);

      // Verify the input and output layouts.
      // TODO(philipphack): Consider supporting more general cases.
      if (!LayoutUtil::IsMonotonicWithDim0Major(dy->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(instr->shape().layout()) ||
          !LayoutUtil::IsMonotonicWithDim0Major(dscale->shape().layout()) ||
          (dbias &&
           !LayoutUtil::IsMonotonicWithDim0Major(dbias->shape().layout()))) {
        VLOG(1) << "Layer norm input and/or output layouts nor supported.";
        return OkStatus();
      }

      // The types of X and DX must match.
      if (x->shape().element_type() != instr->shape().element_type()) {
        VLOG(1) << "The types of X and DX must match.";
        return OkStatus();
      }

      // The types and shapes of scale, Dscale and Dbias (if present) must
      // match.
      if (!ShapeUtil::Equal(
              ShapeUtil::DropDegenerateDimensions(scale->shape()),
              ShapeUtil::DropDegenerateDimensions(dscale->shape())) ||
          (dbias && !ShapeUtil::Equal(
                        ShapeUtil::DropDegenerateDimensions(scale->shape()),
                        ShapeUtil::DropDegenerateDimensions(dbias->shape())))) {
        VLOG(1) << "Backward layer norm types not supported.";
        return OkStatus();
      }

      // Verify the element types.
      if (!CompatibleElementType(dy)) {
        VLOG(1) << "Backward layer norm types not supported.";
        return OkStatus();
      }

      // The byte size of the element type of X must be at least that of DY and
      // scale.
      if (ShapeUtil::ByteSizeOfPrimitiveType(x->shape().element_type()) <
              ShapeUtil::ByteSizeOfPrimitiveType(dy->shape().element_type()) ||
          ShapeUtil::ByteSizeOfPrimitiveType(x->shape().element_type()) <
              ShapeUtil::ByteSizeOfPrimitiveType(
                  scale->shape().element_type())) {
        VLOG(1) << "Backward layer norm types not supported.";
        return OkStatus();
      }

      // Transpose DY applying the stored transpose order of X from the forward
      // graph.
      auto transpose_orders = transpose_orders_.find(custom_call->unique_id());
      if (transpose_orders == transpose_orders_.end()) {
        VLOG(1)
            << "Unable to retrieve transpose orders for forward Custom Call.";
        return OkStatus();
      }
      HloInstruction* transposed_dy = dy;
      if (!transpose_orders->second.x_transpose.empty()) {
        TF_ASSIGN_OR_RETURN(
            transposed_dy,
            MakeTransposeHlo(dy, transpose_orders->second.x_transpose));
      }
      TF_ASSIGN_OR_RETURN(HloInstruction * reshaped_dy,
                          MakeReshapeHlo(x->shape(), transposed_dy));

      Shape dx_shape = ShapeUtil::MakeShape(instr->shape().element_type(),
                                            x->shape().dimensions());

      Shape dscale_dbias_shape = ShapeUtil::MakeShape(
          dscale->shape().element_type(), scale->shape().dimensions());

      GpuBackendConfig gpu_backend_config;
      CudnnNormBackendConfig& backend_config =
          *gpu_backend_config.mutable_cudnn_norm_backend_config();
      backend_config.set_kind(CudnnNormBackendConfig::LAYER_BWD);
      auto* algorithm = backend_config.mutable_algorithm();
      algorithm->set_algo_id(0);
      algorithm->set_math_type(se::dnn::AlgorithmProto::TENSOR_OP_MATH);
      algorithm->set_is_cudnn_frontend(true);

      // Set the workspace size to its upper bound.
      // TODO(philipphack): Consider autotuning the norm kernels.
      TF_ASSIGN_OR_RETURN(const int64_t c_constant,
                          CConstant(cuda_compute_capability_));
      const int64_t workspace_size = (2 * c_constant * (4 + 256)) +
                                     (2 * x->shape().dimensions(0) * 4) + 64;
      algorithm->mutable_workspace_size()->set_value(workspace_size);

      // The output of the Custom Call is a tuple. The output shape of Dscale
      // and Dbias is that of scale.
      Shape custom_call_shape = ShapeUtil::MakeTupleShape(
          {dx_shape, dscale_dbias_shape, dscale_dbias_shape,
           ShapeUtil::MakeShape(U8, {workspace_size})});

      HloInstruction* custom_call =
          instr->AddInstruction(HloInstruction::CreateCustomCall(
              custom_call_shape,
              {x, scale, reshaped_dy, fused_expectation, fused_norm_factor},
              kCudnnNormCallTarget));
      TF_RETURN_IF_ERROR(custom_call->set_backend_config(gpu_backend_config));

      auto replace_with_cc = [custom_call, transpose_orders, transposed_dy,
                              this](HloInstruction* old_instr,
                                    int tuple_index) -> Status {
        TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                            MakeGetTupleElementHlo(custom_call, tuple_index));
        HloInstruction* new_instr;
        // Transpose DX applying the stored transpose order of Y from the
        // forward graph.
        if (tuple_index == 0 && !transpose_orders->second.y_transpose.empty()) {
          TF_ASSIGN_OR_RETURN(new_instr,
                              MakeReshapeHlo(transposed_dy->shape(), gte));
          TF_ASSIGN_OR_RETURN(
              new_instr, MakeTransposeHlo(
                             new_instr, transpose_orders->second.y_transpose));
        } else {
          TF_ASSIGN_OR_RETURN(new_instr,
                              MakeReshapeHlo(old_instr->shape(), gte));
        }
        TF_RETURN_IF_ERROR(ReplaceInstruction(old_instr, new_instr));
        return OkStatus();
      };

      TF_RETURN_IF_ERROR(replace_with_cc(instr, 0));
      TF_RETURN_IF_ERROR(replace_with_cc(dscale, 1));
      if (dbias) {
        TF_RETURN_IF_ERROR(replace_with_cc(dbias, 2));
      }
      VLOG(1) << "Gradients w.r.t. x"
              << (dbias ? ", scale and bias" : " and scale")
              << " rewritten into layer norm backward Custom Call.";
    }

    return OkStatus();
  }

 private:
  se::CudaComputeCapability cuda_compute_capability_;
  struct TransposeOrders {
    std::vector<int64_t> x_transpose, y_transpose;
  };
  absl::flat_hash_map<int, TransposeOrders> transpose_orders_;
};

absl::StatusOr<bool> RunOnComputation(
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

absl::StatusOr<bool> CudnnNormRewriter::Run(
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
