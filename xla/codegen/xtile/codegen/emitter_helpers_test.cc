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

#include "xla/codegen/xtile/codegen/emitter_helpers.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/xtile/ir/xtile_dialect.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla::xtile {
namespace {

// A test-only visitor that reports a fixed replica-id bound for a chosen
// fusion parameter. This is used to simulate a parameter whose tile carries a
// replica_id, which is the precondition for replica-id pointer-table wrapping.
class ReplicaIdForParamVisitor : public DefaultTileRequirementsVisitor {
 public:
  ReplicaIdForParamVisitor(int64_t param_number, int64_t bound)
      : param_number_(param_number), bound_(bound) {}

  absl::StatusOr<llvm::SmallVector<int64_t>> RequiredReplicaIdBounds(
      const HloInstruction& instr) const override {
    if (instr.opcode() == HloOpcode::kParameter &&
        instr.parameter_number() == param_number_) {
      return llvm::SmallVector<int64_t>({bound_});
    }
    return llvm::SmallVector<int64_t>();
  }

 private:
  int64_t param_number_;
  int64_t bound_;
};

class EmitterHelpersTest : public HloHardwareIndependentTestBase {
 public:
  EmitterHelpersTest() : b_(mlir::UnknownLoc::get(&ctx_), &ctx_) {
    ctx_.loadDialect<mlir::arith::ArithDialect,
                     mlir::stablehlo::StablehloDialect, xtile::XTileDialect>();
    module_ = xla::llvm_ir::CreateMlirModuleOp(b_.getLoc());
    b_.setInsertionPointToStart(module_->getBody());
  }

  // Returns the single fusion instruction from the entry computation of the
  // given HLO text.
  absl::StatusOr<std::unique_ptr<HloModule>> ParseFusionModule(
      absl::string_view hlo_text) {
    return ParseAndReturnVerifiedModule(hlo_text);
  }

  const HloFusionInstruction* GetFusion(const HloModule& module) {
    return xla::Cast<HloFusionInstruction>(
        module.entry_computation()->root_instruction());
  }

  mlir::MLIRContext ctx_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  mlir::ImplicitLocOpBuilder b_;
};

constexpr absl::string_view kHloText = R"(
HloModule m

add_fusion {
  p0 = f32[128,128] parameter(0)
  p1 = f32[128,128] parameter(1)
  ROOT add = f32[128,128] add(p0, p1)
}

ENTRY e {
  p0 = f32[128,128] parameter(0)
  p1 = f32[128,128] parameter(1)
  ROOT fusion = f32[128,128] fusion(p0, p1), kind=kCustom,
    calls=add_fusion,
    backend_config={"fusion_backend_config": {kind: "__triton"}}
})";

// When a parameter carries a replica_id and no runtime-contract predicate is
// provided, wrapping is honored unconditionally: the parameter is promoted to a
// 1-D i64 pointer table of size equal to the replica-id bound.
TEST_F(EmitterHelpersTest, WrapsReplicaIdParamWhenPredicateIsNull) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseFusionModule(kHloText));
  const HloFusionInstruction* fusion = GetFusion(*module);

  ReplicaIdForParamVisitor visitor(/*param_number=*/0, /*bound=*/2);
  ASSERT_OK_AND_ASSIGN(
      llvm::SmallVector<mlir::Type> arg_types,
      GetFnArgTypes(b_, *fusion, /*opaque_args_types=*/{}, /*gpu_cc=*/{},
                    visitor, /*is_param_scratch_buffer=*/nullptr));

  // arg0 (replica-id param) -> pointer table memref<2xi64>.
  auto arg0 = mlir::cast<mlir::MemRefType>(arg_types[0]);
  EXPECT_EQ(arg0.getShape(), mlir::ArrayRef<int64_t>({2}));
  EXPECT_TRUE(arg0.getElementType().isInteger(64));

  // arg1 (plain param) -> plain buffer memref<128x128xf32>.
  auto arg1 = mlir::cast<mlir::MemRefType>(arg_types[1]);
  EXPECT_EQ(arg1.getShape(), mlir::ArrayRef<int64_t>({128, 128}));
  EXPECT_TRUE(arg1.getElementType().isF32());
}

// When the runtime-contract predicate reports the replica-id parameter as a
// pointer table (scratch buffer), the parameter is wrapped.
TEST_F(EmitterHelpersTest, WrapsReplicaIdParamWhenPredicateReturnsTrue) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseFusionModule(kHloText));
  const HloFusionInstruction* fusion = GetFusion(*module);

  ReplicaIdForParamVisitor visitor(/*param_number=*/0, /*bound=*/2);
  IsParamScratchBuffer is_param_scratch_buffer =
      [](int64_t param_index) -> bool { return true; };
  ASSERT_OK_AND_ASSIGN(
      llvm::SmallVector<mlir::Type> arg_types,
      GetFnArgTypes(b_, *fusion, /*opaque_args_types=*/{}, /*gpu_cc=*/{},
                    visitor, is_param_scratch_buffer));

  auto arg0 = mlir::cast<mlir::MemRefType>(arg_types[0]);
  EXPECT_EQ(arg0.getShape(), mlir::ArrayRef<int64_t>({2}));
  EXPECT_TRUE(arg0.getElementType().isInteger(64));
}

// The gating: even though the parameter's tile carries a replica_id, when the
// runtime-contract predicate reports that the parameter is NOT passed as a
// pointer table (scratch buffer), it must NOT be wrapped. This is the case for
// the AllGather Triton collective kernel, which passes its input as a plain
// buffer. Wrapping in that case would dereference plain data as pointers and
// cause a GPU memory-access fault.
TEST_F(EmitterHelpersTest, DoesNotWrapReplicaIdParamWhenPredicateReturnsFalse) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseFusionModule(kHloText));
  const HloFusionInstruction* fusion = GetFusion(*module);

  ReplicaIdForParamVisitor visitor(/*param_number=*/0, /*bound=*/2);
  IsParamScratchBuffer is_param_scratch_buffer =
      [](int64_t param_index) -> bool { return false; };
  ASSERT_OK_AND_ASSIGN(
      llvm::SmallVector<mlir::Type> arg_types,
      GetFnArgTypes(b_, *fusion, /*opaque_args_types=*/{}, /*gpu_cc=*/{},
                    visitor, is_param_scratch_buffer));

  // arg0 stays a plain buffer memref<128x128xf32>, NOT a memref<2xi64> pointer
  // table.
  auto arg0 = mlir::cast<mlir::MemRefType>(arg_types[0]);
  EXPECT_EQ(arg0.getShape(), mlir::ArrayRef<int64_t>({128, 128}));
  EXPECT_TRUE(arg0.getElementType().isF32());
}

// The predicate is consulted with the correct fusion parameter number, so it
// can wrap some replica-id parameters while leaving others as plain buffers.
TEST_F(EmitterHelpersTest, PredicateIsGatedPerParameterIndex) {
  constexpr absl::string_view kTwoReplicaIdParamsHlo = R"(
HloModule m

add_fusion {
  p0 = f32[128,128] parameter(0)
  p1 = f32[128,128] parameter(1)
  ROOT add = f32[128,128] add(p0, p1)
}

ENTRY e {
  p0 = f32[128,128] parameter(0)
  p1 = f32[128,128] parameter(1)
  ROOT fusion = f32[128,128] fusion(p0, p1), kind=kCustom,
    calls=add_fusion,
    backend_config={"fusion_backend_config": {kind: "__triton"}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseFusionModule(kTwoReplicaIdParamsHlo));
  const HloFusionInstruction* fusion = GetFusion(*module);

  // Both parameters carry a replica_id.
  class BothParamsVisitor : public DefaultTileRequirementsVisitor {
   public:
    absl::StatusOr<llvm::SmallVector<int64_t>> RequiredReplicaIdBounds(
        const HloInstruction& instr) const override {
      if (instr.opcode() == HloOpcode::kParameter) {
        return llvm::SmallVector<int64_t>({2});
      }
      return llvm::SmallVector<int64_t>();
    }
  } visitor;

  // Only parameter 1 is passed as a pointer table.
  IsParamScratchBuffer is_param_scratch_buffer =
      [](int64_t param_index) -> bool { return param_index == 1; };
  ASSERT_OK_AND_ASSIGN(
      llvm::SmallVector<mlir::Type> arg_types,
      GetFnArgTypes(b_, *fusion, /*opaque_args_types=*/{}, /*gpu_cc=*/{},
                    visitor, is_param_scratch_buffer));

  // arg0 not wrapped (plain buffer), arg1 wrapped (pointer table).
  auto arg0 = mlir::cast<mlir::MemRefType>(arg_types[0]);
  EXPECT_EQ(arg0.getShape(), mlir::ArrayRef<int64_t>({128, 128}));
  EXPECT_TRUE(arg0.getElementType().isF32());

  auto arg1 = mlir::cast<mlir::MemRefType>(arg_types[1]);
  EXPECT_EQ(arg1.getShape(), mlir::ArrayRef<int64_t>({2}));
  EXPECT_TRUE(arg1.getElementType().isInteger(64));
}

// Sanity check: a parameter that does not carry a replica_id is never wrapped,
// regardless of the predicate.
TEST_F(EmitterHelpersTest, NeverWrapsNonReplicaIdParam) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseFusionModule(kHloText));
  const HloFusionInstruction* fusion = GetFusion(*module);

  // No parameter carries a replica_id.
  DefaultTileRequirementsVisitor visitor;
  IsParamScratchBuffer always_true = [](int64_t) -> bool { return true; };
  ASSERT_OK_AND_ASSIGN(llvm::SmallVector<mlir::Type> arg_types,
                       GetFnArgTypes(b_, *fusion, /*opaque_args_types=*/{},
                                     /*gpu_cc=*/{}, visitor, always_true));

  auto arg0 = mlir::cast<mlir::MemRefType>(arg_types[0]);
  EXPECT_EQ(arg0.getShape(), mlir::ArrayRef<int64_t>({128, 128}));
  EXPECT_TRUE(arg0.getElementType().isF32());
}

}  // namespace
}  // namespace xla::xtile
