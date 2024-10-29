/* Copyright 2024 The OpenXLA Authors.

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

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include "xla/service/cpu/onednn_memory_util.h"

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/tests/filecheck.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/test.h"

namespace xla {
namespace cpu {
namespace {

class MemoryUtilTest : public ::testing::Test,
                       public ::testing::WithParamInterface<
                           std::tuple<PrimitiveType, std::vector<int64_t>>> {
 protected:
  constexpr static const char* pad_test_pattern_ = R"(
    CHECK: %[[mref0:[0-9]+]] = insertvalue
    CHECK: %[[mref1:[0-9]+]] = insertvalue
    CHECK-SAME: [[arr:\[12 x i64\]]], [[arr]] } %[[mref0]], i64 255, 3
  )";
};

TEST_P(MemoryUtilTest, VerifyPadTest) {
  PrimitiveType dtype = std::get<0>(GetParam());
  std::string filecheck_input;
  llvm::LLVMContext context = llvm::LLVMContext();
  llvm::IRBuilder builder(context);
  llvm::raw_string_ostream ostream(filecheck_input);
  llvm::Module module("MemoryUtilPad", context);

  llvm::FunctionType* function_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context), {builder.getPtrTy()}, false);
  llvm::Function* function = llvm::Function::Create(
      function_type, llvm::Function::LinkageTypes::ExternalLinkage,
      "memory_util_pad_test", module);
  llvm::BasicBlock* bb = llvm::BasicBlock::Create(context, "BB", function);
  builder.SetInsertPoint(bb);

  Shape shape = ShapeUtil::MakeShape(dtype, std::get<1>(GetParam()));
  llvm::Argument* ptr = function->getArg(0);
  llvm::Type* type = llvm_ir::PrimitiveTypeToIrType(dtype, &module);

  if (shape.IsArray()) {
    for (auto dim : LayoutUtil::MinorToMajor(shape)) {
      type = llvm::ArrayType::get(type, shape.dimensions(dim));
    }
  }

  llvm_ir::IrArray ir_array(ptr, type, shape);
  auto alloca = GetAllocaAndEmitMemrefInfo(builder, ir_array);
  alloca.EmitLifetimeEnd();
  ostream << module;

  absl::StatusOr<bool> match = RunFileCheck(filecheck_input, pad_test_pattern_);
  TF_ASSERT_OK(match.status());
  EXPECT_TRUE(match.value());
}

INSTANTIATE_TEST_SUITE_P(
    MemoryUtilTestSuite, MemoryUtilTest,
    ::testing::Combine(::testing::ValuesIn({S8, S16, BF16, F16, F32}),
                       ::testing::Values(std::vector<int64_t>({30}),
                                         std::vector<int64_t>({30, 40}),
                                         std::vector<int64_t>({30, 40, 50}))),
    [](const ::testing::TestParamInfo<MemoryUtilTest::ParamType>& info) {
      std::ostringstream test_name;
      auto dtype =
          primitive_util::LowercasePrimitiveTypeName(std::get<0>(info.param));
      std::transform(dtype.begin(), dtype.end(), dtype.begin(),
                     [](auto c) { return std::toupper(c); });
      test_name << dtype << "_Rank_" << std::get<1>(info.param).size();
      return test_name.str();
    });

}  // namespace
}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
