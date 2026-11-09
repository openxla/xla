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

#include "xla/backends/gpu/codegen/rng_get_and_update_state.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.h"
#include "xla/tests/literal_test_util.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;

class RngGetAndUpdateStateTest : public GpuPjRtCodegenTest {
 protected:
  RngGetAndUpdateStateTest() : mlir_context_(CreateMlirContext()) {
    mlir_context_->appendDialectRegistry(
        MlirKernelEmitter::GetDialectRegistry());
    mlir_context_->loadAllAvailableDialects();
  }

  std::unique_ptr<mlir::MLIRContext> mlir_context_;
};

TEST_F(RngGetAndUpdateStateTest, CreateAndLowerMlirModule) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule RngStateMlir
    ENTRY main {
      ROOT rng = u64[2]{0} rng-get-and-update-state(), delta=131072
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloRngGetAndUpdateStateInstruction* instr =
      Cast<HloRngGetAndUpdateStateInstruction>(
          module->entry_computation()->root_instruction());
  BufferAllocation state_allocation(/*index=*/0, kRngStateSizeBytes,
                                    /*color=*/0);
  state_allocation.set_persistent(true);
  BufferAllocation out_allocation(/*index=*/1, kRngStateSizeBytes,
                                  /*color=*/0);
  BufferAllocation::Slice state_slice(&state_allocation, 0, kRngStateSizeBytes);
  BufferAllocation::Slice out_slice(&out_allocation, 0, kRngStateSizeBytes);
  ASSERT_OK_AND_ASSIGN(
      emitters::KernelArguments kernel_arguments,
      CreateRngGetAndUpdateStateKernelArguments(instr, state_slice, out_slice));
  // The state is an input as well as an output, so its read survives in
  // KernelThunk::buffer_uses() as BufferUse::Read on the same slice.
  EXPECT_THAT(kernel_arguments.GetArgumentOutputFlags(),
              ElementsAre(false, true, true));
  EXPECT_THAT(kernel_arguments.GetArgumentBufferSlices(),
              ElementsAre(state_slice, state_slice, out_slice));

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       CreateRngGetAndUpdateStateMlirModule(
                           *mlir_context_, instr, kernel_arguments, "rng"));

  std::string mlir_str;
  llvm::raw_string_ostream mlir_stream(mlir_str);
  mlir_stream << *mlir_module;

  EXPECT_THAT(RunFileCheck(mlir_str, R"(
    CHECK-NOT: i128
    CHECK-NOT: llvm.mlir.global
    CHECK: func.func @rng(
    CHECK-SAME: %[[STATE_IN:[^:]*]]: tensor<2xi64> {llvm.align = 256 : index, llvm.dereferenceable = 16 : index, xla.aliased, xla.slice_index = 0 : index},
    CHECK-SAME: %[[STATE_OUT:[^:]*]]: tensor<2xi64> {llvm.align = 256 : index, llvm.dereferenceable = 16 : index, xla.aliased, xla.slice_index = 0 : index},
    CHECK-SAME: %[[OUT:[^:]*]]: tensor<2xi64> {llvm.align = 256 : index, llvm.dereferenceable = 16 : index, xla.slice_index = 1 : index})
    CHECK-SAME: xla.backend_kind = #xla.backend_kind<gpu>
    CHECK:   %[[LO:.*]] = tensor.extract %[[STATE_IN]][%{{.*}}] : tensor<2xi64>
    CHECK:   %[[HI:.*]] = tensor.extract %[[STATE_IN]][%{{.*}}] : tensor<2xi64>
    CHECK:   %[[NEW_LO:.*]], %[[CARRY_I1:.*]] = arith.addui_extended %[[LO]], %[[DELTA:.*]] : i64, i1
    CHECK:   %[[CARRY:.*]] = arith.extui %[[CARRY_I1]] : i1 to i64
    CHECK:   %[[NEW_HI:.*]] = arith.addi %[[HI]], %[[CARRY]] : i64
    CHECK:   %[[S0:.*]] = tensor.insert %[[NEW_LO]] into %[[STATE_OUT]][%{{.*}}] : tensor<2xi64>
    CHECK:   %[[S1:.*]] = tensor.insert %[[NEW_HI]] into %[[S0]][%{{.*}}] : tensor<2xi64>
    CHECK:   %[[O0:.*]] = tensor.insert %[[LO]] into %[[OUT]][%{{.*}}] : tensor<2xi64>
    CHECK:   %[[O1:.*]] = tensor.insert %[[HI]] into %[[O0]][%{{.*}}] : tensor<2xi64>
    CHECK:   return %[[S1]], %[[O1]] : tensor<2xi64>, tensor<2xi64>
  )"),
              IsOkAndHolds(true));

  auto check_lowered_llvm =
      [&](const stream_executor::GpuComputeCapability& cc) {
        ASSERT_OK_AND_ASSIGN(
            mlir::OwningOpRef<mlir::ModuleOp> module_copy,
            CreateRngGetAndUpdateStateMlirModule(*mlir_context_, instr,
                                                 kernel_arguments, "rng"));
        stream_executor::DeviceDescription device;
        device.set_gpu_compute_capability(cc);
        MlirKernelSource mlir_source(nullptr, std::move(module_copy));
        ASSERT_OK_AND_ASSIGN(
            LlvmKernelSource llvm_source,
            CompileMlirToLlvm(device, *module, "rng", /*unroll_factor=*/0,
                              *mlir_context_, std::move(mlir_source)));
        EXPECT_THAT(RunFileCheck(llvm_source.ToString(), R"(
      CHECK-NOT: i128
      CHECK-NOT: @rng_state
      CHECK: define void @rng(ptr align 256 dereferenceable(16) %[[STATE:[^,]*]], ptr align 256 dereferenceable(16) %{{[^,]*}}, ptr noalias align 256 dereferenceable(16) %[[OUT:[^)]*]])
      CHECK:   %[[S_PTR0:.*]] = getelementptr inbounds [2 x i64], ptr %[[STATE]], i32 0, i32 0
      CHECK:   %[[LO:.*]] = load i64, ptr %[[S_PTR0]]
      CHECK:   %[[S_PTR1:.*]] = getelementptr inbounds [2 x i64], ptr %[[STATE]], i32 0, i32 1
      CHECK:   %[[HI:.*]] = load i64, ptr %[[S_PTR1]]
      CHECK:   %[[ADD_RES:.*]] = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %[[LO]], i64 131072)
      CHECK:   %[[NEW_LO:.*]] = extractvalue { i64, i1 } %[[ADD_RES]], 0
      CHECK:   %[[CARRY_I1:.*]] = extractvalue { i64, i1 } %[[ADD_RES]], 1
      CHECK:   %[[CARRY:.*]] = zext i1 %[[CARRY_I1]] to i64
      CHECK:   %[[NEW_HI:.*]] = add i64 %[[HI]], %[[CARRY]]
      CHECK:   store i64 %[[NEW_LO]], ptr %[[S_PTR0]]
      CHECK:   store i64 %[[NEW_HI]], ptr %[[S_PTR1]]
      CHECK:   %[[O_PTR0:.*]] = getelementptr inbounds [2 x i64], ptr %[[OUT]], i32 0, i32 0
      CHECK:   store i64 %[[LO]], ptr %[[O_PTR0]]
      CHECK:   %[[O_PTR1:.*]] = getelementptr inbounds [2 x i64], ptr %[[OUT]], i32 0, i32 1
      CHECK:   store i64 %[[HI]], ptr %[[O_PTR1]]
      CHECK:   ret void
    )"),
                    IsOkAndHolds(true));
      };

  check_lowered_llvm(stream_executor::CudaComputeCapability::Ampere());
  check_lowered_llvm(stream_executor::GpuComputeCapability(
      stream_executor::OneAPIComputeCapability{}));
}

TEST_F(RngGetAndUpdateStateTest, ExecuteAdvancesSharedStateAcrossOps) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule RngStateAdvance
    ENTRY main {
      first = u64[2]{0} rng-get-and-update-state(), delta=131072
      second = u64[2]{0} rng-get-and-update-state(), delta=65536
      ROOT tuple = (u64[2]{0}, u64[2]{0}) tuple(first, second)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> executable,
                       test_runner().CreateExecutable(std::move(module),
                                                      /*run_hlo_passes=*/true));

  constexpr uint64_t kInit = 0x7012395ULL;
  constexpr uint64_t kDeltaFirst = 131072ULL;
  constexpr uint64_t kDeltaSecond = 65536ULL;

  ASSERT_OK_AND_ASSIGN(Literal result1,
                       test_runner().ExecuteWithExecutable(
                           executable.get(), absl::Span<const Literal>{}));
  std::vector<Literal> outputs1 = result1.DecomposeTuple();
  ASSERT_EQ(outputs1.size(), 2);
  EXPECT_EQ(outputs1[0].Get<uint64_t>({1}), 0ULL);
  EXPECT_EQ(outputs1[1].Get<uint64_t>({1}), 0ULL);

  const bool first_ran_first = outputs1[0].Get<uint64_t>({0}) == kInit;
  if (first_ran_first) {
    EXPECT_EQ(outputs1[0].Get<uint64_t>({0}), kInit);
    EXPECT_EQ(outputs1[1].Get<uint64_t>({0}), kInit + kDeltaFirst);
  } else {
    EXPECT_EQ(outputs1[1].Get<uint64_t>({0}), kInit);
    EXPECT_EQ(outputs1[0].Get<uint64_t>({0}), kInit + kDeltaSecond);
  }

  ASSERT_OK_AND_ASSIGN(Literal result2,
                       test_runner().ExecuteWithExecutable(
                           executable.get(), absl::Span<const Literal>{}));
  std::vector<Literal> outputs2 = result2.DecomposeTuple();
  ASSERT_EQ(outputs2.size(), 2);
  EXPECT_EQ(outputs2[0].Get<uint64_t>({1}), 0ULL);
  EXPECT_EQ(outputs2[1].Get<uint64_t>({1}), 0ULL);
  const uint64_t run2_start = kInit + kDeltaFirst + kDeltaSecond;
  if (first_ran_first) {
    EXPECT_EQ(outputs2[0].Get<uint64_t>({0}), run2_start);
    EXPECT_EQ(outputs2[1].Get<uint64_t>({0}), run2_start + kDeltaFirst);
  } else {
    EXPECT_EQ(outputs2[1].Get<uint64_t>({0}), run2_start);
    EXPECT_EQ(outputs2[0].Get<uint64_t>({0}), run2_start + kDeltaSecond);
  }
}

TEST_F(RngGetAndUpdateStateTest, ExecuteLoopAdvancesState) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule RngStateLoop
    cond {
      state = (s32[], u64[2]{0}, u64[2]{0}) parameter(0)
      idx = s32[] get-tuple-element(state), index=0
      limit = s32[] constant(2)
      ROOT lt = pred[] compare(idx, limit), direction=LT
    }
    body {
      state = (s32[], u64[2]{0}, u64[2]{0}) parameter(0)
      idx = s32[] get-tuple-element(state), index=0
      prev = u64[2]{0} get-tuple-element(state), index=2
      curr = u64[2]{0} rng-get-and-update-state(), delta=131072
      one = s32[] constant(1)
      next_idx = s32[] add(idx, one)
      ROOT out = (s32[], u64[2]{0}, u64[2]{0}) tuple(next_idx, prev, curr)
    }
    ENTRY main {
      zero = s32[] constant(0)
      init = u64[2]{0} constant({0, 0})
      state = (s32[], u64[2]{0}, u64[2]{0}) tuple(zero, init, init)
      ROOT loop = (s32[], u64[2]{0}, u64[2]{0}) while(state),
        condition=cond, body=body
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {}));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 3);
  EXPECT_EQ(outputs[1].Get<uint64_t>({0}), 0x7012395ULL);
  EXPECT_EQ(outputs[1].Get<uint64_t>({1}), 0ULL);
  EXPECT_EQ(outputs[2].Get<uint64_t>({0}), 0x7012395ULL + 131072ULL);
  EXPECT_EQ(outputs[2].Get<uint64_t>({1}), 0ULL);
}

TEST_F(RngGetAndUpdateStateTest, ExecuteLoopCarryIntoHighWord) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule RngStateCarry
    cond {
      state = (s32[], u64[2]{0}) parameter(0)
      idx = s32[] get-tuple-element(state), index=0
      limit = s32[] constant(4)
      ROOT lt = pred[] compare(idx, limit), direction=LT
    }
    body {
      state = (s32[], u64[2]{0}) parameter(0)
      idx = s32[] get-tuple-element(state), index=0
      curr = u64[2]{0} rng-get-and-update-state(), delta=8070450532247928832
      one = s32[] constant(1)
      next_idx = s32[] add(idx, one)
      ROOT out = (s32[], u64[2]{0}) tuple(next_idx, curr)
    }
    ENTRY main {
      zero = s32[] constant(0)
      init = u64[2]{0} constant({0, 0})
      state = (s32[], u64[2]{0}) tuple(zero, init)
      ROOT loop = (s32[], u64[2]{0}) while(state),
        condition=cond, body=body
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {}));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 2);
  // 0x7012395 + 3 * 0x7000000000000000 = 0x1_5000000007012395.
  EXPECT_EQ(outputs[1].Get<uint64_t>({0}), 0x5000000007012395ULL);
  EXPECT_EQ(outputs[1].Get<uint64_t>({1}), 1ULL);
}

TEST_F(RngGetAndUpdateStateTest,
       DifferentValuesForMultipleCallsToSameRngComputation) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule MultipleCallsToSameRngComputation
    sample {
      lo = u32[] parameter(0)
      hi = u32[] parameter(1)
      ROOT r = u32[4]{0} rng(lo, hi), distribution=rng_uniform
    }
    ENTRY main {
      lo = u32[] constant(0)
      hi = u32[] constant(1000000)
      a = u32[4]{0} call(lo, hi), to_apply=sample
      b = u32[4]{0} call(lo, hi), to_apply=sample
      ROOT out = (u32[4]{0}, u32[4]{0}) tuple(a, b)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  HloRunnerInterface::ReplicatedExecuteOptions options;
  // A fixed seed pins the Philox key that both calls share, so the outputs are
  // deterministic and can only differ through the shared counter.
  options.seed = 42;
  options.run_hlo_passes = true;
  ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> replica_results,
      test_runner().ExecuteReplicated(std::move(module), options));
  ASSERT_EQ(replica_results.size(), 1);
  std::vector<Literal> results = std::move(replica_results[0]).DecomposeTuple();
  ASSERT_EQ(results.size(), 2);
  EXPECT_FALSE(LiteralTestUtil::Equal(results[0], results[1]));
}

}  // namespace
}  // namespace xla::gpu
