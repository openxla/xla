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
#include "xla/backends/gpu/codegen/emitters/loop.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAreArray;

// There is no gfx1151 test device, so reuse the RDNA3 one with its arch.
se::DeviceDescription Gfx1151DeviceInfo() {
  se::DeviceDescription device = TestGpuDeviceInfo::AMDRX7900DeviceInfo();
  device.set_gpu_compute_capability(
      se::GpuComputeCapability(se::RocmComputeCapability("gfx1151")));
  return device;
}

class LoopFusionTest : public HloHardwareIndependentTestBase {
 public:
  LoopFusionTest() {
    RegisterSymbolicExprStorage(&mlir_context_);
    mlir_context_.loadDialect<mlir::affine::AffineDialect>();
  }

 protected:
  // A loop fusion with result[j, i, k] = operand[i, j, k] + 1 if `transpose`
  // is true and result[i, j, k] = operand[i, j, k] + 1 otherwise. The array has
  // the shape bf16[dim, dim, minor].
  static std::string FusionHlo(int64_t dim, int64_t minor, bool transpose) {
    const std::string shape =
        absl::StrCat("bf16[", dim, ",", dim, ",", minor, "]{2,1,0}");
    return absl::Substitute(R"(
      fused_computation {
        p0 = $0 parameter(0)
        transposed = $0 $1
        one = bf16[] constant(1)
        broadcast = $0 broadcast(one), dimensions={}
        ROOT add = $0 add(transposed, broadcast)
      }

      ENTRY main {
        p0 = $0 parameter(0)
        ROOT fusion = $0 fusion(p0), kind=kLoop, calls=fused_computation
      })",
                            shape,
                            transpose ? "transpose(p0), dimensions={1,0,2}"
                                      : "copy(p0)");
  }

  // Returns the index in the output of the first element that the work item
  // `thread` of the first work group computes, if the fusion is analyzed for
  // `device_info`, and the number of elements per work item.
  struct FirstElement {
    llvm::SmallVector<int64_t, 4> index;
    int64_t unroll_factor;
  };
  absl::StatusOr<FirstElement> GetFirstElement(
      const se::DeviceDescription& device_info, int64_t dim, int64_t minor,
      bool transpose, int64_t thread) {
    TF_ASSIGN_OR_RETURN(auto module,
                        ParseAndReturnVerifiedModule(
                            FusionHlo(dim, minor, transpose)));
    HloFusionAnalysis analysis = HloFusionAnalysis::Create(
        *module->entry_computation()->root_instruction(), device_info);
    CHECK(analysis.emitter_fusion_kind() ==
          HloFusionAnalysis::EmitterFusionKind::kLoop);
    LoopFusion fusion(analysis);

    std::optional<IndexingMap> indexing =
        fusion.ComputeThreadIdToOutputIndexing(/*root_index=*/0,
                                               &mlir_context_);
    CHECK(indexing.has_value());
    // The only symbol left is the index of the element within the work item,
    // because one pass of the grid covers the whole output.
    CHECK_EQ(indexing->GetSymbolCount(), 1);
    auto constant = [&](int64_t value) {
      return CreateSymbolicConstant(value, &mlir_context_);
    };
    // Dimensions are the work item ids followed by the work group ids.
    llvm::SmallVector<SymbolicExpr> dims = {constant(thread), constant(0),
                                            constant(0),      constant(0),
                                            constant(0),      constant(0)};
    return FirstElement{indexing->Evaluate(dims, {constant(0)}),
                        fusion.unroll_factor()};
  }

  // The index of the `position`-th element of a [dim, dim, minor] array when
  // the elements are visited as [dim / block, dim, block, minor]. A block of 1
  // is the memory order.
  static llvm::SmallVector<int64_t, 3> VisitedIndex(int64_t position,
                                                    int64_t dim, int64_t minor,
                                                    int64_t block) {
    int64_t in_block = position / minor % block;
    int64_t i = position / (minor * block) % dim;
    int64_t block_id = position / (minor * block * dim);
    return {block_id * block + in_block, i, position % minor};
  }

  mlir::MLIRContext mlir_context_;
};

TEST_F(LoopFusionTest, Gfx1151ReadsAdjacentRowsOfTransposeOperandTogether) {
  se::DeviceDescription device = Gfx1151DeviceInfo();
  // The rows are 256 bytes long and 256 KiB apart: 16 rows are 4 KiB. The
  // dimension has to be a multiple of the number of rows that are read
  // together, 1032 is only a multiple of 8.
  for (auto [dim, block] : {std::pair{1024, 16}, std::pair{1032, 8}}) {
    for (int64_t thread : {1, 32, 33, 127}) {
      ASSERT_OK_AND_ASSIGN(
          FirstElement element,
          GetFirstElement(device, dim, /*minor=*/128, /*transpose=*/true,
                          thread));
      EXPECT_THAT(element.index,
                  ElementsAreArray(VisitedIndex(thread * element.unroll_factor,
                                                dim, 128, block)))
          << "dim " << dim << " thread " << thread;
    }
  }
}

TEST_F(LoopFusionTest, KeepsMemoryOrderIfRowsDoNotNeedToBeReadTogether) {
  struct TestCase {
    std::string name;
    se::DeviceDescription device;
    int64_t dim;
    int64_t minor;
    bool transpose;
  };
  const se::DeviceDescription gfx1151 = Gfx1151DeviceInfo();
  const TestCase test_cases[] = {
      // The rows are not a multiple of 512 bytes apart.
      {"unaligned rows", gfx1151, 1023, 128, true},
      // Rows of 8 KiB already cover all memory channels.
      {"long rows", gfx1151, 256, 4096, true},
      {"no transpose", gfx1151, 1024, 128, false},
      // Only enabled on gfx1151, where the slowdown was measured.
      {"gfx1100", TestGpuDeviceInfo::AMDRX7900DeviceInfo(), 1024, 128, true},
      {"mi300", TestGpuDeviceInfo::AMDMI300DeviceInfo(), 1024, 128, true},
      {"cuda", TestGpuDeviceInfo::RTXA6000DeviceInfo(), 1024, 128, true},
  };
  for (const TestCase& test_case : test_cases) {
    for (int64_t thread : {1, 32, 33, 127}) {
      ASSERT_OK_AND_ASSIGN(
          FirstElement element,
          GetFirstElement(test_case.device, test_case.dim, test_case.minor,
                          test_case.transpose, thread));
      EXPECT_THAT(element.index,
                  ElementsAreArray(VisitedIndex(thread * element.unroll_factor,
                                                test_case.dim, test_case.minor,
                                                /*block=*/1)))
          << test_case.name << ", thread " << thread;
    }
  }
}

}  // namespace
}  // namespace xla::gpu
