/* Copyright 2018 The OpenXLA Authors.

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

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/xla_test_backend_predicates.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class ScatterTest : public HloPjRtInterpreterReferenceMixin<HloTestBase> {
 protected:
  void RunTest(const absl::string_view hlo_text, Literal* const operand,
               Literal* const scatter_indices, Literal* const updates) {
    RunTest(hlo_text, {operand, scatter_indices, updates});
  }

  void RunTest(const absl::string_view hlo_text,
               const absl::Span<Literal* const> args) {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_text, config));
    EXPECT_TRUE(RunAndCompare(std::move(module), args, std::nullopt));
  }
};

TEST_F(ScatterTest, TensorFlowScatterV1_Update) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, TensorFlowScatterV1_UpdateTwice) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 0});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      Execute(std::move(module), {&operand, &scatter_indices, &updates}));
  auto data = result.data<int32_t>();
  // The first row is written twice, so it is not deterministic which of the two
  // writes "wins".
  EXPECT_TRUE(data[0] == 10 || data[0] == 70);
  EXPECT_TRUE(data[1] == 20 || data[1] == 80);
  EXPECT_TRUE(data[2] == 30 || data[2] == 90);
  EXPECT_EQ(data.subspan(3), (std::vector<int32_t>{4, 5, 6, 7, 8, 9}));
}

TEST_F(ScatterTest, TensorFlowScatterV1_WithFusedAdds) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  p0 = s32[3,3] parameter(0)
  operand = s32[3,3] add(p0, p0)
  p1 = s32[2] parameter(1)
  indices = s32[2] add(p1, p1)
  p2 = s32[2,3] parameter(2)
  updates = s32[2,3] add(p2, p2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 1});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, TensorFlowScatterV2_Update) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterV2

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[3,2] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={0},
      inserted_window_dims={1},
      scatter_dims_to_operand_dims={1},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 30}, {40, 60}, {70, 90}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, TensorFlowScatterV2_InversePermutation) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterV2

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  permutation = s32[3,4] parameter(0)
  reshape = s32[3,4,1] reshape(permutation)
  operand = s32[3,4] iota(), iota_dimension=1
  updates = s32[3,4,1,1] iota(), iota_dimension=1
  iota = s32[3,4,1] iota(), iota_dimension=0
  indices = s32[3,4,2] concatenate(iota, reshape), dimensions={2}
  ROOT scatter = s32[3,4] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={2,3},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=2
}
)";
  Literal permutation = LiteralUtil::CreateR2<int32_t>(
      {{1, 3, 2, 0}, {3, 0, 2, 1}, {2, 3, 1, 0}});
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  TF_ASSERT_OK_AND_ASSIGN(Literal actual,
                          Execute(std::move(module), {&permutation}));
  Literal expected = LiteralUtil::CreateR2<int32_t>(
      {{3, 0, 2, 1}, {1, 3, 2, 0}, {3, 2, 0, 1}});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
}

TEST_F(ScatterTest, SimpleR4) {
  const char* hlo_text = R"(
HloModule SimpleR4

add_f32 (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(f32[] lhs, f32[] rhs)
}

ENTRY main {
  operand = f32[1,2,2,1] parameter(0)
  indices = s32[1,3] parameter(1)
  updates = f32[1,2,2,1] parameter(2)
  ROOT scatter = f32[1,2,2,1] scatter(operand, indices, updates),
      to_apply=add_f32,
      update_window_dims={1,2,3},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0, 2, 1},
      index_vector_dim=1
}
)";

  Literal operand =
      LiteralUtil::CreateR4<float>({{{{0.f}, {0.f}}, {{0.f}, {0.f}}}});
  Literal updates =
      LiteralUtil::CreateR4<float>({{{{0.12}, {0.28}}, {{0.018}, {0.42}}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 0, 0}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, TensorFlowScatter_Add) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatter_Add

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, TensorFlowScatter_Add_UniqueIndices) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatter_Add

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1,
      unique_indices=true
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, TensorFlowScatter_Mul) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatter_Mul

mul_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT mul = s32[] multiply(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=mul_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, TensorFlowScatter_F32) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatter_F32

add_f32 (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(f32[] lhs, f32[] rhs)
}

ENTRY main {
  operand = f32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = f32[2,3] parameter(2)
  ROOT scatter = f32[3,3] scatter(operand, indices, updates),
      to_apply=add_f32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand = LiteralUtil::CreateR2<float>(
      {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({2, 1});
  Literal updates =
      LiteralUtil::CreateR2<float>({{0.4, 1.1, 0.7}, {2.3, 3.1, 1.6}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, TensorFlowScatter_F16) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatter_F16

add_f16 (lhs: f16[], rhs: f16[]) -> f16[] {
  lhs = f16[] parameter(0)
  rhs = f16[] parameter(1)
  ROOT add = f16[] add(f16[] lhs, f16[] rhs)
}

ENTRY main {
  operand = f16[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = f16[2,3] parameter(2)
  ROOT scatter = f16[3,3] scatter(operand, indices, updates),
      to_apply=add_f16,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Array2D<Eigen::half> operand_array(
      {{1.1f, 2.2f, 3.3f}, {4.4f, 5.5f, 6.6f}, {7.7f, 8.8f, 9.9f}});
  Literal operand(ShapeUtil::MakeShape(F16, {3, 3}));
  operand.PopulateR2FromArray2D(operand_array);
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({2, 1});
  Array2D<Eigen::half> updates_array({{0.4f, 1.1f, 0.7f}, {2.3f, 3.1f, 1.6f}});
  Literal updates(ShapeUtil::MakeShape(F16, {2, 3}));
  updates.PopulateR2FromArray2D(updates_array);
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, TensorFlowScatter_RepeatedIndices) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({1, 1});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, TensorFlowScatter_MultipleBatchDims) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterMultipleBatchDims

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s32[2,3,2] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={1},
      inserted_window_dims={1},
      scatter_dims_to_operand_dims={1},
      index_vector_dim=2
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 2}, {2, 1}});
  Literal updates = LiteralUtil::CreateR3<int32_t>(
      {{{10, 30}, {40, 60}, {70, 90}}, {{5, 5}, {5, 5}, {5, 5}}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, TensorFlowScatterNd) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterNd

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s32[2,2] parameter(2)
  ROOT scatter = s32[3,3,2] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  Literal updates = LiteralUtil::CreateR2<int32_t>({{-10, 10}, {-40, 40}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, TensorFlowScatterNdS64) {
  constexpr char hlo_text[] = R"(
HloModule S64Scatter

update {
  lhs = s64[] parameter(0)
  ROOT rhs = s64[] parameter(1)
}

ENTRY main {
  operand = s64[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s64[2,2] parameter(2)
  ROOT scatter = s64[3,3,2] scatter(operand, indices, updates),
      to_apply=update,
      update_window_dims={1},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR3<int64_t>({{{-1, 1LL << 62}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6LL << 59}},  //
                                      {{-7, 7}, {-8, 8LL << 49}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  Literal updates =
      LiteralUtil::CreateR2<int64_t>({{-10, 10LL << 46}, {-(4LL << 38), 40}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, TensorFlowScatterNd_NonDefaultIndexVectorDim) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterNdNonDefaultIndexVectorDim

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s32[2,2] parameter(2)
  ROOT scatter = s32[3,3,2] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=0
}
)";
  Literal operand =
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  Literal updates = LiteralUtil::CreateR2<int32_t>({{-10, 10}, {-20, 20}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, DynamicUpdateSlice) {
  const char* hlo_text = R"(
HloModule DynamicUpdateSlice

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[1,1] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={0,1},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=0
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({1, 1});
  Literal updates = LiteralUtil::CreateR2<int32_t>({{10}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, BatchDynamicUpdateSlice) {
  const char* hlo_text = R"(
HloModule BatchDynamicUpdateSlice

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s32[2,1,1] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1,2},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=0
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{2, 1}, {1, 1}});
  Literal updates = LiteralUtil::CreateR3<int32_t>({{{10}}, {{20}}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, ZeroDimBounds) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter_ZeroDimBounds

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,0] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,0] parameter(2)
  ROOT scatter = s32[3,0] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand = LiteralUtil::CreateR2<int32_t>({{}, {}, {}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates = LiteralUtil::CreateR2<int32_t>({{}, {}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, NoUpdateWindowDims) {
  const std::string hlo_text = R"(
HloModule Scatter_NoUpdateWindowDims

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3] parameter(0)
  indices = s32[2,2,1] parameter(1)
  updates = s32[2,2] parameter(2)
  ROOT scatter = s32[3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=2
}
)";
  Literal operand = LiteralUtil::CreateR1<int32_t>({0, 1, 2});
  Literal scatter_indices =
      LiteralUtil::CreateR3<int32_t>({{{0}, {1}}, {{2}, {1}}});
  Literal updates = LiteralUtil::CreateR2<int32_t>({{10, 20}, {30, 40}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, OutOfBoundsIndex) {
  const std::string hlo_text = R"(
HloModule BatchDynamicSlice

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3]{1,0} parameter(0)
  indices = s32[6,2]{1,0} parameter(1)
  updates = s32[6,1,1]{2,1,0} parameter(2)
  ROOT scatter = s32[3,3]{1,0} scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1,2},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>(
      {{2, 7}, {2, 1}, {1, 1}, {5, 1}, {2147483647, 1}, {1, 2}});
  Literal updates = LiteralUtil::CreateR3<int32_t>(
      {{{10}}, {{20}}, {{30}}, {{40}}, {{50}}, {{60}}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, OutOfBoundsUnsignedIndex) {
  const std::string hlo_text = R"(
HloModule BatchDynamicSlice

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3]{1,0} parameter(0)
  indices = u32[6,2]{1,0} parameter(1)
  updates = s32[6,1,1]{2,1,0} parameter(2)
  ROOT scatter = s32[3,3]{1,0} scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1,2},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<uint32_t>(
      {{2, 7}, {2, 1}, {1, 1}, {5, 1}, {2147483648u, 1}, {1, 2}});
  Literal updates = LiteralUtil::CreateR3<int32_t>(
      {{{10}}, {{20}}, {{30}}, {{40}}, {{50}}, {{60}}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, U8Index) {
  const std::string hlo_text = R"(
HloModule BatchDynamicSlice

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[129,3]{1,0} parameter(0)
  indices = u8[6,2]{1,0} parameter(1)
  updates = s32[6,1,1]{2,1,0} parameter(2)
  ROOT scatter = s32[129,3]{1,0} scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1,2},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateRandomLiteral<S32>(ShapeUtil::MakeShape(S32, {129, 3}),
                                            /*mean=*/500, /*stddev=*/100)
          .value();
  Literal scatter_indices = LiteralUtil::CreateR2<uint8_t>(
      {{2, 7}, {2, 1}, {1, 1}, {5, 1}, {0x80, 1}, {1, 2}});
  Literal updates = LiteralUtil::CreateR3<int32_t>(
      {{{10}}, {{20}}, {{30}}, {{40}}, {{50}}, {{60}}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, U8IndexWithBoundOutsideIndexType) {
  // The valid index bound (299) does not fit into u8; index 200 must still
  // be applied.
  const std::string hlo_text = R"(
HloModule U8IndexWithBoundOutsideIndexType

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  operand = s32[300]{0} parameter(0)
  indices = u8[1,1]{1,0} parameter(1)
  updates = s32[1]{0} parameter(2)
  ROOT scatter = s32[300]{0} scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateRandomLiteral<S32>(ShapeUtil::MakeShape(S32, {300}),
                                            /*mean=*/500, /*stddev=*/100)
          .value();
  Literal scatter_indices = LiteralUtil::CreateR2<uint8_t>({{200}});
  Literal updates = LiteralUtil::CreateR1<int32_t>({1000});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, NegativeIndex) {
  const std::string hlo_text = R"(
HloModule BatchDynamicSlice

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3]{1,0} parameter(0)
  indices = s32[6,2]{1,0} parameter(1)
  updates = s32[6,1,1]{2,1,0} parameter(2)
  ROOT scatter = s32[3,3]{1,0} scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1,2},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices =
      LiteralUtil::CreateR2<int32_t>({{2, 7},
                                      {2, 1},
                                      {1, 1},
                                      {-500, 1},
                                      {static_cast<int32_t>(-2147483648), 1},
                                      {1, 2}});
  Literal updates = LiteralUtil::CreateR3<int32_t>(
      {{{10}}, {{20}}, {{30}}, {{40}}, {{50}}, {{60}}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, OutOfBoundsUpdateWindow) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterNd_OobUpdateWindow

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[1,2] parameter(1)
  updates = s32[1,2,2] parameter(2)
  ROOT scatter = s32[3,3,2] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1,2},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 2}});
  Literal updates = LiteralUtil::CreateR3<int32_t>({{{-10, 10}, {-40, 40}}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, OneScalarIndex) {
  const char* hlo_text = R"(
HloModule OneScalarIndex

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[2,3,2]{2,1,0} parameter(0)
  index = s32[] parameter(1)
  updates = s32[1,3,2]{2,1,0} parameter(2)
  ROOT scatter = s32[2,3,2]{2,1,0} scatter(operand, index, updates),
      to_apply=update_s32,
      update_window_dims={0,1,2},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=0
}
)";
  Literal operand = LiteralUtil::CreateR3<int32_t>(
      {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}});
  Literal scatter_indices = LiteralUtil::CreateR0<int32_t>(1);
  Literal updates =
      LiteralUtil::CreateR3<int32_t>({{{10, 20}, {30, 40}, {50, 60}}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, ScalarUpdate) {
  const char* hlo_text = R"(
HloModule ScalarUpdate

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[4]{0} parameter(0)
  index = s32[] parameter(1)
  updates = s32[] parameter(2)
  ROOT scatter = s32[4]{0} scatter(operand, index, updates),
      to_apply=update_s32,
      update_window_dims={},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=0
}
)";
  Literal operand = LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4});
  Literal scatter_indices = LiteralUtil::CreateR0<int32_t>(1);
  Literal updates = LiteralUtil::CreateR0<int32_t>(25);
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, EmptyIndices) {
  const std::string hlo_text = R"(
HloModule EmptyIndices

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3] parameter(0)
  indices = s32[0] parameter(1)
  updates = s32[0] parameter(2)
  ROOT scatter = s32[3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand = LiteralUtil::CreateR1<int32_t>({1, 2, 3});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({});
  Literal updates = LiteralUtil::CreateR1<int32_t>({});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, ScatterIntoScalar) {
  const char* hlo_text = R"(
HloModule ScatterIntoScalar

update_s32 {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  parameter.1 = s32[] parameter(0)
  parameter.2 = s32[0]{0} parameter(1)
  parameter.3 = s32[] parameter(2)
  ROOT scatter = s32[] scatter(parameter.1, parameter.2, parameter.3),
      update_window_dims={},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={},
      index_vector_dim=0,
      to_apply=update_s32
}
)";
  Literal operand = LiteralUtil::CreateR0<int32_t>(1);
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({});
  Literal updates = LiteralUtil::CreateR0<int32_t>(2);
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, ScatterMultiUpdateIntoScalar) {
  constexpr absl::string_view kS32Hlo = R"hlo(
HloModule ScatterMultiUpdateIntoScalarS32

add_s32 {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT sum = s32[] add(lhs, rhs)
}

ENTRY main {
  operand = s32[] parameter(0)
  indices = s32[2,3,0]{2,1,0} parameter(1)
  updates = s32[2,3]{1,0} parameter(2)
  ROOT scatter = s32[] scatter(operand, indices, updates),
      update_window_dims={},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={},
      index_vector_dim=2,
      to_apply=add_s32
}
)hlo";
  Literal s32_operand = LiteralUtil::CreateR0<int32_t>(10);
  Literal s32_indices(ShapeUtil::MakeShape(S32, {2, 3, 0}));
  Literal s32_updates = LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, -6}});
  RunTest(kS32Hlo, &s32_operand, &s32_indices, &s32_updates);

  constexpr absl::string_view kF32Hlo = R"hlo(
HloModule ScatterMultiUpdateIntoScalarF32

add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY main {
  operand = f32[] parameter(0)
  indices = s32[3,0]{1,0} parameter(1)
  updates = f32[3]{0} parameter(2)
  ROOT scatter = f32[] scatter(operand, indices, updates),
      update_window_dims={},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={},
      index_vector_dim=1,
      to_apply=add_f32
}
)hlo";
  Literal f32_operand = LiteralUtil::CreateR0<float>(10.0f);
  Literal f32_indices(ShapeUtil::MakeShape(S32, {3, 0}));
  Literal f32_updates = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 4.0f});
  RunTest(kF32Hlo, &f32_operand, &f32_indices, &f32_updates);
}

TEST_F(ScatterTest, Multioutput) {
  constexpr char hlo_text[] = R"(
HloModule MultioutputScatter

update {
  lhs0 = s32[] parameter(0)
  lhs1 = f32[] parameter(1)
  rhs0 = s32[] parameter(2)
  rhs1 = f32[] parameter(3)
  ROOT tuple = (s32[], f32[]) tuple(rhs0, rhs1)
}

ENTRY main {
  operand0 = s32[3,3,2] parameter(0)
  operand1 = f32[3,3,2] parameter(1)
  indices = s32[2,2] parameter(2)
  updates0 = s32[2,2] parameter(3)
  updates1 = f32[2,2] parameter(4)
  ROOT scatter = (s32[3,3,2], f32[3,3,2]) scatter(operand0, operand1, indices, updates0, updates1),
      to_apply=update,
      update_window_dims={1},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  Literal operand0 =
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal operand1 =
      LiteralUtil::CreateR3<float>({{{-2, 2}, {-3, 3}, {-4, 4}},  //
                                    {{-5, 5}, {-6, 6}, {-7, 7}},  //
                                    {{-8, 8}, {-9, 9}, {-10, 10}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  Literal updates0 = LiteralUtil::CreateR2<int32_t>({{-10, 10}, {-40, 40}});
  Literal updates1 = LiteralUtil::CreateR2<float>({{-11, 11}, {-41, 41}});
  RunTest(hlo_text,
          {&operand0, &operand1, &scatter_indices, &updates0, &updates1});
}

TEST_F(ScatterTest, MultioutputSameOperand) {
  // TODO(b/435078848): Currently fails on XLA/CPU.
  if (test::DeviceTypeIsOneOf({test::kCpu})) {
    GTEST_SKIP();
  }
  constexpr char hlo_text[] = R"(
HloModule MultioutputScatter

update {
  lhs0 = s32[] parameter(0)
  lhs1 = s32[] parameter(1)
  rhs0 = s32[] parameter(2)
  rhs1 = s32[] parameter(3)
  ROOT tuple = (s32[], s32[]) tuple(rhs0, rhs1)
}

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  updates0 = s32[2,2] parameter(2)
  updates1 = s32[2,2] parameter(3)
  ROOT scatter = (s32[3,3,2], s32[3,3,2]) scatter(operand, operand, indices, updates0, updates1),
      to_apply=update,
      update_window_dims={1},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={0,1},
      index_vector_dim=1
}
)";
  Literal operand =
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  Literal updates0 = LiteralUtil::CreateR2<int32_t>({{-10, 10}, {-40, 40}});
  Literal updates1 = LiteralUtil::CreateR2<int32_t>({{-11, 11}, {-41, 41}});
  RunTest(hlo_text, {&operand, &scatter_indices, &updates0, &updates1});
}

TEST_F(ScatterTest, TensorFlowScatter_Max_F32) {
  const std::string hlo_text = R"(
HloModule TensorFlowScatter_Max_F32

max_f32 (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT max = f32[] maximum(f32[] lhs, f32[] rhs)
}

ENTRY main {
  operand = f32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = f32[2,3] parameter(2)
  ROOT scatter = f32[3,3] scatter(operand, indices, updates),
      to_apply=max_f32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  Literal operand = LiteralUtil::CreateR2<float>(
      {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({2, 1});
  Literal updates =
      LiteralUtil::CreateR2<float>({{0.4, 1.1, 0.7}, {2.3, 3.1, 1.6}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, Scatter_Add_F32) {
  const std::string hlo_text = R"(
HloModule scatter_add

region_0.1 {
  scatter-add.2 = f32[] parameter(0)
  scatter-add.3 = f32[] parameter(1)
  ROOT add.2 = f32[] add(scatter-add.2, scatter-add.3)
}

ENTRY main.2 {
  constant.4 = f32[] constant(0)
  broadcast.4 = f32[2,2,4]{2,1,0} broadcast(constant.4), dimensions={}
  channel_idxs.1 = s32[2,2]{1,0} constant({{0,3}, {1,2}})
  constant.5 = s32[] constant(0)
  broadcast.5 = s32[2,2]{1,0} broadcast(constant.5), dimensions={}
  lt.1 = pred[2,2]{1,0} compare(channel_idxs.1, broadcast.5), direction=LT
  constant.3 = s32[] constant(4)
  broadcast.3 = s32[2,2]{1,0} broadcast(constant.3), dimensions={}
  add.3 = s32[2,2]{1,0} add(channel_idxs.1, broadcast.3)
  select_n.1 = s32[2,2]{1,0} select(lt.1, add.3, channel_idxs.1)
  broadcast_in_dim.1 = s32[2,2,1]{2,1,0} reshape(select_n.1)
  arr.1 = f32[2,2,2]{2,1,0} parameter(0)
  ROOT scatter-add.5 = f32[2,2,4]{2,1,0} scatter(broadcast.4, broadcast_in_dim.1, arr.1), update_window_dims={1}, inserted_window_dims={2}, scatter_dims_to_operand_dims={2}, input_batching_dims={0}, scatter_indices_batching_dims={0}, index_vector_dim=2, to_apply=region_0.1
}
)";
  Literal updates = LiteralUtil::CreateR3<float>(
      {{{1.0, 1.1}, {2.0, 2.1}}, {{3.0, 3.1}, {4.0, 4.1}}});

  RunTest(hlo_text, {&updates});
}

TEST_F(ScatterTest, AddScatterRepeatedParameterNotCombined) {
  constexpr absl::string_view kHloText = R"hlo(
HloModule ScatterAddRepeatedParameter
dp0 { p0 = s16[] parameter(0) p1 = s16[] parameter(1)
      ROOT r = s16[] add(p0, p0) }
dp1 { p0 = s16[] parameter(0) p1 = s16[] parameter(1)
      ROOT r = s16[] add(p1, p1) }
as16 { p0 = s16[] parameter(0) p1 = s16[] parameter(1)
       ROOT r = s16[] add(p0, p1) }
ENTRY main {
  op1 = s16[8] parameter(0) ix1 = s32[2,1] parameter(1)
  up1 = s16[2] parameter(2) op2 = s16[8] parameter(3)
  ix2 = s32[2,1] parameter(4) up2 = s16[2] parameter(5)
  s_lhs = s16[8] scatter(op1, ix1, up1), update_window_dims={}, index_vector_dim=1,
      inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, to_apply=dp0
  out_lhs = s16[8] add(s_lhs, op2)
  s_rhs = s16[8] scatter(op1, ix1, up1), update_window_dims={}, index_vector_dim=1,
      inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, to_apply=dp1
  out_rhs = s16[8] add(op2, s_rhs)
  s_cat0 = s16[8] scatter(op1, ix1, up1), update_window_dims={}, index_vector_dim=1,
      inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, to_apply=dp0
  s_cat1 = s16[8] scatter(op2, ix2, up2), update_window_dims={}, index_vector_dim=1,
      inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, to_apply=as16
  out_cat = s16[8] add(s_cat0, s_cat1)
  s_chn0 = s16[8] scatter(op1, ix1, up1), update_window_dims={}, index_vector_dim=1,
      inserted_window_dims={0}, scatter_dims_to_operand_dims={0},
      unique_indices=true, to_apply=dp0
  s_chn1 = s16[8] scatter(op2, ix1, up2), update_window_dims={}, index_vector_dim=1,
      inserted_window_dims={0}, scatter_dims_to_operand_dims={0},
      unique_indices=true, to_apply=as16
  out_chn = s16[8] add(s_chn0, s_chn1)
  ROOT out = (s16[8], s16[8], s16[8], s16[8]) tuple(
      out_lhs, out_rhs, out_cat, out_chn)
}
)hlo";
  Literal op1 = LiteralUtil::CreateR1<int16_t>({1, 2, 3, 4, 5, 6, 7, 8});
  Literal idx1 = LiteralUtil::CreateR2<int32_t>({{1}, {2}});
  Literal upd1 = LiteralUtil::CreateR1<int16_t>({5, 7});
  Literal op2 =
      LiteralUtil::CreateR1<int16_t>({10, 20, 30, 40, 50, 60, 70, 80});
  Literal idx2 = LiteralUtil::CreateR2<int32_t>({{4}, {5}});
  Literal upd2 = LiteralUtil::CreateR1<int16_t>({100, 200});
  RunTest(kHloText, {&op1, &idx1, &upd1, &op2, &idx2, &upd2});
}

TEST_F(ScatterTest, EmptyIndexVector) {
  const std::string hlo_text = R"(
HloModule ScatterEmptyIndexVector

mul_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT mul = s32[] multiply(lhs, rhs)
}

ENTRY main {
  operand = s32[1]{0} parameter(0)
  indices = s32[2,0]{1,0} parameter(1)
  updates = s32[2,1]{1,0} parameter(2)
  ROOT scatter = s32[1]{0} scatter(operand, indices, updates),
      update_window_dims={1},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={},
      index_vector_dim=1,
      to_apply=mul_s32
}
)";
  Literal operand = LiteralUtil::CreateR1<int32_t>({1});
  Literal scatter_indices(ShapeUtil::MakeShape(S32, {2, 0}));
  Literal updates = LiteralUtil::CreateR2<int32_t>({{2}, {3}});
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

TEST_F(ScatterTest, EmptyIndexVectorSortedLarge) {
  const std::string hlo_text = R"(
HloModule ScatterEmptyIndexVectorSortedLarge

max_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT max = s32[] maximum(lhs, rhs)
}

ENTRY main {
  operand = s32[1]{0} parameter(0)
  indices = s32[2000,0]{1,0} parameter(1)
  updates = s32[2000,1]{1,0} parameter(2)
  ROOT scatter = s32[1]{0} scatter(operand, indices, updates),
      update_window_dims={1},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={},
      index_vector_dim=1,
      indices_are_sorted=true,
      unique_indices=false,
      to_apply=max_s32
}
)";
  Literal operand = LiteralUtil::CreateR1<int32_t>({-100});
  Literal scatter_indices(ShapeUtil::MakeShape(S32, {2000, 0}));
  Array2D<int32_t> updates_array(2000, 1, -50);
  updates_array(1234, 0) = -2;
  Literal updates = LiteralUtil::CreateR2FromArray2D(updates_array);
  RunTest(hlo_text, &operand, &scatter_indices, &updates);
}

// Test min/max/add scatters with edge-case values.
class ScatterEdgeCaseTestP
    : public ScatterTest,
      public ::testing::WithParamInterface<absl::string_view /*operator*/> {};

TEST_P(ScatterEdgeCaseTestP, DoIt) {
  using L = std::numeric_limits<float>;
  std::vector<float> edge_cases = {
      0.f,
      -0.f,
      -1.f,
      1.f,
      L::min(),
      -L::min(),
      L::max(),
      L::lowest(),
      L::epsilon(),
      L::infinity(),
      -L::infinity(),
      L::quiet_NaN(),
      -L::quiet_NaN(),
  };
  int n = edge_cases.size();

  float init_value;
  absl::string_view operation = GetParam();
  if (operation == "minimum") {
    init_value = L::infinity();
  } else if (operation == "maximum") {
    init_value = -L::infinity();
  } else if (operation == "add") {
    init_value = 0;
  } else {
    FAIL() << "Invalid operation " << operation;
  }

  // For each pair of values (x,y) in edge_cases, let the initial value be
  // init_value.  Scatter x and y into the same location.
  const std::string hlo_text = absl::Substitute(R"(
HloModule test

max_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT max = maximum(lhs, rhs)
}

ENTRY main {
  init = f32[$0, $0] broadcast(f32[] constant($2))
  indices = s32[$1] parameter(0)
  updates = f32[$1, $0] parameter(1)
  ROOT scatter = f32[$0, $0] scatter(init, indices, updates),
      to_apply=max_f32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)",
                                                n, 2 * n, init_value);

  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_cpu_enable_fast_min_max(false);

  HloModuleConfig config;
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));

  Literal scatter_indices(ShapeUtil::MakeShape(S32, {2 * n}));
  Literal updates(ShapeUtil::MakeShape(F32, {2 * n, n}));
  for (int i = 0; i < n; i++) {
    scatter_indices.Set({i}, i);
    scatter_indices.Set({i + n}, i);
    for (int j = 0; j < n; j++) {
      updates.Set({i, j}, edge_cases[i]);
      updates.Set({i + n, j}, edge_cases[j]);
    }
  }

  // Expect exact numerical matches.
  ErrorSpec spec(/*aabs=*/0);

  // We pass -NaN in the input, but the output might contain +NaN instead.  This
  // is fine; we don't gurantee that XLA preserves the body of NaNs.
  spec.all_nans_are_equivalent = true;

  EXPECT_TRUE(
      RunAndCompare(std::move(module), {&scatter_indices, &updates}, spec));
}

INSTANTIATE_TEST_SUITE_P(EdgeCases, ScatterEdgeCaseTestP,
                         ::testing::Values("minimum", "maximum", "add"));

TEST_F(ScatterTest, LargeParallelHistogramCollisions) {
  constexpr int64_t kNumUpdates = 131072;
  constexpr int64_t kRealBins = 16384;
  constexpr int64_t kComplexBins = 8192;

  constexpr absl::string_view kRealHlo = R"hlo(
    HloModule LargeParallelHistogramCollisionsReal

    add_s32 {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    ENTRY main {
      s32_init = s32[16384] parameter(0)
      f32_init = f32[16384] parameter(1)
      idx = s32[131072,1] parameter(2)
      s32_upd = s32[131072] parameter(3)
      f32_upd = f32[131072] parameter(4)
      s32_out = s32[16384] scatter(s32_init, idx, s32_upd),
          update_window_dims={},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_s32
      f32_out = f32[16384] scatter(f32_init, idx, f32_upd),
          update_window_dims={},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_f32
      ROOT out = (s32[16384], f32[16384]) tuple(s32_out, f32_out)
    }
  )hlo";

  Literal s32_init(ShapeUtil::MakeShape(S32, {kRealBins}));
  Literal f32_init(ShapeUtil::MakeShape(F32, {kRealBins}));
  absl::Span<int32_t> s32_init_data = s32_init.data<int32_t>();
  absl::Span<float> f32_init_data = f32_init.data<float>();
  std::vector<int32_t> expected_s32(kRealBins);
  std::vector<double> expected_f32(kRealBins);
  for (int64_t b = 0; b < kRealBins; ++b) {
    s32_init_data[b] = static_cast<int32_t>(b * 3 - 500);
    f32_init_data[b] = static_cast<float>((b % 97) - 48) * 0.25f;
    expected_s32[b] = s32_init_data[b];
    expected_f32[b] = static_cast<double>(f32_init_data[b]);
  }

  Literal real_idx(ShapeUtil::MakeShape(S32, {kNumUpdates, 1}));
  Literal s32_upd(ShapeUtil::MakeShape(S32, {kNumUpdates}));
  Literal f32_upd(ShapeUtil::MakeShape(F32, {kNumUpdates}));
  absl::Span<int32_t> real_idx_data = real_idx.data<int32_t>();
  absl::Span<int32_t> s32_upd_data = s32_upd.data<int32_t>();
  absl::Span<float> f32_upd_data = f32_upd.data<float>();
  for (int64_t k = 0; k < kNumUpdates; ++k) {
    int32_t bin = 0;
    if (k % 37 == 0) {
      // Negative out of bounds indices must be ignored.
      bin = -static_cast<int32_t>((k % 512) + 1);
    } else if (k % 41 == 0) {
      // Positive out of bounds indices must be ignored.
      bin = static_cast<int32_t>(kRealBins + (k % 1024));
    } else if (k % 7 == 0) {
      // High collision hot bins.
      bin = static_cast<int32_t>((k * 13) % 64);
    } else {
      bin = static_cast<int32_t>((k * 7919 + 101) & (kRealBins - 1));
    }
    real_idx_data[k] = bin;
    s32_upd_data[k] = static_cast<int32_t>((k % 29) - 11);
    f32_upd_data[k] = static_cast<float>((k % 17) - 8) * 0.1f + 0.03f;
    if (bin >= 0 && bin < kRealBins) {
      expected_s32[bin] += s32_upd_data[k];
      expected_f32[bin] += static_cast<double>(f32_upd_data[k]);
    }
  }

  HloModuleConfig real_config;
  real_config.set_debug_options(GetDebugOptionsForTest());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> real_module,
                       ParseAndReturnVerifiedModule(kRealHlo, real_config));
  ASSERT_OK_AND_ASSIGN(
      Literal real_result,
      Execute(std::move(real_module),
              {&s32_init, &f32_init, &real_idx, &s32_upd, &f32_upd}));
  std::vector<Literal> real_outputs = real_result.DecomposeTuple();
  ASSERT_EQ(real_outputs.size(), 2);
  absl::Span<const int32_t> s32_out_data = real_outputs[0].data<int32_t>();
  absl::Span<const float> f32_out_data = real_outputs[1].data<float>();
  ASSERT_EQ(s32_out_data.size(), kRealBins);
  ASSERT_EQ(f32_out_data.size(), kRealBins);
  for (int64_t b = 0; b < kRealBins; ++b) {
    ASSERT_EQ(s32_out_data[b], expected_s32[b]) << "s32 bin " << b;
    ASSERT_NEAR(f32_out_data[b], static_cast<float>(expected_f32[b]), 1e-3f)
        << "f32 bin " << b;
  }

  if (!test::BackendSupportsComplex128()) {
    return;
  }

  constexpr absl::string_view kComplexHlo = R"hlo(
    HloModule LargeParallelHistogramCollisionsC128

    add_c128 {
      lhs = c128[] parameter(0)
      rhs = c128[] parameter(1)
      ROOT sum = c128[] add(lhs, rhs)
    }

    ENTRY main {
      init = c128[8192] parameter(0)
      idx = s32[131072,1] parameter(1)
      upd = c128[131072] parameter(2)
      ROOT out = c128[8192] scatter(init, idx, upd),
          update_window_dims={},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_c128
    }
  )hlo";

  Literal c128_init(ShapeUtil::MakeShape(C128, {kComplexBins}));
  absl::Span<complex128> c128_init_data = c128_init.data<complex128>();
  std::vector<complex128> expected_c128(kComplexBins);
  for (int64_t b = 0; b < kComplexBins; ++b) {
    c128_init_data[b] =
        complex128(static_cast<double>(b), static_cast<double>(-2 * b));
    expected_c128[b] = c128_init_data[b];
  }

  Literal c128_idx(ShapeUtil::MakeShape(S32, {kNumUpdates, 1}));
  Literal c128_upd(ShapeUtil::MakeShape(C128, {kNumUpdates}));
  absl::Span<int32_t> c128_idx_data = c128_idx.data<int32_t>();
  absl::Span<complex128> c128_upd_data = c128_upd.data<complex128>();
  for (int64_t k = 0; k < kNumUpdates; ++k) {
    int32_t bin = 0;
    if (k % 43 == 0) {
      bin = -static_cast<int32_t>((k % 256) + 1);
    } else if (k % 47 == 0) {
      bin = static_cast<int32_t>(kComplexBins + (k % 512));
    } else if (k % 9 == 0) {
      bin = static_cast<int32_t>((k * 7) % 32);
    } else {
      bin = static_cast<int32_t>((k * 6151 + 53) & (kComplexBins - 1));
    }
    c128_idx_data[k] = bin;
    // Integer valued components stay well within exact f64 mantissa range so
    // atomic reordering still yields exact sums.
    c128_upd_data[k] = complex128(static_cast<double>((k % 19) - 9),
                                  static_cast<double>((k % 23) - 10));
    if (bin >= 0 && bin < kComplexBins) {
      expected_c128[bin] += c128_upd_data[k];
    }
  }

  HloModuleConfig c128_config;
  c128_config.set_debug_options(GetDebugOptionsForTest());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> c128_module,
                       ParseAndReturnVerifiedModule(kComplexHlo, c128_config));
  ASSERT_OK_AND_ASSIGN(
      Literal c128_result,
      Execute(std::move(c128_module), {&c128_init, &c128_idx, &c128_upd}));
  absl::Span<const complex128> c128_out_data = c128_result.data<complex128>();
  ASSERT_EQ(c128_out_data.size(), kComplexBins);
  for (int64_t b = 0; b < kComplexBins; ++b) {
    ASSERT_EQ(c128_out_data[b], expected_c128[b]) << "c128 bin " << b;
  }
}

TEST_F(ScatterTest, LargeParallelUniqueIndicesSingleAndVariadic) {
  constexpr int64_t kNumElems = 65536;
  constexpr int64_t kNumUpdates = 65535;

  constexpr absl::string_view kHlo = R"hlo(
    HloModule LargeParallelUniqueIndicesSingleAndVariadic

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    add_pair {
      lhs_s = s32[] parameter(0)
      lhs_f = f32[] parameter(1)
      rhs_s = s32[] parameter(2)
      rhs_f = f32[] parameter(3)
      ROOT out = (s32[], f32[]) tuple(s32[] add(lhs_s, rhs_s),
                                      f32[] add(lhs_f, rhs_f))
    }

    ENTRY main {
      s_init = s32[65536] parameter(0)
      f_init = f32[65536] parameter(1)
      single_init = f32[65536] parameter(2)
      idx = s32[65535,1] parameter(3)
      s_upd = s32[65535] parameter(4)
      f_upd = f32[65535] parameter(5)
      single_out = f32[65536] scatter(single_init, idx, f_upd),
          update_window_dims={},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          unique_indices=true,
          to_apply=add_f32
      var_out = (s32[65536], f32[65536])
          scatter(s_init, f_init, idx, s_upd, f_upd),
          update_window_dims={},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          unique_indices=true,
          to_apply=add_pair
      var_s = s32[65536] get-tuple-element(var_out), index=0
      var_f = f32[65536] get-tuple-element(var_out), index=1
      ROOT out = (s32[65536], f32[65536], f32[65536])
          tuple(var_s, var_f, single_out)
    }
  )hlo";

  Literal s_init(ShapeUtil::MakeShape(S32, {kNumElems}));
  Literal f_init(ShapeUtil::MakeShape(F32, {kNumElems}));
  Literal single_init(ShapeUtil::MakeShape(F32, {kNumElems}));
  absl::Span<int32_t> s_init_data = s_init.data<int32_t>();
  absl::Span<float> f_init_data = f_init.data<float>();
  absl::Span<float> single_init_data = single_init.data<float>();
  std::vector<int32_t> expected_s(kNumElems);
  std::vector<float> expected_f(kNumElems);
  std::vector<float> expected_single(kNumElems);
  for (int64_t i = 0; i < kNumElems; ++i) {
    s_init_data[i] = static_cast<int32_t>(1000 - i);
    f_init_data[i] = static_cast<float>((i % 251) - 125) * 0.5f;
    single_init_data[i] = static_cast<float>((i % 199) - 99) * 0.25f;
    expected_s[i] = s_init_data[i];
    expected_f[i] = f_init_data[i];
    expected_single[i] = single_init_data[i];
  }

  Literal idx(ShapeUtil::MakeShape(S32, {kNumUpdates, 1}));
  Literal s_upd(ShapeUtil::MakeShape(S32, {kNumUpdates}));
  Literal f_upd(ShapeUtil::MakeShape(F32, {kNumUpdates}));
  absl::Span<int32_t> idx_data = idx.data<int32_t>();
  absl::Span<int32_t> s_upd_data = s_upd.data<int32_t>();
  absl::Span<float> f_upd_data = f_upd.data<float>();
  // Odd multiplier is coprime to 65536, producing 65535 distinct indices and
  // leaving exactly one element untouched.
  for (int64_t k = 0; k < kNumUpdates; ++k) {
    const int32_t dst =
        static_cast<int32_t>((k * 39989 + 12345) & (kNumElems - 1));
    idx_data[k] = dst;
    s_upd_data[k] = static_cast<int32_t>(k * 5 + 7);
    f_upd_data[k] = static_cast<float>((k % 1021) - 510) * 0.125f;
    expected_s[dst] += s_upd_data[k];
    expected_f[dst] += f_upd_data[k];
    expected_single[dst] += f_upd_data[k];
  }

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo, config));
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module),
                                               {&s_init, &f_init, &single_init,
                                                &idx, &s_upd, &f_upd}));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 3);
  absl::Span<const int32_t> out_s = outputs[0].data<int32_t>();
  absl::Span<const float> out_f = outputs[1].data<float>();
  absl::Span<const float> out_single = outputs[2].data<float>();
  for (int64_t i = 0; i < kNumElems; ++i) {
    ASSERT_EQ(out_s[i], expected_s[i]) << "variadic s32 index " << i;
    ASSERT_EQ(out_f[i], expected_f[i]) << "variadic f32 index " << i;
    ASSERT_EQ(out_single[i], expected_single[i]) << "single f32 index " << i;
  }
}

TEST_F(ScatterTest, LargeParallelWindowSliceAddAndOverwrite) {
  constexpr int64_t kRows = 4096;
  constexpr int64_t kCols = 64;
  constexpr int64_t kNumUpdates = 16384;

  constexpr absl::string_view kHlo = R"hlo(
    HloModule LargeParallelWindowSliceAddAndOverwrite

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    overwrite_f32 {
      lhs = f32[] parameter(0)
      ROOT rhs = f32[] parameter(1)
    }

    ENTRY main {
      add_init = f32[4096,64] parameter(0)
      ow_init = f32[4096,64] parameter(1)
      idx = s32[16384,1] parameter(2)
      add_upd = f32[16384,1,64] parameter(3)
      ow_upd = f32[16384,1,64] parameter(4)
      added = f32[4096,64] scatter(add_init, idx, add_upd),
          update_window_dims={1,2},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=add_f32
      overwritten = f32[4096,64] scatter(ow_init, idx, ow_upd),
          update_window_dims={1,2},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          to_apply=overwrite_f32
      ROOT out = (f32[4096,64], f32[4096,64]) tuple(added, overwritten)
    }
  )hlo";

  Literal add_init(ShapeUtil::MakeShape(F32, {kRows, kCols}));
  Literal ow_init(ShapeUtil::MakeShape(F32, {kRows, kCols}));
  absl::Span<float> add_init_data = add_init.data<float>();
  absl::Span<float> ow_init_data = ow_init.data<float>();
  std::vector<double> expected_add(kRows * kCols);
  for (int64_t r = 0; r < kRows; ++r) {
    for (int64_t c = 0; c < kCols; ++c) {
      const int64_t offset = r * kCols + c;
      add_init_data[offset] = static_cast<float>((r % 31) - 15) * 0.5f +
                              static_cast<float>(c) * 0.25f;
      // Negative sentinel distinguishes untouched rows from positive update
      // encodings.
      ow_init_data[offset] = -static_cast<float>(offset + 1);
      expected_add[offset] = static_cast<double>(add_init_data[offset]);
    }
  }

  Literal idx(ShapeUtil::MakeShape(S32, {kNumUpdates, 1}));
  Literal add_upd(ShapeUtil::MakeShape(F32, {kNumUpdates, 1, kCols}));
  Literal ow_upd(ShapeUtil::MakeShape(F32, {kNumUpdates, 1, kCols}));
  absl::Span<int32_t> idx_data = idx.data<int32_t>();
  absl::Span<float> add_upd_data = add_upd.data<float>();
  absl::Span<float> ow_upd_data = ow_upd.data<float>();
  std::vector<int32_t> row_hits(kRows, 0);
  std::vector<int64_t> last_hit_update(kRows, -1);

  for (int64_t k = 0; k < kNumUpdates; ++k) {
    int32_t row = 0;
    if (k % 29 == 0) {
      row = -static_cast<int32_t>((k % 128) + 1);
    } else if (k % 31 == 0) {
      row = static_cast<int32_t>(kRows + (k % 256));
    } else {
      // Leave rows divisible by 16 untouched to verify preserved rows alongside
      // unique and colliding rows.
      const int32_t candidate =
          static_cast<int32_t>((k * 2713 + 19) & (kRows - 1));
      row = (candidate % 16 == 0) ? (candidate + 1) : candidate;
    }
    idx_data[k] = row;

    for (int64_t c = 0; c < kCols; ++c) {
      const int64_t upd_offset = k * kCols + c;
      const float delta =
          static_cast<float>(((k + c) % 13) - 6) * 0.25f + 0.125f;
      add_upd_data[upd_offset] = delta;
      // k * 64 + c + 1 <= 1048576 fits in exact f32 integer precision and
      // uniquely identifies the writing (k, c) pair.
      ow_upd_data[upd_offset] = static_cast<float>(k * kCols + c + 1);
      if (row >= 0 && row < kRows) {
        expected_add[row * kCols + c] += static_cast<double>(delta);
      }
    }
    if (row >= 0 && row < kRows) {
      ++row_hits[row];
      last_hit_update[row] = k;
    }
  }

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo, config));
  ASSERT_OK_AND_ASSIGN(Literal result,
                       Execute(std::move(module),
                               {&add_init, &ow_init, &idx, &add_upd, &ow_upd}));
  std::vector<Literal> outputs = result.DecomposeTuple();
  ASSERT_EQ(outputs.size(), 2);
  absl::Span<const float> added_data = outputs[0].data<float>();
  absl::Span<const float> ow_data = outputs[1].data<float>();
  ASSERT_EQ(added_data.size(), kRows * kCols);
  ASSERT_EQ(ow_data.size(), kRows * kCols);

  for (int64_t r = 0; r < kRows; ++r) {
    for (int64_t c = 0; c < kCols; ++c) {
      const int64_t offset = r * kCols + c;
      ASSERT_EQ(added_data[offset], static_cast<float>(expected_add[offset]))
          << "add row " << r << " col " << c;
      const float actual_ow = ow_data[offset];
      if (row_hits[r] == 0) {
        ASSERT_EQ(actual_ow, ow_init_data[offset])
            << "untouched row " << r << " col " << c;
      } else if (row_hits[r] == 1) {
        const int64_t k = last_hit_update[r];
        ASSERT_EQ(actual_ow, ow_upd_data[k * kCols + c])
            << "single hit row " << r << " col " << c;
      } else {
        ASSERT_GT(actual_ow, 0.0f) << "colliding row " << r << " col " << c;
        const int64_t encoded = static_cast<int64_t>(actual_ow) - 1;
        ASSERT_EQ(actual_ow, static_cast<float>(encoded + 1))
            << "colliding row " << r << " col " << c;
        ASSERT_EQ(encoded % kCols, c) << "colliding row " << r << " col " << c;
        const int64_t writer = encoded / kCols;
        ASSERT_GE(writer, 0);
        ASSERT_LT(writer, kNumUpdates);
        ASSERT_EQ(idx_data[writer], static_cast<int32_t>(r))
            << "colliding row " << r << " col " << c;
      }
    }
  }
}

}  // namespace
}  // namespace xla
