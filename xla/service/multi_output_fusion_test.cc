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

#include "xla/service/multi_output_fusion.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/test_benchmark.h"

namespace xla {
namespace {

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

// Fuses elementwise siblings.
class ElementwiseSiblingFusion : public MultiOutputFusion {
 public:
  explicit ElementwiseSiblingFusion(const AliasInfo* alias_info)
      : MultiOutputFusion(alias_info) {}

  absl::string_view name() const override {
    return "elementwise_sibling_fusion";
  }

 protected:
  bool ShapesCompatibleForFusion(HloInstruction* instr1,
                                 HloInstruction* instr2) override {
    return ShapeUtil::Equal(instr1->shape(), instr2->shape());
  }

  bool IsFusible(HloInstruction* instr) override {
    return instr->IsElementwise();
  }

  int64_t GetProfit(HloInstruction* instr1, HloInstruction* instr2) override {
    return 1;
  }

  bool LegalToFuse(HloInstruction* instr1, HloInstruction* instr2) override {
    return LegalToFuseMainConstraints(instr1, instr2);
  }
};

// Counts how often the sibling worklist build evaluates its predicates.
class CountingMultiOutputFusion : public ElementwiseSiblingFusion {
 public:
  using ElementwiseSiblingFusion::ElementwiseSiblingFusion;

  // Per instruction IsFusible calls, and per ordered pair LegalToFuse calls,
  // made while the worklist of the last computation was built. Keyed by name
  // because the fusions remove instructions afterwards.
  const absl::flat_hash_map<std::string, int>& is_fusible_calls() const {
    return is_fusible_calls_;
  }
  const absl::flat_hash_map<std::pair<std::string, std::string>, int>&
  legal_to_fuse_calls() const {
    return legal_to_fuse_calls_;
  }

 protected:
  bool IsFusible(HloInstruction* instr) override {
    if (building_worklist_) {
      ++is_fusible_calls_[instr->name()];
    }
    return ElementwiseSiblingFusion::IsFusible(instr);
  }

  bool LegalToFuse(HloInstruction* instr1, HloInstruction* instr2) override {
    if (building_worklist_) {
      ++legal_to_fuse_calls_[{std::string(instr1->name()),
                              std::string(instr2->name())}];
    }
    return ElementwiseSiblingFusion::LegalToFuse(instr1, instr2);
  }

  void CreateFusionWorkListForCurrentComputation() override {
    is_fusible_calls_.clear();
    legal_to_fuse_calls_.clear();
    building_worklist_ = true;
    MultiOutputFusion::CreateFusionWorkListForCurrentComputation();
    building_worklist_ = false;
  }

 private:
  bool building_worklist_ = false;
  absl::flat_hash_map<std::string, int> is_fusible_calls_;
  absl::flat_hash_map<std::pair<std::string, std::string>, int>
      legal_to_fuse_calls_;
};

using MultiOutputFusionTest = HloHardwareIndependentTestBase;

TEST_F(MultiOutputFusionTest, WorklistBuildEvaluatesEachFactOnce) {
  // Every sibling pair shares both profitable operands, so every sibling is
  // visited twice per instruction.
  constexpr absl::string_view kHlo = R"(
HloModule m

ENTRY e {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  a = f32[8] add(p0, p1)
  b = f32[8] multiply(p0, p1)
  c = f32[8] subtract(p0, p1)
  ROOT t = (f32[8], f32[8], f32[8]) tuple(a, b, c)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  AliasInfo alias_info;
  CountingMultiOutputFusion fusion(&alias_info);
  ASSERT_OK_AND_ASSIGN(bool changed, fusion.Run(module.get()));
  EXPECT_TRUE(changed);

  EXPECT_THAT(fusion.is_fusible_calls(),
              UnorderedElementsAre(Pair("p0", 1), Pair("p1", 1), Pair("a", 1),
                                   Pair("b", 1), Pair("c", 1), Pair("t", 1)));
  EXPECT_THAT(
      fusion.legal_to_fuse_calls(),
      UnorderedElementsAre(Pair(Pair("a", "b"), 1), Pair(Pair("a", "c"), 1),
                           Pair(Pair("b", "a"), 1), Pair(Pair("b", "c"), 1),
                           Pair(Pair("c", "a"), 1), Pair(Pair("c", "b"), 1)));
}

// num_fusions loop fusions that each add up the same num_params parameters, so
// every pair of them shares num_params profitable operands.
std::string SiblingsSharingOperandsHlo(int num_params, int num_fusions) {
  constexpr absl::string_view kModule = R"(HloModule siblings_sharing_operands

{{COMPUTATIONS}}

ENTRY main {
{{PARAMETERS}}
{{FUSIONS}}
  ROOT t = ({{FUSION_SHAPES}}) tuple({{FUSION_NAMES}})
}
)";
  constexpr absl::string_view kComputation = R"(sum.{{INDEX}} {
{{PARAMETERS}}
{{ADDS}}
})";
  constexpr absl::string_view kShape = "f32[1024,1024]";

  std::vector<std::string> parameters, parameter_names;
  for (int p = 0; p < num_params; ++p) {
    parameters.push_back(
        absl::Substitute("  p$0 = $1 parameter($0)", p, kShape));
    parameter_names.push_back(absl::StrCat("p", p));
  }
  // add.1 = p0 + p1, add.2 = add.1 + p2, and so on; the last add is the root.
  std::vector<std::string> adds;
  std::string sum = "p0";
  for (int p = 1; p < num_params; ++p) {
    absl::string_view root = p + 1 == num_params ? "ROOT " : "";
    adds.push_back(
        absl::Substitute("  $0add.$1 = $2 add($3, p$1)", root, p, kShape, sum));
    sum = absl::StrCat("add.", p);
  }
  const std::string parameter_lines = absl::StrJoin(parameters, "\n");
  const std::string add_lines = absl::StrJoin(adds, "\n");
  const std::string operands = absl::StrJoin(parameter_names, ", ");

  // Fusion i calls sum.i.
  std::vector<std::string> computations, fusions, fusion_names;
  for (int i = 0; i < num_fusions; ++i) {
    computations.push_back(
        absl::StrReplaceAll(kComputation, {{"{{INDEX}}", absl::StrCat(i)},
                                           {"{{PARAMETERS}}", parameter_lines},
                                           {"{{ADDS}}", add_lines}}));
    fusions.push_back(absl::Substitute(
        "  fusion.$0 = $1 fusion($2), kind=kLoop, calls=sum.$0", i, kShape,
        operands));
    fusion_names.push_back(absl::StrCat("fusion.", i));
  }
  const std::vector<absl::string_view> fusion_shapes(num_fusions, kShape);
  return absl::StrReplaceAll(
      kModule, {{"{{COMPUTATIONS}}", absl::StrJoin(computations, "\n\n")},
                {"{{PARAMETERS}}", parameter_lines},
                {"{{FUSIONS}}", absl::StrJoin(fusions, "\n")},
                {"{{FUSION_SHAPES}}", absl::StrJoin(fusion_shapes, ", ")},
                {"{{FUSION_NAMES}}", absl::StrJoin(fusion_names, ", ")}});
}

// The siblings share all 8 parameters, so the worklist build meets every
// sibling once per shared operand.
void BM_SiblingsSharingOperands(::testing::benchmark::State& state) {
  absl::StatusOr<std::unique_ptr<HloModule>> module =
      ParseAndReturnUnverifiedModule(
          SiblingsSharingOperandsHlo(/*num_params=*/8, state.range(0)));
  CHECK_OK(module.status());
  AliasInfo alias_info;
  std::unique_ptr<HloModule> clone;
  for (auto s : state) {
    state.PauseTiming();
    clone = (*module)->Clone();
    ElementwiseSiblingFusion fusion(&alias_info);
    state.ResumeTiming();
    CHECK_OK(fusion.Run(clone.get()).status());
  }
}
BENCHMARK(BM_SiblingsSharingOperands)->Arg(128);

}  // namespace
}  // namespace xla
