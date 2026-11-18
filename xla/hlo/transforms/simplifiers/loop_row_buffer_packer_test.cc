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

#include "xla/hlo/transforms/simplifiers/loop_row_buffer_packer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal.h"
#include "xla/service/collective_pipeliner.h"
#include "xla/service/collective_pipeliner_utils.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/shape_util.h"
#include "xla/tests/test_utils.h"
#include "xla/util.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Matcher;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

// One loop carried buffer of TestLoop: `type`[`rows`], with `columns`
// `type`[`rows`, `columns`], with `depth` too `type`[`rows`, `columns`,
// `depth`].
struct Buffer {
  absl::string_view type = "f32";
  int64_t rows = 8;
  int64_t columns = 0;
  int64_t depth = 0;
  bool dynamic = false;  // `rows` is only the bound of the row count.
};

std::string ShapeOf(const Buffer& b, absl::string_view rows) {
  std::string shape = absl::StrCat(b.type, "[", rows);
  for (int64_t size : {b.columns, b.depth}) {
    if (size > 0) {
      absl::StrAppend(&shape, ",", size);
    }
  }
  return absl::StrCat(shape, "]");
}

std::string ShapeOf(const Buffer& b) {
  return ShapeOf(b, absl::StrCat(b.dynamic ? "<=" : "", b.rows));
}

// The state of TestLoop: the counter, `x` and the buffers.
std::string State(absl::Span<const Buffer> buffers) {
  std::string state = "(s32[], s32[8,160]";
  for (const Buffer& b : buffers) {
    absl::StrAppend(&state, ", ", ShapeOf(b));
  }
  return absl::StrCat(state, ")");
}

// A loop over i = 2..5 that writes row i of every buffer. Buffer k takes its
// row from columns [k, k + width) of row i of the loop invariant `x`. Every
// buffer starts from its own entry parameter, so the result shows the initial
// value in the rows the loop leaves alone.
std::string TestLoop(absl::Span<const Buffer> buffers) {
  std::vector<std::string> shapes, inits, updates, outputs;
  std::string body_reads, body_writes, entry_inits, entry_outputs;
  for (int64_t k = 0; k < buffers.size(); ++k) {
    const Buffer& b = buffers[k];
    const int64_t width =
        std::max<int64_t>(b.columns, 1) * std::max<int64_t>(b.depth, 1);
    shapes.push_back(ShapeOf(b));
    inits.push_back(absl::StrCat("init", k));
    updates.push_back(absl::StrCat("dus", k));
    outputs.push_back(absl::StrCat("out", k));
    absl::StrAppend(&body_reads, "  buf", k, " = ", shapes.back(),
                    " get-tuple-element(p), index=", k + 2, "\n");
    absl::StrAppend(&body_writes, "  val", k, " = s32[1,", width,
                    "] slice(row), slice={[0:1], [", k, ":", k + width, "]}\n");
    absl::StrAppend(&body_writes, "  cvt", k, " = ", b.type, "[1,", width,
                    "] convert(val", k, ")\n");
    absl::StrAppend(&body_writes, "  upd", k, " = ", ShapeOf(b, "1"),
                    " reshape(cvt", k, ")\n");
    absl::StrAppend(&body_writes, "  dus", k, " = ", shapes.back(),
                    " dynamic-update-slice(buf", k, ", upd", k, ", i",
                    b.columns > 0 ? ", zero" : "", b.depth > 0 ? ", zero" : "",
                    ")\n");
    absl::StrAppend(&entry_inits, "  init", k, " = ", shapes.back(),
                    " parameter(", k + 1, ")\n");
    absl::StrAppend(&entry_outputs, "  out", k, " = ", shapes.back(),
                    " get-tuple-element(w), index=", k + 2, "\n");
  }
  const std::string state = State(buffers);
  return absl::StrCat(R"(HloModule m

cond {
  p = )",
                      state, R"( parameter(0)
  i = s32[] get-tuple-element(p), index=0
  n = s32[] constant(6)
  ROOT lt = pred[] compare(i, n), direction=LT
}

body {
  p = )",
                      state, R"( parameter(0)
  i = s32[] get-tuple-element(p), index=0
  x = s32[8,160] get-tuple-element(p), index=1
  one = s32[] constant(1)
  next = s32[] add(i, one)
  zero = s32[] constant(0)
  row = s32[1,160] dynamic-slice(x, i, zero), dynamic_slice_sizes={1,160}
)",
                      body_reads, body_writes, "  ROOT t = ", state,
                      " tuple(next, x, ", absl::StrJoin(updates, ", "), R"()
}

ENTRY e {
  x = s32[8,160] parameter(0)
  two = s32[] constant(2)
)",
                      entry_inits, "  t = ", state, " tuple(two, x, ",
                      absl::StrJoin(inits, ", "), ")\n  w = ", state,
                      " while(t), condition=cond, body=body\n", entry_outputs,
                      "  ROOT r = (", absl::StrJoin(shapes, ", "), ") tuple(",
                      absl::StrJoin(outputs, ", "), ")\n}\n");
}

// `hlo` with its one occurrence of `from` replaced by `to`.
std::string Replace(std::string hlo, absl::string_view from,
                    absl::string_view to) {
  CHECK_EQ(absl::StrReplaceAll({{from, to}}, &hlo), 1) << from;
  return hlo;
}

class LoopRowBufferPackerTest : public HloHardwareIndependentTestBase {
 protected:
  // Runs the packer on `hlo`, then the cleanup the compiler runs after it.
  // Checks that the outputs keep their bits and that the shapes of the updates
  // left in the loop body match `updates`.
  void Pack(absl::string_view hlo,
            const Matcher<std::vector<std::string>>& updates) {
    ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo));
    Pack(updates, /*evaluate=*/true);
  }

  // The same without the outputs, for an `hlo` the evaluator does not run.
  void PackUnevaluated(absl::string_view hlo,
                       const Matcher<std::vector<std::string>>& updates) {
    ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo));
    Pack(updates, /*evaluate=*/false);
  }

  // The same for `module_`.
  void Pack(const Matcher<std::vector<std::string>>& updates, bool evaluate) {
    std::vector<Literal> args;
    Literal before;
    if (evaluate) {
      ASSERT_OK_AND_ASSIGN(args, MakeFakeArguments(module_.get()));
      ASSERT_OK_AND_ASSIGN(before, HloEvaluator().Evaluate(*module_, args));
    }
    HloPassPipeline pipeline("packer");
    pipeline.AddInvariantChecker<HloVerifier>(/*layout_sensitive=*/false,
                                              /*allow_mixed_precision=*/false);
    pipeline.AddPass<LoopRowBufferPacker>();
    HloPassFix<HloPassPipeline>& cleanup =
        pipeline.AddPass<HloPassFix<HloPassPipeline>>("cleanup");
    cleanup.AddPass<TupleSimplifier>();
    cleanup.AddPass<HloDCE>();
    cleanup.AddPass<WhileLoopSimplifier>();
    ASSERT_OK(pipeline.Run(module_.get()).status());
    EXPECT_THAT(Shapes(Body(), HloOpcode::kDynamicUpdateSlice), updates);
    if (evaluate) {
      ASSERT_OK_AND_ASSIGN(Literal after,
                           HloEvaluator().Evaluate(*module_, args));
      EXPECT_EQ(before, after);
    }
  }

  HloComputation* Entry() { return module_->entry_computation(); }

  HloComputation* Body() {
    return hlo_query::FindInstruction(Entry(), HloOpcode::kWhile)->while_body();
  }

  static std::vector<std::string> Shapes(const HloComputation* computation,
                                         HloOpcode opcode) {
    std::vector<std::string> shapes;
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == opcode) {
        shapes.push_back(ShapeUtil::HumanString(instruction->shape()));
      }
    }
    return shapes;
  }

  std::unique_ptr<HloModule> module_;
};

TEST_F(LoopRowBufferPackerTest, PacksPerElementTypeAndRowCount) {
  std::vector<Buffer> buffers(8);
  buffers.push_back({"f32", 8, 3});
  buffers.push_back({"f32", 8, 128});
  buffers.insert(buffers.end(), 8, {"s32"});
  // These clamp the row index: to 3 in the last two iterations, and to 0.
  buffers.insert(buffers.end(), 8, {"f32", 4});
  buffers.insert(buffers.end(), 8, {"f32", 1});
  // Buffer 26, of one row, writes that row with itself.
  Pack(Replace(TestLoop(buffers), "(buf26, upd26, i)", "(buf26, buf26, i)"),
       UnorderedElementsAre("f32[8,139]", "s32[8,8]", "f32[4,8]", "f32[1,8]"));
}

TEST_F(LoopRowBufferPackerTest, KeepsSmallGroups) {
  Pack(TestLoop(std::vector<Buffer>(7)), AllOf(SizeIs(7), Each("f32[8]")));
}

TEST_F(LoopRowBufferPackerTest, ReturnsWhetherItPacked) {
  for (int buffers : {7, 8}) {
    ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(
                                      TestLoop(std::vector<Buffer>(buffers))));
    // No computation runs on this thread.
    EXPECT_THAT(LoopRowBufferPacker().Run(module_.get(), {"other"}),
                IsOkAndHolds(false));
    EXPECT_THAT(LoopRowBufferPacker().Run(module_.get()),
                IsOkAndHolds(buffers == 8));
  }
}

TEST_F(LoopRowBufferPackerTest, KeepsWideRows) {
  Pack(TestLoop(std::vector<Buffer>(8, {"f32", 8, 129})),
       AllOf(SizeIs(8), Each("f32[8,129]")));
}

TEST_F(LoopRowBufferPackerTest, KeepsBuffersOfOtherRanks) {
  Pack(TestLoop(std::vector<Buffer>(8, {"f32", 8, 2, 2})),
       AllOf(SizeIs(8), Each("f32[8,2,2]")));
  // The counter takes its next value through an update of rank 0, which the
  // evaluator does not support.
  PackUnevaluated(
      Replace(TestLoop(std::vector<Buffer>(8)), "  next = s32[] add(i, one)\n",
              "  sum = s32[] add(i, one)\n"
              "  next = s32[] dynamic-update-slice(i, sum)\n"),
      UnorderedElementsAre("f32[8,8]", "s32[]"));
}

TEST_F(LoopRowBufferPackerTest, KeepsBuffersOfDynamicSize) {
  // Literals of dynamic size do not compare reliably, so this loop is not
  // evaluated.
  std::vector<Buffer> buffers(9);
  buffers[3].dynamic = true;
  PackUnevaluated(TestLoop(buffers),
                  UnorderedElementsAre("f32[8,8]", "f32[<=8]"));
  // Buffer 3 takes a row of dynamic size.
  Pack(Replace(TestLoop(std::vector<Buffer>(9)),
               "  upd3 = f32[1] reshape(cvt3)\n",
               "  row3 = f32[1] reshape(cvt3)\n"
               "  upd3 = f32[<=1] set-dimension-size(row3, one), "
               "dimensions={0}\n"),
       UnorderedElementsAre("f32[8,8]", "f32[8]"));
}

TEST_F(LoopRowBufferPackerTest, GroupsByRowIndex) {
  std::string hlo = TestLoop(std::vector<Buffer>(16));
  // The second group writes row 7 - i, at an index of another type.
  hlo = Replace(hlo, "  zero = s32[] constant(0)\n",
                "  zero = s32[] constant(0)\n"
                "  seven = s64[] constant(7)\n"
                "  wide = s64[] convert(i)\n"
                "  j = s64[] subtract(seven, wide)\n");
  for (int k = 8; k < 16; ++k) {
    hlo = Replace(hlo, absl::StrCat("upd", k, ", i)"),
                  absl::StrCat("upd", k, ", j)"));
  }
  Pack(hlo, UnorderedElementsAre("f32[8,8]", "f32[8,8]"));
}

TEST_F(LoopRowBufferPackerTest, PacksBuffersWithUnusedReads) {
  // The body and the condition read buffer 3 once more, for no user.
  std::string hlo = TestLoop(std::vector<Buffer>(8));
  for (absl::string_view line :
       {"  zero = s32[] constant(0)\n", "  n = s32[] constant(6)\n"}) {
    hlo = Replace(
        hlo, line,
        absl::StrCat(line,
                     "  unused3 = f32[8] get-tuple-element(p), index=5\n"));
  }
  Pack(hlo, ElementsAre("f32[8,8]"));
}

TEST_F(LoopRowBufferPackerTest, KeepsBuffersTheBodyReads) {
  // Buffer 3 accumulates the row an earlier iteration wrote.
  Pack(Replace(TestLoop(std::vector<Buffer>(9)),
               "  dus3 = f32[8] dynamic-update-slice(buf3, upd3, i)\n",
               "  old3 = f32[1] slice(buf3), slice={[2:3]}\n"
               "  sum3 = f32[1] add(upd3, old3)\n"
               "  dus3 = f32[8] dynamic-update-slice(buf3, sum3, i)\n"),
       UnorderedElementsAre("f32[8,8]", "f32[8]"));
}

TEST_F(LoopRowBufferPackerTest, KeepsBuffersTheConditionReads) {
  // The loop ends once row 2 of buffer 3 holds the value the body writes.
  Pack(Replace(TestLoop(std::vector<Buffer>(9)),
               "  ROOT lt = pred[] compare(i, n), direction=LT\n",
               "  lt = pred[] compare(i, n), direction=LT\n"
               "  x = s32[8,160] get-tuple-element(p), index=1\n"
               "  val = s32[1,1] slice(x), slice={[2:3], [3:4]}\n"
               "  cvt = f32[1,1] convert(val)\n"
               "  want = f32[] reshape(cvt)\n"
               "  buf3 = f32[8] get-tuple-element(p), index=5\n"
               "  row = f32[1] slice(buf3), slice={[2:3]}\n"
               "  have = f32[] reshape(row)\n"
               "  pending = pred[] compare(have, want), direction=NE\n"
               "  ROOT go = pred[] and(lt, pending)\n"),
       UnorderedElementsAre("f32[8,8]", "f32[8]"));
}

TEST_F(LoopRowBufferPackerTest, KeepsBuffersWhoseUpdateHasOtherUsers) {
  // Buffer 4 takes its row from the updated buffer 3.
  Pack(Replace(TestLoop(std::vector<Buffer>(9)),
               "  upd4 = f32[1] reshape(cvt4)\n",
               "  upd4 = f32[1] slice(dus3), slice={[2:3]}\n"),
       UnorderedElementsAre("f32[8,8]", "f32[8]"));
}

TEST_F(LoopRowBufferPackerTest, KeepsBuffersInTwoStateSlots) {
  // The slot of buffer 4 takes the updated buffer 3.
  Pack(Replace(TestLoop(std::vector<Buffer>(10)), "dus3, dus4, ",
               "dus3, dus3, "),
       UnorderedElementsAre("f32[8,8]", "f32[8]"));
}

TEST_F(LoopRowBufferPackerTest, KeepsBuffersCopiedToAnotherSlot) {
  // The slot of buffer 4 takes buffer 3 as the iteration found it.
  Pack(Replace(TestLoop(std::vector<Buffer>(10)), "dus3, dus4, ",
               "dus3, buf3, "),
       UnorderedElementsAre("f32[8,8]", "f32[8]"));
}

TEST_F(LoopRowBufferPackerTest, KeepsBuffersWrittenIntoAnotherSlot) {
  // Buffers 3 and 4 trade places in every iteration.
  std::string hlo = TestLoop(std::vector<Buffer>(10));
  hlo = Replace(hlo, "(buf3, upd3, i)", "(buf4, upd3, i)");
  hlo = Replace(hlo, "(buf4, upd4, i)", "(buf3, upd4, i)");
  Pack(hlo, UnorderedElementsAre("f32[8,8]", "f32[8]", "f32[8]"));
  // Buffer 3 takes buffer 4 with its row, read as element 5 of a tuple that is
  // not the loop state.
  Pack(Replace(TestLoop(std::vector<Buffer>(10)),
               "  dus3 = f32[8] dynamic-update-slice(buf3, upd3, i)\n",
               "  q = (f32[8], s32[], s32[], s32[], s32[], f32[8]) "
               "tuple(buf3, i, i, i, i, buf4)\n"
               "  alt4 = f32[8] get-tuple-element(q), index=5\n"
               "  dus3 = f32[8] dynamic-update-slice(alt4, upd3, i)\n"),
       UnorderedElementsAre("f32[8,8]", "f32[8]", "f32[8]"));
}

TEST_F(LoopRowBufferPackerTest, KeepsPartialRowWrites) {
  // Buffer 0 takes two of its three columns.
  std::vector<Buffer> buffers = {{"f32", 8, 3}};
  buffers.insert(buffers.end(), 8, {});
  Pack(Replace(TestLoop(buffers), "  upd0 = f32[1,3] reshape(cvt0)\n",
               "  upd0 = f32[1,2] slice(cvt0), slice={[0:1], [0:2]}\n"),
       UnorderedElementsAre("f32[8,8]", "f32[8,3]"));
  // Buffer 0 takes rows of another precision, which the verifier of Pack does
  // not allow.
  ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(Replace(
                   TestLoop(buffers), "  upd0 = f32[1,3] reshape(cvt0)\n",
                   "  upd0 = bf16[1,3] convert(val0)\n")));
  ASSERT_THAT(LoopRowBufferPacker().Run(module_.get()), IsOkAndHolds(true));
  EXPECT_THAT(Shapes(Body(), HloOpcode::kDynamicUpdateSlice),
              UnorderedElementsAre("f32[8,8]", "f32[8,3]"));
}

TEST_F(LoopRowBufferPackerTest, KeepsUpdatesWithVectorIndex) {
  std::string hlo =
      Replace(TestLoop(std::vector<Buffer>(8)), "  zero = s32[] constant(0)\n",
              "  zero = s32[] constant(0)\n"
              "  at = s32[1] reshape(i)\n");
  for (int k = 0; k < 8; ++k) {
    hlo = Replace(hlo, absl::StrCat("upd", k, ", i)"),
                  absl::StrCat("upd", k, ", at)"));
  }
  Pack(hlo, AllOf(SizeIs(8), Each("f32[8]")));
}

TEST_F(LoopRowBufferPackerTest, KeepsUpdatesWithAnnotationsOrOrdering) {
  Pack(Replace(TestLoop(std::vector<Buffer>(9)), "(buf3, upd3, i)\n",
               "(buf3, upd3, i), "
               "frontend_attributes={_scheduling_group_id=\"1\"}\n"),
       UnorderedElementsAre("f32[8,8]", "f32[8]"));
  // The write of buffer 4 waits for the write of buffer 3: both stay.
  Pack(Replace(TestLoop(std::vector<Buffer>(10)), "(buf4, upd4, i)\n",
               "(buf4, upd4, i), control-predecessors={dus3}\n"),
       UnorderedElementsAre("f32[8,8]", "f32[8]", "f32[8]"));
}

TEST_F(LoopRowBufferPackerTest, KeepsBuffersInHostMemory) {
  // Custom calls are not evaluated.
  const std::string hlo = TestLoop(std::vector<Buffer>(9));
  const std::pair<absl::string_view, absl::string_view> annotations[] = {
      // The row of buffer 3 moves to host memory before its write, which
      // takes it through a copy and a reshape, or through a bitcast.
      {"  upd3 = f32[1] reshape(cvt3)\n",
       "  host3 = f32[1,1] custom-call(cvt3), "
       "custom_call_target=\"MoveToHost\"\n"
       "  copy3 = f32[1,1] copy(host3)\n"
       "  upd3 = f32[1] reshape(copy3)\n"},
      {"  upd3 = f32[1] reshape(cvt3)\n",
       "  host3 = f32[1,1] custom-call(cvt3), "
       "custom_call_target=\"MoveToHost\"\n"
       "  upd3 = f32[1] bitcast(host3)\n"},
      // Buffer 3 starts in host memory.
      {"  init3 = f32[8] parameter(4)\n",
       "  device3 = f32[8] parameter(4)\n"
       "  init3 = f32[8] custom-call(device3), "
       "custom_call_target=\"MoveToHost\"\n"},
      // A row written to buffer 3 before the loop moves to host memory.
      {"  init3 = f32[8] parameter(4)\n",
       "  base3 = f32[8] parameter(4)\n"
       "  first3 = f32[1] slice(base3), slice={[0:1]}\n"
       "  host3 = f32[1] custom-call(first3), "
       "custom_call_target=\"MoveToHost\"\n"
       "  init3 = f32[8] dynamic-update-slice(base3, host3, two)\n"},
      // Buffer 3 starts in host memory, with two rows written before the
      // loop.
      {"  init3 = f32[8] parameter(4)\n",
       "  device3 = f32[8] parameter(4)\n"
       "  base3 = f32[8] custom-call(device3), "
       "custom_call_target=\"MoveToHost\"\n"
       "  first3 = f32[1] slice(device3), slice={[0:1]}\n"
       "  next3 = f32[8] dynamic-update-slice(base3, first3, two)\n"
       "  init3 = f32[8] dynamic-update-slice(next3, first3, two)\n"},
      // Buffer 3 starts in host memory, read through a tuple.
      {"  init3 = f32[8] parameter(4)\n",
       "  device3 = f32[8] parameter(4)\n"
       "  host3 = f32[8] custom-call(device3), "
       "custom_call_target=\"MoveToHost\"\n"
       "  pair3 = (f32[8], s32[]) tuple(host3, two)\n"
       "  init3 = f32[8] get-tuple-element(pair3), index=0\n"}};
  for (const auto& [from, to] : annotations) {
    SCOPED_TRACE(to);
    PackUnevaluated(Replace(hlo, from, to),
                    UnorderedElementsAre("f32[8,8]", "f32[8]"));
  }
}

TEST_F(LoopRowBufferPackerTest, SkipsLoopsWhoseNextStateHasOtherUsers) {
  // Every iteration sends the buffer 3 it wrote to the outfeed, which the
  // evaluator does not run.
  PackUnevaluated(
      Replace(TestLoop(std::vector<Buffer>(8)), ")\n}\n\nENTRY e {",
              ")\n  sent3 = f32[8] get-tuple-element(t), index=5\n"
              "  after = token[] after-all()\n"
              "  out = token[] outfeed(sent3, after), outfeed_shape=f32[8]\n"
              "}\n\nENTRY e {"),
      AllOf(SizeIs(8), Each("f32[8]")));
}

TEST_F(LoopRowBufferPackerTest, SkipsLoopsThatUseTheirStateWhole) {
  // Buffer 5 takes its row from buffer 4, read through a barrier on the state.
  const std::vector<Buffer> buffers(9);
  Pack(Replace(TestLoop(buffers), "  upd5 = f32[1] reshape(cvt5)\n",
               absl::StrCat("  q = ", State(buffers),
                            " opt-barrier(p)\n"
                            "  alt4 = f32[8] get-tuple-element(q), index=6\n"
                            "  upd5 = f32[1] slice(alt4), slice={[2:3]}\n")),
       AllOf(SizeIs(9), Each("f32[8]")));
  // The condition could read any buffer through its barrier.
  Pack(Replace(TestLoop(buffers), "  n = s32[] constant(6)\n",
               absl::StrCat("  n = s32[] constant(6)\n  q = ", State(buffers),
                            " opt-barrier(p)\n")),
       AllOf(SizeIs(9), Each("f32[8]")));
}

TEST_F(LoopRowBufferPackerTest, SkipsLoopsWhoseBodyReturnsNoTupleInstruction) {
  const std::vector<Buffer> buffers(8);
  std::string hlo = Replace(TestLoop(buffers), "  ROOT t = ", "  t = ");
  hlo = Replace(hlo, ")\n}\n\nENTRY e {",
                absl::StrCat(")\n  ROOT b = ", State(buffers),
                             " opt-barrier(t)\n}\n\nENTRY e {"));
  Pack(hlo, AllOf(SizeIs(8), Each("f32[8]")));
}

TEST_F(LoopRowBufferPackerTest, SkipsLoopsOffTheMainThread) {
  ASSERT_OK_AND_ASSIGN(
      module_, ParseAndReturnVerifiedModule(TestLoop(std::vector<Buffer>(8))));
  for (HloComputation* computation : module_->computations()) {
    computation->SetExecutionThread("other");
  }
  Pack(AllOf(SizeIs(8), Each("f32[8]")), /*evaluate=*/true);
}

TEST_F(LoopRowBufferPackerTest, SkipsLoopsWithComputeType) {
  Pack(Replace(TestLoop(std::vector<Buffer>(8)), "body=body\n",
               "body=body, frontend_attributes={_xla_compute_type=\"host\"}\n"),
       AllOf(SizeIs(8), Each("f32[8]")));
}

TEST_F(LoopRowBufferPackerTest, PacksLoopsWhoseStateIsNoTupleInstruction) {
  // The loop takes its initial state through an optimization barrier.
  const std::vector<Buffer> buffers(8);
  std::string hlo = Replace(TestLoop(buffers), "  t = ", "  s = ");
  hlo = Replace(
      hlo, "  w = ",
      absl::StrCat("  t = ", State(buffers), " opt-barrier(s)\n  w = "));
  Pack(hlo, ElementsAre("f32[8,8]"));
}

// The form loop peeling leaves: every buffer starts from one fill value, read
// through a tuple, with one row written before the loop.
TEST_F(LoopRowBufferPackerTest, PacksRowsWrittenBeforeTheLoop) {
  // The first four buffers start from `base` with row 7 written, the others
  // from `other_base` with row `other_index` written.
  auto peeled = [](absl::string_view base, absl::string_view other_base,
                   absl::string_view other_index) {
    std::string hlo =
        Replace(TestLoop(std::vector<Buffer>(8)), "  two = s32[] constant(2)\n",
                "  two = s32[] constant(2)\n"
                "  six = s32[] constant(6)\n"
                "  seven = s32[] constant(7)\n"
                "  nine = f32[] constant(9)\n"
                "  fill = f32[8] broadcast(nine), dimensions={}\n"
                "  seed = (f32[8], s32[]) tuple(fill, two)\n"
                "  base = f32[8] get-tuple-element(seed), index=0\n"
                "  three = f32[] constant(3)\n"
                "  other = f32[8] broadcast(three), dimensions={}\n"
                "  spread = f32[8] broadcast(fill), dimensions={0}\n");
    for (int k = 0; k < 8; ++k) {
      hlo = Replace(
          hlo, absl::StrCat("  init", k, " = f32[8] parameter(", k + 1, ")\n"),
          absl::StrCat("  first", k, " = f32[1] parameter(", k + 1, ")\n  init",
                       k, " = f32[8] dynamic-update-slice(",
                       k < 4 ? base : other_base, ", first", k, ", ",
                       k < 4 ? "seven" : other_index, ")\n"));
    }
    return hlo;
  };
  Pack(peeled("base", "base", "seven"), ElementsAre("f32[8,8]"));
  EXPECT_THAT(Shapes(Entry(), HloOpcode::kDynamicUpdateSlice),
              ElementsAre("f32[8,8]"));
  EXPECT_THAT(Shapes(Entry(), HloOpcode::kConcatenate),
              ElementsAre("f32[1,8]"));
  // Rows written at two indices stay separate writes.
  Pack(peeled("base", "base", "six"), ElementsAre("f32[8,8]"));
  EXPECT_THAT(Shapes(Entry(), HloOpcode::kDynamicUpdateSlice),
              AllOf(SizeIs(8), Each("f32[8]")));
  // Two fill values, and a fill that is no scalar.
  Pack(peeled("base", "other", "seven"), ElementsAre("f32[8,8]"));
  Pack(peeled("spread", "spread", "seven"), ElementsAre("f32[8,8]"));
  // A write of two rows is no row write.
  Pack(Replace(peeled("base", "base", "seven"), "first0 = f32[1]",
               "first0 = f32[2]"),
       ElementsAre("f32[8,8]"));
}

// Packing a loop clones its body and, with it, the loop inside, so the pass
// packs that loop first.
TEST_F(LoopRowBufferPackerTest, PacksLoopsInsidePackedLoops) {
  constexpr int kBuffers = 8;
  const std::string state = absl::StrCat(
      "(s32[]",
      absl::StrJoin(std::vector<std::string>(kBuffers, ", f32[8]"), ""), ")");
  // Both loops count to 8 and write row i of their buffers.
  const std::string head = absl::StrCat(
      " {\n  p = ", state,
      " parameter(0)\n  i = s32[] get-tuple-element(p), index=0\n");
  const std::string cond =
      absl::StrCat(head,
                   "  n = s32[] constant(8)\n"
                   "  ROOT lt = pred[] compare(i, n), direction=LT\n}\n\n");
  const std::string step = absl::StrCat(head,
                                        "  one = s32[] constant(1)\n"
                                        "  next = s32[] add(i, one)\n"
                                        "  val = f32[] convert(i)\n");
  std::string inner_writes, outer_writes, parameters;
  std::vector<std::string> updates, inits;
  for (int k = 0; k < kBuffers; ++k) {
    const std::string read = absl::StrCat(
        "  buf", k, " = f32[8] get-tuple-element(p), index=", k + 1, "\n");
    absl::StrAppend(&inner_writes, read, "  dus", k,
                    " = f32[8] dynamic-update-slice(buf", k, ", upd, i)\n");
    // The outer loop writes row k of buffer k of the inner loop's result.
    absl::StrAppend(&outer_writes, "  res", k,
                    " = f32[8] get-tuple-element(inner), index=", k + 1,
                    "\n  upd", k, " = f32[1] slice(res", k, "), slice={[", k,
                    ":", k + 1, "]}\n", read, "  dus", k,
                    " = f32[8] dynamic-update-slice(buf", k, ", upd", k,
                    ", i)\n");
    absl::StrAppend(&parameters, "  init", k, " = f32[8] parameter(", k, ")\n");
    updates.push_back(absl::StrCat("dus", k));
    inits.push_back(absl::StrCat("init", k));
  }
  const std::string root =
      absl::StrCat("  ROOT t = ", state, " tuple(next, ",
                   absl::StrJoin(updates, ", "), ")\n}\n\n");
  // The inner loop starts at row 4, from buffers filled with the outer
  // counter. The outer loop starts at row 2.
  Pack(absl::StrCat(
           "HloModule m\n\ninner_cond", cond, "outer_cond", cond, "inner_body",
           step, "  upd = f32[1] reshape(val)\n", inner_writes, root,
           "outer_body", step,
           "  four = s32[] constant(4)\n"
           "  fill = f32[8] broadcast(val), dimensions={}\n"
           "  start = ",
           state, " tuple(four",
           absl::StrJoin(std::vector<std::string>(kBuffers, ", fill"), ""),
           ")\n  inner = ", state,
           " while(start), condition=inner_cond, body=inner_body\n",
           outer_writes, root, "ENTRY e {\n  two = s32[] constant(2)\n",
           parameters, "  t = ", state, " tuple(two, ",
           absl::StrJoin(inits, ", "), ")\n  ROOT w = ", state,
           " while(t), condition=outer_cond, body=outer_body\n}\n"),
       ElementsAre("f32[8,8]"));
  const HloInstruction* inner =
      hlo_query::FindInstruction(Body(), HloOpcode::kWhile);
  ASSERT_NE(inner, nullptr);
  EXPECT_THAT(Shapes(inner->while_body(), HloOpcode::kDynamicUpdateSlice),
              ElementsAre("f32[8,8]"));
}

// The producer this pass was written for: forward sinking of per iteration
// scalar all reduces leaves one stack per sunk operand.
TEST_F(LoopRowBufferPackerTest, PacksForwardSinkStacks) {
  constexpr int kScalars = 8;
  const std::string state = absl::StrCat(
      "(s32[], f32[16,8]",
      absl::StrJoin(std::vector<std::string>(kScalars, ", f32[16]"), ""), ")");
  std::string body;
  std::vector<std::string> root = {"next", "x"};
  std::vector<std::string> init = {"c0", "x"};
  for (int k = 0; k < kScalars; ++k) {
    absl::StrAppend(&body, "  slc", k,
                    " = f32[1,1] slice(row), slice={[0:1], [", k, ":", k + 1,
                    "]}\n");
    absl::StrAppend(&body, "  red", k, " = f32[] reshape(slc", k, ")\n");
    absl::StrAppend(&body, "  arr", k, " = f32[] all-reduce(red", k,
                    "), replica_groups={}, to_apply=add, channel_id=", k + 1,
                    "\n");
    absl::StrAppend(&body, "  upd", k, " = f32[1] reshape(arr", k, ")\n");
    absl::StrAppend(&body, "  buf", k,
                    " = f32[16] get-tuple-element(p), index=", k + 2, "\n");
    absl::StrAppend(&body, "  dus", k, " = f32[16] dynamic-update-slice(buf", k,
                    ", upd", k, ", i)\n");
    root.push_back(absl::StrCat("dus", k));
    init.push_back("zeros");
  }
  ASSERT_OK_AND_ASSIGN(
      module_,
      ParseAndReturnVerifiedModule(absl::StrCat(
          R"(HloModule m

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

cond {
  p = )",
          state, R"( parameter(0)
  i = s32[] get-tuple-element(p), index=0
  n = s32[] constant(16)
  ROOT lt = pred[] compare(i, n), direction=LT
}

body {
  p = )",
          state, R"( parameter(0)
  i = s32[] get-tuple-element(p), index=0
  x = f32[16,8] get-tuple-element(p), index=1
  one = s32[] constant(1)
  next = s32[] add(i, one)
  zero = s32[] constant(0)
  row = f32[1,8] dynamic-slice(x, i, zero), dynamic_slice_sizes={1,8}
)",
          body, "  ROOT t = ", state, " tuple(", absl::StrJoin(root, ", "),
          R"()
}

ENTRY e {
  x = f32[16,8] parameter(0)
  c0 = s32[] constant(0)
  zf = f32[] constant(0)
  zeros = f32[16] broadcast(zf), dimensions={}
  t = )",
          state, " tuple(", absl::StrJoin(init, ", "), ")\n  ROOT w = ", state,
          " while(t), condition=cond, body=body\n}\n")));
  CollectivePipeliner::Config config;
  config.max_pipelining_per_loop = INT64_MAX;
  config.process_different_sized_ops = true;
  config.pipelining_direction =
      collective_pipeliner_utils::PipeliningDirection::kForwardSink;
  config.should_process = HloPredicateIsOp<HloOpcode::kAllReduce>;
  config.acceptable_formatting = HloPredicateTrue;
  config.reuse_pipelined_op_buffer = HloPredicateTrue;
  HloPassPipeline sink("sink");
  sink.AddPass<CollectivePipeliner>(config);
  sink.AddPass<HloDCE>(/*remove_cross_partition_collective_ops=*/true);
  ASSERT_THAT(sink.Run(module_.get()), IsOkAndHolds(true));
  ASSERT_THAT(Shapes(Body(), HloOpcode::kDynamicUpdateSlice),
              AllOf(SizeIs(kScalars), Each("f32[16]")));

  Pack(ElementsAre("f32[16,8]"), /*evaluate=*/false);
}

}  // namespace
}  // namespace xla
