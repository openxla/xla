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

#include "xla/hlo/translate/mhlo_to_hlo/stack_frame_index_builder.h"

#include <string>
#include <vector>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/test.h"

namespace mlir {

class StackFrameIndexBuilderTestPeer {
 public:
  static int MemoSize(const StackFrameIndexBuilder& builder) {
    return builder.call_stack_to_frame_id_.size();
  }
};

namespace {

Location MakeFrameLoc(MLIRContext* ctx, const std::string& name,
                      const std::string& file, int line) {
  return NameLoc::get(StringAttr::get(ctx, name),
                      FileLineColLoc::get(StringAttr::get(ctx, file), line, 0));
}

std::vector<std::string> GetFrameNames(const xla::StackFrameIndexProto& proto,
                                       int frame_id) {
  std::vector<std::string> names;
  while (frame_id != StackFrameIndexBuilder::kInvalidIndex) {
    const auto& frame = proto.stack_frames(frame_id - 1);
    const auto& loc = proto.file_locations(frame.file_location_id() - 1);
    names.push_back(proto.function_names(loc.function_name_id() - 1));
    frame_id = frame.parent_frame_id();
  }
  return names;
}

TEST(StackFrameIndexBuilderTest, CallSiteLocConcatenation) {
  MLIRContext ctx;
  Location a = MakeFrameLoc(&ctx, "A", "a.py", 1);
  Location b = MakeFrameLoc(&ctx, "B", "b.py", 2);
  Location c = MakeFrameLoc(&ctx, "C", "c.py", 3);
  Location d = MakeFrameLoc(&ctx, "D", "d.py", 4);

  Location ab = CallSiteLoc::get(a, b);
  Location cd = CallSiteLoc::get(c, d);
  Location abcd = CallSiteLoc::get(ab, cd);

  StackFrameIndexBuilder builder;
  int frame_id = builder.AddCallStackAndGetFirstFrameId(abcd);
  auto proto = builder.Build();

  std::vector<std::string> names = GetFrameNames(proto, frame_id);
  std::vector<std::string> expected = {"A", "B", "C", "D"};
  EXPECT_EQ(names, expected);
}

TEST(StackFrameIndexBuilderTest, DeeplyNestedCallSiteLoc) {
  MLIRContext ctx;
  Location a = MakeFrameLoc(&ctx, "A", "a.py", 1);
  Location b = MakeFrameLoc(&ctx, "B", "b.py", 2);
  Location c = MakeFrameLoc(&ctx, "C", "c.py", 3);
  Location d = MakeFrameLoc(&ctx, "D", "d.py", 4);
  Location e = MakeFrameLoc(&ctx, "E", "e.py", 5);
  Location f = MakeFrameLoc(&ctx, "F", "f.py", 6);

  Location ab = CallSiteLoc::get(a, b);
  Location abc = CallSiteLoc::get(ab, c);
  Location ef = CallSiteLoc::get(e, f);
  Location def = CallSiteLoc::get(d, ef);
  Location abcdef = CallSiteLoc::get(abc, def);

  StackFrameIndexBuilder builder;
  int frame_id = builder.AddCallStackAndGetFirstFrameId(abcdef);
  auto proto = builder.Build();

  std::vector<std::string> names = GetFrameNames(proto, frame_id);
  std::vector<std::string> expected = {"A", "B", "C", "D", "E", "F"};
  EXPECT_EQ(names, expected);
}

TEST(StackFrameIndexBuilderTest, LinearChain) {
  MLIRContext ctx;
  Location a = MakeFrameLoc(&ctx, "A", "a.py", 1);
  Location b = MakeFrameLoc(&ctx, "B", "b.py", 2);
  Location c = MakeFrameLoc(&ctx, "C", "c.py", 3);
  Location d = MakeFrameLoc(&ctx, "D", "d.py", 4);

  Location cd = CallSiteLoc::get(c, d);
  Location bcd = CallSiteLoc::get(b, cd);
  Location abcd = CallSiteLoc::get(a, bcd);

  StackFrameIndexBuilder builder;
  int frame_id = builder.AddCallStackAndGetFirstFrameId(abcd);
  auto proto = builder.Build();

  std::vector<std::string> names = GetFrameNames(proto, frame_id);
  std::vector<std::string> expected = {"A", "B", "C", "D"};
  EXPECT_EQ(names, expected);
}

TEST(StackFrameIndexBuilderTest, InlinedOpWithOpNameWrapperLocs) {
  MLIRContext ctx;
  Location a = MakeFrameLoc(&ctx, "A", "a.py", 1);
  Location b = MakeFrameLoc(&ctx, "B", "b.py", 2);
  Location c = MakeFrameLoc(&ctx, "C", "c.py", 3);
  Location d = MakeFrameLoc(&ctx, "D", "d.py", 4);

  // JAX wraps each op's CallSiteLoc in NameLoc("op_type:", NameLoc("op_name",
  // ...)).
  Location callee_op =
      NameLoc::get(StringAttr::get(&ctx, "reduce_sum:"),
                   NameLoc::get(StringAttr::get(&ctx, "jit(fn)/reduce_sum"),
                                CallSiteLoc::get(a, b)));
  Location caller_op =
      NameLoc::get(StringAttr::get(&ctx, "call:"),
                   NameLoc::get(StringAttr::get(&ctx, "jit(fn)/call"),
                                CallSiteLoc::get(c, d)));
  // MLIR's createInlinerPass wraps the inlined op's location in CallSiteLoc.
  Location inlined = CallSiteLoc::get(callee_op, caller_op);

  StackFrameIndexBuilder builder;
  int frame_id = builder.AddCallStackAndGetFirstFrameId(inlined);
  auto proto = builder.Build();

  std::vector<std::string> names = GetFrameNames(proto, frame_id);
  std::vector<std::string> expected = {"A", "B", "C", "D"};
  EXPECT_EQ(names, expected);
}

TEST(StackFrameIndexBuilderTest, FusedLocKeepsFirstCallStack) {
  MLIRContext ctx;
  Location a = MakeFrameLoc(&ctx, "A", "a.py", 1);
  Location b = MakeFrameLoc(&ctx, "B", "b.py", 2);
  Location c = MakeFrameLoc(&ctx, "C", "c.py", 3);

  // Merging two ops fuses their unrelated call stacks.
  Location fused = FusedLoc::get(&ctx, {CallSiteLoc::get(a, b), c});

  StackFrameIndexBuilder builder;
  int frame_id = builder.AddCallStackAndGetFirstFrameId(fused);
  auto proto = builder.Build();

  std::vector<std::string> names = GetFrameNames(proto, frame_id);
  std::vector<std::string> expected = {"A", "B"};
  EXPECT_EQ(names, expected);
}

TEST(StackFrameIndexBuilderTest, FusedLocSkipsSubLocationsWithoutFrames) {
  MLIRContext ctx;
  Location a = MakeFrameLoc(&ctx, "A", "a.py", 1);

  Location no_frames =
      NameLoc::get(StringAttr::get(&ctx, "reduce_sum:"), UnknownLoc::get(&ctx));
  Location fused = FusedLoc::get(&ctx, {no_frames, a});

  StackFrameIndexBuilder builder;
  int frame_id = builder.AddCallStackAndGetFirstFrameId(fused);
  auto proto = builder.Build();

  std::vector<std::string> names = GetFrameNames(proto, frame_id);
  std::vector<std::string> expected = {"A"};
  EXPECT_EQ(names, expected);
}

// Ops sharing a call stack, or a caller, reuse the frames of the first walk;
// the same frame under another caller, or seen as a root, gets its own entry.
TEST(StackFrameIndexBuilderTest, SharedCallStacksAndCallers) {
  MLIRContext ctx;
  Location a = MakeFrameLoc(&ctx, "A", "a.py", 1);
  Location b = MakeFrameLoc(&ctx, "B", "b.py", 2);
  Location c = MakeFrameLoc(&ctx, "C", "c.py", 3);
  Location d = MakeFrameLoc(&ctx, "D", "d.py", 4);
  Location e = MakeFrameLoc(&ctx, "E", "e.py", 5);
  auto wrap = [&](const char* name, Location loc) {
    return NameLoc::get(StringAttr::get(&ctx, name), loc);
  };
  Location callee_op_a =
      wrap("add:", wrap("jit(f)/add", CallSiteLoc::get(a, b)));
  Location callee_op_e =
      wrap("mul:", wrap("jit(f)/mul", CallSiteLoc::get(e, b)));
  Location caller_op =
      wrap("call:", wrap("jit(f)/call", CallSiteLoc::get(c, d)));
  Location inlined_a = CallSiteLoc::get(callee_op_a, caller_op);
  Location inlined_e = CallSiteLoc::get(callee_op_e, caller_op);

  StackFrameIndexBuilder builder;
  int frame_a = builder.AddCallStackAndGetFirstFrameId(inlined_a);
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(inlined_a), frame_a);
  int frame_e = builder.AddCallStackAndGetFirstFrameId(inlined_e);
  int frame_ac = builder.AddCallStackAndGetFirstFrameId(CallSiteLoc::get(a, c));
  int root_a = builder.AddCallStackAndGetFirstFrameId(callee_op_a);
  xla::StackFrameIndexProto proto = builder.Build();

  using Names = std::vector<std::string>;
  EXPECT_EQ(GetFrameNames(proto, frame_a), Names({"A", "B", "C", "D"}));
  EXPECT_EQ(GetFrameNames(proto, frame_e), Names({"E", "B", "C", "D"}));
  EXPECT_EQ(GetFrameNames(proto, frame_ac), Names({"A", "C"}));
  EXPECT_EQ(GetFrameNames(proto, root_a), Names({"A", "B"}));
  // D, C, B, A and E under the shared caller; C and A under C; B and A as a
  // root: no duplicate entries for the repeated root or the shared caller.
  EXPECT_EQ(proto.stack_frames_size(), 9);
}

// Two ops inlined under one caller share the caller's frames and the callee
// frames below their leaves; the memo holds each (location, parent) pair once
// and none of the per op name wrappers.
TEST(StackFrameIndexBuilderTest, MemoHoldsSharedLocationsOnly) {
  MLIRContext ctx;
  Location a = MakeFrameLoc(&ctx, "A", "a.py", 1);
  Location b = MakeFrameLoc(&ctx, "B", "b.py", 2);
  Location c = MakeFrameLoc(&ctx, "C", "c.py", 3);
  Location d = MakeFrameLoc(&ctx, "D", "d.py", 4);
  Location e = MakeFrameLoc(&ctx, "E", "e.py", 5);
  auto wrap = [&](const char* name, Location loc) {
    return NameLoc::get(StringAttr::get(&ctx, name), loc);
  };
  Location caller_op =
      wrap("call:", wrap("jit(f)/call", CallSiteLoc::get(c, d)));
  Location inlined_a = CallSiteLoc::get(
      wrap("add:", wrap("jit(f)/add", CallSiteLoc::get(a, b))), caller_op);
  Location inlined_e = CallSiteLoc::get(
      wrap("mul:", wrap("jit(f)/mul", CallSiteLoc::get(e, b))), caller_op);

  StackFrameIndexBuilder builder;
  int frame_a = builder.AddCallStackAndGetFirstFrameId(inlined_a);
  int frame_e = builder.AddCallStackAndGetFirstFrameId(inlined_e);
  // Pairs walked, with the parent in brackets: inlined_a [0], CallSiteLoc(c,
  // d) [0], d [0], c [D], CallSiteLoc(a, b) [C], b [C], a [B], inlined_e [0],
  // CallSiteLoc(e, b) [C] and e [B]; b [C] is a hit and the wrappers are not
  // keys.
  EXPECT_EQ(StackFrameIndexBuilderTestPeer::MemoSize(builder), 10);
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(inlined_a), frame_a);
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(inlined_e), frame_e);
  EXPECT_EQ(StackFrameIndexBuilderTestPeer::MemoSize(builder), 10);

  xla::StackFrameIndexProto proto = builder.Build();
  using Names = std::vector<std::string>;
  EXPECT_EQ(GetFrameNames(proto, frame_a), Names({"A", "B", "C", "D"}));
  EXPECT_EQ(GetFrameNames(proto, frame_e), Names({"E", "B", "C", "D"}));
  EXPECT_EQ(proto.stack_frames_size(), 5);
}

}  // namespace
}  // namespace mlir
