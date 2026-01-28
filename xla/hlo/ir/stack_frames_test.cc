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
============================================h==================================*/

#include "xla/hlo/ir/stack_frames.h"

#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

TEST(StackFramesTest, Empty) {
  StackFrames dag;
  EXPECT_TRUE(dag.empty());
  EXPECT_EQ(dag.GetStackFrame(StackFrameId{1}).file_name, "");
}

TEST(StackFramesTest, InitializeFromProto) {
  StackFrameIndexProto proto;
  proto.add_file_names("file1.py");
  proto.add_function_names("func1");
  auto* loc = proto.add_file_locations();
  loc->set_file_name_id(1);
  loc->set_function_name_id(1);
  loc->set_line(10);
  loc->set_column(5);
  auto* frame = proto.add_stack_frames();
  frame->set_file_location_id(1);
  frame->set_parent_frame_id(0);

  StackFrames dag(proto);
  EXPECT_FALSE(dag.empty());
  HloStackFrame got = dag.GetStackFrame(StackFrameId{1});
  EXPECT_EQ(got.file_name, "file1.py");
  EXPECT_EQ(got.function_name, "func1");
  EXPECT_EQ(got.line, 10);
  EXPECT_EQ(got.column, 5);
  EXPECT_EQ(got.parent_frame_id, StackFrameId{0});
}

TEST(StackFramesTest, Nested) {
  StackFrameIndexProto proto;
  proto.add_file_names("file1.py");
  proto.add_file_names("file2.py");
  proto.add_function_names("func1");
  proto.add_function_names("func2");

  auto* loc1 = proto.add_file_locations();
  loc1->set_file_name_id(1);
  loc1->set_function_name_id(1);
  loc1->set_line(10);

  auto* loc2 = proto.add_file_locations();
  loc2->set_file_name_id(2);
  loc2->set_function_name_id(2);
  loc2->set_line(20);

  auto* frame1 = proto.add_stack_frames();
  frame1->set_file_location_id(1);
  frame1->set_parent_frame_id(0);

  auto* frame2 = proto.add_stack_frames();
  frame2->set_file_location_id(2);
  frame2->set_parent_frame_id(1);

  StackFrames dag(proto);
  HloStackFrame got2 = dag.GetStackFrame(StackFrameId{2});
  EXPECT_EQ(got2.file_name, "file2.py");
  EXPECT_EQ(got2.parent_frame_id, StackFrameId{1});

  HloStackFrame got1 = dag.GetStackFrame(got2.parent_frame_id);
  EXPECT_EQ(got1.file_name, "file1.py");
  EXPECT_EQ(got1.parent_frame_id, StackFrameId{0});
}

}  // namespace
}  // namespace xla
