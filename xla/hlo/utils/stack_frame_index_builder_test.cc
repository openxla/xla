/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/hlo/utils/stack_frame_index_builder.h"

#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xla/service/hlo.pb.h"

namespace xla {
namespace {

struct StackFrame {
  std::string file_name;
  int line_number;
  int column_number;
  std::string function_name;
};

void CompareSingleFrame(const xla::StackFrameIndexProto& proto,
                        const StackFrame& frame, int root_index) {
  auto file_location = proto.file_locations(
      proto.stack_frames(root_index - 1).file_location_id() - 1);
  EXPECT_EQ(frame.file_name,
            proto.file_names(file_location.file_name_id() - 1));
  EXPECT_EQ(frame.function_name,
            proto.function_names(file_location.function_name_id() - 1));
  EXPECT_EQ(frame.line_number, file_location.line());
}

void CompareStack(xla::StackFrameIndexProto& proto,
                  const std::vector<StackFrame>& frames, int root_index) {
  auto frame_it = frames.rbegin();
  while (root_index != StackFrameIndexBuilder::kInvalidIndex) {
    CompareSingleFrame(proto, *frame_it, root_index);
    ++frame_it;
    root_index = proto.stack_frames(root_index - 1).parent_frame_id();
  }
}

TEST(StackFrameIndexBuilder, CallStacksAreAddedAndDeduplicatedCorrectly) {
  StackFrameIndexBuilder builder;
  std::vector<StackFrame> user_frames;

  user_frames.push_back({"test/file.py", 5, 0, "user_def_function"});
  user_frames.push_back({"test/file.py", 0, 1, "another_function"});
  user_frames.push_back({"test/another_file.py", 6, 2, "yet_another_function"});

  int result_index = StackFrameIndexBuilder::kInvalidIndex;
  for (auto& frame : user_frames) {
    result_index = builder.AddStackFrameAndReturnId(
        frame.file_name, frame.line_number, frame.function_name,
        frame.column_number, result_index);
  }

  auto proto = builder.Build();
  // The proto should contain all frames, and the duplicated filename
  // "test/file.py" should be stored only once.
  EXPECT_EQ(proto.file_locations_size(), 3);
  EXPECT_EQ(proto.file_names_size(), 2);
  EXPECT_EQ(proto.function_names_size(), 3);
  EXPECT_EQ(proto.stack_frames_size(), user_frames.size());
  CompareStack(proto, user_frames, result_index);

  // Adding the same stack again will result in no extra data and return the
  // same result index
  int compare_result_index = StackFrameIndexBuilder::kInvalidIndex;
  for (auto& frame : user_frames) {
    compare_result_index = builder.AddStackFrameAndReturnId(
        frame.file_name, frame.line_number, frame.function_name,
        frame.column_number, compare_result_index);
  }
  EXPECT_EQ(result_index, compare_result_index);

  // Adding a different stack with the same locations will result in a new
  // stack index, but will re-use all the existing locations
  std::reverse(user_frames.begin(), user_frames.end());
  compare_result_index = StackFrameIndexBuilder::kInvalidIndex;
  for (auto& frame : user_frames) {
    compare_result_index = builder.AddStackFrameAndReturnId(
        frame.file_name, frame.line_number, frame.function_name,
        frame.column_number, compare_result_index);
  }

  EXPECT_NE(result_index, compare_result_index);

  proto = builder.Build();
  CompareStack(proto, user_frames, compare_result_index);
}

}  // namespace
}  // namespace xla
