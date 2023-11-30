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

#include <map>
#include <string>
#include <string_view>
#include <utility>

#include "xla/service/hlo.pb.h"

namespace xla {

namespace {
int FindId(const std::string_view key, std::map<std::string_view, int>& index) {
  auto entry_iterator = index.find(key);
  if (entry_iterator == index.end()) {
    return 0;
  } else {
    return entry_iterator->second;
  }
}
}  // namespace

int StackFrameIndexBuilder::FindOrAddFileName(std::string filename) {
  int filename_id = FindId(filename, file_name_to_id_);
  if (filename_id == 0) {
    indexes_.add_file_names(std::move(filename));
    filename_id = indexes_.file_names_size();
    file_name_to_id_[indexes_.file_names(filename_id - 1)] = filename_id;
  }

  return filename_id;
}

int StackFrameIndexBuilder::FindOrAddFunctionName(std::string function_name) {
  int function_name_id = FindId(function_name, function_name_to_id_);
  if (function_name_id == 0) {
    indexes_.add_function_names(std::move(function_name));
    function_name_id = indexes_.function_names_size();
    function_name_to_id_[indexes_.function_names(function_name_id - 1)] =
        function_name_id;
  }

  return function_name_id;
}

int StackFrameIndexBuilder::FindOrAddFileLocation(
    const xla::StackFrameIndexProto::FileLocation& file_location) {
  HashableFileLocation local_loc{file_location.file_name_id(),
                                 file_location.function_name_id(),
                                 file_location.line(), file_location.column()};

  auto file_location_iterator = file_location_to_id_.find(local_loc);
  int file_location_id = 0;
  if (file_location_iterator == file_location_to_id_.end()) {
    *indexes_.add_file_locations() = file_location;
    file_location_id = indexes_.file_locations_size();
    file_location_to_id_[local_loc] = file_location_id;
  } else {
    file_location_id = file_location_iterator->second;
  }

  return file_location_id;
}

int StackFrameIndexBuilder::FindOrAddStackFrame(
    const xla::StackFrameIndexProto::StackFrame& frame) {
  HashableStackFrame local_frame{frame.file_location_id(),
                                 frame.parent_frame_id()};
  auto stack_frame_iterator = frame_to_id_.find(local_frame);
  int stack_frame_id = 0;
  if (stack_frame_iterator == frame_to_id_.end()) {
    *indexes_.add_stack_frames() = frame;
    stack_frame_id = indexes_.stack_frames_size();
    frame_to_id_[local_frame] = stack_frame_id;
  } else {
    stack_frame_id = stack_frame_iterator->second;
  }

  return stack_frame_id;
}

int StackFrameIndexBuilder::AddStackFrameAndReturnId(std::string file_name,
                                                     int line_number,
                                                     std::string function_name,
                                                     int column_number,
                                                     int parent_frame_id) {
  int filename_id = FindOrAddFileName(std::move(file_name));
  int function_name_id = FindOrAddFunctionName(std::move(function_name));

  xla::StackFrameIndexProto::FileLocation file_location;
  file_location.set_file_name_id(filename_id);
  file_location.set_function_name_id(function_name_id);
  file_location.set_line(line_number);
  file_location.set_column(column_number);

  int file_location_id = FindOrAddFileLocation(file_location);

  xla::StackFrameIndexProto::StackFrame proto_frame;
  proto_frame.set_file_location_id(file_location_id);
  proto_frame.set_parent_frame_id(parent_frame_id);

  int stack_frame_id = FindOrAddStackFrame(proto_frame);

  return stack_frame_id;
}

xla::StackFrameIndexProto StackFrameIndexBuilder::Build() const {
  return std::move(indexes_);
}
}  // namespace xla
