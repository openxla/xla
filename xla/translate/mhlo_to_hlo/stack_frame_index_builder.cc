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

#include "xla/translate/mhlo_to_hlo/stack_frame_index_builder.h"

#include <map>
#include <stack>
#include <string>
#include <string_view>
#include <utility>

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/service/hlo.pb.h"

namespace mlir {

namespace {
int FindId(const std::string_view key, std::map<std::string_view, int>& index) {
  auto entry_iterator = index.find(key);
  if (entry_iterator == index.end()) {
    return 0;
  } else {
    return entry_iterator->second;
  }
}

bool IsFrameNameLocation(mlir::Location location) {
  return isa<mlir::NameLoc>(location) &&
         isa<mlir::FileLineColLoc>(cast<mlir::NameLoc>(location).getChildLoc());
}
}  // namespace

int BaseStackFrameIndexBuilder::FindOrAddFileName(std::string filename) {
  int filename_id = FindId(filename, file_name_to_id_);
  if (filename_id == 0) {
    indexes_.add_file_names(std::move(filename));
    filename_id = indexes_.file_names_size();
    file_name_to_id_[indexes_.file_names(filename_id - 1)] = filename_id;
  }

  return filename_id;
}

int BaseStackFrameIndexBuilder::FindOrAddFunctionName(
    std::string function_name) {
  int function_name_id = FindId(function_name, function_name_to_id_);
  if (function_name_id == 0) {
    indexes_.add_function_names(std::move(function_name));
    function_name_id = indexes_.function_names_size();
    function_name_to_id_[indexes_.function_names(function_name_id - 1)] =
        function_name_id;
  }

  return function_name_id;
}

int BaseStackFrameIndexBuilder::FindOrAddFileLocation(
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

int BaseStackFrameIndexBuilder::FindOrAddStackFrame(
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

int BaseStackFrameIndexBuilder::AddStackFrameAndReturnId(
    std::string file_name, int line_number, std::string function_name,
    int column_number, int parent_frame_id) {
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

xla::StackFrameIndexProto BaseStackFrameIndexBuilder::Build() const {
  return std::move(indexes_);
}

int StackFrameIndexBuilder::AddCallStackAndGetFirstFrameId(
    const mlir::Location &root_loc) {
  std::stack<mlir::NameLoc> locations;
  mlir::CallSiteLoc call_site;
  mlir::Location caller = root_loc;
  while ((call_site = dyn_cast<mlir::CallSiteLoc>(caller)) != nullptr) {
    mlir::Location callee = call_site.getCallee();
    caller = call_site.getCaller();

    if (IsFrameNameLocation(callee)) {
      locations.push(cast<mlir::NameLoc>(callee));
    }
    if (IsFrameNameLocation(caller)) {
      locations.push(cast<mlir::NameLoc>(caller));
    }
  }

  // If stack has only one frame it's stored in root location.
  if (IsFrameNameLocation(root_loc)) {
    locations.push(cast<mlir::NameLoc>(root_loc));
  }

  int parent_frame_id = BaseStackFrameIndexBuilder::kInvalidIndex;
  while (!locations.empty()) {
    mlir::NameLoc name_location = locations.top();
    locations.pop();

    mlir::FileLineColLoc file_line_location =
        cast<mlir::FileLineColLoc>(name_location.getChildLoc());

    int line = file_line_location.getLine();
    int column = file_line_location.getColumn();
    std::string filename = file_line_location.getFilename().str();
    std::string function_name = name_location.getName().str();

    parent_frame_id = builder_.AddStackFrameAndReturnId(
        std::move(filename), line, std::move(function_name), column,
        parent_frame_id);
  }

  return parent_frame_id;
}

xla::StackFrameIndexProto StackFrameIndexBuilder::Build() const {
  return builder_.Build();
}

}  // namespace mlir
