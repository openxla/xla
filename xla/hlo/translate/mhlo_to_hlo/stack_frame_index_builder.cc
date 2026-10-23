/* Copyright 2023 The OpenXLA Authors.

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

#include <map>
#include <string>
#include <tuple>
#include <utility>

#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/hlo.pb.h"

namespace mlir {

int FindId(absl::string_view key, std::map<absl::string_view, int>& index) {
  auto entry_iterator = index.find(key);
  if (entry_iterator == index.end()) {
    return 0;
  }
  return entry_iterator->second;
}

int StackFrameIndexBuilder::AddStackFrameLocation(
    const mlir::NameLoc& name_location, int parent_frame_id) {
  mlir::FileLineColLoc file_line_location =
      cast<mlir::FileLineColLoc>(name_location.getChildLoc());

  int line = file_line_location.getLine();
  int end_line = file_line_location.getEndLine();
  int column = file_line_location.getColumn();
  int end_column = file_line_location.getEndColumn();
  std::string filename = file_line_location.getFilename().str();
  std::string function_name = name_location.getName().str();

  int filename_id = FindId(filename, file_name_to_id_);
  if (filename_id == 0) {
    indexes_.add_file_names(std::move(filename));
    filename_id = indexes_.file_names_size();
    file_name_to_id_[indexes_.file_names(filename_id - 1)] = filename_id;
  }

  int function_name_id = FindId(function_name, function_name_to_id_);
  if (function_name_id == 0) {
    indexes_.add_function_names(std::move(function_name));
    function_name_id = indexes_.function_names_size();
    function_name_to_id_[indexes_.function_names(function_name_id - 1)] =
        function_name_id;
  }

  auto location_tuple =
      std::make_tuple(filename_id, function_name_id, line, column);
  auto file_location_iterator = file_location_to_id_.find(location_tuple);
  int file_location_id = 0;
  if (file_location_iterator == file_location_to_id_.end()) {
    auto file_location = indexes_.add_file_locations();
    file_location->set_file_name_id(filename_id);
    file_location->set_function_name_id(function_name_id);
    file_location->set_line(line);
    file_location->set_end_line(end_line);
    file_location->set_column(column);
    file_location->set_end_column(end_column);

    file_location_id = indexes_.file_locations_size();
    file_location_to_id_[location_tuple] = file_location_id;
  } else {
    file_location_id = file_location_iterator->second;
  }

  auto frame_tuple = std::make_tuple(file_location_id, parent_frame_id);
  auto stack_frame_iterator = frame_to_id_.find(frame_tuple);
  int stack_frame_id = 0;
  if (stack_frame_iterator == frame_to_id_.end()) {
    auto frame = indexes_.add_stack_frames();
    frame->set_file_location_id(file_location_id);
    frame->set_parent_frame_id(parent_frame_id);

    stack_frame_id = indexes_.stack_frames_size();
    frame_to_id_[frame_tuple] = stack_frame_id;
  } else {
    stack_frame_id = stack_frame_iterator->second;
  }

  return stack_frame_id;
}

namespace {

bool IsFrameNameLocation(mlir::Location location) {
  return isa<mlir::NameLoc>(location) &&
         isa<mlir::FileLineColLoc>(cast<mlir::NameLoc>(location).getChildLoc());
}

}  // namespace

int StackFrameIndexBuilder::AddFrames(mlir::Location loc, int parent_frame_id) {
  // Based on source_info_to_location in JAX's jax/_src/interpreters/mlir.py
  // and on xla/hlo/translate/hlo_to_mhlo/location_importer.cc: op name
  // wrappers carry no frame and are unique per op, so look past them before
  // memoizing.
  while (isa<mlir::NameLoc>(loc) && !IsFrameNameLocation(loc)) {
    loc = cast<mlir::NameLoc>(loc).getChildLoc();
  }

  // Ops share call stacks and callers, so each (location, parent) pair is
  // walked once; a repeat is one lookup and inserts nothing.
  const std::pair<mlir::Location, int> key(loc, parent_frame_id);
  auto memo_iterator = call_stack_to_frame_id_.find(key);
  if (memo_iterator != call_stack_to_frame_id_.end()) {
    return memo_iterator->second;
  }

  int frame_id = parent_frame_id;
  // Based on JAX's `jaxlib/mlir/_mlir_libs/traceback_to_location.cc`, and on
  // `stackLocations` in `mlir/lib/Transforms/Utils/InliningUtils.cpp`.
  if (auto call_site = dyn_cast<mlir::CallSiteLoc>(loc)) {
    // The caller's frames are the parents of the callee's.
    int caller_frame_id = AddFrames(call_site.getCaller(), parent_frame_id);
    frame_id = AddFrames(call_site.getCallee(), caller_frame_id);
  } else if (IsFrameNameLocation(loc)) {
    // Also `jaxlib/mlir/_mlir_libs/traceback_to_location.cc`, which emits one
    // `NameLoc(<function name>, FileLineColRange)` per Python frame.
    frame_id = AddStackFrameLocation(cast<mlir::NameLoc>(loc), parent_frame_id);
  } else if (auto fused_loc = dyn_cast<mlir::FusedLoc>(loc)) {
    // Based on `xla/hlo/translate/hlo_to_mhlo/location_importer.cc`. The
    // sub-locations are unrelated ops, so keep the first stack instead of
    // concatenating them.
    for (mlir::Location sub_loc : fused_loc.getLocations()) {
      frame_id = AddFrames(sub_loc, parent_frame_id);
      if (frame_id != parent_frame_id) {
        break;
      }
    }
  }

  call_stack_to_frame_id_[key] = frame_id;
  return frame_id;
}

int StackFrameIndexBuilder::AddCallStackAndGetFirstFrameId(
    const mlir::Location& root_loc) {
  return AddFrames(root_loc, StackFrameIndexBuilder::kInvalidIndex);
}

xla::StackFrameIndexProto StackFrameIndexBuilder::Build() const {
  return std::move(indexes_);
}
}  // namespace mlir
