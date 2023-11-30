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

#ifndef XLA_TRANSLATE_MHLO_TO_HLO_STACK_FRAME_INDEX_BUILDER_H_
#define XLA_TRANSLATE_MHLO_TO_HLO_STACK_FRAME_INDEX_BUILDER_H_

#include <map>
#include <string_view>
#include <tuple>

#include "absl/container/flat_hash_map.h"
#include "mlir/IR/Location.h"  // from @llvm-project
#include "xla/service/hlo.pb.h"

namespace mlir {
class BaseStackFrameIndexBuilder {
 public:
  constexpr static int kInvalidIndex = 0;

  xla::StackFrameIndexProto Build() const;

  // Returns the new stack frame index in the indexes_ proto
  // To add a full call stack, the result of one call to this function should be
  // the parent_frame_id of the next call.
  // Use parent_frame_id = StackFrameIndexBuilder::kInvalidIndex for the root
  // of the stack.
  int AddStackFrameAndReturnId(std::string file_name, int line_number,
                               std::string function_name, int column_number,
                               int parent_frame_id);

 private:
  int FindOrAddFileName(std::string filename);
  int FindOrAddFunctionName(std::string function_name);
  int FindOrAddFileLocation(
      const xla::StackFrameIndexProto::FileLocation& file_location);
  int FindOrAddStackFrame(const xla::StackFrameIndexProto::StackFrame& frame);

  xla::StackFrameIndexProto indexes_;

  std::map<std::string_view, int> function_name_to_id_;
  std::map<std::string_view, int> file_name_to_id_;

  // Equivalent to the FileLocation proto
  using HashableFileLocation = std::tuple<int, int, int, int>;
  // Equivalent to the StackFrame proto
  using HashableStackFrame = std::tuple<int, int>;

  absl::flat_hash_map<HashableFileLocation, int> file_location_to_id_;
  absl::flat_hash_map<HashableStackFrame, int> frame_to_id_;
};

class StackFrameIndexBuilder {
 public:
  int AddCallStackAndGetFirstFrameId(const mlir::Location& root_loc);
  xla::StackFrameIndexProto Build() const;

 private:
  int AddStackFrameLocation(const mlir::NameLoc& name_location,
                            int parent_frame_id);

  BaseStackFrameIndexBuilder builder_;
};
}  // namespace mlir

#endif  // XLA_TRANSLATE_MHLO_TO_HLO_STACK_FRAME_INDEX_BUILDER_H_
