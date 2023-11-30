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
#include "xla/hlo/utils/stack_frame_index_builder.h"
#include "xla/service/hlo.pb.h"

namespace mlir {

namespace {
bool IsFrameNameLocation(mlir::Location location) {
  return isa<mlir::NameLoc>(location) &&
         isa<mlir::FileLineColLoc>(cast<mlir::NameLoc>(location).getChildLoc());
}
}  // namespace

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

  int parent_frame_id = xla::StackFrameIndexBuilder::kInvalidIndex;
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
