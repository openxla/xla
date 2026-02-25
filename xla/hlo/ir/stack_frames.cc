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

#include "xla/hlo/ir/stack_frames.h"

#include <utility>

#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/service/hlo.pb.h"

namespace xla {

StackFrames::StackFrames(StackFrameIndexProto proto)
    : proto_(std::move(proto)) {}

HloStackFrame StackFrames::GetStackFrame(StackFrameId id) const {
  if (!id.valid() || id.value > proto_.stack_frames_size()) {
    return {};
  }
  const auto& frame_proto = proto_.stack_frames(id.value - 1);
  const auto& loc_proto =
      proto_.file_locations(frame_proto.file_location_id() - 1);
  return {proto_.file_names(loc_proto.file_name_id() - 1),
          proto_.function_names(loc_proto.function_name_id() - 1),
          loc_proto.line(), loc_proto.column(),
          StackFrameId{frame_proto.parent_frame_id()}};
}

}  // namespace xla
