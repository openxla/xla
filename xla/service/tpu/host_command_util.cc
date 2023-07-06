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

#include "xla/service/tpu/host_command_util.h"

#include <string>

#include "absl/strings/str_format.h"

namespace xla::tpu::host_command {

/*static*/ std::string HostCommandDecoder::DebugString(uint32_t command) {
  return absl::StrFormat("%s: %#x | %s: %#x | %s: %#x | %s: %d", "Command",
                         command, "OpCode", opcode(command), "Operand",
                         operand(command), "Channel", channel(command));
}

}  // namespace xla::tpu::host_command
