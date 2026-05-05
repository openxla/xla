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

// libFuzzer harness for the HLO proto deserialization path.
// Stage 1: deserialize raw bytes into HloModuleProto.
// Stage 2: convert HloModuleProto into an HloModule via CreateFromProto.

#include <cstddef>
#include <cstdint>
#include <limits>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  xla::HloModuleProto proto;
  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) return 0;
  if (!proto.ParseFromArray(data, static_cast<int>(size))) {
    return 0;
  }
  xla::HloModuleConfig config;
  auto result = xla::HloModule::CreateFromProto(proto, config);
  (void)result;
  return 0;
}
