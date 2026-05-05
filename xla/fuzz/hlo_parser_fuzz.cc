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

// libFuzzer harness for the HLO text-format parser.
// Exercises xla::ParseAndReturnUnverifiedModule against arbitrary input bytes.

#include <cstddef>
#include <cstdint>

#include "absl/strings/string_view.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/hlo_module_config.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  absl::string_view input(reinterpret_cast<const char*>(data), size);
  xla::HloModuleConfig config;
  auto result = xla::ParseAndReturnUnverifiedModule(input, config);
  (void)result;
  return 0;
}
