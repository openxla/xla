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

// Usage: heap_simulator_replay module.hlo.pb [alignment_bytes]

#include <cstdint>
#include <iostream>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "xla/service/hlo.pb.h"
#include "xla/tools/heap_simulator_replay.h"
#include "xla/tsl/platform/env.h"

int main(int argc, char** argv) {
  int64_t alignment = 256;
  if (argc < 2 || argc > 3 ||
      (argc == 3 && !absl::SimpleAtoi(argv[2], &alignment)) || alignment <= 0) {
    std::cerr
        << "Usage: heap_simulator_replay module.hlo.pb [alignment_bytes]\n";
    return 1;
  }
  xla::HloProto proto;
  absl::Status status =
      tsl::ReadBinaryProto(tsl::Env::Default(), argv[1], &proto);
  if (status.ok()) {
    status = xla::ReplayHeapSimulator(proto, alignment, std::cout);
  }
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
