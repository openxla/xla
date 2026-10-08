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

#ifndef XLA_TOOLS_HEAP_SIMULATOR_REPLAY_H_
#define XLA_TOOLS_HEAP_SIMULATOR_REPLAY_H_

#include <cstdint>
#include <iosfwd>

#include "absl/status/status.h"

namespace xla {

class HloProto;

// Prints six placement results for the traced heap in the largest ordinary
// temporary allocation. Reports an appended untraced suffix separately, and
// rejects allocations whose trace cannot be replayed as one unconstrained heap.
// Does not mutate the proto or compile the module.
absl::Status ReplayHeapSimulator(const HloProto& proto, int64_t alignment,
                                 std::ostream& output);

}  // namespace xla

#endif  // XLA_TOOLS_HEAP_SIMULATOR_REPLAY_H_
