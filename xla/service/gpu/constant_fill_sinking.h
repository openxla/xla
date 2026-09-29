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

#ifndef XLA_SERVICE_GPU_CONSTANT_FILL_SINKING_H_
#define XLA_SERVICE_GPU_CONSTANT_FILL_SINKING_H_

#include <vector>

#include "xla/hlo/ir/hlo_instruction.h"

namespace xla::gpu {

// Shortens the lifetimes of large, independent scalar-constant fills in a
// scheduled instruction sequence by moving each fill immediately before its
// first user. The relative order of every other instruction is preserved.
//
// A fill is a zero-operand loop fusion whose root is a broadcast of a scalar
// constant. Fills with explicit ordering or stream constraints are left alone.
// A fill inside an async window (between an async start and its done) is only
// moved when its first user lies inside that same window, so the move never
// takes work out from under an in-flight async operation.
//
// Intended as the latency-hiding scheduler's post-processing step, where the
// shorter lifetimes are visible to the scheduler's memory accounting and to
// rematerialization. Returns true if `sequence` changed.
bool SinkConstantFills(std::vector<HloInstruction*>& sequence);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_CONSTANT_FILL_SINKING_H_
