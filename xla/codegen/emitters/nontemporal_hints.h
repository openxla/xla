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

#ifndef XLA_CODEGEN_EMITTERS_NONTEMPORAL_HINTS_H_
#define XLA_CODEGEN_EMITTERS_NONTEMPORAL_HINTS_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"

namespace xla::emitters {

// Argument attributes describing how a fusion reads an operand. Set by
// AnnotateNontemporalHintFacts, used and removed by LowerTensorsPass.

// Read exactly once per output element, with no reuse.
inline constexpr absl::string_view kXlaStreamedOnceAttr = "xla.streamed_once";

// No map reads an element of the operand more than once.
inline constexpr absl::string_view kXlaNotBroadcastAttr = "xla.not_broadcast";

// Another instruction, or another operand of the fusion, reads the buffer too;
// its loads stay temporal below 64x L2.
inline constexpr absl::string_view kXlaReadElsewhereAttr = "xla.read_elsewhere";

// A wavefront load reads part of a 128-byte line; other loads read the rest.
inline constexpr absl::string_view kXlaPiecewiseReadsAttr =
    "xla.piecewise_reads";

// Each output element is written once, as loop and concatenate emitters do.
inline constexpr absl::string_view kXlaContiguousOutputAttr =
    "xla.contiguous_output";

// gfx942 gets load and store hints from 16x L2. Other targets get none.
struct NontemporalHintTargets {
  bool loads = false;
  bool stores = false;
  // Store hints only if a single output qualifies.
  bool single_output_stores = false;
  // No load hints in kernels that overwrite an input.
  bool out_of_place_loads = false;
  // Buffers below this multiple of L2 stay temporal.
  int stream_l2_multiple = 16;
};
NontemporalHintTargets GetNontemporalHintTargets(
    const stream_executor::DeviceDescription& device);

// How a fusion reads an operand, from its indexing maps over all roots.
struct OperandReads {
  // Every map is defined, has no runtime variables, and reads each element at
  // most once.
  bool not_broadcast = false;
  // One map, no range variables, and no other operand of the fusion reads the
  // buffer: read once per output element.
  bool streamed_once = false;
  // Another instruction, or another operand of the fusion, reads the buffer.
  bool read_elsewhere = false;
};

// OperandReads per operand id. Empty for non-fusions and scatter fusions.
// Without `dataflow`, every buffer counts as read elsewhere and no operand as
// streamed once.
std::vector<OperandReads> ComputeOperandReads(
    const HloFusionInstruction& fusion, const HloDataflowAnalysis* dataflow,
    mlir::MLIRContext* mlir_context);

// OperandReads::read_elsewhere per operand id: the only fact that depends on
// more than the fused computation.
std::vector<bool> ComputeReadElsewhere(const HloFusionInstruction& fusion,
                                       const HloDataflowAnalysis* dataflow);

// Whether `map` reads each element of `shape` at most once. False unless the
// linearized index simplifies to a sum of variables, and of variables
// floor-divided by or modulo constants, times constants.
bool ReadsEachElementAtMostOnce(const IndexingMap& map, const Shape& shape);

// Whether a wavefront load reads part of a 128-byte line of an operand.
// Checks the first two wavefronts of the first two blocks with an active
// thread. True if the scan reaches block 1024 before finding two.
bool ReadsLinesInPieces(const IndexingMap& thread_id_to_index,
                        int64_t wavefront_size, int64_t element_bytes,
                        int64_t buffer_bytes);

// Sets the read facts on the entry's operand args, and xla.contiguous_output
// if `contiguous_output`.
absl::Status AnnotateNontemporalHintFacts(mlir::ModuleOp module,
                                          absl::string_view entry_function_name,
                                          const HloInstruction& hlo_instruction,
                                          const HloDataflowAnalysis* dataflow,
                                          bool contiguous_output);

}  // namespace xla::emitters

#endif  // XLA_CODEGEN_EMITTERS_NONTEMPORAL_HINTS_H_
