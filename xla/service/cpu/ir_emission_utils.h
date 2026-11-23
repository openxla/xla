/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_IR_EMISSION_UTILS_H_
#define XLA_SERVICE_CPU_IR_EMISSION_UTILS_H_

#include <cstdint>

#include "xla/backends/cpu/alignment.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/codegen/device_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {

bool PotentiallyImplementedAsEigenConvolution(
    const HloInstruction& convolution,
    const TargetMachineFeatures& target_machine_features);

bool CanUseEigenConvolution(
    const HloInstruction& convolution,
    const TargetMachineFeatures& target_machine_features);

// Computes the minimum alignment guaranteed for a tensor of shape `shape` on
// the target machine.
int64_t GetMinimumAlignmentForArray(
    const Shape& shape, const TargetMachineFeatures& target_machine_features);

// Returns true if no thunks are emitted for `hlo`: its semantics are encoded by
// buffer assignment, by the schedule, or by constants owned by the executable.
bool EmitsNoThunks(const HloInstruction& hlo);

// Returns true if `type` packs multiple elements into a single storage byte in
// CPU scatter lowering (S2, U2, S4, U4, and F4E2M1FN).
bool IsCpuScatterPackedSubByteType(PrimitiveType type);

// Returns true if a scatter reducer discards the existing destination value
// (parameter 0 is dead and not the root) and lowers to an atomic store under
// concurrency.
bool IsCpuScatterOverwriteReducer(const HloComputation& to_apply);

// Returns true if a scatter reducer lowers to a single direct integer
// `llvm.atomicrmw` instruction on CPU rather than a compare and swap loop.
bool IsCpuScatterDirectAtomicRmwReducer(const HloComputation& to_apply,
                                        PrimitiveType element_type);

// Returns true if multi workgroup execution of `scatter` must lower updates
// through atomic memory operations rather than plain loads and stores.
bool CpuScatterRequiresAtomicUpdate(const HloScatterInstruction& scatter);

// Returns the requested workgroup count from `fusion`'s
// `outer_dimension_partitions` backend config (at least 1).
int64_t GetCpuScatterRequestedWorkGroups(const HloFusionInstruction& fusion);

// Returns the safe, clamped workgroup count in `[1, max(1, num_updates)]` for
// a CPU scatter fusion, enforcing identical eligibility rules in
// `ParallelTaskAssigner`, kernel cache fingerprinting, and `CpuScatterFusion`
// (variadic uniqueness, packed sub byte safety, native 64 bit and 128 bit
// atomic lowering capabilities, buffer slice alignment, producer/output buffer
// aliasing, determinism, and degenerate shapes). Returns 1 whenever parallel
// emission is unsupported or unsafe.
int64_t GetCpuScatterEffectiveWorkGroups(
    const HloFusionInstruction& fusion, int64_t requested_workgroups,
    const CpuDeviceSpec& cpu_device_spec,
    const BufferAssignment* buffer_assignment = nullptr,
    int64_t default_buffer_align_bytes = MinAlign());

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_IR_EMISSION_UTILS_H_
