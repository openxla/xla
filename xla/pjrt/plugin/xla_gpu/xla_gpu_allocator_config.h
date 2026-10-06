/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PJRT_PLUGIN_XLA_GPU_XLA_GPU_ALLOCATOR_CONFIG_H_
#define XLA_PJRT_PLUGIN_XLA_GPU_XLA_GPU_ALLOCATOR_CONFIG_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/framework/allocator.h"

namespace xla {

// How much of a device's memory the BFC allocator may use, as fractions of
// total device memory. ParseMemFraction accepts the grammar of JAX's
// XLA_PYTHON_CLIENT_MEM_FRACTION:
//
//   "0.75"       FlexMemFraction{0.75, 1.0}: preallocate 75%, grow to all
//   "0.75-0.85"  FlexMemFraction{0.75, 0.85}: preallocate 75%, grow to 85%
//   "0.75-0.75"  FixedMemFraction{0.75}: hard cap, the pool never grows
//
// Only the shared spatial BFC pool (preallocate=true with
// --xla_gpu_enable_allocator_spatial_partitioning, on device memory) grows.
// Every other allocator kind and mode uses just the start fraction.

// The pool never grows; `fraction` is a hard cap. With unified memory the
// fraction may exceed 1 to oversubscribe device memory.
struct FixedMemFraction {
  double fraction = 0.75;
};

// Preallocate `start` and let default-memory allocations grow the pool up to
// `cap`. Invariant: 0 < start < cap <= 1.
struct FlexMemFraction {
  double start = 0.75;
  double cap = 1.0;
};

using MemFraction = std::variant<FixedMemFraction, FlexMemFraction>;

// Parses "START" or "START-CAP" as described above. Returns InvalidArgument
// for malformed input, a cap below the start, or a growth cap above 1.
absl::StatusOr<MemFraction> ParseMemFraction(absl::string_view spec);

// Policy denoted by a bare fraction: growable to all device memory when
// `fraction` is below 1, otherwise fixed (there is nothing to grow into).
MemFraction MemFractionFromFraction(double fraction);

// Fraction allocated up front: FixedMemFraction::fraction or
// FlexMemFraction::start.
double MemFractionStart(const MemFraction& fraction);

// Canonical spelling accepted by ParseMemFraction.
std::string MemFractionToString(const MemFraction& fraction);

struct GpuAllocatorConfig {
  enum class Kind {
    kDefault,   // Client picks the best option for the platform.
    kPlatform,  // Synchronous passthrough allocator that calls the
                // StreamExecutor Allocate/Deallocate APIs directly via a
                // MultiDeviceAdapter wrapping StreamExecutorAllocator
                // instances, with no BFC caching or pooled growth.
    kBFC,  // Allocator using a "Best-Fit with Coalescing" algorithm. Currently
           // only available for GPU.
    kCudaAsync,  // Use the CUDA async allocator.
    kVmm,  // Use Virtual Memory Management (VMM) allocator. This allocator
           // uses CUDA VMM APIs to manage virtual address space separately from
           // physical memory, enabling features like memory oversubscription
           // and fine-grained memory mapping control.
  };
  Kind kind = Kind::kDefault;

  // Only used if kind == kBFC (or kDefault). How much of total device memory
  // the allocator may use; see MemFraction above. The default preallocates 75%
  // and lets the shared spatial pool grow to all of device memory. This is the
  // default value of XLA_CLIENT_MEM_FRACTION.
  //
  // If `gpu_system_memory_size` is set, it replaces the start fraction as the
  // initial allocation; a growth cap from `memory_fraction` still applies.
  // Other allocator kinds use only the start fraction (MemFractionStart).
  //
  // PJRT C API create options: "memory_fraction" (float, a bare fraction) and
  // "memory_fraction_policy" (string, the full grammar; takes precedence).
  // --xla_gpu_memory_fraction_policy overrides both for the BFC allocator, and
  // --xla_gpu_enable_nccl_user_buffers_in_default_space pins a growable policy
  // to a fixed pool because automatic registration needs a fixed arena.
  MemFraction memory_fraction = FlexMemFraction{};

  // Only used if kind == kBFC. The absolute size of reserved memory space for
  // GPU system in bytes.
  //
  // If null, the default value `memory_fraction` will be used.
  std::optional<int64_t> gpu_system_memory_size = std::nullopt;

  // Only used if kind == kBFC. If true, the allocator will immediately allocate
  // the maximum amount allowed by `memory_fraction`. This reduces
  // fragmentation, allowing more of the total memory to be used. If false, the
  // allocator will allocate more memory as allocations are requested.
  bool preallocate = true;

  // Amount of collective memory (ncclMemAlloc) to preallocate. If this value is
  // 0, collective memory space will be grown as needed to fit the application's
  // usage, with the drawback of potentially higher fragmentation. If set,
  // should be set to a multiple of 512MB to avoid wasting memory due to
  // granularity requirements.
  size_t collective_memory_size = 0;

  // Callbacks that get called when the underlying suballocator allocates or
  // deallocates memory. See `SubAllocator::Visitor` for more details.
  std::vector<tsl::SubAllocator::Visitor> sub_allocator_alloc_visitors;
  std::vector<tsl::SubAllocator::Visitor> sub_allocator_free_visitors;
};

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XLA_GPU_XLA_GPU_ALLOCATOR_CONFIG_H_
