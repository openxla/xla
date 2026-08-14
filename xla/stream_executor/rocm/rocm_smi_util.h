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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_SMI_UTIL_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_SMI_UTIL_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "rocm/rocm_config.h"

// The matching link time switch, and the rationale for the 7.13 boundary, live
// in third_party/gpus/rocm/BUILD.tpl.
#if (TF_ROCM_VERSION >= 71300)
#include "rocm/include/amd_smi/amdsmi.h"
#else
#include "rocm/include/rocm_smi/rocm_smi.h"
#endif  // TF_ROCM_VERSION >= 71300

namespace stream_executor::gpu {

struct BdfComponents {
  uint64_t domain;
  uint64_t bus;
  uint64_t device;
  uint64_t function;
};

#if (TF_ROCM_VERSION >= 71300)
// amd_smi identifies a GPU by an opaque processor handle.
using SmiDeviceHandle = amdsmi_processor_handle;
// Name of the SMI library this build talks to. Included in every SMI log
// message so it is obvious which of the two paths produced it.
inline constexpr absl::string_view kSmiLibraryName = "amd_smi";
#else
// rocm_smi identifies a GPU by a dense monitor device index.
using SmiDeviceHandle = uint32_t;
inline constexpr absl::string_view kSmiLibraryName = "rocm_smi";
#endif  // TF_ROCM_VERSION >= 71300

// Process-global lock serializing all SMI access from XLA. rocm_smi only
// guards state with a per-device mutex, but some of it is global (e.g. the
// shared gpu_metrics object), so concurrent queries on different devices race.
// amd_smi embeds rocm_smi and inherits the same problem. Hold this across each
// full SMI call sequence.
ABSL_CONST_INIT extern absl::Mutex rocm_smi_mutex;

// Returns true if the SMI library was successfully initialized.
bool InitRocmSmi();

// Parses a PCI bus ID string (e.g., "0000:41:00.0") into its BDF components.
// Returns std::nullopt on parse failure.
std::optional<BdfComponents> ParseBdf(absl::string_view pci_bus_id);

// Returns handles for every SMI visible GPU, in SMI enumeration order. Empty
// on failure. Callers must hold rocm_smi_mutex.
std::vector<SmiDeviceHandle> EnumerateDevices();

// Finds the SMI device that matches the given PCI bus ID.
// Returns std::nullopt if not found. Callers must hold rocm_smi_mutex.
std::optional<SmiDeviceHandle> FindDeviceIndex(const BdfComponents& target_bdf);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_SMI_UTIL_H_
