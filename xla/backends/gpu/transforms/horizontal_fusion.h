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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_HORIZONTAL_FUSION_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_HORIZONTAL_FUSION_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Groups consecutive independent small-grid fusion instructions in a scheduled
// HLO module into an horizontal fusions whose fused computation contains nested
// fusions and a root tuple of their outputs.
class HorizontalFusion : public HloModulePass {
 public:
  struct Options {
    // Maximum number of resident waves of the resulting horizontal fusion:
    // `total_blocks * max_threads_per_block <= max_waves *
    // device_info.core_count() * resident_threads_per_sm`, where
    // `resident_threads_per_sm` is a conservative, register-limited estimate.
    // The default of a single wave guarantees that all nested kernels run
    // concurrently; a second wave would serialize part of the grid, and its
    // tail latency can exceed the launch overhead saved by fusing.
    int64_t max_waves = 1;
    // Maximum number of nested kernels in a single horizontal fusion.
    int64_t max_nested_kernels = 8;
    // Maximum estimated execution time (excluding launch overhead) of any
    // single leaf fusion included in a horizontal fusion. Zero disables the
    // check.
    //
    // Horizontal fusion saves one kernel launch (~1 us of GPU-side launch
    // latency, `GpuPerformanceModelBase::kKernelLaunchOverhead`, plus a few us
    // of host-side launch cost when not captured in a command buffer) per fused
    // leaf, so it only pays off for kernels of comparable duration. For longer
    // leaves the saving is negligible, while the fused kernel inherits the
    // largest register and shared-memory footprint of all leaves, which can
    // slow down the other leaves. 5 us is a few launch overheads: large enough
    // to cover the tiny launch-bound kernels in decode steps, small enough to
    // exclude bandwidth- or compute-bound kernels.
    absl::Duration max_leaf_exec_time = absl::Microseconds(5);
    // Maximum output size in bytes of an isolated fusion that may be batched
    // across intervening independent non-fusible instructions (e.g. GEMMs or
    // custom calls) into an earlier horizontal fusion. The fusion is then
    // emitted at the earlier position, so it must not depend on anything
    // scheduled in between. Zero disables cross-barrier batching.
    //
    // Moving a fusion earlier extends the live range of its outputs across the
    // skipped instructions. 2 KiB (e.g. 512 f32 elements, i.e. a single block
    // of a few hundred threads) covers the tiny scalar, step-counter, metadata
    // and RNG-state fusions that are scattered across layers of decode steps,
    // while the extra live memory stays negligible (at most 8 open batches of 8
    // such outputs pin ~128 KiB). Larger outputs would keep proportionally more
    // memory alive across potentially long sequences of GEMMs, so they are
    // left in place.
    int64_t max_cross_barrier_output_bytes = 2048;
  };

  explicit HorizontalFusion(const se::DeviceDescription& device_info)
      : HorizontalFusion(device_info, Options{}) {}

  HorizontalFusion(const se::DeviceDescription& device_info,
                   const Options& options)
      : device_info_(device_info), options_(options) {}

  absl::string_view name() const override { return "horizontal-fusion"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::DeviceDescription device_info_;
  Options options_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_HORIZONTAL_FUSION_H_
