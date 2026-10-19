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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_HOIST_FUSION_BOUNDARY_BITCASTS_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_HOIST_FUSION_BOUNDARY_BITCASTS_H_

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/gpu/mlir_context_pool.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::gpu {

// Moves bitcast instructions at the parameter and root boundaries of fusion
// computations outside the fusion, and recalculates optimal tiling for modified
// tiled Triton fusions.
class HoistFusionBoundaryBitcasts : public HloModulePass {
 public:
  explicit HoistFusionBoundaryBitcasts(
      const se::DeviceDescription& device_info,
      HloCostAnalysis::ShapeSizeFunction shape_size,
      mlir::MLIRContext* mlir_context,
      tsl::thread::ThreadPool* thread_pool = nullptr,
      MlirContextPool* mlir_context_pool = nullptr)
      : device_info_(device_info),
        shape_size_(std::move(shape_size)),
        mlir_context_(mlir_context),
        thread_pool_(thread_pool),
        mlir_context_pool_(mlir_context_pool) {}

  absl::string_view name() const override {
    return "hoist-fusion-boundary-bitcasts";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::DeviceDescription device_info_;
  HloCostAnalysis::ShapeSizeFunction shape_size_;
  mlir::MLIRContext* mlir_context_;
  tsl::thread::ThreadPool* thread_pool_;
  MlirContextPool* mlir_context_pool_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_HOIST_FUSION_BOUNDARY_BITCASTS_H_
