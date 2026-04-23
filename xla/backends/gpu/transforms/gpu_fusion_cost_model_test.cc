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

#include "xla/backends/gpu/transforms/gpu_fusion_cost_model.h"

#include <memory>

#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {
namespace {

using GpuFusionCostModelTest = HloHardwareIndependentTestBase;

TEST_F(GpuFusionCostModelTest, CanInstantiate) {
  se::DeviceDescription device_info = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  GpuHloCostAnalysis::Options options;
  mlir::MLIRContext mlir_context;

  GpuFusionCostModel cost_model(device_info, options, &mlir_context);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
