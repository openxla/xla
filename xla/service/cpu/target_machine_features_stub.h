/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_TARGET_MACHINE_FEATURES_STUB_H_
#define XLA_SERVICE_CPU_TARGET_MACHINE_FEATURES_STUB_H_

#include <cstdint>
#include <functional>
#include <string>
#include <utility>

#include "llvm/IR/Function.h"
#include "tsl/platform/logging.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/codegen/device_spec.h"

namespace xla::cpu {

// Delegates calls to minimum_alignment_for_allocation to a user provided
// std::function, crashes on all other methods.
class TargetMachineFeaturesStub : public TargetMachineFeatures {
 public:
  explicit TargetMachineFeaturesStub(
      std::function<int64_t(int64_t)> min_alignment,
      int32_t vector_register_byte_size = kX86AvxVectorByteSize,
      std::string enabled_target_features = "",
      CpuDeviceSpec cpu_device_spec = {})
      : TargetMachineFeatures(/*target_machine=*/nullptr),
        min_alignment_(std::move(min_alignment)),
        vector_register_byte_size_(vector_register_byte_size),
        enabled_target_features_(std::move(enabled_target_features)),
        cpu_device_spec_(cpu_device_spec) {}

  int vectorization_factor_in_bytes() const final {
    LOG(FATAL) << "Unexpected call to " << __func__;
  }

  int32_t vector_register_byte_size() const final {
    return vector_register_byte_size_;
  }

  int vector_register_byte_size(const llvm::Function& function) const final {
    LOG(FATAL) << "Unexpected call to " << __func__;
  }

  int vector_register_num_elements(const llvm::Function& function,
                                   PrimitiveType type) const final {
    LOG(FATAL) << "Unexpected call to " << __func__;
  }

  int vector_register_count(const llvm::Function& function) const final {
    LOG(FATAL) << "Unexpected call to " << __func__;
  }

  int64_t minimum_alignment_for_allocation(int64_t size_bytes) const final {
    return min_alignment_(size_bytes);
  }

  std::string get_target_feature_string() const final {
    LOG(FATAL) << "Unexpected call to " << __func__;
  }

  std::string get_enabled_target_features() const final {
    return enabled_target_features_;
  }

  CpuDeviceSpec cpu_device_spec() const final { return cpu_device_spec_; }

  void set_cpu_device_spec(CpuDeviceSpec cpu_device_spec) {
    cpu_device_spec_ = cpu_device_spec;
  }

  const llvm::TargetMachine* target_machine() const final {
    LOG(FATAL) << "Unexpected call to " << __func__;
  }

 private:
  std::function<int64_t(int64_t)> min_alignment_;
  int32_t vector_register_byte_size_;
  std::string enabled_target_features_;
  CpuDeviceSpec cpu_device_spec_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_TARGET_MACHINE_FEATURES_STUB_H_
