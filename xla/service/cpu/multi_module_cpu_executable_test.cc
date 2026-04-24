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

#include "xla/service/cpu/multi_module_cpu_executable.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace cpu {
namespace {

// MockAllocator simulates a device memory allocator on the host.
// It uses std::malloc and std::free to allocate and deallocate memory.
class MockAllocator : public se::DeviceMemoryAllocator {
 public:
  MockAllocator() : se::DeviceMemoryAllocator(nullptr) {}

  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override {
    if (size == 0) {
      return se::ScopedDeviceAddress<uint8_t>();
    }
    void* ptr = std::malloc(size);
    return se::ScopedDeviceAddress<uint8_t>(se::DeviceAddressBase(ptr, size),
                                            device_ordinal, this);
  }

  absl::Status Deallocate(int device_ordinal,
                          se::DeviceAddressBase mem) override {
    if (!mem.is_null()) {
      std::free(mem.opaque());
    }
    return absl::OkStatus();
  }

  absl::StatusOr<se::Stream*> GetStream(int device_ordinal) override {
    return absl::UnimplementedError("Not implemented");
  }
};

// DummyExecutable simulates a compiled sub-module executable.
// In its ExecuteAsyncOnStream implementation, it simulates execution by adding
// 42 to the input value and recording that it was called.
class DummyExecutable : public Executable {
 public:
  explicit DummyExecutable(std::shared_ptr<HloModule> module = nullptr)
      : Executable(std::move(module)) {}

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override {
    execute_called_ = true;
    last_arguments_ = std::move(arguments);

    Shape shape = result_shape();
    if (run_options->allocator() == nullptr) {
      return absl::InternalError("Allocator is null");
    }

    ExecutionOutput result(shape, run_options->allocator(), 0);
    ShapeUtil::ForEachSubshape(shape, [&](const Shape& subshape,
                                          const ShapeIndex& index) {
      if (subshape.IsTuple()) {
        return;
      }
      auto buffer_or = run_options->allocator()->Allocate(
          0, ShapeUtil::ByteSizeOf(subshape, 8));
      if (buffer_or.ok()) {
        result.MutableResult()->set_buffer(std::move(buffer_or).value(), index);
      }
    });

    float val = 42.0f;
    if (!last_arguments_.empty()) {
      if (last_arguments_[0].shape().IsTuple()) {
        const auto& leaf_buffer = last_arguments_[0].Buffer({0});
        if (!leaf_buffer.AsDeviceAddress().is_null()) {
          float in_val = 0.0f;
          std::memcpy(&in_val, leaf_buffer.AsDeviceAddress().opaque(),
                      sizeof(float));
          val += in_val;
        }
      } else {
        const auto& buffer = last_arguments_[0].Buffer({});
        if (!buffer.AsDeviceAddress().is_null()) {
          float in_val = 0.0f;
          std::memcpy(&in_val, buffer.AsDeviceAddress().opaque(),
                      sizeof(float));
          val += in_val;
        }
      }
    }

    std::memcpy(result.MutableResult()->buffer({}).opaque(), &val,
                sizeof(float));

    return std::move(result);
  }

  absl::StatusOr<ScopedShapedBuffer> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<const ShapedBuffer* const> arguments) override {
    return absl::UnimplementedError("Not implemented");
  }

  Shape result_shape() const override { return ShapeUtil::MakeShape(F32, {1}); }

  bool execute_called_ = false;
  std::vector<ExecutionInput> last_arguments_;
};

// MockMainExecutable simulates the main module executable.
// In its ExecuteAsyncOnStream implementation, it manually invokes the
// custom call handler "__xla_cpu_multi_module_call" to simulate the bridge
// routing a call to a sub-module named "sub".
class MockMainExecutable : public Executable {
 public:
  MockMainExecutable() : Executable(nullptr) {}

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override {
    auto* registry = CustomCallTargetRegistry::Global();
    void* handler = registry->Lookup("__xla_cpu_multi_module_call", "Host");
    if (handler == nullptr) {
      return absl::InternalError("Custom call handler not found");
    }
    using handler_type = void (*)(void*, const void**, const char*, size_t,
                                  XlaCustomCallStatus*);
    auto* fn = reinterpret_cast<handler_type>(handler);

    float in_val = 10.0f;
    const void* in_ptr = &in_val;
    const void* in[] = {&in_ptr};
    float out_val = 0.0f;
    XlaCustomCallStatus status;
    fn(&out_val, in, "sub", 3, &status);

    custom_call_result_ = out_val;

    Shape shape = ShapeUtil::MakeShape(F32, {1});
    ExecutionOutput result(shape, run_options->allocator(), 0);
    TF_ASSIGN_OR_RETURN(auto buffer, run_options->allocator()->Allocate(
                                         0, ShapeUtil::ByteSizeOf(shape, 8)));
    result.MutableResult()->set_buffer(std::move(buffer), {});
    return std::move(result);
  }

  absl::StatusOr<ScopedShapedBuffer> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<const ShapedBuffer* const> arguments) override {
    return absl::UnimplementedError("Not implemented");
  }

  float custom_call_result_ = 0.0f;
};

// This test verifies that the MultiModuleCpuExecutable correctly initializes
// the custom call bridge and routes calls to the appropriate sub-module.
// It sets up a main executable that triggers the custom call, and a dummy
// sub-executable that simulates execution.
TEST(MultiModuleCpuExecutableTest, CustomCallBridgeWithTuple) {
  auto main_exec = std::make_unique<MockMainExecutable>();
  MockMainExecutable* main_exec_ptr = main_exec.get();

  HloModuleConfig config;
  auto sub_module = std::make_unique<HloModule>("sub", config);
  HloComputation::Builder builder("entry");
  builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {1})}), "p0"));
  sub_module->AddEntryComputation(builder.Build());

  auto sub_exec = std::make_unique<DummyExecutable>(std::move(sub_module));
  DummyExecutable* sub_exec_ptr = sub_exec.get();

  absl::flat_hash_map<std::string, std::unique_ptr<Executable>> sub_modules;
  sub_modules["sub"] = std::move(sub_exec);

  MultiModuleCpuExecutable multi_exec(std::move(main_exec),
                                      std::move(sub_modules));

  MockAllocator allocator;
  ExecutableRunOptions erun_options;
  erun_options.set_allocator(&allocator);
  ServiceExecutableRunOptions run_options(erun_options);

  std::vector<ExecutionInput> args;
  auto result_status =
      multi_exec.ExecuteAsyncOnStream(&run_options, std::move(args));
  TF_ASSERT_OK(result_status.status());

  EXPECT_TRUE(sub_exec_ptr->execute_called_);
  EXPECT_EQ(main_exec_ptr->custom_call_result_, 52.0f);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
