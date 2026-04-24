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
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/executable.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/util/sorted_range.h"

namespace xla {
namespace cpu {

namespace {

struct MultiModuleContext {
  const absl::flat_hash_map<std::string, std::unique_ptr<Executable>>*
      sub_modules;
  const ServiceExecutableRunOptions* run_options;
};

// This thread_local context is used to pass the sub-module registry and run
// options to the custom call handler.
// POTENTIAL BUG: This will not be visible to worker threads if XLA's
// thread pool is used for intra-op parallelism within the custom call bridge.
// A more robust mechanism to propagate the context or pass it through the
// custom call's opaque data should be considered if this becomes an issue.
thread_local MultiModuleContext* g_multi_module_context = nullptr;

void MultiModuleCall(void* out, const void** in, const char* opaque,
                     size_t opaque_len, XlaCustomCallStatus* status) {
  if (g_multi_module_context == nullptr) {
    const char* msg = "Multi-module context not set";
    XlaCustomCallStatusSetFailure(status, msg, std::strlen(msg));
    return;
  }

  // The sub-module name is passed via the `opaque` field of the custom call.
  // While XLA often uses `BackendConfig` for structured metadata, using
  // `opaque` as a raw byte buffer for the name is acceptable here. The
  // `opaque_len` ensures that the `absl::string_view` is correctly formed,
  // avoiding any field mismatch issues related to null termination.
  absl::string_view sub_module_name(opaque, opaque_len);
  auto it = g_multi_module_context->sub_modules->find(sub_module_name);
  if (it == g_multi_module_context->sub_modules->end()) {
    std::string msg = absl::StrCat("Sub-module not found: ", sub_module_name);
    XlaCustomCallStatusSetFailure(status, msg.c_str(), msg.size());
    return;
  }

  Executable* sub_executable = it->second.get();
  const Shape& sub_result_shape = sub_executable->result_shape();

  std::vector<ExecutionInput> sub_arguments;
  if (!sub_executable->has_module()) {
    const char* msg = "Sub-executable does not have a module.";
    XlaCustomCallStatusSetFailure(status, msg, std::strlen(msg));
    return;
  }
  const HloComputation* entry_comp =
      sub_executable->module().entry_computation();
  sub_arguments.reserve(entry_comp->num_parameters());

  auto set_input_buffers = [&](auto self, const Shape& shape,
                               const ShapeIndex& index, void* src,
                               ExecutionInput& input) -> void {
    input.SetBuffer(index,
                    MaybeOwningDeviceAddress(se::DeviceAddressBase(
                        src, ShapeUtil::ByteSizeOf(
                                 shape, /*pointer_size=*/sizeof(void*)))));
    if (shape.IsTuple()) {
      void** src_tuple = static_cast<void**>(src);
      for (int i = 0; i < shape.tuple_shapes_size(); ++i) {
        ShapeIndex child_index = index;
        child_index.push_back(i);
        self(self, shape.tuple_shapes(i), child_index, src_tuple[i], input);
      }
    }
  };

  for (int i = 0; i < entry_comp->num_parameters(); ++i) {
    const Shape& shape = entry_comp->parameter_instruction(i)->shape();
    ExecutionInput input(shape);
    // NOLINTNEXTLINE
    set_input_buffers(set_input_buffers, shape, {}, const_cast<void*>(in[i]),
                      input);
    sub_arguments.push_back(std::move(input));
  }

  auto result_or = sub_executable->ExecuteAsyncOnStream(
      g_multi_module_context->run_options, std::move(sub_arguments));
  if (!result_or.ok()) {
    std::string msg = absl::StrCat("Sub-module execution failed: ",
                                   result_or.status().message());
    XlaCustomCallStatusSetFailure(status, msg.c_str(), msg.size());
    return;
  }

  ExecutionOutput result = std::move(result_or).value();
  const ScopedShapedBuffer& result_buffer = result.Result();

  auto copy_result = [&](auto self, const Shape& shape, const ShapeIndex& index,
                         void* dest) -> void {
    if (shape.IsTuple()) {
      void** dest_tuple = static_cast<void**>(dest);
      for (int i = 0; i < shape.tuple_shapes_size(); ++i) {
        ShapeIndex child_index = index;
        child_index.push_back(i);
        self(self, shape.tuple_shapes(i), child_index, dest_tuple[i]);
      }
    } else {
      const se::DeviceAddressBase& src_addr = result_buffer.buffer(index);
      if (src_addr.opaque() != nullptr && dest != nullptr) {
        std::memcpy(dest, src_addr.opaque(), src_addr.size());
      }
    }
  };

  copy_result(copy_result, sub_result_shape, {}, out);
}

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("__xla_cpu_multi_module_call",
                                             MultiModuleCall);

}  // namespace

MultiModuleCpuExecutable::MultiModuleCpuExecutable(
    std::unique_ptr<Executable> main_executable,
    absl::flat_hash_map<std::string, std::unique_ptr<Executable>> sub_modules)
    : Executable(main_executable->shared_module()),
      main_executable_(std::move(main_executable)),
      sub_modules_(std::move(sub_modules)) {}

absl::StatusOr<ExecutionOutput> MultiModuleCpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments) {
  MultiModuleContext ctx{&sub_modules_, run_options};
  MultiModuleContext* old_ctx = g_multi_module_context;
  g_multi_module_context = &ctx;
  absl::Cleanup cleanup = [old_ctx] { g_multi_module_context = old_ctx; };

  return main_executable_->ExecuteAsyncOnStream(run_options,
                                                std::move(arguments));
}

absl::StatusOr<ScopedShapedBuffer>
MultiModuleCpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<const ShapedBuffer* const> arguments) {
  MultiModuleContext ctx{&sub_modules_, run_options};
  MultiModuleContext* old_ctx = g_multi_module_context;
  g_multi_module_context = &ctx;
  absl::Cleanup cleanup = [old_ctx] { g_multi_module_context = old_ctx; };

  return main_executable_->ExecuteAsyncOnStream(run_options, arguments);
}

int64_t MultiModuleCpuExecutable::SizeOfGeneratedCodeInBytes() const {
  int64_t size = main_executable_->SizeOfGeneratedCodeInBytes();
  for (const auto& [name, sub_module] : tsl::SortedRange(sub_modules_)) {
    size += sub_module->SizeOfGeneratedCodeInBytes();
  }
  return size;
}

}  // namespace cpu
}  // namespace xla
