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

#include "xla/backends/gpu/libraries/cutedsl/kernel_call.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/libraries/cutedsl/ffi_abi.h"
#include "xla/backends/gpu/libraries/cutedsl/module.h"
#include "xla/ffi/ffi.h"

namespace xla::gpu::cutedsl {

namespace {

constexpr absl::string_view kFunctionPrefix = "cutlass_call";

absl::Status RunKernel(const LoadedKernel& kernel,
                       absl::Span<void* const> leading_pointer_values,
                       absl::InlinedVector<CuteXlaFfiBuffer, 8>& buffers,
                       absl::string_view operation) {
  // Generated wrappers use MLIR's packed C interface: every pointer-valued
  // entry points to storage containing the corresponding argument value.
  absl::InlinedVector<void*, 10> pointer_values;
  pointer_values.reserve(leading_pointer_values.size() + buffers.size());
  pointer_values.insert(pointer_values.end(), leading_pointer_values.begin(),
                        leading_pointer_values.end());
  for (CuteXlaFfiBuffer& buffer : buffers) pointer_values.push_back(&buffer);

  absl::InlinedVector<void*, 11> packed_arguments;
  packed_arguments.reserve(pointer_values.size() + 1);
  for (void*& pointer_value : pointer_values) {
    packed_arguments.push_back(&pointer_value);
  }

  // Generated wrappers write the CUDA launch status to the final argument.
  // CUDA error enums use a 32-bit representation, avoiding a dependency on
  // the CUDA runtime ABI here.
  int32_t cuda_error = 0;
  packed_arguments.push_back(&cuda_error);

  absl::Status run_status =
      kernel.Run(packed_arguments.data(), packed_arguments.size());
  if (!run_status.ok()) {
    return absl::InternalError(
        absl::StrFormat("Failed to execute %s: %s; CUDA error %d", operation,
                        run_status.message(), cuda_error));
  }
  if (cuda_error != 0) {
    return absl::InternalError(
        absl::StrFormat("%s returned CUDA error %d", operation, cuda_error));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status LoadedKernel::Run(void** arguments, size_t argument_count) const {
  return module_->Run(function_, arguments, argument_count);
}

absl::StatusOr<LoadedKernel> KernelCallState::Prepare() {
  {
    absl::MutexLock lock(&mutex_);
    if (kernel_.has_value()) return *kernel_;
  }

  ASSIGN_OR_RETURN(std::shared_ptr<LoadedModule> module,
                   loader_.GetOrLoad(image_));
  ASSIGN_OR_RETURN(LoadedModule::FunctionHandle function,
                   module->GetFunction(kFunctionPrefix));
  LoadedKernel loaded(std::move(module), function);

  absl::MutexLock lock(&mutex_);
  if (!kernel_.has_value()) kernel_.emplace(std::move(loaded));
  return *kernel_;
}

absl::StatusOr<LoadedKernel> KernelCallState::prepared_kernel() const {
  absl::MutexLock lock(&mutex_);
  if (!kernel_.has_value()) {
    return absl::FailedPreconditionError(
        "CuTeDSL custom call executed before prepare completed");
  }
  return *kernel_;
}

absl::StatusOr<KernelArguments> KernelArguments::Create(
    ffi::RemainingArgs arguments, ffi::RemainingRets results) {
  KernelArguments kernel_arguments;
  kernel_arguments.buffers_.reserve(arguments.size() + results.size());

  for (size_t i = 0; i < arguments.size(); ++i) {
    ASSIGN_OR_RETURN(ffi::AnyBuffer argument, arguments.get<ffi::AnyBuffer>(i));
    ffi::AnyBuffer::Dimensions dimensions = argument.dimensions();
    kernel_arguments.buffers_.push_back(
        {argument.untyped_data(),
         dimensions.empty() ? nullptr : dimensions.data()});
  }
  for (size_t i = 0; i < results.size(); ++i) {
    ASSIGN_OR_RETURN(ffi::Result<ffi::AnyBuffer> result,
                     results.get<ffi::AnyBuffer>(i));
    ffi::AnyBuffer::Dimensions dimensions = result->dimensions();
    kernel_arguments.buffers_.push_back(
        {result->untyped_data(),
         dimensions.empty() ? nullptr : dimensions.data()});
  }
  return kernel_arguments;
}

absl::Status KernelArguments::Run(const LoadedKernel& kernel, void* stream) {
  std::array<void*, 1> leading_pointer_values = {stream};
  return RunKernel(kernel, absl::MakeConstSpan(leading_pointer_values),
                   buffers_, "CuTeDSL kernel");
}

absl::Status KernelArguments::Run(const LoadedKernel& kernel, void* stream,
                                  CollectiveContextAbi& collective_context) {
  std::array<void*, 2> leading_pointer_values = {stream, &collective_context};
  return RunKernel(kernel, absl::MakeConstSpan(leading_pointer_values),
                   buffers_, "CuTeDSL collective launch");
}

}  // namespace xla::gpu::cutedsl
