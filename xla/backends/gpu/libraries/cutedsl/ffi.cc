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

#include "xla/backends/gpu/ffi.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/libraries/cutedsl/ffi_abi.h"
#include "xla/backends/gpu/libraries/cutedsl/module.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"

namespace xla::gpu::cutedsl {

namespace ffi = ::xla::ffi;

namespace {

constexpr absl::string_view kCutlassCallTarget = "__xla_gpu_cutedsl_call_v3";
constexpr absl::string_view kCutlassCallNoCudaGraphTarget =
    "__xla_gpu_cutedsl_call_no_cuda_graph_v3";
constexpr absl::string_view kFunctionPrefix = "cutlass_call";
// Avoid heap allocation for the common case while retaining support for calls
// with an arbitrary number of buffers.
constexpr size_t kInlineBufferCount = 8;

class CutlassCallStateV3 {
 public:
  struct ModuleAndFunction {
    std::shared_ptr<LoadedModule> module;
    LoadedModule::FunctionHandle function = nullptr;
  };

  explicit CutlassCallStateV3(ModuleImage image) : image_(std::move(image)) {}

  absl::Status LoadModule() {
    {
      absl::MutexLock lock(&mu_);
      if (module_.module != nullptr) return absl::OkStatus();
    }

    ASSIGN_OR_RETURN(std::shared_ptr<LoadedModule> loaded,
                     loader_.GetOrLoad(image_));
    ASSIGN_OR_RETURN(LoadedModule::FunctionHandle function,
                     loaded->GetFunction(kFunctionPrefix));

    absl::MutexLock lock(&mu_);
    if (module_.module == nullptr) {
      module_ = {std::move(loaded), function};
    }
    return absl::OkStatus();
  }

  absl::StatusOr<ModuleAndFunction> GetModule() const {
    absl::MutexLock lock(&mu_);
    if (module_.module == nullptr) {
      return absl::FailedPreconditionError(
          "CuTeDSL custom call executed before prepare completed");
    }
    return module_;
  }

 private:
  ModuleImage image_;
  ModuleLoader loader_;
  mutable absl::Mutex mu_;
  ModuleAndFunction module_ ABSL_GUARDED_BY(mu_);
};

absl::StatusOr<std::unique_ptr<CutlassCallStateV3>> Instantiate(
    absl::string_view module, absl::string_view key) {
  ASSIGN_OR_RETURN(ModuleImage image, ModuleImage::Create(module, key));
  // Runtime access remains deferred to Prepare; Instantiate only validates
  // device-independent attributes.
  return std::make_unique<CutlassCallStateV3>(std::move(image));
}

absl::Status Prepare(CutlassCallStateV3* state, ffi::RemainingArgs,
                     ffi::RemainingRets, absl::string_view, absl::string_view) {
  return state->LoadModule();
}

absl::Status Initialize() { return absl::OkStatus(); }

absl::Status ExecuteFunction(const LoadedModule& module, void* stream,
                             LoadedModule::FunctionHandle function,
                             ffi::RemainingArgs inputs,
                             ffi::RemainingRets outputs) {
  absl::InlinedVector<CuteXlaFfiBuffer, kInlineBufferCount> buffers;
  buffers.reserve(inputs.size() + outputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    absl::StatusOr<ffi::AnyBuffer> input = inputs.get<ffi::AnyBuffer>(i);
    if (!input.ok()) return input.status();
    ffi::AnyBuffer::Dimensions dimensions = input->dimensions();
    buffers.push_back({input->untyped_data(),
                       dimensions.empty() ? nullptr : dimensions.data()});
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    absl::StatusOr<ffi::Result<ffi::AnyBuffer>> output =
        outputs.get<ffi::AnyBuffer>(i);
    if (!output.ok()) return output.status();
    ffi::AnyBuffer::Dimensions dimensions = (*output)->dimensions();
    buffers.push_back({(*output)->untyped_data(),
                       dimensions.empty() ? nullptr : dimensions.data()});
  }

  // CuTeDSL compiled wrappers use MLIR's packed C interface: every entry is a
  // pointer to storage containing the corresponding argument value.
  absl::InlinedVector<void*, kInlineBufferCount + 1> arguments;
  arguments.reserve(buffers.size() + 1);
  arguments.push_back(stream);
  for (CuteXlaFfiBuffer& buffer : buffers) arguments.push_back(&buffer);

  absl::InlinedVector<void*, kInlineBufferCount + 2> packed_arguments;
  packed_arguments.reserve(arguments.size() + 1);
  for (void*& argument : arguments) packed_arguments.push_back(&argument);

  // Generated wrappers write the CUDA launch status to the final argument.
  // CUDA error enums use a 32-bit representation; keeping the ABI type here
  // avoids a link-time dependency on the CUDA runtime.
  int32_t cuda_error = 0;
  packed_arguments.push_back(&cuda_error);

  absl::Status run_status =
      module.Run(function, packed_arguments.data(), packed_arguments.size());
  if (!run_status.ok()) {
    return absl::InternalError(
        absl::StrFormat("Failed to execute CuTeDSL kernel: %s; CUDA error %d",
                        run_status.message(), cuda_error));
  }
  if (cuda_error != 0) {
    return absl::InternalError(
        absl::StrFormat("CuTeDSL kernel returned CUDA error %d", cuda_error));
  }

  return absl::OkStatus();
}

absl::Status Execute(void* stream, CutlassCallStateV3* state,
                     ffi::RemainingArgs inputs, ffi::RemainingRets outputs,
                     absl::string_view, absl::string_view) {
  absl::StatusOr<CutlassCallStateV3::ModuleAndFunction> module =
      state->GetModule();
  if (!module.ok()) return module.status();

  return ExecuteFunction(*module->module, stream, module->function, inputs,
                         outputs);
}

XLA_FFI_DEFINE_HANDLER(kInstantiate, Instantiate,
                       ffi::Ffi::BindInstantiate()
                           .Attr<absl::string_view>("module")
                           .Attr<absl::string_view>("key"));

XLA_FFI_DEFINE_HANDLER(kPrepare, Prepare,
                       ffi::Ffi::BindPrepare()
                           .Ctx<ffi::State<CutlassCallStateV3>>()
                           .RemainingArgs()
                           .RemainingRets()
                           .Attr<absl::string_view>("module")
                           .Attr<absl::string_view>("key"));

XLA_FFI_DEFINE_HANDLER(kInitialize, Initialize, ffi::Ffi::BindInitialize());

XLA_FFI_DEFINE_HANDLER(kExecute, Execute,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<void*>>()
                           .Ctx<ffi::State<CutlassCallStateV3>>()
                           .RemainingArgs()
                           .RemainingRets()
                           .Attr<absl::string_view>("module")
                           .Attr<absl::string_view>("key"),
                       {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER(kExecuteNoCudaGraph, Execute,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<void*>>()
                           .Ctx<ffi::State<CutlassCallStateV3>>()
                           .RemainingArgs()
                           .RemainingRets()
                           .Attr<absl::string_view>("module")
                           .Attr<absl::string_view>("key"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kCutlassCallTarget.data(), "CUDA",
                         (XLA_FFI_Handler_Bundle{/*instantiate=*/kInstantiate,
                                                 /*prepare=*/kPrepare,
                                                 /*initialize=*/kInitialize,
                                                 /*execute=*/kExecute}));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         kCutlassCallNoCudaGraphTarget.data(), "CUDA",
                         (XLA_FFI_Handler_Bundle{
                             /*instantiate=*/kInstantiate,
                             /*prepare=*/kPrepare,
                             /*initialize=*/kInitialize,
                             /*execute=*/kExecuteNoCudaGraph}));

}  // namespace
}  // namespace xla::gpu::cutedsl
