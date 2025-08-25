/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_ONEDNN_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_ONEDNN_THUNK_H_

#if defined(INTEL_MKL)

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/runtime/custom_call_thunk.h"
#include "xla/backends/cpu/runtime/onednn/onednn_threadpool.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi_api.h"
#include "xla/runtime/object_pool.h"
#include "xla/service/cpu/onednn_memory_util.h"
#include "xla/service/custom_call_status.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

typedef CustomCallThunk::OpBuffers OpBuffers;

// Handles XLA's oneDNN custom calls operations.
class OneDnnThunk final : public Thunk {
 public:
  // Function signature for legacy untyped custom call API.
  using CustomCallTarget = std::function<void(void*, const void**, const char*,
                                              size_t, XlaCustomCallStatus*)>;

  static absl::StatusOr<std::unique_ptr<OneDnnThunk>> Create(
      Info info, absl::string_view target_name, OpBuffers op_buffers,
      absl::string_view backend_config, CustomCallApiVersion api_version);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final;

  const std::string& target_name() const { return target_name_; }
  const OpBuffers& op_buffers() const { return op_buffers_; }
  const CustomCallApiVersion& api_version() const { return api_version_; }
  const std::string& backend_config() const { return backend_config_; }

  bool ExecuteMayBlock() const final { return true; }

 private:
  struct OneDnnRuntime {
    OneDnnRuntime(Eigen::ThreadPoolInterface* thread_pool);

    OneDnnRuntime(const OneDnnRuntime&) = delete;
    OneDnnRuntime& operator=(const OneDnnRuntime&) = delete;
    OneDnnRuntime(OneDnnRuntime&&) = default;
    OneDnnRuntime& operator=(OneDnnRuntime&&) = default;

    std::unique_ptr<OneDnnThreadPool> threadpool;

    dnnl::engine cpu_engine;
    dnnl::stream onednn_stream;
    // We initialize the resources struct here to default values, so that we can
    // keep the primitive and memory objects alive for the duration of the
    // runtime. Otherwise, they would be destroyed as soon as we exit the
    // ExecuteOneDnn<primitive> FFI handler. This is a requirement of
    // oneDNN library's asynchronous execution model.
    OneDnnResources resources;
  };

  OneDnnThunk(Info info, absl::string_view target_name,
              std::variant<CustomCallTarget, ffi::HandlerRegistration> target,
              OpBuffers op_buffers, CustomCallApiVersion api_version,
              absl::string_view backend_config,
              std::optional<ffi::CallFrame> call_frame,
              std::unique_ptr<ffi::ExecutionState> execution_state);

  OneDnnThunk(const OneDnnThunk&) = delete;
  OneDnnThunk& operator=(const OneDnnThunk&) = delete;
  OneDnnThunk(OneDnnThunk&&) = default;
  OneDnnThunk& operator=(OneDnnThunk&&) = default;

  std::string target_name_;
  std::variant<CustomCallTarget, ffi::HandlerRegistration> target_;

  OpBuffers op_buffers_;
  CustomCallApiVersion api_version_;
  std::string backend_config_;

  // Reference call frame pre-initialized at construction time.
  std::optional<ffi::CallFrame> call_frame_;

  // A pool of call frames used at run time. Newly created call frames are
  // copied from the reference call frame and updated with buffer addresses.
  ObjectPool<ffi::CallFrame> call_frames_;

  // Execution state bound to the FFI handler. Optional.
  std::unique_ptr<ffi::ExecutionState> execution_state_;
};

}  // namespace xla::cpu

#endif  // INTEL_MKL
#endif  // XLA_BACKENDS_CPU_RUNTIME_ONEDNN_THUNK_H_
