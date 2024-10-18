/* Copyright 2024 The OpenXLA Authors.
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

#ifndef XLA_SERVICE_CPU_RUNTIME_ONEDNN_THUNK_H_
#define XLA_SERVICE_CPU_RUNTIME_ONEDNN_THUNK_H_

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/runtime/custom_call_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

typedef CustomCallThunk::OpBuffers OpBuffers;

class OneDnnThunk final : public Thunk {
 public:
  static absl::StatusOr<std::unique_ptr<OneDnnThunk>> Create(
      const std::string& target, Info info, OpBuffers buffers,
      absl::string_view config);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final;

 private:
  OneDnnThunk(const std::string& target, Info info, OpBuffers buffers,
              absl::string_view config);

  OpBuffers op_buffers_;
  std::string target_;
  std::string config_;
};

}  // namespace xla::cpu

#endif  // INTEL_MKL && && ENABLE_ONEDNN_V3
#endif  // XLA_SERVICE_CPU_RUNTIME_ONEDNN_THUNK_H_
