/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/runtime/topk.h"

#include <stdint.h>

#include <cstddef>
#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/runtime/ffi/ffi_api.h"
#include "xla/runtime/ffi/ffi_c_api.h"
#include "xla/service/gpu/runtime/topk_kernel.h"
#include "xla/status.h"

namespace xla::gpu {

class SpecializeTopkVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleCustomCall(HloInstruction* inst) override {
    HloCustomCallInstruction* topk = DynCast<HloCustomCallInstruction>(inst);
    if (topk == nullptr || topk->custom_call_target() != "TopK") {
      return OkStatus();
    }
    DCHECK_GE(topk->operand_count(), 1);

    return OkStatus();
  }
};

StatusOr<bool> SpecializeTopk::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return SpecializeTopkVisitor().RunOnModule(module, execution_threads);
}

namespace ffi = ::xla::runtime::ffi;

struct TopkFfiModule : ffi::StatelessModule {
  explicit TopkFfiModule(const XLA_FFI_Api* api)
      : StatelessModule(api, "TopkFfiModule", {{"GpuTopK", FFI_TopK}}) {}

  XLA_FFI_DEFINE_FUNCTION(FFI_TopK, TopK,
                          ffi::Ffi::Binding()
                              .Stream<se::gpu::GpuStreamHandle>()
                              .Arg<ffi::StridedBufferArg>()
                              .Arg<ffi::StridedBufferArg>()
                              .Arg<ffi::StridedBufferArg>()
                              .Arg<ffi::StridedBufferArg>());

  static ffi::FfiStatus TopK(se::gpu::GpuStreamHandle stream,
                             ffi::StridedBufferArg data,
                             ffi::StridedBufferArg top_elements,
                             ffi::StridedBufferArg indices,
                             ffi::StridedBufferArg scratch_elements) {
    // TODO(doak): Better validate these arguments.
    if (data.sizes.size() > 2)
      return ffi::FfiStatus::InvalidArgument("Invalid input shape");
    if (indices.dtype != ffi::PrimitiveType::S32)
      return ffi::FfiStatus::InvalidArgument("Indices should be S32");
    const bool has_batch = data.sizes.size() == 2;
    const size_t batch_size = has_batch ? data.sizes[0] : 1;
    const size_t n = has_batch ? data.sizes[1] : data.sizes[0];
    const size_t k = has_batch ? top_elements.sizes[1] : top_elements.sizes[0];
    return RunTopk(stream, data.dtype, data.data, n, top_elements.data,
                   static_cast<uint32_t*>(indices.data), k, batch_size,
                   scratch_elements.data);
  }
};

XLA_REGISTER_FFI_MODULE(std::make_unique<TopkFfiModule>(GetXlaFfiApi()));

}  // namespace xla::gpu
