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
#ifndef XLA_SERVICE_GPU_TRANSFORMS_POST_LAYOUT_CUSTOM_CALL_REWRITER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_POST_LAYOUT_CUSTOM_CALL_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

inline constexpr absl::string_view kAnnotateMemorySpace = "AnnotateMemorySpace";

// Rewrite custom calls after running layout assignment and normalization.
class PostLayoutCustomCallRewriter : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "post-layout-custom-call-rewriter";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_POST_LAYOUT_CUSTOM_CALL_REWRITER_H_
