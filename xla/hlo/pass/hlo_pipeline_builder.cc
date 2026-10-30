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

#include "xla/hlo/pass/hlo_pipeline_builder.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_context.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/pass/hlo_pass_options.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/pass/hlo_pass_registry.h"
#include "xla/hlo/pass/hlo_pipeline.pb.h"
#include "xla/service/compilation_stats.h"
#include "xla/util.h"

namespace xla {
namespace {

// Lets `HloPassFix`, which derives from its pass type, wrap a pass created at
// runtime.
class ForwardingHloPass : public HloPassInterface {
 public:
  explicit ForwardingHloPass(std::unique_ptr<HloPassInterface> pass)
      : pass_(std::move(pass)) {}

  absl::string_view name() const override { return pass_->name(); }
  bool IsPassPipeline() const override { return pass_->IsPassPipeline(); }

  absl::Status RunOnChangedComputations(
      HloModule* module, RunState* run_state,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      override {
    return pass_->RunOnChangedComputations(module, run_state,
                                           execution_threads);
  }

 protected:
  absl::StatusOr<bool> RunImpl(HloModule* module,
                               const absl::flat_hash_set<absl::string_view>&
                                   execution_threads) override {
    return pass_->Run(module, execution_threads);
  }

 private:
  std::unique_ptr<HloPassInterface> pass_;
};

class HloPipelineBuilder {
 public:
  HloPipelineBuilder(const HloPassContext& context,
                     CompilationStats* compilation_stats,
                     const HloPassRegistry& registry)
      : context_(context),
        compilation_stats_(compilation_stats),
        registry_(registry) {}

  absl::StatusOr<std::unique_ptr<HloPassPipeline>> BuildSequence(
      absl::string_view name,
      const HloPipelineNodeSequenceProto& sequence) const {
    auto pipeline = std::make_unique<HloPassPipeline>(name, compilation_stats_);
    for (const HloPipelineNodeProto& node : sequence.nodes()) {
      absl::StatusOr<std::unique_ptr<HloPassInterface>> pass = BuildNode(node);
      // Prefixing every enclosing sequence yields the path to the bad node.
      if (!pass.ok()) {
        return AddStatus(pass.status(), absl::StrCat("pipeline '", name, "'"));
      }
      pipeline->AddPass(*std::move(pass));
    }
    return pipeline;
  }

 private:
  absl::StatusOr<std::unique_ptr<HloPassInterface>> BuildNode(
      const HloPipelineNodeProto& node) const {
    const int32_t max_iterations = node.max_fixed_point_iterations();
    if (max_iterations < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Node '", node.name(), "' has negative max_fixed_point_iterations ",
          max_iterations));
    }
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloPassInterface> pass,
                          BuildNodeOnce(node));
    if (max_iterations <= 1) {
      return pass;
    }
    return HloPassFix<ForwardingHloPass>::Create(max_iterations,
                                                 std::move(pass));
  }

  absl::StatusOr<std::unique_ptr<HloPassInterface>> BuildNodeOnce(
      const HloPipelineNodeProto& node) const {
    if (!node.has_sub_nodes()) {
      return registry_.Create(node.name(), context_,
                              HloPassOptions(node.options()));
    }
    if (!node.options().empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Node sequence '", node.name(), "' has options"));
    }
    return BuildSequence(node.name(), node.sub_nodes());
  }

  const HloPassContext& context_;
  CompilationStats* compilation_stats_;
  const HloPassRegistry& registry_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<HloPassPipeline>> BuildHloPassPipeline(
    const HloPipelineProto& proto, absl::string_view name,
    const HloPassContext& context, CompilationStats* compilation_stats,
    const HloPassRegistry& registry) {
  return HloPipelineBuilder(context, compilation_stats, registry)
      .BuildSequence(name, proto.root());
}

}  // namespace xla
