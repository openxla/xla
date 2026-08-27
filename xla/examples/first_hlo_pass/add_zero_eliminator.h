#ifndef XLA_EXAMPLES_FIRST_HLO_PASS_ADD_ZERO_ELIMINATOR_H_
#define XLA_EXAMPLES_FIRST_HLO_PASS_ADD_ZERO_ELIMINATOR_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

class AddZeroEliminator final : public HloModulePass {
 public:
  absl::string_view name() const override { return "add-zero-eliminator"; }

  using HloModulePass::Run;
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_EXAMPLES_FIRST_HLO_PASS_ADD_ZERO_ELIMINATOR_H_
