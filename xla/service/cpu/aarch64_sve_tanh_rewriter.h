#ifndef XLA_SERVICE_CPU_AARCH64_SVE_TANH_REWRITER_H_
#define XLA_SERVICE_CPU_AARCH64_SVE_TANH_REWRITER_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace cpu {

class Aarch64SveTanhRewriter : public HloModulePass {
 public:
  absl::string_view name() const override { return "aarch64-sve-tanh-rewriter"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_AARCH64_SVE_TANH_REWRITER_H_
