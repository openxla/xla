#include "xla/examples/first_hlo_pass/add_zero_eliminator.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"

namespace xla {
namespace {

bool IsIntegralZeroConstant(const HloInstruction* instruction) {
  return instruction->IsConstant() &&
         primitive_util::IsIntegralType(instruction->shape().element_type()) &&
         instruction->literal().IsAll(0);
}

}  // namespace

absl::StatusOr<bool> AddZeroEliminator::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kAdd) {
        continue;
      }

      HloInstruction* value = instruction->mutable_operand(0);
      HloInstruction* zero = instruction->mutable_operand(1);
      if (!IsIntegralZeroConstant(zero)) {
        value = instruction->mutable_operand(1);
        zero = instruction->mutable_operand(0);
      }
      if (!IsIntegralZeroConstant(zero)) {
        continue;
      }

      ABSL_RETURN_IF_ERROR(computation->ReplaceInstruction(instruction, value));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
