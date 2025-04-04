#include "xla/backends/gpu/codegen/prefetch.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/kernels/prefetch_custom_kernel.h"
#include "xla/service/gpu/kernels/prefetch_kernel_common.h"

namespace xla {
namespace gpu {

namespace {
static const HloInstruction& GetCustomCall(const HloInstruction& hlo) {
  if (hlo.opcode() == HloOpcode::kCustomCall) {
    return hlo;
  }
  for (const HloInstruction* user : hlo.users()) {
    if (user->opcode() != HloOpcode::kTuple) {
      return GetCustomCall(*user);
    }
  }
  LOG(FATAL) << "No custom call found from " << hlo.ToString();
}
}  // namespace

absl::StatusOr<FusionEmissionResult> L2PrefetchFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  const HloInstruction& cc = GetCustomCall(
      *fusion.fused_instructions_computation()->parameter_instruction(0));
  TF_RET_CHECK(cc.operand_count() <= kMaxNumPrefetchBuffers);
  std::array<int, kMaxNumPrefetchBuffers> buffer_sizes;
  std::fill(buffer_sizes.begin() + cc.operand_count(), buffer_sizes.end(), 0);
  for (int i = 0; i < cc.operand_count(); ++i) {
    buffer_sizes[i] = ShapeUtil::ByteSizeOf(cc.operand(i)->shape());
  }

  int num_blocks;
  if (!absl::SimpleAtoi(
          cc.get_frontend_attribute("prefetch_num_blocks").value_or(""),
          &num_blocks)) {
    num_blocks = kPrefetchDefaultBlocks;
  }
  TF_RET_CHECK(num_blocks > 0);

  TF_ASSIGN_OR_RETURN(
      CustomKernel kernel,
      kernel::prefetch::GetL2PrefetchCustomKernel(buffer_sizes, num_blocks));
  TF_ASSIGN_OR_RETURN(
      auto kernel_arguments,
      emitters::KernelArguments::Create(ir_emitter_context.buffer_assignment(),
                                        GetDefaultBufferAlignment(), &fusion,
                                        fusion.operands()));
  FusionEmissionResult result;
  result.thunks.emplace_back(std::make_unique<CustomKernelThunk>(
      &fusion, std::move(kernel), std::move(kernel_arguments.args())));
  return result;
}

}  // namespace gpu
}  // namespace xla
