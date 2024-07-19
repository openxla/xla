// TODO: license

#include "xla/service/experimental/instruction_to_module.h"

#include "xla/hlo/ir/hlo_instructions.h"

namespace xla {

/*********************************************************/
/* Convert instructions to modules                       */
/*********************************************************/

// instruction_to_module.{h,cc}

// clones a parameter instruction specifically 
// for single-instruction HloComputations
std::unique_ptr<HloInstruction> CloneParameterInstruction(
    const HloParameterInstruction* instruction,
    absl::Span<HloInstruction* const> operands) {

  // create parameter-retrieving instruction 
  // with same shape, cloned name, param_no of 0
  Shape s = instruction->shape();
  absl::string_view name = instruction->name();

  return std::move(HloInstruction::CreateParameter(0, s, name));
}

// fixes instructions so that it can be the only one inside of a computation
std::unique_ptr<HloInstruction> CloneSingleInstruction(
    const HloInstruction* instruction,
    absl::Span<HloInstruction* const> operands) {

  std::unique_ptr<HloInstruction> result;

  // choose appropriate correction based on instruction type
  switch (instruction->opcode()) {
    case HloOpcode::kParameter: {
      result = CloneParameterInstruction(
        static_cast<const HloParameterInstruction*>(instruction), 
        operands
      );
      break;
    }
    default: {
      result = instruction->CloneWithNewOperands(
        instruction->shape(), 
        operands
      );
      break;
    }
  }

  return result; 
}

// Takes any instruction in a module and converts it into an entry
// computation where it's operands are turned into HloParameterInstructions
std::unique_ptr<HloComputation> CreateComputationFromInstruction(
    const HloInstruction* instruction) {

  // build operands of instruction as parameters
  std::vector<std::unique_ptr<HloInstruction>> params;
  for (int i = 0; i < instruction->operand_count(); i++) {
    const HloInstruction *operand = instruction->operand(i);
    HloParameterInstruction p(i, operand->shape(), operand->name());
    params.push_back(HloInstruction::CreateParameter(
      i, operand->shape(), operand->name()
    ));
  }

  // copy the instruction so as not to modify the HloModule
  // and with the parameter instructions as parameters
  std::vector<HloInstruction*> param_ptrs;
  param_ptrs.reserve(params.size());
  for (int i = 0; i < params.size(); i++) {
    param_ptrs.push_back(params[i].get());
  }

  std::unique_ptr<HloInstruction> instr_clone 
    = std::move(CloneSingleInstruction(instruction, param_ptrs));

  // construct the computation builder
  HloComputation::Builder builder("single-instr");
  HloInstruction* root_instr = builder.AddInstruction(std::move(instr_clone)); 

  // add operands
  for (int i = 0; i < params.size(); i++) {
    TF_CHECK_OK(builder.AddParameter(std::move(params[i])).status());
  }
  
  // build the resulting computation and return
  return builder.Build(root_instr);

}

// Creates a module from a single instruction for running a simple pass on
std::unique_ptr<HloModule> CreateModuleFromInstruction(
    const HloInstruction* instruction) {

  // create a computation
  std::unique_ptr<HloComputation> computation
    = CreateComputationFromInstruction(instruction);

  // construct the module's configuration
  HloModuleConfig config{computation->ComputeProgramShape()};
  ProgramShape ps = computation->ComputeProgramShape();

  // construct the module from the computation 
  // (unique ptr so cleared out of memory)
  std::unique_ptr<HloModule> module = 
    std::make_unique<HloModule>(std::string(instruction->name()), config);
  module->AddEntryComputation(std::move(computation));

  // create a copy so it is completely separate from original module
  std::unique_ptr<HloModule> module_clone = module->Clone(); 

  return module_clone;
}

} // xla