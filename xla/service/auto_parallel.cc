// TODO: add copyright 

#include "xla/service/auto_parallel.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/logging.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

namespace {

  // static functions only to this file itself
  void CorrectParameterInstruction(HloParameterInstruction* instruction) {

    // set the parameter count to 0


    return;
  }

  // fixes instructions so that it can be the only one inside of a computation
  void CorrectSingleInstruction(HloInstruction* instruction) {

    // choose appropriate correction based on instruction type
    switch (instruction->opcode()) {
      case HloOpcode::kParameter: {
        CorrectParameterInstruction(Cast<HloParameterInstruction>(instruction));
        break;
      }
      default: {
        break;
      }
    }

    return; 
  }
  
  // Creates a module from a single instruction for running a simple pass on
  HloModule* CreateModuleFromInstruction(HloInstruction* instruction) {

    // copy the instruction so as not to modify the HloModule
    std::unique_ptr<HloInstruction> instr_clone = instruction->Clone();
    CorrectSingleInstruction(instr_clone.get());
    
    VLOG(5) << "\t\t\t" << instruction->ToString();

    // create entry computation from the single instruction
    HloComputation::Builder builder{"single-instr"};
    HloInstruction* instrp = builder.AddInstruction(std::move(instr_clone));
    std::unique_ptr<HloComputation> computation = builder.Build(instrp);

    // // construct the module's configuration
    // HloModuleConfig config{computation->ComputeProgramShape()};

    // // construct the module from the computation
    // HloModule* module = new HloModule(std::string(instruction->name()), config);
    // module->AddEmbeddedComputation(std::move(computation));

    // VLOG(5) << "Successful instruction creation for " << module->name();
    // return module;
    
    return nullptr;
  }

}   // namespace

  // overriden functions from class
  absl::StatusOr<bool> AutoParallelizer::Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) {

    VLOG(5) << "Testing run for AutoParallelizer";

    // create a clone of the module, then run off of that 
    std::unique_ptr<HloModule> module_clone = module->Clone();
    VLOG(5) << "AutoParallel: " << "module: " << module_clone->name();

    // iterate through HloModule computations
    for (HloComputation* computation : module_clone->computations()) {
      
      VLOG(5) << "AutoParallel: " << "\t" << "computation: " << computation->name();

      for (HloInstruction* instr : computation->instructions()) {

        VLOG(5) << "AutoParallel: " << "\t\t" << "instruction: " << instr->name();

        CreateModuleFromInstruction(instr);
      }
    }
    
    return true;
  }

    

    
}   // namespace xla