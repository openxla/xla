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

  std::string LOG_HEADER(int x) {
    return ((x == 0) ? ("AutoParallel: ") : ((LOG_HEADER(x - 1)) + "\t"));
  }

  /*********************************************************/
  /* Convert instructions to modules                       */
  /*********************************************************/

  // clones a parameter instruction specifically 
  // for single-instruction HloComputations
  std::unique_ptr<HloInstruction> CloneParameterInstruction(
      HloParameterInstruction* instruction) {

    // create parameter-retrieving instruction 
    // with same shape, cloned name, param_no of 0
    Shape s = instruction->shape();
    absl::string_view name = instruction->name();

    return std::move(HloInstruction::CreateParameter(0, s, name));
  }

  // fixes instructions so that it can be the only one inside of a computation
  std::unique_ptr<HloInstruction> CloneSingleInstruction(
      HloInstruction* instruction) {

    std::unique_ptr<HloInstruction> result;

    // choose appropriate correction based on instruction type
    switch (instruction->opcode()) {
      case HloOpcode::kParameter: {
        result = CloneParameterInstruction(
          Cast<HloParameterInstruction>(instruction));
        break;
      }
      default: {
        result = instruction->Clone();
        break;
      }
    }

    return std::move(result); 
  }
  
  // Creates a module from a single instruction for running a simple pass on
  HloModule* CreateModuleFromInstruction(HloInstruction* instruction) {

    // copy the instruction so as not to modify the HloModule
    std::unique_ptr<HloInstruction> instr_clone 
      = std::move(CloneSingleInstruction(instruction));
    
    // create entry computation from the single instruction
    HloComputation::Builder builder{"single-instr"};
    HloInstruction* instrp = builder.AddInstruction(std::move(instr_clone));
    std::unique_ptr<HloComputation> computation = builder.Build(instrp);

    // construct the module's configuration
    HloModuleConfig config{computation->ComputeProgramShape()};

    // construct the module from the computation
    HloModule* module = new HloModule(std::string(instruction->name()), config);
    module->AddEmbeddedComputation(std::move(computation));

    return module;
  }

  /*********************************************************/
  /* Sharding enumeration                                  */
  /*********************************************************/

  // // creates a list of potential sharding strategies for the instruction
  // std::vector<std::unique_ptr<HloSharding>> GenerateShardingStrats(
  //     HloInstruction* instruction) {

  //   // figure things out here

  //   return nullptr; 
  // }

  void PrintInfo(HloInstruction* instruction) {

    int64_t num_operands = instruction->operand_count();

    VLOG(5) << LOG_HEADER(3) << "num-operands" << num_operands;

    HloInstruction::InstructionVector operands = instruction->operands();
    for (int i = 0; i < num_operands; i++) {
      VLOG(5) << LOG_HEADER(3) << i << ": " << operands[i]->name() << " " << operands[i]->shape().ToString();
    }

    return;
  }

}   // namespace


  /*********************************************************/
  /* AutoParallelizer Pass Implementation                  */
  /*********************************************************/

  // overriden functions from class
  absl::StatusOr<bool> AutoParallelizer::Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) {

    VLOG(5) << "Testing run for AutoParallelizer";

    // create a clone of the module, then run off of that 
    std::unique_ptr<HloModule> module_clone = module->Clone();
    VLOG(5) << LOG_HEADER(0) << "module: " << module_clone->name();

    // iterate through HloModule computations
    for (HloComputation* computation : module_clone->computations()) {
      
      VLOG(5) << LOG_HEADER(1) << "computation: " << computation->name();

      for (HloInstruction* instr : computation->instructions()) {

        VLOG(5) << LOG_HEADER(2) << "instruction: " << instr->name() << " " << instr->shape().ToString();
        PrintInfo(instr);

        if (!instr->has_sharding()) {
          // avoid overwriting user-suggested shardings
          HloModule* single_inst_module = CreateModuleFromInstruction(instr);
        } else {
          VLOG(5) << LOG_HEADER(2) << "instruction has sharding: " << instr->name();
        }

        // now enumerate through various module shardings

      }
    }
    
    return true;
  }

    

    
}   // namespace xla