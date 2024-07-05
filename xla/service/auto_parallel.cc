// TODO: add copyright 

#include "xla/service/auto_parallel.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/logging.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/hlo_pass_interface.h"

#include <stdint.h>

// assuming two dimensional mesh heirarchy of nodes and GPUs within nodes
#define MESH_X_DIM 4 /* number of nodes */
#define MESH_Y_DIM 8 /* number of gpus per node */
#define DEVICE_COUNT (MESH_X_DIM * MESH_Y_DIM) /* total number of devices */

namespace xla {

namespace {

  std::string LOG_HEADER(int x, const char c[]="AutoParallel: ") {
    return ((x == 0) ? (c) : ((LOG_HEADER(x - 1, c)) + "\t"));
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

    return result; 
  }
  
  // Creates a module from a single instruction for running a simple pass on
  std::unique_ptr<HloModule> CreateModuleFromInstruction(
      HloInstruction* instruction) {

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
    // (unique ptr so cleared out of memory)
    std::unique_ptr<HloModule> module = 
      std::make_unique<HloModule>(std::string(instruction->name()), config);
    module->AddEntryComputation(std::move(computation));

    // create a copy so it is completely separate from original module
    std::unique_ptr<HloModule> module_clone = module->Clone(); 

    return module_clone;
  }

  /*********************************************************/
  /* Sharding enumeration                                  */
  /*********************************************************/

  // need to declare
  class InstructionSharding;

  // assuming a 2D mesh grid, enumerates all choice 2 shardings of data
  // TODO: determine if tuples of data will need to be considered for sharding
  std::vector<HloSharding> EnumerateGeneralOpSharding(HloInstruction* operand, 
      HloInstruction* instruction) {

    // two device dimensions currently (assume 4 (nodes) x 8 (gpus per node))
    std::vector<HloSharding> shardings;

    // only sharding array types of data, otherwise no sharding options
    const Shape op_shape = operand->shape();
    if (!op_shape.IsArray()) {
      return shardings;
    }

    int num_dim = op_shape.rank();

    return shardings;
  }

  // Enumerates the shardings of a single operand instruction
  // depending on the user instruction of the operand and whether it is sharded.
  // This is a general function for iterating through shardings of a single
  std::vector<HloSharding> EnumerateOpSharding(
      HloInstruction* operand, HloInstruction* instruction) {
    
    // if sharding already exists for the instruction, only have that sharding
    if (operand->has_sharding()) {
      return { operand->sharding() };
    }

    // otherwise, perform sharding based on type of instruction
    // we are sharding operations for (may want to case on Dot product)
    switch (instruction->opcode()) {
    default:
      return EnumerateGeneralOpSharding(operand, instruction);
    }

  }

  // Enumerates all possible sharding strategies on the inputs of the current
  // instruction
  std::vector<InstructionSharding> EnumerateInstructionShardings(
      HloInstruction* instruction) {

    // enumerate through the shardings for each operator of the instruction
    std::vector<std::vector<HloSharding>> all_op_shardings;

    // TODO: pass index of operand to distinguish from other operands
    // if necessary
    HloInstruction::InstructionVector operands = instruction->operands();
    for (HloInstruction* op : operands) {
      all_op_shardings.push_back(EnumerateOpSharding(op, instruction));
    }

    // combine them to form strategies
    std::vector<InstructionSharding> strats;

    return strats;
  } 

  /*********************************************************/
  /* Helper classes                                        */
  /*********************************************************/

  class InstructionSharding {
  public:
    InstructionSharding() = default;
    ~InstructionSharding() = default;


  private:
    // The sharding of each operand of an instruction. Using shared_ptr
    // as noted by HloInstruction due to large size for many element tuples
    // This vector will be filled by enumerating incomplete sharding strategies
    std::vector<std::shared_ptr<const HloSharding>> operand_shardings_;

    // Sharding of result of computing instruction. This will be completed
    // by GSPMD when given the input shardings and determining the output
    // shardings.
    std::shared_ptr<const HloSharding> result_sharding_;

    // Cost of this specific instruction sharding. This will be assigned
    // after evaluating the cost of the complete HloModule after performing
    // sharding propagation through SPMD.
    uint64_t cost_;

  };

  class InstructionShardingInfo {
  public:
    InstructionShardingInfo(HloInstruction* orig_instr);
    ~InstructionShardingInfo() = default;

  private:

    // Points to the original instruction that will have its
    // sharding strategies enumerated. Eventually, this instruction
    // will be modified with a sharding strategy provided by the solvers
    HloInstruction* orig_instr_;

    // Module containing a single computation of a single instruction that
    // is a clone of the orig_instr. Created on construction of class.
    // After enumerating various incomplete sharding strategies,
    // each incomplete strategy will be applied to this module, be completed
    // by GSPMD, and evaluated for it's computational and communication cost
    // by inspecting the completed module's resulting operations
    std::unique_ptr<HloModule> single_instr_module_;    

    // vector of sharding strategies for the given instruction
    std::vector<InstructionSharding> sharding_strats_;

  };

  InstructionShardingInfo::InstructionShardingInfo(HloInstruction* orig_instr) 
      : orig_instr_(orig_instr),
        single_instr_module_(CreateModuleFromInstruction(orig_instr)),
        sharding_strats_(EnumerateInstructionShardings(orig_instr)) {
    return;
  }

  /*********************************************************/
  /* Debugging                                             */
  /*********************************************************/

  void PrintProtoList(std::function<int()> length_fn, std::function<int64_t(int)> getter, int depth=3, std::string list_name="array") {
    int n = length_fn();

    std::string s = "";

    for (int i = 0; i < n; i++) {
      s += std::to_string(getter(i)) + " ";
    }

    VLOG(5) << LOG_HEADER(depth, "InstInfo: ") << list_name << s;

    return; 
  }

  void PrintShardingInfo(OpSharding sharding, int depth=3) {

    VLOG(5) << LOG_HEADER(depth + 1, "SharInfo: ") << "sharding: ";

    std::function<int64_t(int)> getter = [&sharding](int index) {
      return (sharding.*static_cast<int64_t (OpSharding::*)(int) const>(&OpSharding::tile_assignment_dimensions))(index);
    };

    PrintProtoList(
      std::bind(&OpSharding::tile_assignment_dimensions_size, &sharding),
      getter,
      depth + 1, "tile_assignment_dimensions:"
    );

    getter = [&sharding](int index) {
      return (sharding.*static_cast<int64_t (OpSharding::*)(int) const>(&OpSharding::tile_assignment_devices))(index);
    };

    PrintProtoList(
      std::bind(&OpSharding::tile_assignment_devices_size, &sharding),
      getter,
      depth + 1, "tile_assignment_devices:"
    );

    getter = [&sharding](int index) {
      return (sharding.*static_cast<int64_t (OpSharding::*)(int) const>(&OpSharding::iota_reshape_dims))(index);
    };

    PrintProtoList(
      std::bind(&OpSharding::iota_reshape_dims_size, &sharding),
      getter,
      depth + 1, "iota_reshape_dims:"
    );

    getter = [&sharding](int index) {
      return (sharding.*static_cast<int32_t (OpSharding::*)(int) const>(&OpSharding::iota_transpose_perm))(index);
    };

    PrintProtoList(
      std::bind(&OpSharding::iota_transpose_perm_size, &sharding),
      getter,
      depth + 1, "iota_reshape_dims:"
    );
  }

  void PrintInstructionInfo(HloInstruction* instruction, int depth=3) {

    int64_t num_operands = instruction->operand_count();
    VLOG(5) << LOG_HEADER(depth, "InstInfo: ") << "Name: " << instruction->name() << " " << instruction;
    VLOG(5) << LOG_HEADER(depth + 1, "InstInfo: ") << "num operands: " << num_operands;
    VLOG(5) << LOG_HEADER(depth + 1, "InstInfo: ") << "sharded: " << instruction->has_sharding();
    VLOG(5) << LOG_HEADER(depth + 1, "InstInfo: ") << "shape: " << instruction->shape().ToString();

    if (instruction->has_sharding()) {
      // convert to Proto and print out proto elements
      PrintShardingInfo(instruction->sharding_ptr()->ToProto(), depth + 1);
    }

    HloInstruction::InstructionVector operands = instruction->operands();
    for (int i = 0; i < num_operands; i++) {
      VLOG(5) << LOG_HEADER(depth, "InstInfo: ") << i << ": " << operands[i]->name() << " " << operands[i]->shape().ToString() << " " << operands[i];
    }

    return;
  }

  void PrintComputationInfo(HloComputation* computation, int depth=3) {
    VLOG(5) << LOG_HEADER(depth, "CompInfo: ") << "Name: " << computation->name() << " " << computation;
    VLOG(5) << LOG_HEADER(depth, "CompInfo: ") << "Instruction Count: " << computation->instruction_count();

    for (HloInstruction* instr : computation->instructions()) {
      PrintInstructionInfo(instr, depth + 1);
    }
  }

  void PrintModuleInfo(HloModule* module, int depth=1) {

    VLOG(5) << LOG_HEADER(depth, "ModuInfo: ") << "Name: " << module->name() << " " << module;
    VLOG(5) << LOG_HEADER(depth, "ModuInfo: ") << "Computation count: " << module->computation_count();

    for (HloComputation* computation : module->computations()) {
      PrintComputationInfo(computation, depth + 1);
    }

  }

}   // namespace


  /*********************************************************/
  /* AutoParallelizer Pass Implementation                  */
  /*********************************************************/

  // overriden functions from class
  // modifies the sharding specification of the instructions in the module
  absl::StatusOr<bool> AutoParallelizer::Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) {

    VLOG(5) << "Testing run for AutoParallelizer";

    PrintModuleInfo(module);

    // create a clone of the module, then run off of that 
    std::unique_ptr<HloModule> module_clone = module->Clone();
    VLOG(5) << LOG_HEADER(0) << "module: " << module_clone->name();


    PrintModuleInfo(module_clone.get());


    // iterate through HloModule computations
    for (HloComputation* computation : module_clone->computations()) {
      
      VLOG(5) << LOG_HEADER(1) << "computation: " << computation->name();

      for (HloInstruction* instr : computation->instructions()) {

        VLOG(5) << LOG_HEADER(2) << "instruction: " << instr->name() << " " << instr->shape().ToString() << " " << instr;
        PrintInstructionInfo(instr);

        // create the relevant sharding information for this instruction
        // InstructionShardingInfo i(instr);

      }
    }
    
    return true;
  }

    

    
}   // namespace xla