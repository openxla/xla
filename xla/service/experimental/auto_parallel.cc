// TODO: add copyright 

#include "xla/service/experimental/auto_parallel.h"

#include "xla/service/experimental/module_cost_evaluator.h"
#include "xla/service/experimental/resharding_cost_evaluator.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/errors.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/sharding_propagation.h"
#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"

#include <stdint.h>

// assuming two dimensional mesh heirarchy of nodes and GPUs within nodes
#define NUM_MESH_DIM 2  /* number of dimensions in the mesh grid */
#define MESH_X_DIM 2 /* number of nodes */
#define MESH_Y_DIM 4 /* number of gpus per node */
#define DEVICE_COUNT (MESH_X_DIM * MESH_Y_DIM) /* total number of devices */

namespace xla {

namespace {

  /*********************************************************/
  /* Debugging                                             */
  /*********************************************************/

  std::string LOG_HEADER(int x, const char c[]="AutoParallel: ") {
    return ((x == 0) ? (c) : ((LOG_HEADER(x - 1, c)) + "\t"));
  }

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
      depth + 1, "iota_transpose_perm:"
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
      VLOG(5) << LOG_HEADER(depth + 1, "InstInfo: ") << "op " << i << ": " << operands[i]->name() << " " << operands[i]->shape().ToString() << " " << operands[i];
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

  /*********************************************************/
  /* Convert instructions to modules                       */
  /*********************************************************/

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

  /*********************************************************/
  /* ShardingStrategy Class                             */
  /*********************************************************/

  class ShardingStrategy {
  public:
    ShardingStrategy() = default;
    ~ShardingStrategy() = default;
    ShardingStrategy(const ShardingStrategy& s) = default;
    ShardingStrategy(ShardingStrategy&& s) = default;

    // cost getters and setters
    uint64_t cost() const { return cost_; }
    void set_cost(uint64_t cost) { cost_ = cost; }

    // modifying the operand_shardings
    // TODO: accept a shared pointer
    void AddOpSharding(HloSharding sharding);
    std::shared_ptr<HloSharding> GetOpSharding(int op_idx) {
      return operand_shardings_[op_idx];
    };
    int64_t NumOpShardings() { return operand_shardings_.size(); }

    // modifying resulting sharding
    // TODO: accept a shared pointer
    void set_result_sharding(HloSharding result_sharding);
    std::shared_ptr<HloSharding> result_sharding() { return result_sharding_; };

    void AddUserReshardingCosts(std::vector<uint64_t> resharding_costs);

  private:
    // TODO: make these shared_ptr<const HloSharding>
    // The sharding of each operand of an instruction. Using shared_ptr
    // as noted by HloInstruction due to large size for many element tuples
    // This vector will be filled by enumerating incomplete sharding strategies
    std::vector<std::shared_ptr<HloSharding>> operand_shardings_;

    // Cost of this specific instruction sharding. This will be assigned
    // after evaluating the cost of the complete HloModule after performing
    // sharding propagation through SPMD.
    uint64_t cost_;

    // TODO: make these shared_ptr<const HloSharding>
    // Sharding of result of computing instruction. This will be completed
    // by GSPMD when given the input shardings and determining the output
    // shardings.
    std::shared_ptr<HloSharding> result_sharding_;

    // For each user of this instruction, have a resharding cost for
    // each of the user's potential sharding strategies
    // Cost vectors are added in user order
    std::vector<std::vector<uint64_t>> resharding_costs_;

  };

  void ShardingStrategy::AddOpSharding(HloSharding sharding) {
    operand_shardings_.push_back(std::make_shared<HloSharding>(sharding));
  }

  void ShardingStrategy::set_result_sharding(HloSharding result_sharding) {
    result_sharding_ = std::make_shared<HloSharding>(result_sharding);
  }

  void ShardingStrategy::AddUserReshardingCosts(
      std::vector<uint64_t> costs) {
    resharding_costs_.push_back(std::move(costs));
  }

  /*********************************************************/
  /* Sharding enumeration                                  */
  /*********************************************************/

  // enumerate sharding from the number of dimensions in the data
  // TODO: could be cached
  // Constructs a vector of rank * (rank + 1) shardings
  std::vector<HloSharding> EnumerateShardingsFromRank(int rank) {

    // two device dimensions currently (assume 4 (nodes) x 8 (gpus per node))
    std::vector<HloSharding> shardings;

    // note: this code is only acceptable for a 2D mesh grid,
    // would require more complicated solution for higher-dimensional grids
    for (int x_idx = 0; x_idx < rank; x_idx++) {
      for (int y_idx = 0; y_idx < rank; y_idx++) {
        // TODO: have a simple boolean for whether we would like to shard
        // both mesh grid dimensions on the same data dimension

        // construct tile_assignment_dims
        std::vector<int64_t> tile_assignment_dims(rank, 1);
        tile_assignment_dims[x_idx] *= MESH_X_DIM;
        tile_assignment_dims[y_idx] *= MESH_Y_DIM;

        // NOTE: intentionally may add two shardings if x_idx == y_idx
        // (i.e. when sharding a single data dimension on all devices)
        // because ordering of machines may influence resulting communication
        // costs and overall problem. Adding both shardings to be complete
        // construct the iota_reshape_dims and iota_tranpose_perm
        if (x_idx <= y_idx) {
          shardings.push_back(HloSharding::IotaTile(
            tile_assignment_dims,
            { MESH_X_DIM * MESH_Y_DIM },
            { 0 }
          ));
        }
        if (y_idx <= x_idx) {
          shardings.push_back(HloSharding::IotaTile(
            tile_assignment_dims,
            { MESH_X_DIM, MESH_Y_DIM },
            { 1, 0 }
          ));
        }
      }
    }

    return shardings;
  }

  // assuming a 2D mesh grid, enumerates all choice 2 shardings of data
  // TODO: determine if tuples of data will need to be considered for sharding
  std::vector<HloSharding> EnumerateGeneralOpSharding(HloInstruction* operand, 
      HloInstruction* instruction) {
    
    // operand requires sharding
    assert(operand->has_sharding());

    // only sharding array types of data, otherwise no sharding options
    const Shape op_shape = operand->shape();
    if (!op_shape.IsArray()) {
      return {};
    }

    return EnumerateShardingsFromRank(op_shape.rank());
  }

  // TODO: figure out a better way to deal with tuples for data
  std::vector<HloSharding> EnumerateTupleOpSharding(HloInstruction* operand,
      HloInstruction* instruction) {
    return {};
  }

  // Enumerates the shardings of a single operand instruction
  // depending on the user instruction of the operand and whether it is sharded.
  // This is a general function for iterating through shardings of a single
  // TODO: should give integer argument here and in EnumerateGeneralOpSharding
  std::vector<HloSharding> EnumerateOpSharding(
      HloInstruction* operand, HloInstruction* instruction) {
    
    // if sharding already exists for the instruction, only have that sharding
    if (operand->has_sharding()) {
      return { operand->sharding() };
    }

    // otherwise, perform sharding based on type of instruction
    // we are sharding operations for (may want to case on Dot product)
    switch (instruction->opcode()) {
    case HloOpcode::kTuple:
      return EnumerateTupleOpSharding(operand, instruction);
    default:
      return EnumerateGeneralOpSharding(operand, instruction);
    }

  }

  // Combine shardings for each operator to form sharding strategies
  std::vector<ShardingStrategy> CombineShardingVectors(
      std::vector<std::vector<HloSharding>> sharding_vecs) {
    int num_vecs = sharding_vecs.size();

    if (num_vecs == 0) {
      return {};
    } else if (num_vecs == 1) {
      // only one operator, map each sharding to a separate ShardingStrategy
      std::vector<ShardingStrategy> strats;
      for (HloSharding sharding : sharding_vecs[0]) {
        ShardingStrategy strat;
        strat.AddOpSharding(sharding);
        strats.push_back(strat);
      }
      return strats;
    }

    // otherwise recurse
    std::vector<HloSharding> shardings = sharding_vecs[num_vecs - 1];
    std::vector<ShardingStrategy> sub_strats = CombineShardingVectors(
      std::vector<std::vector<HloSharding>>(sharding_vecs.begin(), 
        sharding_vecs.end() - 1)
    );

    std::vector<ShardingStrategy> strats;
    for (HloSharding sharding : shardings) {
      for (ShardingStrategy strat : sub_strats) {
        // copy the existing sub_strat and add the new sharding
        strat.AddOpSharding(sharding);
        strats.push_back(strat);
      }
    }
    
    return strats;
  }

  // Enumerates all possible sharding strategies on the inputs of the current
  // instruction
  // TODO: need to make instruction sharding use shared pointers
  // going to be many identical copies of the same sharding in memory
  // for larger problems
  std::vector<ShardingStrategy> EnumerateShardingStrategies(
      HloInstruction* instruction) {

    // enumerate through the shardings for each operator of the instruction
    std::vector<std::vector<HloSharding>> all_op_shardings;

    // TODO: pass index of operand to distinguish from other operands
    // if necessary
    HloInstruction::InstructionVector operands = instruction->operands();
    for (HloInstruction* op : operands) {
      all_op_shardings.push_back(EnumerateOpSharding(op, instruction));
    }

    return CombineShardingVectors(all_op_shardings);
  } 

  /*********************************************************/
  /* GSPMD Completion                                      */
  /*********************************************************/

  // Major steps prior to evaluating the cost
  //  0. clone the original module?
  //  1. clear the module of shardings (does GSPMD insert any other metadata?)
  //  2. apply the shardings from a strategy
  //  3. run GSPMD
  //  4. evaluate the cost of the resulting module
  //  5. figure out the output sharding of the complete module

  // This function clears all shardings from instructions in the module
  void ClearHloShardings(HloModule* module) {

    for (HloComputation* computation : module->computations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        instruction->clear_sharding();
      }
    }

    return;
  }

  // This function inserts a sharding strategy into an HloModule
  // Assumes that this is a single instruction HloModule
  void ApplyModuleStrategy(HloModule* module, ShardingStrategy* strat) {

    std::shared_ptr<const HloSharding> sharding_ptr;

    // should only have one computation
    assert(module->computation_count() == 1);
    HloComputation* computation = module->entry_computation();

    // apply the shardings to the operands of the root instruction
    HloInstruction* root_instruction = computation->root_instruction();
    int num_operands = root_instruction->operand_count();
    assert(strat->NumOpShardings() == num_operands);

    for (int i = 0; i < num_operands; i++) {
      root_instruction->mutable_operand(i)->set_sharding(
        strat->GetOpSharding(i)
      );
    }

    return; 
  }

  // This function runs the sharding propagation pass over an HloModule
  void RunShardingPropagation(HloModule* module) {
    // automatically complete the shardings
    HloPassPipeline sharding_pipeline("sharding-propagation");
    sharding_pipeline.AddPass<ShardingPropagation>(
      /* is_spmd */ true,
      /* propagate_metadata */ true,
      /* sharding propagation to output */ absl::Span<const bool>({ true }),
      /* sharding propagation to parameters */ absl::Span<const bool>({ false })
    );

    TF_CHECK_OK(sharding_pipeline.Run(module).status());
  }

  // This function runs the SpmdPartitioner over an HloModule
  void RunSpmdPartitioner(HloModule* module) {
    // fill in communications to produce SPMD program
    HloPassPipeline spmd_pipeline("spmd-partitioner");
    spmd_pipeline.AddPass<spmd::StatefulRngSpmdPartitioner>(
      module->config().num_partitions(),
      module->config().replica_count()
    );

    TF_CHECK_OK(spmd_pipeline.Run(module).status());
  }

  // This function runs the sharding propagation pipeline pass on the module
  HloSharding RunGSPMD(HloModule* module) {

    // TODO: will need to remove manual setting of this eventually
    // TODO: is setting replica_count to 1 okay?
    HloModuleConfig& config = module->mutable_config();
    config.set_num_partitions(DEVICE_COUNT);
    config.set_replica_count(1);
    config.set_use_spmd_partitioning(true);

    // complete the shardings to the output
    RunShardingPropagation(module);

    // extract the output sharding
    HloInstruction* instr = module->entry_computation()->root_instruction();
    HloSharding out_sharding = instr->sharding();

    // now replace shardings with communication operations
    RunSpmdPartitioner(module);

    return out_sharding;
  }

  // This function returns the sharding of the entry computation's 
  // root instruction
  HloSharding GetRootSharding(HloModule* module) {
    HloInstruction* root = module->entry_computation()->root_instruction();
    assert(root->has_sharding());

    return root->sharding();
  }

  // This function will evaluate the sharding strategy on the 
  // single-instruction module by applying the input shardings from the strat
  // onto the operands of the module's root instruction, running GSPMD,
  // and evaluating the communication costs of the resulting module
  // The strat parameter will be updated with this cost and the resulting
  // output sharding
  void EvaluateShardingStrat(const HloModule* module, 
      ShardingStrategy* strat) {

    // clone the module to avoid clobbering future evaluations
    std::unique_ptr<HloModule> eval_module = module->Clone();

    // apply GSPMD to the module with the sharding strategy
    // TODO: should these take in unique pointers or is regulard pointer ok?
    ClearHloShardings(eval_module.get());
    ApplyModuleStrategy(eval_module.get(), strat);
    strat->set_result_sharding(RunGSPMD(eval_module.get()));

    // now evaluate cost
    ModuleCostEvaluator evaluator;
    strat->set_cost(evaluator.Evaluate(eval_module.get()));
    VLOG(5) << LOG_HEADER(0) << "cost: " << strat->cost();
    // PrintModuleInfo(eval_module.get());
    
    // update strat with cost and root instruction's output sharding
    // NOTE: the eval_module after GSPMD doesn't have it's sharding
    // in the output (i.e. root computation of entry computation) which
    // is unexpected, I thought that GSPMD would fill that in

  }

  /*********************************************************/
  /* InstructionStrategies Class                         */
  /*********************************************************/

  class InstructionStrategies {
  public:
    InstructionStrategies(HloInstruction* orig_instr);
    ~InstructionStrategies() = default;
    InstructionStrategies(const InstructionStrategies& info) = default;

    // accessors
    std::vector<ShardingStrategy>& sharding_strats() { 
      return sharding_strats_;
    };

  private:

    // Applies the given sharding strategy to the module and perforsm GSPMD
    // on it to complete the sharding.

    // This function will iterate through the various sharding strategies
    // and apply them to the instruction within the module. Afterwards,
    // GSPMD will be run to complete the module and the appropriate 
    // sharding strategy
    void EstimateStrategyCosts();

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
    std::vector<ShardingStrategy> sharding_strats_;

  };

  InstructionStrategies::InstructionStrategies(HloInstruction* orig_instr) 
      : orig_instr_(orig_instr),
        single_instr_module_(CreateModuleFromInstruction(orig_instr)),
        sharding_strats_(EnumerateShardingStrategies(orig_instr)) {
    EstimateStrategyCosts();
    return;
  }

  void InstructionStrategies::EstimateStrategyCosts() {

    for (int i = 0; i < sharding_strats_.size(); i++) {
      EvaluateShardingStrat(single_instr_module_.get(), &sharding_strats_[i]);
    }

    return;
  }

  /*********************************************************/
  /* Resharding Cost Estimation                            */
  /*********************************************************/

  void EstimateReshardingCosts(std::unordered_map<HloInstruction*, 
      std::unique_ptr<InstructionStrategies>>& map) {
    
    // for each instruction, for each user of it, for each sharding strategy
    // recalculate resharding costs

    ReshardingCostEvaluator evaluator;

    VLOG(5) << "Map size: " << map.size();

    for (auto& [instr, instr_strats] : map) {
      const Shape& shape = instr->shape();
      
      // don't loop if no users
      if (instr->user_count() == 0) {
        continue;
      }

      VLOG(5) << "\tNum sharding strats: " << instr_strats->sharding_strats().size();

      for (ShardingStrategy& strat: instr_strats->sharding_strats()) {
        std::shared_ptr<HloSharding> in_sharding = strat.result_sharding();
        VLOG(5) << "\t\tNum Users: " << instr->user_count();
        // TODO: could honestly cache these to reduce storage

        for (HloInstruction* user : instr->users()) {
          // get index of instr in user's operands
          int op_idx = user->operand_index(instr);
          std::unique_ptr<InstructionStrategies>& user_strats = map[user];

          VLOG(5) << "\t\t\tNum user strats: " << user_strats->sharding_strats().size();

          std::vector<uint64_t> resharding_costs;
          for (ShardingStrategy& out_strat : 
              user_strats->sharding_strats()) {
            uint64_t cost = evaluator.Evaluate(
              shape, *in_sharding.get(), *out_strat.GetOpSharding(op_idx).get()
            );
            VLOG(5) << "Cost: " << cost;
            resharding_costs.push_back(cost);
          }
          strat.AddUserReshardingCosts(resharding_costs);
        }
      }

    }
    
    return;
  }

  /*********************************************************/
  /* Additional Helper Functions                           */
  /*********************************************************/

  // TODO: determine if this is a valid approach to determine 
  // if the module is the main module for the computations
  bool ShardableModule(HloModule* module) {

    std::string name = module->name();
    return name.find("convert_element_type") == std::string::npos &&
      name.find("broadcast_in_dim") == std::string::npos &&
      name.find("_multi_slice") == std::string::npos;
  }

}   // namespace


  /*********************************************************/
  /* AutoParallelizer Pass Implementation                  */
  /*********************************************************/

  // overriden functions from class
  // modifies the sharding specification of the instructions in the module
  // TODO: need to ignore the default modules that are not the core computation
  //  e.g. jit_convert_element_type, jit_broadcast_in_dim, jit__multi_slice
  // otherwise, just too much time spent on these things
  absl::StatusOr<bool> AutoParallelizer::Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) {

    if (!ShardableModule(module)) {
      VLOG(5) << LOG_HEADER(0) << "module = " 
        << module->name() << " not shardable";
      return false;
    }

    VLOG(5) << "Testing AutoParallelizer Run";

    // create a clone of the module, then run off of that 
    std::unique_ptr<HloModule> module_clone = module->Clone();
    VLOG(5) << LOG_HEADER(0) << "module: " << module_clone->name();

    std::unordered_map<HloInstruction*, 
      std::unique_ptr<InstructionStrategies>> info_map;

    // TODO: shouldn't I only be doing this for the main computation?
    // construct relevant sharding information
    for (HloComputation* computation : module_clone->computations()) {
      for (HloInstruction* instr : computation->instructions()) {
        assert(info_map.count(instr) == 0);
        info_map[instr] = std::make_unique<InstructionStrategies>(instr);
      }
    }

    VLOG(5) << "Starting to evaluate the resharding costs";
    EstimateReshardingCosts(info_map);

    VLOG(5) << "Number of instructions: " << info_map.size();

    VLOG(5) << "Done Testing AutoParallelizer Run";
    
    return true;
  }

}   // namespace xla