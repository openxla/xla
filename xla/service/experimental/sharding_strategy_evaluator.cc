// TODO: license

#include "xla/service/experimental/sharding_strategy_evaluator.h"

#include "xla/service/experimental/module_cost_evaluator.h"

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/sharding_propagation.h"
#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"

#include "tsl/platform/errors.h"

// TODO: only here because of mesh information
#include "xla/service/experimental/sharding_enumeration.h"

namespace xla {

/*********************************************************/
/* GSPMD Completion                                      */
/*********************************************************/
// sharding_evaluator.{h,cc}

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
  strat->ApplyToModule(eval_module.get());
  strat->set_result_sharding(RunGSPMD(eval_module.get()));

  // now evaluate cost
  ModuleCostEvaluator evaluator;
  strat->set_cost(evaluator.Evaluate(eval_module.get()));
  
  // update strat with cost and root instruction's output sharding
  // NOTE: the eval_module after GSPMD doesn't have it's sharding
  // in the output (i.e. root computation of entry computation) which
  // is unexpected, I thought that GSPMD would fill that in

}

}