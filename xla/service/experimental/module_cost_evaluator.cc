// TODO: add license

#include "xla/service/experimental/module_cost_evaluator.h"

#include "xla/hlo/ir/hlo_instructions.h"

namespace xla {

namespace {

    // This function returns the number of bytes of a primitive type
    uint64_t NumBytes(PrimitiveType type) {
    }

    // This function evaluates an AllGather communication instruction
    uint64_t EvaluateAllGather(const HloAllGatherInstruction* instr) {

        // get data type of operands

        // 

        return 1;
    }

    // This function evaluates an AllReduce communication instruction
    uint64_t EvaluateAllReduce(const HloAllReduceInstruction* instr) {
        return 1;
    }

    // This function returns an interpretation of the cost of the input
    // HloModule. Currently, this implementation returns the number of
    // bytes that are communicated in the various communication operations
    uint64_t EvaluateCommunicationCost(const HloModule* module) {

        uint64_t cost = 0;

        // iterate through computation and instructions
        // evaluate the cost of each
        for (const HloComputation* comp: module->computations()) {
            for (const HloInstruction* instr: comp->instructions()) {
                switch (instr->opcode()) {
                case HloOpcode::kAllGather:
                    cost += EvaluateAllGather(
                        static_cast<const HloAllGatherInstruction*>(instr)
                    );
                    break;
                case HloOpcode::kAllReduce:
                    cost += EvaluateAllReduce(
                        static_cast<const HloAllReduceInstruction*>(instr)
                    );
                    break;
                case HloOpcode::kCollectiveBroadcast:
                case HloOpcode::kReduce:
                case HloOpcode::kReduceScatter:
                case HloOpcode::kScatter:
                default:
                    break;
                }
            }
        } 
        
        return cost;
    }
    
}  // namespace

    // This function computes the cost of an HloModule following a cost model
    uint64_t ModuleCostEvaluator::Evaluate(const HloModule* module) {
        return EvaluateCommunicationCost(module);
    }

}  // namespace xla