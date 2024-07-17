// TODO: add license

#include "xla/service/experimental/module_cost_evaluator.h"

#include "tsl/platform/logging.h"
#include "tsl/platform/errors.h"
#include "xla/hlo/ir/hlo_instructions.h"

namespace xla {

namespace {

  // This function returns the number of bytes of a primitive type
  uint64_t NumBytesFromType(const PrimitiveType type) {
    assert(shape.IsArray());

    switch (type) {
    case PrimitiveType::PRED:
    case PrimitiveType::S2:
    case PrimitiveType::S4:
    case PrimitiveType::S8:
    case PrimitiveType::U2:
    case PrimitiveType::U4:
    case PrimitiveType::U8:
    case PrimitiveType::F8E5M2:
    case PrimitiveType::F8E4M3FN:
    case PrimitiveType::F8E4M3B11FNUZ:
    case PrimitiveType::F8E5M2FNUZ:
    case PrimitiveType::F8E4M3FNUZ:
      return 1;
    case PrimitiveType::S16:
    case PrimitiveType::U16:
    case PrimitiveType::F16:
      return 2;
    case PrimitiveType::S32:
    case PrimitiveType::U32:
    case PrimitiveType::F32:
      return 4;
    case PrimitiveType::S64:
    case PrimitiveType::U64:
    case PrimitiveType::F64:
    case PrimitiveType::C64:
      return 8;
    case PrimitiveType::C128:
      return 16;
    default:
      // TODO: determine appropriate way to support the other value types
      assert(0);
      return 0;
    }
  }

  // This function returns the number of elements in a shape
  uint64_t NumElementsFromShape(const Shape& shape) {
    assert(shape.IsArray());

    uint64_t num_elems = 1;
    int num_dims = shape.dimensions_size();

    for (int i = 0; i < num_dims; i++) {
      num_elems *= shape.dimensions(i);
    }

    return num_elems;
  }

  // This function returns the number of bytes taken by a shape
  uint64_t NumBytesFromShape(const Shape& shape) {
    assert(!shape.IsToken() && !shape.IsOpaque());

    // base-case in a tuple-tree is an array
    if (shape.IsArray()) {
      return NumElementsFromShape(shape) 
          * NumBytesFromType(shape.element_type());
    }

    assert(shape.IsTuple());
    int num_tuple_shapes = shape.tuple_shapes_size();
    uint64_t total_bytes = 0;
    for (int i = 0; i < num_tuple_shapes; i++) {
      total_bytes += NumBytesFromShape(shape.tuple_shapes(i));
    }

    return total_bytes;
  }

  // This function evaluates an AllGather communication instruction
  uint64_t EvaluateAllGather(const HloAllGatherInstruction* instr) {
    // TODO: assumptions, are there multiple operands, or just a
    // single one that is being gathered?
    assert(instr->operand_count() == 1);
    assert(instr->operand(0)->shape().IsArray());

    int max_group_size = 0;
    uint64_t op_bytes = NumBytesFromShape(instr->operand(0)->shape());

    // get the maximum between replica groups
    std::vector<ReplicaGroup> replica_groups = instr->replica_groups();
    for (const ReplicaGroup& rg : replica_groups) {
      // each replica group transporting some amount of data
      // each device sending some slice to (n - 1) other devices in group
      max_group_size = std::max(max_group_size, rg.replica_ids_size());
    }

    return (max_group_size - 1) * op_bytes;
  }

  // This function evaluates an AllReduce communication instruction
  uint64_t EvaluateAllReduce(const HloAllReduceInstruction* instr) {

    // TODO: approximating with the size of resulting shape
    return NumBytesFromShape(instr->shape());
  }

  uint64_t EvaluateCollectiveBroadcast(
      const HloCollectiveBroadcastInstruction* instr) {
    
    // send out data to all n - 1 devices in device group
    assert(instr->operand_count() == 1);
    assert(instr->opernad(0)->shape().IsArray());

    // NOTE: relatively same implementation as AllGather
    // TODO: determine if correct
    int max_group_size = 0;
    uint64_t op_bytes = NumBytesFromShape(instr->operand(0)->shape());

    // get the maximum between replica groups
    std::vector<ReplicaGroup> replica_groups = instr->replica_groups();
    for (const ReplicaGroup& rg : replica_groups) {
        max_group_size = std::max(max_group_size, rg.replica_ids_size());
    }

    return (max_group_size - 1) * op_bytes;
  }

  // This function returns an interpretation of the cost of the input
  // HloModule. Currently, this implementation returns the number of
  // bytes that are communicated in the various communication operations
  uint64_t EvaluateCommunicationCost(const HloModule* module) {

    uint64_t cost = 0;

    // iterate through computation and instructions
    // evaluate the cost of each
    // shoudl be going through the main computation
    for (const HloComputation* comp: module->computations()) {
      for (const HloInstruction* instr: comp->instructions()) {
        VLOG(5) << "Evaluating instruction: " << instr->name();
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
          cost += EvaluateCollectiveBroadcast(
              static_cast<const HloCollectiveBroadcastInstruction*>(instr)
          );
          break;       
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