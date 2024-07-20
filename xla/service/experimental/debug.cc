// TODO: license

#include "xla/service/experimental/debug.h"

/*********************************************************/
/* Debugging                                             */
/*********************************************************/
// TODO: move into separate file

namespace xla {

std::string LOG_HEADER(int x, const char c[] /* "AutoParallel: " */) {
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

void PrintShardingInfo(OpSharding sharding, int depth /* 3 */) {

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

void PrintInstructionInfo(HloInstruction* instruction, int depth /* 3 */) {

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

void PrintComputationInfo(HloComputation* computation, int depth /* 3 */) {
  VLOG(5) << LOG_HEADER(depth, "CompInfo: ") << "Name: " << computation->name() << " " << computation;
  VLOG(5) << LOG_HEADER(depth, "CompInfo: ") << "Instruction Count: " << computation->instruction_count();

  for (HloInstruction* instr : computation->instructions()) {
    PrintInstructionInfo(instr, depth + 1);
  }
}

void PrintModuleInfo(HloModule* module, int depth /* 1 */) {

  VLOG(5) << LOG_HEADER(depth, "ModuInfo: ") << "Name: " << module->name() << " " << module;
  VLOG(5) << LOG_HEADER(depth, "ModuInfo: ") << "Computation count: " << module->computation_count();

  for (HloComputation* computation : module->computations()) {
    PrintComputationInfo(computation, depth + 1);
  }

}

} // xla