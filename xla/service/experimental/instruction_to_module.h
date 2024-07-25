// TODO: license

#ifndef XLA_SERVICE_EXPERIMENTAL_INSTRUCTION_TO_MODULE_H_
#define XLA_SERVICE_EXPERIMENTAL_INSTRUCTION_TO_MODULE_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// Creates a module from a single instruction for running a simple pass on
std::unique_ptr<HloModule> CreateModuleFromInstruction(
    const HloInstruction* instruction);

} // xla

#endif // XLA_SERVICE_EXPERIMENTAL_INSTRUCTION_TO_MODULE_H_