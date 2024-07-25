// TODO: license

#ifndef XLA_SERVICE_EXPERIMENTAL_DEBUG_H_
#define XLA_SERVICE_EXPERIMENTAL_DEBUG_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/errors.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"

#include <string>
#include <stdint.h>

namespace xla {

std::string LOG_HEADER(int x, const char c[]="AutoParallel: ");
void PrintShardingInfo(OpSharding sharding, int depth=3); 
void PrintInstructionInfo(HloInstruction* instruction, int depth=3);
void PrintComputationInfo(HloComputation* computation, int depth=3);
void PrintModuleInfo(HloModule* module, int depth=1);

} // xla

#endif // XLA_SERVICE_EXPERIMENTAL_DEBUG_H_