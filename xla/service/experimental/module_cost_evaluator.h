// TODO: add license

#ifndef XLA_SERVICE_EXPERIMENTAL_MODULE_COST_EVALUATOR_H_
#define XLA_SERVICE_EXPERIMENTAL_MODULE_COST_EVALUATOR_H_

#include "xla/hlo/ir/hlo_module.h"

#include <stdint.h>

// Class for evaluating the cost of a module based on various configurations

namespace xla {

// TODO: add ability to configure cost evaluator for various cost methods
class ModuleCostEvaluator {
public:
  ModuleCostEvaluator() = default;
  ~ModuleCostEvaluator() = default;

  uint64_t Evaluate(const HloModule* module);
};

} // xla

#endif // XLA_SERVICE_EXPERIMENTAL_MODULE_COST_EVALUATOR_H_