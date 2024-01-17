/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <optional>
#include <vector>

#include "xla/hlo/experimental/auto_sharding/auto_sharding.pb.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_solver.h"
#include "ortools/math_opt/cpp/math_opt.h"

namespace xla {
namespace spmd {

using ::operations_research::math_opt::LinearExpression;
using ::operations_research::math_opt::Model;
using ::operations_research::math_opt::Variable;

std::optional<Variable> CreateMakespanVar(
    const AutoShardingSolverRequest& request,
    const std::vector<std::vector<Variable>>& e, Model& model,
    LinearExpression& objective_expression) {
  return std::nullopt;  // TODO(moffitt): Implement this.
}

double EvaluateMakespan(const AutoShardingSolverRequest& request,
                        const AutoShardingSolverResult& result,
                        AutoShardingEvaluation& evaluation) {
  return 0.0;  // TODO(moffitt): Implement this.
}

}  // namespace spmd
}  // namespace xla
