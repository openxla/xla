/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_ALL_REDUCE_COMBINER_H_
#define XLA_SERVICE_ALL_REDUCE_COMBINER_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "xla/array2d.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/collective_combiner_utils.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Combines small non-dependent AllReduce ops into larger combined
// AllReduce ops. A typical AllReduce implementation has a minimum
// latency-induced time for a AllReduce op so a single combined op can be
// more efficient than many small ones.
class AllReduceCombiner : public HloModulePass {
 public:
  AllReduceCombiner(int64_t combine_threshold_in_bytes,
                    int64_t combine_threshold_count, bool is_async = false,
                    std::string_view async_strategy = "trivial");

  absl::string_view name() const override { return "all-reduce-combiner"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Combine all reduce ops up to this threshold.
  int64_t combine_threshold_in_bytes_;

  // Combine all reduce ops up to this threshold (number of operands).
  int64_t combine_threshold_count_;

  // Whether to operate on Async pairs. Only works post-scheduling.
  bool is_async_;

  AsyncCombinerStrategy async_strategy_;
};

class AsyncAllReduceCombiner : public AllReduceCombiner {
 public:
  AsyncAllReduceCombiner(std::string_view strategy = "trivial")
      : AllReduceCombiner(
            /*combine_threshold_in_bytes=*/INT64_MAX,
            /*combine_threshold_count=*/INT64_MAX,
            /*is_async=*/true, strategy) {}

  absl::string_view name() const override {
    return "async-all-reduce-combiner";
  }
};

}  // namespace xla

#endif  // XLA_SERVICE_ALL_REDUCE_COMBINER_H_
