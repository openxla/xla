// TODO: add copyright


#ifndef XLA_SERVICE_CPU_AUTO_PARALLEL_H_
#define XLA_SERVICE_CPU_AUTO_PARALLEL_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo_pass_interface.h"



/*

Interface:
    - this code will involve taking modules and converting them into modules with an optimal suggested sharding
    - we don't want to overwrite the shardings that were specified by a user
    - ideally we are adding a pass over the code which will make the appropriate modifications
    - I believe that the only thing that this interface will provide is a pass for auto-parallelization
    - will need to determine the type of computer configuration

*/

namespace xla {

  // Pass that searches for a near-optimal sharding strategy for an
  // HloModule to maximize cluster usage. Requires knowledge of cluster
  // configurations. Maintains user-specified shardings and only
  // inserts new ones 
  class AutoParallelizer : public HloModulePass {
  public:
    AutoParallelizer();
    ~AutoParallelizer() = default;

    absl::string_view name() const override { return "auto-parallelizer"; }

    using HloPassInterface::Run;
    absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads
    ) override;


  };
}

#endif // XLA_SERVICE_CPU_AUTO_PARALLEL_H_