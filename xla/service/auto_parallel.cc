// TODO: add copyright 

#include "xla/service/auto_parallel.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo_pass_interface.h"
#include "tsl/platform/logging.h"

namespace xla {

namespace {

  // static functions only to this file itself

}   // namespace

  // overriden functions from class
  absl::StatusOr<bool> AutoParallelizer::Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) {

    VLOG(5) << "Hi!";
    
    return true;
  }

    

    
}   // namespace xla