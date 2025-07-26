#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_STATUS_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_STATUS_H_

#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_result.h"
#include "absl/status/status.h"

namespace xla {
namespace profiler {

absl::Status ToStatus(CUptiResult result);

}
}

#endif // XLA_BACKENDS_PROFILER_GPU_CUPTI_STATUS_H_
