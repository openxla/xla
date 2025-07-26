#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_PM_SAMPLER_FACTORY_H_

#include <memory>

#include "xla/backends/profiler/gpu/cupti_pm_sampler.h"

namespace xla {
namespace profiler {

std::unique_ptr<CuptiPmSampler> CreatePmSampler(
    size_t num_gpus, CuptiPmSamplerOptions& options);

}  // namespace profiler
}  // namespace xla

#endif // XLA_BACKENDS_PROFILER_GPU_CUPTI_PM_SAMPLER_FACTORY_H_
