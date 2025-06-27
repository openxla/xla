#include <memory>

#include "xla/backends/profiler/gpu/cupti_pm_sampler.h"

#if CUPTI_PM_SAMPLING_SUPPORTED
#include "xla/backends/profiler/gpu/cupti_pm_sampler_impl.h"
#else
#include "xla/backends/profiler/gpu/cupti_pm_sampler_stub.h"
#endif

#include "xla/backends/profiler/gpu/cupti_pm_sampler_factory.h"

namespace xla {
namespace profiler {

std::unique_ptr<CuptiPmSampler> CreatePmSampler(
    size_t num_gpus, CuptiPmSamplerOptions& options) {
#if CUPTI_PM_SAMPLING_SUPPORTED
    return std::make_unique<CuptiPmSamplerImpl>(num_gpus, options);
#else
    return std::make_unique<CuptiPmSamplerStub>(num_gpus, options);
#endif
}

}  // namespace profiler
}  // namespace xla
