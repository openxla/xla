#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_PM_SAMPLER_STUB_H_

#include "xla/backends/profiler/gpu/cupti_pm_sampler.h"

namespace xla {
namespace profiler {

class CuptiPmSamplerStub : public CuptiPmSampler {
  // Just declare prototypes for required interfaces
  public:
  CuptiPmSamplerStub() = default;
  CuptiPmSamplerStub(size_t num_gpus, CuptiPmSamplerOptions* options);
  ~CuptiPmSamplerStub() override = default;
  absl::Status Initialize(size_t num_gpus, CuptiPmSamplerOptions& options) override;
  absl::Status StartSampler() override;
  absl::Status StopSampler() override;
  absl::Status Deinitialize() override;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_PM_SAMPLER_STUB_H_
