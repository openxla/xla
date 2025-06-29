#include "xla/backends/profiler/gpu/cupti_pm_sampler.h"

#include "xla/backends/profiler/gpu/cupti_pm_sampler_stub.h"

namespace xla {
namespace profiler {

// Stub implementation of CuptiPmSampler
absl::Status CuptiPmSamplerStub::Initialize(size_t num_gpus,
                                             CuptiPmSamplerOptions& options) {
  return absl::OkStatus();
}
absl::Status CuptiPmSamplerStub::StartSampler() {
  return absl::OkStatus();
}
absl::Status CuptiPmSamplerStub::StopSampler() {
  return absl::OkStatus();
}
absl::Status CuptiPmSamplerStub::Deinitialize() {
  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace xla
