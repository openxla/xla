/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_CPU_RUNTIME_FFT_H_
#define XLA_SERVICE_CPU_RUNTIME_FFT_H_

#include <stdint.h>

namespace xla {
namespace internal {

enum class FftType : int32_t {
  FFT = 0,    // Forward FFT; complex in, complex out.
  IFFT = 1,   // Inverse FFT; complex in, complex out.
  RFFT = 2,   // Forward real FFT; real in, fft_length / 2 + 1 complex out
  IRFFT = 3,  // Inverse real FFT; fft_length / 2 + 1 complex in,
              //                   fft_length real out
};

}  // namespace internal
}  // namespace xla

extern "C" {

extern void __xla_cpu_runtime_DuccFft(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr, void* out,
    void* operand, int32_t fft_type, int32_t double_precision, int32_t fft_rank,
    int64_t input_batch, int64_t const* fft_lengths);

extern void __xla_cpu_runtime_DuccSingleThreadedFft(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr, void* out,
    void* operand, int32_t fft_type, int32_t double_precision, int32_t fft_rank,
    int64_t input_batch, int64_t const* fft_lengths);

}  // extern "C"

#endif  // XLA_SERVICE_CPU_RUNTIME_FFT_H_
