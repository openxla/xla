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

#include "xla/service/cpu/runtime_fft_impl.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/test.h"

TEST(FftTypeTest, MatchesProto) {
  EXPECT_EQ(::xla::FftType_ARRAYSIZE, 4);
  EXPECT_EQ(::xla::internal::FftTypeArraySize(), 4);
  EXPECT_EQ(::xla::FftType::FFT,
            static_cast<int32_t>(::xla::internal::FftType::FFT));
  EXPECT_EQ(::xla::FftType::IFFT,
            static_cast<int32_t>(::xla::internal::FftType::IFFT));
  EXPECT_EQ(::xla::FftType::RFFT,
            static_cast<int32_t>(::xla::internal::FftType::RFFT));
  EXPECT_EQ(::xla::FftType::IRFFT,
            static_cast<int32_t>(::xla::internal::FftType::IRFFT));
}
