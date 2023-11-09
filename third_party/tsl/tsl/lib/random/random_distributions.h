/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_LIB_RANDOM_RANDOM_DISTRIBUTIONS_H_
#define TENSORFLOW_TSL_LIB_RANDOM_RANDOM_DISTRIBUTIONS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>

#include "Eigen/Core"  // from @eigen_archive
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tsl/lib/math/math_util.h"
#include "tsl/lib/random/philox_random.h"
#include "tsl/lib/random/random_distributions_utils.h"
#include "tsl/platform/types.h"

namespace tsl {
namespace random {

// Computes a + b. Requires that the result is representable in the destination
// type and that b is not maximal (i.e. b + 1 is not 0). Notably, the addend b
// need *not* be representable in that type. (The condition on b excludes the
// extremal case INT_MIN + UINT_MAX = INT_MAX, which this function cannot
// compute.)
template <typename Int>
PHILOX_DEVICE_INLINE Int SignedAdd(Int a,
                                   typename std::make_unsigned<Int>::type b) {
  // Implementation note: both b_div_2 and b - b_div_2 are positive and
  // representable as Int.
  auto b_div_2 = b >> 1;
  return a + static_cast<Int>(b_div_2) + static_cast<Int>(b - b_div_2);
}

// A class that generates uniform distribution random numbers from the
// underlying random integer generator.
// Arguments:
//   Generator: a generator type that returns a number of uint32 upon each
//              invocation. It needs to define kResultElementCount for the
//              sample count for each invocation, and ResultType for the
//              actual returned sample type.
//   RealType: the data type of the real numbers that will be returned by the
//             distribution. This could be either float or double for now.
// This class is meant to be implemented through specialization. The default
// is not defined by design.
template <class Generator, typename RealType, typename = void>
class UniformDistribution {};

namespace internal {
template <typename T>
using RequiresFloatingPoint =
    std::enable_if_t<std::numeric_limits<T>::is_specialized &&
                     !std::numeric_limits<T>::is_integer>;
}  // namespace internal

template <class Generator, typename RealType>
class UniformDistribution<Generator, RealType,
                          internal::RequiresFloatingPoint<RealType>> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount =
      Generator::kResultElementCount /
      MathUtil::CeilOfRatio(sizeof(RealType), sizeof(uint32_t));
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<RealType, kResultElementCount> ResultType;
  typedef RealType ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; ++i) {
      if constexpr (std::is_same_v<RealType, double>) {
        result[i] = Uint64ToDouble(sample[2 * i], sample[2 * i + 1]);
      } else {
        result[i] = UintToFloat<RealType>(sample[i]);
      }
    }
    return result;
  }
};

template <class Generator>
class UniformDistribution<Generator, int32> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<int32, kResultElementCount> ResultType;
  typedef int32 ResultElementType;

  // Must have lo < hi
  UniformDistribution(int32_t lo, int32_t hi)
      : lo_(lo), range_(static_cast<uint32>(hi) - static_cast<uint32>(lo)) {}

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; ++i) {
      result[i] = SignedAdd(lo_, sample[i] % range_);
    }
    return result;
  }

 private:
  // Note that lo_ is intentionally signed while range_ is intentionally
  // unsigned.  This is because hi - lo can overflow signed integers if
  // lo < 0 < hi, but always fits in unsigned.
  int32 lo_;
  uint32 range_;
};

template <class Generator>
class UniformDistribution<Generator, int64_t> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = Generator::kResultElementCount / 2;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<int64_t, kResultElementCount> ResultType;
  typedef int64_t ResultElementType;

  // Must have lo < hi
  UniformDistribution(int64_t lo, int64_t hi)
      : lo_(lo), range_(static_cast<uint64>(hi) - static_cast<uint64>(lo)) {}

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; ++i) {
      auto bits = sample[2 * i] | static_cast<uint64>(sample[2 * i + 1]) << 32;
      result[i] = SignedAdd(lo_, bits % range_);
    }
    return result;
  }

 private:
  // Note that lo_ is intentionally signed while range_ is intentionally
  // unsigned.  This is because hi - lo can overflow signed integers if
  // lo < 0 < hi, but always fits in unsigned.
  int64_t lo_;
  uint64 range_;
};

// Similar to `UniformDistribution`, except that instead of generating numbers
// in the range [low, high), it generates numbers covering the whole range of
// the integer type.
template <typename Generator, typename IntType>
class UniformFullIntDistribution;

template <typename Generator, typename IntType>
class UniformFullIntDistribution32 {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<IntType, kResultElementCount> ResultType;
  typedef IntType ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; ++i) {
      result[i] = sample[i];
    }
    return result;
  }
};

template <typename Generator, typename IntType>
class UniformFullIntDistribution64 {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = Generator::kResultElementCount / 2;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<IntType, kResultElementCount> ResultType;
  typedef IntType ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; ++i) {
      result[i] = sample[2 * i] | static_cast<uint64>(sample[2 * i + 1]) << 32;
    }
    return result;
  }
};

template <typename Generator>
class UniformFullIntDistribution<Generator, int32>
    : public UniformFullIntDistribution32<Generator, int32> {};
template <typename Generator>
class UniformFullIntDistribution<Generator, uint32>
    : public UniformFullIntDistribution32<Generator, uint32> {};
template <typename Generator>
class UniformFullIntDistribution<Generator, int64_t>
    : public UniformFullIntDistribution64<Generator, int64_t> {};
template <typename Generator>
class UniformFullIntDistribution<Generator, uint64>
    : public UniformFullIntDistribution64<Generator, uint64> {};

// A class that adapts the underlying native multiple samples to return a single
// sample at a time.
template <class Generator>
class SingleSampleAdapter {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = 1;
  // The number of elements that will be returned by the underlying generator.
  static constexpr int kNativeElementCount = Generator::kResultElementCount;
  typedef typename Generator::ResultElementType ResultType;
  typedef typename Generator::ResultElementType ResultElementType;

  PHILOX_DEVICE_INLINE
  explicit SingleSampleAdapter(Generator* gen)
      : generator_(gen), used_result_index_(Generator::kResultElementCount) {}

  PHILOX_DEVICE_INLINE
  ResultType operator()() {
    if (used_result_index_ == Generator::kResultElementCount) {
      unused_results_ = (*generator_)();
      used_result_index_ = 0;
    }

    return unused_results_[used_result_index_++];
  }

  PHILOX_DEVICE_INLINE
  void Skip(uint64 num_skips) {
    if (!num_skips) {
      return;
    }
    int num_unused_results = kNativeElementCount - used_result_index_;
    if (num_skips <= num_unused_results) {
      used_result_index_ += num_skips;
      return;
    }
    num_skips -= num_unused_results;
    used_result_index_ = kNativeElementCount;
    SkipFromGenerator(num_skips / kNativeElementCount);
    num_skips = num_skips % kNativeElementCount;
    if (num_skips) {
      unused_results_ = (*generator_)();
      used_result_index_ = num_skips;
    }
  }

 private:
  // This implementation iteratively skips over `num_skips` samples
  // from `generator_`. There is an O(1) implementation for PhiloxRandom
  // in random_distributions.cc.
  PHILOX_DEVICE_INLINE
  void SkipFromGenerator(uint64 num_skips) {
    while (num_skips--) {
      (*generator_)();
    }
  }

  Generator* generator_;
  typename Generator::ResultType unused_results_;
  int used_result_index_;
};

// A class that generates unit normal distribution random numbers from the
// underlying random integer generator.
// Arguments:
//   Generator: a generator type that returns a number of uint32 upon each
//              each invocation. It needs to define kResultElementCount for the
//              sample count for each invocation, and ResultType for actual
//              returned sample type.
//   RealType: the data type of the real numbers that will be returned by the
//             distribution. This could be either float or double for now.
// This class is meant to be implemented through specialization. The default
// is not defined by design.
template <class Generator, typename RealType, typename = void>
class NormalDistribution {};

// Exactly like the float version, except that we convert to half afterwards;
// since we don't have half-precision sin/cos even on GPUs, there's nothing to
// gain from working in half internally.
template <class Generator, typename RealType>
class NormalDistribution<Generator, RealType,
                         internal::RequiresFloatingPoint<RealType>> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 70;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<RealType, kResultElementCount> ResultType;
  typedef RealType ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; i += 2) {
      if constexpr (std::is_same_v<RealType, double>) {
        const int i2 = 2 * i;
        uint64_t x0 = (static_cast<uint64_t>(sample[i2]) << 32) |
                      static_cast<uint64_t>(sample[i2 + 1]);
        uint64_t x1 = (static_cast<uint64_t>(sample[i2 + 2]) << 32) |
                      static_cast<uint64_t>(sample[i2 + 3]);
        std::tie(result[i], result[i + 1]) = BoxMullerFloat<double>(x0, x1);
      } else {
        float f[2];
        std::tie(f[0], f[1]) = BoxMullerFloat<float>(sample[i], sample[i + 1]);
        result[i] = static_cast<RealType>(f[0]);
        result[i + 1] = static_cast<RealType>(f[1]);
      }
    }
    return result;
  }
};

// A class that returns standard normal distribution between
// [-kTruncateValue, kTruncateValue].
// Arguments:
//   Generator: a generator type that returns a number of uint32 upon each
//              each invocation. It needs to define kResultElementCount for the
//              sample count for each invocation, and ResultType for actual
//              returned sample type.
//   RealType: the data type of the real numbers that will be returned by the
//             distribution. This could be either float or double for now.
// This class is meant to be implemented through specialization. The default
// is not defined by design.
template <class SingleSampleGenerator, typename RealType, typename = void>
class TruncatedNormalDistribution {};

// Exactly like the float version, except that we convert to half afterwards;
// since we don't have half-precision sin/cos even on GPUs, there's nothing to
// gain from working in half internally.
template <class SingleSampleGenerator, typename RealType>
class TruncatedNormalDistribution<SingleSampleGenerator, RealType,
                                  internal::RequiresFloatingPoint<RealType>> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount =
      SingleSampleGenerator::kNativeElementCount;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 90;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = true;
  // The threshold where the normal distribution is truncated.
  const float kTruncateValue = 2.0f;

  typedef Array<RealType, kResultElementCount> ResultType;
  typedef RealType ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(SingleSampleGenerator* gen) {
    ResultType results;
    int index = 0;
    while (true) {
      // Repeatedly take samples from the normal distribution, until we have
      // the desired number of elements that fall within the predefined cutoff
      // threshold.
      const uint32 x0 = (*gen)();
      const uint32 x1 = (*gen)();
      float f[2];
      std::tie(f[0], f[1]) = BoxMullerFloat<float>(x0, x1);

      if (Eigen::numext::abs(f[0]) < kTruncateValue) {
        results[index++] = static_cast<RealType>(f[0]);
        if (index >= kResultElementCount) {
          return results;
        }
      }
      if (Eigen::numext::abs(f[1]) < kTruncateValue) {
        results[index++] = static_cast<RealType>(f[1]);
        if (index >= kResultElementCount) {
          return results;
        }
      }
    }
  }
};

// Partial specialization for double.
template <class SingleSampleGenerator>
class TruncatedNormalDistribution<SingleSampleGenerator, double> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount =
      (SingleSampleGenerator::kNativeElementCount > 1)
          ? SingleSampleGenerator::kNativeElementCount / 2
          : 1;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 90;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = true;
  typedef Array<double, kResultElementCount> ResultType;
  typedef double ResultElementType;
  const double kTruncateValue = 2.0;

  PHILOX_DEVICE_INLINE
  ResultType operator()(SingleSampleGenerator* gen) {
    ResultType results;
    int index = 0;
    while (true) {
      const uint32 s0 = (*gen)();
      const uint32 s1 = (*gen)();
      const uint32 s2 = (*gen)();
      const uint32 s3 = (*gen)();
      uint64_t x0 =
          (static_cast<uint64_t>(s0) << 32) | static_cast<uint64_t>(s1);
      uint64_t x1 =
          (static_cast<uint64_t>(s2) << 32) | static_cast<uint64_t>(s3);
      double d[2];
      std::tie(d[0], d[1]) = BoxMullerFloat<double>(x0, x1);

      if (Eigen::numext::abs(d[0]) < kTruncateValue) {
        results[index++] = d[0];
        if (index >= kResultElementCount) {
          return results;
        }
      }
      if (Eigen::numext::abs(d[1]) < kTruncateValue) {
        results[index++] = d[1];
        if (index >= kResultElementCount) {
          return results;
        }
      }
    }
  }
};

}  // namespace random
}  // namespace tsl

#endif  // TENSORFLOW_TSL_LIB_RANDOM_RANDOM_DISTRIBUTIONS_H_
