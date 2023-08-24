/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "ducc/google/fft.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <ostream>
#include <thread>
#include <utility>
#include <vector>

#include "ducc/src/ducc0/fft/fft.h"
#include "ducc/src/ducc0/fft/fft1d_impl.h"  // IWYU pragma: keep, DUCC definitions.
#include "ducc/src/ducc0/fft/fftnd_impl.h"  // IWYU pragma: keep, DUCC definitions.
#include "ducc/src/ducc0/infra/mav.h"
#include "ducc/src/ducc0/infra/threading.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/ThreadPool"

namespace ducc0 {
namespace google {

namespace {

// Pseudo thread-pool for single-threaded execution.
class NoThreadPool : public ducc0::detail_threading::thread_pool {
 public:
  size_t nthreads() const override { return 1; }
  size_t adjust_nthreads(size_t nthreads_in) const override { return 1; };
  void submit(std::function<void()> work) override { work(); }
};

// Thread-pool wrapper around Eigen's ThreadPool.
class EigenThreadPool : public ducc0::detail_threading::thread_pool {
 public:
  EigenThreadPool(Eigen::ThreadPoolInterface& pool) : pool_{&pool} {}
  size_t nthreads() const override { return pool_->NumThreads(); }
  size_t adjust_nthreads(size_t nthreads_in) const override {
    // If called by a thread in the pool, return 1
    if (pool_->CurrentThreadId() >= 0) {
      return 1;
    } else if (nthreads_in == 0) {
      return pool_->NumThreads();
    }
    return std::min<size_t>(nthreads_in, pool_->NumThreads());
  };
  void submit(std::function<void()> work) override {
    pool_->Schedule(std::move(work));
  }

 private:
  Eigen::ThreadPoolInterface* pool_;
};

// Default shared global pool.  It is created on first use.
EigenThreadPool* GetGlobalThreadPoolSingleton() {
  static Eigen::ThreadPool* eigen_pool = new Eigen::ThreadPool(std::thread::hardware_concurrency());
  static EigenThreadPool* pool = new EigenThreadPool(*eigen_pool);
  return pool;
}

// Thread-local active pool for current execution.
ducc0::detail_threading::thread_pool*& GetActiveThreadPoolSingleton() {
  thread_local thread_pool* active_pool = nullptr;
  return active_pool;
}

}  // namespace

}  // namespace google

// Implementations required by ducc0.
namespace detail_threading {

thread_pool* set_active_pool(thread_pool* new_pool) {
  return std::exchange(ducc0::google::GetActiveThreadPoolSingleton(), new_pool);
}

thread_pool* get_active_pool() {
  thread_pool* pool = google::GetActiveThreadPoolSingleton();
  if (pool == nullptr) {
    // Set to use a global pool.  This may trigger threadpool creation.
    // Since the active pool is thread-local, this is thread-safe.
    pool = google::GetGlobalThreadPoolSingleton();
    set_active_pool(pool);
  }
  return pool;
}

}  // namespace detail_threading

// Wrappers around DUCC calls.
namespace google {

using Shape = std::vector<std::size_t>;
using Stride = std::vector<std::ptrdiff_t>;

template <typename RealScalar>
void c2c(const std::complex<RealScalar>* in, const Shape& in_shape,
         const Stride& in_stride, std::complex<RealScalar>* out,
         const Shape& out_shape, const Stride& out_stride, const Shape& axes,
         bool forward, RealScalar scale,
         Eigen::ThreadPoolInterface* thread_pool) {
  ducc0::cfmav<std::complex<RealScalar>> m_in(in, in_shape, in_stride);
  ducc0::vfmav<std::complex<RealScalar>> m_out(out, out_shape, out_stride);

  try {
    if (thread_pool == nullptr) {
      // Use a fake threadpool.
      ducc0::google::NoThreadPool no_thread_pool;
      ducc0::detail_threading::ScopedUseThreadPool thread_pool_guard(
          no_thread_pool);
      ducc0::c2c(m_in, m_out, axes, forward, scale, 1);
    } else {
      EigenThreadPool eigen_thread_pool(*thread_pool);
      ducc0::detail_threading::ScopedUseThreadPool thread_pool_guard(
          eigen_thread_pool);
      ducc0::c2c(m_in, m_out, axes, forward, scale,
                 eigen_thread_pool.nthreads());
    }
  } catch (const std::exception& ex) {
    std::cerr << "DUCC FFT c2c failed: " << ex.what() << std::endl;
    std::abort();
  }
}

template <typename RealScalar>
void r2c(const RealScalar* in, const Shape& in_shape, const Stride& in_stride,
         std::complex<RealScalar>* out, const Shape& out_shape,
         const Stride& out_stride, const Shape& axes, bool forward,
         RealScalar scale, Eigen::ThreadPoolInterface* thread_pool) {
  ducc0::cfmav<RealScalar> m_in(in, in_shape, in_stride);
  ducc0::vfmav<std::complex<RealScalar>> m_out(out, out_shape, out_stride);

  try {
    if (thread_pool == nullptr) {
      // Use a fake threadpool.
      ducc0::google::NoThreadPool no_thread_pool;
      ducc0::detail_threading::ScopedUseThreadPool thread_pool_guard(
          no_thread_pool);
      ducc0::r2c(m_in, m_out, axes, forward, scale, 1);
    } else {
      EigenThreadPool eigen_thread_pool(*thread_pool);
      ducc0::detail_threading::ScopedUseThreadPool thread_pool_guard(
          eigen_thread_pool);
      ducc0::r2c(m_in, m_out, axes, forward, scale,
                 eigen_thread_pool.nthreads());
    }
  } catch (const std::exception& ex) {
    std::cerr << "DUCC FFT r2c failed: " << ex.what() << std::endl;
    std::abort();
  }
}

template <typename RealScalar>
void c2r(const std::complex<RealScalar>* in, const Shape& in_shape,
         const Stride& in_stride, RealScalar* out, const Shape& out_shape,
         const Stride& out_stride, const Shape& axes, bool forward,
         RealScalar scale, Eigen::ThreadPoolInterface* thread_pool) {
  ducc0::cfmav<std::complex<RealScalar>> m_in(in, in_shape, in_stride);
  ducc0::vfmav<RealScalar> m_out(out, out_shape, out_stride);

  try {
    if (thread_pool == nullptr) {
      // Use a fake threadpool.
      ducc0::google::NoThreadPool no_thread_pool;
      ducc0::detail_threading::ScopedUseThreadPool thread_pool_guard(
          no_thread_pool);
      ducc0::c2r(m_in, m_out, axes, forward, scale, 1);
    } else {
      EigenThreadPool eigen_thread_pool(*thread_pool);
      ducc0::detail_threading::ScopedUseThreadPool thread_pool_guard(
          eigen_thread_pool);
      ducc0::c2r(m_in, m_out, axes, forward, scale,
                 eigen_thread_pool.nthreads());
    }
  } catch (const std::exception& ex) {
    std::cerr << "DUCC FFT c2r failed: " << ex.what() << std::endl;
    std::abort();
  }
}

#define FFT_DEFINITIONS(RealScalar)                                            \
  template void c2c<RealScalar>(                                               \
      const std::complex<RealScalar>* in, const Shape& in_shape,               \
      const Stride& in_stride, std::complex<RealScalar>* out,                  \
      const Shape& out_shape, const Stride& out_stride, const Shape& axes,     \
      bool forward, RealScalar scale,                                          \
      Eigen::ThreadPoolInterface* thread_pool);                                \
  template void r2c<RealScalar>(                                               \
      const RealScalar* in, const Shape& in_shape, const Stride& in_stride,    \
      std::complex<RealScalar>* out, const Shape& out_shape,                   \
      const Stride& out_stride, const Shape& axes, bool forward,               \
      RealScalar scale, Eigen::ThreadPoolInterface* thread_pool);              \
  template void c2r(const std::complex<RealScalar>* in, const Shape& in_shape, \
                    const Stride& in_stride, RealScalar* out,                  \
                    const Shape& out_shape, const Stride& out_stride,          \
                    const Shape& axes, bool forward, RealScalar scale,         \
                    Eigen::ThreadPoolInterface* thread_pool)
FFT_DEFINITIONS(float);
FFT_DEFINITIONS(double);
#undef FFT_DEFINITIONS

}  // namespace google
}  // namespace ducc0