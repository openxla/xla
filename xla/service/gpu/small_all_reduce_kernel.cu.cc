/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <type_traits>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "third_party/gpus/cuda/include/cuda_runtime.h"
#include "xla/status.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"

namespace xla {
namespace se = stream_executor;
namespace gpu {

#define CUDA_RETURN_IF_ERROR(expression)                                  \
  do {                                                                    \
    if (auto err = expression)                                            \
      return InternalError("CUDA error at %s:%i: %s", __FILE__, __LINE__, \
                           GetErrorMessage(err));                         \
  } while (0)

static const char* GetErrorMessage(cudaError_t err) {
  return cudaGetErrorString(err);
}

static const char* GetErrorMessage(CUresult result) {
  const char* str = nullptr;
  cuGetErrorString(result, &str);
  return str ? str : "<unknown>";
}

static const int kMaxNumGpus = 16;
static const int kLaunchBounds = 256;

// Like std::array<T, kMaxNumGpus>, without the need for `relaxed-constexpr`.
template <typename T>
struct Array {
  __device__ constexpr const T& operator[](int i) { return data[i]; }

 private:
  T data[kMaxNumGpus];
};

struct float2 {
  __device__ explicit float2(__nv_bfloat162 value)
      : x(__bfloat162float(value.x)), y(__bfloat162float(value.y)) {}
  __device__ operator __nv_bfloat162() const {
    __nv_bfloat162 result;
    result.x = __float2bfloat16_rn(x);
    result.y = __float2bfloat16_rn(y);
    return result;
  }
  __device__ float2& operator+=(const float2& rhs) {
    x += rhs.x;
    y += rhs.y;
    return *this;
  }

 private:
  float x, y;
};

template <typename T>
struct MathType {
  using type = T;
};
template <>
struct MathType<__nv_bfloat16> {
  using type = float;
};
template <>
struct MathType<__nv_bfloat162> {
  using type = float2;
};

template <typename T>
static __global__ void __launch_bounds__(kLaunchBounds)
    SmallAllReduceKernel(int num_gpus, Array<const T* __restrict> send_buffers,
                         Array<T* __restrict> recv_buffers,
                         int64_t num_elements) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= num_elements) return;

  T vals[kMaxNumGpus];
  for (int i = 0; i < kMaxNumGpus; ++i) {
    if (i >= num_gpus) break;
    vals[i] = send_buffers[i][tid];
  }
  using MathType = typename MathType<T>::type;
  MathType result = static_cast<MathType>(vals[0]);
  for (int i = 1; i < kMaxNumGpus; ++i) {
    if (i >= num_gpus) break;
    result += static_cast<MathType>(vals[i]);
  }
  for (int i = 0; i < kMaxNumGpus; ++i) {
    if (i >= num_gpus) break;
    recv_buffers[i][tid] = result;
  }
}

// Mark a point in 'stream' for others to wait on later.
static Status RecordStream(se::Stream& stream) {
  CUDA_RETURN_IF_ERROR(
      cuEventRecord(*se::gpu::AsGpuStream(&stream)->completed_event(),
                    se::gpu::AsGpuStreamValue(&stream)));
  return OkStatus();
}

// Make 'stream' wait for all work of 'other' stream up to when 'RecordStream'
// was called.
static Status WaitForStream(se::Stream& stream, se::Stream& other) {
  CUDA_RETURN_IF_ERROR(cuStreamWaitEvent(
      se::gpu::AsGpuStreamValue(&stream),
      *se::gpu::AsGpuStream(&other)->completed_event(), /*flags=*/0));
  return OkStatus();
}

namespace {
class SmallAllReduce {
 public:
  Status Add(se::Stream& stream, int num_gpus, const void* send_buffer,
             void* recv_buffer, PrimitiveType dtype, int64_t num_elements) {
    if (num_gpus_ != num_gpus) {
      assert(num_gpus > 0 && num_gpus <= kMaxNumGpus);
      assert(num_pending_ == num_gpus_ && "inconsistent num_gpus");
      num_pending_ = num_gpus_ = num_gpus;
    }

    // Start filling buffers for next GPU when stream changes.
    if (num_pending_ == num_gpus_ || &stream != streams_[num_pending_]) {
      --num_pending_;
      buffer_idx_ = 0;
    }

    VLOG(0) << "xla::gpu::SmallAllReduce::Add(stream=" << &stream
            << ", device=" << num_gpus - num_pending_ << "/" << num_gpus
            << ", send_buffer=" << send_buffer
            << ", recv_buffer=" << recv_buffer
            << ", dtype=" << PrimitiveType_Name(dtype)
            << ", num_elements=" << num_elements
            << ", buffer_idx=" << buffer_idx_ << ")";

    streams_[num_pending_] = &stream;
    if (buffer_idx_ == 0 && num_pending_ > 0) {
      TF_RETURN_IF_ERROR(RecordStream(stream));
    }

    if (buffers_.size() == buffer_idx_) {
      buffers_.push_back({dtype, num_elements});
    }
    buffers_[buffer_idx_].send_buffers[num_pending_] = send_buffer;
    buffers_[buffer_idx_].recv_buffers[num_pending_] = recv_buffer;
    ++buffer_idx_;

    if (num_pending_ > 0 || buffer_idx_ < buffers_.size()) {
      // Exit early until called once per num_gpus and buffers_.size().
      return OkStatus();
    }

    // Reset state.
    num_pending_ = num_gpus_;
    buffer_idx_ = 0;
    std::vector<Buffers> buffers;
    buffers_.swap(buffers);

    for (int i = 1; i < num_gpus_; ++i) {
      // Synchronize first stream with all others.
      TF_RETURN_IF_ERROR(WaitForStream(stream, *streams_[i]));

      // Allow first device access all others' memory.
      CUcontext peer;
      CUDA_RETURN_IF_ERROR(
          cuStreamGetCtx(se::gpu::AsGpuStreamValue(streams_[i]), &peer));
      CUresult result = cuCtxEnablePeerAccess(peer, 0);
      if (result != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
        CUDA_RETURN_IF_ERROR(result);
      }
    }

    for (auto& buffer : buffers) {
      // Launch kernel for buffer on first stream.
      int64_t num_elements = buffer.num_elements;
      TF_ASSIGN_OR_RETURN(const void* kernel, [&]() -> StatusOr<const void*> {
        switch (buffer.dtype) {
          case PrimitiveType::BF16:
            if (num_elements % 2 == 0) {
              num_elements /= 2;
              return reinterpret_cast<const void*>(
                  &SmallAllReduceKernel<__nv_bfloat162>);
            }
            return reinterpret_cast<const void*>(
                &SmallAllReduceKernel<__nv_bfloat16>);
          case PrimitiveType::F32:
            return reinterpret_cast<const void*>(&SmallAllReduceKernel<float>);
          case PrimitiveType::S32:
            return reinterpret_cast<const void*>(
                &SmallAllReduceKernel<int32_t>);
          default:
            return InternalError("Unsupported: %s",
                                 PrimitiveType_Name(buffer.dtype));
        }
      }());
      int threads_per_block = std::min<int64_t>(kLaunchBounds, num_elements);
      int blocks_per_grid =
          (num_elements + threads_per_block - 1) / threads_per_block;
      void* args[] = {&num_gpus_, &buffer.send_buffers, &buffer.recv_buffers,
                      &num_elements};

      CUDA_RETURN_IF_ERROR(cudaLaunchKernel(
          kernel, blocks_per_grid, threads_per_block, args,
          /*sharedMem=*/0, se::gpu::AsGpuStreamValue(streams_.front())));
    }

    VLOG(0) << "Kernel launched";

    TF_RETURN_IF_ERROR(RecordStream(stream));
    for (int i = 1; i < num_gpus_; ++i) {
      // Synchronize all other streams with first one.
      TF_RETURN_IF_ERROR(WaitForStream(*streams_[i], stream));
    }

    return OkStatus();
  }

  struct Buffers {
    PrimitiveType dtype;
    int64_t num_elements;
    std::array<const void*, kMaxNumGpus> send_buffers;
    std::array<void*, kMaxNumGpus> recv_buffers;
  };

  int num_gpus_ = 0;
  int num_pending_ = 0;
  int buffer_idx_ = 0;
  std::array<se::Stream*, kMaxNumGpus> streams_;
  std::vector<Buffers> buffers_;
};

}  // namespace

Status RunSmallAllReduce(se::Stream& stream, int num_gpus,
                         const void* send_buffer, void* recv_buffer,
                         PrimitiveType dtype, int64_t num_elements) {
  static SmallAllReduce* small = new SmallAllReduce;
  return small->Add(stream, num_gpus, send_buffer, recv_buffer, dtype,
                    num_elements);
}

}  // namespace gpu
}  // namespace xla
