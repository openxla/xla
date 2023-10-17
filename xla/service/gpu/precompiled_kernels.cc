/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/precompiled_kernels.h"

#include <sys/stat.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/asm_compiler.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_pimpl.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

#if TENSORFLOW_USE_ROCM
#include "xla/stream_executor/gpu/gpu_stream.h"
namespace stream_executor {
namespace gpu {

extern void rocm_MakeBatchPointers(void* stream, char* base, int stride, int n,
                                   void** ptrs_out);

}
}  // namespace stream_executor
#endif

namespace xla {
namespace gpu {
namespace {

// GPU kernel to populate an array of pointers:
//
//   [base + stride * i for i in range(n)].
//
// Generated from the following CUDA code.
//
// extern "C" {
// __global__ void __xla_MakeBatchPointers(char* base, int stride,
//                                         int n, void** ptrs_out) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx >= n) return;
//   ptrs_out[idx] = base + idx * stride;
// }
// }
constexpr const char* kMakeBatchPointersPtx = R"(
.version 4.2
.target sm_35
.address_size 64

.visible .entry __xla_MakeBatchPointers(
        .param .u64 __xla_MakeBatchPointers_param_0,
        .param .u32 __xla_MakeBatchPointers_param_1,
        .param .u32 __xla_MakeBatchPointers_param_2,
        .param .u64 __xla_MakeBatchPointers_param_3
)
{
        .reg .pred      %p<2>;
        .reg .b32       %r<8>;
        .reg .b64       %rd<8>;

        ld.param.u32    %r2, [__xla_MakeBatchPointers_param_2];
        mov.u32         %r3, %tid.x;
        mov.u32         %r4, %ctaid.x;
        mov.u32         %r5, %ntid.x;
        mad.lo.s32      %r6, %r4, %r5, %r3;
        setp.ge.s32     %p1, %r6, %r2;
        @%p1 bra        LBB0_2;
        ld.param.u64    %rd3, [__xla_MakeBatchPointers_param_0];
        ld.param.u64    %rd4, [__xla_MakeBatchPointers_param_3];
        cvta.to.global.u64      %rd5, %rd4;
        ld.param.u32    %r1, [__xla_MakeBatchPointers_param_1];
        mul.wide.s32    %rd6, %r6, 8;
        add.s64         %rd1, %rd5, %rd6;
        mul.lo.s32      %r7, %r6, %r1;
        cvt.s64.s32     %rd7, %r7;
        add.s64         %rd2, %rd3, %rd7;
        st.global.u64   [%rd1], %rd2;
LBB0_2:
        ret;
}
)";

// Compiled from:
// __global__ void __xla_Prefetch(char* a, unsigned int n,
//                                unsigned char do_reset) {
//     int tid = blockDim.x * blockIdx.x + threadIdx.x;
//     for (int i = 0; i < n; i ++) {
//         void* addr = &a[(tid * n + i) * 128];
//         if (do_reset) {
//             asm("applypriority.global.L2::evict_normal [%0], 128;" ::
//                 "l"(addr));
//         } else {
//             asm("prefetch.global.L2::evict_last [%0];" :: "l"(addr));
//         }
//     }
// }

static constexpr const char* kPrefetchPtx = R"(
.version 8.1
.target sm_80
.address_size 64

.visible .entry __xla_Prefetch(
        .param .u64 _Z14__xla_PrefetchPcjh_param_0,
        .param .u32 _Z14__xla_PrefetchPcjh_param_1,
        .param .u8 _Z14__xla_PrefetchPcjh_param_2
) {
        .reg .pred      %p<11>;
        .reg .b16       %rs<2>;
        .reg .b32       %r<36>;
        .reg .b64       %rd<22>;
        .loc    1 1 0

        ld.param.u8     %rs1, [_Z14__xla_PrefetchPcjh_param_2];
        ld.param.u64    %rd6, [_Z14__xla_PrefetchPcjh_param_0];
        ld.param.u32    %r15, [_Z14__xla_PrefetchPcjh_param_1];
        .loc    1 2 5
        mov.u32         %r16, %ntid.x;
        mov.u32         %r17, %ctaid.x;
        mov.u32         %r18, %tid.x;
        mad.lo.s32      %r1, %r16, %r17, %r18;
        .loc    1 3 5
        setp.eq.s32     %p1, %r15, 0;
        @%p1 bra        $L__BB0_22;

        .loc    1 5 9
        add.s32         %r20, %r15, -1;
        and.b32         %r35, %r15, 3;
        setp.lt.u32     %p2, %r20, 3;
        mov.u32         %r33, 0;
        @%p2 bra        $L__BB0_16;

        mul.lo.s32      %r22, %r15, %r1;
        shl.b32         %r23, %r22, 7;
        add.s32         %r31, %r23, 384;
        sub.s32         %r4, %r35, %r15;

$L__BB0_3:
        .loc    1 4 9
        add.s32         %r24, %r31, -384;
        cvt.u64.u32     %rd7, %r24;
        add.s64         %rd1, %rd6, %rd7;
        setp.eq.s16     %p3, %rs1, 0;
        .loc    1 5 9
        @%p3 bra        $L__BB0_5;

        .loc    1 7 17
        // begin inline asm
        applypriority.global.L2::evict_normal [%rd1], 128;
        // end inline asm
        bra.uni         $L__BB0_6;

$L__BB0_5:
        .loc    1 9 13
        // begin inline asm
        prefetch.global.L2::evict_last [%rd1];
        // end inline asm

$L__BB0_6:
        .loc    1 4 9
        add.s32         %r25, %r31, -256;
        cvt.u64.u32     %rd10, %r25;
        add.s64         %rd2, %rd6, %rd10;
        .loc    1 5 9
        @%p3 bra        $L__BB0_8;

        .loc    1 7 17
        // begin inline asm
        applypriority.global.L2::evict_normal [%rd2], 128;
        // end inline asm
        bra.uni         $L__BB0_9;

$L__BB0_8:
        .loc    1 9 13
        // begin inline asm
        prefetch.global.L2::evict_last [%rd2];
        // end inline asm

$L__BB0_9:
        .loc    1 4 9
        add.s32         %r26, %r31, -128;
        cvt.u64.u32     %rd13, %r26;
        add.s64         %rd3, %rd6, %rd13;
        .loc    1 5 9
        @%p3 bra        $L__BB0_11;

        .loc    1 7 17
        // begin inline asm
        applypriority.global.L2::evict_normal [%rd3], 128;
        // end inline asm
        bra.uni         $L__BB0_12;

$L__BB0_11:
        .loc    1 9 13
        // begin inline asm
        prefetch.global.L2::evict_last [%rd3];
        // end inline asm

$L__BB0_12:
        .loc    1 4 9
        cvt.u64.u32     %rd16, %r31;
        add.s64         %rd4, %rd6, %rd16;
        .loc    1 5 9
        @%p3 bra        $L__BB0_14;

        .loc    1 7 17
        // begin inline asm
        applypriority.global.L2::evict_normal [%rd4], 128;
        // end inline asm
        bra.uni         $L__BB0_15;

$L__BB0_14:
        .loc    1 9 13
        // begin inline asm
        prefetch.global.L2::evict_last [%rd4];
        // end inline asm

$L__BB0_15:
        .loc    1 3 28
        add.s32         %r33, %r33, 4;
        .loc    1 3 5
        add.s32         %r31, %r31, 512;
        add.s32         %r27, %r4, %r33;
        setp.ne.s32     %p7, %r27, 0;
        @%p7 bra        $L__BB0_3;

$L__BB0_16:
        .loc    1 5 9
        setp.eq.s32     %p8, %r35, 0;
        @%p8 bra        $L__BB0_22;

        mul.lo.s32      %r28, %r15, %r1;
        shl.b32         %r29, %r28, 7;
        shl.b32         %r30, %r33, 7;
        add.s32         %r34, %r29, %r30;

$L__BB0_18:
        .pragma "nounroll";
        .loc    1 4 9
        cvt.u64.u32     %rd19, %r34;
        add.s64         %rd5, %rd6, %rd19;
        setp.eq.s16     %p9, %rs1, 0;
        .loc    1 5 9
        @%p9 bra        $L__BB0_20;

        .loc    1 7 17
        // begin inline asm
        applypriority.global.L2::evict_normal [%rd5], 128;
        // end inline asm
        bra.uni         $L__BB0_21;

$L__BB0_20:
        .loc    1 9 13
        // begin inline asm
        prefetch.global.L2::evict_last [%rd5];
        // end inline asm

$L__BB0_21:
        .loc    1 3 5
        add.s32         %r35, %r35, -1;
        add.s32         %r34, %r34, 128;
        setp.ne.s32     %p10, %r35, 0;
        @%p10 bra       $L__BB0_18;

$L__BB0_22:
        .loc    1 12 1
        ret;
})";

// Lazily compiles ptx kernel, once per StreamExecutor.
//
// Thread-safe.
template <typename... KernelArgs>
class LazyKernel {
 public:
  LazyKernel(absl::string_view kernel_name, const char* ptx,
             const se::GpuAsmOpts& asm_opts)
      : kernel_name_(kernel_name), ptx_(ptx), asm_opts_(asm_opts) {}

  StatusOr<se::TypedKernel<KernelArgs...>*> Get(
      se::StreamExecutor* stream_exec) {
    absl::MutexLock lock(&mu_);

    auto result = kernels_.emplace(stream_exec, nullptr);
    if (result.second) {
      absl::Span<const uint8_t> compiled_ptx;
      StatusOr<absl::Span<const uint8_t>> compiled_ptx_or =
          se::CompileGpuAsmOrGetCached(stream_exec->device_ordinal(), ptx_,
                                       asm_opts_);
      if (compiled_ptx_or.ok()) {
        compiled_ptx = std::move(compiled_ptx_or).value();
      } else {
        static absl::once_flag logged_once;
        absl::call_once(logged_once, [&]() {
          LOG(WARNING)
              << compiled_ptx_or.status()
              << "\nRelying on driver to perform ptx compilation. "
              << "\nSetting XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda "
              << " or modifying $PATH can be used to set the location of ptxas."
              << "\nThis message will only be logged once.";
        });
      }

      auto kernel = stream_exec->CreateTypedKernel<KernelArgs...>(
          kernel_name_, ptx_, compiled_ptx);
      if (kernel.ok()) {
        result.first->second = *std::move(kernel);
      } else {
        kernels_.erase(result.first);
        return kernel.status();
      }
    }
    return result.first->second.get();
  }

 private:
  std::string kernel_name_;
  const char* ptx_;
  se::GpuAsmOpts asm_opts_;

  absl::Mutex mu_;

  // A mutex keyed on StreamExecutor* is ok because StreamExecutors are never
  // destroyed.
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<se::TypedKernel<KernelArgs...>>>
      kernels_ ABSL_GUARDED_BY(mu_);
};

}  // anonymous namespace

Status MakeBatchPointers(se::Stream* stream, const se::GpuAsmOpts& asm_opts,
                         se::DeviceMemoryBase base_ptr, int stride_bytes, int n,
                         se::DeviceMemoryBase ptrs_out) {
#if TENSORFLOW_USE_ROCM
  stream_executor::gpu::rocm_MakeBatchPointers(
      se::gpu::AsGpuStreamValue(stream),
      reinterpret_cast<char*>(base_ptr.opaque()), stride_bytes, n,
      reinterpret_cast<void**>(ptrs_out.opaque()));
#else
  static auto* lazy_kernel =
      new LazyKernel<se::DeviceMemoryBase /*base_ptr*/, int /*stride_bytes*/,
                     int /*n*/, se::DeviceMemoryBase /*ptrs_out*/>(
          "__xla_MakeBatchPointers", kMakeBatchPointersPtx, asm_opts);

  TF_ASSIGN_OR_RETURN(auto kernel, lazy_kernel->Get(stream->parent()));

  constexpr int kThreads = 128;
  TF_RETURN_IF_ERROR(
      stream->ThenLaunch(se::ThreadDim(kThreads, 1, 1),
                         se::BlockDim(CeilOfRatio(n, kThreads), 1, 1), *kernel,
                         base_ptr, stride_bytes, n, ptrs_out));
#endif
  return OkStatus();
}

Status Prefetch(se::Stream* stream, se::DeviceMemoryBase base_ptr,
                const bool do_reset) {
  static auto* lazy_kernel =
      new LazyKernel<se::DeviceMemoryBase /*base_ptr*/, uint32_t /*n*/,
                     uint8_t /*do_reset*/>("__xla_Prefetch", kPrefetchPtx,
                                           se::GpuAsmOpts{});

  TF_ASSIGN_OR_RETURN(auto kernel, lazy_kernel->Get(stream->parent()));

  static constexpr uint64_t kNBlocks = 8;
  static constexpr uint64_t kThreadsPerBlock = 32;
  static constexpr uint64_t kCacheLineSizeBytes = 128;

  const auto do_reset_u = static_cast<uint8_t>(do_reset);
  if (do_reset_u == 1) {
    VLOG(1) << "Cache policy reset.";
  } else if (do_reset_u == 0) {
    VLOG(1) << "Prefetch.";
  }
  TF_RETURN_IF_ERROR(stream->ThenLaunch(
      se::ThreadDim(kThreadsPerBlock), se::BlockDim(kNBlocks), *kernel,
      base_ptr,
      static_cast<uint32_t>(CeilOfRatio(
          base_ptr.size(), kNBlocks * kThreadsPerBlock * kCacheLineSizeBytes)),
      do_reset_u));
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
