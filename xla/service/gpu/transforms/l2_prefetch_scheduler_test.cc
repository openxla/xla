/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/transforms/l2_prefetch_scheduler.h"
#include "absl/status/status_matchers.h"

#include "xla/tests/hlo_test_base.h"

namespace xla::gpu {
namespace {

// FIXME: move to a common header
MATCHER_P(ThunkKindIs, kind, "") {
  return ExplainMatchResult(::testing::Eq(kind), arg->kind(), result_listener);
}

using ::absl_testing::IsOkAndHolds;

using L2PrefetchSchedulerTest = HloTestBase;

TEST_F(L2PrefetchSchedulerTest, SchedulesPrefetchWithUnannotatedAllReduce) {
  RunAndFilecheckHloRewrite(
      R"(
HloModule m, is_scheduled=true

f {
  a = bf16[46000,1024] parameter(0)
  b = bf16[4,1024] parameter(1)
  c = bf16[4,11500,1024] broadcast(b), dimensions={0,2}
  d = bf16[46000,1024] bitcast(c)
  e = bf16[46000,1024] add(a, d)
}

entry {
  y = bf16[4,1024] parameter(1)
  ars = bf16[4,1024] all-reduce-start(y), replica_groups={}, to_apply={
    a = bf16[] parameter(0)
    b = bf16[] parameter(1)
    o = bf16[] add(b, a)
  }
  ard = bf16[4,1024] all-reduce-done(ars)
  x = bf16[46000,1024] parameter(0)
  n = bf16[46000,1024] fusion(x, ard), kind=kLoop, calls=f
})",
      L2PrefetchScheduler{TestGpuDeviceInfo::RTXH100SXMDeviceInfo()}, R"(
// CHECK: slice
// CHECK: calls=%l2_prefetch
)");
}

TEST_F(L2PrefetchSchedulerTest,
       SchedulesPartialPrefetchWithAnnotatedAllReduce) {
  RunAndFilecheckHloRewrite(
      R"(
HloModule m, is_scheduled=true

f {
  a = bf16[460,1024] parameter(0)
  b = bf16[4,1024] parameter(1)
  c = bf16[4,1152,1024] broadcast(b), dimensions={0,2}
  d = bf16[460,1024] bitcast(c)
  e = bf16[460,1024] add(a, d)
}

entry {
  y = bf16[4096] parameter(1)
  ars = bf16[4096] all-reduce-start(y), replica_groups={}, to_apply={
    a = bf16[] parameter(0)
    b = bf16[] parameter(1)
    o = bf16[] add(b, a)
  }
  ard = bf16[4096] all-reduce-done(ars), frontend_attributes={l2_prefetch_opportunity="4488"}
  bc = bf16[4,1024] bitcast(ard)
  x = bf16[460,1024] parameter(0)
  n = bf16[460,1024] fusion(x, bc), kind=kLoop, calls=f
})",
      L2PrefetchScheduler{TestGpuDeviceInfo::RTXH100SXMDeviceInfo()}, R"(
// CHECK: bf16[2244]{0} slice
// CHECK: all-reduce-start
// CHECK-NEXT: calls=%l2_prefetch
// CHECK-NEXT: all-reduce-done
)");
}

TEST_F(L2PrefetchSchedulerTest, SchedulesPrefetchWithAnnotatedAllGather) {
  RunAndFilecheckHloRewrite(
      R"(
HloModule m, is_scheduled=true

f {
  a = bf16[460,1024] parameter(0)
  b = bf16[4,1024] parameter(1)
  c = bf16[4,1152,1024] broadcast(b), dimensions={0,2}
  d = bf16[460,1024] bitcast(c)
  e = bf16[460,1024] add(a, d)
}

entry {
  y = bf16[2048] parameter(1)
  ags = (bf16[2048], bf16[4096]) all-gather-start(y), dimensions={0}, replica_groups={}
  agd = bf16[4096] all-gather-done(ags), frontend_attributes={l2_prefetch_opportunity="12345678"}
  bc = bf16[4,1024] bitcast(agd)
  x = bf16[460,1024] parameter(0)
  n = bf16[460,1024] fusion(x, bc), kind=kLoop, calls=f
})",
      L2PrefetchScheduler{TestGpuDeviceInfo::RTXH100SXMDeviceInfo()}, R"(
// CHECK-NOT: slice
// CHECK: all-gather-start
// CHECK-NEXT: calls=%l2_prefetch
// CHECK-NEXT: all-gather-done
)");
}

TEST_F(L2PrefetchSchedulerTest, SchedulesPrefetchWithAnnotatedReduceScatter) {
  RunAndFilecheckHloRewrite(
      R"(
HloModule m, is_scheduled=true

f {
  a = bf16[460,1024] parameter(0)
  b = bf16[4,1024] parameter(1)
  c = bf16[4,1152,1024] broadcast(b), dimensions={0,2}
  d = bf16[460,1024] bitcast(c)
  e = bf16[460,1024] add(a, d)
}

entry {
  y = bf16[4096] parameter(1)
  rss = ((bf16[4096]), bf16[2048]) reduce-scatter-start(y), dimensions={0}, to_apply={
    a = bf16[] parameter(0)
    b = bf16[] parameter(1)
    o = bf16[] add(b, a)
  }
  rsd = bf16[2048] reduce-scatter-done(rss), frontend_attributes={l2_prefetch_opportunity="12345678"}
  bc = bf16[4,1024] bitcast(rsd)
  x = bf16[460,1024] parameter(0)
  n = bf16[460,1024] fusion(x, bc), kind=kLoop, calls=f
})",
      L2PrefetchScheduler{TestGpuDeviceInfo::RTXH100SXMDeviceInfo()}, R"(
// CHECK-NOT: slice
// CHECK: reduce-scatter-start
// CHECK-NEXT: calls=%l2_prefetch
// CHECK-NEXT: reduce-scatter-done
)");
}

TEST_F(L2PrefetchSchedulerTest,
       SchedulesPartialPrefetchWithAnnotatedCustomCall) {
  RunAndFilecheckHloRewrite(
      R"(
HloModule m, is_scheduled=true

f {
  a = bf16[4608,1024] parameter(0)
  b = bf16[4,1024] parameter(1)
  c = bf16[4,1152,1024] broadcast(b), dimensions={0,2}
  d = bf16[4608,1024] bitcast(c)
  e = bf16[4608,1024] add(a, d)
}

e {
  x = bf16[4608,1024] parameter(0)
  y = bf16[4,1024] parameter(1)
  cc = bf16[1,4,1024] custom-call(y),
    custom_call_target="x", frontend_attributes={l2_prefetch_opportunity="1000000"}
  b = bf16[4,1024] bitcast(cc)
  n = bf16[4608,1024] fusion(x, b), kind=kLoop, calls=f
})",
      L2PrefetchScheduler{TestGpuDeviceInfo::RTXH100SXMDeviceInfo()}, R"(
// CHECK: bf16[500000]{0} slice
// CHECK: custom-call-start
// CHECK-NEXT: calls=%l2_prefetch
// CHECK-NEXT: custom-call-done
)");
}

TEST_F(L2PrefetchSchedulerTest, PrefetchingRunsAndProducesSameOutput) {
  const char* with_prefetch = R"(
HloModule m, is_scheduled=true

f {
  a = s8[10000] parameter(0)
  b = s8[10000] negate(a)
}

%l2_prefetch {
  %p = s8[10000] parameter(0)
  %custom-call = () custom-call(%p), custom_call_target="l2_prefetch",
    custom_call_has_side_effect=true,
    frontend_attributes={prefetch_num_blocks="10"}
}

g {
  a = s8[10000] parameter(0)
  b = s8[10000] negate(a)
}

entry {
  a = s8[10000] parameter(0)
  b = s8[10000] parameter(1)
  fs = ((s8[10000]), s8[10000], s8[0]) fusion-start(a), kind=kLoop, calls=f
  p = () fusion(b), kind=kCustom, calls=%l2_prefetch,
   control-predecessors={fs}, backend_config={"fusion_backend_config":{"kind":"l2_prefetch"}}
  fd = s8[10000] fusion-done(fs)
  g = s8[10000] fusion(b), kind=kLoop, calls=g
  t = tuple(fd, g)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(with_prefetch));

  std::unique_ptr<Executable> executable =
      backend()
          .compiler()
          ->RunBackend(std::move(module), backend().default_stream_executor(),
                       {/*device_allocator=*/nullptr,
                        /*thread_pool=*/nullptr,
                        /*layout_canonicalization_callback=*/{},
                        /*is_autotuning_compilation=*/false})
          .value();
  std::unique_ptr<GpuExecutable> gpu_exec(
      static_cast<GpuExecutable*>(executable.release()));

  EXPECT_EQ(gpu_exec->GetThunk().thunks().size(), 5);
  EXPECT_THAT(gpu_exec->GetThunk().thunks(),
              ::testing::ElementsAre(ThunkKindIs(Thunk::kWaitForStreams),
                                     ThunkKindIs(Thunk::kKernel),
                                     ThunkKindIs(Thunk::kCustomKernel),
                                     ThunkKindIs(Thunk::kWaitForStreams),
                                     ThunkKindIs(Thunk::kKernel)));

  const char* without_prefetch = R"(
HloModule m, is_scheduled=true

f {
  a = s8[10000] parameter(0)
  b = s8[10000] negate(a)
}

g {
  a = s8[10000] parameter(0)
  b = s8[10000] negate(a)
}

entry {
  a = s8[10000] parameter(0)
  b = s8[10000] parameter(1)
  f = s8[10000] fusion(a), kind=kLoop, calls=f
  g = s8[10000] fusion(b), kind=kLoop, calls=g
  t = tuple(f, g)
})";

  EXPECT_TRUE(RunAndCompareTwoModules(with_prefetch, without_prefetch,
                                      std::nullopt, /*run_hlo_passes=*/false));
}

}  // namespace
}  // namespace xla::gpu
