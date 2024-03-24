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

#include "xla/service/send_recv_latency_hiding_scheduler.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

constexpr int kMaxConcurrentAsyncCollectivePermutes = 5;

int GetIndex(absl::Span<HloInstruction* const> instruction_sequence,
             absl::string_view hlo_name) {
  return absl::c_find_if(instruction_sequence,
                         [hlo_name](HloInstruction* instruction) {
                           return instruction->name() == hlo_name;
                         }) -
         instruction_sequence.begin();
}

SchedulerConfig GetDefaultSchedConfig() {
  SchedulerConfig sched_cfg;
  sched_cfg.collective_permute_overlap_limit =
      kMaxConcurrentAsyncCollectivePermutes;
  sched_cfg.send_recv_overlap_limit = INT32_MAX;
  return sched_cfg;
}

StatusOr<bool> RunScheduler(
    HloModule* module, SchedulerConfig sched_config = GetDefaultSchedConfig(),
    std::unique_ptr<LatencyEstimator> latency_estimator =
        std::make_unique<ApproximateLatencyEstimator>()) {
  HloCostAnalysis::ShapeSizeFunction shape_size_bytes =
      [&shape_size_bytes](const Shape& shape) -> int64_t {
    int64_t shape_size = 0;
    if (shape.IsTuple()) {
      for (auto& sub_shape : shape.tuple_shapes()) {
        shape_size += shape_size_bytes(sub_shape);
      }
      return shape_size;
    }
    return ShapeUtil::ByteSizeOfElements(shape);
  };
  auto async_tracker = std::make_unique<AsyncTracker>(sched_config);
  auto scheduler_core = std::make_unique<DefaultSchedulerCore>(
      shape_size_bytes, async_tracker.get(), latency_estimator.get(),
      sched_config, /*target_scheduling_rule=*/nullptr,
      /*early_target_scheduling_rule=*/SendRecvSchedulingRule,
      /*post_processing_fn=*/nullptr);
  TF_ASSIGN_OR_RETURN(
      bool value, LatencyHidingScheduler(
                      std::move(latency_estimator), std::move(async_tracker),
                      std::move(scheduler_core), shape_size_bytes)
                      .Run(module));

  return value;
}

}  // namespace

class LatencyHidingSchedulerTest : public HloTestBase {
 public:
  StatusOr<std::unique_ptr<HloModule>> ParseHloText(
      absl::string_view hlo_string) {
    TF_ASSIGN_OR_RETURN(
        auto hlo_module,
        ParseAndReturnVerifiedModule(hlo_string, GetModuleConfigForTest()));
    return StatusOr<std::unique_ptr<HloModule>>(std::move(hlo_module));
  }
};

TEST_F(LatencyHidingSchedulerTest, SendRecv_PostProcess) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[1665,64,256]{2,1,0} parameter(0)
  p2 = f32[1665,256,256]{2,1,0} parameter(1)
  p3 = f32[1665,256,64]{2,1,0} parameter(2)
  %arg0 = s32[1,1,128]{2,1,0:T(1,128)} parameter(3)
  a0 = f32[1665,64,256]{2,1,0} add(p0, p0)
  after-all = token[] after-all()
  send = (s32[1,1,128]{2,1,0:T(1,128)}, u32[]{:S(2)}, token[]) send(arg0, after-all), channel_id=1, is_host_transfer=true, frontend_attributes={_xla_dcn_recv_channel="2"}
  send-done = token[] send-done(send), channel_id=1, is_host_transfer=true
  recv = (s32[1,1,128]{2,1,0:T(1,128)}, u32[]{:S(2)}, token[]) recv(send-done), channel_id=2, is_host_transfer=true, frontend_attributes={}
  recv-done = (s32[1,1,128]{2,1,0:T(1,128)}, token[]) recv-done(recv), channel_id=2, is_host_transfer=true
  c0 = f32[1665,64,256]{2,1,0} convolution(p2, p3),
    window={size=1665 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  send.1 = (s32[1,1,128]{2,1,0:T(1,128)}, u32[]{:S(2)}, token[]) send(arg0, after-all), channel_id=3, is_host_transfer=true, frontend_attributes={_xla_dcn_recv_channel="4"}
  send-done.1 = token[] send-done(send.1), channel_id=3, is_host_transfer=true
  recv.1 = (s32[1,1,128]{2,1,0:T(1,128)}, u32[]{:S(2)}, token[]) recv(send-done.1), channel_id=4, is_host_transfer=true, frontend_attributes={}
  recv-done.1 = (s32[1,1,128]{2,1,0:T(1,128)}, token[]) recv-done(recv.1), channel_id=4, is_host_transfer=true
  recv-data = s32[1,1,128]{2,1,0:T(1,128)} get-tuple-element(recv-done), index=0
  recv-data.1 = s32[1,1,128]{2,1,0:T(1,128)} get-tuple-element(recv-done.1), index=0
  c1 = f32[1665,256,256]{2,1,0} convolution(a0, c0),
    window={size=1665 stride=15 lhs_dilate=16}, dim_labels=0fb_0io->0fb
  ROOT tuple = (s32[1,1,128]{2,1,0:T(1,128)}, f32[1665,256,256]{2,1,0}, s32[1,1,128]{2,1,0:T(1,128)}) tuple(recv-data, c1, recv-data.1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseHloText(hlo_string));
  HloSchedule& module_schedule = hlo_module->schedule();
  EXPECT_TRUE(hlo_module->has_entry_computation());
  HloComputation* entry_computation = hlo_module->entry_computation();
  std::vector<HloInstruction*> original_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();

  auto sched_config = GetDefaultSchedConfig();
  sched_config.schedule_send_recvs = false;
  EXPECT_TRUE(RunScheduler(hlo_module.get(), sched_config).ok());
  std::vector<HloInstruction*> new_instruction_sequence =
      module_schedule.sequence(entry_computation).instructions();
  if (VLOG_IS_ON(1)) {
    for (auto* new_i : new_instruction_sequence) {
      VLOG(1) << new_i->ToString();
    }
  }
  EXPECT_LT(GetIndex(new_instruction_sequence, "send"),
            GetIndex(new_instruction_sequence, "a0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "send-done"),
            GetIndex(new_instruction_sequence, "a0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "send"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "send-done"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "send"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "send-done"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "send.1"),
            GetIndex(new_instruction_sequence, "a0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "send-done.1"),
            GetIndex(new_instruction_sequence, "a0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "send.1"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "send-done.1"),
            GetIndex(new_instruction_sequence, "c0"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "send.1"),
            GetIndex(new_instruction_sequence, "c1"));
  EXPECT_LT(GetIndex(new_instruction_sequence, "send-done.1"),
            GetIndex(new_instruction_sequence, "c1"));
}

}  // namespace xla
