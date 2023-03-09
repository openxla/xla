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

#ifndef XLA_SERVICE_SEND_RECV_LATENCY_HIDING_SCHEDULER_H_
#define XLA_SERVICE_SEND_RECV_LATENCY_HIDING_SCHEDULER_H_

#include <optional>

#include "xla/service/latency_hiding_scheduler.h"

namespace xla {

std::optional<DefaultSchedulerCore::CandidateResult> SendRecvSchedulingRule(
    DefaultSchedulerCore::ScheduleCandidate& a,
    DefaultSchedulerCore::ScheduleCandidate& b);

}  // namespace xla

#endif  // XLA_SERVICE_SEND_RECV_LATENCY_HIDING_SCHEDULER_H_
