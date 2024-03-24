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

#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/latency_hiding_scheduler.h"

namespace xla {

namespace {
bool IsHostSend(const HloInstruction& instr) {
  return instr.opcode() == HloOpcode::kSend &&
         static_cast<const HloSendRecvInstruction*>(&instr)->is_host_transfer();
}

bool IsHostSendDone(const HloInstruction& instr) {
  return instr.opcode() == HloOpcode::kSendDone &&
         static_cast<const HloSendRecvInstruction*>(&instr)->is_host_transfer();
}
}  // namespace

std::optional<DefaultSchedulerCore::CandidateResult> SendRecvSchedulingRule(
    DefaultSchedulerCore::ScheduleCandidate& a,
    DefaultSchedulerCore::ScheduleCandidate& b) {
  // Delay host send-done.
  if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
          !IsHostSendDone(a.node->GetInstr()), a,
          !IsHostSendDone(b.node->GetInstr()), b, "kDelayHostSendDone")) {
    return *value;
  }
  // Delay host send.
  if (auto value = DefaultSchedulerCore::ChooseBestCandidate(
          !IsHostSend(a.node->GetInstr()), a, !IsHostSend(b.node->GetInstr()),
          b, "kDelayHostSend")) {
    return *value;
  }
  return std::nullopt;
}

}  // namespace xla
