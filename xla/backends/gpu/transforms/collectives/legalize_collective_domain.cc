/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/collectives/legalize_collective_domain.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/transforms/collectives/collective_domain.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/side_effect_util.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

bool HasDomainAttribute(const HloInstruction& instruction) {
  return instruction.frontend_attributes().map().contains(
      kCollectiveCommunicationDomainAttr);
}

HloInstruction* ResolveCollectiveConfigTarget(HloInstruction& instruction) {
  HloInstruction* target = &instruction;
  if (HloPredicateIsOp<HloOpcode::kAsyncStart>(target) &&
      IsTritonCollectiveFusion(*target->async_wrapped_instruction())) {
    target = target->async_wrapped_instruction();
  }
  if (IsTritonCollectiveFusion(*target)) {
    return GetTritonCollectiveHero(*target);
  }
  return target;
}

const HloInstruction* ResolveCollectiveConfigTarget(
    const HloInstruction& instruction) {
  const HloInstruction* target = &instruction;
  if (HloPredicateIsOp<HloOpcode::kAsyncStart>(target) &&
      IsTritonCollectiveFusion(*target->async_wrapped_instruction())) {
    target = target->async_wrapped_instruction();
  }
  if (IsTritonCollectiveFusion(*target)) {
    return GetTritonCollectiveHero(*target);
  }
  return target;
}

absl::StatusOr<CollectiveCommunicationDomain> ReadLocalDomain(
    const HloInstruction& instruction) {
  CollectiveCommunicationDomain domain = kUnspecifiedCollectiveDomain;
  if (auto value = instruction.get_frontend_attribute(
          kCollectiveCommunicationDomainAttr)) {
    ABSL_ASSIGN_OR_RETURN(CollectiveCommunicationDomain attribute_domain,
                          ParseCollectiveCommunicationDomain(*value));
    ABSL_ASSIGN_OR_RETURN(
        domain, JoinCollectiveCommunicationDomains(domain, attribute_domain));
  }

  const HloInstruction* config_target =
      ResolveCollectiveConfigTarget(instruction);
  if (config_target == nullptr) {
    return absl::InternalError(
        "Triton collective fusion has no collective hero instruction");
  }
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig config,
                        config_target->backend_config<GpuBackendConfig>());
  ABSL_ASSIGN_OR_RETURN(
      domain,
      JoinCollectiveCommunicationDomains(
          domain, config.collective_backend_config().communication_domain()));
  return domain;
}

absl::Status ValidateStreamAnnotation(const HloInstruction& instruction,
                                      CollectiveCommunicationDomain domain) {
  if (domain != kScaleUpFabricCollectiveDomain) {
    return absl::OkStatus();
  }
  auto stream = instruction.get_frontend_attribute(kXlaStreamAnnotationAttr);
  if (!stream.has_value() ||
      absl::EqualsIgnoreCase(*stream, kXlaCollectiveStreamAnnotation)) {
    return absl::OkStatus();
  }
  return InvalidArgument("Collective %s selects domain %v and stream %s",
                         instruction.name(), domain, *stream);
}

absl::StatusOr<bool> SetLocalDomain(HloInstruction& instruction,
                                    CollectiveCommunicationDomain domain) {
  ABSL_RETURN_IF_ERROR(ValidateStreamAnnotation(instruction, domain));

  bool changed = false;
  HloInstruction* config_target = ResolveCollectiveConfigTarget(instruction);
  if (config_target == nullptr) {
    return absl::InternalError(
        "Triton collective fusion has no collective hero instruction");
  }
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig config,
                        config_target->backend_config<GpuBackendConfig>());
  if (config.collective_backend_config().communication_domain() != domain) {
    config.mutable_collective_backend_config()->set_communication_domain(
        domain);
    ABSL_RETURN_IF_ERROR(config_target->set_backend_config(config));
    changed = true;
  }

  changed |= instruction.erase_frontend_attribute(
                 kCollectiveCommunicationDomainAttr) != 0;
  return changed;
}

}  // namespace

absl::StatusOr<bool> LegalizeCollectiveDomain::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      const bool supports_domain =
          hlo_query::IsCollectiveCommunicationOp(instruction->opcode()) ||
          IsTritonCollectiveFusion(*instruction) ||
          IsAsyncTritonCollectiveFusionStart(*instruction);
      if (HasDomainAttribute(*instruction) && !supports_domain) {
        return InvalidArgument("Frontend attribute %s is not supported on %s",
                               kCollectiveCommunicationDomainAttr,
                               instruction->name());
      }
      if (supports_domain) {
        ABSL_ASSIGN_OR_RETURN(CollectiveCommunicationDomain domain,
                              ReadLocalDomain(*instruction));
        ABSL_ASSIGN_OR_RETURN(bool instruction_changed,
                              SetLocalDomain(*instruction, domain));
        changed |= instruction_changed;
      }
    }
  }

  return changed;
}

}  // namespace xla::gpu
