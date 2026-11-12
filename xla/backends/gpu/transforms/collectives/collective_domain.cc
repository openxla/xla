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

#include "xla/backends/gpu/transforms/collectives/collective_domain.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/side_effect_util.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

const HloInstruction& CanonicalAsyncStart(const HloInstruction& instruction) {
  const HloInstruction* canonical = &instruction;
  while (canonical->opcode() == HloOpcode::kAllGatherDone ||
         canonical->opcode() == HloOpcode::kAllReduceDone ||
         canonical->opcode() == HloOpcode::kCollectivePermuteDone ||
         canonical->opcode() == HloOpcode::kAsyncUpdate ||
         canonical->opcode() == HloOpcode::kAsyncDone) {
    canonical = canonical->operand(0);
  }
  return *canonical;
}

absl::Status ValidateCollectiveCommunicationDomain(
    CollectiveCommunicationDomain domain) {
  if (domain == kUnspecifiedCollectiveDomain ||
      domain == kScaleUpFabricCollectiveDomain) {
    return absl::OkStatus();
  }
  return InvalidArgument("Unsupported collective communication domain: %v",
                         domain);
}

}  // namespace

absl::StatusOr<CollectiveCommunicationDomain>
ParseCollectiveCommunicationDomain(absl::string_view value) {
  std::string lowercase_value = absl::AsciiStrToLower(value);
  if (lowercase_value == kUnspecifiedCollectiveDomainName) {
    return kUnspecifiedCollectiveDomain;
  }
  if (lowercase_value == kScaleUpFabricCollectiveDomainName) {
    return kScaleUpFabricCollectiveDomain;
  }
  return InvalidArgument("Unsupported collective communication domain: %s",
                         value);
}

absl::StatusOr<CollectiveCommunicationDomain>
JoinCollectiveCommunicationDomains(CollectiveCommunicationDomain lhs,
                                   CollectiveCommunicationDomain rhs) {
  ABSL_RETURN_IF_ERROR(ValidateCollectiveCommunicationDomain(lhs));
  ABSL_RETURN_IF_ERROR(ValidateCollectiveCommunicationDomain(rhs));
  if (lhs == kUnspecifiedCollectiveDomain) {
    return rhs;
  }
  if (rhs == kUnspecifiedCollectiveDomain || lhs == rhs) {
    return lhs;
  }
  return InvalidArgument(
      "Conflicting collective communication domains: %v and %v", lhs, rhs);
}

bool SupportsCollectiveCommunicationDomain(const HloInstruction& instruction) {
  const HloInstruction& canonical = CanonicalAsyncStart(instruction);
  if (&canonical != &instruction) {
    return SupportsCollectiveCommunicationDomain(canonical);
  }

  if (hlo_query::IsCollectiveCommunicationOp(instruction.opcode()) ||
      IsTritonCollectiveFusion(instruction)) {
    return true;
  }
  if (HloPredicateIsOp<HloOpcode::kAsyncStart>(&instruction)) {
    return hlo_query::IsCollectiveCommunicationOp(
               instruction.async_wrapped_opcode()) ||
           IsTritonCollectiveFusion(*instruction.async_wrapped_instruction()) ||
           instruction.frontend_attributes().map().contains(
               kCollectiveGroupMarkerAttr);
  }
  return false;
}

absl::StatusOr<CollectiveCommunicationDomain> GetCollectiveCommunicationDomain(
    const HloInstruction& instruction) {
  const HloInstruction& canonical = CanonicalAsyncStart(instruction);
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig canonical_config,
                        canonical.backend_config<GpuBackendConfig>());
  ABSL_ASSIGN_OR_RETURN(
      CollectiveCommunicationDomain domain,
      JoinCollectiveCommunicationDomains(
          kUnspecifiedCollectiveDomain,
          canonical_config.collective_backend_config().communication_domain()));

  const HloInstruction* fusion = &canonical;
  if (HloPredicateIsOp<HloOpcode::kAsyncStart>(fusion) &&
      IsTritonCollectiveFusion(*fusion->async_wrapped_instruction())) {
    fusion = fusion->async_wrapped_instruction();
  }
  if (IsTritonCollectiveFusion(*fusion)) {
    const HloInstruction* hero = GetTritonCollectiveHero(*fusion);
    if (hero == nullptr) {
      return absl::InternalError(
          "Triton collective fusion has no collective hero instruction");
    }
    ABSL_ASSIGN_OR_RETURN(GpuBackendConfig hero_config,
                          hero->backend_config<GpuBackendConfig>());
    ABSL_ASSIGN_OR_RETURN(
        domain,
        JoinCollectiveCommunicationDomains(
            domain,
            hero_config.collective_backend_config().communication_domain()));
  }
  return domain;
}

}  // namespace xla::gpu
