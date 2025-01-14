/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_COLLECTIVES_INTERFACE_H_
#define XLA_SERVICE_CPU_COLLECTIVES_INTERFACE_H_

#include <cstddef>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/core/collectives/communicator.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/global_device_id.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

class CollectivesInterface {
 public:
  virtual ~CollectivesInterface() = default;

  // Builds a context for a collective group.
  // Args:
  //  devices: the devices participating in this collective.
  //  rank: the rank of this process.
  virtual absl::StatusOr<std::shared_ptr<Communicator>> GetCommunicator(
      absl::Span<GlobalDeviceId const> devices, int rank) = 0;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_COLLECTIVES_INTERFACE_H_
