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

#ifndef XLA_PJRT_CPU_MPI_COLLECTIVES_H_
#define XLA_PJRT_CPU_MPI_COLLECTIVES_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "mpi.h"  // NOLINT
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/cpu/collectives_interface.h"
#include "xla/service/global_device_id.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

class MpiCollectivesCommunicator : public Communicator {
 public:
  explicit MpiCollectivesCommunicator(int color, int key);
  ~MpiCollectivesCommunicator() override;

  absl::Status AllReduce(se::DeviceMemoryBase send_buffer,
                         se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                         size_t count, ReductionKind reduction_kind,
                         const Executor& executor) override;
  absl::Status CollectivePermute(se::DeviceMemoryBase send_buffer,
                                 se::DeviceMemoryBase recv_buffer,
                                 PrimitiveType dtype, size_t count,
                                 std::optional<RankId> source_rank,
                                 absl::Span<const RankId> target_ranks,
                                 const Executor& executor) override;
  absl::Status AllToAll(absl::Span<const se::DeviceMemoryBase> send_buffers,
                        absl::Span<const se::DeviceMemoryBase> recv_buffers,
                        PrimitiveType dtype, size_t count,
                        const Executor& executor) override;
  absl::Status AllGather(se::DeviceMemoryBase send_buffer,
                         se::DeviceMemoryBase recv_buffer, PrimitiveType dtype,
                         size_t count, const Executor& executor) override;
  absl::Status ReduceScatter(se::DeviceMemoryBase send_buffer,
                             se::DeviceMemoryBase recv_buffer,
                             PrimitiveType dtype, size_t count,
                             ReductionKind reduction_kind,
                             const Executor& executor) override;

  absl::Status Broadcast(se::DeviceMemoryBase, se::DeviceMemoryBase,
                         PrimitiveType, size_t, RankId,
                         const Executor&) override {
    return Unimplemented("Broadcast is not implemented");
  }

  absl::Status Send(se::DeviceMemoryBase, PrimitiveType, size_t, RankId,
                    const Executor&) override {
    return Unimplemented("Send is not implemented");
  }

  absl::Status Recv(se::DeviceMemoryBase, PrimitiveType, size_t, RankId,
                    const Executor&) override {
    return Unimplemented("Recv is not implemented");
  }

  absl::StatusOr<size_t> NumRanks() const override { return mpi_size_; }

  std::string ToString() const override {
    return absl::StrCat("MpiCommunicator [rank: ", mpi_rank_,
                        " num_ranks: ", mpi_size_, "]");
  }

 private:
  MPI_Comm comm_;
  int mpi_rank_;
  int mpi_size_;
};

class MpiCollectives : public CollectivesInterface {
 public:
  /*
  The user has to explicitly call Init() and Finalize() before and
  after use.
  For example, using the Python client, this can be achieved with:

  collectives = xla_client._xla.make_mpi_collectives()
  collectives.Init()
  atexit.register(collectives.Finalize)
  */
  void Init();
  void Finalize();

  absl::StatusOr<std::shared_ptr<Communicator>> GetCommunicator(
      absl::Span<GlobalDeviceId const> global_devices, int rank) override;

 private:
  absl::Status ExchangeGlobalDeviceIds(
      absl::Span<GlobalDeviceId const> global_devices, int rank);

  int mpi_world_rank_;
  int mpi_world_size_;
  absl::flat_hash_map<std::tuple<std::vector<GlobalDeviceId>, int>,
                      std::shared_ptr<MpiCollectivesCommunicator>>
      contexts_;
};

}  // namespace xla::cpu

#endif  // XLA_PJRT_CPU_MPI_COLLECTIVES_H_
