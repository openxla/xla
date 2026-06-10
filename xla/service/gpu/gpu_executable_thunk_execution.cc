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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_cache.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/scratch_memory.h"
#include "xla/backends/gpu/runtime/scratch_memory_requests.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/hang_watchdog.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/rendezvous.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/stream_pool.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/util.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::gpu {
namespace {

absl::StatusOr<bool> ShouldCollectiveUseMinimalResource(
    const HloModule& module) {
  int64_t sync_collective_count = 0;
  int64_t total_sync_coll_size = 0;

  int64_t async_collective_count = 0;
  int64_t total_async_coll_size = 0;
  for (HloComputation* computation : module.MakeNonfusionComputations()) {
    for (HloInstruction* inst : computation->instructions()) {
      if (IsCollective(inst)) {
        ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                         inst->backend_config<GpuBackendConfig>());

        bool is_sync = gpu_backend_config.collective_backend_config().is_sync();
        int64_t total_size =
            ShapeUtil::ByteSizeOfElementsRecursive(inst->shape());

        if (is_sync) {
          sync_collective_count++;
          total_sync_coll_size += total_size;
        } else {
          async_collective_count++;
          total_async_coll_size += total_size;
        }
      }
    }
  }
  // Simple Heuristics to determine if we should minimize SM usage.
  // If we have more async collective count or larger message sizes
  // for async collectives, then we will choose to minimize SM
  // usage to give resource for computes.
  return (sync_collective_count <= async_collective_count ||
          total_sync_coll_size <= total_async_coll_size);
}

}  // namespace

using ::tsl::profiler::ScopedAnnotation;

GpuExecutable::BorrowedStreams GpuExecutable::BorrowedStreams::Assign(
    se::Stream* stream, int num_streams) {
  return BorrowedStreams{std::vector<se::Stream*>(num_streams, stream), {}};
}

absl::StatusOr<GpuExecutable::BorrowedStreams> GpuExecutable::BorrowStreams(
    const ServiceExecutableRunOptions& run_options, int device_ordinal,
    int num_streams, se::StreamPriority priority) {
  ASSIGN_OR_RETURN(
      std::vector<StreamPool::Ptr> owners,
      run_options.BorrowStreams(device_ordinal, num_streams, priority));

  std::vector<se::Stream*> streams;
  streams.reserve(num_streams);
  for (auto& stream : owners) {
    streams.push_back(stream.get());
  }

  return BorrowedStreams{std::move(streams), std::move(owners)};
}

absl::Status MaybeSyncAndProfile(const ServiceExecutableRunOptions* run_options,
                                 se::EventBasedTimer* execution_timer,
                                 se::Stream* stream_to_sync);

absl::Status RendezvousAfterInitialization(
    const ServiceExecutableRunOptions& run_options,
    const DebugOptions* absl_nullable debug_options);

absl::Status BarrierAfterExecutable(
    const ServiceExecutableRunOptions& run_options,
    const DebugOptions* absl_nullable debug_options, se::Stream& stream_to_sync,
    size_t num_participants);

absl::Status GpuExecutable::ExecuteThunksImpl(
    const DebugOptions* debug_options, const std::string& module_name,
    ModuleIdentifier module_id, ThunkExecutor& thunk_executor,
    Thunk::ExecutableSource executable_source,
    const ServiceExecutableRunOptions* run_options,
    const BufferAllocations& buffer_allocations,
    const Thunk::CommandBufferUpdateInfo* command_buffer_update_info,
    bool block_host_until_done,
    GpuExecutable::NumAdditionalStreams num_additional_streams,
    CollectiveMemoryCache& collective_memory_cache,
    bool collective_use_minimal_resource) {
  bool mock_collectives =
      run_options->run_options().gpu_executable_run_options()
          ? run_options->run_options()
                .gpu_executable_run_options()
                ->enable_mock_collectives()
          : false;

  int64_t collective_max_nchannels =
      debug_options ? debug_options->xla_gpu_nccl_collective_max_nchannels()
                    : 0;
  int64_t p2p_max_nchannels =
      debug_options ? debug_options->xla_gpu_nccl_p2p_max_nchannels() : 0;
  bool use_highest_priority_for_async_stream =
      debug_options
          ? debug_options->xla_gpu_enable_highest_priority_async_stream()
          : false;

  se::Stream* main_stream = run_options->stream();
  se::StreamExecutor* executor = main_stream->parent();
  se::StreamPriority communication_stream_priority =
      se::StreamPriority::Default;

  // TODO(intel-tf): Enable stream priorities for sycl backend.
  if (executor->GetPlatform()->id() == se::sycl::kSyclPlatformId) {
    use_highest_priority_for_async_stream = false;
  }
  if (use_highest_priority_for_async_stream) {
    communication_stream_priority = se::StreamPriority::Highest;
  }

  // Maybe install progress tracker for this execution.
  int32_t progress_tracking_n =
      debug_options ? debug_options->xla_gpu_execution_progress_tracking() : 0;

  std::optional<ThunkExecutor::ScopedProgressTracker> tracker;
  if (progress_tracking_n > 0) {
    ASSIGN_OR_RETURN(tracker, InstallProgressTracker(executor, thunk_executor));
  }

  // Maybe add a watch guard for this execution.
  absl::Duration watchdog_timeout = absl::InfiniteDuration();
  if (debug_options &&
      !debug_options->xla_gpu_execution_terminate_timeout().empty()) {
    TF_RET_CHECK(absl::ParseDuration(
        debug_options->xla_gpu_execution_terminate_timeout(),
        &watchdog_timeout))
        << "Failed to parse XLA execution terminate timeout";
  }

  std::shared_ptr<HangWatchdog::Guard> guard = nullptr;
  if (watchdog_timeout < absl::InfiniteDuration()) {
    int32_t device_ordinal = executor->device_ordinal();
    std::string watchdog_name = absl::StrFormat("[%d] XLA GPU execution `%s`",
                                                device_ordinal, module_name);

    // If we have installed progress tracker, log how far thunk execution
    // progressed before getting stuck. This is helpful for identifying kernels
    // that never finish and stall the stream execution.
    HangWatchdog::CancelCallback pre_abort;
    if (tracker.has_value()) {
      pre_abort = [&tracker, progress_tracking_n, device_ordinal] {
        auto log_progress = [&](auto label, auto thunks) {
          LOG(ERROR) << absl::StreamFormat("[%d] %s: size=%d", device_ordinal,
                                           label, thunks.size());
          // We want to report all thunks in chronological order for
          // readability according to the time they were executed.
          absl::c_sort(thunks, [](const auto& a, const auto& b) {
            return a.executed < b.executed;
          });
          for (auto& thunk : thunks) {
            std::string loop_info;
            for (const auto& state : thunk.loop_nest) {
              absl::StrAppend(&loop_info,
                              absl::StrFormat(" [%s iter=%d]", state.loop_name,
                                              state.loop_iteration));
            }
            LOG(ERROR) << absl::StreamFormat(
                "  - exec[%d] thunk[%d/%d] %v: %s at %s%s", thunk.exec_idx,
                thunk.thunk_idx, tracker->num_thunks(), thunk.kind, thunk.name,
                absl::FormatTime("%Y-%m-%d %H:%M:%S.%E6f", thunk.executed,
                                 absl::LocalTimeZone()),
                loop_info);
          }
        };

        size_t num_executions = tracker->num_executions();
        LOG(ERROR) << absl::StreamFormat(
            "[%d] Completed thunks: %d/%d (unique thunks: %d)", device_ordinal,
            tracker->NumCompletedThunks(), num_executions,
            tracker->num_thunks());
        LOG(ERROR) << absl::StreamFormat(
            "[%d] Pending thunks: %d/%d (unique thunks: %d)", device_ordinal,
            tracker->NumPendingThunks(), num_executions, tracker->num_thunks());

        log_progress("Last completed thunks",
                     tracker->LastCompletedThunks(progress_tracking_n));
        log_progress("First pending thunks",
                     tracker->FirstPendingThunks(progress_tracking_n));
        log_progress("Last pending thunks",
                     tracker->LastPendingThunks(progress_tracking_n));
      };
    }

    guard = HangWatchdog::Global().Watch(
        watchdog_name, watchdog_timeout,
        HangWatchdog::Abort(watchdog_name, watchdog_timeout,
                            std::move(pre_abort)));
  }

  // Borrow stream for tracing command buffers.
  se::Stream* command_buffer_trace_stream = nullptr;
  StreamPool::Ptr borrowed_command_buffer_trace_stream;
  if (run_options->HasStreamBorrower()) {
    ASSIGN_OR_RETURN(borrowed_command_buffer_trace_stream,
                     run_options->BorrowStream(executor->device_ordinal()));
    command_buffer_trace_stream = borrowed_command_buffer_trace_stream.get();
  }

  // Borrow streams for communication.
  BorrowedStreams communication_streams = BorrowedStreams::Assign(
      main_stream, num_additional_streams.communication);
  if (run_options->HasStreamBorrower()) {
    ASSIGN_OR_RETURN(communication_streams,
                     BorrowStreams(*run_options, executor->device_ordinal(),
                                   num_additional_streams.communication,
                                   communication_stream_priority));
    XLA_VLOG_DEVICE(2, run_options->device_ordinal())
        << absl::StreamFormat("Using %d additional communication streams.",
                              num_additional_streams.communication);
  }

  // Borrow streams for computations.
  BorrowedStreams compute_streams =
      BorrowedStreams::Assign(main_stream, num_additional_streams.compute);
  if (run_options->HasStreamBorrower()) {
    ASSIGN_OR_RETURN(compute_streams,
                     BorrowStreams(*run_options, executor->device_ordinal(),
                                   num_additional_streams.compute,
                                   se::StreamPriority::Default));
    XLA_VLOG_DEVICE(2, run_options->device_ordinal()) << absl::StreamFormat(
        "Using %d additional compute streams.", num_additional_streams.compute);
  }

  tsl::profiler::TraceMe hlo_module_activity(
      [&] { return absl::StrCat(module_name, ":XLA GPU module"); },
      tsl::profiler::TraceMeLevel::kInfo);

  std::unique_ptr<se::EventBasedTimer> execution_timer;
  if (ExecutionProfile* profile =
          run_options->run_options().execution_profile();
      profile) {
    ASSIGN_OR_RETURN(execution_timer, main_stream->CreateEventBasedTimer(
                                          profile->warmup_run_executed()));
  }

  // A state container for this execution.
  Thunk::ExecutionScopedState execution_scoped_state;

  // Parameters for executing collective operations.
  std::optional<std::string> collectives_impl_name;
  if (debug_options &&
      !debug_options->xla_gpu_collectives_implementation().empty()) {
    collectives_impl_name = debug_options->xla_gpu_collectives_implementation();
  }

  ASSIGN_OR_RETURN(
      CollectiveParams collective_params,
      CollectiveParams::Create(
          *run_options, communication_streams.streams,
          LocalDeviceId(main_stream->parent()->device_ordinal()),
          std::move(collectives_impl_name), collective_max_nchannels,
          p2p_max_nchannels, collective_use_minimal_resource));

  CollectiveCliqueRequests collective_clique_requests;
  CollectiveMemoryRequests collective_memory_requests(buffer_allocations);
  ScratchMemoryRequests scratch_memory_requests;

  {  // Prepare thunks for execution and collect requested GPU cliques.
    Thunk::PrepareParams prepare_params{&collective_params,
                                        &collective_clique_requests,
                                        &collective_memory_requests,
                                        &scratch_memory_requests,
                                        executor,
                                        &buffer_allocations,
                                        &execution_scoped_state};

    tsl::profiler::TraceMe trace_prepare("Thunks::Prepare");
    RETURN_IF_ERROR(thunk_executor.Prepare(prepare_params));
  }

  XLA_VLOG_DEVICE(3, run_options->device_ordinal()) << absl::StreamFormat(
      "Prepared GPU executable module: %s for execution: "
      "#collective=[cliques=%d, symmetric=%d]",
      module_name, collective_clique_requests.size(),
      collective_memory_requests.symmetric_size());

  std::vector<std::unique_ptr<CliqueKey>>* clique_keys =
      run_options->run_options().clique_keys();
  if (clique_keys != nullptr) {
    for (const GpuCliqueKey& clique_key :
         collective_clique_requests.RequestedCliques()) {
      clique_keys->push_back(std::make_unique<GpuCliqueKey>(clique_key));
    }
  }

  // Acquire collective cliques requested by thunks.
  CollectiveCliques collective_cliques;
  if (!mock_collectives) {
    ASSIGN_OR_RETURN(collective_cliques,
                     AcquireCollectiveCliques(collective_params,
                                              collective_clique_requests));
  }

  ASSIGN_OR_RETURN(ScratchMemory scratch_memory,
                   AcquireScratchMemory(
                       collective_params, scratch_memory_requests,
                       collective_memory_cache, executor, collective_cliques));
  // Acquire collective memories requested by thunks.
  ASSIGN_OR_RETURN(CollectiveMemory collective_memory,
                   AcquireCollectiveMemory(
                       collective_params, collective_cliques,
                       collective_memory_requests, collective_memory_cache));
  {  // Initialize thunks using prepared resources before execution.
    Thunk::InitializeParams initialize_params{
        executor,
        executable_source,
        &buffer_allocations,
        main_stream,
        command_buffer_trace_stream,
        &collective_params,
        &collective_cliques,
        &collective_memory,
        &scratch_memory,
        run_options->run_options().ffi_execution_context(),
        run_options->local_device_count(),
        &execution_scoped_state};
    initialize_params.command_buffer_update_info = command_buffer_update_info;

    tsl::profiler::TraceMe trace_initialize("Thunks::Initialize");
    RETURN_IF_ERROR(thunk_executor.Initialize(initialize_params));
  }

  // Join a round of rendezvous after thunk initialization. We do this only in
  // presence of newly acquired collective cliques which means that we have
  // collective operations and clique initialization is famous for introducing
  // deadlocks if we try to execute it concurrently with other potentially
  // memory-allocating operations.
  if (!collective_cliques.empty()) {
    RETURN_IF_ERROR(RendezvousAfterInitialization(*run_options, debug_options));
  }

  // Prepare parameters for thunks execution.
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      *run_options, buffer_allocations, main_stream,
      command_buffer_trace_stream, &collective_params, &collective_cliques,
      &collective_memory, std::move(compute_streams.streams),
      &execution_scoped_state, command_buffer_update_info);

  XLA_VLOG_DEVICE(1, run_options->device_ordinal())
      << "Start GpuExecutable::ExecuteOnStream module: " << module_name;
  RETURN_IF_ERROR(thunk_executor.ExecuteOnStream(execute_params));
  XLA_VLOG_DEVICE(1, run_options->device_ordinal())
      << "End GpuExecutable::ExecuteOnStream module: " << module_name;

  // Collective kernel thunks may request a barrier after the module execution.
  // This might be needed for several reasons:
  // 1. To make sure that at the end of graph execution all reads and writes to
  //    the symmetric buffers are finished.
  // 2. To make sure that cuda module which uses a multimem handler used by
  //    another GPU will be unloaded only after all kernels are finished.
  //    Otherwise module unloading can cause a deadlock.
  absl::flat_hash_set<GlobalDeviceId> requested_barrier_devices =
      collective_clique_requests.GetDevicesRequiringBarrier();
  if (absl::c_linear_search(requested_barrier_devices,
                            collective_params.global_device_id.value())) {
    XLA_VLOG_DEVICE(1, collective_params.global_device_id.value())
        << "Barrier after executable required by participants: ("
        << absl::StrJoin(requested_barrier_devices, ", ") << ")";
    RETURN_IF_ERROR(BarrierAfterExecutable(*run_options, debug_options,
                                           *main_stream,
                                           requested_barrier_devices.size()));
  }

  return MaybeSyncAndProfile(run_options, execution_timer.get(),
                             block_host_until_done ? main_stream : nullptr);
}

namespace {
// Wrap RunId into a unique struct to guarantee we do not accidentally try to
// run multiple unrelated rendezvous for a same key.
struct InitializationKey {
  RunId run_id;

  template <typename H>
  friend H AbslHashValue(H h, const InitializationKey& key) {
    return H::combine(std::move(h), key.run_id);
  }
};

bool operator==(const InitializationKey& a, const InitializationKey& b) {
  return a.run_id == b.run_id;
}
}  // namespace

absl::Status RendezvousAfterInitialization(
    const ServiceExecutableRunOptions& run_options,
    const DebugOptions* absl_nullable debug_options) {
  // Thunk initialization can allocate new control data structures on device
  // that can lead to deadlocks if other replicas are executing concurrently
  // (i.e. this happens if we try to instantiate CUDA graph when other replica
  // is executing NCCL kernels). If we detect that we are running in multi-gpu
  // setup we synchronize after first initialization to make sure that all
  // replicas completed initialization process before we start execution.
  auto* gpu_opts = run_options.run_options().gpu_executable_run_options();
  auto* device_assn = run_options.run_options().device_assignment();

  // If we don't have Gpu executable options or device assignment it means we
  // are running in a single Gpu config and don't need a rendezvous.
  if (!gpu_opts || !device_assn) {
    return absl::OkStatus();
  }

  // Assume that all participants execute locally first, if we have a local
  // device id to global device id map we will use it to get the real number of
  // participating local devices.
  int64_t num_local_participants =
      device_assn->replica_count() * device_assn->computation_count();

  // Find what local devices are part of the device assignment.
  if (gpu_opts->gpu_global_device_ids()) {
    auto d2l_map = device_assn->GetDeviceToLogicalIdMap();

    num_local_participants = 0;
    for (auto& [local_id, global_id] : *gpu_opts->gpu_global_device_ids()) {
      num_local_participants += d2l_map.contains(global_id);
    }

    if (num_local_participants == 0) {
      return absl::InternalError(
          "Cound't find the number of local participants");
    }
  }

  XLA_VLOG_DEVICE(1, run_options.device_ordinal()) << absl::StreamFormat(
      "Join thunks initialization rendezvous with %d local participants",
      num_local_participants);

  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode(
        "RendezvousAfterInitialization",
        {{"run_id", run_options.run_options().run_id().ToInt()},
         {"num_local_participants", num_local_participants}});
  });

  auto rendezvous_key = InitializationKey{run_options.run_options().run_id()};
  auto rendezvous_name = absl::StrFormat(
      "thunk initialization completion for device ordinal %d; run_id=%d",
      run_options.device_ordinal(), run_options.run_options().run_id().ToInt());

  return Rendezvous(
      rendezvous_name, rendezvous_key, num_local_participants,
      absl::Seconds(
          debug_options
              ? debug_options->xla_gpu_executable_warn_stuck_timeout_seconds()
              : 10),
      absl::Seconds(
          debug_options
              ? debug_options->xla_gpu_executable_terminate_timeout_seconds()
              : 30));
}

absl::Status MaybeSyncAndProfile(const ServiceExecutableRunOptions* run_options,
                                 se::EventBasedTimer* execution_timer,
                                 se::Stream* stream_to_sync = nullptr) {
  // If we're measuring the execution time then it's important to queue the
  // stop event before triggering any synchronization.
  if (ExecutionProfile* profile =
          run_options->run_options().execution_profile();
      profile) {
    ASSIGN_OR_RETURN(absl::Duration elapsed,
                     execution_timer->GetElapsedDuration());
    profile->set_compute_time_ns(
        std::max(absl::ToDoubleNanoseconds(elapsed), 1.0));
  }

  // Make sure kernels are completed before deallocating temporary buffers or
  // the profiler state.
  // TODO(b/30100571): we could potentially postpone deallocating the temp
  // buffers until a different computation is executed.
  if (stream_to_sync) {
    absl::Status block_status = stream_to_sync->BlockHostUntilDone();
    if (!block_status.ok()) {
      return Internal(
          "Failed to complete all kernels launched on stream %p: %s",
          stream_to_sync, block_status.message());
    }
  }

  return absl::OkStatus();
}

absl::Status BarrierAfterExecutable(
    const ServiceExecutableRunOptions& run_options,
    const DebugOptions* absl_nullable debug_options, se::Stream& stream,
    const size_t num_participants) {
  RETURN_IF_ERROR(stream.BlockHostUntilDone());

  XLA_VLOG_DEVICE(1, run_options.device_ordinal()) << absl::StreamFormat(
      "Join thunks in barrier after module execution rendezvous with %d "
      "local "
      "participants",
      num_participants);

  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode(
        "RendezvousAfterExecution",
        {{"run_id", run_options.run_options().run_id().ToInt()},
         {"num_local_participants", num_participants}});
  });

  auto rendezvous_key = InitializationKey{run_options.run_options().run_id()};
  auto rendezvous_name = absl::StrFormat(
      "thunk barrier after module execution completion for device ordinal "
      "%d; run_id=%d",
      run_options.device_ordinal(), run_options.run_options().run_id().ToInt());

  return Rendezvous(
      rendezvous_name, rendezvous_key, num_participants,
      absl::Seconds(
          debug_options
              ? debug_options->xla_gpu_executable_warn_stuck_timeout_seconds()
              : 10),
      absl::Seconds(
          debug_options
              ? debug_options->xla_gpu_executable_terminate_timeout_seconds()
              : 30));
}

absl::Status GpuExecutable::ExecuteThunks(
    const BufferAllocations& buffer_allocations,
    const ServiceExecutableRunOptions* run_options,
    const Thunk::CommandBufferUpdateInfo* command_buffer_update_info) {
  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode(
        absl::StrFormat("[%d] GpuExecutable::ExecuteThunks",
                        run_options->device_ordinal()),
        {{"module_name", module_name_}});
  });

  if (VLOG_IS_ON(5)) {
    // Debug code to compare current allocation's address with previous run's
    // address, and report the allocation info if memory addressed changed.
    // Useful for identify in user's model if it is command buffer perf friendly
    // (no command buffer update cost).
    se::StreamExecutor* executor = run_options->stream()->parent();

    // Collect the set of allocations that changed between executions.
    std::vector<std::pair<int32_t, std::string>> changed_allocations;

    absl::MutexLock lock(module_handle_mutex_);
    if (module_allocations_.find(executor) == module_allocations_.end()) {
      std::vector<se::DeviceAddressBase> allocs_addr;
      allocs_addr.reserve(buffer_allocations.size());
      for (int i = 0; i < buffer_allocations.size(); i++) {
        allocs_addr.push_back(buffer_allocations.GetDeviceAddress(i));
      }
      module_allocations_[executor] = std::move(allocs_addr);
    } else {
      for (int i = 0; i < buffer_allocations.size(); i++) {
        if (module_allocations_[executor][i].IsSameAs(
                buffer_allocations.GetDeviceAddress(i))) {
          continue;
        }
        module_allocations_[executor][i] =
            buffer_allocations.GetDeviceAddress(i);
        const BufferAllocation& allocation =
            buffer_assignment_->GetAllocation(i);
        const char* allocation_type =
            allocation.is_entry_computation_parameter() ? "parameter"
            : allocation.maybe_live_out()               ? "live-out"
                                                        : "temp";
        changed_allocations.emplace_back(i, allocation_type);
      }
    }

    if (!changed_allocations.empty()) {
      XLA_VLOG_DEVICE(5, executor->device_ordinal()) << absl::StreamFormat(
          "Buffer allocations changed address between module %s executions: "
          "[%s]",
          module_name_,
          absl::StrJoin(changed_allocations, ", ", absl::PairFormatter(":")));
    }
  }

  se::DeviceAddressAllocator* const memory_allocator = run_options->allocator();
  // Force synchronous execution if the allocator requires it.
  const bool block_host_until_done =
      !memory_allocator->AllowsAsynchronousDeallocation();

  RETURN_IF_ERROR(
      CheckCompatibilityWithServiceExecutableRunOptions(run_options));

  ScopedAnnotation annotation([&] { return module_annotations_.top_level; });
  ScopedModuleAnnotations module_annotations(&module_annotations_);

  ModuleIdentifier unique_id = has_module() ? module().unique_id() : -1;
  Thunk::ExecutableSource executable_source = {text_, binary_,
                                               dnn_compiled_graphs_};

  se::StreamExecutor* executor = run_options->stream()->parent();

  XLA_VLOG_DEVICE(3, executor->device_ordinal()) << absl::StreamFormat(
      "ExecuteThunks: command_buffer_allocation_indexes_.size()=%d",
      command_buffer_va_remapping_->allocation_indices().size());

  bool collective_use_minimal_resource = false;
  if (has_module()) {
    ASSIGN_OR_RETURN(collective_use_minimal_resource,
                     ShouldCollectiveUseMinimalResource(module()));
  }
  RETURN_IF_ERROR(ExecuteThunksImpl(
      has_module() ? &module_config().debug_options() : nullptr, module_name_,
      unique_id, *thunk_executor_, executable_source, run_options,
      buffer_allocations, command_buffer_update_info, block_host_until_done,
      num_additional_streams_, collective_memory_cache_,
      collective_use_minimal_resource));
  return absl::OkStatus();
}

}  // namespace xla::gpu
