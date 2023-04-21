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

#include "xla/pjrt/pjrt_host_memory_space_and_buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/tracked_tfrt_cpu_device_buffer.h"
#include "xla/runtime/cpu_event.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/cpu_xfeed.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/traceme.h"
#include "tfrt/concurrency/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace xla {

static constexpr size_t kSmallDataTransferByteSize = 102400;  // 100 KiB

static std::vector<tfrt::RCReference<tfrt::AsyncValue>> CopyAsyncValues(
    absl::Span<const tfrt::RCReference<tfrt::AsyncValue>> events) {
  std::vector<tfrt::RCReference<tfrt::AsyncValue>> avs;
  avs.reserve(events.size());
  for (const auto& ev : events) {
    avs.push_back(ev.CopyRef());
  }
  return avs;
}

static void EnqueueWork(tsl::thread::ThreadPool* pool,
                        absl::AnyInvocable<void()> callee) {
  // TSL TheadPool expects std::function that must be copyable, so we are
  // forced to do a little bit of manual memory management here.
  pool->Schedule([ptr = new absl::AnyInvocable<void()>(std::move(callee))]() {
    (*ptr)();
    delete ptr;
  });
}

// Enqueue to PjRtClient pool when all `values` are ready.
static void EnqueueWorkWhenReady(
    tsl::thread::ThreadPool* pool,
    llvm::ArrayRef<tfrt::RCReference<tfrt::AsyncValue>> values,
    absl::AnyInvocable<void()> callee) {
  RunWhenReady(values, [pool, callee = std::move(callee)]() mutable {
    EnqueueWork(pool, std::move(callee));
  });
}

HostMemorySpace::HostMemorySpace(int id, PjRtClient* client, PjRtDevice* device)
    : id_(id), client_(client) {
  CHECK(device->client() == client_) << absl::StrFormat(
      "Cannot attach the HostMemorySpace to "
      "a device owned by a different client, the device's client: %s",
      device->client()->platform_name());
  devices_.push_back(device);
  debug_string_ = absl::StrFormat("HostMemorySpace(id=%i), client: %s", id_,
                                  client_->platform_name());
}

TfrtCpuBuffer::TfrtCpuBuffer(
    Shape on_device_shape,
    std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer,
    HostMemorySpace* memory_space)
    : on_device_shape_(std::move(on_device_shape)),
      memory_space_(memory_space),
      tracked_device_buffer_(std::move(tracked_device_buffer)) {}

TfrtCpuBuffer::~TfrtCpuBuffer() {
  Delete();
  CHECK_EQ(external_reference_counter_, 0);
}

PjRtDevice* TfrtCpuBuffer::device() const {
  CHECK_EQ(memory_space_->devices().size(), 1);
  return memory_space_->devices()[0];
}

StatusOr<size_t> TfrtCpuBuffer::GetOnDeviceSizeInBytes() const {
  return ShapeUtil::ByteSizeOf(on_device_shape_);
}

StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
TfrtCpuBuffer::AcquireExternalReference() {
  class ScopedExternalReference : public PjRtBuffer::ExternalReference {
   public:
    explicit ScopedExternalReference(TfrtCpuBuffer* buffer,
                                     std::shared_ptr<MaybeOwningCpuMemory> data)
        : buffer_(buffer), data_(std::move(data)) {
      DCHECK(data_);
      data_ptr_ = data_->data();
    }

    ~ScopedExternalReference() override { buffer_->DropExternalReference(); }

   private:
    TfrtCpuBuffer* buffer_ = nullptr;
    // Keep a reference to the underlying data used. Note that it is still
    // users' responsibility to synchronize reads and writes to the data.
    std::shared_ptr<MaybeOwningCpuMemory> data_;
  };

  absl::MutexLock lock(&mu_);
  if (tracked_device_buffer_ == nullptr) {
    return InvalidArgument("Buffer has been deleted or donated.");
  }

  ++external_reference_counter_;

  return {std::make_unique<ScopedExternalReference>(
      this, tracked_device_buffer_->Buffers()[0])};
}

class TrackedCpuDeviceBufferExternalReference
    : public PjRtBuffer::ExternalReference {
 public:
  explicit TrackedCpuDeviceBufferExternalReference(
      std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer)
      : tracked_device_buffer_(std::move(tracked_device_buffer)) {
    data_ptr_ = tracked_device_buffer_->Buffers()[0]->data();
  }

  ~TrackedCpuDeviceBufferExternalReference() override = default;

 private:
  std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer_;
};

StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
TfrtCpuBuffer::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  if (on_device_shape_.IsTuple()) {
    return InvalidArgument(
        "ReleaseDeviceMemoryOwnership allowed only for non-tuple");
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer,
      Release(wait_for_operations_to_complete));

  std::unique_ptr<PjRtBuffer::ExternalReference> ref;
  if (tracked_device_buffer) {
    ref = std::make_unique<TrackedCpuDeviceBufferExternalReference>(
        std::move(tracked_device_buffer));
  }
  return ref;
}

void TfrtCpuBuffer::CommitDonation() {
  absl::MutexLock lock(&mu_);
  CHECK(pending_donation_);
  CHECK(!tracked_device_buffer_);
  pending_donation_ = false;
}

void TfrtCpuBuffer::AbortDonation(
    std::unique_ptr<TrackedTfrtCpuDeviceBuffer> device_buffer) {
  absl::MutexLock lock(&mu_);
  CHECK(pending_donation_);
  CHECK(!tracked_device_buffer_);
  pending_donation_ = false;
  tracked_device_buffer_ = std::move(device_buffer);
}

void TfrtCpuBuffer::Delete() {
  auto device_buffer = ReleaseBufferLocked();
  if (device_buffer == nullptr) return;

  // Now that all holds have completed and no more can be added, we can get
  // the final set of usage events.
  absl::InlinedVector<tfrt::AsyncValueRef<runtime::CpuEvent>, 4> usage_events =
      device_buffer->LockUseAndTransferUsageEvents();

  std::vector<tfrt::AsyncValue*> event_avs;
  event_avs.reserve(usage_events.size() + 1);
  for (auto& event : usage_events) {
    event_avs.push_back(event.GetAsyncValue());
  }

  // We should also wait for the definition event.
  event_avs.push_back(device_buffer->definition_event().GetAsyncValue());

  RunWhenReady(event_avs, [device_buffer = std::move(device_buffer)]() mutable {
    device_buffer.reset();
  });
}

bool TfrtCpuBuffer::IsDeleted() {
  absl::MutexLock lock(&mu_);
  return tracked_device_buffer_ == nullptr;
}

std::unique_ptr<TrackedTfrtCpuDeviceBuffer>
TfrtCpuBuffer::ReleaseBufferLocked() {
  absl::MutexLock lock(&mu_);
  auto condition = [this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return !pending_donation_;
  };
  mu_.Await(absl::Condition(&condition));
  return std::move(tracked_device_buffer_);
}

StatusOr<std::unique_ptr<TrackedTfrtCpuDeviceBuffer>> TfrtCpuBuffer::Release(
    bool wait_for_operations_to_complete) {
  std::unique_ptr<TrackedTfrtCpuDeviceBuffer> device_buffer =
      ReleaseBufferLocked();
  if (device_buffer == nullptr) return {nullptr};

  absl::InlinedVector<tfrt::AsyncValueRef<runtime::CpuEvent>, 4> events;
  // Now that all holds have completed and no more can be added, we can get
  // the final set of usage events.
  events = device_buffer->LockUseAndTransferUsageEvents();

  if (wait_for_operations_to_complete) {
    // Block the host until all usage events have completed. Usage events
    // dominate definition events, so this also waits for the buffer to be
    // defined. Return the first error encountered.
    Status first_error;
    for (const auto& av : events) {
      BlockUntilReady(av.GetAsyncValue());
      if (auto* error = av.GetErrorIfPresent()) {
        first_error.Update(
            InternalError("Error Execute: %s", error->message()));
      }
    }
    if (!first_error.ok()) return std::move(first_error);
  }

  return device_buffer;
}

TrackedTfrtCpuDeviceBuffer* TfrtCpuBuffer::AcquireUsage(
    tfrt::AsyncValueRef<runtime::CpuEvent> usage_event) {
  absl::MutexLock lock(&mu_);
  if (!tracked_device_buffer_) {
    return nullptr;
  }

  tracked_device_buffer_->AddUsageEvents(absl::MakeSpan(&usage_event, 1));
  return tracked_device_buffer_.get();
}

StatusOr<TfrtCpuBuffer::DonationTransaction> TfrtCpuBuffer::AcquireDonation() {
  absl::MutexLock lock(&mu_);

  if (tracked_device_buffer_ == nullptr) {
    return InvalidArgument("Donation requested for invalid buffer");
  }

  if (external_reference_counter_ > 0) {
    return InvalidArgument(
        "Donation requested for buffer with external reference");
  }

  CHECK(!pending_donation_);
  pending_donation_ = true;

  // Swap out `tracked_device_buffer_` so that no one can acquire a usage event
  // after this point.
  return DonationTransaction(this, std::move(tracked_device_buffer_));
}

static ShapedBuffer AsShapedBuffer(
    int device_ordinal, const Shape& on_device_shape,
    absl::Span<const std::shared_ptr<MaybeOwningCpuMemory>> buffers) {
  ShapedBuffer shaped_buffer(on_device_shape, device_ordinal);
  ShapeTree<se::DeviceMemoryBase>::iterator iterator =
      shaped_buffer.buffers().begin();
  for (const auto& buf : buffers) {
    CHECK(iterator != shaped_buffer.buffers().end());
    iterator->second = se::DeviceMemoryBase(buf->data(), buf->size());
    ++iterator;
  }
  CHECK(iterator == shaped_buffer.buffers().end());
  return shaped_buffer;
}

StatusOr<Shape> TfrtCpuBuffer::logical_on_device_shape() {
  if (on_device_shape_.is_static()) {
    return on_device_shape_;
  }

  auto usage_event = tfrt::MakeConstructedAsyncValueRef<runtime::CpuEvent>();
  auto* device_buffer = AcquireUsage(usage_event);
  if (device_buffer == nullptr) {
    return InvalidArgument(
        "logical_on_device_shape() called on deleted or donated buffer");
  }
  MarkEventReadyOnExit ready_on_exit(std::move(usage_event));

  // Wait for the definition event.
  const auto& av = device_buffer->definition_event();
  BlockUntilReady(av.GetAsyncValue());
  if (auto* error = av.GetErrorIfPresent()) {
    return InternalError("Error Execute: %s", error->message());
  }

  CHECK_EQ(memory_space_->devices().size(), 1);
  ShapedBuffer shaped_buffer =
      AsShapedBuffer(memory_space_->devices()[0]->local_hardware_id(),
                     on_device_shape_, device_buffer->Buffers());
  Shape ret_shape = on_device_shape_;
  TF_RETURN_IF_ERROR(ReadDynamicShapesOnCpu(
      &shaped_buffer, &ret_shape, cpu::CpuExecutable::ShapeSizeBytes));
  return ret_shape;
}

PjRtFuture<Status> TfrtCpuBuffer::ToLiteral(MutableLiteralBase* literal) {
  tsl::profiler::TraceMe traceme("TfrtCpuBuffer::ToLiteral");
  if (IsEmptyTuple()) {
    return PjRtFuture<Status>(
        InvalidArgument("ToLiteral called on empty tuple"));
  }
  auto usage_event = tfrt::MakeConstructedAsyncValueRef<runtime::CpuEvent>();
  auto* device_buffer = AcquireUsage(usage_event);
  if (device_buffer == nullptr) {
    return PjRtFuture<Status>(InvalidArgument(
        "CopyToHostAsync() called on deleted or donated buffer"));
  }
  MarkEventReadyOnExit ready_on_exit(std::move(usage_event));

  std::vector<tfrt::RCReference<tfrt::AsyncValue>> device_buffer_wait_avs = {
      device_buffer->definition_event().CopyRCRef()};
  std::vector<tfrt::RCReference<tfrt::AsyncValue>> device_buffer_wait_avs_copy =
      CopyAsyncValues(device_buffer_wait_avs);

  bool should_sync_copy = device_buffer_wait_avs.empty() &&
                          literal->size_bytes() < kSmallDataTransferByteSize;
  StatusOr<Shape> device_shape = logical_on_device_shape();
  if (!device_shape.ok()) {
    return PjRtFuture<Status>(device_shape.status());
  }
  if (should_sync_copy) {
    if (!on_device_shape().IsTuple()) {
      const std::shared_ptr<MaybeOwningCpuMemory>& b =
          device_buffer->Buffers()[0];
      std::memcpy(literal->untyped_data(), b->data(),
                  ShapeUtil::ByteSizeOf(*device_shape));
    } else {
      // Tuple case.
      int num_leaves = literal->shape().tuple_shapes().size();
      for (int i = 0; i < num_leaves; ++i) {
        const std::shared_ptr<MaybeOwningCpuMemory>& b =
            device_buffer->Buffers()[i];
        std::memcpy(
            literal->untyped_data({i}), b->data(),
            ShapeUtil::ByteSizeOf(ShapeUtil::GetSubshape(*device_shape, {i})));
      }
    }
    // Unblock ToLiteral caller.
    return PjRtFuture<Status>(OkStatus());
  } else {
    auto ready_event = tfrt::MakeUnconstructedAsyncValueRef<Status>();
    // Wait for buffer definition events to finish before d2h dispatch. D2H
    // dispatch should be in parallel, e.g. one Execute event finish may trigger
    // multiple outputs' D2H, they should happen in different threads in
    // parallel.
    EnqueueWorkWhenReady(
        client()->pjrt_client_thread_pool(), device_buffer_wait_avs,
        [this, device_buffer_wait_avs = std::move(device_buffer_wait_avs_copy),
         literal, ready_event = ready_event.CopyRef(), device_buffer,
         device_shape, ready_on_exit = std::move(ready_on_exit)]() mutable {
          tsl::profiler::TraceMe traceme("D2H Dispatch");
          // Errors in src buffer are surfaced to user.
          for (const auto& av : device_buffer_wait_avs) {
            if (auto* error = av->GetErrorIfPresent()) {
              ready_event.emplace(Internal("Error converting to literal: %s",
                                           error->message()));
              return;
            }
          }

          if (!on_device_shape().IsTuple()) {
            const std::shared_ptr<MaybeOwningCpuMemory>& b =
                device_buffer->Buffers()[0];
            std::memcpy(literal->untyped_data(), b->data(),
                        ShapeUtil::ByteSizeOf(*device_shape));
          } else {
            // Tuple case.
            int num_leaves = literal->shape().tuple_shapes().size();
            for (int i = 0; i < num_leaves; ++i) {
              const std::shared_ptr<MaybeOwningCpuMemory>& b =
                  device_buffer->Buffers()[i];
              std::memcpy(literal->untyped_data({i}), b->data(),
                          ShapeUtil::ByteSizeOf(
                              ShapeUtil::GetSubshape(*device_shape, {i})));
            }
          }

          // Unblock ToLiteral event.
          ready_event.emplace(OkStatus());
        });
    return PjRtFuture<Status>(
        std::move(ready_event),
        /*on_block_start=*/
        []() {
          tsl::profiler::TraceMeProducer traceme("TfrtCpuBuffer::ToLiteral");
          VLOG(1) << "TfrtCpuBuffer::ToLiteral";
          return PjRtFutureHelpers::ProfilingKeys(
              {/*traceme_context_id =*/traceme.GetContextId()});
        },
        /*on_block_end=*/
        [](PjRtFutureHelpers::ProfilingKeys keys) {
          tsl::profiler::TraceMeConsumer traceme("TfrtCpuBuffer::ToLiteral",
                                                 keys.traceme_context_id);
        });
  }
}

// TODO(zhangqiaorjc): Consider disallowing multiple CPU devices and assign
// multiple pmap replicas to the same CPU device for multi-CPU pmap testing.
StatusOr<std::unique_ptr<PjRtBuffer>> TfrtCpuBuffer::CopyToDevice(
    PjRtDevice* dst_device) {
  tsl::profiler::TraceMe traceme("TfrtCpuBuffer::CopyToDevice");
  // TODO(zhangqiaorjc): Remove this restriction after removing the test that
  // explicitly asserts this.
  CHECK_EQ(memory_space_->devices().size(), 1);
  if (dst_device == memory_space_->devices()[0]) {
    return InvalidArgument(
        "CopyToDevice cannot accept the same source and destination devices");
  }

  // Copying across PjRtClients involves a copy through the host.
  if (dst_device->client() != this->client()) {
    TF_ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal, ToLiteralSync());
    // Avoid use-after-free on `literal` due to unsequenced move and use.
    Literal* literal_pointer = literal.get();
    absl::InlinedVector<int64_t, 4> byte_strides(
        literal->shape().dimensions_size());
    TF_RETURN_IF_ERROR(
        ShapeUtil::ByteStrides(literal->shape(), absl::MakeSpan(byte_strides)));
    return dst_device->client()->BufferFromHostBuffer(
        literal_pointer->untyped_data(),
        literal_pointer->shape().element_type(),
        literal_pointer->shape().dimensions(), byte_strides,
        PjRtClient::HostBufferSemantics::kZeroCopy,
        [literal{std::move(literal)}]() { /* frees literal */ }, dst_device);
  }

  // Copy each leaf buffer to a destination buffer.
  auto usage_event = tfrt::MakeConstructedAsyncValueRef<runtime::CpuEvent>();
  auto* src_device_buffer = AcquireUsage(usage_event);
  if (src_device_buffer == nullptr) {
    return InvalidArgument("CopyToDevice called on deleted or donated buffer");
  }
  MarkEventReadyOnExit ready_on_exit(std::move(usage_event));

  int num_leaf_buffers = src_device_buffer->Buffers().size();
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> src_buffers;
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> dst_buffers;
  absl::InlinedVector<tfrt::AsyncValueRef<runtime::CpuEvent>, 4>
      dst_definition_events;
  src_buffers.reserve(num_leaf_buffers);
  dst_buffers.reserve(num_leaf_buffers);
  dst_definition_events.reserve(num_leaf_buffers);

  for (int i = 0; i < num_leaf_buffers; ++i) {
    auto src_buffer = src_device_buffer->Buffers()[i];
    TF_ASSIGN_OR_RETURN(auto dst_buffer, MaybeOwningCpuMemory::AllocateShared(
                                             src_buffer->size()));
    src_buffers.push_back(std::move(src_buffer));
    dst_buffers.push_back(std::move(dst_buffer));
    dst_definition_events.push_back(
        tfrt::MakeConstructedAsyncValueRef<runtime::CpuEvent>());
  }

  // Wait for src buffer definition events to finish before d2d dispatch.
  // Errors are propagated asynchronously in dst buffer's definition events.
  const auto& src_definition_event = src_device_buffer->definition_event();

  auto copy_task = [num_leaf_buffers, src_buffers = std::move(src_buffers),
                    dst_buffers_copies = dst_buffers, dst_definition_events,
                    src_definition_event,
                    ready_on_exit = std::move(ready_on_exit)]() mutable {
    tsl::profiler::TraceMe traceme("D2D Dispatch");
    if (auto* error = src_definition_event.GetErrorIfPresent()) {
      for (int i = 0; i < num_leaf_buffers; ++i) {
        // Any error discovered in src buffer are propagated to dst buffer
        // definition events, which will surface to users in
        // dst_buffer->ToLiteral().
        dst_definition_events[i].SetError(*error);
      }
      return;
    }

    for (int i = 0; i < num_leaf_buffers; ++i) {
      std::memcpy(dst_buffers_copies[i]->data(), src_buffers[i]->data(),
                  src_buffers[i]->size());
      dst_definition_events[i].SetStateConcrete();
    }
  };

  src_definition_event.AndThen([pool = client()->pjrt_client_thread_pool(),
                                copy_task = std::move(copy_task)]() mutable {
    EnqueueWork(pool, std::move(copy_task));
  });

  return std::unique_ptr<PjRtBuffer>(std::make_unique<TfrtCpuBuffer>(
      on_device_shape_,
      std::make_unique<TrackedTfrtCpuDeviceBuffer>(
          on_device_shape_.IsTuple(), std::move(dst_buffers),
          std::move(dst_definition_events)),
      tensorflow::down_cast<HostMemorySpace*>(
          dst_device->host_memory_space())));
}

PjRtFuture<Status> TfrtCpuBuffer::GetReadyFuture() {
  tfrt::AsyncValueRef<runtime::CpuEvent> definition_event;
  {
    absl::MutexLock lock(&mu_);
    if (!tracked_device_buffer_) {
      return PjRtFuture<Status>(InvalidArgument(
          "GetReadyFuture() called on deleted or donated buffer"));
    }
    definition_event = tracked_device_buffer_->definition_event();
  }
  DCHECK(definition_event);

  if (definition_event.IsAvailable()) {
    if (definition_event.IsError()) {
      return PjRtFuture<Status>(
          FailedPrecondition("Buffer Definition Event: %s",
                             definition_event.GetError().message()));
    }
    return PjRtFuture<Status>(OkStatus());
  } else {
    tfrt::AsyncValueRef<Status> status_event =
        tfrt::MakeUnconstructedAsyncValueRef<Status>();

    definition_event.AndThen(
        [definition_event = definition_event.AsPtr(), status_event]() {
          if (definition_event.IsError()) {
            status_event.emplace(
                FailedPrecondition("Buffer Definition Event: %s",
                                   definition_event.GetError().message()));
          } else {
            status_event.emplace(OkStatus());
          }
        });

    return PjRtFuture<Status>(
        std::move(status_event),
        /*on_block_start=*/
        []() {
          tsl::profiler::TraceMeProducer traceme("TfrtCpuBuffer::Await");
          VLOG(1) << "TfrtCpuBuffer::Await";
          return PjRtFutureHelpers::ProfilingKeys(
              {/*traceme_context_id=*/traceme.GetContextId()});
        },
        /*on_block_end=*/
        [](PjRtFutureHelpers::ProfilingKeys keys) {
          tsl::profiler::TraceMeConsumer traceme("TfrtCpuBuffer::Await",
                                                 keys.traceme_context_id);
        });
  }
}

}  // namespace xla
