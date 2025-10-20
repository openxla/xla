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

#include "xla/pjrt/se_raw_buffer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/buffer_sequencing_event.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/tracked_device_buffer.h"
#include "xla/pjrt/transpose.h"
#include "xla/primitive_util.h"
#include "xla/service/generic_transfer_manager.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/casts.h"
#include "tsl/profiler/lib/connected_traceme.h"

namespace xla {

Future<> PjRtStreamExecutorDeviceEvent::GetReadyFuture() {
  auto [promise, future] = Future<>::MakePromise();
  event_.AndThen([promise = std::move(promise), event = event_]() mutable {
    if (auto* error = event.GetErrorIfPresent()) {
      promise.Set(*error);
    } else {
      promise.Set();
    }
  });

  return FutureHelpers::WithProfiling(
      std::move(future),
      /*on_block_start=*/
      [callee_method = callee_method_, callee_type = callee_type_]() {
        tsl::profiler::TraceMeProducer traceme(
            [&] { return absl::StrCat(callee_type, "::", callee_method); });
        return FutureHelpers::ProfilingKeys({traceme.GetContextId()});
      },
      /*on_block_end=*/
      [callee_method = callee_method_,
       callee_type = callee_type_](FutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme(
            [&] { return absl::StrCat(callee_type, "::", callee_method); },
            keys.traceme_context_id);
      });
}

PjRtStreamExecutorDeviceEventPromise::PjRtStreamExecutorDeviceEventPromise(
    PjRtMemorySpace* memory_space, LocalDeviceState* local_device,
    tsl::thread::ThreadPool* thread_pool)
    : memory_space_(memory_space),
      local_device_(local_device),
      av_(tsl::MakeIndirectAsyncValue()),
      event_(tsl::MakeConstructedAsyncValueRef<BufferSequencingEvent>(
          thread_pool,
          tsl::AsyncValueRef<BufferSequencingEvent::EventState>(av_))) {}

void PjRtStreamExecutorDeviceEventPromise::Set(
    tsl::RCReference<PjRtDeviceEvent> event) {
  SetFromSEEvent(
      tensorflow::down_cast<PjRtStreamExecutorDeviceEvent*>(event.get())
          ->event());
}

void PjRtStreamExecutorDeviceEventPromise::SetFromSEEvent(
    BufferSequencingEventRef event) {
  av_->ForwardTo(event->event().CopyRCRef());
  event.AndThen([event = event_, original_event = event]() {
    if (auto* error = original_event.GetErrorIfPresent()) {
      event.SetError(*error);
    }
    event.SetStateConcrete();
  });
}

void PjRtStreamExecutorDeviceEventPromise::SetReady() {
  auto* client =
      tensorflow::down_cast<PjRtStreamExecutorClient*>(memory_space_->client());
  auto result = BufferSequencingEvent::Create(client->thread_pool());
  auto stream = local_device_->BorrowStreamFromPool();
  auto status =
      client->AllocateAndRecordEvent(result, local_device_, stream.get());
  local_device_->ReturnStreamToPool(std::move(stream));
  if (!status.ok()) {
    SetError(status);
  } else {
    SetFromSEEvent(result);
  }
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
PjRtStreamExecutorRawBuffer::CopyRawHostToDeviceAndReturnEvent(
    const void* src, int64_t offset, int64_t transfer_size) {
  se::Stream* stream = local_device_->host_to_device_stream();
  auto device_event = BufferSequencingEvent::Create(client_->thread_pool());
  client_->thread_pool()->Schedule(
      [client = client_, device_event, local_device = local_device_, stream,
       src, offset, transfer_size, buf = tsl::FormRef(this)]() mutable {
        se::DeviceMemoryBase sub_buffer = buf->device_buffer_->mem();
        if (transfer_size < sub_buffer.size()) {
          sub_buffer = sub_buffer.GetByteSlice(offset, transfer_size);
        }
        client->WaitForAllocation(stream, *buf);
        auto status = stream->Memcpy(&sub_buffer, src, transfer_size);
        if (status.ok()) {
          status = client->AllocateAndRecordEvent(device_event, local_device,
                                                  stream);
        }
        if (!status.ok()) {
          client->SetEventAsError(device_event, status);
        }
      });
  return tsl::MakeRef<PjRtStreamExecutorDeviceEvent>(
      std::move(device_event), "PjRtStreamExecutorRawBuffer",
      "CopyRawHostToDevice");
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
PjRtStreamExecutorRawBuffer::CopyRawDeviceToHostAndReturnEvent(
    void* dst, int64_t offset, int64_t transfer_size) {
  se::Stream* stream = local_device_->GetDeviceToHostStream();
  auto device_event = BufferSequencingEvent::Create(client_->thread_pool());
  client_->thread_pool()->Schedule(
      [client = client_, device_event, local_device = local_device_, stream,
       dst, offset, transfer_size, buf = tsl::FormRef(this)]() mutable {
        se::DeviceMemoryBase sub_buffer = buf->device_buffer_->mem();
        if (transfer_size < sub_buffer.size()) {
          sub_buffer = sub_buffer.GetByteSlice(offset, transfer_size);
        }
        client->WaitForAllocation(stream, *buf);
        auto status = stream->Memcpy(dst, sub_buffer, transfer_size);
        if (status.ok()) {
          status = client->AllocateAndRecordEvent(device_event, local_device,
                                                  stream);
        }
        if (!status.ok()) {
          client->SetEventAsError(device_event, status);
        }
      });
  return tsl::MakeRef<PjRtStreamExecutorDeviceEvent>(
      std::move(device_event), "PjRtStreamExecutorRawBuffer",
      "CopyRawDeviceToHost");
}

ShapedBuffer PjRtStreamExecutorRawBuffer::AsShapedBuffer(
    const xla::Shape& shape) {
  auto* device = memory_space()->devices()[0];
  ShapedBuffer shaped_buffer(shape, device->local_device_id().value(),
                             device->local_hardware_id().value());
  ShapeTree<se::DeviceMemoryBase>::iterator iterator =
      shaped_buffer.buffers().begin();
  if (device_buffer_) {
    CHECK(iterator != shaped_buffer.buffers().end());
    iterator->second = device_buffer_->mem();
    ++iterator;
  }
  CHECK(iterator == shaped_buffer.buffers().end());
  return shaped_buffer;
}

void PjRtStreamExecutorRawBuffer::ReadDynamicShape(
    tsl::AsyncValueRef<xla::Shape> output_shape, xla::Shape shape) {
  auto* stream = local_device_->GetDeviceToHostStream();
  auto shaped_buffer = AsShapedBuffer(shape);
  TransferManager* transfer_manager =
      client_->client()->backend().transfer_manager();
  auto status = transfer_manager->ReadDynamicShapes(stream, &shaped_buffer,
                                                    &*output_shape);
  if (!status.ok()) {
    output_shape.SetError(status);
  } else {
    output_shape.SetStateConcrete();
  }
}

void PjRtStreamExecutorRawBuffer::CopyToLiteralAsync(
    Promise<> promise, tsl::RCReference<PjRtDeviceEventPromise> device_promise,
    MutableLiteralBase* literal, xla::Shape shape) {
  auto usage_event = BufferSequencingEvent::Create(client_->thread_pool());
  usage_event.AndThen([device_buffer = device_buffer_]() {});
  client_->async_work_runner()->Schedule(
      [usage_event, local_device = local_device_,
       on_device_shape = std::move(shape), promise = std::move(promise),
       literal, client = client_, memory_space = memory_space_,
       device_buffer = device_buffer_]() mutable {
        std::shared_ptr<TransposePlan> transpose;
        se::Stream* stream = local_device->GetDeviceToHostStream();
        TransferManager* transfer_manager =
            client->client()->backend().transfer_manager();
        if (on_device_shape.IsArray()) {
          xla::Layout literal_layout;
          if (literal->shape().has_layout()) {
            literal_layout = literal->shape().layout();
          } else {
            literal_layout = LayoutUtil::MakeDescendingLayout(
                on_device_shape.dimensions().size());
          }

          if (on_device_shape.layout() != literal_layout) {
            absl::InlinedVector<int64_t, 4> byte_strides(
                on_device_shape.dimensions().size());
            absl::Status s = ShapeUtil::ByteStrides(
                on_device_shape, absl::MakeSpan(byte_strides));
            if (!s.ok()) {
              promise.Set(s);
              client->SetEventAsError(usage_event, s);
              return;
            }
            absl::Span<const int64_t> dims = on_device_shape.dimensions();
            absl::InlinedVector<int64_t, 4> permutation(dims.size());
            absl::c_reverse_copy(literal_layout.minor_to_major(),
                                 permutation.begin());
            TransposePlan::Options options;
            options.elem_size_in_bytes =
                primitive_util::ByteWidth(on_device_shape.element_type());
            options.dims = on_device_shape.dimensions();
            options.permutation = permutation;
            options.input_layout = TransposePlan::Striding{byte_strides};
            {
              absl::MutexLock lock(&client->transpose_mu_);
              absl::StatusOr<std::shared_ptr<TransposePlan>> t =
                  client->transpose_cache_.GetOrCreate(options);
              if (!t.ok()) {
                promise.Set(t.status());
                client->SetEventAsError(usage_event, t.status());
                return;
              }
              transpose = *std::move(t);
            }
          }
        }

        absl::StatusOr<EventPool::Handle> event_or =
            local_device->event_pool().AllocateEvent(stream->parent());
        if (!event_or.ok()) {
          promise.Set(event_or.status());
          client->SetEventAsError(usage_event, event_or.status());
          return;
        }

        ShapedBuffer shaped_buffer = device_buffer->AsShapedBuffer(
            memory_space->devices()[0], on_device_shape);

        GenericTransferManager::LiteralFromDeviceMetadata transfer_metadata;
        // We never call device functions from the `done` callback.
        transfer_metadata.callback_is_host_callback_safe = true;

        TransferManager::TransferMetadata* transfer_metadata_ptr =
            (dynamic_cast<GenericTransferManager*>(transfer_manager) != nullptr)
                ? &transfer_metadata
                : nullptr;

        if (transpose) {
          // Copy the device buffer to a temporary literal with descending
          // layout and transpose to the requested layout.

          Shape stage_shape = literal->shape();
          *stage_shape.mutable_layout() =
              LayoutUtil::MakeDescendingLayout(stage_shape.dimensions().size());
          auto staged = std::make_shared<Literal>(stage_shape);

          transfer_manager->TransferLiteralFromDevice(
              stream, shaped_buffer, staged.get(),
              [transpose = std::move(transpose),
               promise = std::move(promise).ToShared(), staged,
               literal = std::move(literal)](absl::Status status) mutable {
                if (status.ok()) {
                  transpose->Execute(staged->untyped_data(),
                                     literal->untyped_data());
                }
                promise->Set(std::move(status));
              },
              transfer_metadata_ptr);
        } else {
          transfer_manager->TransferLiteralFromDevice(
              stream, shaped_buffer, literal,
              [promise =
                   std::move(promise).ToShared()](absl::Status status) mutable {
                promise->Set(std::move(status));
              },
              transfer_metadata_ptr);
        }

        client->ThenRecordEvent(usage_event, local_device,
                                std::move(event_or).value(), stream);
      });

  device_promise->Set(
      tsl::MakeRef<PjRtStreamExecutorDeviceEvent>(std::move(usage_event)));
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
PjRtStreamExecutorRawBuffer::MakeAllocationReadyEvent() {
  auto* client =
      tensorflow::down_cast<PjRtStreamExecutorClient*>(memory_space_->client());
  TF_ASSIGN_OR_RETURN(auto result,
                      device_buffer_->GetDefinitionEvent(
                          client->thread_pool(), /*nullptr_if_past=*/false));
  if (!result) {
    result = BufferSequencingEvent::Create(client->thread_pool());
    auto stream = local_device_->BorrowStreamFromPool();
    auto status =
        client->AllocateAndRecordEvent(result, local_device_, stream.get());
    local_device_->ReturnStreamToPool(std::move(stream));
    TF_RETURN_IF_ERROR(status);
  }
  return tsl::MakeRef<PjRtStreamExecutorDeviceEvent>(std::move(result));
}

void PjRtStreamExecutorRawBuffer::CopyTo(
    tsl::RCReference<CommonPjRtRawBuffer> dst_raw_buffer,
    tsl::RCReference<PjRtDeviceEventPromise> definition_event_promise,
    tsl::RCReference<PjRtDeviceEventPromise> src_usage_event_promise,
    ::tsl::AsyncValueRef<bool> allocation_event) {
  auto status = absl::UnimplementedError("CopyTo not implemented");
  src_usage_event_promise->SetError(status);
  if (allocation_event) {
    allocation_event.SetError(status);
  }
  definition_event_promise->SetError(status);
}

}  // namespace xla
