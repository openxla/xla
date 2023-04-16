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

#include "xla/stream_executor/singleton_stream_manager.h"

#include <memory>

namespace stream_executor {

StreamGroup::StreamGroup(StreamExecutor* executor, StreamGroupOptions options)
    : priority(options.priority) {
  compute_stream = std::make_shared<Stream>(executor);
  compute_stream->implementation()->SetPriority(priority);
  compute_stream->Init();

  host_to_device_stream = std::make_shared<Stream>(executor);
  host_to_device_stream->implementation()->SetPriority(priority);
  host_to_device_stream->Init();

  for (int i = 0; i < options.num_device_to_host_streams; ++i) {
    auto stream = std::make_shared<Stream>(executor);
    stream->implementation()->SetPriority(priority);
    stream->Init();
    device_to_host_streams.push_back(stream);
  }

  for (int i = 0; i < options.num_device_to_device_streams; ++i) {
    auto stream = std::make_shared<Stream>(executor);
    stream->implementation()->SetPriority(priority);
    stream->Init();
    device_to_device_streams.push_back(stream);
  }

  LOG(INFO) << "Created stream_executor::StreamGroup. Compute stream: "
            << compute_stream.get() << " StreamExecutor: " << executor;
}

const StreamGroup& StreamManager::GetOrCreate(
    int device_ordinal, StreamExecutor* executor, int priority,
    int num_device_to_host_streams, int num_device_to_device_streams) {
  tsl::mutex_lock guard(lock_);
  StreamGroup::StreamGroupOptions options;
  options.priority = priority;
  options.num_device_to_host_streams = num_device_to_host_streams;
  options.num_device_to_device_streams = num_device_to_device_streams;
  auto emplace_result = stream_groups_.try_emplace(
      device_ordinal, StreamGroup(executor, options));
  const auto& iter = emplace_result.first;
  if (!emplace_result.second) {
    int d2h_streams_size = iter->second.device_to_host_streams.size();
    int d2d_streams_size = iter->second.device_to_device_streams.size();
    if (d2h_streams_size != num_device_to_device_streams) {
      LOG(ERROR) << "stream_executor::StreamGroup for device[" << device_ordinal
                 << "] already created but number of device to host streams: "
                 << d2h_streams_size << " is not equal to requested number: "
                 << num_device_to_host_streams;
    }
    if (d2d_streams_size != num_device_to_device_streams) {
      LOG(ERROR) << "stream_executor::StreamGroup for device[" << device_ordinal
                 << "] already created but number of device to device streams: "
                 << d2d_streams_size << " is not equal to requested number: "
                 << num_device_to_device_streams;
    }
  }
  return iter->second;
}

}  // namespace stream_executor
