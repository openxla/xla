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

#ifndef XLA_STREAM_EXECUTOR_SINGLETON_STREAM_MANAGER_H_
#define XLA_STREAM_EXECUTOR_SINGLETON_STREAM_MANAGER_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/mutex.h"

namespace stream_executor {

struct StreamGroup {
  struct StreamGroupOptions {
    int priority = 0;
    int num_device_to_host_streams = 1;
    int num_device_to_device_streams = 1;
  };

  StreamGroup(StreamExecutor* executor, StreamGroupOptions options);
  std::shared_ptr<Stream> compute_stream;
  std::shared_ptr<Stream> host_to_device_stream;
  std::vector<std::shared_ptr<Stream>> device_to_host_streams;
  std::vector<std::shared_ptr<Stream>> device_to_device_streams;
  int priority;
};

class StreamManager {
 public:
  const StreamGroup& GetOrCreate(int device_ordinal, StreamExecutor* executor,
                                 int priority = 0,
                                 int num_device_to_host_streams = 1,
                                 int num_device_to_device_streams = 1);

  // Returns a reference to the StreamManager singleton. Note that this is never
  // destroyed, so the objects it owns are never deleted.
  static StreamManager& Global() {
    static StreamManager* instance = new StreamManager();
    return *instance;
  }

 private:
  std::unordered_map<int, StreamGroup> stream_groups_;
  tsl::mutex lock_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SINGLETON_STREAM_MANAGER_H_
