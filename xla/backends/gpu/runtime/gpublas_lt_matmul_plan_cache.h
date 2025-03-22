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

#ifndef XLA_BACKENDS_GPU_RUNTIME_GPUBLAS_LT_MATMUL_PLAN_CACHE_H_
#define XLA_BACKENDS_GPU_RUNTIME_GPUBLAS_LT_MATMUL_PLAN_CACHE_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/container/node_hash_map.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

struct MatmulPlanCache {

  static MatmulPlanCache& GetCacheForExecutor(se::StreamExecutor *exec) {
    static absl::Mutex m(absl::kConstInit);
    // Each stream executor gets a different cache instance
    static absl::node_hash_map< se::StreamExecutor *, MatmulPlanCache > meta;
    absl::MutexLock lock(&m);
    return meta[exec];
  }

  template < class Func >
  absl::StatusOr<se::gpu::BlasLt::MatmulPlan *> 
          GetOrCreate(const std::string& key, Func&& create) {
    // each GPU has a different mutex => hence different GPU instances can
    // create matmul plans in parallel
    absl::MutexLock lock(&mutex_); 
    auto res = map_.emplace(key, se::gpu::BlasLt::MatmulPlanPtr{});
    // New entry inserted: always create a new matmul plan if key is empty, 
    // this is used by command_buffer_thunk test.
    if(res.second || key.empty()) { 
      VLOG(2) << "Creating a plan for: " << key;
      TF_ASSIGN_OR_RETURN(res.first->second, std::forward<Func>(create)());
      VLOG(2) << "Plan created: cache size: " << map_.size();
    } 
    return res.first->second.get();
  }

  size_t size() const {
    absl::MutexLock lock(&mutex_); 
    return map_.size();
  }

  void clear() {
    absl::MutexLock lock(&mutex_); 
    map_.clear();
  }

  MatmulPlanCache() = default;
  
private:
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<std::string, se::gpu::BlasLt::MatmulPlanPtr> map_
          ABSL_GUARDED_BY(mutex_);
};

} // namespace gpu
} // namespace xla

#endif // XLA_BACKENDS_GPU_RUNTIME_GPUBLAS_LT_MATMUL_PLAN_CACHE_H_
