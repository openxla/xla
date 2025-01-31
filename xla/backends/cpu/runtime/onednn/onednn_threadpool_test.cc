/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/onednn/onednn_threadpool.h"

#include <unordered_map>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"
#include "third_party/intel_dnnl/include/oneapi/dnnl/dnnl_common.hpp"
#include "third_party/intel_dnnl/include/oneapi/dnnl/dnnl_threadpool.hpp"
#include "third_party/intel_dnnl/include/oneapi/dnnl/dnnl_types.h"
#include "xla/backends/cpu/runtime/parallel_loop_runner.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

TEST(OneDnnThreadPoolTest, Binary) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 32);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());

  ParallelLoopRunner runner(&device);
  OneDnnThreadPool threadpool(&runner);

  dnnl::engine engine(dnnl::engine::kind::cpu, 0);

  int64_t d0 = 1000;
  int64_t d1 = 1000;
  int64_t num_elements = d0 * d1;

  dnnl::memory::dims src_dims = {d0, d1};
  dnnl::memory::dims dst_dims = {d0, d1};

  std::vector<float> src_data(num_elements, 1.0f);
  std::vector<float> dst_data(num_elements, 0.0f);

  // Create src and dst memory descriptors and memory objects.
  auto src_md = dnnl::memory::desc(src_dims, dnnl::memory::data_type::f32,
                                   dnnl::memory::format_tag::ab);
  auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32,
                                   dnnl::memory::format_tag::ab);
  auto src_mem = dnnl::memory(src_md, engine, src_data.data());
  auto dst_mem = dnnl::memory(dst_md, engine, dst_data.data());

  // Create description and primitive for element-wise exp.
  auto exp_pd = dnnl::eltwise_forward::primitive_desc(
      engine, dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_exp,
      src_md, dst_md);

  auto exp_prim = dnnl::eltwise_forward(exp_pd);

  // Create onednn stream backed by parallel loop runner.
  auto stream =
      dnnl::stream(dnnl::threadpool_interop::make_stream(engine, &threadpool));

  // Execute the primitive using our custom threadpool.
  std::unordered_map<int, dnnl::memory> exp_args;
  exp_args.insert({DNNL_ARG_SRC, src_mem});
  exp_args.insert({DNNL_ARG_DST, dst_mem});

  exp_prim.execute(stream, exp_args);

  // Wait for the completion of parallel loops scheduled into the runner.
  tsl::BlockUntilReady(runner.done_event());

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_NEAR(dst_data[i], std::exp(1.0f), 1e-5);
  }
}

}  // namespace
}  // namespace xla::cpu
