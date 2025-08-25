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

#include "xla/backends/cpu/runtime/onednn/onednn_thunk.h"

#include "gtest/gtest.h"
#include "xla/backends/cpu/runtime/custom_call_thunk.h"
#include "xla/backends/cpu/runtime/onednn/onednn_threadpool.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/ffi/execution_context.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {
using dnnl::engine;
using dnnl::stream;

// Simple handler: sum two float32 buffers element-wise, uses transpose attrs.
tsl::AsyncValueRef<OneDnnThunk::ExecuteEvent> DummyOneDnnHandler(
    ffi::AnyBuffer a, ffi::AnyBuffer b, ffi::Result<ffi::AnyBuffer> out,
    ffi::Dictionary attrs, dnnl::engine* cpu_engine,
    dnnl::stream* onednn_stream, OneDnnThreadPool* threadpool,
    OneDnnResources* resources) {
  // Get shape size (assuming 1D shape)
  int64_t N = a.dimensions()[0];
  CHECK_EQ(b.dimensions()[0], N);
  CHECK_EQ(out->dimensions()[0], N);

  // Cast void* data to float*
  float* a_data = static_cast<float*>(a.untyped_data());
  float* b_data = static_cast<float*>(b.untyped_data());
  float* out_data = static_cast<float*>(out->untyped_data());

  MemrefInfo a_minfo(static_cast<void*>(resources->arg_memrefs[0].get()));
  MemrefInfo b_minfo(static_cast<void*>(resources->arg_memrefs[1].get()));
  MemrefInfo out_minfo(static_cast<void*>(resources->result_memrefs[0].get()));

  auto a_md = a_minfo.GetOneDnnMemDesc();
  auto b_md = b_minfo.GetOneDnnMemDesc();
  auto out_md = out_minfo.GetOneDnnMemDesc();

  // Create memory objects
  resources->src_mem = dnnl::memory(a_md, *cpu_engine, a_data);
  resources->wei_mem = dnnl::memory(b_md, *cpu_engine, b_data);
  resources->dst_mem = dnnl::memory(out_md, *cpu_engine, out_data);

  // Create a simple element-wise addition primitive
  dnnl::binary::primitive_desc binary_pd(
      *cpu_engine, dnnl::algorithm::binary_add, resources->src_mem.get_desc(),
      resources->wei_mem.get_desc(), resources->dst_mem.get_desc());
  dnnl::binary binary_prim(binary_pd);
  resources->primitive = binary_prim;

  std::unordered_map<int, dnnl::memory> binary_args;
  binary_args.insert({DNNL_ARG_SRC_0, resources->src_mem});
  binary_args.insert({DNNL_ARG_SRC_1, resources->wei_mem});
  binary_args.insert({DNNL_ARG_DST, resources->dst_mem});

  resources->primitive.execute(*onednn_stream, binary_args);

  return threadpool->done_event();
}

// Register handler under name "onednn_add"
XLA_FFI_DEFINE_HANDLER(handler, DummyOneDnnHandler,
                       xla::ffi::Ffi::Bind()
                           .Arg<xla::ffi::AnyBuffer>()
                           .Arg<xla::ffi::AnyBuffer>()
                           .Ret<xla::ffi::AnyBuffer>()
                           .Attrs()
                           .Ctx<ffi::UserData<dnnl::engine>>()
                           .Ctx<ffi::UserData<dnnl::stream>>()
                           .Ctx<ffi::UserData<OneDnnThreadPool>>()
                           .Ctx<ffi::UserData<OneDnnResources>>());
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "onednn_add", "Host", handler);

TEST(OneDnnCustomCallThunkTest, SimpleOneDnnCustomCall) {
  // Set up a thread pool for parallel execution
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());

  // Define shapes for input/output buffers: e.g. vector of length 4 floats
  Shape shape = ShapeUtil::MakeShape(F32, {4});

  // Prepare dummy data for inputs and output (host side)
  std::vector<float> input_a = {1.f, 2.f, 3.f, 4.f};
  std::vector<float> input_b = {10.f, 20.f, 30.f, 40.f};
  std::vector<float> output(4, 0.f);

  // Create Literals from data
  Literal lhs_literal = LiteralUtil::CreateR1<float>(input_a);
  Literal rhs_literal = LiteralUtil::CreateR1<float>(input_b);
  Literal out_literal = LiteralUtil::CreateR1<float>(output);

  // Create buffer allocations and slices
  auto lhs_alloc = CreateBufferAllocation(0, lhs_literal);
  auto rhs_alloc = CreateBufferAllocation(1, rhs_literal);
  auto out_alloc = CreateBufferAllocation(2, out_literal);

  auto lhs_slice = CreateBufferAllocationSlice(lhs_alloc);
  auto rhs_slice = CreateBufferAllocationSlice(rhs_alloc);
  auto out_slice = CreateBufferAllocationSlice(out_alloc);

  BufferAllocations allocations =
      CreateBufferAllocations(lhs_literal, rhs_literal, out_literal);

  // Set up op_buffers
  CustomCallThunk::OpBuffers op_buffers;
  op_buffers.arguments_buffers = {lhs_slice, rhs_slice};
  op_buffers.arguments_shapes = {shape, shape};
  op_buffers.results_buffers = {out_slice};
  op_buffers.results_shapes = {shape};

  // Create thunk
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      OneDnnThunk::Create(Thunk::Info(), "onednn_add", op_buffers, {},
                          CustomCallApiVersion::API_VERSION_TYPED_FFI));

  // Set up execute params
  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  // Set up custom call parameters
  Thunk::CustomCallExecuteParams custom_call_params(
      xla::RunId{42}, 0, &device,
      nullptr);  // Example run ID and device ordinal
  params.custom_call_params = &custom_call_params;

  // Execute the thunk
  auto exec_event = thunk->Execute(params);
  tsl::BlockUntilReady(exec_event);
  ASSERT_FALSE(exec_event.IsError()) << "OneDnnThunk execution failed";

  // Load output and verify
  EXPECT_EQ(out_literal,
            LiteralUtil::CreateR1<float>({11.f, 22.f, 33.f, 44.f}));
}

}  // namespace
}  // namespace xla::cpu
