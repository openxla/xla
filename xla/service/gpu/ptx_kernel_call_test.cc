/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/ptx_kernel_call.h"

#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"

namespace xla::gpu {
namespace {

class PtxKernelCallTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mlir_context_ = std::make_unique<mlir::MLIRContext>();
  }

  std::unique_ptr<mlir::MLIRContext> mlir_context_;
};

TEST_F(PtxKernelCallTest, ParseBasicConfiguration) {
  const char* backend_config = R"({
    name = "test_kernel",
    source = ".version 7.0\n.target sm_70\n.entry test_kernel() { ret; }",
    grid_x = 1,
    grid_y = 2,
    grid_z = 3,
    block_x = 4,
    block_y = 5,
    block_z = 6,
    shared_mem_bytes = 1024
  })";

  PtxCall ptx_call = PtxCall::Parse(backend_config, mlir_context_.get());

  EXPECT_EQ(ptx_call.name, "test_kernel");
  EXPECT_EQ(ptx_call.source,
            ".version 7.0\n.target sm_70\n.entry test_kernel() { ret; }");
  EXPECT_EQ(ptx_call.block_dim.x, 1);
  EXPECT_EQ(ptx_call.block_dim.y, 2);
  EXPECT_EQ(ptx_call.block_dim.z, 3);
  EXPECT_EQ(ptx_call.thread_dim.x, 4);
  EXPECT_EQ(ptx_call.thread_dim.y, 5);
  EXPECT_EQ(ptx_call.thread_dim.z, 6);
  EXPECT_EQ(ptx_call.shared_mem, 1024);
  EXPECT_TRUE(ptx_call.output_indices.empty());
}

TEST_F(PtxKernelCallTest, ParseWithOutputIndices) {
  const char* backend_config = R"({
    name = "kernel_with_outputs",
    source = ".version 7.0\n.target sm_70\n.entry kernel_with_outputs() { ret; }",
    grid_x = 10,
    grid_y = 20,
    grid_z = 1,
    block_x = 32,
    block_y = 1,
    block_z = 1,
    shared_mem_bytes = 0,
    output_indices = [1, 3, 5]
  })";

  PtxCall ptx_call = PtxCall::Parse(backend_config, mlir_context_.get());

  EXPECT_EQ(ptx_call.name, "kernel_with_outputs");
  EXPECT_EQ(ptx_call.block_dim.x, 10);
  EXPECT_EQ(ptx_call.block_dim.y, 20);
  EXPECT_EQ(ptx_call.block_dim.z, 1);
  EXPECT_EQ(ptx_call.thread_dim.x, 32);
  EXPECT_EQ(ptx_call.thread_dim.y, 1);
  EXPECT_EQ(ptx_call.thread_dim.z, 1);
  EXPECT_EQ(ptx_call.shared_mem, 0);

  ASSERT_EQ(ptx_call.output_indices.size(), 3);
  EXPECT_EQ(ptx_call.output_indices[0], 1);
  EXPECT_EQ(ptx_call.output_indices[1], 3);
  EXPECT_EQ(ptx_call.output_indices[2], 5);
}

TEST_F(PtxKernelCallTest, ParseMinimalConfiguration) {
  const char* backend_config = R"({
    name = "minimal_kernel",
    source = ".entry minimal_kernel() { ret; }",
    grid_x = 1,
    grid_y = 1,
    grid_z = 1,
    block_x = 1,
    block_y = 1,
    block_z = 1,
    shared_mem_bytes = 0
  })";

  PtxCall ptx_call = PtxCall::Parse(backend_config, mlir_context_.get());

  EXPECT_EQ(ptx_call.name, "minimal_kernel");
  EXPECT_EQ(ptx_call.source, ".entry minimal_kernel() { ret; }");
  EXPECT_EQ(ptx_call.block_dim.x, 1);
  EXPECT_EQ(ptx_call.block_dim.y, 1);
  EXPECT_EQ(ptx_call.block_dim.z, 1);
  EXPECT_EQ(ptx_call.thread_dim.x, 1);
  EXPECT_EQ(ptx_call.thread_dim.y, 1);
  EXPECT_EQ(ptx_call.thread_dim.z, 1);
  EXPECT_EQ(ptx_call.shared_mem, 0);
  EXPECT_TRUE(ptx_call.output_indices.empty());
}

TEST_F(PtxKernelCallTest, ParseLargeDimensions) {
  const char* backend_config = R"({
    name = "large_kernel",
    source = ".entry large_kernel() { ret; }",
    grid_x = 65535,
    grid_y = 65535,
    grid_z = 65535,
    block_x = 1024,
    block_y = 1024,
    block_z = 64,
    shared_mem_bytes = 49152
  })";

  PtxCall ptx_call = PtxCall::Parse(backend_config, mlir_context_.get());

  EXPECT_EQ(ptx_call.name, "large_kernel");
  EXPECT_EQ(ptx_call.block_dim.x, 65535);
  EXPECT_EQ(ptx_call.block_dim.y, 65535);
  EXPECT_EQ(ptx_call.block_dim.z, 65535);
  EXPECT_EQ(ptx_call.thread_dim.x, 1024);
  EXPECT_EQ(ptx_call.thread_dim.y, 1024);
  EXPECT_EQ(ptx_call.thread_dim.z, 64);
  EXPECT_EQ(ptx_call.shared_mem, 49152);
}

TEST_F(PtxKernelCallTest, ParseEmptyOutputIndices) {
  const char* backend_config = R"({
    name = "no_outputs",
    source = ".entry no_outputs() { ret; }",
    grid_x = 1,
    grid_y = 1,
    grid_z = 1,
    block_x = 32,
    block_y = 1,
    block_z = 1,
    shared_mem_bytes = 512,
    output_indices = []
  })";

  PtxCall ptx_call = PtxCall::Parse(backend_config, mlir_context_.get());

  EXPECT_EQ(ptx_call.name, "no_outputs");
  EXPECT_EQ(ptx_call.shared_mem, 512);
  EXPECT_TRUE(ptx_call.output_indices.empty());
}

TEST_F(PtxKernelCallTest, ParseSingleOutputIndex) {
  const char* backend_config = R"({
    name = "single_output",
    source = ".entry single_output() { ret; }",
    grid_x = 2,
    grid_y = 1,
    grid_z = 1,
    block_x = 64,
    block_y = 1,
    block_z = 1,
    shared_mem_bytes = 256,
    output_indices = [0]
  })";

  PtxCall ptx_call = PtxCall::Parse(backend_config, mlir_context_.get());

  EXPECT_EQ(ptx_call.name, "single_output");
  EXPECT_EQ(ptx_call.shared_mem, 256);
  ASSERT_EQ(ptx_call.output_indices.size(), 1);
  EXPECT_EQ(ptx_call.output_indices[0], 0);
}

TEST_F(PtxKernelCallTest, ParseComplexPtxSource) {
  const char* backend_config = R"({
    name = "complex_kernel",
    source = ".version 7.5\n.target sm_80\n.address_size 64\n\n.entry complex_kernel(.param .u64 ptr) {\n  .reg .u64 %r1;\n  ld.param.u64 %r1, [ptr];\n  ret;\n}",
    grid_x = 100,
    grid_y = 50,
    grid_z = 1,
    block_x = 256,
    block_y = 1,
    block_z = 1,
    shared_mem_bytes = 8192,
    output_indices = [2, 4, 6, 8]
  })";

  PtxCall ptx_call = PtxCall::Parse(backend_config, mlir_context_.get());

  EXPECT_EQ(ptx_call.name, "complex_kernel");
  EXPECT_TRUE(ptx_call.source.find(".version 7.5") != std::string::npos);
  EXPECT_TRUE(ptx_call.source.find(".target sm_80") != std::string::npos);
  EXPECT_TRUE(ptx_call.source.find("complex_kernel") != std::string::npos);
  EXPECT_EQ(ptx_call.block_dim.x, 100);
  EXPECT_EQ(ptx_call.block_dim.y, 50);
  EXPECT_EQ(ptx_call.thread_dim.x, 256);
  EXPECT_EQ(ptx_call.shared_mem, 8192);

  ASSERT_EQ(ptx_call.output_indices.size(), 4);
  EXPECT_EQ(ptx_call.output_indices[0], 2);
  EXPECT_EQ(ptx_call.output_indices[1], 4);
  EXPECT_EQ(ptx_call.output_indices[2], 6);
  EXPECT_EQ(ptx_call.output_indices[3], 8);
}

}  // namespace
}  // namespace xla::gpu