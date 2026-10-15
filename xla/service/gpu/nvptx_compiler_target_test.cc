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

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/cuda/compilation_provider.h"
#include "xla/stream_executor/cuda/compilation_provider_options.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/mock_compilation_provider.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"

namespace xla::gpu {

class NVPTXCompilerTestPeer {
 public:
  static void SetCompilationProvider(
      NVPTXCompiler& compiler, const DebugOptions& debug_options,
      std::unique_ptr<se::cuda::CompilationProvider> provider) {
    absl::MutexLock lock(compiler.compilation_providers_mutex_);
    compiler.compilation_providers_[se::cuda::CompilationProviderOptions::
                                        FromDebugOptions(debug_options)] =
        std::move(provider);
  }

  static absl::StatusOr<std::vector<uint8_t>> LinkModules(
      NVPTXCompiler& compiler, const se::DeviceDescription& device_description,
      const DebugOptions& debug_options) {
    return compiler.LinkModules(device_description, {{0}}, debug_options,
                                nullptr);
  }
};

namespace {

using ::testing::_;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;

class NVPTXCompilerTargetTest
    : public ::testing::TestWithParam<std::tuple<int, bool>> {
 protected:
  void SetUp() override {
    module_config_.mutable_debug_options().set_xla_gpu_cuda_data_dir(
        tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "service/gpu",
                          "nvptx_compiler_target_test_cuda"));
    device_description_.set_cuda_compute_capability(
        {10, 7,
         se::CudaComputeCapability::FeatureExtension::kAcceleratedFeatures});
    auto provider =
        std::make_unique<NiceMock<se::cuda::MockCompilationProvider>>();
    provider_ = provider.get();
    ON_CALL(*provider_, GetLatestPtxIsaVersion())
        .WillByDefault(Return(std::get<0>(GetParam())));
    NVPTXCompilerTestPeer::SetCompilationProvider(
        compiler_, module_config_.debug_options(), std::move(provider));
  }

  NVPTXCompiler compiler_;
  HloModuleConfig module_config_;
  se::DeviceDescription device_description_;
  se::cuda::MockCompilationProvider* provider_;
};

TEST_P(NVPTXCompilerTargetTest, CompilationAndLinkingUseSameTarget) {
  using FeatureExtension = se::CudaComputeCapability::FeatureExtension;
  se::CudaComputeCapability expected_cc =
      std::get<0>(GetParam()) == 94
          ? se::CudaComputeCapability{10, 7,
                                      FeatureExtension::kAcceleratedFeatures}
          : se::CudaComputeCapability{
                10, 3, FeatureExtension::kFamilyCompatibleFeatures};
  auto expected_ptx = HasSubstr(absl::StrCat(
      ".target ", expected_cc.GetPtxAsTargetName(
                      se::CudaComputeCapability::CompileMode::kSass)));

  // Linking must also work before the LLVM backend has compiled any PTX.
  EXPECT_CALL(*provider_, CompileAndLink(expected_cc, _, _))
      .WillOnce(Return(se::cuda::Assembly{}));
  EXPECT_OK(NVPTXCompilerTestPeer::LinkModules(compiler_, device_description_,
                                               module_config_.debug_options()));

  llvm::LLVMContext context;
  llvm::Module module("test", context);
  new llvm::GlobalVariable(
      module, llvm::Type::getInt32Ty(context), true,
      llvm::GlobalValue::ExternalLinkage,
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 0), "gv");
  bool relocatable = std::get<1>(GetParam());
  if (relocatable) {
    EXPECT_CALL(*provider_,
                CompileToRelocatableModule(expected_cc, expected_ptx, _))
        .WillOnce(Return(se::cuda::RelocatableModule{}));
  } else {
    EXPECT_CALL(*provider_, Compile(expected_cc, expected_ptx, _))
        .WillOnce(Return(se::cuda::Assembly{}));
  }
  EXPECT_OK(compiler_.CompileTargetBinary(module_config_, &module,
                                          device_description_, relocatable,
                                          nullptr, std::nullopt));
}

INSTANTIATE_TEST_SUITE_P(ProviderPtxVersion, NVPTXCompilerTargetTest,
                         ::testing::Combine(::testing::Values(90, 94),
                                            ::testing::Bool()));

}  // namespace
}  // namespace xla::gpu
