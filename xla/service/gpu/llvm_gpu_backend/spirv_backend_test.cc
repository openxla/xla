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

#include "xla/service/gpu/llvm_gpu_backend/spirv_backend.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.h"
#include "xla/xla.pb.h"

namespace xla::gpu::spirv {
namespace {

constexpr uint32_t kSpirvMagic = 0x07230203;
constexpr uint16_t kOpTypePointer = 32;
constexpr uint32_t kStorageClassUniformConstant = 0;
constexpr uint32_t kStorageClassWorkgroup = 4;
constexpr uint32_t kStorageClassCrossWorkgroup = 5;

std::vector<uint32_t> DecodeSpirvWords(std::string_view binary) {
  std::vector<uint32_t> words(binary.size() / sizeof(uint32_t));
  std::memcpy(words.data(), binary.data(), binary.size());
  return words;
}

bool ContainsOpTypePointerWithStorageClass(const std::vector<uint32_t>& words,
                                           uint32_t storage_class) {
  constexpr int kHeaderWordCount = 5;
  for (size_t i = kHeaderWordCount; i < words.size();) {
    uint16_t instruction_opcode = words[i] & 0xffff;
    uint16_t instruction_word_count = words[i] >> 16;
    if (instruction_opcode == kOpTypePointer && instruction_word_count >= 3 &&
        words[i + 2] == storage_class) {
      return true;
    }
    if (instruction_word_count == 0) {
      return false;
    }
    i += instruction_word_count;
  }
  return false;
}

stream_executor::GpuComputeCapability TestComputeCapability() {
  return stream_executor::GpuComputeCapability(
      stream_executor::OneAPIComputeCapability::BMG());
}

std::unique_ptr<llvm::Module> ParseLlvmIr(std::string_view ir,
                                          llvm::LLVMContext& context) {
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseAssemblyString(ir, diagnostic, context);
  if (module == nullptr) {
    ADD_FAILURE() << "Failed to parse LLVM IR: "
                  << diagnostic.getMessage().str();
  }
  return module;
}

std::vector<uint32_t> CompileAndDecode(std::string_view ir) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ParseLlvmIr(ir, context);
  if (module == nullptr) {
    return {};
  }

  absl::StatusOr<std::string> spirv =
      CompileToSPIRV(module.get(), TestComputeCapability(), DebugOptions());
  EXPECT_TRUE(spirv.ok()) << spirv.status();
  if (!spirv.ok()) {
    return {};
  }
  EXPECT_EQ(spirv->size() % sizeof(uint32_t), 0);
  return DecodeSpirvWords(*spirv);
}

TEST(SpirvBackendTest, TestSPIRVExtensions) {
  auto extensions = SPIRVExtensionsEnumToString(common_spirv_extensions);
  auto extensions_set =
      std::set<std::string>(extensions.begin(), extensions.end());

  EXPECT_NE(extensions_set.find("SPV_EXT_optnone"), extensions_set.end());
  EXPECT_NE(extensions_set.find("SPV_KHR_uniform_group_instructions"),
            extensions_set.end());
  EXPECT_NE(extensions_set.find("SPV_KHR_linkonce_odr"), extensions_set.end());
  EXPECT_NE(extensions_set.find("SPV_KHR_cooperative_matrix"),
            extensions_set.end());
  EXPECT_NE(extensions_set.find("SPV_EXT_shader_atomic_float_add"),
            extensions_set.end());
  EXPECT_EQ(extensions_set.find("SPV_NV_cooperative_matrix"),
            extensions_set.end());
}

TEST(SpirvBackendTest, AddressSpaceKernelScalarArgumentsArePreserved) {
  std::vector<uint32_t> words = CompileAndDecode(R"(
define spir_kernel void @scalar_kernel_arg(i32 %value, ptr addrspace(1) %out) {
entry:
  store i32 %value, ptr addrspace(1) %out, align 4
  ret void
}
)");
  ASSERT_GE(words.size(), 5);
  EXPECT_EQ(words[0], kSpirvMagic);
  EXPECT_TRUE(
      ContainsOpTypePointerWithStorageClass(words, kStorageClassCrossWorkgroup));
}

TEST(SpirvBackendTest, AddressSpaceScalarArgumentsSurvivePointerRewrite) {
  std::vector<uint32_t> words = CompileAndDecode(R"(
define spir_kernel void @default_pointer_kernel(
    i32 %value, ptr %in, ptr addrspace(1) %out) {
entry:
  %loaded = load i32, ptr %in, align 4
  %sum = add i32 %loaded, %value
  store i32 %sum, ptr addrspace(1) %out, align 4
  ret void
}
)");
  ASSERT_GE(words.size(), 5);
  EXPECT_EQ(words[0], kSpirvMagic);
  EXPECT_TRUE(
      ContainsOpTypePointerWithStorageClass(words, kStorageClassCrossWorkgroup));
}

TEST(SpirvBackendTest, AddressSpaceMappingsMatchSpirvContract) {
  std::vector<uint32_t> words = CompileAndDecode(R"(
define spir_kernel void @address_space_map(ptr addrspace(1) %out,
                                           ptr addrspace(1) %global,
                                           ptr addrspace(2) %constant,
                                           ptr addrspace(3) %workgroup) {
entry:
  %global_value = load i32, ptr addrspace(1) %global, align 4
  %constant_value = load i32, ptr addrspace(2) %constant, align 4
  %workgroup_value = load i32, ptr addrspace(3) %workgroup, align 4
  %sum0 = add i32 %global_value, %constant_value
  %sum1 = add i32 %sum0, %workgroup_value
  store i32 %sum1, ptr addrspace(1) %out, align 4
  ret void
}
)");
  ASSERT_GE(words.size(), 5);
  EXPECT_EQ(words[0], kSpirvMagic);
  EXPECT_TRUE(
      ContainsOpTypePointerWithStorageClass(words, kStorageClassCrossWorkgroup));
  EXPECT_TRUE(
      ContainsOpTypePointerWithStorageClass(words, kStorageClassUniformConstant));
  EXPECT_TRUE(
      ContainsOpTypePointerWithStorageClass(words, kStorageClassWorkgroup));
}

}  // namespace
}  // namespace xla::gpu::spirv
