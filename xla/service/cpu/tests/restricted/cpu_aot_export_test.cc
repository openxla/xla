/* Copyright 2022 The OpenXLA Authors.

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/executable.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/restricted/hlo_test_base_legacy.h"
#include "xla/xla.pb.h"

namespace xla::cpu {

using ::testing::IsEmpty;

class CpuAotCompilationTest : public HloTestBaseLegacy {
 protected:
  void ExportAndLoad(absl::string_view hlo_string) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(hlo_string));

    auto compiler = backend().compiler();
    auto name = absl::AsciiStrToUpper(
        PlatformUtil::CanonicalPlatformName("host").value());
    ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                         se::PlatformManager::PlatformWithName(name));
    ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                         platform->ExecutorForDevice(0));

    // JIT compile executable
    ASSERT_OK_AND_ASSIGN(
        std::vector<std::unique_ptr<Executable>> executables,
        compiler->Compile(std::move(module), {stream_exec}, nullptr));

    ASSERT_OK_AND_ASSIGN(std::unique_ptr<CompiledModule> exported_aot_result,
                         compiler->Export(executables[0].get()));

    // Serialize-deserialize AOT compilation result.
    ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                         exported_aot_result->SerializeAsString());
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<CompiledModule> loaded_aot_result,
        compiler->LoadAotCompilationResult(serialized_aot_result));

    // Load Executable from AOT compilation result.
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                         std::move(*loaded_aot_result).LoadExecutable());
  }
};

TEST_F(CpuAotCompilationTest, ExportAndLoadExecutable) {
  const absl::string_view hlo_string = R"(
    HloModule Test

    ENTRY main {
      a = f32[2, 2]{1,0} parameter(0)
      ROOT b = f32[2, 2]{1,0} add(a, a)
    })";

  ExportAndLoad(hlo_string);
}

TEST_F(CpuAotCompilationTest, ExportAndLoadExecutableNoKernels) {
  // Copy operation implemented in the runtime and this module does not have
  // any jit compiled kernels. We test that we still can export and load such
  // executable.
  const absl::string_view hlo_string = R"(
    HloModule Test

    ENTRY main {
      a = f32[2, 2]{1,0} parameter(0)
      ROOT b = f32[2, 2]{1,0} copy(a)
    })";

  ExportAndLoad(hlo_string);
}

class CpuAotPicCompilationTest : public CpuAotCompilationTest {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        CpuAotCompilationTest::GetDebugOptionsForTest();
    debug_options.set_xla_cpu_aot_emit_shared_object(true);
    return debug_options;
  }
};

// Absolute relocations in executable sections prevent linking the object files
// into a shared object (they'd require text relocations or aren't supported).
static bool IsAbsoluteRelocation(const llvm::object::ELFObjectFileBase& obj,
                                 uint64_t type) {
  switch (obj.getEMachine()) {
    case llvm::ELF::EM_X86_64:
      return type == llvm::ELF::R_X86_64_64 || type == llvm::ELF::R_X86_64_32 ||
             type == llvm::ELF::R_X86_64_32S;
    case llvm::ELF::EM_AARCH64:
      return type == llvm::ELF::R_AARCH64_ABS64 ||
             type == llvm::ELF::R_AARCH64_MOVW_UABS_G0 ||
             type == llvm::ELF::R_AARCH64_MOVW_UABS_G0_NC ||
             type == llvm::ELF::R_AARCH64_MOVW_UABS_G1 ||
             type == llvm::ELF::R_AARCH64_MOVW_UABS_G1_NC ||
             type == llvm::ELF::R_AARCH64_MOVW_UABS_G2 ||
             type == llvm::ELF::R_AARCH64_MOVW_UABS_G2_NC ||
             type == llvm::ELF::R_AARCH64_MOVW_UABS_G3;
    default:
      return false;
  }
}

TEST_F(CpuAotPicCompilationTest, ExportPicObjectFiles) {
  // Exponential uses constants from the constant pool, which the large code
  // model addresses with absolute relocations.
  const absl::string_view hlo_string = R"(
    HloModule Test

    ENTRY main {
      a = f32[1024]{0} parameter(0)
      ROOT b = f32[1024]{0} exponential(a)
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_TRUE(
      module->config().debug_options().xla_cpu_aot_emit_shared_object());

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<Executable>> executables,
      backend().compiler()->Compile(std::move(module),
                                    {backend().default_stream_executor()},
                                    /*device_allocator=*/nullptr));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CompiledModule> exported,
                       backend().compiler()->Export(executables[0].get()));
  auto* aot_result = absl::down_cast<CpuAotCompilationResult*>(exported.get());
  ASSERT_NE(aot_result, nullptr);
  ASSERT_FALSE(aot_result->obj_files().empty());

  int num_relocations = 0;
  std::vector<std::string> absolute_relocations;
  for (absl::string_view obj_file : aot_result->obj_files()) {
    llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> obj =
        llvm::object::ObjectFile::createObjectFile(llvm::MemoryBufferRef(
            llvm::StringRef(obj_file.data(), obj_file.size()), "obj_file"));
    ASSERT_TRUE(static_cast<bool>(obj)) << llvm::toString(obj.takeError());
    auto* elf = llvm::dyn_cast<llvm::object::ELFObjectFileBase>(obj->get());
    ASSERT_NE(elf, nullptr);

    for (const llvm::object::SectionRef& section : elf->sections()) {
      llvm::Expected<llvm::object::section_iterator> relocated =
          section.getRelocatedSection();
      ASSERT_TRUE(static_cast<bool>(relocated))
          << llvm::toString(relocated.takeError());
      if (*relocated == elf->section_end() || !(*relocated)->isText()) {
        continue;
      }
      for (const llvm::object::RelocationRef& relocation :
           section.relocations()) {
        ++num_relocations;
        if (IsAbsoluteRelocation(*elf, relocation.getType())) {
          llvm::SmallString<32> type_name;
          relocation.getTypeName(type_name);
          absolute_relocations.push_back(type_name.str().str());
        }
      }
    }
  }
  EXPECT_GT(num_relocations, 0);
  EXPECT_THAT(absolute_relocations, IsEmpty());

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(CpuAotPicCompilationTest, JitCompilePicWithExternalCalls) {
  // Most of these ops call libm, which PIC code reaches through the PLT, so
  // ORC needs a stub and a GOT entry for each of them.
  const absl::string_view hlo_string = R"(
    HloModule Test

    ENTRY main {
      a = f32[64]{0} parameter(0)
      b = f32[64]{0} parameter(1)
      exp = f32[64]{0} exponential(a)
      log = f32[64]{0} log(b)
      pow = f32[64]{0} power(b, a)
      atan2 = f32[64]{0} atan2(a, b)
      tanh = f32[64]{0} tanh(a)
      sin = f32[64]{0} sine(a)
      cos = f32[64]{0} cosine(b)
      rem = f32[64]{0} remainder(b, a)
      cbrt = f32[64]{0} cbrt(b)
      sum0 = f32[64]{0} add(exp, log)
      sum1 = f32[64]{0} add(sum0, pow)
      sum2 = f32[64]{0} add(sum1, atan2)
      sum3 = f32[64]{0} add(sum2, tanh)
      sum4 = f32[64]{0} add(sum3, sin)
      sum5 = f32[64]{0} add(sum4, cos)
      sum6 = f32[64]{0} add(sum5, rem)
      ROOT sum7 = f32[64]{0} add(sum6, cbrt)
    })";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-4, 1e-4}));
  ExportAndLoad(hlo_string);
}

}  // namespace xla::cpu
