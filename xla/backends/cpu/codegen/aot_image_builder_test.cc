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

#include "xla/backends/cpu/codegen/aot_image_builder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/backends/cpu/codegen/builtin_symbols.h"
#include "xla/backends/cpu/codegen/pre_relocated_image_loader.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/service/cpu/executable.pb.h"

namespace xla::cpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::Contains;
using ::testing::Property;

constexpr absl::string_view kTestIr = R"(
  @table = internal constant [4 x float] [float 1.0, float 2.0, float 3.0, float 4.0]
  @ptr_table = internal constant [2 x ptr] [ptr @table, ptr @helper]
  @counter = internal global i64 0
  @extern_global = external global i32

  declare float @expf(float)
  declare float @__powisf2(float, i32)
  declare float @custom_fn(float)

  define internal float @helper(float %x) {
    %y = fmul float %x, 2.0
    ret float %y
  }

  define float @lookup(i64 %i) {
    %p = getelementptr [4 x float], ptr @table, i64 0, i64 %i
    %v = load float, ptr %p
    %h = call float @helper(float %v)
    ret float %h
  }

  define float @call_external(float %x) {
    %e = call float @expf(float %x)
    %p = call float @__powisf2(float %e, i32 2)
    %c = call float @custom_fn(float %p)
    ret float %c
  }

  define ptr @get_ptr(i64 %i) {
    %p = getelementptr [2 x ptr], ptr @ptr_table, i64 0, i64 %i
    %v = load ptr, ptr %p
    ret ptr %v
  }

  define i64 @bump() {
    %v = load i64, ptr @counter
    %n = add i64 %v, 1
    store i64 %n, ptr @counter
    ret i64 %n
  }

  define i32 @read_extern() {
    %v = load i32, ptr @extern_global
    ret i32 %v
  }
)";

const std::vector<std::string>& ExportedSymbols() {
  static const auto* symbols = new std::vector<std::string>{
      "lookup", "call_external", "get_ptr", "bump", "read_extern"};
  return *symbols;
}

int32_t extern_global = 42;

float custom_fn(float x) { return x + 1.0f; }

void* TestResolver(absl::string_view name) {
  if (name == "custom_fn") return reinterpret_cast<void*>(custom_fn);
  if (name == "extern_global") return &extern_global;
  return LookupBuiltinSymbol(name);
}

void InitializeTargets() {
  static absl::once_flag once;
  absl::call_once(once, [] {
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeAArch64Target();
    LLVMInitializeAArch64TargetInfo();
    LLVMInitializeAArch64TargetMC();
    LLVMInitializeAArch64AsmPrinter();
  });
}

absl::StatusOr<std::string> CompileToObjectFile(
    absl::string_view triple_str, llvm::Reloc::Model reloc_model,
    std::optional<llvm::CodeModel::Model> code_model) {
  InitializeTargets();

  llvm::Triple triple((std::string(triple_str)));
  std::string error;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (target == nullptr) {
    return absl::InternalError(error);
  }
  std::unique_ptr<llvm::TargetMachine> target_machine(
      target->createTargetMachine(triple, "generic", "", llvm::TargetOptions(),
                                  reloc_model, code_model,
                                  llvm::CodeGenOptLevel::Default));

  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module = llvm::parseAssemblyString(
      llvm::StringRef(kTestIr.data(), kTestIr.size()), diagnostic, context);
  if (module == nullptr) {
    return absl::InternalError(diagnostic.getMessage().str());
  }
  module->setTargetTriple(triple);
  module->setDataLayout(target_machine->createDataLayout());

  llvm::SmallVector<char, 0> buffer;
  llvm::raw_svector_ostream stream(buffer);
  llvm::legacy::PassManager pm;
  if (target_machine->addPassesToEmitFile(pm, stream, nullptr,
                                          llvm::CodeGenFileType::ObjectFile)) {
    return absl::InternalError("Target can't emit object files");
  }
  pm.run(*module);
  return std::string(buffer.begin(), buffer.end());
}

absl::StatusOr<PreRelocatedImageProto> BuildImage(
    absl::string_view triple, llvm::Reloc::Model reloc_model,
    std::optional<llvm::CodeModel::Model> code_model = std::nullopt) {
  absl::StatusOr<std::string> obj =
      CompileToObjectFile(triple, reloc_model, code_model);
  if (!obj.ok()) {
    return obj.status();
  }
  absl::string_view obj_files[] = {*obj};
  return BuildPreRelocatedImage(obj_files, ExportedSymbols());
}

void ExpectWellFormed(const PreRelocatedImageProto& image) {
  EXPECT_EQ(image.rx_size() % kPreRelocatedImageSegmentAlignment, 0);
  EXPECT_EQ(image.ro_size() % kPreRelocatedImageSegmentAlignment, 0);
  EXPECT_LE(image.rx_bytes().size(), image.rx_size());
  EXPECT_LE(image.ro_bytes().size(), image.ro_size());
  EXPECT_LE(image.rw_bytes().size(), image.rw_mem_size());
  for (const std::string& name : ExportedSymbols()) {
    EXPECT_THAT(image.exported_symbols(),
                Contains(Property(&PreRelocatedImageProto::ExportedSymbol::name,
                                  name)));
  }
  for (absl::string_view name :
       {"expf", "__powisf2", "custom_fn", "extern_global"}) {
    EXPECT_THAT(
        image.external_fixups(),
        Contains(Property(&PreRelocatedImageProto::ExternalFixup::symbol_name,
                          std::string(name))));
  }
}

// Loads the image in the current process and runs every exported function.
void ExpectExecutesCorrectly(const PreRelocatedImageProto& image) {
  auto library = LoadPreRelocatedImage(image, TestResolver);
  ASSERT_THAT(library, IsOk());

  auto lookup = (*library)->ResolveFunction<float(int64_t)>("lookup");
  auto call_external =
      (*library)->ResolveFunction<float(float)>("call_external");
  auto get_ptr = (*library)->ResolveFunction<void*(int64_t)>("get_ptr");
  auto bump = (*library)->ResolveFunction<int64_t()>("bump");
  auto read_extern = (*library)->ResolveFunction<int32_t()>("read_extern");
  ASSERT_THAT(lookup, IsOk());
  ASSERT_THAT(call_external, IsOk());
  ASSERT_THAT(get_ptr, IsOk());
  ASSERT_THAT(bump, IsOk());
  ASSERT_THAT(read_extern, IsOk());

  EXPECT_EQ((*lookup)(2), 6.0f);
  EXPECT_FLOAT_EQ((*call_external)(1.0f),
                  std::exp(1.0f) * std::exp(1.0f) + 1.0f);

  auto* table = static_cast<float*>((*get_ptr)(0));
  EXPECT_EQ(table[3], 4.0f);
  auto* helper = reinterpret_cast<float (*)(float)>((*get_ptr)(1));
  EXPECT_EQ(helper(3.0f), 6.0f);

  EXPECT_EQ((*bump)(), 1);
  EXPECT_EQ((*bump)(), 2);
  EXPECT_EQ((*read_extern)(), 42);
}

std::optional<std::string> HostTriple() {
  llvm::Triple host(llvm::sys::getProcessTriple());
  if (host.getArch() == llvm::Triple::x86_64 && host.isOSLinux()) {
    return "x86_64-unknown-linux-gnu";
  }
  if (host.getArch() == llvm::Triple::aarch64 && host.isOSLinux()) {
    return "aarch64-unknown-linux-gnu";
  }
  return std::nullopt;
}

class AotImageBuilderTest : public ::testing::TestWithParam<absl::string_view> {
};

TEST_P(AotImageBuilderTest, BuildsPicImage) {
  auto image = BuildImage(GetParam(), llvm::Reloc::PIC_);
  ASSERT_THAT(image, IsOk());
  ExpectWellFormed(*image);

  if (HostTriple() == GetParam()) {
    ExpectExecutesCorrectly(*image);
  }
}

INSTANTIATE_TEST_SUITE_P(Targets, AotImageBuilderTest,
                         ::testing::Values("x86_64-unknown-linux-gnu",
                                           "aarch64-unknown-linux-gnu"));

// The JIT compiles x86-64 kernels with the large code model, which stores
// absolute addresses in `movabs` immediates instead of using PC-relative code.
TEST(AotImageBuilderX86Test, BuildsLargeCodeModelImage) {
  auto image = BuildImage("x86_64-unknown-linux-gnu", llvm::Reloc::Static,
                          llvm::CodeModel::Large);
  ASSERT_THAT(image, IsOk());
  ExpectWellFormed(*image);

  if (HostTriple() == "x86_64-unknown-linux-gnu") {
    ExpectExecutesCorrectly(*image);
  }
}

// AArch64 large code model materializes addresses with MOVZ/MOVK sequences,
// which can't be patched with 64-bit slots.
TEST(AotImageBuilderAArch64Test, RejectsLargeCodeModel) {
  EXPECT_THAT(BuildImage("aarch64-unknown-linux-gnu", llvm::Reloc::Static,
                         llvm::CodeModel::Large),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(AotImageBuilderTest, FailsOnMissingExportedSymbol) {
  auto obj = CompileToObjectFile("x86_64-unknown-linux-gnu", llvm::Reloc::PIC_,
                                 std::nullopt);
  ASSERT_THAT(obj, IsOk());
  absl::string_view obj_files[] = {*obj};
  std::vector<std::string> symbols = {"does_not_exist"};
  EXPECT_THAT(BuildPreRelocatedImage(obj_files, symbols),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(AotImageBuilderTest, FailsOnInvalidObjectFile) {
  absl::string_view obj_files[] = {"not an object file"};
  EXPECT_THAT(BuildPreRelocatedImage(obj_files, ExportedSymbols()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla::cpu
