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

#include "xla/backends/cpu/codegen/wasm_object_linker.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/backends/cpu/codegen/wasm_function_library.h"

#ifdef __EMSCRIPTEN__
#include "xla/backends/cpu/runtime/function_library.h"
#endif

namespace xla::cpu {
namespace {

constexpr const char* kLookupIr = R"(
    @constants = constant [4 x i32] [i32 11, i32 23, i32 37, i32 41]
    define i32 @lookup(i32 %index) {
      %p = getelementptr [4 x i32], ptr @constants, i32 0, i32 %index
      %value = load i32, ptr %p
      ret i32 %value
    }
  )";

absl::StatusOr<std::unique_ptr<llvm::MemoryBuffer>> CreateObject(
    const char* triple, llvm::Reloc::Model relocation_model,
    llvm::StringRef ir = kLookupIr) {
  LLVMInitializeWebAssemblyTargetInfo();
  LLVMInitializeWebAssemblyTarget();
  LLVMInitializeWebAssemblyTargetMC();
  LLVMInitializeWebAssemblyAsmPrinter();
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseAssemblyString(ir, diagnostic, context);
  if (!module) return absl::InternalError("Failed to parse test IR");
  std::string error;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(llvm::Triple(triple), error);
  if (!target) return absl::InternalError(error);
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      llvm::Triple(triple), "generic", "", llvm::TargetOptions(),
      relocation_model));
  module->setTargetTriple(llvm::Triple(triple));
  module->setDataLayout(machine->createDataLayout());
  if (relocation_model == llvm::Reloc::PIC_) {
    module->setPICLevel(llvm::PICLevel::BigPIC);
  }
  llvm::SmallVector<char, 0> bytes;
  llvm::raw_svector_ostream stream(bytes);
  llvm::legacy::PassManager passes;
  llvm::MCContext* mc_context;
  machine->addPassesToEmitMC(passes, mc_context, stream);
  passes.run(*module);
  return llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(bytes.data(), bytes.size()), "lookup");
}

TEST(WasmObjectLinkerTest, RejectsMalformedObject) {
  auto module = LinkWasmObject(llvm::MemoryBufferRef("invalid", "invalid"));
  EXPECT_EQ(module.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(WasmObjectLinkerTest, LinksBothWidthsAndRecoversAfterLinkErrors) {
  for (const char* triple : {"wasm32-unknown-unknown",
                             "wasm64-unknown-unknown"}) {
    SCOPED_TRACE(triple);
    ASSERT_OK_AND_ASSIGN(auto non_pic,
                         CreateObject(triple, llvm::Reloc::Static));
    auto failed = LinkWasmObject(non_pic->getMemBufferRef());
    ASSERT_FALSE(failed.ok());
    EXPECT_THAT(failed.status().message(), testing::HasSubstr("-fPIC"));

    ASSERT_OK_AND_ASSIGN(auto pic, CreateObject(triple, llvm::Reloc::PIC_));
    for (int i = 0; i < 2; ++i) {
      ASSERT_OK_AND_ASSIGN(auto module, LinkWasmObject(pic->getMemBufferRef()));
      auto object = llvm::object::ObjectFile::createObjectFile(*module);
      ASSERT_TRUE(static_cast<bool>(object))
          << llvm::toString(object.takeError());
      EXPECT_TRUE((*object)->isWasm());
      EXPECT_FALSE((*object)->isRelocatableObject());
      EXPECT_EQ((*object)->getBytesInAddress(),
                llvm::Triple(triple).isArch64Bit() ? 8 : 4);
      auto* wasm = llvm::cast<llvm::object::WasmObjectFile>(object->get());
      auto exports = wasm->exports();
      EXPECT_TRUE(std::any_of(exports.begin(), exports.end(),
                              [](const auto& symbol) {
                                return symbol.Name == "lookup";
                              }));
      // Definitions must not be imported through the process-wide GOT, where
      // a different compiled module may already have the same symbol name.
      EXPECT_FALSE(std::any_of(wasm->imports().begin(), wasm->imports().end(),
                               [](const auto& symbol) {
                                 return symbol.Field == "constants";
                               }));
      EXPECT_FALSE(llvm::sys::fs::exists(module->getBufferIdentifier()));
    }
  }
}

#ifdef __EMSCRIPTEN__

#ifdef __wasm64__
constexpr const char* kHostTriple = "wasm64-unknown-emscripten";
#else
constexpr const char* kHostTriple = "wasm32-unknown-emscripten";
#endif

TEST(WasmObjectLinkerTest, LoadsAndExecutesIndependentModules) {
  const FunctionLibrary::Symbol symbols[] = {
      FunctionLibrary::Sym<int(int)>("lookup")};
  ASSERT_OK_AND_ASSIGN(auto first_object,
                       CreateObject(kHostTriple, llvm::Reloc::PIC_));
  ASSERT_OK_AND_ASSIGN(
      auto first,
      LoadWasmFunctionLibrary(symbols, first_object->getMemBufferRef()));
  ASSERT_OK_AND_ASSIGN(auto lookup_first,
                       first->ResolveFunction<int(int)>("lookup"));
  EXPECT_EQ(lookup_first(0), 11);
  EXPECT_EQ(lookup_first(1), 23);

  ASSERT_OK_AND_ASSIGN(auto second_object,
                       CreateObject(kHostTriple, llvm::Reloc::PIC_, R"(
    @constants = constant i32 59
    define i32 @lookup(i32 %index) {
      %value = load i32, ptr @constants
      ret i32 %value
    }
  )"));
  ASSERT_OK_AND_ASSIGN(
      auto second,
      LoadWasmFunctionLibrary(symbols, second_object->getMemBufferRef()));
  ASSERT_OK_AND_ASSIGN(auto lookup_second,
                       second->ResolveFunction<int(int)>("lookup"));
  EXPECT_EQ(lookup_second(0), 59);
  EXPECT_EQ(lookup_first(1), 23);
  first.reset();
  EXPECT_EQ(lookup_second(0), 59);
}

TEST(WasmObjectLinkerTest, RecoversAfterMissingFunction) {
  ASSERT_OK_AND_ASSIGN(auto object,
                       CreateObject(kHostTriple, llvm::Reloc::PIC_));
  const FunctionLibrary::Symbol missing[] = {
      FunctionLibrary::Sym<int(int)>("missing")};
  auto failed = LoadWasmFunctionLibrary(missing, object->getMemBufferRef());
  EXPECT_EQ(failed.status().code(), absl::StatusCode::kNotFound);
  const FunctionLibrary::Symbol symbols[] = {
      FunctionLibrary::Sym<int(int)>("lookup")};
  ASSERT_OK_AND_ASSIGN(
      auto library, LoadWasmFunctionLibrary(symbols, object->getMemBufferRef()));
  ASSERT_OK_AND_ASSIGN(auto lookup,
                       library->ResolveFunction<int(int)>("lookup"));
  EXPECT_EQ(lookup(2), 37);
  EXPECT_EQ(library->ResolveFunction<int(int)>("missing").status().code(),
            absl::StatusCode::kNotFound);
}

#endif  // __EMSCRIPTEN__

}  // namespace
}  // namespace xla::cpu
