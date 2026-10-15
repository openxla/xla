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

#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "lld/Common/Driver.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

LLD_HAS_DRIVER(wasm)

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<llvm::MemoryBuffer>> LinkWasmObject(
    llvm::MemoryBufferRef object) {
  auto parsed = llvm::object::ObjectFile::createObjectFile(object);
  if (!parsed) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid WebAssembly object: ", llvm::toString(parsed.takeError())));
  }
  if (!(*parsed)->isWasm() || !(*parsed)->isRelocatableObject()) {
    return absl::InvalidArgumentError("Expected a relocatable WebAssembly object");
  }
  bool is_wasm64 = (*parsed)->getBytesInAddress() == 8;

  llvm::SmallString<128> directory;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory("xla-wasm", directory)) {
    return absl::InternalError(
        absl::StrCat("Creating WebAssembly link directory: ", error.message()));
  }
  llvm::scope_exit cleanup([&] { llvm::sys::fs::remove_directories(directory); });
  llvm::SmallString<128> input_path(directory);
  llvm::sys::path::append(input_path, "input.o");
  llvm::SmallString<128> output_path(directory);
  llvm::sys::path::append(output_path, "output.wasm");

  std::error_code error;
  llvm::raw_fd_ostream output(input_path, error, llvm::sys::fs::OF_None);
  if (error) {
    return absl::InternalError(
        absl::StrCat("Writing WebAssembly object: ", error.message()));
  }
  output << object.getBuffer();
  output.close();
  if (output.has_error()) {
    error = output.error();
    output.clear_error();
    return absl::InternalError(
        absl::StrCat("Writing WebAssembly object: ", error.message()));
  }

  // Separate XLA compilations can reuse symbol names. Bind definitions locally
  // so the dynamic loader cannot resolve them to an earlier module.
  std::vector<const char*> args = {"wasm-ld", "-shared", "--import-memory",
                                   "--allow-undefined", "-Bsymbolic"};
  if (is_wasm64) args.push_back("-mwasm64");
  args.insert(args.end(), {input_path.c_str(), "-o", output_path.c_str()});
  const lld::DriverDef drivers[] = {{lld::Wasm, &lld::wasm::link}};
  std::string diagnostics;
  llvm::raw_string_ostream diagnostic_stream(diagnostics);
  llvm::raw_null_ostream stdout_stream;
  lld::Result result;
  {
    // LLD owns process-global state, including its current linking context.
    static absl::Mutex mutex;
    absl::MutexLock lock(mutex);
    result = lld::lldMain(args, stdout_stream, diagnostic_stream, drivers);
    if (!result.canRunAgain) {
      llvm::report_fatal_error("LLD crashed and cannot safely be reused");
    }
  }
  if (result.retCode != 0) {
    return absl::InternalError(
        absl::StrCat("Linking WebAssembly object: ", diagnostics));
  }

  auto module = llvm::MemoryBuffer::getFileAsStream(output_path);
  if (!module) {
    return absl::InternalError(absl::StrCat(
        "Reading linked WebAssembly module: ", module.getError().message()));
  }
  return std::move(*module);
}

}  // namespace xla::cpu
