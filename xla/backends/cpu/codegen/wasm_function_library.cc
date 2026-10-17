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

#include "xla/backends/cpu/codegen/wasm_function_library.h"

#ifdef __EMSCRIPTEN__

#include <dlfcn.h>

#include <memory>
#include <string>
#include <system_error>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/backends/cpu/codegen/aot_compiled_function_library.h"
#include "xla/backends/cpu/codegen/wasm_object_linker.h"
#include "xla/backends/cpu/runtime/function_library.h"

namespace xla::cpu {
namespace {

class WasmFunctionLibrary final : public AotCompiledFunctionLibrary {
 public:
  WasmFunctionLibrary(void* handle,
                      absl::flat_hash_map<std::string, void*> symbols)
      : AotCompiledFunctionLibrary(std::move(symbols)), handle_(handle) {}

  ~WasmFunctionLibrary() override { dlclose(handle_); }

 private:
  void* handle_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<FunctionLibrary>> LoadWasmFunctionLibrary(
    absl::Span<const FunctionLibrary::Symbol> symbols,
    llvm::MemoryBufferRef object) {
  ABSL_ASSIGN_OR_RETURN(auto module, LinkWasmObject(object));
  int fd;
  llvm::SmallString<128> path;
  if (std::error_code error =
          llvm::sys::fs::createTemporaryFile("xla-wasm", "wasm", fd, path)) {
    return absl::InternalError(
        absl::StrCat("Creating WebAssembly module file: ", error.message()));
  }
  llvm::scope_exit remove_file([&] { llvm::sys::fs::remove(path); });
  llvm::raw_fd_ostream stream(fd, /*shouldClose=*/true);
  stream << module->getBuffer();
  stream.close();
  if (stream.has_error()) {
    std::error_code error = stream.error();
    stream.clear_error();
    return absl::InternalError(
        absl::StrCat("Writing WebAssembly module: ", error.message()));
  }

  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char* error = dlerror();
    return absl::InternalError(absl::StrCat(
        "Loading WebAssembly module: ", error ? error : "unknown loader error"));
  }
  llvm::scope_exit close_handle([&] { dlclose(handle); });
  absl::flat_hash_map<std::string, void*> resolved;
  for (const FunctionLibrary::Symbol& symbol : symbols) {
    void* address = dlsym(handle, symbol.name.c_str());
    if (address == nullptr) {
      return absl::NotFoundError(
          absl::StrCat("WebAssembly function not found: ", symbol.name));
    }
    resolved.emplace(symbol.name, address);
  }
  auto library =
      std::make_unique<WasmFunctionLibrary>(handle, std::move(resolved));
  close_handle.release();
  return library;
}

}  // namespace xla::cpu

#endif  // __EMSCRIPTEN__
