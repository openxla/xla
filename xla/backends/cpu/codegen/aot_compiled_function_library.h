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

#ifndef XLA_BACKENDS_CPU_CODEGEN_AOT_COMPILED_FUNCTION_LIBRARY_H_
#define XLA_BACKENDS_CPU_CODEGEN_AOT_COMPILED_FUNCTION_LIBRARY_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/function_library.h"

namespace xla::cpu {

// A AotCompiledFunctionLibrary is a FunctionLibrary that utilizes a symbol
// table to resolve function names to functions compiled into the object file
// linked with the current library.
class AotCompiledFunctionLibrary : public FunctionLibrary {
 public:
  using FunctionPtr = void*;

  struct MappedMemory {
    void* base = nullptr;
    size_t size = 0;

    MappedMemory() = default;
    MappedMemory(void* base, size_t size) : base(base), size(size) {}
    MappedMemory(MappedMemory&& other) noexcept;
    MappedMemory& operator=(MappedMemory&& other) noexcept;
    MappedMemory(const MappedMemory&) = delete;
    MappedMemory& operator=(const MappedMemory&) = delete;
    ~MappedMemory();
  };

  // Constructs a new AotCompiledFunctionLibrary.
  //
  // `symbols_map` is a map from symbol names to resolved symbols.
  explicit AotCompiledFunctionLibrary(
      absl::flat_hash_map<std::string, FunctionPtr> symbols_map,
      std::vector<MappedMemory> mapped_memories = {});

  // Resolves the function with the given name and type ID.
  absl::StatusOr<void*> ResolveFunction(TypeId type_id,
                                        absl::string_view name) final;

 private:
  // Caches the resolved symbols so we don't have to look them up every time a
  // function is resolved.
  absl::flat_hash_map<std::string, FunctionPtr> symbols_map_;
  std::vector<MappedMemory> mapped_memories_;
};

// A lightweight, zero-LLVM-dependency ELF object loader for AOT compiled
// XLA:CPU executables.
class AotObjectLoader {
 public:
  using Symbol = FunctionLibrary::Symbol;
  using SymbolResolver = std::function<void*(absl::string_view)>;

  explicit AotObjectLoader(SymbolResolver external_symbol_resolver = nullptr);

  AotObjectLoader(AotObjectLoader&& other) = default;
  AotObjectLoader& operator=(AotObjectLoader&& other) = default;

  absl::Status AddObjFile(absl::string_view obj_file,
                          absl::string_view memory_buffer_name = "",
                          size_t dylib_index = 0);

  absl::StatusOr<std::unique_ptr<FunctionLibrary>> Load(
      absl::Span<const Symbol> symbols) &&;

  static absl::StatusOr<std::unique_ptr<FunctionLibrary>> LoadFunctionLibrary(
      absl::Span<const Symbol> symbols, absl::Span<const std::string> obj_files,
      SymbolResolver external_symbol_resolver = nullptr);

 private:
  struct ObjFileEntry {
    std::string contents;
    std::string name;
  };

  std::vector<ObjFileEntry> obj_files_;
  SymbolResolver external_symbol_resolver_;
};

using LiteObjectLoader = AotObjectLoader;

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_AOT_COMPILED_FUNCTION_LIBRARY_H_
