/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/object_loader.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "xla/backends/cpu/codegen/compiled_function_library.h"
#include "xla/backends/cpu/codegen/contiguous_section_memory_manager.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/service/cpu/orc_jit_memory_mapper.h"

namespace xla::cpu {

static std::unique_ptr<llvm::orc::RTDyldObjectLinkingLayer>
CreateObjectLinkingLayer(llvm::orc::ExecutionSession& execution_session) {
  return std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
      execution_session, [] {
        return std::make_unique<ContiguousSectionMemoryManager>(
            orc_jit_memory_mapper::GetInstance());
      });
}

ObjectLoader::ObjectLoader(size_t num_dylibs)
/*: target_machine_(std::move(target_machine))*/ {
  // LLVM execution session that holds jit-compiled functions.
  execution_session_ = std::make_unique<llvm::orc::ExecutionSession>(
      std::make_unique<llvm::orc::UnsupportedExecutorProcessControl>(
          /*SSP=*/nullptr, /*D=*/nullptr));

  execution_session_->setErrorReporter([](llvm::Error err) {
    LOG(ERROR) << "LLVM compilation error: " << llvm::toString(std::move(err));
  });

  // Create at least one dynamic library for the given jit compiler.
  dylibs_.resize(std::max<size_t>(1, num_dylibs));
  for (size_t i = 0; i < dylibs_.size(); ++i) {
    dylibs_[i] = &execution_session_->createBareJITDylib(
        absl::StrCat("<xla_jit_dylib_", i, ">"));
    // TODO using target machine might bring some deps we don't need.
    // as a first attempt fully remove it, consider pruning the reqs
    // if (definition_generator) {
    //   dylibs_[i]->addGenerator(definition_generator(target_machine_.get()));
    // }
  }

  object_layer_ = CreateObjectLinkingLayer(*execution_session_);
}

absl::Status ObjectLoader::AddObjFile(const std::string& obj_file,
                                      const std::string& memory_buffer_name,
                                      size_t dylib_index) {
  if (dylib_index >= dylibs_.size()) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Invalid dylib index %d (num dylibs: %d))", dylib_index,
                        dylibs_.size()));
  }

  llvm::StringRef data(obj_file.data(), obj_file.size());

  auto obj_file_mem_buffer =
      llvm::MemoryBuffer::getMemBuffer(data, memory_buffer_name);

  if (!obj_file_mem_buffer) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Failed to create memory buffer");
  }

  llvm::orc::JITDylib* dylib = dylibs_[dylib_index];
  if (auto err = object_layer_->add(*dylib, std::move(obj_file_mem_buffer))) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Failed to add object file to dylib %d: %s",
                        dylib_index, llvm::toString(std::move(err))));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<FunctionLibrary>> ObjectLoader::Load(
    absl::Span<const Symbol> symbols, const llvm::DataLayout& data_layout) && {
  // Mangle symbol names for the target machine data layout.
  auto mangle = [&](absl::string_view name) {
    llvm::SmallVector<char, 40> mangled;
    llvm::Mangler::getNameWithPrefix(mangled, name, data_layout);
    return std::string(mangled.begin(), mangled.end());
  };

  // Build a symbol lookup set.
  llvm::orc::SymbolLookupSet lookup_set;
  for (const auto& symbol : symbols) {
    VLOG(5) << absl::StreamFormat(" - look up symbol: %s", symbol.name);
    lookup_set.add(execution_session_->intern(mangle(symbol.name)));
  }

  // Build a search order for the dynamic libraries.
  llvm::orc::JITDylibSearchOrder search_order(dylibs_.size());
  for (size_t i = 0; i < dylibs_.size(); ++i) {
    search_order[i] = std::make_pair(
        dylibs_[i], llvm::orc::JITDylibLookupFlags::MatchExportedSymbolsOnly);
  }

  // Look up all requested symbols in the execution session.
  auto symbol_map = execution_session_->lookup(std::move(search_order),
                                               std::move(lookup_set));

  if (auto err = symbol_map.takeError()) {
    return absl::Status(absl::StatusCode::kInternal,
                        absl::StrFormat("%s", llvm::toString(std::move(err))));
  }

  // Resolve type-erased symbol pointers from the symbol map.
  using ResolvedSymbol = CompiledFunctionLibrary::ResolvedSymbol;
  absl::flat_hash_map<std::string, ResolvedSymbol> resolved_map;

  for (const auto& symbol : symbols) {
    auto symbol_name = execution_session_->intern(mangle(symbol.name));
    llvm::orc::ExecutorSymbolDef symbol_def = symbol_map->at(symbol_name);
    llvm::orc::ExecutorAddr symbol_addr = symbol_def.getAddress();
    void* ptr = reinterpret_cast<void*>(symbol_addr.getValue());
    resolved_map[symbol.name] = ResolvedSymbol{symbol.type_id, ptr};
  }

  return std::make_unique<CompiledFunctionLibrary>(
      std::move(execution_session_), std::move(object_layer_),
      std::move(resolved_map));
}

ObjectLoader::~ObjectLoader() {
  if (execution_session_) {
    if (auto err = execution_session_->endSession()) {
      execution_session_->reportError(std::move(err));
    }
  }
}

}  // namespace xla::cpu
