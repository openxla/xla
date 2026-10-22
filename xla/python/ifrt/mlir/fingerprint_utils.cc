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

#include "xla/python/ifrt/mlir/fingerprint_utils.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "highwayhash/arch_specific.h"
#include "highwayhash/hh_types.h"
#include "highwayhash/highwayhash.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Bytecode/Encoding.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/hlo.pb.h"
#include "xla/status_macros.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"

namespace xla {
namespace ifrt {

namespace {

// Name of the Shardy mesh op. Its symbol name only identifies the mesh.
constexpr llvm::StringLiteral kSdyMeshOpName = "sdy.mesh";

// Bytecode attribute group of the encodings of canonicalized names.
constexpr llvm::StringLiteral kCanonicalNameGroup = "ifrt.canonical_name";
constexpr uint64_t kSymbolDefinitionTag = 0;
constexpr uint64_t kSymbolReferenceTag = 1;

absl::string_view ToStringView(llvm::StringRef str) {
  return absl::string_view(str.data(), str.size());
}

mlir::StringAttr SymbolName(mlir::Operation& op) {
  return op.getAttrOfType<mlir::StringAttr>(
      mlir::SymbolTable::getSymbolAttrName());
}

// Returns whether `dict` holds exactly the attributes that the bytecode writer
// emits for `op` before native properties encoding, i.e. its inherent and
// discardable attributes.
bool IsAttrDictionaryOf(mlir::DictionaryAttr dict, mlir::Operation* op) {
  size_t num_attrs = 0;
  bool equal = true;
  auto visit = [&](llvm::StringRef name, mlir::Attribute value) {
    ++num_attrs;
    equal = equal && dict.get(name) == value;
  };
  if (op->getPropertiesStorage()) {
    op->getName().walkInherentAttrs(
        op, [&](llvm::StringRef name, mlir::Attribute& value) {
          visit(name, value);
        });
    for (const mlir::NamedAttribute attr : op->getDiscardableAttrs()) {
      visit(attr.getName().getValue(), attr.getValue());
    }
  } else {
    for (const mlir::NamedAttribute attr : op->getRawDictionaryAttrs()) {
      visit(attr.getName().getValue(), attr.getValue());
    }
  }
  return equal && num_attrs == dict.size();
}

// Bytecode writer callback that encodes the names canonicalized by
// `FingerprintOptions` as IDs: 0 for the module, and 1 + the rank of the name
// among the mesh names for a mesh. It writes custom encodings instead of
// creating renamed attributes, because modules that share an MLIR context may
// be fingerprinted concurrently.
class NameCanonicalizingWriter {
 public:
  NameCanonicalizingWriter(mlir::ModuleOp module_op,
                           const FingerprintOptions& options) {
    if (options.canonicalize_sdy_mesh_names) {
      // A reference in a nested symbol table resolves to a symbol of that
      // table, so a mesh with the name of such a symbol keeps its name.
      absl::flat_hash_set<absl::string_view> nested_symbol_names;
      module_op->walk([&](mlir::Operation* op) {
        if (op == module_op.getOperation() ||
            !op->hasTrait<mlir::OpTrait::SymbolTable>()) {
          return;
        }
        for (mlir::Region& region : op->getRegions()) {
          for (mlir::Operation& symbol : region.getOps()) {
            if (mlir::StringAttr name = SymbolName(symbol)) {
              nested_symbol_names.insert(ToStringView(name.getValue()));
            }
          }
        }
      });
      std::vector<std::pair<llvm::StringRef, mlir::Operation*>> meshes;
      for (mlir::Operation& op : *module_op.getBody()) {
        if (op.getName().getStringRef() != kSdyMeshOpName) {
          continue;
        }
        if (mlir::StringAttr name = SymbolName(op);
            name &&
            !nested_symbol_names.contains(ToStringView(name.getValue()))) {
          meshes.push_back({name.getValue(), &op});
        }
      }
      absl::c_sort(meshes, [](const auto& a, const auto& b) {
        return a.first < b.first;
      });
      for (size_t i = 0; i < meshes.size(); ++i) {
        const uint64_t id = i + 1;
        const auto& [name, op] = meshes[i];
        definitions_.push_back({op, id});
        mesh_ids_.try_emplace(ToStringView(name), id);
      }
    }
    if (options.canonicalize_module_name &&
        module_op.getSymName().has_value()) {
      definitions_.push_back({module_op.getOperation(), 0});
    }
  }

  mlir::LogicalResult Write(mlir::Attribute attr,
                            std::optional<llvm::StringRef>& group,
                            mlir::DialectBytecodeWriter& writer) const {
    if (auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(attr)) {
      return WriteDefinition(dict, group, writer);
    }
    if (auto ref = mlir::dyn_cast<mlir::SymbolRefAttr>(attr)) {
      return WriteReference(ref, group, writer);
    }
    return mlir::failure();
  }

 private:
  struct Definition {
    mlir::Operation* op;
    uint64_t id;
  };

  // Encodes the attribute dictionary of the module or of a mesh op with the ID
  // of the module or mesh in place of its name.
  mlir::LogicalResult WriteDefinition(
      mlir::DictionaryAttr dict, std::optional<llvm::StringRef>& group,
      mlir::DialectBytecodeWriter& writer) const {
    const llvm::StringRef name_attr_name =
        mlir::SymbolTable::getSymbolAttrName();
    mlir::Attribute name = dict.get(name_attr_name);
    if (!name) {
      return mlir::failure();
    }
    for (const Definition& definition : definitions_) {
      if (definition.op->getAttr(name_attr_name) != name ||
          !IsAttrDictionaryOf(dict, definition.op)) {
        continue;
      }
      group = kCanonicalNameGroup;
      writer.writeVarInt(kSymbolDefinitionTag);
      writer.writeVarInt(definition.id);
      writer.writeVarInt(dict.size() - 1);
      for (const mlir::NamedAttribute attr : dict) {
        if (attr.getName() != name_attr_name) {
          writer.writeAttribute(attr.getName());
          writer.writeAttribute(attr.getValue());
        }
      }
      return mlir::success();
    }
    return mlir::failure();
  }

  // Encodes a reference to a mesh with the ID of the mesh in place of its name.
  mlir::LogicalResult WriteReference(
      mlir::SymbolRefAttr ref, std::optional<llvm::StringRef>& group,
      mlir::DialectBytecodeWriter& writer) const {
    auto it = mesh_ids_.find(ToStringView(ref.getRootReference().getValue()));
    if (it == mesh_ids_.end()) {
      return mlir::failure();
    }
    group = kCanonicalNameGroup;
    writer.writeVarInt(kSymbolReferenceTag);
    writer.writeVarInt(it->second);
    writer.writeVarInt(ref.getNestedReferences().size());
    for (const mlir::FlatSymbolRefAttr nested : ref.getNestedReferences()) {
      writer.writeAttribute(nested);
    }
    return mlir::success();
  }

  std::vector<Definition> definitions_;
  absl::flat_hash_map<absl::string_view, uint64_t> mesh_ids_;
};

// Calculates a HighwayHash fingerprint in a streaming manner.
class HighwayHashStream final : public llvm::raw_ostream {
 public:
  HighwayHashStream() : hash_(kHighwayHashKey), pos_(0) { SetUnbuffered(); }

  ~HighwayHashStream() override { CHECK_EQ(buffer_pos_, 0); }

  // Destructively calculates the fingerprint of the data consumed so far.
  uint64_t fingerprint() && {
    flush_buffer();
    highwayhash::HHResult64 result;
    hash_.Finalize(&result);
    return result;
  }

 private:
  // `HighwayHashCatT::Append` updates the hash directly from the source buffer
  // if the provided buffer size is larger than `sizeof(HHPacket)`.
  static constexpr size_t kBufferSize = sizeof(highwayhash::HHPacket);

  // Arbitrarily chosen, forever-unchanging hash key required by HighwayHash.
  static constexpr highwayhash::HHKey kHighwayHashKey = {
      0x4451e30f87db9609ULL,
      0xca7358a1fd2737f8ULL,
      0x4b2c991fcee4fdeaULL,
      0x0b2658e18326f6baULL,
  };

  void write_impl(const char* Ptr, size_t Size) final {
    // For tiny writes, it is more efficient to accumulate the data to a buffer
    // first and flush it since `HighwayHashCatT::Append` is optimized for
    // large writes.
    static constexpr size_t kSmallWriteSize = 4;
    static_assert(kSmallWriteSize <= kBufferSize);

    if (Size <= kSmallWriteSize) {
      if (buffer_pos_ + Size > kBufferSize) {
        flush_buffer();
      }
      std::memcpy(buffer_ + buffer_pos_, Ptr, Size);
      buffer_pos_ += Size;
    } else {
      flush_buffer();
      hash_.Append(Ptr, Size);
    }
    pos_ += Size;
  }

  uint64_t current_pos() const final { return pos_; }

  void flush_buffer() {
    if (buffer_pos_ > 0) {
      hash_.Append(buffer_, buffer_pos_);
      buffer_pos_ = 0;
    }
  }

  highwayhash::HighwayHashCatT<HH_TARGET> hash_;
  uint64_t pos_;

  char buffer_[kBufferSize];
  uint64_t buffer_pos_ = 0;
};

}  // namespace

absl::StatusOr<uint64_t> FingerprintModuleOp(
    mlir::ModuleOp module_op, const FingerprintOptions& options) {
  tsl::StatusScopedDiagnosticHandler diag_handler(module_op->getContext());

  mlir::BytecodeWriterConfig config;
  config.setElideLocations(true);

  // Use a version before `kUseListOrdering` due to an MLIR bug where use list
  // ordering is not stable.
  //
  // TODO(b/503120525): Remove this workaround once
  // https://github.com/llvm/llvm-project/pull/191942 lands.
  config.setDesiredBytecodeVersion(
      mlir::bytecode::BytecodeVersion::kLazyLoading);

  if (options.canonicalize_module_name || options.canonicalize_sdy_mesh_names) {
    // `NameCanonicalizingWriter` finds symbol names in the attribute
    // dictionaries of ops, which hold inherent attributes only before native
    // properties.
    TF_RET_CHECK(config.getDesiredBytecodeVersion() <
                 mlir::bytecode::BytecodeVersion::kNativePropertiesEncoding);
    config.attachAttributeCallback(
        [writer = NameCanonicalizingWriter(module_op, options)](
            mlir::Attribute attr, std::optional<llvm::StringRef>& group,
            mlir::DialectBytecodeWriter& dialect_writer) {
          return writer.Write(attr, group, dialect_writer);
        });
  }

  HighwayHashStream os;
  mlir::LogicalResult result = mlir::writeBytecodeToFile(module_op, os, config);
  ABSL_RETURN_IF_ERROR(diag_handler.consumeStatus());
  TF_RET_CHECK(mlir::succeeded(result));
  return std::move(os).fingerprint();
}

}  // namespace ifrt
}  // namespace xla
