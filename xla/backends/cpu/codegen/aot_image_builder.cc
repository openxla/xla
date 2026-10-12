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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/backends/cpu/codegen/pre_relocated_image_loader.h"
#include "xla/service/cpu/executable.pb.h"

namespace xla::cpu {
namespace {

namespace ELF = ::llvm::ELF;
using ::llvm::object::ELFObjectFileBase;
using ::llvm::object::ELFRelocationRef;
using ::llvm::object::ELFSectionRef;
using ::llvm::object::ELFSymbolRef;
using ::llvm::object::ObjectFile;
using ::llvm::object::RelocationRef;
using ::llvm::object::SectionRef;
using ::llvm::object::SymbolRef;

constexpr uint64_t kSegmentAlignment = kPreRelocatedImageSegmentAlignment;
constexpr uint64_t kGotSlotSize = 8;

template <typename T>
absl::StatusOr<T> Unwrap(llvm::Expected<T> expected) {
  if (!expected) {
    return absl::InvalidArgumentError(llvm::toString(expected.takeError()));
  }
  return std::move(*expected);
}

uint64_t AlignTo(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

uint64_t Page(uint64_t offset) { return offset & ~uint64_t{0xFFF}; }

bool IsInt(int64_t value, int bits) {
  return value >= -(int64_t{1} << (bits - 1)) &&
         value < (int64_t{1} << (bits - 1));
}

absl::Status Unsupported(absl::string_view what) {
  return absl::UnimplementedError(
      absl::StrCat("Can't build a pre-relocated image: ", what));
}

enum class Segment { kRx, kRo, kRw, kBss };

struct Section {
  size_t obj_index;
  SectionRef ref;
  Segment segment;
  uint64_t size;
  uint64_t alignment;
  uint64_t offset = 0;  // Offset in the image, assigned by `Layout`.
};

// A symbol definition inside the image.
struct Definition {
  size_t section_id;
  uint64_t value;  // Offset within the section.
  bool weak;
};

// The target of a relocation: either an external symbol resolved at load time,
// or a definition inside the image.
struct Target {
  bool external;
  std::string name;
  Definition definition;
};

class ImageBuilder {
 public:
  absl::Status AddObjectFile(absl::string_view contents);
  absl::Status AllocateGotAndPlt();
  absl::Status Layout();
  absl::Status ApplyRelocations();
  absl::StatusOr<PreRelocatedImageProto> Finish(
      absl::Span<const std::string> exported_symbols);

 private:
  // Calls `fn(target_section, obj_index, relocation)` for every relocation
  // applied to a section included in the image.
  template <typename Fn>
  absl::Status ForEachRelocation(Fn fn);

  absl::StatusOr<Target> ResolveSymbol(size_t obj_index,
                                       const RelocationRef& relocation);

  uint64_t ImageOffset(const Definition& definition) const {
    return sections_[definition.section_id].offset + definition.value;
  }

  static std::string GotKey(const Target& target) {
    return target.external ? absl::StrCat("E:", target.name)
                           : absl::StrCat("I:", target.definition.section_id,
                                          ":", target.definition.value);
  }

  bool IsGotRelocation(uint32_t type) const;
  bool IsBranchRelocation(uint32_t type) const;
  uint64_t PltStubSize() const;

  absl::Status ApplyX86_64(uint32_t type, uint64_t place, const Target& target,
                           int64_t addend);
  absl::Status ApplyAArch64(uint32_t type, uint64_t place, const Target& target,
                            int64_t addend);

  // Writes a 64-bit absolute pointer to `target + addend` as a load-time fixup.
  absl::Status WriteAbs64(uint64_t place, const Target& target, int64_t addend);
  void WritePltStubs();

  absl::StatusOr<uint64_t> GotSlot(const Target& target) const;
  absl::StatusOr<uint64_t> PltStub(const std::string& name) const;

  absl::Status CheckBounds(uint64_t place, uint64_t size) const;
  uint32_t Read32(uint64_t place) const;
  void Write32(uint64_t place, uint32_t value);
  void Write64(uint64_t place, uint64_t value);

  std::vector<std::unique_ptr<ObjectFile>> objs_;
  llvm::Triple::ArchType arch_ = llvm::Triple::UnknownArch;

  std::vector<Section> sections_;
  absl::flat_hash_map<std::pair<size_t, uint64_t>, size_t> section_ids_;
  absl::flat_hash_map<std::string, Definition> globals_;

  // GOT and PLT entries in deterministic (first use) order.
  std::vector<Target> got_entries_;
  absl::flat_hash_map<std::string, size_t> got_index_;
  std::vector<std::string> plt_entries_;
  absl::flat_hash_map<std::string, size_t> plt_index_;

  uint64_t plt_offset_ = 0;
  uint64_t got_offset_ = 0;
  uint64_t rx_end_ = 0;
  uint64_t rx_size_ = 0;
  uint64_t ro_end_ = 0;
  uint64_t ro_size_ = 0;
  uint64_t rw_file_end_ = 0;
  uint64_t rw_mem_end_ = 0;

  std::string image_;
  PreRelocatedImageProto proto_;
};

absl::Status ImageBuilder::AddObjectFile(absl::string_view contents) {
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<ObjectFile> obj,
      Unwrap(ObjectFile::createObjectFile(llvm::MemoryBufferRef(
          llvm::StringRef(contents.data(), contents.size()), "obj"))));

  auto* elf = llvm::dyn_cast<ELFObjectFileBase>(obj.get());
  if (elf == nullptr || !elf->is64Bit() || !elf->isLittleEndian()) {
    return Unsupported("only little-endian ELF64 objects are supported");
  }
  if (elf->getEType() != ELF::ET_REL) {
    return Unsupported("only relocatable ELF objects are supported");
  }
  llvm::Triple::ArchType arch = obj->getArch();
  if (arch != llvm::Triple::x86_64 && arch != llvm::Triple::aarch64) {
    return Unsupported(absl::StrCat("unsupported architecture ",
                                    llvm::Triple::getArchTypeName(arch).str()));
  }
  if (arch_ != llvm::Triple::UnknownArch && arch != arch_) {
    return absl::InvalidArgumentError("Object files have different targets");
  }
  arch_ = arch;

  size_t obj_index = objs_.size();
  for (const SectionRef& section : obj->sections()) {
    ELFSectionRef elf_section(section);
    uint64_t flags = elf_section.getFlags();
    if (!(flags & ELF::SHF_ALLOC)) {
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(llvm::StringRef name, Unwrap(section.getName()));
    // We don't register the image with the unwinder.
    if (name == ".eh_frame") {
      continue;
    }
    if (flags & ELF::SHF_TLS) {
      return Unsupported(absl::StrCat("TLS section ", name.str()));
    }
    uint32_t type = elf_section.getType();
    if (type == ELF::SHT_INIT_ARRAY || type == ELF::SHT_FINI_ARRAY ||
        type == ELF::SHT_PREINIT_ARRAY) {
      return Unsupported(absl::StrCat("constructor section ", name.str()));
    }
    uint64_t alignment = section.getAlignment().value();
    if (alignment > kSegmentAlignment) {
      return Unsupported(
          absl::StrFormat("section %s alignment %d", name.str(), alignment));
    }

    Segment segment = Segment::kRo;
    if (type == ELF::SHT_NOBITS) {
      segment = Segment::kBss;
    } else if (flags & ELF::SHF_EXECINSTR) {
      segment = Segment::kRx;
    } else if (flags & ELF::SHF_WRITE) {
      segment = Segment::kRw;
    }

    section_ids_[{obj_index, section.getIndex()}] = sections_.size();
    sections_.push_back(
        Section{obj_index, section, segment, section.getSize(), alignment});
  }

  for (const SymbolRef& symbol : obj->symbols()) {
    ELFSymbolRef elf_symbol(symbol);
    ABSL_ASSIGN_OR_RETURN(uint32_t flags, Unwrap(symbol.getFlags()));
    if ((flags & SymbolRef::SF_Undefined) ||
        elf_symbol.getBinding() == ELF::STB_LOCAL) {
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(llvm::StringRef name, Unwrap(symbol.getName()));
    if (flags & SymbolRef::SF_Common) {
      return Unsupported(absl::StrCat("common symbol ", name.str()));
    }
    ABSL_ASSIGN_OR_RETURN(auto section, Unwrap(symbol.getSection()));
    if (section == obj->section_end()) {
      continue;
    }
    auto it = section_ids_.find({obj_index, section->getIndex()});
    if (it == section_ids_.end()) {
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(uint64_t value, Unwrap(symbol.getValue()));
    Definition definition{it->second, value,
                          elf_symbol.getBinding() == ELF::STB_WEAK};

    auto [existing, inserted] = globals_.try_emplace(name.str(), definition);
    if (!inserted && !definition.weak) {
      if (!existing->second.weak) {
        return Unsupported(absl::StrCat("duplicate symbol ", name.str()));
      }
      existing->second = definition;
    }
  }

  objs_.push_back(std::move(obj));
  return absl::OkStatus();
}

template <typename Fn>
absl::Status ImageBuilder::ForEachRelocation(Fn fn) {
  for (size_t obj_index = 0; obj_index < objs_.size(); ++obj_index) {
    const ObjectFile& obj = *objs_[obj_index];
    for (const SectionRef& section : obj.sections()) {
      ABSL_ASSIGN_OR_RETURN(auto relocated,
                            Unwrap(section.getRelocatedSection()));
      if (relocated == obj.section_end()) {
        continue;
      }
      auto it = section_ids_.find({obj_index, relocated->getIndex()});
      if (it == section_ids_.end()) {
        continue;
      }
      for (const RelocationRef& relocation : section.relocations()) {
        ABSL_RETURN_IF_ERROR(fn(sections_[it->second], obj_index, relocation));
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<Target> ImageBuilder::ResolveSymbol(
    size_t obj_index, const RelocationRef& relocation) {
  const ObjectFile& obj = *objs_[obj_index];
  auto symbol = relocation.getSymbol();
  if (symbol == obj.symbol_end()) {
    return Unsupported("relocation without a symbol");
  }

  ABSL_ASSIGN_OR_RETURN(uint32_t flags, Unwrap(symbol->getFlags()));
  ABSL_ASSIGN_OR_RETURN(llvm::StringRef name, Unwrap(symbol->getName()));
  if (flags & (SymbolRef::SF_Common | SymbolRef::SF_Absolute)) {
    return Unsupported(absl::StrCat("common or absolute symbol ", name.str()));
  }

  bool is_local = ELFSymbolRef(*symbol).getBinding() == ELF::STB_LOCAL;
  if (!is_local) {
    // Global symbols may be defined in another object file, or overridden by a
    // strong definition if this one is weak.
    if (auto it = globals_.find(absl::string_view(name.data(), name.size()));
        it != globals_.end()) {
      return Target{false, "", it->second};
    }
    if (flags & SymbolRef::SF_Undefined) {
      return Target{true, name.str(), {}};
    }
  }

  ABSL_ASSIGN_OR_RETURN(auto section, Unwrap(symbol->getSection()));
  auto it = section == obj.section_end()
                ? section_ids_.end()
                : section_ids_.find({obj_index, section->getIndex()});
  if (it == section_ids_.end()) {
    return Unsupported(absl::StrCat("reference to symbol ", name.str(),
                                    " outside of the image"));
  }
  ABSL_ASSIGN_OR_RETURN(uint64_t value, Unwrap(symbol->getValue()));
  return Target{false, "", Definition{it->second, value, false}};
}

bool ImageBuilder::IsGotRelocation(uint32_t type) const {
  if (arch_ == llvm::Triple::x86_64) {
    return type == ELF::R_X86_64_GOTPCREL || type == ELF::R_X86_64_GOTPCRELX ||
           type == ELF::R_X86_64_REX_GOTPCRELX;
  }
  return type == ELF::R_AARCH64_ADR_GOT_PAGE ||
         type == ELF::R_AARCH64_LD64_GOT_LO12_NC;
}

bool ImageBuilder::IsBranchRelocation(uint32_t type) const {
  if (arch_ == llvm::Triple::x86_64) {
    return type == ELF::R_X86_64_PLT32;
  }
  return type == ELF::R_AARCH64_CALL26 || type == ELF::R_AARCH64_JUMP26;
}

uint64_t ImageBuilder::PltStubSize() const {
  return arch_ == llvm::Triple::x86_64 ? 8 : 16;
}

absl::Status ImageBuilder::AllocateGotAndPlt() {
  return ForEachRelocation(
      [&](const Section&, size_t obj_index,
          const RelocationRef& relocation) -> absl::Status {
        uint32_t type = relocation.getType();
        bool needs_plt = IsBranchRelocation(type);
        if (!IsGotRelocation(type) && !needs_plt) {
          return absl::OkStatus();
        }
        ABSL_ASSIGN_OR_RETURN(Target target,
                              ResolveSymbol(obj_index, relocation));
        if (needs_plt && !target.external) {
          return absl::OkStatus();
        }

        auto [got_it, got_inserted] =
            got_index_.try_emplace(GotKey(target), got_entries_.size());
        if (got_inserted) {
          got_entries_.push_back(target);
        }
        if (needs_plt) {
          auto [plt_it, plt_inserted] =
              plt_index_.try_emplace(target.name, plt_entries_.size());
          if (plt_inserted) {
            plt_entries_.push_back(target.name);
          }
        }
        return absl::OkStatus();
      });
}

absl::Status ImageBuilder::Layout() {
  auto place_sections = [&](Segment segment, uint64_t offset) {
    for (Section& section : sections_) {
      if (section.segment == segment) {
        section.offset = AlignTo(offset, section.alignment);
        offset = section.offset + section.size;
      }
    }
    return offset;
  };

  uint64_t offset = place_sections(Segment::kRx, 0);
  plt_offset_ = AlignTo(offset, 16);
  rx_end_ = plt_offset_ + plt_entries_.size() * PltStubSize();
  rx_size_ = AlignTo(rx_end_, kSegmentAlignment);

  offset = place_sections(Segment::kRo, rx_size_);
  got_offset_ = AlignTo(offset, kGotSlotSize);
  ro_end_ = got_offset_ + got_entries_.size() * kGotSlotSize;
  ro_size_ = AlignTo(ro_end_ - rx_size_, kSegmentAlignment);

  rw_file_end_ = place_sections(Segment::kRw, rx_size_ + ro_size_);
  rw_mem_end_ = place_sections(Segment::kBss, rw_file_end_);

  image_.assign(rw_file_end_, '\0');
  for (const Section& section : sections_) {
    if (section.segment == Segment::kBss) {
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(llvm::StringRef contents,
                          Unwrap(section.ref.getContents()));
    std::memcpy(image_.data() + section.offset, contents.data(),
                contents.size());
  }
  return absl::OkStatus();
}

absl::Status ImageBuilder::CheckBounds(uint64_t place, uint64_t size) const {
  if (place > image_.size() || size > image_.size() - place) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Relocation at offset %d is out of bounds", place));
  }
  return absl::OkStatus();
}

uint32_t ImageBuilder::Read32(uint64_t place) const {
  uint32_t value;
  std::memcpy(&value, image_.data() + place, sizeof(value));
  return value;
}

void ImageBuilder::Write32(uint64_t place, uint32_t value) {
  std::memcpy(image_.data() + place, &value, sizeof(value));
}

void ImageBuilder::Write64(uint64_t place, uint64_t value) {
  std::memcpy(image_.data() + place, &value, sizeof(value));
}

absl::StatusOr<uint64_t> ImageBuilder::GotSlot(const Target& target) const {
  auto it = got_index_.find(GotKey(target));
  if (it == got_index_.end()) {
    return absl::InternalError("Missing GOT entry");
  }
  return got_offset_ + it->second * kGotSlotSize;
}

absl::StatusOr<uint64_t> ImageBuilder::PltStub(const std::string& name) const {
  auto it = plt_index_.find(name);
  if (it == plt_index_.end()) {
    return absl::InternalError(absl::StrCat("Missing PLT entry for ", name));
  }
  return plt_offset_ + it->second * PltStubSize();
}

absl::Status ImageBuilder::WriteAbs64(uint64_t place, const Target& target,
                                      int64_t addend) {
  ABSL_RETURN_IF_ERROR(CheckBounds(place, 8));
  Write64(place, 0);
  if (target.external) {
    auto* fixup = proto_.add_external_fixups();
    fixup->set_image_offset(place);
    fixup->set_symbol_name(target.name);
    fixup->set_addend(addend);
  } else {
    auto* fixup = proto_.add_relative_fixups();
    fixup->set_image_offset(place);
    fixup->set_target_image_offset(ImageOffset(target.definition) +
                                   static_cast<uint64_t>(addend));
  }
  return absl::OkStatus();
}

absl::Status ImageBuilder::ApplyX86_64(uint32_t type, uint64_t place,
                                       const Target& target, int64_t addend) {
  auto write_pc32 = [&](uint64_t dest) -> absl::Status {
    ABSL_RETURN_IF_ERROR(CheckBounds(place, 4));
    int64_t value = static_cast<int64_t>(dest + addend - place);
    if (!IsInt(value, 32)) {
      return Unsupported("PC-relative displacement doesn't fit in 32 bits");
    }
    Write32(place, static_cast<uint32_t>(value));
    return absl::OkStatus();
  };

  switch (type) {
    case ELF::R_X86_64_NONE:
      return absl::OkStatus();
    case ELF::R_X86_64_64:
      return WriteAbs64(place, target, addend);
    case ELF::R_X86_64_PLT32:
      if (target.external) {
        ABSL_ASSIGN_OR_RETURN(uint64_t stub, PltStub(target.name));
        return write_pc32(stub);
      }
      return write_pc32(ImageOffset(target.definition));
    case ELF::R_X86_64_PC32:
      if (target.external) {
        return Unsupported(
            absl::StrCat("R_X86_64_PC32 to external symbol ", target.name));
      }
      return write_pc32(ImageOffset(target.definition));
    case ELF::R_X86_64_PC64:
      if (target.external) {
        return Unsupported(
            absl::StrCat("R_X86_64_PC64 to external symbol ", target.name));
      }
      ABSL_RETURN_IF_ERROR(CheckBounds(place, 8));
      Write64(place, ImageOffset(target.definition) + addend - place);
      return absl::OkStatus();
    case ELF::R_X86_64_GOTPCREL:
    case ELF::R_X86_64_GOTPCRELX:
    case ELF::R_X86_64_REX_GOTPCRELX: {
      ABSL_ASSIGN_OR_RETURN(uint64_t slot, GotSlot(target));
      return write_pc32(slot);
    }
    default:
      return Unsupported(absl::StrCat("x86-64 relocation type ", type));
  }
}

absl::Status ImageBuilder::ApplyAArch64(uint32_t type, uint64_t place,
                                        const Target& target, int64_t addend) {
  if (type == ELF::R_AARCH64_NONE) {
    return absl::OkStatus();
  }
  if (type == ELF::R_AARCH64_ABS64) {
    return WriteAbs64(place, target, addend);
  }
  if (type == ELF::R_AARCH64_PREL64) {
    if (target.external) {
      return Unsupported(
          absl::StrCat("PC-relative reference to external "
                       "symbol ",
                       target.name));
    }
    ABSL_RETURN_IF_ERROR(CheckBounds(place, 8));
    Write64(place, ImageOffset(target.definition) + addend - place);
    return absl::OkStatus();
  }

  ABSL_RETURN_IF_ERROR(CheckBounds(place, 4));
  uint32_t insn = Read32(place);

  auto set_adr_imm = [&](int64_t imm) {
    insn &= ~((0x3u << 29) | (0x7FFFFu << 5));
    insn |= (static_cast<uint32_t>(imm) & 0x3) << 29;
    insn |= ((static_cast<uint32_t>(imm) >> 2) & 0x7FFFF) << 5;
  };
  auto set_imm12 = [&](uint64_t imm) {
    insn &= ~(0xFFFu << 10);
    insn |= (static_cast<uint32_t>(imm) & 0xFFF) << 10;
  };

  // GOT relocations.
  if (type == ELF::R_AARCH64_ADR_GOT_PAGE ||
      type == ELF::R_AARCH64_LD64_GOT_LO12_NC) {
    if (addend != 0) {
      return Unsupported("GOT relocation with a non-zero addend");
    }
    ABSL_ASSIGN_OR_RETURN(uint64_t slot, GotSlot(target));
    if (type == ELF::R_AARCH64_ADR_GOT_PAGE) {
      set_adr_imm(static_cast<int64_t>(Page(slot) - Page(place)) >> 12);
    } else {
      set_imm12((slot & 0xFFF) >> 3);
    }
    Write32(place, insn);
    return absl::OkStatus();
  }

  // Branches, which go through a PLT stub for external symbols.
  if (type == ELF::R_AARCH64_CALL26 || type == ELF::R_AARCH64_JUMP26) {
    uint64_t dest = 0;
    if (target.external) {
      ABSL_ASSIGN_OR_RETURN(dest, PltStub(target.name));
    } else {
      dest = ImageOffset(target.definition);
    }
    int64_t value = static_cast<int64_t>(dest + addend - place);
    if (!IsInt(value, 28) || value % 4 != 0) {
      return Unsupported("branch target is out of range");
    }
    insn = (insn & ~0x03FFFFFFu) |
           ((static_cast<uint32_t>(value) >> 2) & 0x03FFFFFFu);
    Write32(place, insn);
    return absl::OkStatus();
  }

  if (target.external) {
    return Unsupported(absl::StrFormat(
        "AArch64 relocation type %d to external symbol %s", type, target.name));
  }
  uint64_t dest = ImageOffset(target.definition) + addend;
  int64_t pc_rel = static_cast<int64_t>(dest - place);

  auto lo12_shift = [](uint32_t lo12_type) -> int {
    switch (lo12_type) {
      case ELF::R_AARCH64_ADD_ABS_LO12_NC:
      case ELF::R_AARCH64_LDST8_ABS_LO12_NC:
        return 0;
      case ELF::R_AARCH64_LDST16_ABS_LO12_NC:
        return 1;
      case ELF::R_AARCH64_LDST32_ABS_LO12_NC:
        return 2;
      case ELF::R_AARCH64_LDST64_ABS_LO12_NC:
        return 3;
      case ELF::R_AARCH64_LDST128_ABS_LO12_NC:
        return 4;
      default:
        return -1;
    }
  };

  switch (type) {
    case ELF::R_AARCH64_PREL32:
      if (!IsInt(pc_rel, 32)) {
        return Unsupported("R_AARCH64_PREL32 out of range");
      }
      insn = static_cast<uint32_t>(pc_rel);
      break;
    case ELF::R_AARCH64_ADR_PREL_PG_HI21:
    case ELF::R_AARCH64_ADR_PREL_PG_HI21_NC: {
      int64_t pages = static_cast<int64_t>(Page(dest) - Page(place)) >> 12;
      if (type == ELF::R_AARCH64_ADR_PREL_PG_HI21 && !IsInt(pages, 21)) {
        return Unsupported("ADRP target is out of range");
      }
      set_adr_imm(pages);
      break;
    }
    case ELF::R_AARCH64_ADR_PREL_LO21:
      if (!IsInt(pc_rel, 21)) {
        return Unsupported("ADR target is out of range");
      }
      set_adr_imm(pc_rel);
      break;
    case ELF::R_AARCH64_CONDBR19:
      if (!IsInt(pc_rel, 21) || pc_rel % 4 != 0) {
        return Unsupported("conditional branch target is out of range");
      }
      insn = (insn & ~(0x7FFFFu << 5)) |
             (((static_cast<uint32_t>(pc_rel) >> 2) & 0x7FFFF) << 5);
      break;
    case ELF::R_AARCH64_TSTBR14:
      if (!IsInt(pc_rel, 16) || pc_rel % 4 != 0) {
        return Unsupported("test branch target is out of range");
      }
      insn = (insn & ~(0x3FFFu << 5)) |
             (((static_cast<uint32_t>(pc_rel) >> 2) & 0x3FFF) << 5);
      break;
    default: {
      int shift = lo12_shift(type);
      if (shift < 0) {
        return Unsupported(absl::StrCat("AArch64 relocation type ", type));
      }
      uint64_t lo12 = dest & 0xFFF;
      if (lo12 % (uint64_t{1} << shift) != 0) {
        return absl::InvalidArgumentError("Misaligned LO12 relocation");
      }
      set_imm12(lo12 >> shift);
      break;
    }
  }
  Write32(place, insn);
  return absl::OkStatus();
}

absl::Status ImageBuilder::ApplyRelocations() {
  return ForEachRelocation(
      [&](const Section& section, size_t obj_index,
          const RelocationRef& relocation) -> absl::Status {
        uint32_t type = relocation.getType();
        if ((arch_ == llvm::Triple::x86_64 && type == ELF::R_X86_64_NONE) ||
            (arch_ == llvm::Triple::aarch64 && type == ELF::R_AARCH64_NONE)) {
          return absl::OkStatus();
        }
        if (section.segment == Segment::kBss) {
          return absl::InvalidArgumentError("Relocation in a NOBITS section");
        }
        ABSL_ASSIGN_OR_RETURN(Target target,
                              ResolveSymbol(obj_index, relocation));
        ABSL_ASSIGN_OR_RETURN(int64_t addend,
                              Unwrap(ELFRelocationRef(relocation).getAddend()));
        uint64_t place = section.offset + relocation.getOffset();
        if (arch_ == llvm::Triple::x86_64) {
          return ApplyX86_64(type, place, target, addend);
        }
        return ApplyAArch64(type, place, target, addend);
      });
}

void ImageBuilder::WritePltStubs() {
  for (size_t i = 0; i < plt_entries_.size(); ++i) {
    uint64_t stub = plt_offset_ + i * PltStubSize();
    uint64_t slot =
        got_offset_ +
        got_index_.at(absl::StrCat("E:", plt_entries_[i])) * kGotSlotSize;
    if (arch_ == llvm::Triple::x86_64) {
      // jmp *slot(%rip); int3; int3
      int32_t disp = static_cast<int32_t>(slot - (stub + 6));
      image_[stub + 0] = '\xff';
      image_[stub + 1] = '\x25';
      Write32(stub + 2, static_cast<uint32_t>(disp));
      image_[stub + 6] = '\xcc';
      image_[stub + 7] = '\xcc';
    } else {
      // adrp x16, slot; ldr x16, [x16, :lo12:slot]; br x16; brk #0
      int64_t pages = static_cast<int64_t>(Page(slot) - Page(stub)) >> 12;
      uint32_t adrp = 0x90000010u |
                      ((static_cast<uint32_t>(pages) & 0x3) << 29) |
                      (((static_cast<uint32_t>(pages) >> 2) & 0x7FFFF) << 5);
      uint32_t ldr = 0xF9400210u | (((slot & 0xFFF) >> 3) << 10);
      Write32(stub + 0, adrp);
      Write32(stub + 4, ldr);
      Write32(stub + 8, 0xD61F0200u);
      Write32(stub + 12, 0xD4200000u);
    }
  }
}

absl::StatusOr<PreRelocatedImageProto> ImageBuilder::Finish(
    absl::Span<const std::string> exported_symbols) {
  WritePltStubs();

  for (size_t i = 0; i < got_entries_.size(); ++i) {
    ABSL_RETURN_IF_ERROR(
        WriteAbs64(got_offset_ + i * kGotSlotSize, got_entries_[i], 0));
  }

  for (const std::string& name : exported_symbols) {
    auto it = globals_.find(name);
    if (it == globals_.end()) {
      return absl::NotFoundError(
          absl::StrCat("Exported symbol ", name, " is not defined"));
    }
    auto* symbol = proto_.add_exported_symbols();
    symbol->set_name(name);
    symbol->set_image_offset(ImageOffset(it->second));
  }

  absl::string_view image = image_;
  uint64_t rw_begin = rx_size_ + ro_size_;
  proto_.set_rx_bytes(image.substr(0, rx_end_));
  proto_.set_ro_bytes(image.substr(rx_size_, ro_end_ - rx_size_));
  proto_.set_rw_bytes(image.substr(rw_begin, rw_file_end_ - rw_begin));
  proto_.set_rx_size(rx_size_);
  proto_.set_ro_size(ro_size_);
  proto_.set_rw_mem_size(rw_mem_end_ - rw_begin);
  return std::move(proto_);
}

}  // namespace

absl::StatusOr<PreRelocatedImageProto> BuildPreRelocatedImage(
    absl::Span<const absl::string_view> obj_files,
    absl::Span<const std::string> exported_symbols) {
  ImageBuilder builder;
  for (absl::string_view obj_file : obj_files) {
    ABSL_RETURN_IF_ERROR(builder.AddObjectFile(obj_file));
  }
  ABSL_RETURN_IF_ERROR(builder.AllocateGotAndPlt());
  ABSL_RETURN_IF_ERROR(builder.Layout());
  ABSL_RETURN_IF_ERROR(builder.ApplyRelocations());
  return builder.Finish(exported_symbols);
}

}  // namespace xla::cpu
