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

#include "xla/backends/cpu/codegen/aot_compiled_function_library.h"

#include <dlfcn.h>
#include <elf.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/codegen/builtin_fp16.h"
#include "xla/backends/cpu/codegen/builtin_pow.h"
#include "xla/backends/cpu/runtime/function_library.h"

namespace xla::cpu {

namespace {

// Converts an F32 value to a BF16 (matching MLIR / compiler-rt ABI).
uint16_t FloatToBFloat16Bits(float float_value) {
  if (std::isnan(float_value)) {
    return std::signbit(float_value) ? 0xFFC0u : 0x7FC0u;
  }
  uint32_t float_bits = 0;
  std::memcpy(&float_bits, &float_value, sizeof(float_bits));
  uint32_t lsb = (float_bits >> 16) & 1u;
  uint32_t rounding_bias = 0x7fffu + lsb;
  float_bits += rounding_bias;
  return static_cast<uint16_t>(float_bits >> 16);
}

// On x86_64 psABI, __bf16 is returned in XMM0 (same low 16 bits as float).
// Returning float with the low 16 bits populated satisfies both XMM0 and W0
// (or uint16_t on AArch64) when callers read the architecture return register.
float TruncSFBF2Fallback(float a) {
  uint16_t bf = FloatToBFloat16Bits(a);
  float ret = 0.0f;
  std::memcpy(&ret, &bf, sizeof(bf));
  return ret;
}

float TruncDFBF2Fallback(double a) {
  return TruncSFBF2Fallback(static_cast<float>(a));
}

template <typename R, typename... Args>
void* FnVoidPtr(R (*func)(Args...)) {
  return reinterpret_cast<void*>(func);
}

using BuiltinSymbolMap = absl::flat_hash_map<std::string, void*>;

#define REGISTER_LIBM_SYMBOL(name, double_sig) \
  registry[#name "f"] = FnVoidPtr(name##f);    \
  registry[#name] = FnVoidPtr(static_cast<double_sig>(name));

BuiltinSymbolMap CreateBuiltinSymbolMap() {
  BuiltinSymbolMap registry;

  registry["memcpy"] =
      FnVoidPtr(static_cast<void* (*)(void*, const void*, size_t)>(memcpy));
  registry["memmove"] =
      FnVoidPtr(static_cast<void* (*)(void*, const void*, size_t)>(memmove));
  registry["memset"] =
      FnVoidPtr(static_cast<void* (*)(void*, int, size_t)>(memset));
  registry["malloc"] = FnVoidPtr(static_cast<void* (*)(size_t)>(malloc));
  registry["free"] = FnVoidPtr(static_cast<void (*)(void*)>(free));

  registry["__gnu_f2h_ieee"] = FnVoidPtr(__gnu_f2h_ieee);
  registry["__gnu_h2f_ieee"] = FnVoidPtr(__gnu_h2f_ieee);
  registry["__truncdfhf2"] = FnVoidPtr(__truncdfhf2);
  registry["__truncdfbf2"] = FnVoidPtr(TruncDFBF2Fallback);
  registry["__truncsfbf2"] = FnVoidPtr(TruncSFBF2Fallback);

  registry["__powisf2"] = FnVoidPtr(__powisf2);
  registry["__powidf2"] = FnVoidPtr(__powidf2);

  REGISTER_LIBM_SYMBOL(acos, double (*)(double));
  REGISTER_LIBM_SYMBOL(acosh, double (*)(double));
  REGISTER_LIBM_SYMBOL(asin, double (*)(double));
  REGISTER_LIBM_SYMBOL(asinh, double (*)(double));
  REGISTER_LIBM_SYMBOL(atan, double (*)(double));
  REGISTER_LIBM_SYMBOL(atan2, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(atanh, double (*)(double));
  REGISTER_LIBM_SYMBOL(cbrt, double (*)(double));
  REGISTER_LIBM_SYMBOL(ceil, double (*)(double));
  REGISTER_LIBM_SYMBOL(copysign, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(cos, double (*)(double));
  REGISTER_LIBM_SYMBOL(cosh, double (*)(double));
  REGISTER_LIBM_SYMBOL(erf, double (*)(double));
  REGISTER_LIBM_SYMBOL(erfc, double (*)(double));
  REGISTER_LIBM_SYMBOL(exp, double (*)(double));
  REGISTER_LIBM_SYMBOL(exp2, double (*)(double));
  REGISTER_LIBM_SYMBOL(expm1, double (*)(double));
  REGISTER_LIBM_SYMBOL(fabs, double (*)(double));
  REGISTER_LIBM_SYMBOL(fdim, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(floor, double (*)(double));
  REGISTER_LIBM_SYMBOL(fma, double (*)(double, double, double));
  REGISTER_LIBM_SYMBOL(fmax, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(fmin, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(fmod, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(frexp, double (*)(double, int*));
  REGISTER_LIBM_SYMBOL(hypot, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(ilogb, int (*)(double));
  REGISTER_LIBM_SYMBOL(ldexp, double (*)(double, int));
  REGISTER_LIBM_SYMBOL(lgamma, double (*)(double));
  REGISTER_LIBM_SYMBOL(llrint, long long (*)(double));   // NOLINT
  REGISTER_LIBM_SYMBOL(llround, long long (*)(double));  // NOLINT
  REGISTER_LIBM_SYMBOL(log, double (*)(double));
  REGISTER_LIBM_SYMBOL(log10, double (*)(double));
  REGISTER_LIBM_SYMBOL(log1p, double (*)(double));
  REGISTER_LIBM_SYMBOL(log2, double (*)(double));
  REGISTER_LIBM_SYMBOL(logb, double (*)(double));
  REGISTER_LIBM_SYMBOL(lrint, long (*)(double));   // NOLINT
  REGISTER_LIBM_SYMBOL(lround, long (*)(double));  // NOLINT
  REGISTER_LIBM_SYMBOL(modf, double (*)(double, double*));
  REGISTER_LIBM_SYMBOL(nan, double (*)(const char*));
  REGISTER_LIBM_SYMBOL(nearbyint, double (*)(double));
  REGISTER_LIBM_SYMBOL(nextafter, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(nexttoward, double (*)(double, long double));
  REGISTER_LIBM_SYMBOL(pow, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(remainder, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(remquo, double (*)(double, double, int*));
  REGISTER_LIBM_SYMBOL(rint, double (*)(double));
  REGISTER_LIBM_SYMBOL(round, double (*)(double));
  REGISTER_LIBM_SYMBOL(scalbln, double (*)(double, long));  // NOLINT
  REGISTER_LIBM_SYMBOL(scalbn, double (*)(double, int));
  REGISTER_LIBM_SYMBOL(sin, double (*)(double));
  REGISTER_LIBM_SYMBOL(sinh, double (*)(double));
  REGISTER_LIBM_SYMBOL(sqrt, double (*)(double));
  REGISTER_LIBM_SYMBOL(tan, double (*)(double));
  REGISTER_LIBM_SYMBOL(tanh, double (*)(double));
  REGISTER_LIBM_SYMBOL(tgamma, double (*)(double));
  REGISTER_LIBM_SYMBOL(trunc, double (*)(double));
  REGISTER_LIBM_SYMBOL(sincos, void (*)(double, double*, double*));

#undef REGISTER_LIBM_SYMBOL

  return registry;
}

void* LookupBuiltinOrProcessSymbol(
    absl::string_view name,
    const AotObjectLoader::SymbolResolver& external_resolver) {
  if (external_resolver) {
    if (void* addr = external_resolver(name)) {
      return addr;
    }
  }
  static const absl::NoDestructor<BuiltinSymbolMap> kBuiltins(
      CreateBuiltinSymbolMap());
  std::string name_str(name);
  if (void* dlsym_addr = dlsym(RTLD_DEFAULT, name_str.c_str())) {
    return dlsym_addr;
  }
  if (auto it = kBuiltins->find(name); it != kBuiltins->end()) {
    return it->second;
  }
  if (name.size() > 1 && name.front() == '_') {
    absl::string_view stripped = name.substr(1);
    std::string stripped_str(stripped);
    if (void* dlsym_addr = dlsym(RTLD_DEFAULT, stripped_str.c_str())) {
      return dlsym_addr;
    }
    if (auto it = kBuiltins->find(stripped); it != kBuiltins->end()) {
      return it->second;
    }
  }
  return nullptr;
}

size_t AlignUp(size_t value, size_t alignment) {
  if (alignment <= 1) return value;
  return (value + alignment - 1) & ~(alignment - 1);
}

template <typename T>
T ReadUnaligned(const void* ptr) {
  T val;
  std::memcpy(&val, ptr, sizeof(T));
  return val;
}

template <typename T>
void WriteUnaligned(void* ptr, T val) {
  std::memcpy(ptr, &val, sizeof(T));
}

void Or32LE(void* ptr, uint32_t mask) {
  uint32_t v = ReadUnaligned<uint32_t>(ptr) | mask;
  WriteUnaligned<uint32_t>(ptr, v);
}

void Or32AArch64Imm(void* ptr, uint64_t imm) {
  Or32LE(ptr, static_cast<uint32_t>((imm & 0xFFFu) << 10));
}

void Write32AArch64Addr(void* ptr, uint64_t imm) {
  uint32_t imm_lo = static_cast<uint32_t>((imm & 0x3u) << 29);
  uint32_t imm_hi = static_cast<uint32_t>((imm & 0x1FFFFCu) << 3);
  uint32_t mask = (0x3u << 29) | (0x1FFFFCu << 3);
  uint32_t cur = ReadUnaligned<uint32_t>(ptr);
  WriteUnaligned<uint32_t>(ptr, (cur & ~mask) | imm_lo | imm_hi);
}

uint64_t GetBits(uint64_t val, int start, int end) {
  uint64_t mask = (uint64_t{1} << (end + 1 - start)) - 1;
  return (val >> start) & mask;
}

struct ParsedElfObject {
  absl::string_view data;
  std::string name;
  const Elf64_Ehdr* ehdr = nullptr;
  const Elf64_Shdr* shdrs = nullptr;
  const Elf64_Sym* symtab = nullptr;
  size_t num_syms = 0;
  const char* strtab = nullptr;
  size_t strtab_size = 0;

  std::vector<uint8_t*> section_addrs;
  std::vector<uint8_t*> section_stub_cursors;
  absl::flat_hash_map<uint64_t, uint8_t*> stubs;
  absl::flat_hash_map<uint64_t, uint8_t*> got_entries;
  uint8_t* got_cursor = nullptr;
  uint8_t* got_base = nullptr;
};

absl::Status ParseElfHeader(absl::string_view bytes, absl::string_view name,
                            ParsedElfObject& out) {
  if (bytes.size() < sizeof(Elf64_Ehdr)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ELF object '%s' is too small (%d bytes)", name, bytes.size()));
  }
  const auto* ehdr = reinterpret_cast<const Elf64_Ehdr*>(bytes.data());
  if (std::memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0 ||
      ehdr->e_ident[EI_CLASS] != ELFCLASS64 ||
      ehdr->e_ident[EI_DATA] != ELFDATA2LSB) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Object '%s' is not a 64-bit little-endian ELF file", name));
  }
  if (ehdr->e_shoff == 0 || ehdr->e_shentsize != sizeof(Elf64_Shdr) ||
      ehdr->e_shoff + static_cast<size_t>(ehdr->e_shnum) * sizeof(Elf64_Shdr) >
          bytes.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ELF object '%s' has invalid section header table", name));
  }
  out.data = bytes;
  out.name = std::string(name);
  out.ehdr = ehdr;
  out.shdrs = reinterpret_cast<const Elf64_Shdr*>(bytes.data() + ehdr->e_shoff);
  out.section_addrs.assign(ehdr->e_shnum, nullptr);
  out.section_stub_cursors.assign(ehdr->e_shnum, nullptr);

  for (uint16_t i = 0; i < ehdr->e_shnum; ++i) {
    const Elf64_Shdr& sh = out.shdrs[i];
    if (sh.sh_type != SHT_NOBITS &&
        (sh.sh_offset + sh.sh_size > bytes.size())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Section %d in ELF object '%s' exceeds buffer bounds", i, name));
    }
    if (sh.sh_type == SHT_SYMTAB) {
      if (sh.sh_entsize != sizeof(Elf64_Sym) || sh.sh_link >= ehdr->e_shnum) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Invalid SHT_SYMTAB in ELF object '%s'", name));
      }
      const Elf64_Shdr& str_sh = out.shdrs[sh.sh_link];
      if (str_sh.sh_offset + str_sh.sh_size > bytes.size()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Invalid string table in ELF object '%s'", name));
      }
      out.symtab =
          reinterpret_cast<const Elf64_Sym*>(bytes.data() + sh.sh_offset);
      out.num_syms = sh.sh_size / sizeof(Elf64_Sym);
      out.strtab = bytes.data() + str_sh.sh_offset;
      out.strtab_size = str_sh.sh_size;
    }
  }
  return absl::OkStatus();
}

absl::string_view GetSymbolName(const ParsedElfObject& obj,
                                const Elf64_Sym& sym) {
  if (obj.strtab == nullptr || sym.st_name == 0 ||
      sym.st_name >= obj.strtab_size) {
    return "";
  }
  return absl::string_view(obj.strtab + sym.st_name);
}

absl::Status ApplyX86_64Relocation(ParsedElfObject& obj, uint16_t target_sec,
                                   uint64_t offset, uint64_t sym_val,
                                   uint32_t sym_idx, bool is_external_sym,
                                   uint32_t r_type, int64_t addend) {
  uint8_t* sec_base = obj.section_addrs[target_sec];
  uint8_t* loc = sec_base + offset;
  uint64_t place = reinterpret_cast<uint64_t>(loc);

  switch (r_type) {
    case R_X86_64_NONE:
      return absl::OkStatus();
    case R_X86_64_64: {
      WriteUnaligned<uint64_t>(
          loc, static_cast<uint64_t>(static_cast<int64_t>(sym_val) + addend));
      return absl::OkStatus();
    }
    case R_X86_64_32:
    case R_X86_64_32S: {
      int64_t v = static_cast<int64_t>(sym_val) + addend;
      WriteUnaligned<uint32_t>(loc, static_cast<uint32_t>(v));
      return absl::OkStatus();
    }
    case R_X86_64_PC32: {
      int64_t delta =
          static_cast<int64_t>(sym_val) + addend - static_cast<int64_t>(place);
      if (delta < INT32_MIN || delta > INT32_MAX) {
        return absl::InternalError(absl::StrFormat(
            "R_X86_64_PC32 relocation overflow in '%s'", obj.name));
      }
      WriteUnaligned<int32_t>(loc, static_cast<int32_t>(delta));
      return absl::OkStatus();
    }
    case R_X86_64_PLT32: {
      int64_t direct_delta =
          static_cast<int64_t>(sym_val) + addend - static_cast<int64_t>(place);
      if (!is_external_sym && direct_delta >= INT32_MIN &&
          direct_delta <= INT32_MAX) {
        WriteUnaligned<int32_t>(loc, static_cast<int32_t>(direct_delta));
        return absl::OkStatus();
      }
      uint64_t key = (static_cast<uint64_t>(target_sec) << 32) | sym_idx;
      uint8_t* stub_addr = nullptr;
      if (auto it = obj.stubs.find(key); it != obj.stubs.end()) {
        stub_addr = it->second;
      } else {
        stub_addr = reinterpret_cast<uint8_t*>(AlignUp(
            reinterpret_cast<size_t>(obj.section_stub_cursors[target_sec]), 8));
        obj.section_stub_cursors[target_sec] = stub_addr + 8;
        obj.stubs[key] = stub_addr;

        uint8_t* got_entry = obj.got_cursor;
        obj.got_cursor += 8;
        WriteUnaligned<uint64_t>(got_entry, sym_val);

        // jmp *(%rip + disp32): 0xFF 0x25 <disp32>
        stub_addr[0] = 0xFF;
        stub_addr[1] = 0x25;
        int64_t got_disp = reinterpret_cast<int64_t>(got_entry) -
                           reinterpret_cast<int64_t>(stub_addr + 6);
        WriteUnaligned<int32_t>(stub_addr + 2, static_cast<int32_t>(got_disp));
        stub_addr[6] = 0x90;
        stub_addr[7] = 0x90;
      }
      int64_t stub_delta = reinterpret_cast<int64_t>(stub_addr) + addend -
                           static_cast<int64_t>(place);
      if (stub_delta < INT32_MIN || stub_delta > INT32_MAX) {
        return absl::InternalError(absl::StrFormat(
            "R_X86_64_PLT32 stub displacement overflow in '%s'", obj.name));
      }
      WriteUnaligned<int32_t>(loc, static_cast<int32_t>(stub_delta));
      return absl::OkStatus();
    }
    case R_X86_64_GOTPCREL:
    case R_X86_64_GOTPCRELX:
    case R_X86_64_REX_GOTPCRELX: {
      uint64_t key = (static_cast<uint64_t>(sym_idx) << 32) |
                     static_cast<uint32_t>(sym_val & 0xFFFFFFFFu);
      uint8_t* got_entry = nullptr;
      if (auto it = obj.got_entries.find(key); it != obj.got_entries.end()) {
        got_entry = it->second;
      } else {
        got_entry = obj.got_cursor;
        obj.got_cursor += 8;
        WriteUnaligned<uint64_t>(got_entry, sym_val);
        obj.got_entries[key] = got_entry;
      }
      int64_t got_delta = reinterpret_cast<int64_t>(got_entry) + addend -
                          static_cast<int64_t>(place);
      if (got_delta < INT32_MIN || got_delta > INT32_MAX) {
        return absl::InternalError(
            absl::StrFormat("R_X86_64_GOTPCREL overflow in '%s'", obj.name));
      }
      WriteUnaligned<int32_t>(loc, static_cast<int32_t>(got_delta));
      return absl::OkStatus();
    }
    case R_X86_64_PC64: {
      int64_t delta =
          static_cast<int64_t>(sym_val) + addend - static_cast<int64_t>(place);
      WriteUnaligned<int64_t>(loc, delta);
      return absl::OkStatus();
    }
    case R_X86_64_GOTOFF64: {
      int64_t delta = static_cast<int64_t>(sym_val) -
                      reinterpret_cast<int64_t>(obj.got_base) + addend;
      WriteUnaligned<int64_t>(loc, delta);
      return absl::OkStatus();
    }
    case R_X86_64_8: {
      WriteUnaligned<uint8_t>(loc,
                              static_cast<uint8_t>((sym_val + addend) & 0xFFu));
      return absl::OkStatus();
    }
    case R_X86_64_16: {
      WriteUnaligned<uint16_t>(
          loc, static_cast<uint16_t>((sym_val + addend) & 0xFFFFu));
      return absl::OkStatus();
    }
    case R_X86_64_PC8: {
      int64_t delta =
          static_cast<int64_t>(sym_val) + addend - static_cast<int64_t>(place);
      WriteUnaligned<int8_t>(loc, static_cast<int8_t>(delta & 0xFF));
      return absl::OkStatus();
    }
    default:
      return absl::InternalError(
          absl::StrFormat("Unsupported x86_64 ELF relocation type %d in '%s'",
                          r_type, obj.name));
  }
}

absl::Status ApplyAArch64Relocation(ParsedElfObject& obj, uint16_t target_sec,
                                    uint64_t offset, uint64_t sym_val,
                                    uint32_t sym_idx, uint32_t r_type,
                                    int64_t addend) {
  uint8_t* sec_base = obj.section_addrs[target_sec];
  uint8_t* loc = sec_base + offset;
  uint64_t place = reinterpret_cast<uint64_t>(loc);
  uint64_t target =
      static_cast<uint64_t>(static_cast<int64_t>(sym_val) + addend);

  constexpr uint32_t kElfAArch64Plt32 = 314;
  switch (r_type) {
    case R_AARCH64_NONE:
      return absl::OkStatus();
    case R_AARCH64_ABS16:
      WriteUnaligned<uint16_t>(loc, static_cast<uint16_t>(target & 0xFFFFu));
      return absl::OkStatus();
    case R_AARCH64_ABS32:
      WriteUnaligned<uint32_t>(loc,
                               static_cast<uint32_t>(target & 0xFFFFFFFFu));
      return absl::OkStatus();
    case R_AARCH64_ABS64:
      WriteUnaligned<uint64_t>(loc, target);
      return absl::OkStatus();
    case R_AARCH64_PREL16:
      WriteUnaligned<uint16_t>(
          loc, static_cast<uint16_t>((target - place) & 0xFFFFu));
      return absl::OkStatus();
    case kElfAArch64Plt32:
    case R_AARCH64_PREL32:
      WriteUnaligned<uint32_t>(
          loc, static_cast<uint32_t>((target - place) & 0xFFFFFFFFu));
      return absl::OkStatus();
    case R_AARCH64_PREL64:
      WriteUnaligned<uint64_t>(loc, target - place);
      return absl::OkStatus();
    case R_AARCH64_CALL26:
    case R_AARCH64_JUMP26: {
      int64_t branch_imm =
          static_cast<int64_t>(target) - static_cast<int64_t>(place);
      if (branch_imm < -(1LL << 27) || branch_imm >= (1LL << 27)) {
        uint64_t key = (static_cast<uint64_t>(target_sec) << 32) | sym_idx;
        uint8_t* stub_addr = nullptr;
        if (auto it = obj.stubs.find(key); it != obj.stubs.end()) {
          stub_addr = it->second;
        } else {
          stub_addr = reinterpret_cast<uint8_t*>(AlignUp(
              reinterpret_cast<size_t>(obj.section_stub_cursors[target_sec]),
              8));
          obj.section_stub_cursors[target_sec] = stub_addr + 20;
          obj.stubs[key] = stub_addr;
          // Emit AArch64 indirect branch stub via x16:
          // movz x16, #g3, lsl #48
          // movk x16, #g2, lsl #32
          // movk x16, #g1, lsl #16
          // movk x16, #g0
          // br x16
          uint32_t movz_g3 = 0xD2E00010u | static_cast<uint32_t>(
                                               ((target >> 48) & 0xFFFFu) << 5);
          uint32_t movk_g2 = 0xF2C00010u | static_cast<uint32_t>(
                                               ((target >> 32) & 0xFFFFu) << 5);
          uint32_t movk_g1 = 0xF2A00010u | static_cast<uint32_t>(
                                               ((target >> 16) & 0xFFFFu) << 5);
          uint32_t movk_g0 =
              0xF2800010u | static_cast<uint32_t>((target & 0xFFFFu) << 5);
          uint32_t br_x16 = 0xD61F0200u;
          WriteUnaligned<uint32_t>(stub_addr + 0, movz_g3);
          WriteUnaligned<uint32_t>(stub_addr + 4, movk_g2);
          WriteUnaligned<uint32_t>(stub_addr + 8, movk_g1);
          WriteUnaligned<uint32_t>(stub_addr + 12, movk_g0);
          WriteUnaligned<uint32_t>(stub_addr + 16, br_x16);
        }
        branch_imm =
            reinterpret_cast<int64_t>(stub_addr) - static_cast<int64_t>(place);
      }
      Or32LE(loc, static_cast<uint32_t>((branch_imm & 0x0FFFFFFC) >> 2));
      return absl::OkStatus();
    }
    case R_AARCH64_ADR_PREL_PG_HI21: {
      uint64_t result = (target & ~0xFFFULL) - (place & ~0xFFFULL);
      Write32AArch64Addr(loc, result >> 12);
      return absl::OkStatus();
    }
    case R_AARCH64_ADD_ABS_LO12_NC:
    case R_AARCH64_LDST8_ABS_LO12_NC:
      Or32AArch64Imm(loc, GetBits(target, 0, 11));
      return absl::OkStatus();
    case R_AARCH64_LDST16_ABS_LO12_NC:
      Or32AArch64Imm(loc, GetBits(target, 1, 11));
      return absl::OkStatus();
    case R_AARCH64_LDST32_ABS_LO12_NC:
      Or32AArch64Imm(loc, GetBits(target, 2, 11));
      return absl::OkStatus();
    case R_AARCH64_LDST64_ABS_LO12_NC:
      Or32AArch64Imm(loc, GetBits(target, 3, 11));
      return absl::OkStatus();
    case R_AARCH64_LDST128_ABS_LO12_NC:
      Or32AArch64Imm(loc, GetBits(target, 4, 11));
      return absl::OkStatus();
    case R_AARCH64_ADR_GOT_PAGE: {
      uint64_t key = (static_cast<uint64_t>(sym_idx) << 32) |
                     static_cast<uint32_t>(target & 0xFFFFFFFFu);
      uint8_t* got_entry = nullptr;
      if (auto it = obj.got_entries.find(key); it != obj.got_entries.end()) {
        got_entry = it->second;
      } else {
        got_entry = obj.got_cursor;
        obj.got_cursor += 8;
        WriteUnaligned<uint64_t>(got_entry, target);
        obj.got_entries[key] = got_entry;
      }
      uint64_t got_addr = reinterpret_cast<uint64_t>(got_entry);
      uint64_t result = (got_addr & ~0xFFFULL) - (place & ~0xFFFULL);
      Write32AArch64Addr(loc, result >> 12);
      return absl::OkStatus();
    }
    case R_AARCH64_LD64_GOT_LO12_NC: {
      uint64_t key = (static_cast<uint64_t>(sym_idx) << 32) |
                     static_cast<uint32_t>(target & 0xFFFFFFFFu);
      uint8_t* got_entry = nullptr;
      if (auto it = obj.got_entries.find(key); it != obj.got_entries.end()) {
        got_entry = it->second;
      } else {
        got_entry = obj.got_cursor;
        obj.got_cursor += 8;
        WriteUnaligned<uint64_t>(got_entry, target);
        obj.got_entries[key] = got_entry;
      }
      uint64_t got_addr = reinterpret_cast<uint64_t>(got_entry);
      Or32AArch64Imm(loc, GetBits(got_addr, 3, 11));
      return absl::OkStatus();
    }
    case R_AARCH64_MOVW_UABS_G3:
      Or32LE(loc, static_cast<uint32_t>(((target >> 48) & 0xFFFFu) << 5));
      return absl::OkStatus();
    case R_AARCH64_MOVW_UABS_G2_NC:
      Or32LE(loc, static_cast<uint32_t>(((target >> 32) & 0xFFFFu) << 5));
      return absl::OkStatus();
    case R_AARCH64_MOVW_UABS_G1_NC:
      Or32LE(loc, static_cast<uint32_t>(((target >> 16) & 0xFFFFu) << 5));
      return absl::OkStatus();
    case R_AARCH64_MOVW_UABS_G0_NC:
      Or32LE(loc, static_cast<uint32_t>((target & 0xFFFFu) << 5));
      return absl::OkStatus();
    default:
      return absl::InternalError(
          absl::StrFormat("Unsupported AArch64 ELF relocation type %d in '%s'",
                          r_type, obj.name));
  }
}

}  // namespace

AotCompiledFunctionLibrary::MappedMemory::MappedMemory(
    MappedMemory&& other) noexcept
    : base(other.base), size(other.size) {
  other.base = nullptr;
  other.size = 0;
}

AotCompiledFunctionLibrary::MappedMemory&
AotCompiledFunctionLibrary::MappedMemory::operator=(
    MappedMemory&& other) noexcept {
  if (this != &other) {
    if (base != nullptr && size > 0) {
      munmap(base, size);
    }
    base = other.base;
    size = other.size;
    other.base = nullptr;
    other.size = 0;
  }
  return *this;
}

AotCompiledFunctionLibrary::MappedMemory::~MappedMemory() {
  if (base != nullptr && size > 0) {
    munmap(base, size);
  }
}

AotCompiledFunctionLibrary::AotCompiledFunctionLibrary(
    absl::flat_hash_map<std::string, FunctionPtr> symbols_map,
    std::vector<MappedMemory> mapped_memories)
    : symbols_map_(std::move(symbols_map)),
      mapped_memories_(std::move(mapped_memories)) {}

absl::StatusOr<void*> AotCompiledFunctionLibrary::ResolveFunction(
    TypeId type_id, absl::string_view name) {
  if (auto it = symbols_map_.find(name); it != symbols_map_.end()) {
    // NOTE(basioli) there is no type checking here.
    return it->second;
  }
  return absl::Status(absl::StatusCode::kNotFound,
                      absl::StrFormat("Function %s not found (type id: %d)",
                                      name, type_id.value()));
}

AotObjectLoader::AotObjectLoader(SymbolResolver external_symbol_resolver)
    : external_symbol_resolver_(std::move(external_symbol_resolver)) {}

absl::Status AotObjectLoader::AddObjFile(absl::string_view obj_file,
                                         absl::string_view memory_buffer_name,
                                         size_t dylib_index) {
  (void)dylib_index;
  if (obj_file.empty()) {
    return absl::InvalidArgumentError("Object file contents are empty");
  }
  obj_files_.push_back(
      {std::string(obj_file), std::string(memory_buffer_name)});
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<FunctionLibrary>> AotObjectLoader::Load(
    absl::Span<const Symbol> symbols) && {
  if (obj_files_.empty()) {
    absl::flat_hash_map<std::string, AotCompiledFunctionLibrary::FunctionPtr>
        empty_symbols;
    return std::make_unique<AotCompiledFunctionLibrary>(
        std::move(empty_symbols));
  }

  std::vector<ParsedElfObject> parsed_objs(obj_files_.size());
  for (size_t i = 0; i < obj_files_.size(); ++i) {
    absl::Status status = ParseElfHeader(obj_files_[i].contents,
                                         obj_files_[i].name, parsed_objs[i]);
    if (!status.ok()) {
      return status;
    }
  }

  const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  size_t code_size = 0;
  size_t rodata_size = 0;
  size_t rwdata_size = 0;
  size_t code_align = page_size;
  size_t rodata_align = page_size;
  size_t rwdata_align = page_size;

  for (const auto& obj : parsed_objs) {
    size_t total_relocs = 0;
    for (uint16_t i = 0; i < obj.ehdr->e_shnum; ++i) {
      const Elf64_Shdr& sh = obj.shdrs[i];
      if (sh.sh_type == SHT_RELA && sh.sh_entsize == sizeof(Elf64_Rela)) {
        total_relocs += sh.sh_size / sizeof(Elf64_Rela);
      }
    }

    for (uint16_t i = 0; i < obj.ehdr->e_shnum; ++i) {
      const Elf64_Shdr& sh = obj.shdrs[i];
      if ((sh.sh_flags & SHF_ALLOC) == 0) continue;
      size_t align = std::max<size_t>(sh.sh_addralign, 16);
      if (sh.sh_flags & SHF_EXECINSTR) {
        code_align = std::max(code_align, align);
        code_size = AlignUp(code_size, align) + sh.sh_size + total_relocs * 24;
      } else if (sh.sh_flags & SHF_WRITE) {
        rwdata_align = std::max(rwdata_align, align);
        rwdata_size = AlignUp(rwdata_size, align) + sh.sh_size;
      } else {
        rodata_align = std::max(rodata_align, align);
        rodata_size = AlignUp(rodata_size, align) + sh.sh_size;
      }
    }
    // Reserve GOT entries for each object in rodata (before mprotect).
    rodata_size = AlignUp(rodata_size, 8) + (total_relocs + 1) * 8;
  }

  code_size = AlignUp(std::max<size_t>(code_size, page_size), page_size);
  rodata_size = AlignUp(std::max<size_t>(rodata_size, page_size), page_size);
  rwdata_size = AlignUp(std::max<size_t>(rwdata_size, page_size), page_size);

  const size_t total_alloc_size = code_size + rodata_size + rwdata_size +
                                  code_align + rodata_align + rwdata_align;

  void* raw_map = mmap(nullptr, total_alloc_size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw_map == MAP_FAILED) {
    return absl::InternalError(absl::StrFormat(
        "mmap(%d bytes) failed for AOT ELF loader", total_alloc_size));
  }
  AotCompiledFunctionLibrary::MappedMemory mapped_memory(raw_map,
                                                         total_alloc_size);

  uintptr_t base_addr = reinterpret_cast<uintptr_t>(raw_map);
  uintptr_t code_block_start = AlignUp(base_addr, code_align);
  uintptr_t rodata_block_start =
      AlignUp(code_block_start + code_size, rodata_align);
  uintptr_t rwdata_block_start =
      AlignUp(rodata_block_start + rodata_size, rwdata_align);

  uintptr_t code_cursor = code_block_start;
  uintptr_t rodata_cursor = rodata_block_start;
  uintptr_t rwdata_cursor = rwdata_block_start;

  absl::flat_hash_map<std::string, void*> global_symbols;

  // Allocate and copy sections for all objects.
  for (auto& obj : parsed_objs) {
    size_t total_relocs = 0;
    for (uint16_t i = 0; i < obj.ehdr->e_shnum; ++i) {
      const Elf64_Shdr& sh = obj.shdrs[i];
      if (sh.sh_type == SHT_RELA && sh.sh_entsize == sizeof(Elf64_Rela)) {
        total_relocs += sh.sh_size / sizeof(Elf64_Rela);
      }
    }

    for (uint16_t i = 0; i < obj.ehdr->e_shnum; ++i) {
      const Elf64_Shdr& sh = obj.shdrs[i];
      if ((sh.sh_flags & SHF_ALLOC) == 0) continue;
      size_t align = std::max<size_t>(sh.sh_addralign, 16);
      uint8_t* dst = nullptr;
      if (sh.sh_flags & SHF_EXECINSTR) {
        code_cursor = AlignUp(code_cursor, align);
        dst = reinterpret_cast<uint8_t*>(code_cursor);
        obj.section_stub_cursors[i] = dst + sh.sh_size;
        code_cursor += sh.sh_size + total_relocs * 24;
      } else if (sh.sh_flags & SHF_WRITE) {
        rwdata_cursor = AlignUp(rwdata_cursor, align);
        dst = reinterpret_cast<uint8_t*>(rwdata_cursor);
        rwdata_cursor += sh.sh_size;
      } else {
        rodata_cursor = AlignUp(rodata_cursor, align);
        dst = reinterpret_cast<uint8_t*>(rodata_cursor);
        rodata_cursor += sh.sh_size;
      }
      obj.section_addrs[i] = dst;
      if (sh.sh_type != SHT_NOBITS && sh.sh_size > 0) {
        std::memcpy(dst, obj.data.data() + sh.sh_offset, sh.sh_size);
      } else if (sh.sh_type == SHT_NOBITS && sh.sh_size > 0) {
        std::memset(dst, 0, sh.sh_size);
      }
    }

    rodata_cursor = AlignUp(rodata_cursor, 8);
    obj.got_base = reinterpret_cast<uint8_t*>(rodata_cursor);
    obj.got_cursor = obj.got_base;
    rodata_cursor += (total_relocs + 1) * 8;

    // Collect defined symbols in this object.
    for (size_t s = 1; s < obj.num_syms; ++s) {
      const Elf64_Sym& sym = obj.symtab[s];
      if (sym.st_shndx == SHN_UNDEF) continue;
      void* sym_addr = nullptr;
      if (sym.st_shndx == SHN_ABS) {
        sym_addr =
            reinterpret_cast<void*>(static_cast<uintptr_t>(sym.st_value));
      } else if (sym.st_shndx < obj.section_addrs.size() &&
                 obj.section_addrs[sym.st_shndx] != nullptr) {
        sym_addr = obj.section_addrs[sym.st_shndx] + sym.st_value;
      }
      if (sym_addr != nullptr) {
        absl::string_view sym_name = GetSymbolName(obj, sym);
        if (!sym_name.empty()) {
          uint8_t bind = ELF64_ST_BIND(sym.st_info);
          if (bind == STB_GLOBAL || !global_symbols.contains(sym_name)) {
            global_symbols[std::string(sym_name)] = sym_addr;
          }
        }
      }
    }
  }

  // Apply relocations for all objects.
  for (auto& obj : parsed_objs) {
    for (uint16_t i = 0; i < obj.ehdr->e_shnum; ++i) {
      const Elf64_Shdr& rel_sh = obj.shdrs[i];
      if (rel_sh.sh_type != SHT_RELA) continue;
      if (rel_sh.sh_info >= obj.ehdr->e_shnum) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Invalid sh_info on SHT_RELA section in '%s'", obj.name));
      }
      uint16_t target_sec = static_cast<uint16_t>(rel_sh.sh_info);
      if (obj.section_addrs[target_sec] == nullptr) {
        // Relocation applies to a non-SHF_ALLOC section (e.g. .eh_frame /
        // .debug_*); skip.
        continue;
      }
      const auto* relas = reinterpret_cast<const Elf64_Rela*>(obj.data.data() +
                                                              rel_sh.sh_offset);
      size_t num_relas = rel_sh.sh_size / sizeof(Elf64_Rela);

      for (size_t r = 0; r < num_relas; ++r) {
        const Elf64_Rela& rela = relas[r];
        uint32_t sym_idx = ELF64_R_SYM(rela.r_info);
        uint32_t r_type = ELF64_R_TYPE(rela.r_info);
        int64_t addend = rela.r_addend;

        uint64_t sym_val = 0;
        bool is_external_sym = false;
        if (sym_idx != 0) {
          if (sym_idx >= obj.num_syms) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Relocation references out-of-range symbol %d in '%s'", sym_idx,
                obj.name));
          }
          const Elf64_Sym& sym = obj.symtab[sym_idx];
          if (sym.st_shndx == SHN_UNDEF) {
            is_external_sym = true;
            absl::string_view sym_name = GetSymbolName(obj, sym);
            void* resolved = nullptr;
            if (auto it = global_symbols.find(sym_name);
                it != global_symbols.end()) {
              resolved = it->second;
            } else {
              resolved = LookupBuiltinOrProcessSymbol(
                  sym_name, external_symbol_resolver_);
            }
            if (resolved == nullptr) {
              return absl::InternalError(absl::StrFormat(
                  "Unresolved external symbol '%s' in ELF object '%s'",
                  sym_name, obj.name));
            }
            sym_val = reinterpret_cast<uint64_t>(resolved);
          } else if (sym.st_shndx == SHN_ABS) {
            sym_val = sym.st_value;
          } else if (sym.st_shndx < obj.section_addrs.size() &&
                     obj.section_addrs[sym.st_shndx] != nullptr) {
            sym_val = reinterpret_cast<uint64_t>(
                obj.section_addrs[sym.st_shndx] + sym.st_value);
          } else {
            return absl::InternalError(absl::StrFormat(
                "Relocation references unallocated section %d in '%s'",
                sym.st_shndx, obj.name));
          }
        }

        absl::Status rel_status;
        if (obj.ehdr->e_machine == EM_X86_64) {
          rel_status =
              ApplyX86_64Relocation(obj, target_sec, rela.r_offset, sym_val,
                                    sym_idx, is_external_sym, r_type, addend);
        } else if (obj.ehdr->e_machine == EM_AARCH64) {
          rel_status = ApplyAArch64Relocation(obj, target_sec, rela.r_offset,
                                              sym_val, sym_idx, r_type, addend);
        } else {
          return absl::InternalError(
              absl::StrFormat("Unsupported ELF machine architecture %d in '%s'",
                              obj.ehdr->e_machine, obj.name));
        }
        if (!rel_status.ok()) {
          return rel_status;
        }
      }
    }
  }

  // Finalize memory permissions and flush instruction cache.
  if (mprotect(reinterpret_cast<void*>(code_block_start), code_size,
               PROT_READ | PROT_EXEC) != 0) {
    return absl::InternalError("mprotect(PROT_READ | PROT_EXEC) failed");
  }
  if (mprotect(reinterpret_cast<void*>(rodata_block_start), rodata_size,
               PROT_READ) != 0) {
    return absl::InternalError("mprotect(PROT_READ) failed");
  }
  __builtin___clear_cache(
      reinterpret_cast<char*>(code_block_start),
      reinterpret_cast<char*>(code_block_start + code_size));

  absl::flat_hash_map<std::string, AotCompiledFunctionLibrary::FunctionPtr>
      resolved_map;
  for (const auto& symbol : symbols) {
    auto it = global_symbols.find(symbol.name);
    if (it == global_symbols.end()) {
      // Also check with leading underscore stripped/added if needed.
      std::string alt_name = absl::StrFormat("_%s", symbol.name);
      it = global_symbols.find(alt_name);
    }
    if (it == global_symbols.end()) {
      return absl::NotFoundError(absl::StrFormat(
          "Symbol '%s' not found in loaded AOT ELF objects", symbol.name));
    }
    resolved_map[symbol.name] = it->second;
  }

  std::vector<AotCompiledFunctionLibrary::MappedMemory> memories;
  memories.push_back(std::move(mapped_memory));
  return std::make_unique<AotCompiledFunctionLibrary>(std::move(resolved_map),
                                                      std::move(memories));
}

absl::StatusOr<std::unique_ptr<FunctionLibrary>>
AotObjectLoader::LoadFunctionLibrary(absl::Span<const Symbol> symbols,
                                     absl::Span<const std::string> obj_files,
                                     SymbolResolver external_symbol_resolver) {
  AotObjectLoader loader(std::move(external_symbol_resolver));
  for (size_t i = 0; i < obj_files.size(); ++i) {
    absl::Status status = loader.AddObjFile(obj_files[i]);
    if (!status.ok()) {
      return status;
    }
  }
  return std::move(loader).Load(symbols);
}

}  // namespace xla::cpu
