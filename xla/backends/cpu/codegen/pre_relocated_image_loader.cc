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

#include "xla/backends/cpu/codegen/pre_relocated_image_loader.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/aot_compiled_function_library.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/service/cpu/executable.pb.h"

// Anonymous executable memory is the only platform-specific part of the loader,
// so we keep the platform conditional confined to these wrappers.
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>

namespace xla::cpu {
namespace {

absl::StatusOr<void*> MapMemory(size_t size) {
  void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (addr == MAP_FAILED) {
    return absl::ResourceExhaustedError(
        absl::StrFormat("Failed to map %d bytes for pre-relocated image: %s",
                        size, std::strerror(errno)));
  }
  return addr;
}

void UnmapMemory(void* addr, size_t size) { munmap(addr, size); }

// Makes `[addr, addr + size)` read-only, and executable if `executable`.
absl::Status ProtectMemory(void* addr, size_t size, bool executable) {
  int prot = executable ? PROT_READ | PROT_EXEC : PROT_READ;
  if (size == 0 || mprotect(addr, size, prot) == 0) {
    return absl::OkStatus();
  }
  return absl::InternalError(
      absl::StrFormat("mprotect failed: %s", std::strerror(errno)));
}

void FlushInstructionCache(void* addr, size_t size) {
  __builtin___clear_cache(static_cast<char*>(addr),
                          static_cast<char*>(addr) + size);
}

}  // namespace
}  // namespace xla::cpu

#else

namespace xla::cpu {
namespace {

absl::StatusOr<void*> MapMemory(size_t) {
  return absl::UnimplementedError(
      "Loading pre-relocated images is only supported on POSIX hosts");
}

void UnmapMemory(void*, size_t) {}

absl::Status ProtectMemory(void*, size_t, bool) {
  return absl::UnimplementedError(
      "Loading pre-relocated images is only supported on POSIX hosts");
}

void FlushInstructionCache(void*, size_t) {}

}  // namespace
}  // namespace xla::cpu

#endif

namespace xla::cpu {
namespace {

constexpr uint64_t kAlignment = kPreRelocatedImageSegmentAlignment;

absl::Status CheckSlot(uint64_t offset, uint64_t image_size,
                       absl::string_view kind) {
  if (image_size < sizeof(uint64_t) || offset > image_size - sizeof(uint64_t)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s fixup at offset %d is out of bounds of the image "
                        "(size %d)",
                        kind, offset, image_size));
  }
  return absl::OkStatus();
}

// Validates the image layout and returns the total image size in memory.
absl::StatusOr<uint64_t> ValidateImage(const PreRelocatedImageProto& image) {
  if (image.rx_size() % kAlignment != 0 || image.ro_size() % kAlignment != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "RX and RO segment sizes must be multiples of %d, got %d and %d",
        kAlignment, image.rx_size(), image.ro_size()));
  }
  if (image.rx_bytes().size() > image.rx_size() ||
      image.ro_bytes().size() > image.ro_size() ||
      image.rw_bytes().size() > image.rw_mem_size()) {
    return absl::InvalidArgumentError(
        "Segment contents are larger than the segment sizes");
  }

  // Leave room for the alignment padding we add when mapping the image.
  constexpr uint64_t kMaxImageSize =
      std::numeric_limits<size_t>::max() - kAlignment;
  if (image.rx_size() > kMaxImageSize ||
      image.ro_size() > kMaxImageSize - image.rx_size() ||
      image.rw_mem_size() > kMaxImageSize - image.rx_size() - image.ro_size()) {
    return absl::InvalidArgumentError("Image is too large");
  }
  uint64_t image_size = image.rx_size() + image.ro_size() + image.rw_mem_size();
  if (image_size == 0) {
    return absl::InvalidArgumentError("Image is empty");
  }

  // Fixups may also target RX: e.g. x86-64 objects compiled with the large
  // code model store 64-bit absolute addresses as `movabs` immediates. We patch
  // them before the RX segment becomes executable.
  for (const auto& fixup : image.external_fixups()) {
    ABSL_RETURN_IF_ERROR(
        CheckSlot(fixup.image_offset(), image_size, "External"));
  }
  for (const auto& fixup : image.relative_fixups()) {
    ABSL_RETURN_IF_ERROR(
        CheckSlot(fixup.image_offset(), image_size, "Relative"));
  }
  for (const auto& symbol : image.exported_symbols()) {
    if (symbol.image_offset() >= image_size) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Exported symbol %s at offset %d is out of bounds of the image "
          "(size %d)",
          symbol.name(), symbol.image_offset(), image_size));
    }
  }
  return image_size;
}

// Owns an anonymous memory mapping.
class MappedMemory {
 public:
  MappedMemory(void* addr, size_t size) : addr_(addr), size_(size) {}

  MappedMemory(MappedMemory&& other)
      : addr_(std::exchange(other.addr_, nullptr)),
        size_(std::exchange(other.size_, 0)) {}
  MappedMemory& operator=(MappedMemory&&) = delete;

  ~MappedMemory() {
    if (addr_ != nullptr) {
      UnmapMemory(addr_, size_);
    }
  }

 private:
  void* addr_;
  size_t size_;
};

// An `AotCompiledFunctionLibrary` that keeps the mapped image alive.
class PreRelocatedFunctionLibrary final : public AotCompiledFunctionLibrary {
 public:
  PreRelocatedFunctionLibrary(
      MappedMemory memory,
      absl::flat_hash_map<std::string, FunctionPtr> symbols_map)
      : AotCompiledFunctionLibrary(std::move(symbols_map)),
        memory_(std::move(memory)) {}

 private:
  MappedMemory memory_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<FunctionLibrary>> LoadPreRelocatedImage(
    const PreRelocatedImageProto& image,
    absl::FunctionRef<void*(absl::string_view)> symbol_resolver) {
  ABSL_ASSIGN_OR_RETURN(uint64_t image_size, ValidateImage(image));

  // Over-allocate so we can align the image base to 64 KiB, which keeps AArch64
  // ADRP page deltas resolved at compile time valid.
  size_t mapping_size = image_size + kAlignment;
  ABSL_ASSIGN_OR_RETURN(void* mapping, MapMemory(mapping_size));
  MappedMemory memory(mapping, mapping_size);

  uintptr_t aligned = (reinterpret_cast<uintptr_t>(mapping) + kAlignment - 1) &
                      ~(kAlignment - 1);
  uint8_t* base = reinterpret_cast<uint8_t*>(aligned);
  uint8_t* ro = base + image.rx_size();
  uint8_t* rw = ro + image.ro_size();

  // Anonymous mappings are zero-initialized, which covers .bss and padding.
  std::memcpy(base, image.rx_bytes().data(), image.rx_bytes().size());
  std::memcpy(ro, image.ro_bytes().data(), image.ro_bytes().size());
  std::memcpy(rw, image.rw_bytes().data(), image.rw_bytes().size());

  for (const auto& fixup : image.external_fixups()) {
    void* addr = symbol_resolver(fixup.symbol_name());
    if (addr == nullptr) {
      return absl::NotFoundError(absl::StrFormat(
          "Failed to resolve external symbol %s", fixup.symbol_name()));
    }
    uint64_t value = reinterpret_cast<uint64_t>(addr) +
                     static_cast<uint64_t>(fixup.addend());
    std::memcpy(base + fixup.image_offset(), &value, sizeof(value));
  }

  for (const auto& fixup : image.relative_fixups()) {
    uint64_t value =
        reinterpret_cast<uint64_t>(base) + fixup.target_image_offset();
    std::memcpy(base + fixup.image_offset(), &value, sizeof(value));
  }

  ABSL_RETURN_IF_ERROR(
      ProtectMemory(base, image.rx_size(), /*executable=*/true));
  ABSL_RETURN_IF_ERROR(
      ProtectMemory(ro, image.ro_size(), /*executable=*/false));
  FlushInstructionCache(base, image.rx_size());

  absl::flat_hash_map<std::string, AotCompiledFunctionLibrary::FunctionPtr>
      symbols_map;
  for (const auto& symbol : image.exported_symbols()) {
    symbols_map[symbol.name()] = base + symbol.image_offset();
  }

  return std::make_unique<PreRelocatedFunctionLibrary>(std::move(memory),
                                                       std::move(symbols_map));
}

}  // namespace xla::cpu
