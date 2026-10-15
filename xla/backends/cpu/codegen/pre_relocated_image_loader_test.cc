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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/service/cpu/executable.pb.h"

namespace xla::cpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

constexpr uint64_t k64KiB = 64 * 1024;

// Offsets of the 8-byte slots read by the test functions.
constexpr uint64_t kExternalSlot = k64KiB;      // first RO slot
constexpr uint64_t kRelativeSlot = 2 * k64KiB;  // first RW slot
constexpr uint64_t kBssSlot = 2 * k64KiB + 64;  // past rw_bytes

int external_value = 0;

void PutU32(std::string& bytes, uint64_t offset, uint32_t value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

// Emits a function at `func_offset` that returns the 8-byte value stored at
// image offset `slot_offset`, encoded for the host architecture. Returns false
// if the host architecture isn't supported by this test.
bool EmitLoadSlotFunction(std::string& rx, uint64_t func_offset,
                          uint64_t slot_offset) {
  llvm::Triple::ArchType arch =
      llvm::Triple(llvm::sys::getProcessTriple()).getArch();

  if (arch == llvm::Triple::x86_64) {
    // mov rax, qword ptr [rip + disp32]; ret
    int32_t disp = static_cast<int32_t>(slot_offset - (func_offset + 7));
    rx[func_offset + 0] = '\x48';
    rx[func_offset + 1] = '\x8b';
    rx[func_offset + 2] = '\x05';
    PutU32(rx, func_offset + 3, static_cast<uint32_t>(disp));
    rx[func_offset + 7] = '\xc3';
    return true;
  }

  if (arch == llvm::Triple::aarch64) {
    // adrp x0, slot; ldr x0, [x0, :lo12:slot]; ret
    int64_t pages = static_cast<int64_t>((slot_offset & ~0xFFFull) -
                                         (func_offset & ~0xFFFull)) >>
                    12;
    uint32_t adrp =
        0x90000000u | ((pages & 0x3) << 29) | (((pages >> 2) & 0x7FFFF) << 5);
    uint32_t ldr = 0xF9400000u | (((slot_offset & 0xFFF) / 8) << 10);
    PutU32(rx, func_offset + 0, adrp);
    PutU32(rx, func_offset + 4, ldr);
    PutU32(rx, func_offset + 8, 0xD65F03C0u);
    return true;
  }

  return false;
}

// Builds an image with three functions that read an external fixup slot (RO),
// a relative fixup slot pointing at function `f0` (RW), and a .bss slot.
PreRelocatedImageProto CreateTestImage(bool& supported) {
  PreRelocatedImageProto image;
  image.set_rx_size(k64KiB);
  image.set_ro_size(k64KiB);
  image.set_rw_mem_size(k64KiB);

  std::string rx(64, '\0');
  supported = EmitLoadSlotFunction(rx, 0, kExternalSlot) &&
              EmitLoadSlotFunction(rx, 16, kRelativeSlot) &&
              EmitLoadSlotFunction(rx, 32, kBssSlot);
  image.set_rx_bytes(rx);
  image.set_ro_bytes(std::string(8, '\0'));
  image.set_rw_bytes(std::string(16, '\x5a'));

  auto* external = image.add_external_fixups();
  external->set_image_offset(kExternalSlot);
  external->set_symbol_name("external_value");
  external->set_addend(4);

  auto* relative = image.add_relative_fixups();
  relative->set_image_offset(kRelativeSlot);
  relative->set_target_image_offset(0);

  const std::pair<absl::string_view, uint64_t> exports[] = {
      {"f0", 0}, {"f1", 16}, {"f2", 32}};
  for (const auto& [name, offset] : exports) {
    auto* symbol = image.add_exported_symbols();
    symbol->set_name(name);
    symbol->set_image_offset(offset);
  }
  return image;
}

void* TestResolver(absl::string_view name) {
  return name == "external_value" ? &external_value : nullptr;
}

TEST(PreRelocatedImageLoaderTest, LoadsAndPatchesImage) {
  bool supported = false;
  PreRelocatedImageProto image = CreateTestImage(supported);
  if (!supported) {
    GTEST_SKIP() << "Test machine code is not available for this host.";
  }

  auto library = LoadPreRelocatedImage(image, TestResolver);
  ASSERT_THAT(library, IsOk());

  using Fn = uint64_t (*)();
  auto resolve = [&](absl::string_view name) -> Fn {
    auto fn = (*library)->ResolveFunction<uint64_t()>(name);
    EXPECT_THAT(fn, IsOk());
    return fn.value_or(nullptr);
  };

  Fn f0 = resolve("f0");
  Fn f1 = resolve("f1");
  Fn f2 = resolve("f2");

  EXPECT_EQ(f0(), reinterpret_cast<uint64_t>(&external_value) + 4);
  EXPECT_EQ(f1(), reinterpret_cast<uint64_t>(f0));
  EXPECT_EQ(f2(), 0);

  EXPECT_EQ(reinterpret_cast<uint64_t>(f0) % k64KiB, 0);
}

TEST(PreRelocatedImageLoaderTest, RejectsUnalignedSegments) {
  bool supported = false;
  PreRelocatedImageProto image = CreateTestImage(supported);
  image.set_rx_size(4096);
  EXPECT_THAT(LoadPreRelocatedImage(image, TestResolver),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PreRelocatedImageLoaderTest, RejectsOversizedSegmentBytes) {
  bool supported = false;
  PreRelocatedImageProto image = CreateTestImage(supported);
  image.set_rw_mem_size(8);
  EXPECT_THAT(LoadPreRelocatedImage(image, TestResolver),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PreRelocatedImageLoaderTest, RejectsOutOfBoundsFixups) {
  bool supported = false;
  PreRelocatedImageProto image = CreateTestImage(supported);
  image.mutable_relative_fixups(0)->set_image_offset(3 * k64KiB - 4);
  EXPECT_THAT(LoadPreRelocatedImage(image, TestResolver),
              StatusIs(absl::StatusCode::kInvalidArgument));

  image = CreateTestImage(supported);
  image.mutable_external_fixups(0)->set_image_offset(~0ull - 2);
  EXPECT_THAT(LoadPreRelocatedImage(image, TestResolver),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PreRelocatedImageLoaderTest, RejectsOutOfBoundsExportedSymbols) {
  bool supported = false;
  PreRelocatedImageProto image = CreateTestImage(supported);
  image.mutable_exported_symbols(0)->set_image_offset(3 * k64KiB);
  EXPECT_THAT(LoadPreRelocatedImage(image, TestResolver),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PreRelocatedImageLoaderTest, FailsOnUnresolvedSymbol) {
  bool supported = false;
  PreRelocatedImageProto image = CreateTestImage(supported);
  image.mutable_external_fixups(0)->set_symbol_name("does_not_exist");
  EXPECT_THAT(
      LoadPreRelocatedImage(image, TestResolver),
      StatusIs(absl::StatusCode::kNotFound, HasSubstr("does_not_exist")));
}

}  // namespace
}  // namespace xla::cpu
