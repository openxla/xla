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

#ifndef XLA_STREAM_EXECUTOR_SYCL_ONEAPI_COMPUTE_CAPABILITY_H_
#define XLA_STREAM_EXECUTOR_SYCL_ONEAPI_COMPUTE_CAPABILITY_H_

#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.pb.h"

namespace stream_executor {

enum class XeGeneration : uint32_t {
  kUnknown = 0,
  kXe = 12,
  kXe2 = 20,
  kXe3 = 30,
  kXe3P = 35
};

// Known Xe architectures as (architecture, generation, min_version,
// max_version) rows. A row matches a device of the given generation whose
// release version lies in the inclusive range [min_version, max_version]. The
// architecture name is the string used by architecture(), ToString() and the
// proto.
//
// Use EMIT_ARCH_DEFINE once per arch (it defines the XeArch enumerator) and
// EMIT_ARCH_EXTEND for additional version ranges of the same arch. When
// inferring the arch, the first row whose generation matches and whose range
// contains the version is selected.
#define FOR_KNOWN_ONEAPI_ARCHS(EMIT_ARCH_DEFINE, EMIT_ARCH_EXTEND) \
  EMIT_ARCH_DEFINE(XeLP, Xe, 0, 4)        /* TGL, RKL, ADL */      \
  EMIT_ARCH_EXTEND(XeLP, Xe, 10, 10)      /* DG1 */                \
  EMIT_ARCH_DEFINE(XeHPG, Xe, 55, 57)     /* DG2 */                \
  EMIT_ARCH_DEFINE(XeHPC, Xe, 60, 61)     /* PVC */                \
  EMIT_ARCH_DEFINE(XeLPG, Xe, 70, 71)     /* MTL */                \
  EMIT_ARCH_DEFINE(XeLPGPlus, Xe, 74, 74) /* ARL-H */              \
  /* Add new Xe rows above this. */                                \
  EMIT_ARCH_DEFINE(XeOther, Xe, 0, 0xff) /* Other Xe */            \
  EMIT_ARCH_DEFINE(Xe2LPG, Xe2, 4, 4)    /* LNL */                 \
  EMIT_ARCH_DEFINE(Xe2HPG, Xe2, 1, 2)    /* BMG */                 \
  /* Add new Xe2 rows above this. */                               \
  EMIT_ARCH_DEFINE(Xe2Other, Xe2, 0, 0xff) /* Other Xe2 */         \
  EMIT_ARCH_DEFINE(Xe3, Xe3, 0, 0xff)      /* PTL, WCL, NVL-S/U */ \
  EMIT_ARCH_DEFINE(Xe3PLPG, Xe3P, 10, 10)  /* NVL-P */             \
  EMIT_ARCH_DEFINE(Xe3PXPC, Xe3P, 11, 11)  /* CRI */               \
  /* Add new Xe3P rows above this. */                              \
  EMIT_ARCH_DEFINE(Xe3POther, Xe3P, 0, 0xff) /* Other Xe3P */

#define ARCH_SKIP(...)
#define ARCH_ENTRIES(ARCH, ...) k##ARCH,
#define ARCH_INFO(ARCH, GEN, MIN, MAX) \
  {#ARCH, XeArch::k##ARCH, XeGeneration::k##GEN, MIN, MAX},

// Convenience utilities for each known arch. For example, IsXeLPG()
// checks whether this capability corresponds to an XeLPG device.
#define ARCH_UTILITIES(ARCH, ...) \
  bool Is##ARCH() const { return arch_ == XeArch::k##ARCH; }

// Only EMIT_ARCH_DEFINE rows become enumerators, so each name appears once.
enum class XeArch : uint8_t {
  FOR_KNOWN_ONEAPI_ARCHS(ARCH_ENTRIES, ARCH_SKIP) kUnknown
};

class OneAPIComputeCapability {
 public:
  OneAPIComputeCapability() = default;

  explicit OneAPIComputeCapability(XeGeneration generation, uint8_t version);

  explicit OneAPIComputeCapability(const OneAPIComputeCapabilityProto& proto)
      : OneAPIComputeCapability(FromProto(proto)) {}

  // The IP version can be queried via Level Zero. Its 10 most significant bits
  // hold the Xe generation and the next 8 bits hold the release version. The
  // pair selects the Xe architecture from the FOR_KNOWN_ONEAPI_ARCHS table.
  explicit OneAPIComputeCapability(const uint32_t ip_version)
      : OneAPIComputeCapability(
            static_cast<XeGeneration>((ip_version >> 22) & 0x3ff),
            static_cast<uint8_t>((ip_version >> 14) & 0xff)) {}

  explicit OneAPIComputeCapability(absl::string_view arch);

  std::string architecture() const;

  std::string ToString() const;

  OneAPIComputeCapabilityProto ToProto() const;

  static OneAPIComputeCapability FromProto(
      const OneAPIComputeCapabilityProto& proto);

  FOR_KNOWN_ONEAPI_ARCHS(ARCH_UTILITIES, ARCH_SKIP)

  static OneAPIComputeCapability BMG() {
    return OneAPIComputeCapability("Xe2HPG");
  }
  static OneAPIComputeCapability PVC() {
    return OneAPIComputeCapability("XeHPC");
  }
  static OneAPIComputeCapability DG2() {
    return OneAPIComputeCapability("XeHPG");
  }
  static OneAPIComputeCapability CRI() {
    return OneAPIComputeCapability("Xe3PXPC");
  }

  bool operator==(const OneAPIComputeCapability& other) const {
    return arch_ == other.arch_;
  }

  bool operator!=(const OneAPIComputeCapability& other) const {
    return !this->operator==(other);
  }

 private:
  struct OneAPIArchInfo {
    const char* name;
    XeArch arch;
    XeGeneration generation;
    uint8_t min_version;
    uint8_t max_version;
  };

  static constexpr OneAPIArchInfo kXeArchs[]{
      FOR_KNOWN_ONEAPI_ARCHS(ARCH_INFO, ARCH_INFO)};

  // Maps a (generation, version) pair to an XeArch.
  static XeArch InferDeviceArch(XeGeneration generation, uint8_t version);

  XeGeneration generation_ = XeGeneration::kUnknown;
  XeArch arch_ = XeArch::kUnknown;
};

#undef ARCH_UTILITIES
#undef ARCH_SKIP
#undef ARCH_ENTRIES
#undef ARCH_INFO
#undef FOR_KNOWN_ONEAPI_ARCHS

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_ONEAPI_COMPUTE_CAPABILITY_H_
