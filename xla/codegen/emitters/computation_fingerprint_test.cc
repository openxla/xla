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

#include "xla/codegen/emitters/computation_fingerprint.h"

#include <gtest/gtest.h>

#include <memory>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"

namespace xla::emitters {
namespace {

using ComputationFingerprintTest = HloHardwareIndependentTestBase;

TEST_F(ComputationFingerprintTest, ClonedFusionsHaveEqualFingerprints) {
  static constexpr absl::string_view kHlo = R"(
    HloModule m

    fused_a {
      p0 = f32[16] parameter(0)
      c0 = f32[16] constant({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
      ROOT add_a = f32[16] add(p0, c0), metadata={op_name="op_a"}, origin={{"orig_a"}}
    }

    fused_b {
      x0 = f32[16] parameter(0)
      k0 = f32[16] constant({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
      ROOT add_b = f32[16] add(x0, k0), metadata={op_name="op_b"}, origin={{"orig_b"}}
    }

    ENTRY main {
      a = f32[16] parameter(0)
      b = f32[16] parameter(1)
      f0 = f32[16] fusion(a), kind=kLoop, calls=fused_a
      f1 = f32[16] fusion(b), kind=kLoop, calls=fused_b
      ROOT out = (f32[16], f32[16]) tuple(f0, f1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloComputation* fused_a = module->GetComputationWithName("fused_a");
  const HloComputation* fused_b = module->GetComputationWithName("fused_b");
  ASSERT_NE(fused_a, nullptr);
  ASSERT_NE(fused_b, nullptr);

  EXPECT_EQ(GetComputationFingerprint(fused_a, {}),
            GetComputationFingerprint(fused_b, {}));
  EXPECT_NE(GetComputationFingerprint(fused_a, {}, "disc_a"),
            GetComputationFingerprint(fused_b, {}, "disc_b"));
}

TEST_F(ComputationFingerprintTest,
       DistinctLargeArrayConstantsProduceDistinctFingerprints) {
  static constexpr absl::string_view kHlo = R"(
    HloModule m

    fused_a {
      p0 = f32[16] parameter(0)
      c0 = f32[16] constant({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
      ROOT add_a = f32[16] add(p0, c0)
    }

    fused_b {
      p0 = f32[16] parameter(0)
      c0 = f32[16] constant({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17})
      ROOT add_b = f32[16] add(p0, c0)
    }

    ENTRY main {
      a = f32[16] parameter(0)
      b = f32[16] parameter(1)
      f0 = f32[16] fusion(a), kind=kLoop, calls=fused_a
      f1 = f32[16] fusion(b), kind=kLoop, calls=fused_b
      ROOT out = (f32[16], f32[16]) tuple(f0, f1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloComputation* fused_a = module->GetComputationWithName("fused_a");
  const HloComputation* fused_b = module->GetComputationWithName("fused_b");
  ASSERT_NE(fused_a, nullptr);
  ASSERT_NE(fused_b, nullptr);

  EXPECT_NE(GetComputationFingerprint(fused_a, {}),
            GetComputationFingerprint(fused_b, {}));
}

TEST_F(ComputationFingerprintTest,
       DistinctInstructionBackendConfigsProduceDistinctFingerprints) {
  static constexpr absl::string_view kHlo = R"(
    HloModule m

    fused_a {
      p0 = f32[16,32] parameter(0)
      p1 = f32[32,16] parameter(1)
      ROOT d = f32[16,16] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0},
        backend_config={"sizes":["16"]}
    }

    fused_b {
      p0 = f32[16,32] parameter(0)
      p1 = f32[32,16] parameter(1)
      ROOT d = f32[16,16] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0},
        backend_config={"sizes":["32"]}
    }

    ENTRY main {
      a = f32[16,32] parameter(0)
      b = f32[32,16] parameter(1)
      f0 = f32[16,16] fusion(a, b), kind=kLoop, calls=fused_a
      f1 = f32[16,16] fusion(a, b), kind=kLoop, calls=fused_b
      ROOT out = (f32[16,16], f32[16,16]) tuple(f0, f1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloComputation* fused_a = module->GetComputationWithName("fused_a");
  const HloComputation* fused_b = module->GetComputationWithName("fused_b");
  ASSERT_NE(fused_a, nullptr);
  ASSERT_NE(fused_b, nullptr);

  EXPECT_NE(GetComputationFingerprint(fused_a, {}),
            GetComputationFingerprint(fused_b, {}));
}

TEST_F(ComputationFingerprintTest,
       JsonBackendConfigKeyOrderDoesNotAffectFingerprint) {
  static constexpr absl::string_view kHlo = R"(
    HloModule m

    fused_a {
      p0 = f32[16] parameter(0)
      ROOT r = f32[16] negate(p0), backend_config={"a":1,"b":2}
    }

    fused_b {
      p0 = f32[16] parameter(0)
      ROOT r = f32[16] negate(p0), backend_config={"b":2,"a":1}
    }

    ENTRY main {
      a = f32[16] parameter(0)
      f0 = f32[16] fusion(a), kind=kLoop, calls=fused_a
      f1 = f32[16] fusion(a), kind=kLoop, calls=fused_b
      ROOT out = (f32[16], f32[16]) tuple(f0, f1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const HloComputation* fused_a = module->GetComputationWithName("fused_a");
  const HloComputation* fused_b = module->GetComputationWithName("fused_b");
  ASSERT_NE(fused_a, nullptr);
  ASSERT_NE(fused_b, nullptr);

  EXPECT_EQ(GetComputationFingerprint(fused_a, {}),
            GetComputationFingerprint(fused_b, {}));
}

}  // namespace
}  // namespace xla::emitters
