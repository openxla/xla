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

// AOT compatibility of data-movement collective thunks using symmetric memory
// (ncclPutSignal) on 2 GPUs.

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/literal.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/aot_compatibility/gpu/collective_aot_test_base.h"
#include "xla/tests/aot_compatibility/test_lib.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"

namespace xla {
namespace aot_compatibility {
namespace {

using ::testing::ValuesIn;

constexpr absl::string_view kTargetBase =
    "data_movement_collectives_aot_symmetric_test_2gpu";
constexpr int64_t kNumReplicas = 2;

constexpr absl::string_view kDataMovementHlo = R"hlo(
  HloModule test
  ENTRY main {
    id = u32[] replica-id()
    one = u32[] constant(1)
    id1 = u32[] add(id, one)
    v0 = u32[2] broadcast(id1), dimensions={}
    ag = u32[4] all-gather(v0), dimensions={0}, replica_groups={{0,1}}

    ten = u32[] constant(10)
    base = u32[] multiply(id, ten)
    b4 = u32[4] broadcast(base), dimensions={}
    iota4 = u32[4] iota(), iota_dimension=0
    in4 = u32[4] add(b4, iota4)
    a2a_split = u32[4] all-to-all(in4), dimensions={0}, replica_groups={{0,1}}

    b2 = u32[2] broadcast(base), dimensions={}
    c12 = u32[2] constant({1, 2})
    c34 = u32[2] constant({3, 4})
    t0_in = u32[2] add(b2, c12)
    t1_in = u32[2] add(b2, c34)
    a2a_tuple = (u32[2], u32[2]) all-to-all(t0_in, t1_in),
                                 replica_groups={{0,1}}
    a2a_t0 = u32[2] get-tuple-element(a2a_tuple), index=0
    a2a_t1 = u32[2] get-tuple-element(a2a_tuple), index=1

    cb_static = u32[2] collective-broadcast(t0_in), replica_groups={{1, 0}}
    root_idx = s32[1] constant({1})
    cb_dyn = u32[2] collective-broadcast(t1_in, root_idx),
                    replica_groups={{0, 1}}, has_dynamic_root=true
    ROOT out = (u32[4], u32[4], u32[2], u32[2], u32[2], u32[2])
               tuple(ag, a2a_split, a2a_t0, a2a_t1, cb_static, cb_dyn)
  }
)hlo";

void ExpectDataMovementResults(std::vector<Literal> results) {
  std::vector<Literal> r0 = results[0].DecomposeTuple();
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1, 2, 2}, r0[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({0, 1, 10, 11}, r0[1]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 2}, r0[2]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 12}, r0[3]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 12}, r0[4]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({13, 14}, r0[5]);

  std::vector<Literal> r1 = results[1].DecomposeTuple();
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1, 2, 2}, r1[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({2, 3, 12, 13}, r1[1]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({3, 4}, r1[2]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({13, 14}, r1[3]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 12}, r1[4]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({13, 14}, r1[5]);
}

class DataMovementCollectivesAotSymmetricTest : public CollectiveAotTestBase {};

// COLLECTIVES_SYMMETRIC_MEMORY all-gather and use_symmetric_buffer on every
// collective of the module.
TEST_P(DataMovementCollectivesAotSymmetricTest, SymmetricMemory) {
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  DebugOptions& debug_options = config.mutable_debug_options();
  debug_options.set_xla_gpu_all_gather_mode(
      DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY);
  EnableNcclSymmetricBuffers(debug_options);

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kDataMovementHlo, config));
  ExpectDataMovementResults(std::move(results));
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility,
                         DataMovementCollectivesAotSymmetricTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification,
                         DataMovementCollectivesAotSymmetricTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
