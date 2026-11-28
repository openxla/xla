/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/collective_permute_decomposer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/named_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/collective_conflict_analysis.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/hlo_verifier.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::NotNull;

using Pass = CollectivePermuteDecomposer;

struct Decomposed {
  std::string cp_name;
  HloInstruction* after_all;
  HloInstruction* send;
  HloInstruction* recv;
  HloInstruction* send_done;
  HloInstruction* recv_done;
};

class DecomposerTest : public HloHardwareIndependentTestBase {
 protected:
  Decomposed FindComponents(HloModule* module, absl::string_view cp_name) {
    Decomposed result;
    result.cp_name = cp_name;
    result.after_all =
        FindInstruction(module, absl::StrCat(cp_name, "-after-all"));
    result.send = FindInstruction(module, absl::StrCat(cp_name, "-send"));
    result.recv = FindInstruction(module, absl::StrCat(cp_name, "-recv"));
    result.send_done =
        FindInstruction(module, absl::StrCat(cp_name, "-send-done"));
    result.recv_done =
        FindInstruction(module, absl::StrCat(cp_name, "-recv-done"));
    CHECK(result.after_all != nullptr) << cp_name;
    CHECK(result.send != nullptr) << cp_name;
    CHECK(result.recv != nullptr) << cp_name;
    CHECK(result.send_done != nullptr) << cp_name;
    CHECK(result.recv_done != nullptr) << cp_name;
    return result;
  }
};

const char* kSimpleHloWhileLoopTemplate = R"(
  HloModule module
  cond {
    param = (u32[], f32[64]) parameter(0)
    i = get-tuple-element(param), index=0
    n = u32[] constant(2)
    ROOT result = pred[] compare(i, n), direction=LT
  }

  $hlo_while_body

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c42 = f32[] constant(42.0)
    init = f32[64] broadcast(c42), dimensions={}
    while_init = (u32[], f32[64]) tuple(c0, init)
    while_result = (u32[], f32[64]) while(while_init), body=body, condition=cond
    ROOT result = f32[64] get-tuple-element(while_result), index=1
  }
)";

static std::string GetSimpleHloWhileLoopStr(absl::string_view hlo_while_body) {
  return absl::StrReplaceAll(kSimpleHloWhileLoopTemplate,
                             {
                                 {"$hlo_while_body", hlo_while_body},
                             });
}

TEST_F(DecomposerTest, WithCycleNotTransformed) {
  std::string hlo = GetSimpleHloWhileLoopStr(R"(
  body {
    param = (u32[], f32[64]) parameter(0)
    i = get-tuple-element(param), index=0
    data = get-tuple-element(param), index=1
    cp = f32[64] collective-permute(data), channel_id=1,
        source_target_pairs={{0,1}, {1,0}}
    ROOT result = tuple(i, cp)
  }
  )");
  ASSERT_OK(RunAndCheckHloRewrite(
      hlo,
      Pass(/*threshold_in_bytes=*/0,
           DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
      false));
}

TEST_F(DecomposerTest, ThresholdNotTransformed) {
  const int64_t kThreshold = 64 * 8;
  std::string hlo = GetSimpleHloWhileLoopStr(R"(
  body {
    param = (u32[], f32[64]) parameter(0)
    i = get-tuple-element(param), index=0
    data = get-tuple-element(param), index=1
    cp = f32[64] collective-permute(data),
        source_target_pairs={{0,1}, {1,2}, {2,3}}
    ROOT result = tuple(i, cp)
  }
  )");
  ASSERT_OK(RunAndCheckHloRewrite(
      hlo,
      Pass(/*threshold_in_bytes=*/kThreshold,
           DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE),
      false));
}

TEST_F(DecomposerTest, ThresholdIgnoredWhenPipelineParallelismOptEnabled) {
  const int64_t kThreshold = 64 * 8;
  std::string hlo = GetSimpleHloWhileLoopStr(R"(
  body {
    param = (u32[], f32[64]) parameter(0)
    i = get-tuple-element(param), index=0
    data = get-tuple-element(param), index=1
    cp = f32[64] collective-permute(data),
        source_target_pairs={{0,1}, {1,2}, {2,3}}
    ROOT result = tuple(i, cp)
  }
  )");
  ASSERT_OK(RunAndCheckHloRewrite(
      hlo,
      Pass(/*threshold_in_bytes=*/kThreshold,
           DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
      true));
}

TEST_F(DecomposerTest, Basic) {
  std::string hlo = GetSimpleHloWhileLoopStr(R"(
  body {
    param = (u32[], f32[64]) parameter(0)
    i = get-tuple-element(param), index=0
    data = get-tuple-element(param), index=1
    cp = f32[64] collective-permute(data), channel_id=1,
        source_target_pairs={{0,1}, {1,2}}
    ROOT result = tuple(i, cp)
  }
  )");
  ASSERT_OK(RunAndCheckHloRewrite(
      hlo,
      Pass(/*threshold_in_bytes=*/0,
           DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
      true));
}

TEST_F(DecomposerTest, NoChannelId) {
  std::string hlo = GetSimpleHloWhileLoopStr(R"(
  body {
    param = (u32[], f32[64]) parameter(0)
    i = get-tuple-element(param), index=0
    data = get-tuple-element(param), index=1
    cp = f32[64] collective-permute(data), source_target_pairs={{0,1}, {1,2}}
    ROOT result = tuple(i, cp)
  }
  )");
  ASSERT_OK(RunAndCheckHloRewrite(
      hlo,
      Pass(/*threshold_in_bytes=*/0,
           DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
      true));
}

TEST_F(DecomposerTest, OutsideOfWhileLoop) {
  std::string hlo = R"(HloModule test
    ENTRY test_computation {
      data = u32[] parameter(0)
      ROOT cp = u32[] collective-permute(data), channel_id=1,
          source_target_pairs={{0,1}, {1,2}}
    })";
  ASSERT_OK(RunAndCheckHloRewrite(
      hlo,
      Pass(/*threshold_in_bytes=*/0,
           DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
      false));
}

TEST_F(DecomposerTest, ControlDependency_IndependentCPs) {
  absl::string_view hlo = R"(
    HloModule test

    cond {
      param = (u32[], f32[64], f32[64], f32[64]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      n = u32[] constant(2)
      ROOT result = pred[] compare(i, n), direction=LT
    }

    body {
      param = (u32[], f32[64], f32[64], f32[64]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      data1 = f32[64] get-tuple-element(param), index=1
      data2 = f32[64] get-tuple-element(param), index=2
      cp3 = f32[64] collective-permute(data2), source_target_pairs={{6,7}}
      cp1 = f32[64] collective-permute(data1), source_target_pairs={{3,0}}
      cp2 = f32[64] collective-permute(data2),
          source_target_pairs={{0,1},{1,2},{2,3}}
      ROOT out = (u32[], f32[64], f32[64], f32[64]) tuple(i, cp2, cp3, cp1)
    }

    ENTRY test_computation {
      c0 = u32[] constant(0)
      c42 = f32[] constant(42.0)
      init = f32[64] broadcast(c42), dimensions={}
      while_init = (u32[], f32[64], f32[64], f32[64])
          tuple(c0, init, init, init)
      ROOT while_result = (u32[], f32[64], f32[64], f32[64]) while(while_init),
          body=body, condition=cond
    })";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      RunAndCheckHloRewrite(
          hlo,
          Pass(/*threshold_in_bytes=*/0,
               DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
          true));
  // cp1 and cp3 won't get decomposed.
  HloInstruction* cp1 = FindInstruction(module.get(), "cp1");
  HloInstruction* cp3 = FindInstruction(module.get(), "cp3");
  ASSERT_THAT(cp1, NotNull());
  ASSERT_THAT(cp3, NotNull());
  EXPECT_EQ(cp1->opcode(), HloOpcode::kCollectivePermute);
  EXPECT_EQ(cp3->opcode(), HloOpcode::kCollectivePermute);
  Decomposed cp2 = FindComponents(module.get(), "cp2");
  // Sequence in tuple determines the port order and therefore control
  // dependency of consecutive CPs.
  EXPECT_THAT(cp1->control_predecessors(), ElementsAre(cp2.send_done));
}

TEST_F(DecomposerTest, ControlDependency_BasicDependency) {
  const std::string kHlo = GetSimpleHloWhileLoopStr(R"(
    body {
      param = (u32[], f32[64]) parameter(0)
      i = get-tuple-element(param), index=0
      data = get-tuple-element(param), index=1
      cp-a = f32[64] collective-permute(data),
          source_target_pairs={{0,1}, {1,2}, {2,3}}
      cp-b = f32[64] collective-permute(cp-a), source_target_pairs={{3,0}}
      ROOT result = (u32[], f32[64]) tuple(i, cp-b)
    }
  )");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      RunAndCheckHloRewrite(
          kHlo,
          Pass(/*threshold_in_bytes=*/0,
               DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
          true));
  Decomposed cp_a = FindComponents(module.get(), "cp-a");
  EXPECT_NE(FindInstruction(module.get(), "cp-b"), nullptr);
}

TEST_F(DecomposerTest, ControlDependency_MoreDependencies) {
  std::string kHlo = GetSimpleHloWhileLoopStr(R"(
    body {
      param = (u32[], f32[64]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      data = f32[64] get-tuple-element(param), index=1

      // misordered names to assure that dependencies are honored
      cp1 = f32[64] collective-permute(data), source_target_pairs={{3,0}}
      cp2 = f32[64] collective-permute(cp1),
          source_target_pairs={{0,1},{1,2},{2,3}}
      cp3 = f32[64] collective-permute(cp2), source_target_pairs={{6,7}}
      ROOT out = (u32[], f32[64]) tuple(i, cp3)
    })");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      RunAndCheckHloRewrite(
          kHlo,
          Pass(/*threshold_in_bytes=*/0,
               DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
          true));
  Decomposed cp1 = FindComponents(module.get(), "cp1");
  EXPECT_NE(FindInstruction(module.get(), "cp2"), nullptr);
  EXPECT_NE(FindInstruction(module.get(), "cp3"), nullptr);
}

void EnsurePreservedInfo(const HloInstruction* instr) {
  SCOPED_TRACE("AssurePreservedInfo for: " + instr->ToString());
  EXPECT_EQ(instr->channel_id().value(), 1);
  EXPECT_EQ(instr->metadata().op_name(), "op1/op2/add");
  EXPECT_EQ(instr->metadata().source_file(), "foo/bar/mysource.py");
  EXPECT_EQ(instr->metadata().source_line(), 35);
  EXPECT_THAT(
      instr->ToString(),
      HasSubstr(
          "_xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3},{3,4}}"));
}

std::string PipelineAttr(const HloInstruction* instr) {
  return instr->get_frontend_attribute(kSendRecvPipelineAttr).value_or("");
}
std::string OtherAttr(const HloInstruction* instr) {
  const FrontendAttributes& attributes = instr->frontend_attributes();
  return attributes.map().find("_xla_other_attribute")->second;
}

void EnsurePipelineAttr(Decomposed cp, std::string val) {
  SCOPED_TRACE("ExpectePipelineAttr for " + cp.cp_name);
  EXPECT_EQ(PipelineAttr(cp.recv), val);
  EXPECT_EQ(PipelineAttr(cp.send), val);
  EXPECT_EQ(PipelineAttr(cp.recv_done), val);
  EXPECT_EQ(PipelineAttr(cp.send_done), val);
}

void EnsureControlDependency(Decomposed cp) {
  SCOPED_TRACE("ExpectOpControlDependency for " + cp.cp_name);
  EXPECT_EQ(cp.recv->operand(0), cp.after_all);
  EXPECT_EQ(cp.send->operand(1), cp.after_all);
  EXPECT_EQ(cp.recv_done->operand(0), cp.recv);
  EXPECT_EQ(cp.send_done->operand(0), cp.send);
  EXPECT_TRUE(absl::c_find_if(
      cp.send->control_predecessors(),
      [&cp](const HloInstruction* it) { return it == cp.recv; }))
      << "Send should depend on recv when decoposed";
  EXPECT_TRUE(absl::c_find_if(
      cp.recv_done->control_predecessors(),
      [&cp](const HloInstruction* it) { return it == cp.send; }))
      << "Recv-done should depend on send when decoposed";
}

TEST_F(DecomposerTest, StructureAndMetadata) {
  std::string kHlo = GetSimpleHloWhileLoopStr(R"(
    body {
      param = (u32[], f32[64]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      data = f32[64] get-tuple-element(param), index=1
      cp = f32[64] collective-permute(data), channel_id=1,
        source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}},
        metadata={op_name="op1/op2/add" source_file="foo/bar/mysource.py"
        source_line=35}
      ROOT result = (u32[], f32[64]) tuple(i, cp)
    })");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      RunAndCheckHloRewrite(
          kHlo,
          Pass(/*threshold_in_bytes=*/0,
               DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
          true));
  Decomposed cp = FindComponents(module.get(), "cp");
  EnsurePreservedInfo(cp.send);
  EnsurePreservedInfo(cp.recv);
  EnsureControlDependency(cp);
}

TEST_F(DecomposerTest, Pipeline1) {
  absl::string_view hlo = R"(
    HloModule module

    cond {
      param = (u32[], u32[2]) parameter(0)
      count = get-tuple-element(param), index=0
      ub = u32[] constant(2)
      ROOT result = pred[] compare(count, ub), direction=LT
    }

    body {
      param = (u32[], u32[2]) parameter(0)
      count = get-tuple-element(param), index=0
      send-data = get-tuple-element(param), index=1

      cp = u32[2] collective-permute(send-data), channel_id=1,
        source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}},
        frontend_attributes={_xla_other_attribute="xyz"}

      c1 = u32[] constant(1)
      new_count = u32[] add(count, c1)

      r = u32[2] broadcast(c1), dimensions={}
      s = u32[2] add(r, cp)

      ROOT result = (u32[], u32[2]) tuple(new_count, s)
    }

    ENTRY test_computation {
      c0 = u32[] constant(0)
      c1 = u32[] constant(1)
      r = u32[] replica-id()
      a = u32[] add(c1, r)
      init = u32[2] broadcast(a), dimensions={}
      while_init = (u32[], u32[2]) tuple(c0, init)
      while_result = (u32[], u32[2]) while(while_init), body=body,
          condition=cond
      ROOT result = u32[2] get-tuple-element(while_result), index=1
    })";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      RunAndCheckHloRewrite(
          hlo,
          Pass(/*threshold_in_bytes=*/0,
               DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
          true));
  Decomposed cp = FindComponents(module.get(), "cp");
  EnsurePipelineAttr(cp, "0");
  EXPECT_EQ(OtherAttr(cp.recv), "xyz") << "Preseving other attributes";
  EXPECT_EQ(OtherAttr(cp.send), "xyz") << "Preseving other attributes";
  EnsureControlDependency(cp);
}

TEST_F(DecomposerTest, ForwardPipeline2) {
  absl::string_view hlo = R"(
  HloModule module
  cond {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(2)
    ROOT result = pred[] compare(count, ub), direction=LT
  }

  body {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    send-data = get-tuple-element(param), index=1

    cp_fwd = u32[2] collective-permute(send-data), channel_id=2,
      source_target_pairs={{0,1}, {1,2}, {2,3}}

    cp_back = u32[2] collective-permute(send-data), channel_id=1,
      source_target_pairs={{3,0}}

    replica = u32[] replica-id()
    constant0 = u32[] constant(0)
    compare0 = pred[] compare(replica, constant0), direction=EQ
    compare = pred[2] broadcast(compare0), dimensions={}
    recv-data = u32[2] select(compare, cp_back, cp_fwd)

    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)

    r = u32[2] broadcast(c1), dimensions={}
    s = u32[2] add(r, recv-data)

    ROOT result = (u32[], u32[2]) tuple(new_count, s)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    r = u32[] replica-id()
    a = u32[] add(c1, r)
    init = u32[2] broadcast(a), dimensions={}
    while_init = (u32[], u32[2]) tuple(c0, init)
    while_result = (u32[], u32[2]) while(while_init), body=body, condition=cond
    ROOT result = u32[2] get-tuple-element(while_result), index=1
  })";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      RunAndCheckHloRewrite(
          hlo,
          Pass(/*threshold_in_bytes=*/0,
               DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
          true));
  HloInstruction* cp_back = FindInstruction(module.get(), "cp_back");
  ASSERT_THAT(cp_back, NotNull());
  EXPECT_EQ(cp_back->opcode(), HloOpcode::kCollectivePermute);
  Decomposed cp_fwd = FindComponents(module.get(), "cp_fwd");
  EXPECT_THAT(cp_back->control_predecessors(), ElementsAre(cp_fwd.send_done));
  EXPECT_EQ(cp_fwd.recv->channel_id().value(), 2);
  EnsurePipelineAttr(cp_fwd, "1");
}

TEST_F(DecomposerTest, ForwardPipelineWithMatmul) {
  // The HLO module below is generated by passing the HLO in
  // CollectiveOpsTest.CollectivePermute_CircularPipelinePreOptimization through
  // the collective_permute_cycle_decomposer.transformation.
  absl::string_view hlo = R"(
  HloModule test

  while_body {
    inputs = (u32[], f32[2,2], f32[2,2]) parameter(0)
    iter = u32[] get-tuple-element(inputs), index=0
    iter_increment = u32[] constant(1)
    next_iter = u32[] add(iter, iter_increment)
    partition-id = u32[] partition-id()
    zero = u32[] constant(0)
    compare = pred[] compare(partition-id, zero), direction=EQ
    broadcast = pred[2,2] broadcast(compare), dimensions={}

    weights = f32[2,2] get-tuple-element(inputs), index=2
    data = f32[2,2] get-tuple-element(inputs), index=1

    cp_back = f32[2,2] collective-permute(data), channel_id=1,
      source_target_pairs={{3,0}},
      frontend_attributes={_xla_send_recv_validation="{{3,10}}"}
    cp_fwd = f32[2,2] collective-permute(data), channel_id=2,
      source_target_pairs={{0,1},{1,2},{2,3}},
      frontend_attributes={_xla_send_recv_validation="{{0,7},{1,8},{2,9}}"}

    select = f32[2,2] select(broadcast, cp_back, cp_fwd)

    matmul = f32[2,2] dot(weights, select), lhs_contracting_dims={1},
        rhs_contracting_dims={0}

    ROOT result = (u32[], f32[2,2], f32[2,2]) tuple(next_iter, matmul, weights)
  }

  while_cond {
    inputs = (u32[], f32[2,2], f32[2,2]) parameter(0)
    iter = u32[] get-tuple-element(inputs), index=0
    max_iter = u32[] constant(3)
    ROOT compare = pred[] compare(iter, max_iter), direction=LT
  }

  ENTRY test_computation {
    start_iter = u32[] constant(0)
    input_data = f32[2,2] parameter(0)
    input_weights = f32[2,2] parameter(1)
    input = (u32[], f32[2,2], f32[2,2]) tuple(start_iter, input_data,
        input_weights)
    while_result = (u32[], f32[2,2], f32[2,2]) while(input),
        condition=while_cond, body=while_body
    ROOT data_out = f32[2,2] get-tuple-element(while_result), index=1
  }
  )";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      RunAndCheckHloRewrite(
          hlo,
          Pass(/*threshold_in_bytes=*/0,
               DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
          true));
  HloInstruction* cp_back = FindInstruction(module.get(), "cp_back");
  ASSERT_THAT(cp_back, NotNull());
  EXPECT_EQ(cp_back->opcode(), HloOpcode::kCollectivePermute);
  Decomposed cp_fwd = FindComponents(module.get(), "cp_fwd");
  EXPECT_EQ(cp_fwd.recv->channel_id().value(), 2);
  EXPECT_THAT(cp_back->control_predecessors(), ElementsAre(cp_fwd.send_done));
  EnsurePipelineAttr(cp_fwd, "1");
}

TEST_F(DecomposerTest, BackwardPipeline2) {
  absl::string_view hlo = R"(
  HloModule module
  cond {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(2)
    ROOT result = pred[] compare(count, ub), direction=LT
  }

  body {
    param = (u32[], u32[2]) parameter(0)
    count = get-tuple-element(param), index=0
    send-data = get-tuple-element(param), index=1

    cp_fwd = u32[2] collective-permute(send-data), channel_id=1,
      source_target_pairs={{1,0},{2,1},{3,2}}

    cp_back = u32[2] collective-permute(send-data), channel_id=2,
      source_target_pairs={{0,3}}

    replica = u32[] replica-id()
    constant0 = u32[] constant(0)
    compare0 = pred[] compare(replica, constant0), direction=NE
    compare = pred[2] broadcast(compare0), dimensions={}
    recv-data = u32[2] select(compare, cp_fwd, cp_back)

    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)

    r = u32[2] broadcast(c1), dimensions={}
    s = u32[2] add(r, recv-data)

    ROOT result = (u32[], u32[2]) tuple(new_count, s)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    r = u32[] replica-id()
    a = u32[] add(c1, r)
    init = u32[2] broadcast(a), dimensions={}
    while_init = (u32[], u32[2]) tuple(c0, init)
    while_result = (u32[], u32[2]) while(while_init), body=body, condition=cond
    ROOT result = u32[2] get-tuple-element(while_result), index=1
  })";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      RunAndCheckHloRewrite(
          hlo,
          Pass(/*threshold_in_bytes=*/0,
               DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
          true));
  HloInstruction* cp_back = FindInstruction(module.get(), "cp_back");
  ASSERT_THAT(cp_back, NotNull());
  EXPECT_EQ(cp_back->opcode(), HloOpcode::kCollectivePermute);
  Decomposed cp_fwd = FindComponents(module.get(), "cp_fwd");
  EXPECT_THAT(cp_back->control_predecessors(), ElementsAre(cp_fwd.send_done));
  EXPECT_EQ(cp_fwd.recv->channel_id().value(), 1);
  EnsurePipelineAttr(cp_fwd, "1");
  EnsureControlDependency(cp_fwd);
}

TEST_F(DecomposerTest, OneSendRecvWithOneConflictingCollectivePermute) {
  absl::string_view hlo = R"(
  HloModule module

  cond {
    param = (u32[], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(param), index=0
    n = u32[] constant(2)
    ROOT result = pred[] compare(i, n), direction=LT
  }

  body {
    param = (u32[], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(param), index=0
    data_a = f32[64] get-tuple-element(param), index=1
    data_b = f32[64] get-tuple-element(param), index=2

    // cp_fwd can be decomposed.
    cp_fwd = f32[64] collective-permute(data_a), channel_id=1,
        source_target_pairs={{0,1},{1,2},{2,3}}

    // cp_cycle cannot be decomposed and is conflicting with cp_fwd.
    cp_cycle = f32[64] collective-permute(data_b), channel_id=2,
        source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
    c1 = u32[] constant(1)
    i_ = u32[] add(i, c1)

    ROOT result = (u32[], f32[64], f32[64]) tuple(i_, cp_fwd, cp_cycle)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    a = f32[] constant(42.0)
    data = f32[64] broadcast(a), dimensions={}
    while_init = (u32[], f32[64], f32[64]) tuple(c0, data, data)
    ROOT while_result = (u32[], f32[64], f32[64]) while(while_init), body=body,
        condition=cond
  })";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      RunAndCheckHloRewrite(
          hlo,
          Pass(/*threshold_in_bytes=*/0,
               DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
          true));

  // Expect the resulting send/recv ops to be strictly ordered before the
  // remaining collective-permute. This is to avoid deadlocks where one device
  // is waiting on its send/recv peer while its peer expects to perform a
  // collective-permute.
  HloInstruction* cp_cycle = FindInstruction(module.get(), "cp_cycle");
  HloInstruction* cp_fwd_recv = FindInstruction(module.get(), "cp_fwd-recv");
  HloInstruction* cp_fwd_recv_done =
      FindInstruction(module.get(), "cp_fwd-recv-done");
  HloInstruction* cp_fwd_send = FindInstruction(module.get(), "cp_fwd-send");
  HloInstruction* cp_fwd_send_done =
      FindInstruction(module.get(), "cp_fwd-send-done");
  ASSERT_THAT(cp_cycle, NotNull());
  ASSERT_THAT(cp_fwd_recv, NotNull());
  ASSERT_THAT(cp_fwd_recv_done, NotNull());
  ASSERT_THAT(cp_fwd_send, NotNull());
  ASSERT_THAT(cp_fwd_send_done, NotNull());
  EXPECT_THAT(cp_fwd_send->control_predecessors(),
              ElementsAre(cp_fwd_recv_done));
  EXPECT_THAT(cp_cycle->control_predecessors(), ElementsAre(cp_fwd_send_done));

  // Expect all conflicting collectives to be annotated with the collective
  // stream attribute.
  EXPECT_EQ(
      cp_fwd_recv->frontend_attributes().map().at(kCollectiveStreamAttrName),
      kCollectiveStreamP2P);
  EXPECT_EQ(
      cp_fwd_send->frontend_attributes().map().at(kCollectiveStreamAttrName),
      kCollectiveStreamP2P);
  EXPECT_EQ(cp_cycle->frontend_attributes().map().at(kCollectiveStreamAttrName),
            kCollectiveStreamP2P);
}

TEST_F(DecomposerTest, OneSendRecvWithOneConflictingAllReduce) {
  absl::string_view hlo = R"(
  HloModule module

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  cond {
    param = (u32[], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(param), index=0
    n = u32[] constant(2)
    ROOT result = pred[] compare(i, n), direction=LT
  }

  body {
    param = (u32[], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(param), index=0
    data_a = f32[64] get-tuple-element(param), index=1
    data_b = f32[64] get-tuple-element(param), index=2

    // cp_fwd can be decomposed.
    cp_fwd = f32[64] collective-permute(data_a), channel_id=1,
        source_target_pairs={{0,1},{1,2},{2,3}}

    // ar is conflicting with cp_fwd.
    ar = f32[64] all-reduce(data_b), channel_id=2, replica_groups={{0,1,2,3}},
        to_apply=add

    c1 = u32[] constant(1)
    i_ = u32[] add(i, c1)

    ROOT result = (u32[], f32[64], f32[64]) tuple(i_, cp_fwd, ar)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    a = f32[] constant(42.0)
    data = f32[64] broadcast(a), dimensions={}
    while_init = (u32[], f32[64], f32[64]) tuple(c0, data, data)
    ROOT while_result = (u32[], f32[64], f32[64]) while(while_init), body=body,
        condition=cond
  })";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      RunAndCheckHloRewrite(
          hlo,
          Pass(/*threshold_in_bytes=*/0,
               DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
          true));

  // Expect the resulting send/recv ops to be strictly ordered before the
  // all-reduce. This is to avoid deadlocks where one device is waiting on its
  // send/recv peer while its peer expects to perform an all-reduce.
  HloInstruction* ar = FindInstruction(module.get(), "ar");
  HloInstruction* cp_fwd_recv = FindInstruction(module.get(), "cp_fwd-recv");
  HloInstruction* cp_fwd_recv_done =
      FindInstruction(module.get(), "cp_fwd-recv-done");
  HloInstruction* cp_fwd_send = FindInstruction(module.get(), "cp_fwd-send");
  HloInstruction* cp_fwd_send_done =
      FindInstruction(module.get(), "cp_fwd-send-done");
  ASSERT_THAT(ar, NotNull());
  ASSERT_THAT(cp_fwd_recv, NotNull());
  ASSERT_THAT(cp_fwd_recv_done, NotNull());
  ASSERT_THAT(cp_fwd_send, NotNull());
  ASSERT_THAT(cp_fwd_send_done, NotNull());
  EXPECT_THAT(cp_fwd_send->control_predecessors(),
              ElementsAre(cp_fwd_recv_done));
  EXPECT_THAT(ar->control_predecessors(), ElementsAre(cp_fwd_send_done));

  // Expect all conflicting collectives to be annotated with the collective
  // stream attribute.
  EXPECT_EQ(
      cp_fwd_recv->frontend_attributes().map().at(kCollectiveStreamAttrName),
      kCollectiveStreamP2P);
  EXPECT_EQ(
      cp_fwd_send->frontend_attributes().map().at(kCollectiveStreamAttrName),
      kCollectiveStreamP2P);
  EXPECT_EQ(ar->frontend_attributes().map().at(kCollectiveStreamAttrName),
            kCollectiveStreamP2P);
}

TEST_F(DecomposerTest, OneSendRecvWithConflictingSendRecv) {
  absl::string_view hlo = R"(
  HloModule module

  cond {
    param = (u32[], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(param), index=0
    n = u32[] constant(2)
    ROOT result = pred[] compare(i, n), direction=LT
  }

  body {
    param = (u32[], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(param), index=0
    data_a = f32[64] get-tuple-element(param), index=1
    data_b = f32[64] get-tuple-element(param), index=2

    // cp_fwd can be decomposed.
    cp_fwd = f32[64] collective-permute(data_a), channel_id=1,
        source_target_pairs={{0,1},{1,2},{2,3}}

    // These send/recv ops are conflicting with cp_fwd.
    tok = token[] after-all()
    recv_ctx = (f32[64], u32[], token[]) recv(tok), channel_id=2,
        frontend_attributes={_xla_send_recv_source_target_pairs={{3,0}}}
    recv_done = (f32[64], token[]) recv-done(recv_ctx), channel_id=2
    send_ctx = (f32[64], u32[], token[]) send(tok, data_b), channel_id=2,
        frontend_attributes={_xla_send_recv_source_target_pairs={{3,0}}}
    send_done = token[] send-done(send_ctx), channel_id=2
    recv_data = f32[64] get-tuple-element(recv_done), index=0

    c1 = u32[] constant(1)
    i_ = u32[] add(i, c1)

    ROOT result = (u32[], f32[64], f32[64]) tuple(i_, cp_fwd, recv_data)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    a = f32[] constant(42.0)
    data = f32[64] broadcast(a), dimensions={}
    while_init = (u32[], f32[64], f32[64]) tuple(c0, data, data)
    ROOT while_result = (u32[], f32[64], f32[64]) while(while_init), body=body,
        condition=cond
  })";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      RunAndCheckHloRewrite(
          hlo,
          Pass(/*threshold_in_bytes=*/0,
               DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
          true));

  // Expect decomposed send/recv ops to be strictly ordered before the
  // preexisting send/recv ops. This is to avoid deadlocks where one device is
  // waiting on its decomposed send/recv peer while its peer is stuck in some
  // different send/recv communication.
  HloInstruction* conflicting_recv = FindInstruction(module.get(), "recv_ctx");
  HloInstruction* conflicting_send = FindInstruction(module.get(), "send_ctx");
  HloInstruction* cp_fwd_recv = FindInstruction(module.get(), "cp_fwd-recv");
  HloInstruction* cp_fwd_recv_done =
      FindInstruction(module.get(), "cp_fwd-recv-done");
  HloInstruction* cp_fwd_send = FindInstruction(module.get(), "cp_fwd-send");
  HloInstruction* cp_fwd_send_done =
      FindInstruction(module.get(), "cp_fwd-send-done");
  ASSERT_THAT(conflicting_recv, NotNull());
  ASSERT_THAT(conflicting_send, NotNull());
  ASSERT_THAT(cp_fwd_recv, NotNull());
  ASSERT_THAT(cp_fwd_recv_done, NotNull());
  ASSERT_THAT(cp_fwd_send, NotNull());
  ASSERT_THAT(cp_fwd_send_done, NotNull());
  EXPECT_THAT(cp_fwd_send->control_predecessors(),
              ElementsAre(cp_fwd_recv_done));
  EXPECT_THAT(conflicting_recv->control_predecessors(),
              ElementsAre(cp_fwd_send_done));
  EXPECT_THAT(conflicting_send->control_predecessors(),
              ElementsAre(cp_fwd_send_done));

  // Expect all conflicting collectives to be annotated with the collective
  // stream attribute.
  EXPECT_EQ(
      cp_fwd_recv->frontend_attributes().map().at(kCollectiveStreamAttrName),
      kCollectiveStreamP2P);
  EXPECT_EQ(
      cp_fwd_send->frontend_attributes().map().at(kCollectiveStreamAttrName),
      kCollectiveStreamP2P);
  EXPECT_EQ(conflicting_recv->frontend_attributes().map().at(
                kCollectiveStreamAttrName),
            kCollectiveStreamP2P);
  EXPECT_EQ(conflicting_send->frontend_attributes().map().at(
                kCollectiveStreamAttrName),
            kCollectiveStreamP2P);
}

TEST_F(DecomposerTest, OneSendRecvWithNonConflictingAllReduce) {
  absl::string_view hlo = R"(
  HloModule module

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  cond {
    param = (u32[], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(param), index=0
    n = u32[] constant(2)
    ROOT result = pred[] compare(i, n), direction=LT
  }

  body {
    param = (u32[], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(param), index=0
    data_a = f32[64] get-tuple-element(param), index=1
    data_b = f32[64] get-tuple-element(param), index=2

    // cp_fwd can be decomposed.
    cp_fwd = f32[64] collective-permute(data_a), channel_id=1,
        source_target_pairs={{0,2},{1,3}}

    // ar is non-conflicting with cp_fwd. Cliques overlap in no more than one
    // rank.
    ar = f32[64] all-reduce(data_b), channel_id=2, replica_groups={{0,1},{2,3}},
        to_apply=add
    c1 = u32[] constant(1)
    i_ = u32[] add(i, c1)

    ROOT result = (u32[], f32[64], f32[64]) tuple(i_, cp_fwd, ar)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    a = f32[] constant(42.0)
    data = f32[64] broadcast(a), dimensions={}
    while_init = (u32[], f32[64], f32[64]) tuple(c0, data, data)
    ROOT while_result = (u32[], f32[64], f32[64]) while(while_init), body=body,
        condition=cond
  })";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      RunAndCheckHloRewrite(
          hlo,
          Pass(/*threshold_in_bytes=*/0,
               DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
          true));

  // Expect decomposed send/recv ops to be unordered wrt. to preexisting
  // non-conflicting all-reduce. These collectivves will not deadlock as their
  // NCCL cliques overlap in no more than one rank.
  HloInstruction* ar = FindInstruction(module.get(), "ar");
  HloInstruction* cp_fwd_recv = FindInstruction(module.get(), "cp_fwd-recv");
  HloInstruction* cp_fwd_recv_done =
      FindInstruction(module.get(), "cp_fwd-recv-done");
  HloInstruction* cp_fwd_send = FindInstruction(module.get(), "cp_fwd-send");
  HloInstruction* cp_fwd_send_done =
      FindInstruction(module.get(), "cp_fwd-send-done");
  ASSERT_THAT(ar, NotNull());
  ASSERT_THAT(cp_fwd_recv, NotNull());
  ASSERT_THAT(cp_fwd_recv_done, NotNull());
  ASSERT_THAT(cp_fwd_send, NotNull());
  ASSERT_THAT(cp_fwd_send_done, NotNull());
  EXPECT_THAT(cp_fwd_send->control_predecessors(),
              ElementsAre(cp_fwd_recv_done));
  EXPECT_THAT(ar->control_predecessors(), ElementsAre());

  // Expect all conflicting collectives to be annotated with the collective
  // stream attribute.
  EXPECT_EQ(
      cp_fwd_recv->frontend_attributes().map().at(kCollectiveStreamAttrName),
      kCollectiveStreamP2P);
  EXPECT_EQ(
      cp_fwd_send->frontend_attributes().map().at(kCollectiveStreamAttrName),
      kCollectiveStreamP2P);

  // Expect all non-conflicting collectives to not be annotated with the
  // collective
  // stream attribute.
  EXPECT_FALSE(ar->has_frontend_attribute(kCollectiveStreamAttrName));
}

TEST_F(DecomposerTest, OneSendRecvWithConflictingAndNonConflictingCollectives) {
  absl::string_view hlo = R"(
  HloModule module

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  cond {
    param = (u32[], f32[64], f32[64], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(param), index=0
    n = u32[] constant(2)
    ROOT result = pred[] compare(i, n), direction=LT
  }

  body {
    param = (u32[], f32[64], f32[64], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(param), index=0
    data_a = f32[64] get-tuple-element(param), index=1
    data_b = f32[64] get-tuple-element(param), index=2

    // cp_fwd can be decomposed.
    cp_fwd = f32[64] collective-permute(data_a), channel_id=1,
        source_target_pairs={{0,2},{1,3}}

    // cp_cycle is conflicting with cp_fwd.
    cp_cycle = f32[64] collective-permute(data_b), channel_id=2,
        source_target_pairs={{0,1},{1,2},{2,3},{3,0}}

    // ar is non-conflicting with cp_fwd. Cliques overlap in no more than one
    // rank.
    ar = f32[64] all-reduce(data_b), channel_id=3,
        replica_groups={{0},{1},{2},{3}}, to_apply=add

    // arc is conflicting with cp_fwd.
    arc = f32[64] all-reduce(data_b), channel_id=4, replica_groups={{0,1,2,3}},
        to_apply=add

    c1 = u32[] constant(1)
    i_ = u32[] add(i, c1)

    ROOT result = (u32[], f32[64], f32[64], f32[64], f32[64]) tuple(i_, cp_fwd,
        cp_cycle, ar, arc)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    a = f32[] constant(42.0)
    data = f32[64] broadcast(a), dimensions={}
    while_init = (u32[], f32[64], f32[64], f32[64], f32[64]) tuple(c0, data,
        data, data, data)
    ROOT while_result = (u32[], f32[64], f32[64], f32[64], f32[64])
        while(while_init), body=body, condition=cond
  })";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      RunAndCheckHloRewrite(
          hlo,
          Pass(/*threshold_in_bytes=*/0,
               DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
          true));

  // Expect decomposed send/recv ops to be strictly ordered before the
  // conflicting all-reduce arc and the conflicting collective-permute cp_cycle.
  // Expect them to be unordered wrt. to the non-conflicting all-reduce ar.
  HloInstruction* cp_cycle = FindInstruction(module.get(), "cp_cycle");
  HloInstruction* ar = FindInstruction(module.get(), "ar");
  HloInstruction* arc = FindInstruction(module.get(), "arc");
  HloInstruction* cp_fwd_recv = FindInstruction(module.get(), "cp_fwd-recv");
  HloInstruction* cp_fwd_recv_done =
      FindInstruction(module.get(), "cp_fwd-recv-done");
  HloInstruction* cp_fwd_send = FindInstruction(module.get(), "cp_fwd-send");
  HloInstruction* cp_fwd_send_done =
      FindInstruction(module.get(), "cp_fwd-send-done");
  ASSERT_THAT(cp_fwd_recv, NotNull());
  ASSERT_THAT(cp_fwd_recv_done, NotNull());
  ASSERT_THAT(cp_fwd_send, NotNull());
  ASSERT_THAT(cp_fwd_send_done, NotNull());
  ASSERT_THAT(cp_cycle, NotNull());
  ASSERT_THAT(ar, NotNull());
  ASSERT_THAT(arc, NotNull());
  EXPECT_THAT(cp_fwd_recv->control_predecessors(), ElementsAre());
  EXPECT_THAT(cp_fwd_recv_done->control_predecessors(), ElementsAre());
  EXPECT_THAT(cp_fwd_send->control_predecessors(),
              ElementsAre(cp_fwd_recv_done));
  EXPECT_THAT(cp_fwd_send_done->control_predecessors(), ElementsAre());
  EXPECT_THAT(cp_cycle->control_predecessors(), ElementsAre(cp_fwd_send_done));
  EXPECT_THAT(ar->control_predecessors(), ElementsAre());
  EXPECT_THAT(arc->control_predecessors(), ElementsAre(cp_fwd_send_done));

  // Expect all conflicting collectives to be annotated with the collective
  // stream attribute.
  EXPECT_EQ(
      cp_fwd_recv->frontend_attributes().map().at(kCollectiveStreamAttrName),
      kCollectiveStreamP2P);
  EXPECT_EQ(
      cp_fwd_send->frontend_attributes().map().at(kCollectiveStreamAttrName),
      kCollectiveStreamP2P);
  EXPECT_EQ(cp_cycle->frontend_attributes().map().at(kCollectiveStreamAttrName),
            kCollectiveStreamP2P);
  EXPECT_EQ(arc->frontend_attributes().map().at(kCollectiveStreamAttrName),
            kCollectiveStreamP2P);

  // Expect all non-conflicting collectives to not be annotated with the
  // collective stream attribute.
  EXPECT_FALSE(ar->has_frontend_attribute(kCollectiveStreamAttrName));
}

TEST_F(DecomposerTest, OneSendRecvWithIndirectlyConflictingCollectives) {
  absl::string_view hlo = R"(
  HloModule module

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  cond {
    param = (u32[], f32[64], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(param), index=0
    n = u32[] constant(2)
    ROOT result = pred[] compare(i, n), direction=LT
  }

  body {
    param = (u32[], f32[64], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(param), index=0
    data_a = f32[64] get-tuple-element(param), index=1
    data_b = f32[64] get-tuple-element(param), index=2

    // cp_fwd can be decomposed.
    cp_fwd = f32[64] collective-permute(data_a), channel_id=1,
        source_target_pairs={{0,1},{1,2},{2,3}}

    // These collective-permute ops are conflicting with cp_fwd, some through
    // indirection.
    cp_cycle = f32[64] collective-permute(data_b), channel_id=2,
        source_target_pairs={{4,5},{5,6},{6,7},{7,4}}
    cp_cycle2 = f32[64] collective-permute(data_b), channel_id=3,
        source_target_pairs={{2,3},{3,4},{4,5},{5,2}}

    c1 = u32[] constant(1)
    i_ = u32[] add(i, c1)

    ROOT result = (u32[], f32[64], f32[64], f32[64]) tuple(i_, cp_fwd, cp_cycle,
        cp_cycle2)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    c1 = u32[] constant(1)
    a = f32[] constant(42.0)
    data = f32[64] broadcast(a), dimensions={}
    while_init = (u32[], f32[64], f32[64], f32[64]) tuple(c0, data, data, data)
    ROOT while_result = (u32[], f32[64], f32[64], f32[64]) while(while_init),
        body=body, condition=cond
  })";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      RunAndCheckHloRewrite(
          hlo,
          Pass(/*threshold_in_bytes=*/0,
               DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
          true));

  // Expect send/recv ops to be strictly ordered before the conflicting
  // collective-permute ops. This is to avoid deadlocks where one device is
  // waiting on its send/recv peer while its peer is stuck in a different
  // collective-permute.
  HloInstruction* cp_cycle = FindInstruction(module.get(), "cp_cycle");
  HloInstruction* cp_cycle2 = FindInstruction(module.get(), "cp_cycle2");
  HloInstruction* cp_fwd_recv = FindInstruction(module.get(), "cp_fwd-recv");
  HloInstruction* cp_fwd_recv_done =
      FindInstruction(module.get(), "cp_fwd-recv-done");
  HloInstruction* cp_fwd_send = FindInstruction(module.get(), "cp_fwd-send");
  HloInstruction* cp_fwd_send_done =
      FindInstruction(module.get(), "cp_fwd-send-done");
  ASSERT_THAT(cp_cycle, NotNull());
  ASSERT_THAT(cp_cycle2, NotNull());
  ASSERT_THAT(cp_fwd_recv, NotNull());
  ASSERT_THAT(cp_fwd_recv_done, NotNull());
  ASSERT_THAT(cp_fwd_send, NotNull());
  ASSERT_THAT(cp_fwd_send_done, NotNull());
  ASSERT_THAT(cp_fwd_recv->control_predecessors(), ElementsAre());
  ASSERT_THAT(cp_fwd_recv_done->control_predecessors(), ElementsAre());
  EXPECT_THAT(cp_fwd_send->control_predecessors(),
              ElementsAre(cp_fwd_recv_done));
  EXPECT_THAT(cp_fwd_send_done->control_predecessors(), ElementsAre());
  EXPECT_THAT(cp_cycle->control_predecessors(), ElementsAre(cp_fwd_send_done));
  EXPECT_THAT(cp_cycle2->control_predecessors(), ElementsAre(cp_fwd_send_done));

  // Expect all conflicting collectives to be annotated with the collective
  // stream attribute.
  EXPECT_EQ(
      cp_fwd_recv->frontend_attributes().map().at(kCollectiveStreamAttrName),
      kCollectiveStreamP2P);
  EXPECT_EQ(
      cp_fwd_send->frontend_attributes().map().at(kCollectiveStreamAttrName),
      kCollectiveStreamP2P);
  EXPECT_EQ(cp_cycle->frontend_attributes().map().at(kCollectiveStreamAttrName),
            kCollectiveStreamP2P);
  EXPECT_EQ(
      cp_cycle2->frontend_attributes().map().at(kCollectiveStreamAttrName),
      kCollectiveStreamP2P);
}

TEST_F(DecomposerTest,
       OrdersUpstreamAndSandwichedConflictingCollectivesWithoutCycles) {
  // Upstream conflicting all-reduce feeding cp (DISABLE) and sandwiched
  // cp0 -> ar -> cp1: no HLO cycle, exact topological ordering.
  constexpr absl::string_view kSandwiched = R"(
  HloModule m
  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT s = f32[] add(a, b)
  }
  cond {
    p = (u32[], f32[64]) parameter(0)
    i = u32[] get-tuple-element(p), index=0
    c = u32[] constant(2)
    ROOT cmp = pred[] compare(i, c), direction=LT
  }
  body {
    p = (u32[], f32[64]) parameter(0)
    i = u32[] get-tuple-element(p), index=0
    x = f32[64] get-tuple-element(p), index=1
    cp0 = f32[64] collective-permute(x), channel_id=1,
      source_target_pairs={{0,1},{1,2}}
    ar = f32[64] all-reduce(cp0), channel_id=2, use_global_device_ids=true,
      replica_groups={{0,1,2,3}}, to_apply=add
    cp1 = f32[64] collective-permute(ar), channel_id=3,
      source_target_pairs={{1,2},{2,3}}
    i1 = u32[] add(i, u32[] constant(1))
    ROOT r = (u32[], f32[64]) tuple(i1, cp1)
  }
  ENTRY e {
    z = f32[64] broadcast(f32[] constant(0)), dimensions={}
    init = (u32[], f32[64]) tuple(u32[] constant(0), z)
    ROOT w = (u32[], f32[64]) while(init), body=body, condition=cond
  })";
  ASSERT_OK_AND_ASSIGN(
      auto m, RunAndCheckHloRewrite(
                  kSandwiched,
                  Pass(/*threshold_in_bytes=*/0,
                       DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE),
                  true));
  Decomposed d0 = FindComponents(m.get(), "cp0");
  Decomposed d1 = FindComponents(m.get(), "cp1");
  HloInstruction* ar = FindInstruction(m.get(), "ar");
  ASSERT_NE(ar, nullptr);
  EXPECT_THAT(ar->control_predecessors(),
              ElementsAre(d0.send_done, d0.recv_done));
  EXPECT_THAT(d1.recv->control_predecessors(), ElementsAre(d0.send, ar));

  // Upstream conflicting all-reduce with control edge into cp under ENABLE.
  constexpr absl::string_view kCtrlEnable = R"(
  HloModule m
  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT s = f32[] add(a, b)
  }
  cond {
    p = (u32[], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(p), index=0
    ROOT cmp = pred[] compare(i, u32[] constant(2)), direction=LT
  }
  body {
    p = (u32[], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(p), index=0
    x = f32[64] get-tuple-element(p), index=1
    y = f32[64] get-tuple-element(p), index=2
    ar = f32[64] all-reduce(y), channel_id=2, use_global_device_ids=true,
      replica_groups={{0,1,2,3}}, to_apply=add
    cp = f32[64] collective-permute(x), channel_id=1,
      source_target_pairs={{0,1},{1,2}}, control-predecessors={ar}
    ROOT r = (u32[], f32[64], f32[64]) tuple(
      u32[] add(i, u32[] constant(1)), cp, ar)
  }
  ENTRY e {
    z = f32[64] broadcast(f32[] constant(0)), dimensions={}
    ROOT w = (u32[], f32[64], f32[64]) while(
      (u32[], f32[64], f32[64]) tuple(u32[] constant(0), z, z)),
      body=body, condition=cond
  })";
  ASSERT_OK_AND_ASSIGN(
      auto me, RunAndCheckHloRewrite(
                   kCtrlEnable,
                   Pass(/*threshold_in_bytes=*/0,
                        DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
                   true));
  HloInstruction* are = FindInstruction(me.get(), "ar");
  Decomposed de = FindComponents(me.get(), "cp");
  ASSERT_NE(are, nullptr);
  EXPECT_TRUE(are->control_predecessors().empty());
  EXPECT_THAT(de.recv->control_predecessors(), ElementsAre(are));
  EXPECT_EQ(are->frontend_attributes().map().at(kCollectiveStreamAttrName),
            kCollectiveStreamP2P);
}

TEST_F(DecomposerTest, RelaysControlPredecessorsAndSuccessorsAcrossDecompose) {
  constexpr absl::string_view kHlo = R"(
  HloModule m
  cond {
    p = (u32[], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(p), index=0
    ROOT cmp = pred[] compare(i, u32[] constant(2)), direction=LT
  }
  body {
    p = (u32[], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(p), index=0
    x = f32[64] get-tuple-element(p), index=1
    y = f32[64] get-tuple-element(p), index=2
    t0 = token[] after-all()
    cp0 = f32[64] collective-permute(x), channel_id=1,
      source_target_pairs={{0,1}}, control-predecessors={t0}
    cp1 = f32[64] collective-permute(y), channel_id=2,
      source_target_pairs={{1,2}}, control-predecessors={cp0}
    sink = f32[64] negate(y), control-predecessors={cp1}
    ROOT r = (u32[], f32[64], f32[64]) tuple(
      u32[] add(i, u32[] constant(1)), cp0, f32[64] add(cp1, sink))
  }
  ENTRY e {
    z = f32[64] broadcast(f32[] constant(0)), dimensions={}
    ROOT w = (u32[], f32[64], f32[64]) while(
      (u32[], f32[64], f32[64]) tuple(u32[] constant(0), z, z)),
      body=body, condition=cond
  })";
  ASSERT_OK_AND_ASSIGN(
      auto m, RunAndCheckHloRewrite(
                  kHlo,
                  Pass(/*threshold_in_bytes=*/0,
                       DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE),
                  true));
  Decomposed d0 = FindComponents(m.get(), "cp0");
  Decomposed d1 = FindComponents(m.get(), "cp1");
  HloInstruction* t0 = FindInstruction(m.get(), "t0");
  HloInstruction* sink = FindInstruction(m.get(), "sink");
  EXPECT_THAT(d0.recv->control_predecessors(), ElementsAre(t0));
  EXPECT_THAT(d1.recv->control_predecessors(),
              ElementsAre(d0.recv_done, d0.send_done, d0.send));
  EXPECT_THAT(sink->control_predecessors(),
              ElementsAre(d1.recv_done, d1.send_done));
}

TEST_F(DecomposerTest, SkipsInPlaceAndWhileConditionCollectivePermutes) {
  constexpr absl::string_view kInPlace = R"(
  HloModule inplace_cp
  cond {
    p = (u32[], f32[8], f32[16]) parameter(0)
    i = u32[] get-tuple-element(p), index=0
    ROOT cmp = pred[] compare(i, u32[] constant(2)), direction=LT
  }
  body {
    p = (u32[], f32[8], f32[16]) parameter(0)
    i = u32[] get-tuple-element(p), index=0
    src = f32[8] get-tuple-element(p), index=1
    dst = f32[16] get-tuple-element(p), index=2
    off = (s32[]) tuple(s32[] constant(0))
    cp = f32[16] collective-permute(src, dst, off, off), channel_id=1,
      source_target_pairs={{0,1}}, slice_sizes={{8}}
    ROOT r = (u32[], f32[8], f32[16]) tuple(
      u32[] add(i, u32[] constant(1)), src, cp)
  }
  ENTRY e {
    s = f32[8] broadcast(f32[] constant(1)), dimensions={}
    d = f32[16] broadcast(f32[] constant(0)), dimensions={}
    ROOT w = (u32[], f32[8], f32[16]) while(
      (u32[], f32[8], f32[16]) tuple(u32[] constant(0), s, d)),
      body=body, condition=cond
  })";
  ASSERT_OK_AND_ASSIGN(
      auto m_ip, RunAndCheckHloRewrite(
                     kInPlace,
                     Pass(/*threshold_in_bytes=*/0,
                          DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
                     false));

  constexpr absl::string_view kWhileCond = R"(
  HloModule cond_cp
  cond {
    p = (u32[], f32[4]) parameter(0)
    x = f32[4] get-tuple-element(p), index=1
    cp = f32[4] collective-permute(x), channel_id=1,
      source_target_pairs={{0,1}}
    c0 = f32[4] broadcast(f32[] constant(0)), dimensions={}
    eq = pred[4] compare(cp, c0), direction=LT
    i = u32[] get-tuple-element(p), index=0
    ROOT cmp = pred[] compare(i, u32[] constant(2)), direction=LT
  }
  body {
    p = (u32[], f32[4]) parameter(0)
    i = u32[] get-tuple-element(p), index=0
    x = f32[4] get-tuple-element(p), index=1
    ROOT r = (u32[], f32[4]) tuple(u32[] add(i, u32[] constant(1)), x)
  }
  ENTRY e {
    z = f32[4] broadcast(f32[] constant(0)), dimensions={}
    ROOT w = (u32[], f32[4]) while(
      (u32[], f32[4]) tuple(u32[] constant(0), z)),
      body=body, condition=cond
  })";
  ASSERT_OK_AND_ASSIGN(
      auto m_wc, RunAndCheckHloRewrite(
                     kWhileCond,
                     Pass(/*threshold_in_bytes=*/0,
                          DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
                     false));

  ASSERT_OK_AND_ASSIGN(auto m_dyn, ParseAndReturnUnverifiedModule(R"(
  HloModule unbounded_cp
  cond {
    p = (u32[], f32[?]) parameter(0)
    i = u32[] get-tuple-element(p), index=0
    ROOT cmp = pred[] compare(i, u32[] constant(2)), direction=LT
  }
  body {
    p = (u32[], f32[?]) parameter(0)
    i = u32[] get-tuple-element(p), index=0
    x = f32[?] get-tuple-element(p), index=1
    cp = f32[?] collective-permute(x), channel_id=1,
      source_target_pairs={{0,1}}
    ROOT r = (u32[], f32[?]) tuple(u32[] add(i, u32[] constant(1)), cp)
  }
  ENTRY e {
    x0 = f32[?] parameter(0)
    ROOT w = (u32[], f32[?]) while(
      (u32[], f32[?]) tuple(u32[] constant(0), x0)),
      body=body, condition=cond
  })"));
  ASSERT_THAT(Pass(/*threshold_in_bytes=*/0,
                   DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE)
                  .Run(m_dyn.get()),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(DecomposerTest,
       PreservesResultLayoutMetadataAndV2V3ShardingsOnAllDecomposedOps) {
  constexpr absl::string_view kHlo = R"(
  HloModule layout_meta_sharding
  cond {
    p = (u32[], f32[4,8]{1,0}) parameter(0)
    i = u32[] get-tuple-element(p), index=0
    ROOT cmp = pred[] compare(i, u32[] constant(2)), direction=LT
  }
  body {
    p = (u32[], f32[4,8]{1,0}) parameter(0)
    i = u32[] get-tuple-element(p), index=0
    x = f32[4,8]{1,0} get-tuple-element(p), index=1,
      sharding={devices=[2,1]0,1}
    cp = f32[4,8]{0,1:S(1)} collective-permute(x), channel_id=1,
      source_target_pairs={{0,1}}, sharding={devices=[2,1]0,1},
      metadata={op_name="layer/cp" source_line=42}
    out = f32[4,8]{1,0} copy(cp), sharding={devices=[2,1]0,1}
    ROOT r = (u32[], f32[4,8]{1,0}) tuple(
      u32[] add(i, u32[] constant(1)), out)
  }
  ENTRY e {
    z = f32[4,8]{1,0} broadcast(f32[] constant(0)), dimensions={}
    ROOT w = (u32[], f32[4,8]{1,0}) while(
      (u32[], f32[4,8]{1,0}) tuple(u32[] constant(0), z)),
      body=body, condition=cond
  })";
  auto check_sharding = [&](const HloSharding& sh) {
    ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnUnverifiedModule(kHlo));
    HloInstruction* cp = FindInstruction(m.get(), "cp");
    ASSERT_NE(cp, nullptr);
    cp->set_sharding(sh);
    ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                            /*allow_mixed_precision=*/false)
                    .Run(m.get()),
                absl_testing::IsOk());
    ASSERT_THAT(Pass(/*threshold_in_bytes=*/0,
                     DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE)
                    .Run(m.get()),
                absl_testing::IsOkAndHolds(true));
    ASSERT_THAT(HloVerifier(/*layout_sensitive=*/true,
                            /*allow_mixed_precision=*/false)
                    .Run(m.get()),
                absl_testing::IsOk());
    Decomposed d = FindComponents(m.get(), "cp");
    HloInstruction* data = FindInstruction(m.get(), "cp-recv-data");
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->shape().layout().memory_space(), 1);
    EXPECT_EQ(data->shape().layout().minor_to_major(0), 0);
    EXPECT_EQ(data->sharding(), sh);
    for (HloInstruction* h :
         {d.after_all, d.recv, d.send, d.recv_done, d.send_done, data}) {
      EXPECT_EQ(h->metadata().op_name(), "layer/cp");
      EXPECT_TRUE(h->has_sharding());
    }
  };
  OpMetadata md;
  md.set_op_name("sh_md");
  check_sharding(HloSharding::Manual({md}));
  check_sharding(HloSharding::SingleDevice(0, {md}));
  Mesh mesh({2}, {"x"});
  check_sharding(HloSharding(NamedSharding::Replicate({md})));
  check_sharding(HloSharding(NamedSharding(
      mesh,
      {NamedSharding::DimensionSharding({AxisRef(0)}, /*is_closed=*/true),
       NamedSharding::DimensionSharding({}, /*is_closed=*/true)},
      {}, {}, {}, {md})));
}

TEST_F(DecomposerTest,
       HandlesAsyncStartDoneCollectivesWithCompletionAnchorsAndNoCrash) {
  constexpr absl::string_view kHlo = R"(
  HloModule async_collectives
  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT s = f32[] add(a, b)
  }
  async_cp_comp {
    p = f32[64] parameter(0)
    ROOT r = f32[64] collective-permute(p), channel_id=4,
      source_target_pairs={{1,2}}
  }
  cond {
    p = (u32[], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(p), index=0
    ROOT cmp = pred[] compare(i, u32[] constant(2)), direction=LT
  }
  body {
    p = (u32[], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(p), index=0
    x = f32[64] get-tuple-element(p), index=1
    y = f32[64] get-tuple-element(p), index=2
    ars = f32[64] all-reduce-start(y), channel_id=2,
      use_global_device_ids=true, replica_groups={{0,1}}, to_apply=add
    ard = f32[64] all-reduce-done(ars)
    cp = f32[64] collective-permute(x), channel_id=1,
      source_target_pairs={{0,1},{1,2}}, control-predecessors={ard}
    cps = (f32[64], f32[64]) collective-permute-start(y), channel_id=3,
      source_target_pairs={{1,2}}
    cpd = f32[64] collective-permute-done(cps)
    as = ((f32[64]), f32[64]) async-start(y), calls=async_cp_comp
    au = ((f32[64]), f32[64]) async-update(as)
    ad = f32[64] async-done(au)
    sum = f32[64] add(ard, f32[64] add(cpd, ad))
    ROOT r = (u32[], f32[64], f32[64]) tuple(
      u32[] add(i, u32[] constant(1)), cp, sum)
  }
  ENTRY e {
    z = f32[64] broadcast(f32[] constant(0)), dimensions={}
    ROOT w = (u32[], f32[64], f32[64]) while(
      (u32[], f32[64], f32[64]) tuple(u32[] constant(0), z, z)),
      body=body, condition=cond
  })";
  ASSERT_OK_AND_ASSIGN(
      auto m, RunAndCheckHloRewrite(
                  kHlo,
                  Pass(/*threshold_in_bytes=*/0,
                       DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE),
                  true));
  Decomposed d = FindComponents(m.get(), "cp");
  HloInstruction* ars = FindInstruction(m.get(), "ars");
  HloInstruction* ard = FindInstruction(m.get(), "ard");
  HloInstruction* cps = FindInstruction(m.get(), "cps");
  HloInstruction* cpd = FindInstruction(m.get(), "cpd");
  HloInstruction* as = FindInstruction(m.get(), "as");
  HloInstruction* au = FindInstruction(m.get(), "au");
  HloInstruction* ad = FindInstruction(m.get(), "ad");
  EXPECT_THAT(d.recv->control_predecessors(), ElementsAre(ard));
  EXPECT_THAT(cps->control_predecessors(), ElementsAre(d.send_done));
  EXPECT_THAT(as->control_predecessors(), ElementsAre(d.send_done));
  EXPECT_TRUE(
      ars->frontend_attributes().map().contains(kCollectiveStreamAttrName));
  EXPECT_FALSE(
      ard->frontend_attributes().map().contains(kCollectiveStreamAttrName));
  EXPECT_FALSE(
      cpd->frontend_attributes().map().contains(kCollectiveStreamAttrName));
  EXPECT_FALSE(
      au->frontend_attributes().map().contains(kCollectiveStreamAttrName));
  EXPECT_FALSE(
      ad->frontend_attributes().map().contains(kCollectiveStreamAttrName));
}

TEST_F(DecomposerTest,
       DetectsEmptyReplicaGroupCrossMode2DAndGuardsNegativeReplicaIds) {
  // Negative IDs and self-merges do not OOM or corrupt AbstractReplicaGroups.
  AbstractReplicaGroups g1, g2;
  g1.merge_groups(-1, 0);
  g1.merge_groups(0, -5);
  g1.merge_groups(0, 0);
  EXPECT_TRUE(g1.groups.empty());
  g1.merge_groups(0, 1);
  g2.groups.push_back({-3, 0, 999999});
  const size_t before = g1.index_map.size();
  EXPECT_FALSE(IsConflictingAbstractReplicaGroups(g1, g2));
  EXPECT_EQ(g1.index_map.size(), before);

  // Empty replica_groups={} on R=1,P=1 (wildcard), R=4,P=1, and R=1,P=4.
  for (auto [r, p] : {std::pair<int64_t, int64_t>{1, 1}, {4, 1}, {1, 4}}) {
    ASSERT_OK_AND_ASSIGN(auto m,
                         ParseAndReturnVerifiedModule(R"(
  HloModule empty_rg
  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT s = f32[] add(a, b)
  }
  cond {
    p = (u32[], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(p), index=0
    ROOT cmp = pred[] compare(i, u32[] constant(2)), direction=LT
  }
  body {
    p = (u32[], f32[64], f32[64]) parameter(0)
    i = u32[] get-tuple-element(p), index=0
    x = f32[64] get-tuple-element(p), index=1
    y = f32[64] get-tuple-element(p), index=2
    cp = f32[64] collective-permute(x), channel_id=1,
      source_target_pairs={{0,1}}
    ar = f32[64] all-reduce(y), channel_id=2, replica_groups={}, to_apply=add
    ROOT r = (u32[], f32[64], f32[64]) tuple(
      u32[] add(i, u32[] constant(1)), cp, ar)
  }
  ENTRY e {
    z = f32[64] broadcast(f32[] constant(0)), dimensions={}
    ROOT w = (u32[], f32[64], f32[64]) while(
      (u32[], f32[64], f32[64]) tuple(u32[] constant(0), z, z)),
      body=body, condition=cond
  })",
                                                      /*replica_count=*/r,
                                                      /*num_partitions=*/p));
    ASSERT_THAT(Pass(/*threshold_in_bytes=*/0,
                     DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE)
                    .Run(m.get()),
                absl_testing::IsOkAndHolds(true));
    EXPECT_THAT(FindInstruction(m.get(), "ar")->control_predecessors(),
                ElementsAre(FindInstruction(m.get(), "cp-send-done")));
  }

  // R=2, P=2 cross-mode: kCrossPartition cp {{0,1}} (and annotated device
  // send/recv) conflicts with kFlattenedID ar {{2,3}} and
  // kCrossReplicaAndPartition ar {{0},{1}}, but NOT with orthogonal
  // kCrossReplica ar {{0,1}} or {}. Also checks R=1,P=4 singleton {{0}}.
  ASSERT_OK_AND_ASSIGN(
      auto m2d,
      ParseAndReturnVerifiedModule(R"(
  HloModule r2p2, replica_count=2, num_partitions=2
  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT s = f32[] add(a, b)
  }
  ENTRY e {
    x = f32[16] parameter(0)
    cp = f32[16] collective-permute(x), channel_id=1,
      source_target_pairs={{0,1}}
    tok = token[] after-all()
    snd = (f32[16], u32[], token[]) send(x, tok), channel_id=2,
      frontend_attributes={_xla_send_recv_source_target_pairs="{{0,1}}"}
    sd = token[] send-done(snd), channel_id=2
    ar_flat = f32[16] all-reduce(x), channel_id=3, use_global_device_ids=true,
      replica_groups={{0},{1},{2,3}}, to_apply=add
    ar_crp = f32[16] all-reduce(x), channel_id=4, replica_groups={{0},{1}},
      to_apply=add
    ar_cr = f32[16] all-reduce(x), replica_groups={{0,1}}, to_apply=add
    ar_cr_empty = f32[16] all-reduce(x), replica_groups={}, to_apply=add
    cp_empty = f32[16] collective-permute(x), channel_id=5,
      source_target_pairs={}
    ROOT t = (f32[16], token[], f32[16], f32[16], f32[16], f32[16], f32[16])
      tuple(cp, sd, ar_flat, ar_crp, ar_cr, ar_cr_empty, cp_empty)
  })",
                                   /*replica_count=*/2, /*num_partitions=*/2));
  EXPECT_TRUE(
      FindAllConflictingCollectives(FindInstruction(m2d.get(), "cp_empty"))
          .empty());
  auto conf = FindAllConflictingCollectives(FindInstruction(m2d.get(), "cp"));
  EXPECT_THAT(conf, ElementsAre(FindInstruction(m2d.get(), "snd"),
                                FindInstruction(m2d.get(), "ar_flat"),
                                FindInstruction(m2d.get(), "ar_crp")));

  ASSERT_OK_AND_ASSIGN(auto m_crp1,
                       ParseAndReturnVerifiedModule(R"(
  HloModule r1p4_single, num_partitions=4
  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT s = f32[] add(a, b)
  }
  ENTRY e {
    x = f32[16] parameter(0)
    cp = f32[16] collective-permute(x), channel_id=1,
      source_target_pairs={{2,3}}
    ar_s = f32[16] all-reduce(x), channel_id=2, replica_groups={{0}},
      to_apply=add
    ROOT t = (f32[16], f32[16]) tuple(cp, ar_s)
  })",
                                                    /*replica_count=*/1,
                                                    /*num_partitions=*/4));
  EXPECT_THAT(
      FindAllConflictingCollectives(FindInstruction(m_crp1.get(), "cp")),
      ElementsAre(FindInstruction(m_crp1.get(), "ar_s")));

  // Over-65536 device grid uses bounded wildcard fallback without OOM.
  ASSERT_OK_AND_ASSIGN(auto m_huge,
                       ParseAndReturnVerifiedModule(R"(
  HloModule huge_grid
  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT s = f32[] add(a, b)
  }
  ENTRY e {
    x = f32[16] parameter(0)
    cp = f32[16] collective-permute(x), channel_id=1,
      source_target_pairs={{0,1}}
    ar = f32[16] all-reduce(x), channel_id=2, replica_groups={}, to_apply=add
    ROOT t = (f32[16], f32[16]) tuple(cp, ar)
  })",
                                                    /*replica_count=*/1000,
                                                    /*num_partitions=*/1000));
  EXPECT_THAT(
      FindAllConflictingCollectives(FindInstruction(m_huge.get(), "cp")),
      ElementsAre(FindInstruction(m_huge.get(), "ar")));
}

}  // namespace
}  // namespace xla
