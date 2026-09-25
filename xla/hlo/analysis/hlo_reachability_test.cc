/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/analysis/hlo_reachability.h"

#if defined(__linux__)
#include <sys/mman.h>
#include <sys/prctl.h>
#include <unistd.h>
// The same fallbacks as hlo_reachability.cc for headers older than the
// kernel.
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
#ifndef PR_GET_THP_DISABLE
#define PR_GET_THP_DISABLE 42
#endif
#ifndef PR_THP_DISABLE_EXCEPT_ADVISED
#define PR_THP_DISABLE_EXCEPT_ADVISED (1 << 1)
#endif
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal_util.h"
#include "xla/service/device_assignment.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

class HloReachabilityTest : public HloHardwareIndependentTestBase {};

TEST_F(HloReachabilityTest, Reachability) {
  // Construct and test a reachability graph of the following form:
  /*
       a
      / \
     b   c
      \ / \
       d   e
  */
  auto builder = HloComputation::Builder(TestName());
  auto a = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto b = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto c = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto d = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto e = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  HloReachabilityMap reachability({a, b, c, d, e});
  reachability.SetReachable(a, a);
  EXPECT_TRUE(reachability.SetReachabilityToUnion({a}, b));
  EXPECT_TRUE(reachability.SetReachabilityToUnion({a}, c));
  EXPECT_TRUE(reachability.SetReachabilityToUnion({b, c}, d));
  EXPECT_TRUE(reachability.SetReachabilityToUnion({c}, e));

  EXPECT_TRUE(reachability.IsReachable(a, a));
  EXPECT_TRUE(reachability.IsReachable(a, b));
  EXPECT_TRUE(reachability.IsReachable(a, c));
  EXPECT_TRUE(reachability.IsReachable(a, d));
  EXPECT_TRUE(reachability.IsReachable(a, e));

  EXPECT_FALSE(reachability.IsReachable(b, a));
  EXPECT_TRUE(reachability.IsReachable(b, b));
  EXPECT_FALSE(reachability.IsReachable(b, c));
  EXPECT_TRUE(reachability.IsReachable(b, d));
  EXPECT_FALSE(reachability.IsReachable(b, e));

  EXPECT_FALSE(reachability.IsReachable(e, a));
  EXPECT_FALSE(reachability.IsReachable(e, b));
  EXPECT_FALSE(reachability.IsReachable(e, c));
  EXPECT_FALSE(reachability.IsReachable(e, d));
  EXPECT_TRUE(reachability.IsReachable(e, e));

  // Recomputing the same reachability for a previously computed instruction
  // should return false (no change).
  EXPECT_FALSE(reachability.SetReachabilityToUnion({a}, b));
  EXPECT_FALSE(reachability.SetReachabilityToUnion({b, c}, d));
}

TEST_F(HloReachabilityTest, NonTrivialReachability) {
  // Test reachability of a non-trivial computation:
  //
  // const1    const2
  //    |         |
  //    | +-------+
  //    | |       |
  //    add ..   negate
  //     |   .     |
  //     |   .... exp
  //     |         |
  //     +---+   +-+---+
  //         |   |     |
  //       multiply   copy
  //
  // There is a control dependency from 'add' to 'exp'.
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f)));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kAdd, constant1, constant2));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kNegate, constant2));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, negate));
  auto mul = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kMultiply, add, exp));
  auto copy = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kCopy, exp));

  auto module = CreateNewVerifiedModule();
  auto computation =
      module->AddEntryComputation(builder.Build(/*root_instruction=*/mul));

  CHECK_OK(add->AddControlDependencyTo(exp));
  auto reachability = HloReachabilityMap::Build(computation);

  EXPECT_TRUE(reachability->IsReachable(constant1, constant1));
  EXPECT_FALSE(reachability->IsReachable(constant1, constant2));
  EXPECT_TRUE(reachability->IsReachable(constant1, add));
  EXPECT_FALSE(reachability->IsReachable(constant1, negate));
  EXPECT_TRUE(reachability->IsReachable(constant1, exp));
  EXPECT_TRUE(reachability->IsReachable(constant1, mul));
  EXPECT_TRUE(reachability->IsReachable(constant1, copy));

  EXPECT_FALSE(reachability->IsReachable(constant2, constant1));
  EXPECT_TRUE(reachability->IsReachable(constant2, constant2));
  EXPECT_TRUE(reachability->IsReachable(constant2, add));
  EXPECT_TRUE(reachability->IsReachable(constant2, negate));
  EXPECT_TRUE(reachability->IsReachable(constant2, exp));
  EXPECT_TRUE(reachability->IsReachable(constant2, mul));
  EXPECT_TRUE(reachability->IsReachable(constant2, copy));

  EXPECT_FALSE(reachability->IsReachable(exp, constant1));
  EXPECT_FALSE(reachability->IsReachable(exp, constant2));
  EXPECT_FALSE(reachability->IsReachable(exp, add));
  EXPECT_FALSE(reachability->IsReachable(exp, negate));
  EXPECT_TRUE(reachability->IsReachable(exp, exp));
  EXPECT_TRUE(reachability->IsReachable(exp, mul));
  EXPECT_TRUE(reachability->IsReachable(exp, copy));

  EXPECT_FALSE(reachability->IsReachable(mul, constant1));
  EXPECT_FALSE(reachability->IsReachable(mul, constant2));
  EXPECT_FALSE(reachability->IsReachable(mul, add));
  EXPECT_FALSE(reachability->IsReachable(mul, negate));
  EXPECT_FALSE(reachability->IsReachable(mul, exp));
  EXPECT_TRUE(reachability->IsReachable(mul, mul));
  EXPECT_FALSE(reachability->IsReachable(mul, copy));

  EXPECT_TRUE(reachability->IsConnected(constant1, copy));
  EXPECT_TRUE(reachability->IsConnected(copy, constant1));
  EXPECT_FALSE(reachability->IsConnected(negate, add));
  EXPECT_FALSE(reachability->IsConnected(add, negate));

  // Remove the control dependency then update and verify the reachability map
  ASSERT_IS_OK(add->RemoveControlDependencyTo(exp));
  reachability->UpdateReachabilityThroughInstruction(exp);

  EXPECT_TRUE(reachability->IsReachable(constant1, constant1));
  EXPECT_FALSE(reachability->IsReachable(constant1, constant2));
  EXPECT_TRUE(reachability->IsReachable(constant1, add));
  EXPECT_FALSE(reachability->IsReachable(constant1, negate));
  EXPECT_FALSE(reachability->IsReachable(constant1, exp));
  EXPECT_TRUE(reachability->IsReachable(constant1, mul));
  EXPECT_FALSE(reachability->IsReachable(constant1, copy));

  // Change a use within the graph then update and verify the reachability map
  ASSERT_IS_OK(constant2->ReplaceUseWith(negate, constant1));
  reachability->UpdateReachabilityThroughInstruction(negate);

  EXPECT_FALSE(reachability->IsReachable(constant2, constant1));
  EXPECT_TRUE(reachability->IsReachable(constant2, constant2));
  EXPECT_TRUE(reachability->IsReachable(constant2, add));
  EXPECT_FALSE(reachability->IsReachable(constant2, negate));
  EXPECT_FALSE(reachability->IsReachable(constant2, exp));
  EXPECT_TRUE(reachability->IsReachable(constant2, mul));
  EXPECT_FALSE(reachability->IsReachable(constant2, copy));
}

TEST_F(HloReachabilityTest, ChannelReachability) {
  const Shape shape = ShapeUtil::MakeShape(F32, {5, 7});
  HloComputation::Builder builder("ChannelReachability");
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto token0 = builder.AddInstruction(HloInstruction::CreateToken());
  auto send = builder.AddInstruction(HloInstruction::CreateSend(
      param, token0, /*channel_id=*/1, /*is_host_transfer=*/false));
  auto send_done = builder.AddInstruction(HloInstruction::CreateSendDone(
      send, send->channel_id(), /*is_host_transfer=*/false));
  auto token1 = builder.AddInstruction(HloInstruction::CreateToken());
  auto recv = builder.AddInstruction(HloInstruction::CreateRecv(
      shape, token1, /*channel_id=*/1, /*is_host_transfer=*/false));
  auto recv_done = builder.AddInstruction(HloInstruction::CreateRecvDone(
      recv, recv->channel_id(), /*is_host_transfer=*/false));

  auto module = CreateNewVerifiedModule();
  module->mutable_config().set_use_spmd_partitioning(false);
  module->mutable_config().set_static_device_assignment(DeviceAssignment(1, 2));
  auto computation = module->AddEntryComputation(builder.Build(recv_done));
  auto reachability = HloReachabilityMap::Build(computation);
  EXPECT_FALSE(reachability->IsReachable(param, recv_done));
  EXPECT_FALSE(reachability->IsReachable(send, recv));
  EXPECT_FALSE(reachability->IsReachable(send_done, recv));
}

TEST_F(HloReachabilityTest, ReplaceInstructions) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY entry {
      p0 = f32[28,28]{1,0} parameter(0)
      ROOT add = f32[28,28]{1,0} add(p0, p0)
    })")
                    .value();
  auto computation = module->entry_computation();
  auto reachability = HloReachabilityMap::Build(computation);
  auto* add = module->entry_computation()->root_instruction();
  auto* p0 = add->operand(0);
  EXPECT_TRUE(reachability->IsReachable(p0, add));

  // Replacing an instruction with itself is a noop.
  reachability->Replace(add, add);
  EXPECT_TRUE(reachability->IsReachable(p0, add));

  // Introduce a fusion instruction taking the place of `add`.
  auto* fusion = computation->AddInstruction(HloInstruction::CreateFusion(
      add->shape(), HloInstruction::FusionKind::kLoop, add));
  EXPECT_FALSE(reachability->IsPresent(fusion));
  EXPECT_TRUE(reachability->IsReachable(p0, add));

  // Replace `add` with `fusion` in the readability map.
  reachability->Replace(add, fusion);
  EXPECT_FALSE(reachability->IsPresent(add));
  EXPECT_TRUE(reachability->IsReachable(p0, fusion));
}

TEST_F(HloReachabilityTest, UpdateMultipleInstructions) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  auto builder = HloComputation::Builder(TestName());
  auto a = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  auto b = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f)));
  auto c = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kNegate, a));
  auto d = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, b));
  auto e = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kCopy, c));
  auto f = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kCopy, d));

  auto module = CreateNewVerifiedModule();
  auto computation =
      module->AddEntryComputation(builder.Build(/*root_instruction=*/f));

  auto reachability = HloReachabilityMap::Build(computation);

  EXPECT_TRUE(reachability->IsReachable(a, c));
  EXPECT_TRUE(reachability->IsReachable(c, e));
  EXPECT_TRUE(reachability->IsReachable(a, e));

  EXPECT_FALSE(reachability->IsReachable(b, c));
  EXPECT_FALSE(reachability->IsReachable(b, e));
  EXPECT_FALSE(reachability->IsReachable(d, e));
  EXPECT_FALSE(reachability->IsReachable(a, d));
  EXPECT_FALSE(reachability->IsReachable(a, f));

  // Add a control dependency from b to c, and d to e.
  ASSERT_IS_OK(b->AddControlDependencyTo(c));
  ASSERT_IS_OK(d->AddControlDependencyTo(e));

  absl::flat_hash_map<const HloInstruction*,
                      absl::flat_hash_set<const HloInstruction*>>
      to_update;
  to_update[c].insert(b);
  to_update[e].insert(d);

  reachability->UpdateMultipleInstructions(to_update);

  // Now b should be reachable to c, e
  EXPECT_TRUE(reachability->IsReachable(b, c));
  EXPECT_TRUE(reachability->IsReachable(b, e));

  // d should be reachable to e
  EXPECT_TRUE(reachability->IsReachable(d, e));

  // a is still reachable to c, e
  EXPECT_TRUE(reachability->IsReachable(a, c));
  EXPECT_TRUE(reachability->IsReachable(a, e));

  // a is still not reachable to d, f
  EXPECT_FALSE(reachability->IsReachable(a, d));
  EXPECT_FALSE(reachability->IsReachable(a, f));
}

#if defined(__linux__)
// A copy of the helper in hlo_reachability.cc: the selected option of a
// sysfs setting, "always [madvise] never" giving "madvise"; empty when the
// file cannot be read.
std::string SelectedOption(const std::string& path) {
  std::ifstream file(path);
  std::string line;
  std::getline(file, line);
  const size_t open_bracket = line.find('[');
  if (open_bracket == std::string::npos) {
    return "";
  }
  const size_t close_bracket = line.find(']', open_bracket);
  if (close_bracket == std::string::npos) {
    return "";
  }
  return line.substr(open_bracket + 1, close_bracket - open_bracket - 1);
}
#endif

// Whether HloReachabilityMap memory maps the 8 MiB matrices of the test
// below on this host. It mirrors the gate in hlo_reachability.cc, so keep the
// two in step: huge pages of at most 2 MiB, so that 8 MiB holds the four huge
// pages a mapping must span; transparent huge pages enabled for that size and
// not disabled for the process; and Linux 5.14 or later, which has the
// populate advice.
bool HostMemoryMapsLargeMatrices() {
#if defined(__linux__)
  const std::string thp_dir = "/sys/kernel/mm/transparent_hugepage/";
  std::ifstream size_file(thp_dir + "hpage_pmd_size");
  size_t huge_page_bytes = size_t{2} << 20;
  if (size_t read_bytes = 0; size_file >> read_bytes && read_bytes > 0) {
    huge_page_bytes = read_bytes;
  }
  if (huge_page_bytes > (size_t{2} << 20)) {
    return false;
  }
  std::string enabled = SelectedOption(thp_dir + "enabled");
  const std::string pmd_enabled =
      SelectedOption(thp_dir + "hugepages-" +
                     std::to_string(huge_page_bytes / 1024) + "kB/enabled");
  if (!pmd_enabled.empty() && pmd_enabled != "inherit") {
    enabled = pmd_enabled;
  }
  if (enabled != "always" && enabled != "madvise") {
    return false;
  }
  const int disabled = prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0);
  if (disabled > 0 && (disabled & PR_THP_DISABLE_EXCEPT_ADVISED) == 0) {
    return false;
  }
  const size_t page_bytes = sysconf(_SC_PAGESIZE);
  void* page = mmap(nullptr, page_bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (page == MAP_FAILED) {
    return false;
  }
  const bool populates = madvise(page, page_bytes, MADV_POPULATE_WRITE) == 0;
  PCHECK(munmap(page, page_bytes) == 0);
  return populates;
#else
  return false;
#endif
}

// Builds a chain of `num_instructions` exponentials and its reachability map.
std::pair<std::unique_ptr<HloReachabilityMap>, std::vector<HloInstruction*>>
BuildChainReachability(HloModule* module, int num_instructions) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  auto builder = HloComputation::Builder("chain");
  std::vector<HloInstruction*> chain;
  chain.push_back(builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f))));
  for (int i = 1; i < num_instructions; ++i) {
    chain.push_back(builder.AddInstruction(
        HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, chain.back())));
  }
  HloComputation* computation =
      module->AddEntryComputation(builder.Build(chain.back()));
  return {HloReachabilityMap::Build(computation), std::move(chain)};
}

}  // namespace

class HloReachabilityMapTestPeer {
 public:
  static bool IsMemoryMapped(const HloReachabilityMap& map) {
    return map.bit_storage_.is_memory_mapped();
  }
  static bool PopulateCostAcceptable(int64_t minor_faults, int64_t cpu_micros,
                                     size_t chunk_bytes,
                                     size_t huge_page_bytes) {
    return HloReachabilityMap::BitStorage::PopulateCostAcceptable(
        minor_faults, cpu_micros, chunk_bytes, huge_page_bytes);
  }
};

namespace {

// The probe judges the first chunk of a mapping by its page faults (one per
// huge page, one per base page otherwise) and its CPU time (about 50 us per
// MiB with huge pages, more when the kernel compacts memory for them).
TEST(HloReachabilityMapBitStorageTest, PopulateCostAcceptable) {
  using Peer = HloReachabilityMapTestPeer;
  constexpr size_t kMiB = size_t{1} << 20;
  // A 64 MiB chunk of 2 MiB huge pages on a 4 KiB page kernel.
  EXPECT_TRUE(Peer::PopulateCostAcceptable(32, 3200, 64 * kMiB, 2 * kMiB));
  // Up to a quarter small pages and up to 500 us per MiB pass.
  EXPECT_TRUE(Peer::PopulateCostAcceptable(4096, 3200, 64 * kMiB, 2 * kMiB));
  EXPECT_FALSE(Peer::PopulateCostAcceptable(4097, 3200, 64 * kMiB, 2 * kMiB));
  EXPECT_TRUE(Peer::PopulateCostAcceptable(32, 32000, 64 * kMiB, 2 * kMiB));
  EXPECT_FALSE(Peer::PopulateCostAcceptable(32, 32001, 64 * kMiB, 2 * kMiB));
  // All small pages, and huge pages assembled by compaction.
  EXPECT_FALSE(Peer::PopulateCostAcceptable(16384, 3200, 64 * kMiB, 2 * kMiB));
  EXPECT_FALSE(Peer::PopulateCostAcceptable(32, 64000, 64 * kMiB, 2 * kMiB));
  // A 64 KiB page kernel: one 512 MiB huge page per chunk, or 8192 small
  // pages; 128 faults for the one huge page, 500 us per MiB as before.
  EXPECT_TRUE(Peer::PopulateCostAcceptable(1, 25600, 512 * kMiB, 512 * kMiB));
  EXPECT_TRUE(
      Peer::PopulateCostAcceptable(128, 256000, 512 * kMiB, 512 * kMiB));
  EXPECT_FALSE(
      Peer::PopulateCostAcceptable(129, 25600, 512 * kMiB, 512 * kMiB));
  EXPECT_FALSE(Peer::PopulateCostAcceptable(1, 256001, 512 * kMiB, 512 * kMiB));
  EXPECT_FALSE(
      Peer::PopulateCostAcceptable(8192, 25600, 512 * kMiB, 512 * kMiB));
}

// 8191 instructions give an 8192 row by 128 word matrix of exactly 8 MiB, the
// smallest matrix that HloReachabilityMap memory maps where the host allows
// it; 8192 instructions need one more block and a mapping that is not a whole
// number of huge pages.
TEST_F(HloReachabilityTest, LargeMatrixIsMemoryMapped) {
  auto small_module = CreateNewVerifiedModule();
  EXPECT_FALSE(HloReachabilityMapTestPeer::IsMemoryMapped(
      *BuildChainReachability(small_module.get(), 8190).first));

  const bool host_maps = HostMemoryMapsLargeMatrices();
  // So that the test report shows which path ran.
  RecordProperty("memory_mapped", host_maps);
  for (int num_instructions : {8191, 8192}) {
    SCOPED_TRACE(num_instructions);
    auto module = CreateNewVerifiedModule();
    auto [reachability, chain] =
        BuildChainReachability(module.get(), num_instructions);
    EXPECT_EQ(HloReachabilityMapTestPeer::IsMemoryMapped(*reachability),
              host_maps);
    EXPECT_TRUE(reachability->IsReachable(chain.front(), chain.back()));
    EXPECT_FALSE(reachability->IsReachable(chain.back(), chain.front()));
    EXPECT_TRUE(reachability->IsReachable(chain[1000], chain[7000]));
    EXPECT_FALSE(reachability->IsReachable(chain[7000], chain[1000]));
    // Rows 4095 and 4096 are the last row of one block and the first of the
    // next. Recomputing an unchanged row goes through the temporary row, the
    // last row of the matrix, and reports no change.
    EXPECT_FALSE(
        reachability->SetReachabilityToUnion({chain[4095]}, chain[4096]));
    EXPECT_FALSE(reachability->SetReachabilityToUnion(
        {chain[num_instructions - 2]}, chain.back()));
  }
}

}  // namespace

class HloReachabilityMapBitSetBenchmark {
 public:
  explicit HloReachabilityMapBitSetBenchmark(int size) {
    size_t nwords = (size + 63) / 64;
    space_.resize(2 * nwords);
    a_ = HloReachabilityMap::BitSet(&space_[0], nwords);
    b_ = HloReachabilityMap::BitSet(&space_[nwords], nwords);
    // Initialize the bit sets to random inputs. Done out of caution -- note
    // that a sufficiently smart optimizer might realize that the bit sets
    // are otherwise initialized to 0.
    absl::BitGen gen;
    for (int i = 0; i < size; ++i) {
      if (absl::Bernoulli(gen, 0.5)) a_.Set(i);
      if (absl::Bernoulli(gen, 0.5)) b_.Set(i);
    }
  }
  void Union() { a_ |= b_; }

  void OrUpdatePartial(
      const std::vector<std::pair<size_t, HloReachabilityMap::BitSet::Word>>&
          diff) {
    a_.OrUpdatePartial(diff);
  }

  std::vector<std::pair<size_t, HloReachabilityMap::BitSet::Word>> GenerateDiff(
      int num_elements) {
    std::vector<std::pair<size_t, HloReachabilityMap::BitSet::Word>> diff;
    size_t nwords = a_.NumWords();
    if (nwords == 0) {
      return diff;
    }
    absl::BitGen gen;
    if (num_elements >= nwords) {
      for (size_t i = 0; i < nwords; ++i) {
        diff.push_back(
            {i, absl::Uniform<HloReachabilityMap::BitSet::Word>(gen)});
      }
    } else {
      absl::flat_hash_set<uint64_t> indices;
      while (indices.size() < num_elements) {
        indices.insert(absl::Uniform<size_t>(gen, 0, nwords));
      }
      std::vector<uint64_t> sorted_indices(indices.begin(), indices.end());
      std::sort(sorted_indices.begin(), sorted_indices.end());
      for (uint64_t idx : sorted_indices) {
        diff.push_back(
            {idx, absl::Uniform<HloReachabilityMap::BitSet::Word>(gen)});
      }
    }
    return diff;
  }

 private:
  std::vector<uint64_t> space_;
  HloReachabilityMap::BitSet a_;
  HloReachabilityMap::BitSet b_;
};

namespace {

void BM_HloReachabilityBitSetUnion(benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  for (auto s : state) {
    bm.Union();
  }
}
#define BM_ARGS Arg(1)->Arg(64)->Arg(128)->Arg(256)->Range(512, 256 * 1024)
BENCHMARK(BM_HloReachabilityBitSetUnion)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartialRandom2Diff(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  auto diff = bm.GenerateDiff(2);
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartialRandom2Diff)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartialRandom10Diff(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  auto diff = bm.GenerateDiff(10);
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartialRandom10Diff)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartial1Percent(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  int nwords = (state.range(0) + 63) / 64;
  auto diff = bm.GenerateDiff(std::max<int>(1, nwords / 100));
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartial1Percent)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartial10Percent(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  int nwords = (state.range(0) + 63) / 64;
  auto diff = bm.GenerateDiff(std::max<int>(1, nwords / 10));
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartial10Percent)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartial25Percent(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  int nwords = (state.range(0) + 63) / 64;
  auto diff = bm.GenerateDiff(std::max<int>(1, nwords / 4));
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartial25Percent)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartial50Percent(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  int nwords = (state.range(0) + 63) / 64;
  auto diff = bm.GenerateDiff(std::max<int>(1, nwords / 2));
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartial50Percent)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartial75Percent(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  int nwords = (state.range(0) + 63) / 64;
  auto diff = bm.GenerateDiff(std::max<int>(1, nwords * 3 / 4));
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartial75Percent)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartialDenseRandom(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  auto diff = bm.GenerateDiff((state.range(0) + 63) / 64);
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartialDenseRandom)->BM_ARGS;

class HloReachabilityBenchmark {
 public:
  HloReachabilityBenchmark(int size, absl::string_view name) : name_(name) {
    Shape r0f32 = ShapeUtil::MakeShape(F32, {});
    auto builder = HloComputation::Builder(name);

    // Build a graph of chained Exponentials, i.e. Exp(...(Exp(Input))...).
    HloInstruction* constant = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f)));
    HloInstruction* prev = constant;
    for (int i = 1; i < size; ++i) {
      prev = builder.AddInstruction(
          HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, prev));
    }

    HloModuleConfig hlo_config;
    module_ = std::make_unique<HloModule>(name_, hlo_config);
    computation_ =
        module_->AddEntryComputation(builder.Build(/*root_instruction=*/prev));
  }
  std::unique_ptr<HloReachabilityMap> Build() {
    return HloReachabilityMap::Build(computation_);
  }

 private:
  std::unique_ptr<HloModule> module_;
  HloComputation* computation_;
  const std::string name_;
};

void BM_HloReachabilityBuild(benchmark::State& state) {
  HloReachabilityBenchmark bm(state.range(0), state.name());
  for (auto s : state) {
    benchmark::DoNotOptimize(bm.Build());
  }
}
BENCHMARK(BM_HloReachabilityBuild)->BM_ARGS;

}  // namespace

}  // namespace xla
