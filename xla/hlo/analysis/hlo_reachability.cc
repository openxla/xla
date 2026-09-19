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
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
// Linux 5.14 added the populate advice and 6.18 the except advised flag;
// older headers lack the constants. Older kernels reject the populate advice
// with EINVAL, and PopulateMapping then gives the mapping up.
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
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

namespace {

// A populated huge page costs one page fault; on 4 KiB page kernels, small
// pages cost 512 per 2 MiB. A chunk with up to a quarter of its pages small
// still costs less than the heap's fresh pages.
constexpr int64_t kMaxFaultsPerHugePage = 128;

// Zeroing a chunk costs the kernel about 50 us of CPU time per MiB with huge
// pages and about 400 us with small pages. Above this the kernel is
// compacting memory to assemble huge pages, and the heap is cheaper.
constexpr int64_t kMaxCpuMicrosPerMiB = 500;

// The populate probe's decision, see BitStorage::PopulateCostAcceptable.
bool PopulateCostAcceptable(int64_t minor_faults, int64_t cpu_micros,
                            size_t chunk_bytes, size_t huge_page_bytes) {
  const int64_t huge_pages = chunk_bytes / huge_page_bytes;
  const int64_t mebibytes = chunk_bytes >> 20;
  return minor_faults <= kMaxFaultsPerHugePage * huge_pages &&
         cpu_micros <= kMaxCpuMicrosPerMiB * mebibytes;
}

#if defined(__linux__)

// The size from which a matrix is memory mapped instead of heap allocated,
// the same on every host; 8 MiB is four huge pages on 4 KiB page kernels, so
// both floors coincide there. The heap keeps serving the many small matrices
// as before.
constexpr size_t kMinMappedBytes = size_t{8} << 20;

// A matrix also spans at least this many huge pages, so that huge pages can
// back nearly all of it. Huge pages are usually 2 MiB on 4 KiB page kernels
// and up to 512 MiB on 64 KiB page kernels.
constexpr size_t kMinMappedHugePages = 4;

// The mapping is populated in chunks of whole huge pages, so that the first
// chunk can show whether the kernel hands out huge pages at a reasonable
// price and so that the kernel's address space lock is released between
// chunks.
constexpr size_t kPopulateChunkBytes = size_t{64} << 20;

// MADV_HUGEPAGE reports a transient shortage of kernel memory as EAGAIN.
constexpr int kHugePageAdviceAttempts = 3;

constexpr char kThpDir[] = "/sys/kernel/mm/transparent_hugepage/";

struct TransparentHugePages {
  bool enabled = false;
  // The huge page size; a mapping is aligned to it.
  size_t page_bytes = size_t{2} << 20;
};

// The selected option of a sysfs setting, "always [madvise] never" giving
// "madvise"; empty when the file cannot be read.
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

// Whether the kernel hands out transparent huge pages to this process, on
// request at least. Without them a fresh mapping faults in small pages on
// every map, which the heap's reuse of resident memory beats. The sysfs reads
// come first, so that a sandbox without /sys never reaches the prctl below.
const TransparentHugePages& GetTransparentHugePages() {
  static const TransparentHugePages thp = [] {
    TransparentHugePages result;
    const std::string dir = kThpDir;
    std::ifstream size_file(dir + "hpage_pmd_size");
    size_t page_bytes = 0;
    if (size_file >> page_bytes && page_bytes > 0) {
      result.page_bytes = page_bytes;
    }
    std::string enabled = SelectedOption(dir + "enabled");
    // Linux 6.8 added a setting per huge page size; any value but inherit
    // overrides the one above.
    const std::string pmd_enabled =
        SelectedOption(dir + "hugepages-" +
                       std::to_string(result.page_bytes / 1024) + "kB/enabled");
    if (!pmd_enabled.empty() && pmd_enabled != "inherit") {
      enabled = pmd_enabled;
    }
    if (enabled != "always" && enabled != "madvise") {
      return result;
    }
    // A process can opt out of huge pages (container runtimes do so for some
    // tasks), with or without an exception for regions that ask for them.
    const int disabled = prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0);
    result.enabled =
        disabled <= 0 || (disabled & PR_THP_DISABLE_EXCEPT_ADVISED) != 0;
    return result;
  }();
  return thp;
}

// What a populate cost the calling thread. Small pages show as one fault per
// base page, compaction as CPU time out of proportion to the bytes.
struct ThreadCost {
  int64_t minor_faults = 0;
  int64_t cpu_micros = 0;
};

bool GetThreadCost(ThreadCost* cost) {
  struct rusage usage = {};
  struct timespec cpu_time = {};
  if (getrusage(RUSAGE_THREAD, &usage) != 0 ||
      clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_time) != 0) {
    return false;
  }
  cost->minor_faults = usage.ru_minflt;
  cost->cpu_micros =
      int64_t{cpu_time.tv_sec} * 1000000 + cpu_time.tv_nsec / 1000;
  return true;
}

// Faults the mapping in with write faults, chunk by chunk. Returns false when
// the kernel does not support the advice or, for a mapping of several chunks,
// when the first chunk came back with too many small pages or cost too much:
// the caller then gives the mapping up and uses the heap, which reuses
// resident memory.
bool PopulateMapping(char* base, size_t bytes, size_t huge_page_bytes) {
  const size_t chunk_bytes = std::max(kPopulateChunkBytes, huge_page_bytes);
  for (size_t offset = 0; offset < bytes;) {
    const size_t length = std::min(chunk_bytes, bytes - offset);
    // A single chunk is populated by the time it could be judged. Giving it
    // up would only add work, and a chunk of small pages costs about what
    // the heap's fresh pages cost.
    const bool probe = offset == 0 && length < bytes;
    ThreadCost before;
    if (probe && !GetThreadCost(&before)) {
      return false;
    }
    if (madvise(base + offset, length, MADV_POPULATE_WRITE) != 0) {
      return false;
    }
    if (probe) {
      ThreadCost after;
      if (!GetThreadCost(&after) ||
          !PopulateCostAcceptable(after.minor_faults - before.minor_faults,
                                  after.cpu_micros - before.cpu_micros, length,
                                  huge_page_bytes)) {
        return false;
      }
    }
    offset += length;
  }
  return true;
}

// Returns an anonymous memory mapping of at least `bytes` zero bytes, aligned
// to a huge page, faulted in and backed by huge pages where the kernel grants
// them, or nullptr when the matrix should live on the heap instead.
// `*mapped_bytes` receives the length. A mapping that is given up holds at
// most the chunks populated so far, usually one, so unmapping it is cheap.
void* MapZeroedBytes(uint64_t bytes, size_t* mapped_bytes) {
  if (bytes < kMinMappedBytes) {
    return nullptr;
  }
  const TransparentHugePages& thp = GetTransparentHugePages();
  if (!thp.enabled || bytes < kMinMappedHugePages * thp.page_bytes) {
    return nullptr;
  }
  const uint64_t base_page_bytes = sysconf(_SC_PAGESIZE);
  const uint64_t length =
      (bytes + base_page_bytes - 1) / base_page_bytes * base_page_bytes;
  // The kernel aligns anonymous mappings to the base page only, or on some
  // kernels to a huge page (6.13 and later only for a whole number of huge
  // pages). One huge page of slack lets the mapping be trimmed to a huge page
  // boundary, so that no small pages at its start count against the probe.
  const uint64_t span = length + thp.page_bytes;
  // On 32 bit hosts a matrix can exceed the address space; the heap path
  // then fails as before.
  if (span > std::numeric_limits<size_t>::max()) {
    return nullptr;
  }
  void* raw = mmap(nullptr, span, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw == MAP_FAILED) {
    return nullptr;
  }
  const size_t head =
      (thp.page_bytes - reinterpret_cast<uintptr_t>(raw) % thp.page_bytes) %
      thp.page_bytes;
  char* mapping = static_cast<char*>(raw) + head;
  // A trim splits the new region off a neighboring mapping the kernel may
  // have merged it with, which fails at the mapping count limit. On failure
  // the code unmaps what it still owns, best effort.
  if (head > 0 && munmap(raw, head) != 0) {
    munmap(raw, span);
    return nullptr;
  }
  if (munmap(mapping + length, span - head - length) != 0) {
    munmap(mapping, span - head);
    return nullptr;
  }
  // Huge pages are opt in under the madvise setting; the populate step judges
  // the outcome either way.
  for (int attempt = 0; attempt < kHugePageAdviceAttempts; ++attempt) {
    if (madvise(mapping, length, MADV_HUGEPAGE) == 0 || errno != EAGAIN) {
      break;
    }
  }
  if (!PopulateMapping(mapping, length, thp.page_bytes)) {
    PCHECK(munmap(mapping, length) == 0);
    return nullptr;
  }
  *mapped_bytes = length;
  return mapping;
}

void UnmapBytes(void* mapping, size_t mapped_bytes) {
  PCHECK(munmap(mapping, mapped_bytes) == 0);
}

#else

void* MapZeroedBytes(uint64_t, size_t*) { return nullptr; }

void UnmapBytes(void*, size_t) {}

#endif

}  // namespace

bool HloReachabilityMap::BitStorage::PopulateCostAcceptable(
    int64_t minor_faults, int64_t cpu_micros, size_t chunk_bytes,
    size_t huge_page_bytes) {
  // Forwards to the file local decision above; the member exists so that the
  // test can reach it. The qualifier keeps this from calling itself.
  return xla::PopulateCostAcceptable(minor_faults, cpu_micros, chunk_bytes,
                                     huge_page_bytes);
}

HloReachabilityMap::BitStorage::BitStorage(size_t num_rows,
                                           size_t words_per_row) {
  const size_t num_blocks =
      (num_rows + kRowsPerAllocation - 1) / kRowsPerAllocation;
  blocks_.reserve(num_blocks);
  mapping_ =
      MapZeroedBytes(uint64_t{num_rows} * words_per_row * sizeof(BitSet::Word),
                     &mapped_bytes_);
  if (mapping_ != nullptr) {
    BitSet::Word* base = static_cast<BitSet::Word*>(mapping_);
    for (size_t block = 0; block < num_blocks; ++block) {
      blocks_.push_back(base + block * kRowsPerAllocation * words_per_row);
    }
    return;
  }
  heap_blocks_.reserve(num_blocks);
  for (size_t row = 0; row < num_rows; row += kRowsPerAllocation) {
    const size_t rows_in_block =
        std::min<size_t>(kRowsPerAllocation, num_rows - row);
    // make_unique initializes the array of words to 0.
    heap_blocks_.push_back(
        std::make_unique<BitSet::Word[]>(rows_in_block * words_per_row));
    blocks_.push_back(heap_blocks_.back().get());
  }
}

HloReachabilityMap::BitStorage::~BitStorage() {
  if (mapping_ != nullptr) {
    UnmapBytes(mapping_, mapped_bytes_);
  }
}

HloReachabilityMap::HloReachabilityMap(
    absl::Span<const HloInstruction* const> instructions)
    : words_per_bitset_((instructions.size() + BitSet::kBits - 1) /
                        BitSet::kBits),
      bit_storage_(instructions.size() + 1 /*for tmp_bit_set_*/,
                   words_per_bitset_) {
  if (!instructions.empty()) {
    CHECK(instructions[0]->parent() != nullptr)
        << "Instruction must be in a computation.";
    computation_id_ = instructions[0]->parent()->unique_id();
  } else {
    computation_id_ = kComputationIdAbsent;
  }

  tmp_bit_set_ = BitSetFromIndex(instructions.size());
  int32_t max_local_id = 0;
  for (const HloInstruction* instruction : instructions) {
    max_local_id = std::max(max_local_id, instruction->local_id());
  }
  indices_.resize(max_local_id + 1, kValueAbsent);
  for (size_t i = 0; i < instructions.size(); ++i) {
    BitSetFromIndex(i).Set(i);  // Instructions are reachable from themselves.
    indices_[GetKey(instructions[i])] = i;
  }
}

bool HloReachabilityMap::SetReachabilityToUnion(
    absl::Span<const HloInstruction* const> inputs,
    const HloInstruction* instruction) {
  Index index = GetIndex(instruction);
  BitSet bit_set = BitSetFromIndex(index);
  tmp_bit_set_.CopyBitSet(bit_set);
  SetReachabilityToUnionHelper(inputs, index);
  return bit_set != tmp_bit_set_;
}

void HloReachabilityMap::FastSetReachabilityToUnion(
    absl::Span<const HloInstruction* const> inputs,
    const HloInstruction* instruction) {
  SetReachabilityToUnionHelper(inputs, GetIndex(instruction));
}

void HloReachabilityMap::FastSetReachabilityToUnion(
    absl::Span<const Index> input_indices, Index index) {
  SetReachabilityToUnionHelper(input_indices, index);
}

void HloReachabilityMap::SetReachabilityToUnionHelper(
    absl::Span<const HloInstruction* const> inputs, Index index) {
  absl::InlinedVector<Index, 16> input_indices;
  input_indices.reserve(inputs.size());
  for (const HloInstruction* input : inputs) {
    input_indices.push_back(GetIndex(input));
  }
  SetReachabilityToUnionHelper(input_indices, index);
}

void HloReachabilityMap::SetReachabilityToUnionHelper(
    absl::Span<const Index> input_indices, Index index) {
  BitSet bit_set = BitSetFromIndex(index);
  // If instruction is part of inputs, don't reset the bit-set.
  if (!absl::c_linear_search(input_indices, index)) {
    bit_set.SetToZero();
  }
  bit_set.Set(index);
  for (Index input_index : input_indices) {
    if (input_index != index) {
      bit_set |= BitSetFromIndex(input_index);
    }
  }
}

void HloReachabilityMap::Replace(const HloInstruction* original,
                                 const HloInstruction* replacement) {
  Key original_key = GetKey(original);
  Key replacement_key = GetKey(replacement);
  if (original_key != replacement_key) {
    DCHECK_LT(original_key, indices_.size());
    if (replacement_key >= indices_.size()) {
      indices_.resize(replacement_key + 1, kValueAbsent);
    }
    indices_[replacement_key] = GetIndex(original);
    indices_[original_key] = kValueAbsent;
  }
}

std::unique_ptr<HloReachabilityMap> HloReachabilityMap::BuildWithRestrictions(
    const HloComputation* computation,
    absl::FunctionRef<void(const HloInstruction*,
                           std::vector<HloInstruction*>*)>
        add_dependencies) {
  const auto& all = computation->MakeInstructionPostOrder();
  auto result = std::make_unique<HloReachabilityMap>(all);

  std::vector<HloInstruction*> inputs;
  for (const HloInstruction* hlo : all) {
    inputs.clear();
    add_dependencies(hlo, &inputs);
    result->FastSetReachabilityToUnion(inputs, hlo);
  }
  return result;
}

std::unique_ptr<HloReachabilityMap> HloReachabilityMap::Build(
    const HloComputation* computation) {
  std::vector<HloInstruction*> instructions =
      computation->MakeInstructionPostOrder();
  auto result = std::make_unique<HloReachabilityMap>(instructions);

  auto get_bit_set = [&](const HloInstruction* instruction) -> BitSet {
    return result->BitSetFromIndex(result->GetIndex(instruction));
  };

  for (const HloInstruction* instruction : instructions) {
    BitSet bit_set = get_bit_set(instruction);

    auto add_dependencies = [&](const HloInstruction* instruction) {
      for (const HloInstruction* operand : instruction->operands()) {
        bit_set |= get_bit_set(operand);
      }
      for (const HloInstruction* predecessor :
           instruction->control_predecessors()) {
        bit_set |= get_bit_set(predecessor);
      }
    };

    add_dependencies(instruction);
  }
  return result;
}

void HloReachabilityMap::UpdateReachabilityThroughInstruction(
    const HloInstruction* instruction) {
  std::queue<const HloInstruction*> worklist;
  worklist.push(instruction);

  std::vector<HloInstruction*> inputs;

  // Keep track of the number of times an instruction is in the worklist and
  // only process it only if it is the last occurrence. Note that this might
  // still mean that an instruction is processed multiple times.
  absl::flat_hash_map<const HloInstruction*, int64_t> in_worklist;

  while (!worklist.empty()) {
    const HloInstruction* item = worklist.front();
    worklist.pop();
    --in_worklist[item];
    if (in_worklist[item] > 0) {
      continue;
    }

    inputs.assign(item->operands().begin(), item->operands().end());
    inputs.insert(inputs.end(), item->control_predecessors().begin(),
                  item->control_predecessors().end());

    if (SetReachabilityToUnion(inputs, item)) {
      // Add immediate successors to worklist.
      for (const HloInstruction* user : item->users()) {
        worklist.push(user);
        ++in_worklist[user];
      }
      for (const HloInstruction* succ : item->control_successors()) {
        worklist.push(succ);
        ++in_worklist[succ];
      }
    }
  }
}

// Use ptr tagging in `worklist` to check if current instruction is successor of
// left or right.
static constexpr uintptr_t FROM_LEFT_FLAG_MASK = 1;
static constexpr uintptr_t PTR_MASK = ~FROM_LEFT_FLAG_MASK;
static_assert(alignof(HloInstruction) >= 2,
              "HloInstruction must be aligned to at least 2 bytes");
void HloReachabilityMap::UpdateReachabilityForMerge(
    const HloInstruction* left, const HloInstruction* right) {
  DCHECK(tmp_worklist_.empty());
  DCHECK(tmp_indices_to_update_.empty());
  DCHECK(IsKeyPresent(GetKey(left)));
  DCHECK(IsKeyPresent(GetKey(right)));

  if (left == right) {
    return;
  }

  Index left_index = GetIndex(left);
  Index right_index = GetIndex(right);
  BitSet left_bit_set = BitSetFromIndex(left_index);
  BitSet right_bit_set = BitSetFromIndex(right_index);

  absl::flat_hash_set<const HloInstruction*> visited;
  auto add_to_worklist = [&](const HloInstruction* instr,
                             bool from_left) -> void {
    if (visited.insert(instr).second) {
      if (IsKeyPresent(GetKey(instr))) {
        BitSet bit_set = BitSetFromIndex(GetIndex(instr));
        // If the node is already reachable from both sides, we can skip it.
        if ((from_left && bit_set.Get(right_index)) ||
            (!from_left && bit_set.Get(left_index))) {
          return;
        }
      }
      uintptr_t raw_addr = reinterpret_cast<uintptr_t>(instr);
      tmp_worklist_.push_back(raw_addr | from_left);
      return;
    }
    return;
  };
  add_to_worklist(left, /*from_left=*/true);
  const bool left_added = !tmp_worklist_.empty();
  add_to_worklist(right, /*from_left=*/false);
  if (tmp_worklist_.empty()) {
    return;
  }
  left_bit_set.GetDifferingWordUnions(right_bit_set, tmp_changed_words_);
  if (tmp_changed_words_.empty()) {
    tmp_worklist_.clear();
    return;
  }
  while (!tmp_worklist_.empty()) {
    const uintptr_t item_and_from_left = tmp_worklist_.back();
    tmp_worklist_.pop_back();

    // Use ptr tagging to show if instruction is successor of left or right.
    const bool from_left = (item_and_from_left & FROM_LEFT_FLAG_MASK);
    const HloInstruction* item =
        reinterpret_cast<const HloInstruction*>(item_and_from_left & PTR_MASK);

    if (IsKeyPresent(GetKey(item))) {
      tmp_indices_to_update_.push_back(GetIndex(item));
    }
    for (const HloInstruction* user : item->users()) {
      add_to_worklist(user, from_left);
    }
    for (const HloInstruction* succ : item->control_successors()) {
      add_to_worklist(succ, from_left);
    }
  }
  DCHECK(tmp_worklist_.empty());
  // Based on the benchmarks, if the number of changed words is more than 25% of
  // the size of the bitset, it is faster do full |= instead of using
  // OrUpdatePartial.
  if (tmp_changed_words_.size() > words_per_bitset_ * 0.25) {
    // Is is guaranteed that either left_bit_set or right_bit_set is in
    // tmp_indices_to_update_ based on the logic above.
    BitSet info = left_added ? left_bit_set : right_bit_set;
    info.OrUpdatePartial(tmp_changed_words_);
    for (Index index : tmp_indices_to_update_) {
      BitSet bit_set = BitSetFromIndex(index);
      bit_set |= info;
    }
  } else {
    for (Index index : tmp_indices_to_update_) {
      BitSet bit_set = BitSetFromIndex(index);
      bit_set.OrUpdatePartial(tmp_changed_words_);
    }
  }
  tmp_changed_words_.clear();
  tmp_indices_to_update_.clear();
}

void HloReachabilityMap::UpdateMultipleInstructions(
    absl::flat_hash_map<const HloInstruction*,
                        absl::flat_hash_set<const HloInstruction*>>
        to_update) {
  while (!to_update.empty()) {
    auto it = to_update.begin();
    const HloInstruction* instruction = it->first;

    BitSet bit_set = BitSetFromIndex(GetIndex(instruction));
    bool changed = false;
    // NOLINTNEXTLINE the loop aggregation is order independent.
    for (const HloInstruction* operand : it->second) {
      BitSet operand_bit_set = BitSetFromIndex(GetIndex(operand));
      changed |= bit_set.OrUpdate(operand_bit_set);
    }
    to_update.erase(it);
    if (changed) {
      for (const HloInstruction* user : instruction->users()) {
        to_update[user].insert(instruction);
      }
      for (const HloInstruction* succ : instruction->control_successors()) {
        to_update[succ].insert(instruction);
      }
    }
  }
}

}  // namespace xla
