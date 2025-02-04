#include "xla/service/collective_combiner_utils.h"

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"

namespace xla {

// Merges op metadata from a list of instructions to be combined
// as much as possible.
OpMetadata MergeOpMetadata(absl::Span<HloInstruction* const> to_combine) {
  OpMetadata op_metadata;
  absl::flat_hash_set<std::string> source_info_set;
  for (const HloInstruction* instr : to_combine) {
    const auto& metadata = instr->metadata();
    if (metadata.source_file().empty()) {
      continue;
    }
    std::string source_info;
    absl::StrAppend(&source_info, metadata.source_file(), ":",
                    metadata.source_line());
    source_info_set.insert(std::move(source_info));
  }
  if (!source_info_set.empty()) {
    op_metadata.set_source_file(absl::StrJoin(source_info_set, ","));
  }
  return op_metadata;
}

std::string MaybeMergeBackendConfigString(
    absl::Span<HloInstruction* const> to_combine) {
  if (to_combine.empty()) {
    return "";
  }

  std::string result = to_combine.front()->raw_backend_config_string();
  for (const auto* instr : to_combine) {
    if (instr->raw_backend_config_string() != result) {
      return "";
    }
  }
  return result;
}
}  // namespace xla
