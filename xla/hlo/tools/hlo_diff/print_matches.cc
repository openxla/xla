#include <iostream>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"

#include "xla/debug_options_flags.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_diff.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_module_util.h"

ABSL_FLAG(std::string, first, "", "Path to first HLO text/proto");
ABSL_FLAG(std::string, second, "", "Path to second HLO text/proto");
ABSL_FLAG(bool, ignore_shape, false, "Ignore shapes in fingerprints");

using xla::HloModule;
using xla::hlo_diff::ComputeDiff;
using xla::hlo_diff::DiffOptions;

xla::HloParserOptions ParserOpts() {
  xla::HloParserOptions opts;
  opts.set_fill_shortform_constants_with_random_values(false);
  return opts;
}

absl::StatusOr<std::unique_ptr<HloModule>> LoadModule(const std::string& path) {
  // 这里直接按 HLO text 读；如需 proto，可仿照 hlo_diff_main 追加读取逻辑
  return xla::ReadModuleFromHloTextFile(
      path, xla::GetDebugOptionsFromFlags(), ParserOpts());
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string first = absl::GetFlag(FLAGS_first);
  const std::string second = absl::GetFlag(FLAGS_second);
  if (first.empty() || second.empty()) {
    LOG(ERROR) << "--first/--second are required";
    return 1;
  }

  auto m1 = LoadModule(first);
  auto m2 = LoadModule(second);
  if (!m1.ok() || !m2.ok()) {
    LOG(ERROR) << "Parse failed: " << m1.status() << " / " << m2.status();
    return 1;
  }

  DiffOptions opts;
  opts.fingerprint_options.ignore_shape = absl::GetFlag(FLAGS_ignore_shape);

  auto diff = ComputeDiff(**m1, **m2, opts);
  if (!diff.ok()) {
    LOG(ERROR) << diff.status();
    return 1;
  }

  const auto& res = *diff->diff_result;
  LOG(INFO) << "Unchanged: " << res.unchanged_instructions.size()
            << ", Changed: " << res.changed_instructions.size()
            << ", Unmatched L/R: "
            << res.left_module_unmatched_instructions.size()
            << "/" << res.right_module_unmatched_instructions.size();

  std::cout << "\n";

  // 把“两边都有的”都算进去：unchanged + changed
  std::vector<std::pair<const xla::HloInstruction*, const xla::HloInstruction*>>
      matches;
  matches.reserve(res.unchanged_instructions.size() +
                  res.changed_instructions.size());

  auto append_map =
      [&](const auto& m) {
        for (const auto& kv : m) {
          matches.emplace_back(kv.first, kv.second);
        }
      };

  append_map(res.unchanged_instructions);
  append_map(res.changed_instructions);

  for (const auto& kv : matches) {
    std::cout << kv.first->name() << " <-> " << kv.second->name() << "\n";
  }

  return 0;
}
