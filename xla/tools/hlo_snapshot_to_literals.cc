#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "xla/service/hlo.pb.h"              // HloSnapshot / LiteralProto
#include "xla/tools/run_hlo_module.pb.h"     // RunHloModuleLiterals
#include "tsl/platform/env.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

ABSL_FLAG(std::string, input, "", "Path to .hlo_snapshot.pb");
ABSL_FLAG(std::string, output, "", "Path to RunHloModuleLiterals proto to write");

namespace {

tsl::StatusOr<xla::HloSnapshot> ReadSnapshot(const std::string& path) {
  xla::HloSnapshot snapshot;
  TF_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), path, &snapshot));
  return snapshot;
}


tsl::Status WriteRunLiterals(const std::string& path,
                             const xla::RunHloModuleLiterals& run_lits) {
  return tsl::WriteBinaryProto(tsl::Env::Default(), path, run_lits);
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string input = absl::GetFlag(FLAGS_input);
  const std::string output = absl::GetFlag(FLAGS_output);

  if (input.empty() || output.empty()) {
    LOG(QFATAL) << "--input and --output must be set";
  }

  auto snapshot_or = ReadSnapshot(input);
  if (!snapshot_or.ok()) {
    LOG(QFATAL) << "Failed to read HloSnapshot from " << input << ": "
                << snapshot_or.status();
  }
  xla::HloSnapshot snapshot = *snapshot_or;

  xla::RunHloModuleLiterals run_lits;
  // RunHloModuleLiterals 里是 iterations[]
  auto* iter = run_lits.add_iterations();

  // 把 snapshot 里的所有 parameter（输入+权重+其他）原样拷过去
  for (const auto& arg : snapshot.arguments()) {
    *iter->add_arguments() = arg;
  }

  // 可选：顺便把 result 带上（有些场景方便对比）
  if (snapshot.has_result()) {
    *iter->mutable_result() = snapshot.result();
  }

  auto status = WriteRunLiterals(output, run_lits);
  if (!status.ok()) {
    LOG(QFATAL) << "Failed to write RunHloModuleLiterals to " << output
                << ": " << status;
  }

  LOG(INFO) << "Wrote RunHloModuleLiterals with " << iter->arguments_size()
            << " arguments to " << output;
  return 0;
}
