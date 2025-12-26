#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/json_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

#include "xla/tools/hlo_module_loader.h"
#include "xla/service/hlo.pb.h"  // 提供 xla::HloModuleProto（路径/名字可能随版本变化）

ABSL_FLAG(std::string, input_format, "", "hlo|mhlo|stablehlo|pb|pbtxt");
ABSL_FLAG(std::string, output_format, "pb", "pb|pbtxt|json");
ABSL_FLAG(std::string, output, "out.pb", "output path");

static tsl::Status WriteString(const std::string& path, const std::string& s) {
  return tsl::WriteStringToFile(tsl::Env::Default(), path, s);
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  if (argc < 2) {
    LOG(QFATAL)
        << "Usage: hlo_to_proto --input_format=hlo --output_format=pb "
           "--output=out.pb file.hlo";
  }

  const std::string input_format = absl::GetFlag(FLAGS_input_format);
  const std::string output_format = absl::GetFlag(FLAGS_output_format);
  const std::string output = absl::GetFlag(FLAGS_output);

  const std::string path = argv[argc - 1];

  tsl::StatusOr<std::unique_ptr<xla::HloModule>> module_or =
      xla::LoadModuleFromFile(
          path, input_format,
          xla::hlo_module_loader_details::Config(),
          /*config_modifier_hook=*/{},
          /*buffer_assignment_proto=*/nullptr,
          /*fill_missing_layouts=*/true);

  TF_QCHECK_OK(module_or.status());
  std::unique_ptr<xla::HloModule> module = std::move(module_or).value();

  xla::HloModuleProto proto = module->ToProto();

  if (output_format == "pb") {
    std::string bytes = proto.SerializeAsString();
    TF_QCHECK_OK(WriteString(output, bytes));
    return 0;
  }

  if (output_format == "pbtxt") {
    std::string text;
    google::protobuf::TextFormat::PrintToString(proto, &text);
    TF_QCHECK_OK(WriteString(output, text));
    return 0;
  }

  if (output_format == "json") {
    std::string json;
    google::protobuf::util::JsonPrintOptions opts;
    opts.add_whitespace = true;
    opts.preserve_proto_field_names = true;
    auto st = google::protobuf::util::MessageToJsonString(proto, &json, opts);
    if (!st.ok()) {
      LOG(QFATAL) << "MessageToJsonString failed: " << st.message();
    }
    TF_QCHECK_OK(WriteString(output, json));
    return 0;
  }

  LOG(QFATAL) << "Unknown --output_format=" << output_format
              << " (expected pb|pbtxt|json)";
}
