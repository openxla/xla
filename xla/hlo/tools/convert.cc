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

#include <stdio.h>

#include <string>
#include <vector>

#include "xla/service/hlo_module_util.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/tsl/util/fixed_option_set_flag.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/protobuf.h"

const std::string kUsage = R"(
Reads an HLO module and outputs it in the requested format.
)";

namespace xla {
namespace {

enum class OutputFormat {
  kText,
  kProtoBinary,
  kProtoText,
};

bool AbslParseFlag(absl::string_view text, OutputFormat* output_format,
                   std::string* error) {
  return GetFixedOptionSetFlagParser<OutputFormat>(
             {{"text", OutputFormat::kText},
              {"proto_binary", OutputFormat::kProtoBinary},
              {"proto_text", OutputFormat::kProtoText}})
      .Parse(text, output_format, error);
}

absl::Status RealMain(absl::string_view input_file,
                      absl::string_view output_file,
                      absl::string_view input_format_str,
                      absl::string_view output_format_str) {
  InputFormat input_format;
  OutputFormat output_format;
  if (!input_format_str.empty()) {
    if (std::string error;
        !AbslParseFlag(input_format_str, &input_format, &error)) {
      return absl::InternalError("Failed parsing input format: " + error);
    }
  }
  if (!output_format_str.empty()) {
    if (std::string error;
        !AbslParseFlag(output_format_str, &output_format, &error)) {
      return absl::InternalError("Failed parsing output format: " + error);
    }
  }
  TF_ASSIGN_OR_RETURN(const auto module_and_arguments,
                      LoadHloModuleAndArguments(input_file, input_format));
  const auto& module = module_and_arguments.hlo_module;
  std::string result;
  switch (output_format) {
    case OutputFormat::kText:
      result = module->ToString();
      break;
    case OutputFormat::kProtoText:
      if (!tsl::protobuf::TextFormat::PrintToString(MakeHloProto(*module),
                                                    &result)) {
        return absl::InternalError("Proto to text conversion failed.");
      }
      break;
    case OutputFormat::kProtoBinary:
      MakeHloProto(*module).AppendToString(&result);
  }
  if (output_file == "-") {
    std::cout << result;
    return absl::OkStatus();
  }
  return tsl::WriteStringToFile(tsl::Env::Default(), std::string(output_file),
                                result);
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::string output_file = "-", input_format = "text", output_format = "text";
  const std::vector<tsl::Flag> flag_list = {
      tsl::Flag("output", &output_file, "Output file. '-' for stdout."),
      tsl::Flag("input_format", &input_format,
                "Input format: text / proto_text / proto_binary / "
                "snapshot_proto_binary."),
      tsl::Flag("output_format", &output_format,
                "Output format: text / proto_text / proto_binary."),
  };
  const std::string kUsageAndFlags =
      absl::StrCat(kUsage, "\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(kUsageAndFlags.c_str(), &argc, &argv);
  CHECK(parse_ok && argc == 2) << "\n" << kUsageAndFlags;

  absl::Status result =
      xla::RealMain(argv[1], output_file, input_format, output_format);
  if (!result.ok()) {
    LOG(ERROR) << result.message();
  }
  return result.raw_code();
}
