#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "tsl/platform/env.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "xla/debug_options_flags.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/tools/hlo_module_loader.h"

ABSL_FLAG(std::string, input_format, "",
          "hlo|mhlo|stablehlo|pb|pbtxt");
ABSL_FLAG(std::string, output, "out.html",
          "html output path");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  // 至少要有一个位置参数（输入文件）
  if (argc < 2) {
    LOG(QFATAL)
        << "Usage: hlo_to_html --input_format=hlo --output=out.html file.hlo";
  }

  const std::string input_format = absl::GetFlag(FLAGS_input_format);
  const std::string output = absl::GetFlag(FLAGS_output);

  // 直接用“最后一个 argv”当成 HLO 文件路径
  const std::string path = argv[argc - 1];

  xla::DebugOptions debug_opts = xla::GetDebugOptionsFromFlags();

  tsl::StatusOr<std::unique_ptr<xla::HloModule>> module_or =
      xla::LoadModuleFromFile(
          path, input_format,
          xla::hlo_module_loader_details::Config(),
          /*config_modifier_hook=*/{},
          /*buffer_assignment_proto=*/nullptr,
          /*fill_missing_layouts=*/true);
  TF_QCHECK_OK(module_or.status());

  std::unique_ptr<xla::HloModule> module = std::move(module_or).value();

  tsl::StatusOr<std::string> html_or =
      xla::RenderGraph(*module->entry_computation(), module->name(),
                       debug_opts, xla::RenderedGraphFormat::kHtml,
                       xla::HloRenderOptions());
  TF_QCHECK_OK(html_or.status());

  TF_QCHECK_OK(
      tsl::WriteStringToFile(tsl::Env::Default(), output, html_or.value()));

  return 0;
}
