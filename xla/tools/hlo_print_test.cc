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

    // for (xla::HloComputation* comp : module->computations()) {
    //     LOG(INFO) << "computation " << comp->name();
    //     for (xla::HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    //         LOG(INFO) << instr->ToString();
    //     }
    // }
    // for (xla::HloComputation* comp : module->computations()) {
    //     LOG(INFO) << comp->name();
    // }


    for (xla::HloComputation* comp : module->computations()) {
        // LOG(INFO) << "computation: " << comp->name();

        // // 谁调用我（按调用次数）
        // for (auto& [caller, count] : comp->caller_computations()) {
        //     LOG(INFO) << "  caller: " << caller->name() << " x" << count;
        // }

        // // 我调用谁（按调用次数）
        // for (auto& [callee, count] : comp->callee_computations()) {
        //     LOG(INFO) << "  callee: " << callee->name() << " x" << count;
        // }

        // // 可选：具体哪些指令作为 caller
        // for (xla::HloInstruction* instr : comp->caller_instructions()) {
        //     LOG(INFO) << "  caller instr: " << instr->ToString();
        // }
        // int64_t n = comp->instruction_count();
        // LOG(INFO) << "count = " << n;

        // for (xla::HloInstruction* instr : comp->MakeInstructionPostOrder()) {
        // LOG(INFO) << HloOpcodeString(instr->opcode()) << " | " << instr->name();
        // }








        // for (xla::HloInstruction* instr : comp->MakeInstructionPostOrder()) {
        // LOG(INFO) << instr->name() << " (" << HloOpcodeString(instr->opcode()) << ")";
        // for (int i = 0; i < instr->operand_count(); ++i) {
        //     const xla::HloInstruction* op = instr->operand(i);
        //     LOG(INFO) << "  arg" << i << ": " << op->name()
        //             << " (" << HloOpcodeString(op->opcode()) << ")";
        //     LOG(INFO) << "arg" << i << " shape: " << op->shape().ToString();
        //     auto& layout = instr->shape().layout();
        // LOG(INFO) << instr->name() << " layout: " << layout.ToString();
        // LOG(INFO) << "minor_to_major: " << absl::StrJoin(layout.minor_to_major(), ",");

        // }








        for (xla::HloInstruction* instr : comp->MakeInstructionPostOrder()) {
        LOG(INFO) << instr->name() << " | operands: " << instr->operand_count();
        }
        // 如果指令有嵌套 computation（如 kMap/kWhile/kCall 等），也能列出：
        // for (xla::HloComputation* callee : instr->called_computations()) {
        //     LOG(INFO) << "  callee: " << callee->name();
        // }
        // }


    }




//   xla::HloModuleProto proto = module->ToProto();

//   if (output_format == "pb") {
//     std::string bytes = proto.SerializeAsString();
//     TF_QCHECK_OK(WriteString(output, bytes));
//     return 0;
//   }

//   if (output_format == "pbtxt") {
//     std::string text;
//     google::protobuf::TextFormat::PrintToString(proto, &text);
//     TF_QCHECK_OK(WriteString(output, text));
//     return 0;
//   }

//   if (output_format == "json") {
//     std::string json;
//     google::protobuf::util::JsonPrintOptions opts;
//     opts.add_whitespace = true;
//     opts.preserve_proto_field_names = true;
//     auto st = google::protobuf::util::MessageToJsonString(proto, &json, opts);
//     if (!st.ok()) {
//       LOG(QFATAL) << "MessageToJsonString failed: " << st.message();
//     }
//     TF_QCHECK_OK(WriteString(output, json));
//     return 0;
//   }

//   LOG(QFATAL) << "Unknown --output_format=" << output_format
//               << " (expected pb|pbtxt|json)";
}
