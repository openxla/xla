#include "xla/hlo/dialect/hlo/parser/hlo_parser.h"
#include "xla/hlo/transforms/hlo_module_diff.h"

#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: my_diff a.hlo b.hlo\n";
        return 1;
    }

    auto mod_a_status = xla::ParseAndReturnUnverifiedModule(argv[1]);
    auto mod_b_status = xla::ParseAndReturnUnverifiedModule(argv[2]);

    if (!mod_a_status.ok() || !mod_b_status.ok()) {
        std::cerr << "Failed to parse HLO modules\n";
        return 1;
    }

    std::unique_ptr<xla::HloModule> mod_a = std::move(mod_a_status.value());
    std::unique_ptr<xla::HloModule> mod_b = std::move(mod_b_status.value());

    xla::HloMatcherConfig config;
    xla::HloModuleDiff diff(*mod_a, *mod_b, config);

    auto matches = diff.SuggestedMatches();

    for (const auto& m : matches) {
        std::cout << "A: " << m.instruction_a->ToString() << "\n";
        std::cout << "B: " << m.instruction_b->ToString() << "\n";
        std::cout << "----\n";
    }

    return 0;
}
