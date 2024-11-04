// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "filesystem.hpp"

#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"


namespace xla {
namespace common {
struct source_location
{
    std::string function = {};
    std::string file     = {};
    uint32_t    line     = 0;
    std::string context  = {};
};

using call_stack_t = std::vector<source_location>;

inline void
print_call_stack(std::string         ofname,
                 const call_stack_t& _call_stack,
                 const char*         env_variable = "ROCPROFILER_SAMPLE_OUTPUT_FILE")
{
    if(auto* eofname = getenv(env_variable)) ofname = eofname;

    std::ostream* ofs     = nullptr;
    auto          cleanup = std::function<void(std::ostream*&)>{};

    if(ofname == "stdout")
        ofs = &std::cout;
    else if(ofname == "stderr")
        ofs = &std::cerr;
    else
    {
        ofs = new std::ofstream{ofname};
        if(ofs && *ofs)
            cleanup = [](std::ostream*& _os) { delete _os; };
        else
        {
            std::cerr << "Error outputting to " << ofname << ". Redirecting to stderr...\n";
            ofname = "stderr";
            ofs    = &std::cerr;
        }
    }

    // LOG(INFO) << "Outputting collected data to " << ofname;

    size_t n = 0;
    for(const auto& itr : _call_stack)
    {
        *ofs << std::left << std::setw(2) << ++n << "/" << std::setw(2) << _call_stack.size()
             << " [" << common::fs::path{itr.file}.filename() << ":" << itr.line << "] "
             << std::setw(20) << itr.function;
        if(!itr.context.empty()) *ofs << " :: " << itr.context;
        *ofs << "\n";
    }

    *ofs << std::flush;

    if(cleanup) cleanup(ofs);
}
}  // namespace common
}  // namespace xla