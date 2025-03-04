/* Copyright 2022 The OpenXLA Authors.

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

#include <fstream>
#include <sstream>
#include <regex>
#include "xla/error_spec.h"
#include "xla/literal_comparison.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_utils.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"

namespace xla {
namespace gpu {

namespace {

template <class T>
std::vector<T*> MakePointerVector(std::vector<T>& input_vec) {
  std::vector<T*> output_pointers;
  output_pointers.reserve(input_vec.size());
  for (auto& input : input_vec) {
    output_pointers.push_back(&input);
  }
  return output_pointers;
}

void WriteLiteralToTempFile(const LiteralSlice& literal, const std::string& name) {
  // Bazel likes for tests to write "debugging outputs" like these to
  // TEST_UNDECLARED_OUTPUTS_DIR.  This plays well with tools that inspect test
  // results, especially when they're run on remote machines.
  std::string outdir;
  const char* undeclared_outputs_dir = getenv("TEST_TMPDIR");
  if (undeclared_outputs_dir != nullptr) {
    outdir = undeclared_outputs_dir;
  } else {
    outdir = tsl::testing::TmpDir();
  }

  auto* env = tsl::Env::Default();
  std::string filename = outdir + "/" + name;
  TF_CHECK_OK(tsl::WriteBinaryProto(env, absl::StrCat(filename, ".pb"),
                                           literal.ToProto()));
  TF_CHECK_OK(tsl::WriteStringToFile(env, absl::StrCat(filename, ".txt"),
                                           literal.ToString()));
  LOG(ERROR) << "wrote Literal to " << name << " file: " << filename
             << ".{pb,txt}";
}

absl::StatusOr<Literal> ReadLiteralFromFile(const std::string& name) {

  std::string baseDir = "/data/tf-default/ref_files/";
  auto* env = tsl::Env::Default();
  auto path = baseDir + name + ".pb";

  if (!env->FileExists(path).ok()) {
    VLOG(0) << "Unable to find file: " << path;
    return absl::InternalError("ops");
  }

  LiteralProto proto;
  TF_RETURN_IF_ERROR(tsl::ReadBinaryProto(env, path, &proto));

  return Literal::CreateFromProto(proto);
}

template < class Func >
absl::StatusOr<Literal> MakeVarLiteral(const Shape& shape, Func&& F) {
  
  if (shape.IsTuple()) {
    std::vector<Literal> elements;
    for (const Shape& element_shape : shape.tuple_shapes()) {
      TF_ASSIGN_OR_RETURN(Literal element, MakeVarLiteral(element_shape, 
                                              std::move(F)));
      elements.push_back(std::move(element));
    }
    return LiteralUtil::MakeTupleOwned(std::move(elements));
  }
  Literal literal(shape);
  primitive_util::ArrayTypeSwitch<void>(
    [&](auto prim_const) {
      using T = primitive_util::NativeTypeOf<prim_const>;
      int idx = 0;
      for (T& val : literal.data<T>()) {
        val = (T)std::forward<Func>(F)(idx++);
      }
    },
    shape.element_type());
  return std::move(literal);
}

absl::StatusOr<std::vector<Literal>> MakeSpecialArguments(HloModule* const module) {

  const auto params = module->entry_computation()->parameter_instructions();
  using T = double;
  std::vector<Literal> arguments(params.size());
  for (int i = 0; i < params.size(); ++i) {
    TF_ASSIGN_OR_RETURN(arguments[i], 
        MakeVarLiteral(params[i]->shape(), 
          [](int idx){ return idx + 1; }
        ));
  }
  return std::move(arguments);
}

} // namespace 


#define DO_REFERENCE_CHECK 1
#define USE_MULTIPLE_GPUS 0
#define USE_SPECIAL_ARGUMENTS 0

class HloRunnerTest : public GpuCodegenTest {

protected:
  constexpr static const char *CsvSep = " , ";

  void run_internal(std::istream& ifs, std::ostream& ofs, const std::string& name) {

    const static std::pair< std::regex, const char *> replaces[] = {
        { std::regex(R"x(\[\(0\),)x"), "[" }, // remove dynamic shapes
        { std::regex(R"x(u32\[\] get-dimension-size\()x"), "s32[] get-dimension-size(" },
        { std::regex(R"x(u32\[\] %get-dimension-size\.)x"), "s32[] %get-dimension-size."},
        { std::regex(R"x(u32\[\] get-dimension-size\.)x"), "s32[] get-dimension-size."},
    };
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    auto input = buffer.str();
    for (const auto& rep : replaces) {
      input = std::regex_replace(input, rep.first, rep.second);
    }

    HloModuleConfig config = GetModuleConfigForTest();
#if !USE_MULTIPLE_GPUS
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(input, 
          config));
  
#if !USE_SPECIAL_ARGUMENTS
  TF_ASSERT_OK_AND_ASSIGN(auto fake_arguments, xla::MakeFakeArguments(module.get(), 
        true, /*pseudo-random*/
        false /* use large range*/));
#else
  TF_ASSERT_OK_AND_ASSIGN(auto fake_arguments, MakeSpecialArguments(module.get()));
#endif
  auto arg_ptrs = MakePointerVector<xla::Literal>(fake_arguments);

  auto ref_module = module->Clone();  
  TF_ASSERT_OK_AND_ASSIGN(auto exec, CreateExecutable(std::move(module), true));

  //  TF_ASSERT_OK_AND_ASSIGN(auto truth, 
  //        ReadLiteralFromProto("/tf/xla/expected.pb"));
  // TF_ASSERT_OK_AND_ASSIGN(auto truth, 
  // ref_runner.ExecuteWithExecutable(ref_exec.get(), arg_ptrs, nullptr));
  // WriteLiteralToTempFile(truth, "expected");
  //VLOG(0) << "Got expected literal from file.. running test";

  auto& runner = test_runner_as_hlo_runner();

  int num_runs = 0, num_warmups = 0;
  TF_ASSERT_OK_AND_ASSIGN(auto argument_buffers,
                      runner.TransferLiteralsToDevice(arg_ptrs));
  
  uint64_t timeNs = 0;
  for(int i = 0; i < num_runs + num_warmups; i++) {
    if(i == num_warmups) {
      VLOG(0) << "Warmup finished.. running";
      ASSERT_TRUE(backend().default_stream_executor()->SynchronizeAllActivity());
    }
    xla::ExecutionProfile profile;
    //profile.set_warmup_run_executed(true);
    TF_ASSERT_OK_AND_ASSIGN(auto result,
                      runner.ExecuteWithDeviceBuffers(
                          /*executable=*/exec.get(),
                          /*arguments=*/argument_buffers,
                          /*profile=*/&profile));
    if (i == 0) {
      TF_ASSERT_OK_AND_ASSIGN(auto host_res, 
                runner.TransferLiteralFromDevice(result.Result()));
      //WriteLiteralToTempFile(host_res, name); // write execution results to file
    }
    if (i >= num_warmups) timeNs += profile.compute_time_ns();
    //VLOG(0) << i << " compute time: " << profile.compute_time_ns();
  }
  double usec = (double)timeNs  / (num_runs * 1000);
  VLOG(0) << "Time elapsed: " << usec << " usec";
  ofs << usec;

#if DO_REFERENCE_CHECK
  VLOG(0) << "Performing correctness check.";
  TF_ASSERT_OK_AND_ASSIGN(auto test_res,
                      runner.ExecuteWithExecutable(
                          /*executable=*/exec.get(),
                          /*arguments=*/arg_ptrs));
  VLOG(0) << test_res.ToString();
 
  auto& ref_runner = reference_runner();
  TF_ASSERT_OK_AND_ASSIGN(auto truth, ref_runner.Execute(std::move(ref_module),
                                       fake_arguments, /*run_hlo_passes*/true));
  
  // TF_ASSERT_OK_AND_ASSIGN(auto truth, ReadLiteralFromFile(name));
  
  // VLOG(0) << "Running reference exec..";
  // auto& ref_runner = HloTestBase::reference_runner_;
  // TF_ASSERT_OK_AND_ASSIGN(
  //      auto ref_exec, ref_runner.CreateExecutable(std::move(ref_module), true));

  // TF_ASSERT_OK_AND_ASSIGN(
  //      auto truth, ref_runner.ExecuteWithExecutable(ref_exec.get(), arg_ptrs));

  // //ErrorSpec error_spec{1e-2, 1e-3};
  ErrorSpec error_spec(1e-5 /*abs*/, 1e-5 /*rel*/);
  ASSERT_EQ(literal_comparison::Near(/*expected=*/truth,
                                   /*actual=*/test_res,
                                   /*error=*/error_spec,
                            /*detailed_message=*/true, {}), absl::OkStatus());
#endif // DO_REFERENCE_CHECK
 //    EXPECT_TRUE(RunAndCompare(std::move(module), 
  // //     absl::Span< xla::Literal * const>(arg_ptrs.data(), arg_ptrs.size()), error_spec));
#else // USE_MULTIPLE_GPUS
  int NumReplicas = 8, NumParts = 1;
  config.set_replica_count(NumReplicas);
  config.set_num_partitions(NumParts);

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(input, config));
  DeviceAssignment assn(/*replica_count=*/NumReplicas,
                        /*computation_count=*/NumParts);
  for (int64_t i = 0, k = 0; i < NumReplicas; i++)
  for (int64_t j = 0; j < NumParts; j++) {
    assn(i, j) = k++;
  }

  auto fake_arguments = xla::MakeFakeArguments(
      module.get(),
      true, /*pseudo-random*/
      false /* use large range*/).ValueOrDie();
  TF_ASSERT_OK_AND_ASSIGN(auto exec, 
      test_runner_.CreateExecutable(std::move(module), true));

 for(int i = 0; i < 10; i++) {
   VLOG(0) << "Running iteration #" << i;
   TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
         HloTestBase::ExecuteReplicated(
          [&](int64_t){ return exec.get(); },
          [&fake_arguments](int64_t replica_id)
          { return fake_arguments.size(); },
          [&fake_arguments](int64_t replica_id, int64_t idx)
          { return &fake_arguments[idx]; },
          NumReplicas, false /*run hlo*/, &assn));
   ASSERT_EQ(results.size(), NumReplicas);
 }
#endif // USE_MULTIPLE_GPUS
  }

};

TEST_F(HloRunnerTest, RunSingle) {
  
  if (std::ifstream ifs("input.hlo"); ifs.good()) {
    std::ofstream ofs;
    return run_internal(ifs, ofs, "input");
  }
  std::ifstream ifs("pattern.txt");
  if (!ifs.good()) {
    GTEST_SKIP() << "No input files provided!";
  }

  ASSERT_TRUE(ifs.good());

  std::string line;
  std::getline(ifs, line);
  VLOG(0) << "Using file pattern: " << line;

  auto env = tsl::Env::Default();
  std::string csv("hlo_runner_results.csv");
  bool exists = env->FileExists(csv).ok();
  std::ofstream ofs(csv, std::ios_base::app);

  std::vector<std::string> matches, short_names;
  ASSERT_TRUE(env->GetMatchingPaths(line, &matches).ok());
  std::sort(matches.begin(), matches.end());
  short_names.resize(matches.size());
  
  if (!exists) ofs << CsvSep; // add one column for the header

  const std::regex match_no(R"x([A-Za-z_]*([0-9]+).*)x");

  for(size_t i = 0; i < matches.size(); i++) {
    auto s = matches[i];
    auto res = s.find_last_of('/');
    if (res != std::string::npos) s = s.substr(res + 1);
    res = s.find_last_of(".");
    if (res != std::string::npos) s = s.substr(0, res);
    
    short_names[i] = s;
    std::smatch base_match;
    if (std::regex_match(s, base_match, match_no)) {
      if (base_match.size() == 2) {
        s = base_match[1].str();
        short_names[i] = "module_" + s;
      }
    } 
    if (!exists) {
      ofs << s << (i == matches.size() - 1 ? "\n" : CsvSep);
    }
  }

  ofs << "v0.5.0" << CsvSep;
  for(size_t i = 0; i < matches.size(); i++) {
    auto s = matches[i];
    std::ifstream ifs(s);
    if (!ifs.good()) {
      VLOG(0) << "Skipping file: " << s;
      ofs << CsvSep;
      continue;
    }
    VLOG(0) << i << " of " << matches.size() << ": HLO test for: " 
            << s << " ---------------------";
   
    run_internal(ifs, ofs, short_names[i]);
    ofs << (i == matches.size() - 1 ? "\n" : CsvSep);
    std::flush(ofs);
  }
}

}  // namespace gpu
}  // namespace xla
 