bazel build //xla/hlo/tools/hlo_diff:print_matches   --repo_env TF_NEED_CUDA=0
bazel build //xla/tools:run_hlo_module --config=cuda
bazel run //xla/tools:hlo_to_html -- \
  --input_format=hlo --output=/home/linux/Documents/test_hlo_to_html/mod.html /path/to/module.hlo

run_hlo_module \
  --input_format=hlo \
  --platform=CUDA \
  --reference_platform="" \
  --input_literals_file=empty_inputs.pb \
  --print_literals=true \
  your_module.hlo

bazel run //xla/tools:hlo_print_test /home/linux/Documents/constable_impl/t5/t5_hlo.txt