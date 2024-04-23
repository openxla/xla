
workspace=$1
cd $workspace/xla 

if [ -z ${SYCL_TOOLKIT_PATH+x} ];
then
  export SYCL_TOOLKIT_PATH=$workspace/oneapi/compiler/2024.1/
fi
bazel_bin=$(ls $workspace/bazel/)
./configure.py --backend=SYCL --host_compiler=GCC
$workspace/bazel/$bazel_bin build --config=verbose_logs -s --verbose_failures --nocheck_visibility //xla/tools:run_hlo_module
