
workspace=$1
cd $workspace/xla
bazel_version=$(head -1 .bazelversion)
bazel_base_url="https://github.com/bazelbuild/bazel/releases/download"
bazel_url=$bazel_base_url/$bazel_version/"bazel-$bazel_version-linux-x86_64"
echo $bazel_url
mkdir -p $workspace/bazel
cd $workspace/bazel
wget $bazel_url
bazel_bin=$(ls)
chmod +x $bazel_bin
