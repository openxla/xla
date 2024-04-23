
if [ -z ${SYCL_TOOLKIT_PATH+x} ];
then
workspace=$1
action=$2
echo "Install Intel OneAPI in $workspace/oneapi"
cd $workspace
mkdir -p oneapi
if ! [ -f $workspace/l_BaseKit_p_2024.1.0.596.sh ]; then
  echo "Download oneAPI package"
  wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/fdc7a2bc-b7a8-47eb-8876-de6201297144/l_BaseKit_p_2024.1.0.596.sh
fi
bash l_BaseKit_p_2024.1.0.596.sh -a -s --eula accept --action $action --install-dir $workspace/oneapi --log-dir $workspace/oneapi/log --download-cache $workspace/oneapi/cache --components=intel.oneapi.lin.dpcpp-cpp-compiler:intel.oneapi.lin.mkl.devel
else
  echo "SYCL_TOOLKIT_PATH set to $SYCL_TOOLKIT_PATH", skip install/remove oneAPI;
fi
