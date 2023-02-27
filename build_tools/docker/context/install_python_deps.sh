# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set -euo pipefail

if ! [[ -f build_requirements.txt ]]; then
  echo "Couldn't find build_requirements.txt file in current directory" >&2
  exit 1
fi

PYTHON_VERSION="$1"

apt-get update

apt-get install -y \
  "python${PYTHON_VERSION}" \
  "python${PYTHON_VERSION}-dev"

update-alternatives --install /usr/bin/python3 python3 "/usr/bin/python${PYTHON_VERSION}" 1

apt-get install -y \
  python3-pip \
  python3-setuptools \
  python3-distutils \
  python3-venv \
  "python${PYTHON_VERSION}-venv"

# Note that we use --ignore-installed when installing packages that may have
# been auto-installed by the OS package manager (i.e. PyYAML is often an
# implicit OS-level dep). This should not break so long as we do not
# subsequently reinstall it on the OS side. Failing to do this will yield a
# hard error with pip along the lines of:
#   Cannot uninstall 'PyYAML'. It is a distutils installed project and thus we
#   cannot accurately determine which files belong to it which would lead to
#   only a partial uninstall.
python3 -m pip install --ignore-installed --upgrade -r build_requirements.txt
