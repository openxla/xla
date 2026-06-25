# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

rocm_redist = {
    "rocm_7.12.0_gfx94X": struct(
        packages = [
            {
                "url": "https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx94X-dcgpu-7.12.0.tar.gz",
                "sha256": "b88e1f167abe4cb3ab0d0c44431eed3ca1b77e1de6843e153c9ea6ac1e29f2f2",
            },
        ],
        required_softlinks = [],
        rocm_root = "",
    ),
    "rocm_7.12.0_gfx908": struct(
        packages = [
            {
                "url": "https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx908-7.12.0.tar.gz",
                "sha256": "8645100bd43761253114f175a6b5e5e928a72a437094e9e35d750ea089d41d6c",
            },
        ],
        required_softlinks = [],
        rocm_root = "",
    ),
    "rocm_7.12.0_gfx90a": struct(
        packages = [
            {
                "url": "https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx90a-7.12.0.tar.gz",
                "sha256": "d1dc2d3cb113e433cf3d3a77f8e414dfd9537b8e7d4f655df4c2d3604a736700",
            },
        ],
        required_softlinks = [],
        rocm_root = "",
    ),
}

def _parse_rocm_distro_links(distro_links):
    result = []
    if distro_links == "":
        return result

    for pair in distro_links.split(","):
        link = pair.split(":")
        result.append(struct(target = link[0], link = link[1]))
    return result

def create_rocm_distro(distro_url, distro_hash, symlinks):
    return struct(
        packages = [
            {
                "url": distro_url,
                "sha256": distro_hash,
            },
        ],
        required_softlinks = _parse_rocm_distro_links(symlinks),
        rocm_root = "",
    )
