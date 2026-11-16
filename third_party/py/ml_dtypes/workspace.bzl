# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

"""Provides the repo macro to import ml_dtypes.

ml_dtypes provides machine-learning-specific data-types like bfloat16,
float8 varieties, and int4.
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    ML_DTYPES_COMMIT = "7b0b6b80f90404b58771805c91d4e75286b8b62c"
    ML_DTYPES_SHA256 = "e817ad910c70fbf7808b914e42bb27484ec013fb93ed5b76ca40a92173136a67"
    tf_http_archive(
        name = "ml_dtypes_py",
        build_file = "//third_party/py/ml_dtypes:ml_dtypes_py.BUILD",
        link_files = {
            "//third_party/py/ml_dtypes:ml_dtypes.BUILD": "ml_dtypes/BUILD.bazel",
        },
        sha256 = ML_DTYPES_SHA256,
        strip_prefix = "ml_dtypes-{commit}".format(commit = ML_DTYPES_COMMIT),
        urls = tf_mirror_urls("https://github.com/jax-ml/ml_dtypes/archive/{commit}/ml_dtypes-{commit}.tar.gz".format(commit = ML_DTYPES_COMMIT)),
    )
