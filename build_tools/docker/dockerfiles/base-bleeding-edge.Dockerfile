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

# A Docker image that contains all the latest stuff.

# 22.04
FROM ubuntu@sha256:4b1d0c4a2d2aaf63b37111f34eb9fa89fa1bf53dd6e4ca954d47caebca4005c2

SHELL ["/bin/bash", "-e", "-u", "-o", "pipefail", "-c"]

# Disable apt-key parse waring. If someone knows how to do whatever the "proper"
# thing is then feel free. The warning complains about parsing apt-key output,
# which we're not even doing.
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

######## Basic stuff ########
WORKDIR /install-basics
# Useful utilities for building child images. Best practices would tell us to
# use multi-stage builds
# (https://docs.docker.com/develop/develop-images/multistage-build/) but it
# turns out that Dockerfile is a thoroughly non-composable awful format and that
# doesn't actually work that well. These deps are pretty small.
RUN apt-get update \
  && apt-get install -y \
    git \
    unzip \
    wget \
    curl \
    gnupg2 \
  && rm -rf /install-basics

######## Bazel ########
WORKDIR /install-bazel
COPY build_tools/docker/context/install_bazel.sh .bazelversion ./
RUN ./install_bazel.sh && rm -rf /install-bazel

##############

WORKDIR /install-python

# Max supported Python version
ARG PYTHON_VERSION=3.10

# Versions for things required to build IREE. We install the latest version of
# packages in the requirements file. Unlike most places in the project, these
# dependencies are *not* pinned in general. This adds non-determinism to the
# Docker image build, but reduces maintenance burden. If a new build fails
# because of a new package version, we should add a max version constraint to
# the requirement.
COPY build_tools/docker/python/build_requirements.txt build_tools/docker/context/install_python_deps.sh ./
RUN ./install_python_deps.sh "${PYTHON_VERSION}" \
  && rm -rf /install-python

ENV PYTHON_BIN /usr/bin/python3

##############

### Clean up
RUN apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /
