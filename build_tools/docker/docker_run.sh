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

# Drawn from https://github.com/openxla/iree/blob/95c56190d75725ca4a7251ba8b4ac4edd9b8970d/build_tools/docker/docker_run.sh

set -euo pipefail

# Doing things this way allows runners to change their working directory,
# ensuring that different runners use the correct path names. This also enables
# local reproduction of issues.
DOCKER_CONTAINER_WORKDIR="${DOCKER_CONTAINER_WORKDIR:-/work}"

# Sets up files and environment to enable running scripts in docker.
# In particular, does some shenanigans to enable running with the current user.
# Some of this setup is only strictly necessary for Bazel, but it doesn't hurt
# for anything else.
# Requires that DOCKER_HOST_WORKDIR and DOCKER_HOST_TMPDIR have been set
function docker_run() {
    # Make the source repository available and launch containers in that
    # directory.
    DOCKER_RUN_ARGS=(
      --mount="type=bind,source=${DOCKER_HOST_WORKDIR},dst=${DOCKER_CONTAINER_WORKDIR}"
      --workdir="${DOCKER_CONTAINER_WORKDIR}"
      --env "CCACHE_BASE_DIR=${DOCKER_CONTAINER_WORKDIR}"
    )

    # Delete the container after the run is complete.
    DOCKER_RUN_ARGS+=(--rm)


    # Run as the current user and group. If only it were this simple...
    DOCKER_RUN_ARGS+=(--user="$(id -u):$(id -g)")

    # Use the host network stack. Improves network performance and makes it
    # possible for the container to talk to localhost.
    DOCKER_RUN_ARGS+=(--network="host")

    # The Docker container doesn't know about the users and groups of the host
    # system. We have to tell it. This is just a mapping of IDs to names though.
    # The thing that really matters is the IDs, so the key thing is that Docker
    # writes files as the same ID as the current user, which we set above, but
    # without the group and passwd file, lots of things get upset because they
    # don't recognize the current user ID (e.g. `whoami` fails). Bazel in
    # particular looks for a home directory and is not happy when it can't find
    # one.
    # So we make the container share the host mapping, which guarantees that the
    # current user is mapped. If there was any user or group in the container
    # that we cared about, this wouldn't necessarily work because the host and
    # container don't necessarily map the ID to the same user. Luckily though,
    # we don't.
    # We don't just mount the real /etc/passwd and /etc/group because Google
    # Linux workstations do some interesting stuff with user/group permissions
    # such that they don't contain the information about normal users and we
    # want these scripts to be runnable locally for debugging.
    # Instead we dump the results of `getent` to some fake files.
    local fake_etc_dir="${DOCKER_HOST_TMPDIR}/fake_etc"
    mkdir -p "${fake_etc_dir}"

    local fake_group="${fake_etc_dir}/group"
    local fake_passwd="${fake_etc_dir}/passwd"

    getent group > "${fake_group}"
    getent passwd > "${fake_passwd}"

    DOCKER_RUN_ARGS+=(
      --mount="type=bind,src=${fake_group},dst=/etc/group,readonly"
      --mount="type=bind,src=${fake_passwd},dst=/etc/passwd,readonly"
    )


    # Bazel stores its cache in the user home directory by default. It's
    # possible to override this, but that would require changing our Bazel
    # startup options, which means polluting all our scripts and making them not
    # runnable locally. Instead, we give it a special home directory to write
    # into. We don't just mount the user home directory (or some subset thereof)
    # for two reasons:
    #   1. We probably don't want Docker to just write into the user's home
    #      directory when running locally.
    #   2. This allows us to control the device the home directory is in. Bazel
    #      tends to be IO bound at even moderate levels of CPU parallelism and
    #      the difference between a persistent SSD and a local scratch SSD can
    #      be huge. In particular, Kokoro has the home directory on the former
    #      and the work directory on the latter.
    local fake_home_dir="${DOCKER_HOST_TMPDIR}/fake_home"
    mkdir -p "${fake_home_dir}"

    DOCKER_RUN_ARGS+=(
      --mount="type=bind,src=${fake_home_dir},dst=${HOME}"
    )

    # Make gcloud credentials available if they are present. This isn't
    # necessary when running in GCE but enables using this script locally with
    # remote caching.
    if [[ -d "${HOME}/.config/gcloud" ]]; then
      DOCKER_RUN_ARGS+=(
        --mount="type=bind,src=${HOME}/.config/gcloud,dst=${HOME}/.config/gcloud,readonly"
      )
    fi

    # Give the container a ramdisk and set the Bazel sandbox base to point to
    # it. This helps a lot with Bazel getting IO bound. Note that SANDBOX_BASE
    # is a custom environment variable we translate into the corresponding Bazel
    # option.
    DOCKER_RUN_ARGS+=(
      --mount="type=tmpfs,dst=/dev/shm"
      --env SANDBOX_BASE=/dev/shm
    )

    # Some scripts need elevated permissions to control system-level scheduling.
    # Since we're not using Docker for sandboxing, it is fine to run in
    # privileged mode.
    DOCKER_RUN_ARGS+=(
      --privileged
    )

    docker run "${DOCKER_RUN_ARGS[@]}" "$@"
}

docker_run "$@"
