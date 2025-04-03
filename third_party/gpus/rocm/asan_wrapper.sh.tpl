#!/bin/bash

set -e

export ASAN_OPTIONS="suppressions=$(pwd)/external/local_config_rocm/rocm/asan_blacklist.txt"
exec "$@"
