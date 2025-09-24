#!/bin/bash

TAG_FILTERS=(
    -no_gpu
    -requires-gpu-intel
    -requires-gpu-nvidia
    -no_oss
    -oss_excluded
    -oss_serial
    -cuda-only
    -oneapi-only
    -requires-gpu-sm60
    -requires-gpu-sm60-only
    -requires-gpu-sm70
    -requires-gpu-sm70-only
    -requires-gpu-sm80
    -requires-gpu-sm80-only
    -requires-gpu-sm86
    -requires-gpu-sm86-only
    -requires-gpu-sm89
    -requires-gpu-sm89-only
    -requires-gpu-sm90
    -requires-gpu-sm90-only
)

echo $(IFS=, ; echo "${TAG_FILTERS[*]}")
