#!/usr/bin/env sh
# Usage: ./convert.sh /path/to/video [extra simple-ehm args...]
# Build once with: docker build -t simple-ehm .
# NOTE: for the default bind-mount to work, input files must be inside this folder.
set -eu
docker run --rm -v "$PWD:/simple_ehm" simple-ehm "$@"
