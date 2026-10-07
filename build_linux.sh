#!/usr/bin/env bash
# Build Linux onefile binaries (run on Debian/Ubuntu from this folder).
# Requires: python3.11 + pip + ffmpeg. For NVIDIA GPU use the CUDA TF build:
#   pip install "tensorflow[and-cuda]>=2.16,<2.22"  (else CPU-only, fine too)
set -eux
sudo apt-get update
sudo apt-get install -y ffmpeg python3-tk
pip3 install --upgrade pip
pip3 install -r requirements.txt
pip3 install pyinstaller
pyinstaller simple-ehm-cli.spec
pyinstaller simple-ehm-gui.spec
echo "Binaries in dist/: simple-ehm (CLI), simple-ehm-gui (GUI)"
