# Build Windows .exe (run in PowerShell from this folder).
# Requires: Python 3.11 + pip. ffmpeg must be installed separately (PATH).
$ErrorActionPreference = "Stop"
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller
pyinstaller simple-ehm-cli.spec
pyinstaller simple-ehm-gui.spec
Write-Host "Binaries in dist\: simple-ehm.exe (CLI), simple-ehm-gui.exe (GUI)"
