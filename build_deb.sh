#!/usr/bin/env bash
# Build .deb from the PyInstaller Linux binaries (run AFTER build_linux.sh).
set -eux
VER=$(python3 -c "import re; print(re.search(r'\"([^\"]+)\"', open('version.py').read()).group(1))")
PKG="simple-ehm_${VER}_amd64"
rm -rf "$PKG"
mkdir -p "$PKG/DEBIAN" \
  "$PKG/usr/share/simple-ehm" \
  "$PKG/usr/bin" \
  "$PKG/usr/share/applications" \
  "$PKG/usr/share/icons/hicolor/1024x1024/apps"
cp dist/simple-ehm dist/simple-ehm-gui "$PKG/usr/share/simple-ehm/"
cp assets/logo-icon.png "$PKG/usr/share/icons/hicolor/1024x1024/apps/simple-ehm.png"
ln -s /usr/share/simple-ehm/simple-ehm "$PKG/usr/bin/simple-ehm"
ln -s /usr/share/simple-ehm/simple-ehm-gui "$PKG/usr/bin/simple-ehm-gui"
cat > "$PKG/usr/share/applications/simple-ehm.desktop" <<'EOF'
[Desktop Entry]
Name=simple-ehm
Comment=Remove "ehm" filler sounds from videos
Exec=simple-ehm-gui
Icon=simple-ehm
Terminal=false
Type=Application
Categories=AudioVideo;Video;
EOF
cat > "$PKG/DEBIAN/control" <<EOF
Package: simple-ehm
Version: $VER
Section: video
Priority: optional
Architecture: amd64
Depends: ffmpeg, python3-tk
Maintainer: simple-ehm <simple-ehm@localhost>
Description: Remove "ehm" filler sounds from videos (AI powered).
 CLI (simple-ehm) + GUI (simple-ehm-gui), IT/EN auto language,
 NVIDIA GPU auto-detect, models auto-update from GitHub.
EOF
dpkg-deb --build "$PKG"
echo "Built: $PKG.deb (install with: sudo apt install ./$PKG.deb)"
