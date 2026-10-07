# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the CLI. Run from this folder: pyinstaller simple-ehm-cli.spec
a = Analysis(
    ["simple_ehm-runnable.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("langs/it.json", "langs"),
        ("langs/en.json", "langs"),
        ("model.keras", "."),
        ("model.keras.version", "."),
    ],
    hiddenimports=["i18n", "update", "version"],
    excludes=["matplotlib", "seaborn", "cv2", "IPython", "notebook",
              "torch", "torchvision", "torchaudio", "pandas", "scipy", "jax"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name="simple-ehm",
    debug=False,
    strip=False,
    upx=False,
    console=True,
)
