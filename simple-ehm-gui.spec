# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the GUI. Run from this folder: pyinstaller simple-ehm-gui.spec
a = Analysis(
    ["simple_ehm_gui.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("langs/it.json", "langs"),
        ("langs/en.json", "langs"),
        ("model.keras", "."),
        ("model.keras.version", "."),
        ("simple_ehm-runnable.py", "."),
    ],
    hiddenimports=["i18n", "update", "version"],
    excludes=["matplotlib", "seaborn", "IPython", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name="simple-ehm-gui",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
