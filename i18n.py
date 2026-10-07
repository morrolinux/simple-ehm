"""Shared i18n: automatic language from the PC (Windows + Linux).

Order: SIMPLE_EHM_LANG env -> Windows API -> POSIX LANG/LC_* -> locale module.
Only 'it'/'en' ship in langs/; anything else falls back to 'en'.
Stdlib only so CLI --help, GUI and frozen binaries all work.
"""
import json
import os
from pathlib import Path

SUPPORTED = ("it", "en")


def _norm(tag: str) -> str:
    t = (tag or "").strip().lower().replace("_", "-")
    if t.startswith("it"):
        return "it"
    return "en" if not t else ("it" if t == "it" else "en")


def detect_lang() -> str:
    # 1) explicit override
    env = os.environ.get("SIMPLE_EHM_LANG", "").strip()
    if env and env.lower() != "auto":
        return _norm(env)
    # 2) Windows native API (works even when env vars are missing)
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(85)
        if ctypes.windll.kernel32.GetUserDefaultLocaleName(buf, 85):
            return _norm(buf.value)
    except Exception:
        pass
    # 3) POSIX env (Linux/macOS): LC_ALL > LC_MESSAGES > LANG
    for var in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        v = os.environ.get(var, "")
        if v:
            return _norm(v.split(".")[0].split(":")[0])
    # 4) locale module (deprecated getdefaultlocale avoided)
    try:
        import locale
        loc = (locale.getlocale()[0] or "")
        if loc:
            return _norm(loc)
    except Exception:
        pass
    return "en"


def load_strings(base_dir: Path, lang: str) -> dict:
    for code in (lang, "en"):
        try:
            p = Path(base_dir) / "langs" / f"{code}.json"
            if p.is_file():
                return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return {}


def resolve_lang(code: str) -> str:
    if code and code != "auto":
        return _norm(code)
    return detect_lang()
