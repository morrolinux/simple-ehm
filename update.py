"""Update logic: program version check + model download only if remote is newer.

Conventions on GitHub (repo overridable via SIMPLE_EHM_REPO, default below):
- Program: standard GitHub Release with tag like "v2.0.1".
  API: https://api.github.com/repos/{repo}/releases/latest -> tag_name, html_url
- Models: release asset "manifest.json", fallback raw file
  https://raw.githubusercontent.com/{repo}/main/models/manifest.json
  manifest format:
    {"model.keras": {"version": "2026.10.07", "url": "<full https url>",
                     "sha256": "<optional hex>", "size": 6812345}}
- Local state: sidecar "<model>.version" text file with the bundled version.
  The binary ships model.keras + model.keras.version; download happens ONLY
  if remote version differs (or file missing).

All network failures are non-fatal: caller decides (warn and continue offline).
Stdlib only (urllib + json) so --help / binary work without extra deps.
"""
import json
import os
import urllib.request
from contextlib import suppress
from pathlib import Path

from version import __version__

REPO = os.environ.get("SIMPLE_EHM_REPO", "morrolinux/simple-ehm")
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
MANIFEST_ASSET = (
    f"https://github.com/{REPO}/releases/latest/download/manifest.json"
)
MANIFEST_RAW = (
    f"https://raw.githubusercontent.com/{REPO}/main/models/manifest.json"
)
TIMEOUT = 15


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "simple-ehm"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def check_program_update(current: str = __version__, repo: str = REPO):
    """Return (has_update, latest_tag, release_url). Never raises."""
    try:
        data = _get_json(
            f"https://api.github.com/repos/{repo}/releases/latest"
        )
        tag = str(data.get("tag_name", "")).strip()
        url = str(data.get("html_url", ""))
        if not tag:
            return False, current, ""
        norm = lambda v: v.lstrip("v").strip()  # noqa: E731
        has = norm(tag) != norm(current)
        # naive "newer" check: different tag counts as update available
        return has, tag, url
    except Exception:
        return False, current, ""


def _get_manifest(repo: str = REPO):
    for url in (
        f"https://github.com/{repo}/releases/latest/download/manifest.json",
        f"https://raw.githubusercontent.com/{repo}/main/models/manifest.json",
    ):
        try:
            return _get_json(url), url
        except Exception:
            continue
    return {}, ""


def local_model_version(model_path: Path) -> str:
    sidecar = Path(str(model_path) + ".version")
    for cand in (sidecar, model_path.with_suffix(".version"),
                 model_path.parent / "model.version"):
        with suppress(Exception):
            if cand.is_file():
                v = cand.read_text(encoding="utf-8").strip()
                if v:
                    return v
    return ""


def _write_version(model_path: Path, version: str):
    with suppress(Exception):
        Path(str(model_path) + ".version").write_text(version, encoding="utf-8")


def _verify(path: Path, sha256: str = "", size: int = 0) -> bool:
    try:
        if size and path.stat().st_size != size:
            return False
        if sha256:
            import hashlib
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest().lower() == sha256.lower()
        return path.stat().st_size > 100_000
    except Exception:
        return False


def _seed_from_bundle(target: Path):
    """Copy model bundled in the frozen binary (or script dir) to the target."""
    import sys
    cands = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cands.append(Path(meipass) / target.name)
    with suppress(Exception):
        cands.append(Path(__file__).resolve().parent / target.name)
    for cand in cands:
        if cand.is_file() and cand.resolve() != target.resolve():
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copyfile(cand, target)
                for ext in (".version",):
                    sv = Path(str(cand) + ext)
                    if sv.is_file():
                        shutil.copyfile(sv, Path(str(target) + ext))
                return True
            except Exception:
                continue
    return False


def ensure_model_smart(model_arg: str, base_dir: Path,
                       model_url: str = "", no_download: bool = False,
                       repo: str = REPO):
    """Return (path, changed, note). Downloads ONLY if remote version is newer.

    - explicit file/dir in --model that exists -> use it, no network.
    - --model-url set -> legacy behaviour: download only if target missing.
    - else: compare local .version sidecar vs remote manifest; download the
      model.keras entry only when versions differ.
    Raises SystemExit with human message when download needed but impossible.
    """
    from pathlib import Path as _P
    p = _P(model_arg)
    if p.exists() and (p.is_file() or p.is_dir()):
        return p, False, "local"
    target = p if p.suffix == ".keras" else (base_dir / "model.keras")
    if not target.is_file():
        _seed_from_bundle(target)  # frozen binary ships its own copy
    if target.is_file() and not model_url:
        # may still be updateable -> check manifest for newer version
        local_v = local_model_version(target)
        manifest, _ = _get_manifest(repo)
        entry = (manifest or {}).get(target.name, {})
        remote_v = str(entry.get("version", "")).strip()
        if not remote_v or (local_v and remote_v == local_v):
            return target, False, "up-to-date" if remote_v else "local"
        if no_download:
            return target, False, "update-available"
        # fall through to download remote entry
        return _download_entry(target, entry, repo)
    if target.is_file():  # --model-url given but file exists: keep local
        return target, False, "local"
    if no_download:
        raise SystemExit(f"Model not found: {p} (downloads disabled).")
    if model_url:
        entry = {"version": "", "url": model_url}
        return _download_entry(target, entry, repo)
    manifest, _ = _get_manifest(repo)
    entry = (manifest or {}).get(target.name, {})
    if not entry.get("url"):
        raise SystemExit(
            f"Model not found: {target} and no manifest entry on GitHub "
            f"(repo {repo}). Publish models/manifest.json + model.keras "
            f"as release assets, or pass --model / --model-url.")
    return _download_entry(target, entry, repo)


def _download_entry(target: Path, entry: dict, repo: str):
    url = entry.get("url", "")
    if not url:  # last resort: release asset with same name
        url = f"https://github.com/{repo}/releases/latest/download/{target.name}"
    print(f"Downloading model v{entry.get('version', '?')} from {url} ...")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".keras.downloading")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "simple-ehm"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        if not _verify(tmp, entry.get("sha256", ""), int(entry.get("size", 0) or 0)):
            raise RuntimeError("downloaded file failed verification (sha256/size)")
        tmp.replace(target)
        if entry.get("version"):
            _write_version(target, str(entry["version"]))
        print(f"Model saved to {target}")
        return target, True, "downloaded"
    except Exception as e:
        with suppress(Exception):
            tmp.unlink()
        raise SystemExit(f"Model download failed: {e}")
