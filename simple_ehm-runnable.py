#!/usr/bin/env python3
"""simple-ehm: remove filler sounds ("ehm") from pre-recorded speeches.

Modernized for Python 3.10-3.13, TensorFlow 2.16-2.21 (Keras 3) and Windows/Linux.
CLI is backward compatible with the original script.
"""
import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------- paths
def _base_dir() -> Path:
    """Dir with bundled files: script dir, or PyInstaller _MEIPASS when frozen."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parent


def _data_dir() -> Path:
    """Writable dir for model/tmp: user profile when frozen, script dir else."""
    if getattr(sys, "frozen", False):
        if os.name == "nt":
            d = Path(os.environ.get("APPDATA", str(Path.home()))) / "simple-ehm"
        else:
            d = Path(os.environ.get("XDG_CACHE_HOME",
                                    str(Path.home() / ".cache"))) / "simple-ehm"
        d.mkdir(parents=True, exist_ok=True)
        return d
    return _base_dir()


BASE_DIR = _base_dir()
DATA_DIR = _data_dir()
# Prefer an existing model: user data (frozen) / script dir, .keras then legacy.
_CANDS = [DATA_DIR / "model.keras", BASE_DIR / "model.keras",
          DATA_DIR / "model", BASE_DIR / "model"]
DEFAULT_MODEL_DIR = next((c for c in _CANDS if c.exists()), DATA_DIR / "model.keras")
DEFAULT_TMP_DIR = DATA_DIR / "tmp"
DEFAULT_TD_DIR = DATA_DIR / "training_data"

# ---------------------------------------------------------------- CLI (stdlib only, so --help works without TF)
parser = argparse.ArgumentParser(
    description='Remove "ehm" filler sounds from a video file (AI powered).'
)
parser.add_argument("filename", nargs="?", help="video file name (or full file path) to classify")
parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR),
                    help="path to the SavedModel / .keras model (default: model.keras if present, else model/)")
parser.add_argument("--model-url", default=os.environ.get("SIMPLE_EHM_MODEL_URL", ""),
                    help="URL to download model.keras from on first run "
                    "(default: SIMPLE_EHM_MODEL_URL env or built-in GitHub release URL)")
parser.add_argument("--no-download", default=False, action="store_true",
                    help="never download the model automatically (fail if missing)")
parser.add_argument("--fastcut", default=False, action='store_true',
                    help="cut and merge an mp4 video without re-encoding using an edit list."
                    "Might not work on some players. see https://stackoverflow.com/a/18449609")
parser.add_argument("--window-size-divide", type=float, default=1,
                    help="divide window size (default: 1s) by this factor")
parser.add_argument("--window-slide-divide", type=float, default=2,
                    help="divide the window slide by this factor (default: half the window size)")
parser.add_argument("--fps", type=int, default=-1,
                    help="frames per second of the encoded video. Lower FPS mean faster encoding (default: original)")
parser.add_argument("--crf", type=int, default=-1, help="CRF factor for h264 encoding.")
parser.add_argument("--spectrogram", default=False, action='store_true',
                    help="show spectrogram window during analysis (debugging only, needs GUI)")
parser.add_argument("--generate-training-data", default=False, action='store_true',
                    help="export extracted ehm(s) and silences to a separate folder. Useful for training on false positives")
parser.add_argument("--srt", default=False, action='store_true',
                    help="generate subtitle track for easier accuracy evaluation")
parser.add_argument("--keep", nargs="+", default=["speech"],
                    help="space separated tags to be kept in the final video. Eg: ehm silence. Default: speech")
parser.add_argument("--keep-junk", default=False, action="store_true", help="keep tmp files")
parser.add_argument("--concat-only", default=False, action="store_true",
                    help="skip analysis/cut, concat existing tmp/inputs.txt "
                    "(useful to resume an interrupted merge)")
parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"],
                    help="compute device for TF inference (default: auto = GPU if available, else CPU)")
parser.add_argument("--encoder", default="auto", choices=["auto", "libx264", "h264_nvenc"],
                    help="video encoder for re-encode cuts (default: auto = NVENC if NVIDIA "
                    "encoder available, else libx264; ignored with --fastcut)")
parser.add_argument("--list-devices", default=False, action="store_true",
                    help="print detected GPUs / NVENC support and exit")
parser.add_argument("--lang", default="auto", choices=["auto", "it", "en"],
                    help="interface language (default: auto from system locale)")

args = parser.parse_args()

# ------------------------------------------------- i18n (stdlib only)
try:
    from i18n import load_strings as _ls, resolve_lang as _rl
    LANG = _rl(args.lang)
    _STRINGS = _ls(BASE_DIR, LANG)
except Exception:
    LANG = "en" if (args.lang or "auto") != "it" else args.lang
    _STRINGS = {}


def T(key: str, default: str = "") -> str:
    return _STRINGS.get(key, default or key)

# ------------------------------------------------- heavy imports (after --help)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import numpy as np
except ImportError:
    sys.exit("Missing dependency 'numpy'. Run: pip install -r requirements.txt")

try:
    import tensorflow as tf
except ImportError:
    sys.exit(
        "Missing dependency 'tensorflow'. Install it first:\n"
        '  pip install "tensorflow>=2.16,<2.22"\n'
        "Windows/Python 3.11: pip install tensorflow==2.21.0"
    )

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("Missing dependency 'tqdm'. Run: pip install -r requirements.txt")

try:
    import psutil  # noqa: F401  (used for cpu count)
except ImportError:
    psutil = None

import datetime
import io
import signal
import subprocess
import time
from contextlib import suppress
from functools import wraps

try:
    import cv2
except ImportError:
    cv2 = None  # only needed for --spectrogram

# -------------------------------------------- compute device (NVIDIA auto)
# Windows native TF is CPU-only (TF>=2.11): GPU works on Linux + NVIDIA CUDA,
# or WSL2. --device auto uses the GPU when TF sees one, else CPU.
BUILTIN_MODEL_URL = (
    "https://github.com/morrolinux/simple-ehm/releases/latest/download/model.keras"
)


def configure_device(mode: str = "auto") -> str:
    """Return 'gpu' or 'cpu'. Sets memory growth + optional mixed precision."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception as e:
        print(T("gpu_warn", f"GPU detection failed ({e}), using CPU."))
        return "cpu"
    want_gpu = (mode == "gpu") or (mode == "auto" and bool(gpus))
    if want_gpu and not gpus:
        print(T("gpu_missing",
                "No GPU visible to TensorFlow (Windows native is CPU-only; "
                "use Linux + NVIDIA CUDA or WSL2 for GPU). Falling back to CPU."))
        return "cpu"
    if want_gpu:
        try:
            for g in gpus:
                with suppress(Exception):
                    tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
        # Mixed precision speeds up NVIDIA inference; inference-only so safe.
        with suppress(Exception):
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
        print(T("gpu_on", f"GPU enabled: {[g.name for g in gpus]}"))
        return "gpu"
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    print(T("cpu_on", "Using CPU for inference."))
    return "cpu"


def _detect_nvenc() -> bool:
    """True if this ffmpeg build supports NVIDIA NVENC encoding."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-h", "encoder=h264_nvenc"],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except Exception:
        return False


_NVENC = None


def resolve_encoder() -> str:
    """Pick encoder: explicit choice, or NVENC when available (auto)."""
    global _NVENC
    if args.encoder != "auto":
        return args.encoder
    if _NVENC is None:
        _NVENC = _detect_nvenc()
    return "h264_nvenc" if _NVENC else "libx264"


def list_devices() -> str:
    """One-shot diagnostic: GPUs visible to TF + NVENC + recommendation."""
    err = ""
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception as e:
        gpus = []
        err = f" ({e})"
    names = [getattr(g, "name", str(g)) for g in gpus]
    nvenc = _detect_nvenc()
    if names:
        rec = "gpu inference + " + ("h264_nvenc" if nvenc else "libx264")
    elif nvenc:
        rec = "cpu inference + h264_nvenc (encode-only)"
    else:
        rec = "cpu inference + libx264"
    return "\n".join([
        f"tensorflow gpus: {names if names else 'none' + err}",
        f"ffmpeg nvenc: {'yes' if nvenc else 'no'}",
        f"recommended: --device auto --encoder auto  ->  {rec}",
    ])


def ensure_model(model_arg: str, model_url: str = "", no_download: bool = False) -> Path:
    """Return usable model path, downloading model.keras on first run if needed."""
    p = Path(model_arg)
    if p.exists() and (p.is_file() or p.is_dir()):
        return p
    # Only auto-download single-file .keras targets, never legacy SavedModel dirs.
    target = p if p.suffix == ".keras" else (BASE_DIR / "model.keras")
    if target.is_file():
        return target
    url = (model_url or "").strip() or BUILTIN_MODEL_URL
    if no_download or not url:
        sys.exit(T("model_missing",
                    f"Model not found: {p}. Pass --model or set SIMPLE_EHM_MODEL_URL."));
    print(T("model_dl", f"Downloading model from {url} ..."))
    try:
        import urllib.request
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_dl = target.with_suffix(".keras.downloading")

        def _hook(block, bsize, total):
            pass  # keep quiet; tqdm not easily hookable here w/o extra code

        urllib.request.urlretrieve(url, str(tmp_dl))
        if tmp_dl.stat().st_size < 100_000:
            raise RuntimeError(f"suspiciously small file ({tmp_dl.stat().st_size} B)")
        tmp_dl.replace(target)
        print(T("model_ok", f"Model saved to {target}"))
        return target
    except Exception as e:
        with suppress(Exception):
            tmp_dl.unlink()
        sys.exit(T("model_fail",
                   f"Model download failed: {e}\nSet SIMPLE_EHM_MODEL_URL or "
                   f"--model-url to a valid model.keras URL."))


# ------------------------------------------------- globals
video_path = Path(args.filename) if args.filename else None
audio_len = None
pbar = None
tmp_folder = DEFAULT_TMP_DIR
# NOTE: training_data stays next to the script for backward compat
td_folder = DEFAULT_TD_DIR
_perf = dict()
stats = None
labels = ["ehm", "silence", "speech"]
keep = set()
trash = set()

cuts = []       # edits for ffmpeg to split
mergelist = []  # files for ffmpeg to merge back


# ------------------------------------------------- helpers
def signal_handler(sig, frame):
    try:
        if not args.keep_junk:
            wav = video_path.with_suffix(".wav")
            wav.unlink(missing_ok=True)
            if tmp_folder.is_dir():
                for f in tmp_folder.iterdir():
                    with suppress(OSError):
                        f.unlink()
    except Exception:
        pass
    sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)


def format_ts(seconds: float) -> str:
    """Return HH:MM:SS.mmm timestamp for SRT/ffmpeg."""
    if seconds < 0:
        seconds = 0
    td = datetime.timedelta(seconds=seconds)
    total_ms = int(td.total_seconds() * 1000)
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def timeit(f):
    @wraps(f)
    def timed(*a, **kw):
        start = datetime.datetime.now()
        out = f(*a, **kw)
        _perf[f.__name__] = datetime.datetime.now() - start
        return out
    return timed


@timeit
def convert_input(path: Path) -> Path:
    wav_path = path.with_suffix(".wav")
    print("extracting audio track...")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
           "-filter:a", "dynaudnorm", str(wav_path), "-y"]
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        sys.exit(f"ffmpeg audio extraction failed:\n{r.stderr.decode(errors='replace')}")
    return wav_path


@timeit
def decode_audio(audio_binary):
    audio, rate = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1), int(rate)


def get_spectrogram(waveform, seek: int, window_size: int):
    """1s (or window_size) slice -> (124, 129, 1) spectrogram.

    Uses Python ints for shapes (TF tensors can't be used in `if`).
    """
    window_size = int(window_size)
    seek = int(seek)
    waveform = tf.cast(waveform, tf.float32)
    total = int(waveform.shape[0])

    end = min(seek + window_size, total)
    segment = waveform[seek:end] if end > seek else waveform[0:0]
    seg_len = int(segment.shape[0])
    if seg_len < window_size:
        segment = tf.concat(
            [segment, tf.zeros([window_size - seg_len], dtype=tf.float32)], 0)

    spectrogram = tf.signal.stft(segment, frame_length=255, frame_step=128, pad_end=True)
    spectrogram = tf.abs(spectrogram)

    n_frames = int(spectrogram.shape[0])
    if n_frames > 124:
        spectrogram = spectrogram[:124, :]
    elif n_frames < 124:
        spectrogram = tf.pad(spectrogram, [(0, 124 - n_frames), (0, 0)])

    return tf.expand_dims(spectrogram, axis=-1)


def load_predictor(model_path: Path):
    """Return fn(batch) -> numpy logits, supporting Keras 3 (.keras) and legacy SavedModel."""
    # 1) try Keras native format
    with suppress(Exception):
        m = tf.keras.models.load_model(str(model_path))
        def _keras(batch):
            return np.asarray(m(batch, training=False))
        # sanity call with dummy input
        _keras(tf.zeros((1, 124, 129, 1), dtype=tf.float32))
        print(f"model loaded (keras): {model_path}")
        return _keras

    # 2) legacy SavedModel directory
    loaded = tf.saved_model.load(str(model_path))
    sig = None
    if hasattr(loaded, "signatures") and loaded.signatures:
        sig = loaded.signatures.get("serving_default")
        if sig is None:
            sig = next(iter(loaded.signatures.values()))

    if sig is None:
        # last resort: callable SavedModel
        def _call(batch):
            out = loaded(tf.convert_to_tensor(batch))
            if isinstance(out, dict):
                out = next(iter(out.values()))
            return np.asarray(out)
        print(f"model loaded (saved_model callable): {model_path}")
        return _call

    input_names = list(sig.structured_input_signature[1].keys())

    def _sig(batch):
        t = tf.convert_to_tensor(batch, dtype=tf.float32)
        try:
            if len(input_names) == 1:
                out = sig(**{input_names[0]: t})
            else:
                out = sig(t)
        except Exception:
            out = sig(t)
        if isinstance(out, dict):
            out = next(iter(out.values()))
        return np.asarray(out)

    # sanity call
    _sig(np.zeros((1, 124, 129, 1), dtype=np.float32))
    print(f"model loaded (saved_model signature): {model_path}")
    return _sig


def td_folder_init():
    td_folder.mkdir(exist_ok=True)
    for f in td_folder.glob("*.wav"):
        with suppress(OSError):
            f.unlink()


def generate_tdata(ss, to, count, label):
    filename = str(td_folder / f"{label}-{count}.wav")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", ss,
           "-i", str(video_path), "-t", "1",
           "-c:a", "pcm_s16le", "-ac", "1", "-ar", "16000",
           "-filter:a", "dynaudnorm", filename]
    cuts.append(cmd)


def generate_cut(ss, to, count):
    out_name = str(count) + video_path.suffix
    # NOTE: un solo -ss. Con fastcut (stream copy) lo mettiamo prima di -i
    # (seek veloce); con re-encode dopo -i (seek accurato al frame).
    if args.fastcut:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-ss", ss, "-to", to, "-i", str(video_path), "-copyts"]
    else:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-i", str(video_path), "-ss", ss, "-to", to, "-copyts"]
    if args.crf > 0:
        cmd.extend(["-crf", str(args.crf)])
    if args.fps > 0:
        cmd.extend(["-filter:v", "fps=fps=" + str(args.fps)])

    if args.fastcut:
        cmd.extend(["-c:a", "copy", "-c:v", "copy", "-avoid_negative_ts", "1"])
    else:
        cmd.extend(["-c:v", resolve_encoder(), "-crf", "23"])
    cmd.extend([str(tmp_folder / out_name), "-y"])
    cuts.append(cmd)
    mergelist.append("file '" + out_name + "'")


@timeit
def analyze_track(predict, waveform, sample_rate: int):
    global cuts, mergelist, pbar, audio_len, labels, stats, keep, trash

    lastc = -1
    lasts = 0.0
    lastts = "00:00:00.000"
    count = 0
    lastwf = 0
    stats = [[0, 0.0] for _ in range(len(labels))]

    if args.srt:
        sub = open(video_path.with_suffix(".srt"), 'w', encoding='utf-8')
    else:
        sub = io.StringIO()

    window_size = int(sample_rate / args.window_size_divide)
    window_slide = int(window_size / args.window_slide_divide)

    print("analyzing track...")
    last_i = window_slide * int(audio_len / window_slide)
    pbar = tqdm(total=max(last_i, 1))
    for i in range(0, audio_len, window_slide):
        pbar.update(n=min(window_slide, audio_len - i))
        spectrogram = get_spectrogram(waveform, i, window_size)
        batch = tf.expand_dims(spectrogram, axis=0)

        logits = predict(batch)[0]
        probs = tf.nn.softmax(logits).numpy()
        cls = int(np.argmax(probs))
        conf = float(probs[cls])

        if lastc == -1:
            lastc = cls
            continue

        if cls != lastc or i == last_i:
            s = i / sample_rate
            if i == last_i:
                s += (audio_len - i) / sample_rate

            ts = format_ts(s)
            if labels[cls] not in trash and i < lastwf and i < last_i:
                continue
            record = (f"{count}\n{lastts.replace('.', ',')} --> "
                      f"{ts.replace('.', ',')}\n{labels[lastc]}\n"
                      f"[{conf * 100:.1f}]\n\n")
            count += 1
            sub.write(record)
            stats[lastc][0] += 1
            stats[lastc][1] += s - lasts
            lasts = s
            if labels[lastc] in keep:
                generate_cut(lastts, ts, count)
            elif args.generate_training_data:
                generate_tdata(lastts, ts, count, labels[lastc])
            lastts = ts
            lastc = cls
        lastwf = i + window_size

        if not args.spectrogram:
            continue
        if cv2 is None:
            print("opencv not installed, --spectrogram disabled.", file=sys.stderr)
            args.spectrogram = False
            continue
        img = spectrogram.numpy().T
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(img, labels[cls], (5, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("spectrogram", img)
        cv2.waitKey(1) & 0xFF
        time.sleep(0.2)

    sub.close()
    pbar.close()


@timeit
def cut_and_merge(out_stem: Path):
    global pbar
    if not cuts:
        print("Nothing to cut/merge (no kept segments?).")
        return None
    if psutil is not None:
        cores = max(1, int((psutil.cpu_count(logical=True) or 2) / 2))
    else:
        cores = max(1, int((os.cpu_count() or 2) / 2))
    print("CUT and MERGE: running", cores, "ffmpeg simultaneous instances.")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run_cut(cmd):
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode != 0:
            raise RuntimeError(
                f"ffmpeg cut failed (cmd={' '.join(cmd)}):\n"
                f"{r.stderr.decode(errors='replace')}")
        return True

    pbar = tqdm(total=len(cuts))
    with ThreadPoolExecutor(max_workers=cores) as ex:
        futs = [ex.submit(_run_cut, c) for c in cuts]
        for f in as_completed(futs):
            f.result()  # solleva subito in caso di errore
            pbar.update(n=1)
    pbar.close()
    print("all cut processes finished.")

    mergelist_path = tmp_folder / "inputs.txt"
    with open(mergelist_path, 'w', encoding='utf-8') as f:
        for m in mergelist:
            f.write(m + "\n")

    return concat_inputs(mergelist_path, out_stem)


def concat_inputs(mergelist_path: Path, out_stem: Path):
    """Concat demuxer su inputs.txt esistente (usato anche da --concat-only)."""
    if not mergelist_path.is_file():
        sys.exit(f"concat-only: {mergelist_path} non trovato.")
    now = datetime.datetime.now()
    out_filename = f"{out_stem}_{now.hour:02d}-{now.minute:02d}-{now.second:02d}{video_path.suffix}"

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", str(mergelist_path),
           "-c", "copy", str(out_filename), "-y"]
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        sys.exit(f"ffmpeg concat failed:\n{r.stderr.decode(errors='replace')}")
    print("saved:", out_filename)
    return out_filename


if __name__ == '__main__':
    if args.list_devices:
        print(list_devices())
        raise SystemExit(0)

    if video_path is None or not video_path.is_file():
        parser.print_usage(sys.stderr)
        raise SystemExit(f"Error! {video_path} doesn't exist.")

    tmp_folder.mkdir(exist_ok=True)

    if args.concat_only:
        out = concat_inputs(tmp_folder / "inputs.txt", video_path.with_suffix(""))
        print("\nFATTO (concat-only)!")
        print("saved:", out)
        raise SystemExit(0)

    if args.generate_training_data:
        td_folder_init()

    keep = set(args.keep)
    trash = set(labels) - keep

    configure_device(args.device)
    try:
        from update import ensure_model_smart
        model_path, _changed, _note = ensure_model_smart(
            args.model, DATA_DIR, args.model_url, args.no_download)
    except SystemExit:
        raise
    except Exception:
        model_path = ensure_model(args.model, args.model_url, args.no_download)
    predict = load_predictor(model_path)
    wav_path = convert_input(video_path)
    audio_binary = tf.io.read_file(str(wav_path))
    waveform, sample_rate = decode_audio(audio_binary)

    audio_len = int(waveform.shape[0])

    analyze_track(predict, waveform, sample_rate)
    cut_and_merge(video_path.with_suffix(""))

    if not args.keep_junk:
        with suppress(OSError):
            wav_path.unlink()
        if tmp_folder.is_dir():
            for file in tmp_folder.iterdir():
                with suppress(OSError):
                    if file.is_file():
                        file.unlink()

    print("\nFATTO!")
    for k, v in _perf.items():
        print(k, "ha impiegato", str(v))

    saved_time = 0.0
    print("")
    for i in range(len(labels)):
        if labels[i] == "speech":
            continue
        saved_time += stats[i][1]
        print(f"Rimosso {stats[i][0]} {labels[i]}(s) per un ammontare di "
              f"{str(datetime.timedelta(seconds=stats[i][1]))[:8]}")

    print("Tempo totale risparmiato:", str(datetime.timedelta(seconds=saved_time))[:8])
