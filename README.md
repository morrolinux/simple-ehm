# simple-ehm

![logo](assets/logo-horizontal.png)

A simple tool for a simple task: remove filler sounds ("ehm") from pre-recorded speeches. AI powered.
Istruzioni in italiano in fondo al documento.

## Quick start

```bash
pip install -r requirements.txt
python simple_ehm-runnable.py /path/to/video.mp4 --srt --fastcut
# GUI (same folder): python simple_ehm_gui.py
```

Output video + `.srt` debug track are written next to the original file.

## Features (v2.x)

- **Modern stack:** Python 3.10–3.13, TensorFlow 2.16–2.21 / Keras 3, Windows + Linux.
- **NVIDIA GPU auto:** `--device auto` (default) uses the GPU when TensorFlow sees
  one (memory growth + mixed precision), else CPU. `--device cpu|gpu` to force.
  `--encoder auto` (default) uses `h264_nvenc` when your ffmpeg supports it,
  else `libx264` (ignored with `--fastcut`). `--list-devices` prints what
  your PC can use (TF GPUs + NVENC) and the recommended combo.
  Note: native Windows TF is CPU-only; for GPU inference use Linux + CUDA
  or WSL2 (NVENC encoding works on Windows too).
- **Auto language:** IT/EN picked from the PC (Windows API / POSIX locale),
  override with `--lang it|en` or `SIMPLE_EHM_LANG`. GUI language menu included.
  Strings live in `langs/*.json` – add a file to add a language.
- **Models from GitHub:** first run downloads `model.keras` only if the bundled
  one is missing or older than the release `manifest.json`
  (`SIMPLE_EHM_REPO`, default `morrolinux/simple-ehm`).
  Override: `--model`, `--model-url`, `SIMPLE_EHM_MODEL_URL`, `--no-download`.
  Program update check included (release tag vs `version.py`).
- **Resume merges:** `--concat-only` re-runs only the final `ffmpeg concat`
  from `tmp/inputs.txt` (no re-analysis).
- **GUI:** `simple_ehm_gui.py` (tkinter, stdlib only) exposes every parameter
  with log window, Run/Stop and IT/EN switch.

## Binaries

- Windows: `build_windows.ps1` → `dist/simple-ehm.exe` (≈3,5 GB con TF),
  `dist/simple-ehm-gui.exe` (≈16 MB). `dist/` non va su git: i binary si
  pubblicano come asset della GitHub Release insieme a `SHA256SUMS.txt`.
- **Prima di eseguire: verifica gli hash** come spiegato in `VERIFY.md`
  (confronto SHA256 + Defender/VirusTotal + sandbox).
- Linux: `build_linux.sh` → `dist/simple-ehm`, `dist/simple-ehm-gui`.
- Debian: `build_deb.sh` → `simple-ehm_<ver>_amd64.deb`
  (`sudo apt install ./simple-ehm_*_amd64.deb`, anche nel menu app).
  PyInstaller specs: `simple-ehm-cli.spec`, `simple-ehm-gui.spec`.

## Publishing models (maintainers)

Attach to a GitHub Release: `model.keras` + `manifest.json`
(`models/manifest.json` is the template: `version`, `url`, `sha256`, `size`).
Clients download only when `manifest.version != model.keras.version`.

Docker: `docker build -t simple-ehm .`, then `./convert.sh /path/to/video [args...]`
(input files must be inside this folder for the bind-mount).

# ITA

![logo](assets/logo-horizontal.png)

Un semplice strumento per un semplice compito: rimuovere gli "ehm" dai discorsi
pre-registrati.

```bash
pip install -r requirements.txt
python simple_ehm-runnable.py /percorso/al/video.mp4 --srt --fastcut
# GUI: python simple_ehm_gui.py
```

- **GPU NVIDIA automatica:** `--device auto` usa la GPU se TensorFlow la vede,
  altrimenti CPU. `--encoder auto` usa `h264_nvenc` se il tuo ffmpeg lo
  supporta, altrimenti `libx264`. `--list-devices` dice cosa può usare il PC.
  Su Windows nativo inferenza solo CPU; per la GPU serve Linux + CUDA o WSL2
  (l'encoding NVENC funziona anche su Windows).
- **Lingua automatica:** IT/EN dal PC (API Windows / locale Linux),
  `--lang it|en` o `SIMPLE_EHM_LANG` per forzare.
- **Modelli da GitHub:** al primo avvio scarica `model.keras` solo se manca o è
  più vecchio del `manifest.json` della release.
- **Binary:** `build_windows.ps1` (exe), `build_linux.sh` + `build_deb.sh` (.deb).

## Contribuire al modello / dataset

Vedi le istruzioni originali nel notebook `simple_ehm.ipynb` e
`trainingset/`: clip WAV `pcm_s16le` mono 16 kHz per classe
(`silence`, `speech`, `ehm`), estraibili con `--generate-training-data`.
