#!/usr/bin/env python3
"""simple-ehm GUI (tkinter, stdlib only): set every CLI parameter and run."""
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path

BASE_DIR = Path(getattr(sys, "_MEIPASS", None) or Path(__file__).resolve().parent)
FROZEN = bool(getattr(sys, "frozen", False))


def _find_cli() -> list[str]:
    """CLI command: sibling frozen binary if packaged, else runnable script."""
    if FROZEN:
        exe_dir = Path(sys.executable).resolve().parent
        for name in ("simple-ehm.exe", "simple-ehm"):
            cand = exe_dir / name
            if cand.is_file():
                return [str(cand)]
    return [sys.executable, str(BASE_DIR / "simple_ehm-runnable.py")]

try:
    from i18n import detect_lang as _detect, load_strings as _ls
    _HAS_I18N = True
except Exception:
    _HAS_I18N = False


def detect_lang() -> str:
    if _HAS_I18N:
        return _detect()
    try:
        import locale
        return "it" if ((locale.getlocale()[0] or "").lower().startswith("it")) else "en"
    except Exception:
        return "en"


def load_strings(lang: str) -> dict:
    if _HAS_I18N:
        return _ls(BASE_DIR, lang)
    try:
        import json
        return json.loads((BASE_DIR / "langs" / f"{lang}.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.lang = tk.StringVar(value=detect_lang())
        self.strings = load_strings(self.lang.get())
        self.proc: subprocess.Popen | None = None
        self.log_q: queue.Queue = queue.Queue()
        self._build()
        self._refresh_texts()
        self.after(100, self._poll_log)
        threading.Thread(target=self._refresh_hw, daemon=True).start()

    def _refresh_hw(self):
        """Show what the PC can use (TF GPUs + NVENC) without loading the GUI."""
        try:
            out = subprocess.run(
                _find_cli() + ["--list-devices"], capture_output=True, text=True,
                errors="replace", timeout=120,
                cwd=(str(Path(sys.executable).resolve().parent) if FROZEN
                     else str(BASE_DIR)))
            txt = (out.stdout or out.stderr or "").strip().replace("\n", " · ")
            self.after(0, lambda: self.lbl_hw.config(text=txt or "hw check failed"))
        except Exception as e:
            self.after(0, lambda: self.lbl_hw.config(text=f"hw check failed: {e}"))

    def T(self, key: str) -> str:
        return self.strings.get(key, key)

    def _build(self):
        self.title("simple-ehm")
        self.geometry("640x560")
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # video + model rows
        self.var_file = tk.StringVar()
        self.var_model = tk.StringVar(value=str(BASE_DIR / "model.keras"))
        self.var_device = tk.StringVar(value="auto")
        self.var_encoder = tk.StringVar(value="auto")
        self.var_keep = tk.StringVar(value="speech")
        self.var_ws = tk.StringVar(value="1")
        self.var_wsl = tk.StringVar(value="2")
        self.var_fps = tk.StringVar(value="-1")
        self.var_crf = tk.StringVar(value="-1")
        self.var_fastcut = tk.BooleanVar(value=True)
        self.var_srt = tk.BooleanVar(value=True)
        self.var_training = tk.BooleanVar(value=False)
        self.var_spectro = tk.BooleanVar(value=False)
        self.var_junk = tk.BooleanVar(value=False)
        self.var_concat = tk.BooleanVar(value=False)

        r = 0
        self.lbl_file = ttk.Label(frm, text="Video")
        self.lbl_file.grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.var_file, width=50).grid(row=r, column=1, sticky="ew")
        self.btn_browse = ttk.Button(frm, text="…", width=4, command=self._browse)
        self.btn_browse.grid(row=r, column=2)
        r += 1
        self.lbl_model = ttk.Label(frm, text="Model")
        self.lbl_model.grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.var_model, width=50).grid(row=r, column=1, sticky="ew")
        ttk.Button(frm, text="…", width=4, command=self._browse_model).grid(row=r, column=2)
        r += 1

        # device + lang
        self.lbl_device = ttk.Label(frm, text="Device")
        self.lbl_device.grid(row=r, column=0, sticky="w")
        ttk.Combobox(frm, textvariable=self.var_device,
                     values=["auto", "cpu", "gpu"], width=8, state="readonly").grid(
            row=r, column=1, sticky="w")
        self.lbl_lang = ttk.Label(frm, text="Language")
        self.lbl_lang.grid(row=r, column=1, sticky="e", padx=(0, 110))
        ttk.Combobox(frm, textvariable=self.lang, values=["it", "en"],
                     width=5, state="readonly").grid(row=r, column=1, sticky="e")
        self.lang.trace_add("write", lambda *_: self._refresh_texts())
        r += 1

        # encoder (ffmpeg GPU) + detected status
        self.lbl_encoder = ttk.Label(frm, text="Encoder")
        self.lbl_encoder.grid(row=r, column=0, sticky="w")
        ttk.Combobox(frm, textvariable=self.var_encoder,
                     values=["auto", "libx264", "h264_nvenc"],
                     width=12, state="readonly").grid(row=r, column=1, sticky="w")
        r += 1
        self.lbl_hw = ttk.Label(frm, text="…", foreground="gray")
        self.lbl_hw.grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1

        # options grid
        self.cb_fast = ttk.Checkbutton(frm, text="fastcut", variable=self.var_fastcut)
        self.cb_fast.grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        self.cb_srt = ttk.Checkbutton(frm, text="srt", variable=self.var_srt)
        self.cb_srt.grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        self.cb_tr = ttk.Checkbutton(frm, text="training", variable=self.var_training)
        self.cb_tr.grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        self.cb_sp = ttk.Checkbutton(frm, text="spectrogram", variable=self.var_spectro)
        self.cb_sp.grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        self.cb_junk = ttk.Checkbutton(frm, text="keep junk", variable=self.var_junk)
        self.cb_junk.grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        self.cb_concat = ttk.Checkbutton(frm, text="concat only", variable=self.var_concat)
        self.cb_concat.grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1

        # keep + numeric params
        self.lbl_keep = ttk.Label(frm, text="keep")
        self.lbl_keep.grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.var_keep, width=20).grid(row=r, column=1, sticky="w")
        r += 1
        for label, var in (("win-size÷", self.var_ws), ("win-slide÷", self.var_wsl),
                           ("fps", self.var_fps), ("crf", self.var_crf)):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w")
            ttk.Entry(frm, textvariable=var, width=10).grid(row=r, column=1, sticky="w")
            r += 1

        # run/stop
        btns = ttk.Frame(frm)
        btns.grid(row=r, column=0, columnspan=3, pady=8)
        self.btn_run = ttk.Button(btns, text="Run", command=self._run)
        self.btn_run.pack(side=tk.LEFT, padx=4)
        self.btn_stop = ttk.Button(btns, text="Stop", command=self._stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=4)
        r += 1

        self.log = tk.Text(frm, height=12, state=tk.DISABLED)
        self.log.grid(row=r, column=0, columnspan=3, sticky="nsew")
        frm.rowconfigure(r, weight=1)
        frm.columnconfigure(1, weight=1)

    def _refresh_texts(self):
        self.strings = load_strings(self.lang.get())
        self.title(self.T("app_title"))
        self.lbl_file.config(text=self.T("gui_file"))
        self.btn_browse.config(text=self.T("gui_browse"))
        self.lbl_model.config(text=self.T("gui_model"))
        self.lbl_device.config(text=self.T("gui_device"))
        self.lbl_encoder.config(text=self.T("gui_encoder"))
        self.lbl_lang.config(text=self.T("gui_lang"))
        self.cb_fast.config(text=self.T("gui_fastcut"))
        self.cb_srt.config(text=self.T("gui_srt"))
        self.cb_tr.config(text=self.T("gui_training"))
        self.cb_sp.config(text=self.T("gui_spectrogram"))
        self.lbl_keep.config(text=self.T("gui_keep"))
        self.cb_junk.config(text=self.T("gui_keep_junk"))
        self.cb_concat.config(text=self.T("gui_concat_only"))
        self.btn_run.config(text=self.T("gui_run"))
        self.btn_stop.config(text=self.T("gui_stop"))

    def _browse(self):
        f = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mkv *.mov *.avi"), ("All", "*.*")])
        if f:
            self.var_file.set(f)

    def _browse_model(self):
        f = filedialog.askopenfilename(filetypes=[("Keras", "*.keras"), ("All", "*.*")])
        if f:
            self.var_model.set(f)

    def _append(self, s: str):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, s)
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def _poll_log(self):
        try:
            while True:
                line = self.log_q.get_nowait()
                self._append(line)
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    def _build_cmd(self) -> list[str]:
        cmd = _find_cli()
        if self.var_file.get():
            cmd.append(self.var_file.get())
        cmd += ["--model", self.var_model.get(), "--device", self.var_device.get(),
                "--encoder", self.var_encoder.get(), "--lang", self.lang.get(),
                "--window-size-divide", self.var_ws.get(),
                "--window-slide-divide", self.var_wsl.get(),
                "--fps", self.var_fps.get(), "--crf", self.var_crf.get(),
                "--keep"] + self.var_keep.get().split()
        for flag, var in (("--fastcut", self.var_fastcut), ("--srt", self.var_srt),
                           ("--generate-training-data", self.var_training),
                           ("--spectrogram", self.var_spectro),
                           ("--keep-junk", self.var_junk),
                           ("--concat-only", self.var_concat)):
            if var.get():
                cmd.append(flag)
        return cmd

    def _run(self):
        if not self.var_file.get():
            self._append("⚠ select a video first\n")
            return
        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self._append(self.T("gui_starting") + "\n" + " ".join(self._build_cmd()) + "\n")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            self.proc = subprocess.Popen(
                self._build_cmd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace",
                cwd=(str(Path(sys.executable).resolve().parent) if FROZEN
                     else str(BASE_DIR)))
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.log_q.put(line)
            self.proc.wait()
            self.log_q.put(f"\n[{self.T('gui_done')} rc={self.proc.returncode}]\n")
        except Exception as e:
            self.log_q.put(f"\nERROR: {e}\n")
        finally:
            self.proc = None
            self.after(0, lambda: (self.btn_run.config(state=tk.NORMAL),
                                   self.btn_stop.config(state=tk.DISABLED)))

    def _stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


if __name__ == "__main__":
    App().mainloop()
