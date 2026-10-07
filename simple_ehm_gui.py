#!/usr/bin/env python3
"""simple-ehm GUI (tkinter, stdlib only).

Thin front-end over the CLI: builds the command line, runs it in a
subprocess, streams the log and turns tqdm output into a progress bar.
Usage: python simple_ehm_gui.py [video]
"""
import codecs
import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, ttk

BASE_DIR = Path(getattr(sys, "_MEIPASS", None) or Path(__file__).resolve().parent)
FROZEN = bool(getattr(sys, "frozen", False))
WORK_DIR = Path(sys.executable).resolve().parent if FROZEN else BASE_DIR
IS_WIN = os.name == "nt"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if IS_WIN else 0

try:
    from i18n import detect_lang, load_strings
except Exception:  # i18n.py missing: English keys only
    def detect_lang() -> str:
        return "en"

    def load_strings(base_dir, lang) -> dict:
        return {}

try:
    from version import __version__
except Exception:
    __version__ = ""

LABELS = ("speech", "ehm", "silence")
VIDEO_TYPES = "*.mp4 *.MP4 *.mkv *.mov *.MOV *.avi *.m4v *.webm"
TQDM_RE = re.compile(r"(\d{1,3})%\|.*?\|\s*(\S+)\s*\[([^\]<]*)<?([^,\]]*)")
PHASES = (("extracting audio track", "phase_audio"),
          ("analyzing track", "phase_analyze"),
          ("CUT and MERGE", "phase_cut"),
          ("all cut processes finished", "phase_merge"))

DARK = dict(bg="#11131A", card="#1A1E2A", border="#2A3042", fg="#E8EBF2",
            muted="#8E97AB", field="#232838", log_bg="#0C0E14",
            log_fg="#CDD3E0", accent="#4C8DFF", accent_hover="#6A9FFF",
            accent_fg="#FFFFFF", ok="#3ECF8E", err="#FF6B6B", warn="#F5B94A")
LIGHT = dict(bg="#EEF0F5", card="#FFFFFF", border="#D5D9E3", fg="#1B2030",
             muted="#5E6779", field="#F6F7FA", log_bg="#FAFBFC",
             log_fg="#1B2030", accent="#2F6BFF", accent_hover="#4C80FF",
             accent_fg="#FFFFFF", ok="#1E8E5A", err="#D93B3B", warn="#B7791F")


def _settings_path() -> Path:
    if IS_WIN:
        root = Path(os.environ.get("APPDATA", str(Path.home())))
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return root / "simple-ehm" / "gui.json"


def _find_cli() -> list[str]:
    """CLI command: sibling frozen binary if packaged, else runnable script."""
    if FROZEN:
        for name in ("simple-ehm.exe", "simple-ehm"):
            cand = WORK_DIR / name
            if cand.is_file():
                return [str(cand)]
    return [sys.executable, str(BASE_DIR / "simple_ehm-runnable.py")]


def _child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"      # live log instead of block-buffered output
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _open_path(p: Path):
    try:
        if IS_WIN:
            os.startfile(str(p))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])
    except Exception:
        pass


class App(tk.Tk):
    def __init__(self, initial_file: str = ""):
        super().__init__()
        self.cfg = self._load_settings()
        self.lang_choice = tk.StringVar(value=self.cfg.get("lang", "auto"))
        self.lang = self._resolve_lang()
        self.S = load_strings(BASE_DIR, self.lang)
        self.dark = tk.BooleanVar(value=self.cfg.get("dark", True))

        g = self.cfg.get
        self.var_file = tk.StringVar(value=initial_file)
        self.var_device = tk.StringVar(value=g("device", "auto"))
        self.var_encoder = tk.StringVar(value=g("encoder", "auto"))
        self.var_fastcut = tk.BooleanVar(value=g("fastcut", True))
        self.var_srt = tk.BooleanVar(value=g("srt", True))
        keep = set(g("keep", ["speech"]))
        self.var_keep = {k: tk.BooleanVar(value=k in keep) for k in LABELS}
        self.var_crf = tk.StringVar(value=g("crf", ""))
        self.var_fps = tk.StringVar(value=g("fps", ""))
        self.var_win = tk.StringVar(value=g("win", "1"))
        self.var_slide = tk.StringVar(value=g("slide", "2"))
        self.var_model = tk.StringVar(value=g("model", ""))
        self.var_training = tk.BooleanVar(value=False)
        self.var_keep_junk = tk.BooleanVar(value=False)
        self.var_no_output = tk.BooleanVar(value=False)
        self.var_concat_only = tk.BooleanVar(value=False)
        self.adv_open = tk.BooleanVar(value=g("adv_open", False))

        self.proc: subprocess.Popen | None = None
        self.stopped = False
        self.last_output: Path | None = None
        self.hw_text = ""
        self.q: queue.Queue = queue.Queue()
        self._texts: list[tuple] = []   # (widget, key, prefix) for live re-translation

        self.title("simple-ehm")
        self._set_icon()
        self._setup_fonts()
        self.style = ttk.Style(self)
        with_suppress(lambda: self.style.theme_use("clam"))
        self._build()
        self._apply_theme()
        self._refresh_texts()
        self._sync_states()
        self.update_idletasks()
        self.minsize(self.winfo_reqwidth(), min(self.winfo_reqheight(), self.winfo_screenheight() - 80))
        if g("geometry"):
            self.geometry(g("geometry"))
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(80, self._poll)
        threading.Thread(target=self._probe_hw, daemon=True).start()

    # ------------------------------------------------------------ helpers
    def T(self, key: str) -> str:
        return self.S.get(key, key)

    def _resolve_lang(self) -> str:
        c = self.lang_choice.get()
        return c if c in ("it", "en") else detect_lang()

    def _tr(self, widget, key: str, prefix: str = ""):
        """Register a widget whose text comes from a translation key."""
        self._texts.append((widget, key, prefix))
        return widget

    def _set_icon(self):
        self.logo = None
        p = BASE_DIR / "assets" / "logo-icon.png"
        try:
            if p.is_file():
                img = tk.PhotoImage(file=str(p))
                self.iconphoto(True, img)
                f = max(1, img.height() // 48)
                self.logo = img.subsample(f, f)
        except Exception:
            self.logo = None

    def _setup_fonts(self):
        family = "Segoe UI" if IS_WIN else None
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            with_suppress(lambda n=name: tkfont.nametofont(n).configure(
                size=10, **({"family": family} if family else {})))
        base = tkfont.nametofont("TkDefaultFont").actual()
        self.f_title = tkfont.Font(family=base["family"], size=17, weight="bold")
        self.f_head = tkfont.Font(family=base["family"], size=11, weight="bold")
        self.f_small = tkfont.Font(family=base["family"], size=9)
        mono = "Consolas" if IS_WIN else tkfont.nametofont("TkFixedFont").actual()["family"]
        self.f_mono = tkfont.Font(family=mono, size=9)

    # ------------------------------------------------------------ layout
    def _card(self, parent, key: str):
        card = ttk.Frame(parent, style="Card.TFrame", padding=(14, 10, 14, 12))
        card.pack(fill=tk.X, pady=(0, 10))
        self._tr(ttk.Label(card, style="Head.Card.TLabel"), key).pack(anchor="w", pady=(0, 6))
        return card

    def _build(self):
        root = ttk.Frame(self, padding=(16, 14, 16, 10))
        root.pack(fill=tk.BOTH, expand=True)

        # header
        hd = ttk.Frame(root)
        hd.pack(fill=tk.X, pady=(0, 12))
        if self.logo is not None:
            ttk.Label(hd, image=self.logo).pack(side=tk.LEFT, padx=(0, 12))
        tt = ttk.Frame(hd)
        tt.pack(side=tk.LEFT)
        ttk.Label(tt, text="simple-ehm", style="Title.TLabel").pack(anchor="w")
        self._tr(ttk.Label(tt, style="Muted.TLabel"), "app_title").pack(anchor="w")
        right = ttk.Frame(hd)
        right.pack(side=tk.RIGHT)
        self.btn_theme = ttk.Button(right, command=self._toggle_theme, style="Ghost.TButton")
        self.btn_theme.pack(side=tk.RIGHT)
        self.cb_lang = ttk.Combobox(right, width=11, state="readonly", values=())
        self.cb_lang.pack(side=tk.RIGHT, padx=(0, 8))
        self.cb_lang.bind("<<ComboboxSelected>>", self._on_lang)

        # video
        c = self._card(root, "gui_file")
        fr = ttk.Frame(c, style="Inner.TFrame")
        fr.pack(fill=tk.X)
        self.ent_file = ttk.Entry(fr, textvariable=self.var_file, width=50)
        self.ent_file.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)
        self._tr(ttk.Button(fr, command=self._browse), "gui_browse").pack(side=tk.LEFT, padx=(8, 0))
        self.lbl_out = ttk.Label(c, style="Muted.Card.TLabel")
        self.lbl_out.pack(anchor="w", pady=(6, 0))
        self.var_file.trace_add("write", lambda *_: self._update_out_hint())

        # options
        c = self._card(root, "gui_proc")
        grid = ttk.Frame(c, style="Inner.TFrame")
        grid.pack(fill=tk.X)
        self._tr(ttk.Label(grid, style="Card.TLabel"), "gui_device").grid(row=0, column=0, sticky="w")
        ttk.Combobox(grid, textvariable=self.var_device, width=9, state="readonly",
                     values=("auto", "cpu", "gpu")).grid(row=0, column=1, sticky="w", padx=(8, 24))
        self._tr(ttk.Label(grid, style="Card.TLabel"), "gui_encoder").grid(row=0, column=2, sticky="w")
        self.cb_enc = ttk.Combobox(grid, textvariable=self.var_encoder, width=12, state="readonly",
                                   values=("auto", "libx264", "h264_nvenc"))
        self.cb_enc.grid(row=0, column=3, sticky="w", padx=(8, 0))

        kf = ttk.Frame(c, style="Inner.TFrame")
        kf.pack(fill=tk.X, pady=(10, 0))
        self._tr(ttk.Label(kf, style="Card.TLabel"), "gui_keep").pack(side=tk.LEFT, padx=(0, 8))
        for k in LABELS:
            ttk.Checkbutton(kf, text=k, variable=self.var_keep[k],
                            style="Card.TCheckbutton").pack(side=tk.LEFT, padx=(0, 10))

        self._tr(ttk.Checkbutton(c, variable=self.var_fastcut, style="Card.TCheckbutton",
                                 command=self._sync_states), "gui_fastcut").pack(anchor="w", pady=(10, 0))
        self._tr(ttk.Checkbutton(c, variable=self.var_srt, style="Card.TCheckbutton"),
                 "gui_srt").pack(anchor="w", pady=(2, 0))

        # advanced (collapsible)
        self.btn_adv = ttk.Button(c, command=self._toggle_adv, style="Link.Card.TButton")
        self.btn_adv.pack(anchor="w", pady=(8, 0))
        self.adv = ttk.Frame(c, style="Inner.TFrame")
        ag = ttk.Frame(self.adv, style="Inner.TFrame")
        ag.pack(fill=tk.X, pady=(4, 0))
        self.ent_crf = self._spin(ag, "gui_crf", self.var_crf, 0, 51, 0, 0)
        self.ent_fps = self._spin(ag, "gui_fps", self.var_fps, 1, 120, 0, 2)
        self._spin(ag, "gui_win", self.var_win, 1, 8, 1, 0)
        self._spin(ag, "gui_slide", self.var_slide, 1, 8, 1, 2)
        mf = ttk.Frame(self.adv, style="Inner.TFrame")
        mf.pack(fill=tk.X, pady=(8, 0))
        self._tr(ttk.Label(mf, style="Card.TLabel"), "gui_model").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Entry(mf, textvariable=self.var_model).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._tr(ttk.Button(mf, command=self._browse_model), "gui_browse").pack(side=tk.LEFT, padx=(8, 0))
        for var, key in ((self.var_training, "gui_training"),
                         (self.var_no_output, "gui_no_output"),
                         (self.var_keep_junk, "gui_keep_junk"),
                         (self.var_concat_only, "gui_concat_only")):
            self._tr(ttk.Checkbutton(self.adv, variable=var, style="Card.TCheckbutton",
                                     command=self._sync_states), key).pack(anchor="w", pady=(4, 0))
        if self.adv_open.get():
            self.adv.pack(fill=tk.X)

        # actions + progress
        ac = ttk.Frame(root)
        ac.pack(fill=tk.X, pady=(2, 8))
        self.btn_run = self._tr(ttk.Button(ac, command=self._run, style="Accent.TButton"),
                                "gui_run", "▶  ")
        self.btn_run.pack(side=tk.LEFT, ipadx=18)
        self.btn_stop = self._tr(ttk.Button(ac, command=self._stop, state=tk.DISABLED),
                                 "gui_stop", "■  ")
        self.btn_stop.pack(side=tk.LEFT, padx=(8, 0))
        self.btn_open = self._tr(ttk.Button(ac, command=self._open_output, state=tk.DISABLED),
                                 "gui_open")
        self.btn_open.pack(side=tk.RIGHT)

        pf = ttk.Frame(root)
        pf.pack(fill=tk.X, pady=(0, 8))
        self.pbar = ttk.Progressbar(pf, mode="determinate", maximum=100)
        self.pbar.pack(fill=tk.X)
        self.lbl_status = ttk.Label(pf, style="Muted.TLabel")
        self.lbl_status.pack(anchor="w", pady=(4, 0))

        # log
        lh = ttk.Frame(root)
        lh.pack(fill=tk.X)
        self._tr(ttk.Label(lh, style="Head.TLabel"), "gui_log").pack(side=tk.LEFT)
        self._tr(ttk.Button(lh, command=self._clear_log, style="Link.TButton"),
                 "gui_clear").pack(side=tk.RIGHT)
        lf = ttk.Frame(root, style="Border.TFrame", padding=1)
        lf.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.log = tk.Text(lf, height=10, wrap=tk.WORD, state=tk.DISABLED, relief=tk.FLAT,
                           borderwidth=0, padx=10, pady=8, font=self.f_mono)
        sb = ttk.Scrollbar(lf, orient=tk.VERTICAL, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # footer
        ft = ttk.Frame(root)
        ft.pack(fill=tk.X, pady=(8, 0))
        self.lbl_hw = ttk.Label(ft, style="Small.TLabel")
        self.lbl_hw.pack(anchor="w")
        cr = ttk.Frame(root)
        cr.pack(fill=tk.X, pady=(2, 0))
        self._tr(ttk.Label(cr, style="Small.TLabel"), "gui_credits").pack(side=tk.LEFT)
        if __version__:
            ttk.Label(cr, text=f"v{__version__}", style="Small.TLabel").pack(side=tk.RIGHT)

    def _spin(self, parent, key, var, lo, hi, row, col):
        self._tr(ttk.Label(parent, style="Card.TLabel"), key).grid(
            row=row, column=col, sticky="w", pady=2)
        sp = ttk.Spinbox(parent, from_=lo, to=hi, textvariable=var, width=7)
        sp.grid(row=row, column=col + 1, sticky="w", padx=(8, 24), pady=2)
        return sp

    # ------------------------------------------------------------ theme / texts
    def _apply_theme(self):
        c = DARK if self.dark.get() else LIGHT
        self.c = c
        s = self.style
        self.configure(bg=c["bg"])
        for opt, val in (("background", c["field"]), ("foreground", c["fg"]),
                         ("selectBackground", c["accent"]), ("selectForeground", c["accent_fg"])):
            self.option_add(f"*TCombobox*Listbox.{opt}", val)

        s.configure(".", background=c["bg"], foreground=c["fg"], bordercolor=c["border"],
                    darkcolor=c["bg"], lightcolor=c["bg"], troughcolor=c["field"],
                    focuscolor=c["accent"], selectbackground=c["accent"],
                    selectforeground=c["accent_fg"], insertcolor=c["fg"])
        s.configure("TFrame", background=c["bg"])
        s.configure("Card.TFrame", background=c["card"], relief="solid",
                    borderwidth=1, bordercolor=c["border"])
        s.configure("Inner.TFrame", background=c["card"], borderwidth=0)
        s.configure("Border.TFrame", background=c["border"])
        s.configure("TLabel", background=c["bg"], foreground=c["fg"])
        s.configure("Title.TLabel", font=self.f_title)
        s.configure("Head.TLabel", font=self.f_head)
        s.configure("Muted.TLabel", foreground=c["muted"])
        s.configure("Small.TLabel", foreground=c["muted"], font=self.f_small)
        s.configure("Card.TLabel", background=c["card"])
        s.configure("Head.Card.TLabel", background=c["card"], font=self.f_head)
        s.configure("Muted.Card.TLabel", background=c["card"], foreground=c["muted"],
                    font=self.f_small)

        s.configure("TButton", background=c["field"], foreground=c["fg"], padding=(12, 6),
                    bordercolor=c["border"], lightcolor=c["field"], darkcolor=c["field"],
                    focusthickness=0, relief="flat")
        s.map("TButton",
              background=[("disabled", c["card"]), ("pressed", c["border"]), ("active", c["border"])],
              foreground=[("disabled", c["muted"])],
              bordercolor=[("focus", c["accent"])])
        s.configure("Accent.TButton", background=c["accent"], foreground=c["accent_fg"],
                    bordercolor=c["accent"], lightcolor=c["accent"], darkcolor=c["accent"],
                    font=self.f_head)
        s.map("Accent.TButton",
              background=[("disabled", c["border"]), ("pressed", c["accent"]),
                          ("active", c["accent_hover"])],
              foreground=[("disabled", c["muted"])])
        s.configure("Ghost.TButton", background=c["bg"], bordercolor=c["border"],
                    lightcolor=c["bg"], darkcolor=c["bg"], padding=(10, 4))
        s.map("Ghost.TButton", background=[("active", c["field"])])
        for name, bg in (("Link.TButton", c["bg"]), ("Link.Card.TButton", c["card"])):
            s.configure(name, background=bg, foreground=c["accent"], bordercolor=bg,
                        lightcolor=bg, darkcolor=bg, padding=(0, 2))
            s.map(name, background=[("active", bg)], foreground=[("active", c["accent_hover"])],
                  bordercolor=[("focus", bg)])

        s.configure("Card.TCheckbutton", background=c["card"], foreground=c["fg"],
                    indicatorbackground=c["field"], indicatorforeground=c["accent"],
                    upperbordercolor=c["border"], lowerbordercolor=c["border"],
                    focusthickness=0, padding=(0, 2))
        s.map("Card.TCheckbutton",
              background=[("active", c["card"])],
              indicatorbackground=[("selected", c["field"]), ("active", c["field"])],
              foreground=[("disabled", c["muted"])])

        for w in ("TEntry", "TCombobox", "TSpinbox"):
            s.configure(w, fieldbackground=c["field"], foreground=c["fg"], background=c["field"],
                        bordercolor=c["border"], lightcolor=c["field"], darkcolor=c["field"],
                        arrowcolor=c["muted"], insertcolor=c["fg"], padding=4)
            s.map(w, bordercolor=[("focus", c["accent"])],
                  lightcolor=[("focus", c["accent"])],
                  fieldbackground=[("readonly", c["field"]), ("disabled", c["card"])],
                  foreground=[("disabled", c["muted"]), ("readonly", c["fg"])],
                  selectbackground=[("readonly", c["field"])],
                  selectforeground=[("readonly", c["fg"])],
                  background=[("active", c["field"]), ("readonly", c["field"])],
                  arrowcolor=[("disabled", c["border"])])
        s.configure("TProgressbar", background=c["accent"], troughcolor=c["field"],
                    bordercolor=c["border"], lightcolor=c["accent"], darkcolor=c["accent"],
                    thickness=8)
        s.configure("Vertical.TScrollbar", background=c["field"], troughcolor=c["log_bg"],
                    bordercolor=c["log_bg"], arrowcolor=c["muted"],
                    lightcolor=c["field"], darkcolor=c["field"])
        s.map("Vertical.TScrollbar", background=[("active", c["border"])])

        self.log.configure(bg=c["log_bg"], fg=c["log_fg"], insertbackground=c["fg"],
                           selectbackground=c["accent"], selectforeground=c["accent_fg"])
        self.log.tag_configure("err", foreground=c["err"])
        self.log.tag_configure("ok", foreground=c["ok"])
        self.log.tag_configure("info", foreground=c["accent"])
        self.btn_theme.configure(text="☀" if self.dark.get() else "☾")

    def _refresh_texts(self):
        for w, key, prefix in self._texts:
            w.configure(text=prefix + self.T(key))
        names = (self.T("gui_lang_auto"), "Italiano", "English")
        self.cb_lang.configure(values=names)
        self.cb_lang.set(names[("auto", "it", "en").index(self.lang_choice.get())
                               if self.lang_choice.get() in ("auto", "it", "en") else 0])
        self.btn_adv.configure(text=("▾  " if self.adv_open.get() else "▸  ") + self.T("gui_advanced"))
        self._update_out_hint()
        self.lbl_hw.configure(text=self.hw_text or self.T("gui_hw_probe"))
        if not self.proc:
            self._set_status(self.T("gui_ready"))

    def _toggle_theme(self):
        self.dark.set(not self.dark.get())
        self._apply_theme()

    def _on_lang(self, _e=None):
        self.lang_choice.set(("auto", "it", "en")[max(0, self.cb_lang.current())])
        self.lang = self._resolve_lang()
        self.S = load_strings(BASE_DIR, self.lang)
        self._refresh_texts()
        self.cb_lang.selection_clear()

    def _toggle_adv(self):
        self.adv_open.set(not self.adv_open.get())
        if self.adv_open.get():
            self.adv.pack(fill=tk.X)
        else:
            self.adv.pack_forget()
        self._refresh_texts()

    def _sync_states(self):
        """Grey out options that the current combination makes irrelevant."""
        reencode = not self.var_fastcut.get()
        for w in (self.cb_enc,):
            w.configure(state="readonly" if reencode else "disabled")
        for w in (self.ent_crf, self.ent_fps):
            w.configure(state="normal" if reencode else "disabled")

    def _update_out_hint(self):
        f = self.var_file.get().strip()
        if f:
            folder = str(Path(f).parent)
            if len(folder) > 70:  # keep the window from growing with long paths
                folder = folder[:30] + "…" + folder[-38:]
            self.lbl_out.configure(text=self.T("gui_out_hint").format(folder=folder))
        else:
            self.lbl_out.configure(text=self.T("gui_pick_hint"))

    # ------------------------------------------------------------ actions
    def _browse(self):
        init = self.cfg.get("last_dir") or str(Path.home())
        f = filedialog.askopenfilename(initialdir=init, filetypes=[
            ("Video", VIDEO_TYPES), (self.T("gui_all_files"), "*.*")])
        if f:
            self.var_file.set(f)
            self.cfg["last_dir"] = str(Path(f).parent)

    def _browse_model(self):
        f = filedialog.askopenfilename(initialdir=str(BASE_DIR),
                                       filetypes=[("Keras", "*.keras"), (self.T("gui_all_files"), "*.*")])
        if f:
            self.var_model.set(f)

    def _clear_log(self):
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

    def _append(self, s: str, tag: str = ""):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, s, (tag,) if tag else ())
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _set_status(self, text: str, color: str = ""):
        self.lbl_status.configure(text=text, foreground=color or self.c["muted"])

    def _open_output(self):
        if self.last_output and self.last_output.exists():
            _open_path(self.last_output.parent)
        elif self.var_file.get():
            _open_path(Path(self.var_file.get()).parent)

    @staticmethod
    def _int_or_none(v: str):
        v = (v or "").strip()
        return int(v) if v.lstrip("-").isdigit() and int(v) > 0 else None

    def _build_cmd(self) -> list[str] | None:
        f = self.var_file.get().strip().strip('"')
        if not f or not Path(f).is_file():
            self._append("⚠ " + self.T("gui_err_file") + "\n", "err")
            return None
        keep = [k for k in LABELS if self.var_keep[k].get()]
        if not keep and not self.var_no_output.get():
            self._append("⚠ " + self.T("gui_err_keep") + "\n", "err")
            return None
        cmd = _find_cli() + [f, "--device", self.var_device.get(),
                             "--encoder", self.var_encoder.get(), "--lang", self.lang]
        if keep:
            cmd += ["--keep", *keep]
        flags = ((self.var_fastcut, "--fastcut"), (self.var_srt, "--srt"),
                 (self.var_training, "--generate-training-data"),
                 (self.var_no_output, "--no-output"), (self.var_keep_junk, "--keep-junk"),
                 (self.var_concat_only, "--concat-only"))
        cmd += [flag for var, flag in flags if var.get()]
        if not self.var_fastcut.get():
            for var, flag in ((self.var_crf, "--crf"), (self.var_fps, "--fps")):
                n = self._int_or_none(var.get())
                if n:
                    cmd += [flag, str(n)]
        for var, flag, default in ((self.var_win, "--window-size-divide", "1"),
                                   (self.var_slide, "--window-slide-divide", "2")):
            v = var.get().strip()
            if v and v != default:
                cmd += [flag, v]
        if self.var_model.get().strip():
            cmd += ["--model", self.var_model.get().strip()]
        return cmd

    def _run(self):
        if self.proc:
            return
        cmd = self._build_cmd()
        if not cmd:
            return
        self.stopped = False
        self.last_output = None
        self.btn_run.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.btn_open.configure(state=tk.DISABLED)
        self.pbar.configure(value=0)
        self._set_status(self.T("gui_starting"))
        self._append("$ " + " ".join(f'"{a}"' if " " in a else a for a in cmd[len(_find_cli()):]) + "\n", "info")
        self._save_settings()
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, cwd=str(WORK_DIR), env=_child_env(),
                creationflags=NO_WINDOW, start_new_session=not IS_WIN)
        except Exception as e:
            self.q.put(("line", f"ERROR: {e}\n"))
            self.q.put(("end", -1))
            return
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc: subprocess.Popen):
        """Split output on \\n and \\r so tqdm redraws become progress events."""
        buf = ""
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        fd = proc.stdout.fileno()
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += dec.decode(chunk)
            parts = re.split(r"(\r\n|\n|\r)", buf)
            buf = parts.pop()
            for i in range(0, len(parts), 2):
                self._dispatch(parts[i], parts[i + 1] != "\r")
        if buf:
            self._dispatch(buf, True)
        rc = proc.wait()
        self.q.put(("end", rc))

    def _dispatch(self, line: str, newline: bool):
        m = TQDM_RE.search(line)
        if m:
            self.q.put(("progress", int(m.group(1)), m.group(3).strip(), m.group(4).strip()))
            return
        if not line.strip():
            return
        self.q.put(("line", line + "\n"))

    def _poll(self):
        try:
            while True:
                ev = self.q.get_nowait()
                if ev[0] == "line":
                    self._on_line(ev[1])
                elif ev[0] == "progress":
                    _, pct, elapsed, left = ev
                    self.pbar.configure(value=pct)
                    phase = getattr(self, "_phase", self.T("gui_starting"))
                    eta = f" · {elapsed}" + (f" / -{left}" if left and left != "?" else "")
                    self._set_status(f"{phase} — {pct}%{eta}")
                elif ev[0] == "end":
                    self._on_end(ev[1])
        except queue.Empty:
            pass
        self.after(80, self._poll)

    def _on_line(self, line: str):
        low = line.lower()
        for needle, key in PHASES:
            if needle.lower() in low:
                self._phase = self.T(key)
                self.pbar.configure(value=0)
                self._set_status(self._phase)
        if line.startswith("saved:"):
            p = line.split(":", 1)[1].strip()
            self.last_output = Path(p) if Path(p).is_absolute() else WORK_DIR / p
        tag = "err" if ("error" in low or "traceback" in low or "failed" in low) else ""
        self._append(line, tag)

    def _on_end(self, rc: int):
        self.proc = None
        self._phase = self.T("gui_starting")
        self.btn_run.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        if self.stopped:
            self._set_status(self.T("gui_stopped"), self.c["warn"])
            self._append(f"\n[{self.T('gui_stopped')}]\n", "err")
        elif rc == 0:
            self.pbar.configure(value=100)
            self._set_status(self.T("gui_done"), self.c["ok"])
            self._append(f"\n[{self.T('gui_done')}]\n", "ok")
            self.btn_open.configure(state=tk.NORMAL)
        else:
            self._set_status(self.T("gui_failed").format(rc=rc), self.c["err"])
            self._append(f"\n[{self.T('gui_failed').format(rc=rc)}]\n", "err")

    def _stop(self):
        p = self.proc
        if not p or p.poll() is not None:
            return
        self.stopped = True
        self._set_status(self.T("gui_stopping"), self.c["warn"])
        try:
            if IS_WIN:  # kill the whole tree (ffmpeg children included)
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                               capture_output=True, creationflags=NO_WINDOW)
            else:
                import signal
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            with_suppress(p.kill)

    def _probe_hw(self):
        try:
            out = subprocess.run(
                _find_cli() + ["--list-devices"], capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=180, cwd=str(WORK_DIR),
                env=_child_env(), stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
            info = {}
            for ln in (out.stdout or "").splitlines():
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    info[k.strip()] = v.strip()
            gpus = info.get("tensorflow gpus", "?")
            gpu = "—" if gpus.startswith("none") else gpus.strip("[]'")
            text = f"GPU (TensorFlow): {gpu}   ·   NVENC: {info.get('ffmpeg nvenc', '?')}"
        except Exception:
            text = self.T("gui_hw_fail")
        self.hw_text = text
        self.after(0, lambda: self.lbl_hw.configure(text=text))

    # ------------------------------------------------------------ settings
    def _load_settings(self) -> dict:
        try:
            return json.loads(_settings_path().read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self):
        self.cfg.update(
            lang=self.lang_choice.get(), dark=self.dark.get(),
            device=self.var_device.get(), encoder=self.var_encoder.get(),
            fastcut=self.var_fastcut.get(), srt=self.var_srt.get(),
            keep=[k for k in LABELS if self.var_keep[k].get()],
            crf=self.var_crf.get(), fps=self.var_fps.get(), win=self.var_win.get(),
            slide=self.var_slide.get(), model=self.var_model.get(),
            adv_open=self.adv_open.get(), geometry=self.geometry())
        try:
            p = _settings_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self.cfg, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_close(self):
        if self.proc and self.proc.poll() is None:
            self._stop()
        self._save_settings()
        self.destroy()


def with_suppress(fn):
    try:
        return fn()
    except Exception:
        return None


if __name__ == "__main__":
    if IS_WIN:  # crisp rendering on HiDPI screens
        with_suppress(lambda: __import__("ctypes").windll.shcore.SetProcessDpiAwareness(1))
    App(sys.argv[1] if len(sys.argv) > 1 else "").mainloop()
