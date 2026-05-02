"""ClothingSnap main application window."""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
import os
import queue

from processing.pipeline import ProcessingConfig, collect_images
from processing.batch import BatchProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

BG = "#0F0F11"
SURFACE = "#1A1A1F"
SURFACE2 = "#24242B"
BORDER = "#2E2E38"
ACCENT = "#6C63FF"
ACCENT_H = "#8B84FF"
TEXT = "#F0F0F5"
TEXT_DIM = "#8888A0"
SUCCESS = "#4CAF82"
ERROR = "#E05C5C"
FONT_MAIN = ("Segoe UI", 10)
FONT_LABEL = ("Segoe UI", 9)
FONT_SMALL = ("Segoe UI", 8)


class ClothingSnapApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("")
        self.root.geometry("620x500")
        self.root.minsize(540, 460)
        self.root.configure(bg=BG)

        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()

        self.processor = BatchProcessor()
        self._queue: queue.Queue = queue.Queue()
        self._processing = False

        self._setup_styles()
        self._build_ui()
        self._poll_queue()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground=TEXT,
            font=("Segoe UI Semibold", 10),
            borderwidth=0,
            focusthickness=0,
            padding=(20, 10),
        )
        style.map(
            "Accent.TButton",
            background=[("active", ACCENT_H), ("disabled", BORDER)],
            foreground=[("disabled", TEXT_DIM)],
        )

        style.configure(
            "Ghost.TButton",
            background=SURFACE2,
            foreground=TEXT_DIM,
            font=FONT_MAIN,
            borderwidth=0,
            focusthickness=0,
            padding=(16, 10),
        )
        style.map(
            "Ghost.TButton",
            background=[("active", BORDER)],
            foreground=[("active", TEXT)],
        )

        style.configure(
            "TProgressbar",
            troughcolor=SURFACE2,
            background=ACCENT,
            thickness=6,
            borderwidth=0,
        )

    def _build_ui(self):
        root = self.root

        header = tk.Frame(root, bg=BG)
        header.pack(fill="x", padx=24, pady=(20, 0))

        tk.Label(
            header,
            text="Clothing Image Optimizer ", 
            font=("Segoe UI Semibold", 18),
            bg=BG,
            fg=TEXT,
        ).pack(side="left")
        tk.Label(
            header,
            text="Studio Editor",
            font=FONT_LABEL,
            bg=BG,
            fg=TEXT_DIM,
        ).pack(side="left", padx=(10, 0), pady=(6, 0))

        self._divider(root)

        self._section_label(root, "FOLDERS")
        folders_card = self._card(root)
        self._folder_row(folders_card, "Input folder", self.input_folder, self._browse_input)
        self._thin_divider(folders_card)
        self._folder_row(folders_card, "Output folder", self.output_folder, self._browse_output)

        self._section_label(root, "PROGRESS")
        prog_card = self._card(root)
        self._build_progress(prog_card)

        btn_frame = tk.Frame(root, bg=BG)
        btn_frame.pack(fill="x", padx=24, pady=(16, 24))

        self.start_btn = ttk.Button(
            btn_frame,
            text="Start Processing",
            style="Accent.TButton",
            command=self._start,
        )
        self.start_btn.pack(side="left")

        self.cancel_btn = ttk.Button(
            btn_frame,
            text="Cancel",
            style="Ghost.TButton",
            command=self._cancel,
            state="disabled",
        )
        self.cancel_btn.pack(side="left", padx=(10, 0))

        self.status_var = tk.StringVar(value="Ready - select folders to begin")
        tk.Label(
            btn_frame,
            textvariable=self.status_var,
            font=FONT_SMALL,
            bg=BG,
            fg=TEXT_DIM,
            anchor="e",
        ).pack(side="right")

    def _divider(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=24, pady=(16, 0))

    def _thin_divider(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=16, pady=0)

    def _section_label(self, parent, text):
        tk.Label(
            parent,
            text=text,
            font=("Segoe UI", 8, "bold"),
            bg=BG,
            fg=TEXT_DIM,
            anchor="w",
        ).pack(fill="x", padx=26, pady=(16, 6))

    def _card(self, parent):
        frame = tk.Frame(
            parent,
            bg=SURFACE,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        frame.pack(fill="x", padx=24, pady=(0, 0))
        return frame

    def _folder_row(self, parent, label, var, command):
        row = tk.Frame(parent, bg=SURFACE, pady=12, padx=16)
        row.pack(fill="x")

        tk.Label(row, text=label, font=FONT_LABEL, bg=SURFACE, fg=TEXT_DIM, width=12, anchor="w").pack(side="left")

        entry = tk.Entry(
            row,
            textvariable=var,
            font=FONT_LABEL,
            bg=SURFACE2,
            fg=TEXT,
            relief="flat",
            insertbackground=TEXT,
            bd=0,
            highlightthickness=0,
        )
        entry.pack(side="left", fill="x", expand=True, padx=(8, 8), ipady=6)

        tk.Button(
            row,
            text="Browse",
            font=FONT_SMALL,
            bg=SURFACE2,
            fg=TEXT_DIM,
            activebackground=BORDER,
            activeforeground=TEXT,
            relief="flat",
            cursor="hand2",
            bd=0,
            padx=10,
            pady=4,
            command=command,
        ).pack(side="right")

    def _build_progress(self, parent):
        inner = tk.Frame(parent, bg=SURFACE, padx=16, pady=14)
        inner.pack(fill="x")

        counter_row = tk.Frame(inner, bg=SURFACE)
        counter_row.pack(fill="x", pady=(0, 8))

        self.counter_var = tk.StringVar(value="0 / 0 images")
        tk.Label(counter_row, textvariable=self.counter_var, font=("Segoe UI Semibold", 11), bg=SURFACE, fg=TEXT).pack(
            side="left"
        )

        self.pct_var = tk.StringVar(value="0%")
        tk.Label(counter_row, textvariable=self.pct_var, font=FONT_LABEL, bg=SURFACE, fg=ACCENT).pack(side="right")

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(inner, variable=self.progress_var, maximum=100, style="TProgressbar")
        self.progress_bar.pack(fill="x")

        self.result_var = tk.StringVar(value="")
        tk.Label(inner, textvariable=self.result_var, font=FONT_SMALL, bg=SURFACE, fg=TEXT_DIM, anchor="w").pack(
            fill="x", pady=(8, 0)
        )

    def _browse_input(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_folder.set(folder)
            images = collect_images(folder)
            self.status_var.set(f"{len(images)} image(s) found in input folder")

    def _browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)

    def _start(self):
        inp = self.input_folder.get().strip()
        out = self.output_folder.get().strip()

        if not inp or not os.path.isdir(inp):
            messagebox.showerror("Error", "Please select a valid input folder.")
            return
        if not out:
            messagebox.showerror("Error", "Please select an output folder.")
            return

        images = collect_images(inp)
        if not images:
            messagebox.showwarning("No Images", "No supported images found in the input folder.")
            return

        config = ProcessingConfig(
            canvas_width=1200,
            canvas_height=1600,
            fill_ratio=0.85,
            output_format="WEBP",
            smooth_edges=True,
            alpha_threshold=2,
            use_cropped_size_canvas=True,
        )

        self._processing = True
        self._total = len(images)
        self._set_running_state(True)
        self.counter_var.set(f"0 / {self._total} images")
        self.pct_var.set("0%")
        self.progress_var.set(0)
        self.result_var.set("")
        self.status_var.set("Processing...")

        logger.info("[ui] Starting batch: %s images", len(images))
        self.processor.start(
            image_paths=images,
            output_folder=out,
            config=config,
            on_progress=self._on_progress,
            on_complete=self._on_complete,
        )

    def _cancel(self):
        self.processor.cancel()
        self.status_var.set("Cancelling...")
        self.cancel_btn.config(state="disabled")

    def _on_progress(self, done: int, total: int, msg: str):
        self._queue.put(("progress", done, total, msg))

    def _on_complete(self, success: int, failed: int):
        self._queue.put(("complete", success, failed))

    def _poll_queue(self):
        try:
            while True:
                item = self._queue.get_nowait()
                if item[0] == "progress":
                    _, done, total, msg = item
                    pct = (done / total * 100) if total else 0
                    self.progress_var.set(pct)
                    self.counter_var.set(f"{done} / {total} images")
                    self.pct_var.set(f"{int(pct)}%")
                    self.status_var.set(msg)
                elif item[0] == "complete":
                    _, success, failed = item
                    self._processing = False
                    self._set_running_state(False)
                    self.result_var.set(
                        f"✓ {success} succeeded" + (f"  ✗ {failed} failed" if failed else "")
                    )
                    self.status_var.set(f"Done - {success}/{success + failed} images processed")
                    self.progress_var.set(100)
        except queue.Empty:
            pass
        except Exception:
            logger.exception("[ui] Queue polling error")
        self.root.after(100, self._poll_queue)

    def _set_running_state(self, running: bool):
        if running:
            self.start_btn.config(state="disabled")
            self.cancel_btn.config(state="normal")
        else:
            self.start_btn.config(state="normal")
            self.cancel_btn.config(state="disabled")

    def run(self):
        self.root.mainloop()
