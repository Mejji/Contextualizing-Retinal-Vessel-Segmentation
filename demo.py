import re
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk


class RetinalDemo(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Retinal Vessel Segmentation Demo")
        self.geometry("1200x800")

        # Base assets folder: THESIS/system-demo/assets
        self.base_dir = Path(__file__).resolve().parent / "assets"

        # Display names in the combo boxes  -> actual folder names
        self.model_map = {
            "DRIU_VGN (Baseline)": "DRIU_VGN_BASELINE",
            "DRIU_VGN_AFF (Static)": "DRIU_VGN_AFF_STATIC",
            "DRIU_VGN_AFF (Dynamic)": "DRIU_VGN_AFF_DYNAMIC",
            "(DA-U)²Net_VGN": "DAU2Net_VGN",
            "(DA-U)²Net_AFF (Dynamic)": "DAU2Net_AFF_Dynamic",  # note: CamelCase 'Dynamic'
        }

        self.dataset_map = {
            "DRIVE": "drive",
            "CHASE": "chase",
            "HRF": "hrf",
        }

        # Pipeline stages (index = slider value)
        self.stage_keys = ["fundus", "cnn", "graph", "seg", "gt"]
        self.stage_names = {
            "fundus": "Fundus (Input)",
            "cnn": "CNN Prob. Map",
            "graph": "Graph Refinement",
            "seg": "Final Segmentation",
            "gt": "Ground Truth",
        }

        # Loaded cases for current model+dataset
        self.cases = []
        self.case_index = 0
        self.image_cache = None  # keep reference to avoid GC
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        self.pan_start = None
        self.canvas_image_id = None
        self.stage_buttons = []
        self.legend_labels = []
        self.legend_mode = None

        self._build_ui()
        self.load_cases()  # initial load

    # ---------- UI ----------

    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=5)

        # Model selection
        ttk.Label(top, text="Model:").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value="DRIU_VGN (Baseline)")
        self.model_combo = ttk.Combobox(
            top,
            textvariable=self.model_var,
            state="readonly",
            width=28,
            values=list(self.model_map.keys()),
        )
        self.model_combo.grid(row=0, column=1, sticky="w", padx=(2, 15))
        self.model_combo.bind("<<ComboboxSelected>>", lambda e: self.load_cases())

        # Dataset selection
        ttk.Label(top, text="Dataset:").grid(row=0, column=2, sticky="w")
        self.dataset_var = tk.StringVar(value="DRIVE")
        self.dataset_combo = ttk.Combobox(
            top,
            textvariable=self.dataset_var,
            state="readonly",
            width=10,
            values=list(self.dataset_map.keys()),
        )
        self.dataset_combo.grid(row=0, column=3, sticky="w")
        self.dataset_combo.bind("<<ComboboxSelected>>", lambda e: self.load_cases())

        # Case navigation
        self.case_label_var = tk.StringVar(value="Case: 0 / 0")
        ttk.Label(top, textvariable=self.case_label_var).grid(
            row=0, column=4, padx=(30, 5)
        )
        ttk.Button(top, text="<", width=3, command=self.prev_case).grid(
            row=0, column=5, padx=2
        )
        ttk.Button(top, text=">", width=3, command=self.next_case).grid(
            row=0, column=6, padx=2
        )

        # Stage slider
        stage_frame = ttk.Frame(self)
        stage_frame.pack(fill="x", padx=10, pady=(5, 0))

        ttk.Label(stage_frame, text="Stage:").grid(row=0, column=0, sticky="w")

        self.stage_var = tk.IntVar(value=0)
        stage_buttons = ttk.Frame(stage_frame)
        stage_buttons.grid(row=0, column=1, sticky="w")
        for idx, key in enumerate(self.stage_keys):
            btn = ttk.Radiobutton(
                stage_buttons,
                text=self.stage_names[key],
                variable=self.stage_var,
                value=idx,
                command=self.on_stage_button,
            )
            btn.grid(row=0, column=idx, padx=2)
            self.stage_buttons.append(btn)

        control_frame = ttk.Frame(stage_frame)
        control_frame.grid(row=0, column=2, sticky="e", padx=(30, 0))

        ttk.Button(control_frame, text="Compare", command=self.on_compare).grid(
            row=0, column=0, padx=(0, 8)
        )

        zoom_frame = ttk.Frame(control_frame)
        zoom_frame.grid(row=0, column=1, sticky="e")
        ttk.Label(zoom_frame, text="Zoom:").grid(row=0, column=0, sticky="e")
        ttk.Button(zoom_frame, text="-", width=3, command=lambda: self.adjust_zoom(0.85)).grid(
            row=0, column=1, padx=2
        )
        ttk.Button(zoom_frame, text="+", width=3, command=lambda: self.adjust_zoom(1.15)).grid(
            row=0, column=2, padx=2
        )
        ttk.Button(zoom_frame, text="Reset", width=6, command=self.reset_zoom).grid(
            row=0, column=3, padx=(6, 0)
        )
        self.zoom_label_var = tk.StringVar(value="100%")
        ttk.Label(zoom_frame, textvariable=self.zoom_label_var).grid(
            row=0, column=4, padx=(6, 0)
        )

        stage_frame.columnconfigure(1, weight=1)

        self.update_stage_label()

        # Image canvas
        self.canvas = tk.Canvas(self, bg="black")
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas.bind("<ButtonPress-1>", self.on_pan_start)
        self.canvas.bind("<B1-Motion>", self.on_pan_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_pan_end)
        self.canvas.bind("<Leave>", self.on_pan_end)

        # Legend panel (hidden until used)
        self.legend_frame = ttk.Frame(self)
        for text, color in [
            ("TP: correct vessel", "#ffffff"),
            ("TN: correct background", "#000000"),
            ("FP: false positive", "#00ff00"),
            ("FN: missed vessel", "#ff0000"),
        ]:
            box = tk.Label(self.legend_frame, width=2, background=color, relief="ridge")
            box.pack(side="left", padx=(0, 4))
            lbl = ttk.Label(self.legend_frame, text=text)
            lbl.pack(side="left", padx=(0, 12))
            self.legend_labels.append((box, lbl))
        self.set_legend_visible(False)

    # ---------- Case loading ----------

    def load_cases(self):
        """Load all cases for current model+dataset."""
        model_display = self.model_var.get()
        dataset_display = self.dataset_var.get()

        model_dir = self.model_map[model_display]
        dataset_dir = self.dataset_map[dataset_display]

        folder = self.base_dir / model_dir / dataset_dir

        if not folder.exists():
            messagebox.showwarning(
                "Folder not found",
                f"Folder does not exist:\n{folder}\n\n"
                f"Check the model/dataset folder names.",
            )
            self.cases = []
            self.case_index = 0
            self.update_case_label()
            self.clear_canvas()
            return

        self.cases = self.scan_cases(folder)
        self.case_index = 0
        self.stage_var.set(0)
        self.update_stage_label()
        self.update_case_label()
        self.zoom_factor = 1.0
        if hasattr(self, "zoom_label_var"):
            self.zoom_label_var.set("100%")
        self.reset_pan()
        self.set_legend_visible(False)

        if not self.cases:
            messagebox.showwarning(
                "No cases",
                f"No images found for:\n"
                f"Model: {model_display}\nDataset: {dataset_display}",
            )
            self.clear_canvas()
        else:
            self.ensure_valid_stage(self.cases[0])
            self.show_current_image()

    def scan_cases(self, folder: Path):
        """
        Group images into cases by stage using filename substrings.

        Expected filenames (order doesn’t matter, just consistent):
          fundus_*.jpg / .png
          cnn_* or cnn_prob_*.png
          graph_*.png
          seg_*.png
          gt_*.png
        """
        stage_files = {k: [] for k in self.stage_keys}

        stage_keywords = {
            "fundus": ["fundus"],
            "cnn": ["cnn_prob", "cnn"],
            "graph": ["graph"],
            "seg": ["seg", "segment"],
            "gt": ["gt", "groundtruth", "ground_truth"],
        }

        allowed_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".JPG", ".JPEG", ".gif"}

        for path in sorted(folder.glob("*")):
            if path.suffix.lower() not in allowed_exts:
                continue

            stem = path.stem.lower()
            for stage, keywords in stage_keywords.items():
                if any(kw in stem for kw in keywords):
                    stage_files[stage].append(path)
                    break

        if all(len(v) == 0 for v in stage_files.values()):
            return []

        # If every non-empty stage has the same number of files, pair by index
        # (useful when numeric suffixes are inconsistent across stages).
        nonempty_stages = [k for k in self.stage_keys if stage_files[k]]
        counts = [len(stage_files[k]) for k in nonempty_stages]
        if counts and len(set(counts)) == 1:
            n = counts[0]
            cases = []
            for i in range(n):
                case = {}
                for stage in self.stage_keys:
                    case[stage] = stage_files[stage][i] if i < len(stage_files[stage]) else None
                cases.append(case)
            return cases

        cases_by_id = {}
        for stage, items in stage_files.items():
            for idx, path in enumerate(items):
                case_id = self.extract_case_id(path.stem)
                cid = case_id if case_id is not None else f"{stage}_{idx}"
                entry = cases_by_id.setdefault(cid, {k: None for k in self.stage_keys})
                entry[stage] = path

        def sort_key(cid):
            try:
                return (0, int(cid))
            except (TypeError, ValueError):
                return (1, str(cid))

        sorted_ids = sorted(cases_by_id.keys(), key=sort_key)
        return [cases_by_id[cid] for cid in sorted_ids]

    @staticmethod
    def extract_case_id(stem: str):
        """Return trailing integer in filename stem, if present."""
        match = re.search(r"(\d+)(?!.*\d)", stem)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return match.group(1)
        return None

    # ---------- Navigation ----------

    def prev_case(self):
        if not self.cases:
            return
        self.case_index = (self.case_index - 1) % len(self.cases)
        self.update_case_label()
        self.reset_pan()
        self.ensure_valid_stage(self.cases[self.case_index])
        self.set_legend_visible(False)
        self.show_current_image()

    def next_case(self):
        if not self.cases:
            return
        self.case_index = (self.case_index + 1) % len(self.cases)
        self.update_case_label()
        self.reset_pan()
        self.ensure_valid_stage(self.cases[self.case_index])
        self.set_legend_visible(False)
        self.show_current_image()

    def update_case_label(self):
        total = len(self.cases)
        if total == 0:
            txt = "Case: 0 / 0"
        else:
            txt = f"Case: {self.case_index + 1} / {total}"
        self.case_label_var.set(txt)

    # ---------- Stage slider ----------

    def update_stage_label(self):
        idx = self.stage_var.get()
        idx = max(0, min(idx, len(self.stage_keys) - 1))
        self.stage_var.set(idx)

    def update_stage_buttons(self, enabled_stages):
        for stage, btn in zip(self.stage_keys, self.stage_buttons):
            if stage in enabled_stages:
                btn.state(["!disabled"])
            else:
                btn.state(["disabled"])

    def set_legend_visible(self, visible: bool):
        """Show or hide the overlay legend."""
        if visible:
            if self.legend_frame.winfo_manager() != "pack":
                self.legend_frame.pack(fill="x", padx=10, pady=(0, 8))
            self.legend_mode = "diff"
        else:
            if self.legend_frame.winfo_manager():
                self.legend_frame.pack_forget()
            self.legend_mode = None

    def ensure_valid_stage(self, case):
        enabled = [s for s, path in case.items() if path is not None]
        self.update_stage_buttons(set(enabled))
        if not enabled:
            return False
        current_stage = self.stage_keys[self.stage_var.get()]
        if current_stage not in enabled:
            self.stage_var.set(self.stage_keys.index(enabled[0]))
        return True

    # ---------- Overlays ----------

    @staticmethod
    def load_gray(path: Path):
        """Load an image as grayscale float32 in [0,1]."""
        img = Image.open(path).convert("L")
        arr = np.array(img).astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        return arr

    @staticmethod
    def prob_to_binary(prob, thr=0.5):
        return (prob >= thr).astype(np.uint8)

    @staticmethod
    def resize_like(arr: np.ndarray, target_shape, resample=Image.NEAREST):
        """Resize numpy image array to target (H, W) shape using PIL."""
        if arr.shape == tuple(target_shape):
            return arr
        h, w = target_shape
        img = Image.fromarray(arr)
        return np.array(img.resize((w, h), resample=resample))

    def overlay_seg_vs_gt(self, gt_path: Path, seg_path: Path, thr=0.5):
        """Compare final segmentation vs GT; return RGB overlay and stats."""
        colors = {
            "tp": (255, 255, 255),  # white
            "tn": (0, 0, 0),        # black
            "fp": (0, 255, 0),      # green
            "fn": (255, 0, 0),      # red
        }

        gt = self.load_gray(gt_path)
        seg = self.load_gray(seg_path)

        if seg.shape != gt.shape:
            seg = self.resize_like(seg, gt.shape, resample=Image.NEAREST)

        gt_bin = self.prob_to_binary(gt, thr=0.5)
        seg_bin = self.prob_to_binary(seg, thr=thr)

        h, w = gt_bin.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        tp = (gt_bin == 1) & (seg_bin == 1)
        tn = (gt_bin == 0) & (seg_bin == 0)
        fp = (gt_bin == 0) & (seg_bin == 1)
        fn = (gt_bin == 1) & (seg_bin == 0)

        overlay[tp] = colors["tp"]
        overlay[tn] = colors["tn"]
        overlay[fp] = colors["fp"]
        overlay[fn] = colors["fn"]

        stats = {
            "TP": int(tp.sum()),
            "TN": int(tn.sum()),
            "FP": int(fp.sum()),
            "FN": int(fn.sum()),
        }

        return Image.fromarray(overlay, mode="RGB"), stats

    def overlay_diff_two_models(self, gt_path: Path, prob_a_path: Path, prob_b_path: Path, thr=0.5):
        """
        Compare baseline model A vs new model B against GT.
        Colors follow VGN-style diff map.
        """
        colors = {
            "FN_to_TP": (255, 0, 0),  # red
            "FP_to_TN": (0, 255, 0),  # green
            "TP_to_FN": (0, 0, 255),  # blue
            "TN_to_FP": (255, 255, 0),  # yellow
        }

        gt = self.load_gray(gt_path)
        prob_a = self.load_gray(prob_a_path)
        prob_b = self.load_gray(prob_b_path)

        if prob_a.shape != gt.shape:
            prob_a = self.resize_like(prob_a, gt.shape, resample=Image.BILINEAR)
        if prob_b.shape != gt.shape:
            prob_b = self.resize_like(prob_b, gt.shape, resample=Image.BILINEAR)

        gt_bin = self.prob_to_binary(gt, thr=0.5)
        a_bin = self.prob_to_binary(prob_a, thr=thr)
        b_bin = self.prob_to_binary(prob_b, thr=thr)

        h, w = gt_bin.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        fn_to_tp = (gt_bin == 1) & (a_bin == 0) & (b_bin == 1)
        fp_to_tn = (gt_bin == 0) & (a_bin == 1) & (b_bin == 0)
        tp_to_fn = (gt_bin == 1) & (a_bin == 1) & (b_bin == 0)
        tn_to_fp = (gt_bin == 0) & (a_bin == 0) & (b_bin == 1)

        overlay[fn_to_tp] = colors["FN_to_TP"]
        overlay[fp_to_tn] = colors["FP_to_TN"]
        overlay[tp_to_fn] = colors["TP_to_FN"]
        overlay[tn_to_fp] = colors["TN_to_FP"]

        stats = {
            "FN->TP": int(fn_to_tp.sum()),
            "FP->TN": int(fp_to_tn.sum()),
            "TP->FN": int(tp_to_fn.sum()),
            "TN->FP": int(tn_to_fp.sum()),
        }

        return Image.fromarray(overlay, mode="RGB"), stats

    def on_compare(self):
        """Generate overlay (final segmentation vs GT) and show it."""
        if not self.cases:
            return

        case = self.cases[self.case_index]
        gt_path = case.get("gt")
        seg_path = case.get("seg")

        missing = [label for label, path in [("GT", gt_path), ("Seg", seg_path)] if path is None]
        if missing:
            messagebox.showwarning(
                "Compare unavailable",
                f"Missing required images: {', '.join(missing)}",
            )
            return

        try:
            overlay_img, stats = self.overlay_seg_vs_gt(gt_path, seg_path)
        except Exception as e:
            messagebox.showerror("Compare failed", f"Could not build overlay:\n{e}")
            return

        # Show legend and overlay
        self.set_legend_visible(True)
        self.render_image(overlay_img)

        print("\nPixel counts (Segmentation vs GT):")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print("\nColor legend:")
        print("  white  = TP (correct vessel)")
        print("  black  = TN (correct background)")
        print("  green  = FP (false positive vessel)")
        print("  red    = FN (missed vessel)")

    def on_stage_button(self):
        self.update_stage_label()
        self.ensure_valid_stage(self.cases[self.case_index])
        self.reset_pan()
        self.set_legend_visible(False)
        self.show_current_image()

    def adjust_zoom(self, multiplier):
        new_zoom = self.zoom_factor * multiplier
        new_zoom = max(0.25, min(new_zoom, 5.0))
        self.zoom_factor = new_zoom
        self.zoom_label_var.set(f"{int(self.zoom_factor * 100)}%")
        self.ensure_valid_stage(self.cases[self.case_index]) if self.cases else None
        self.show_current_image()

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.zoom_label_var.set("100%")
        self.reset_pan()
        self.show_current_image()

    # ---------- Panning ----------

    def on_pan_start(self, event):
        if not self.cases or self.image_cache is None:
            return
        self.pan_start = (event.x, event.y)

    def on_pan_move(self, event):
        if self.pan_start is None:
            return
        dx = event.x - self.pan_start[0]
        dy = event.y - self.pan_start[1]
        self.pan_offset[0] += dx
        self.pan_offset[1] += dy
        self.pan_start = (event.x, event.y)
        self.update_canvas_image_position()

    def on_pan_end(self, _event=None):
        self.pan_start = None

    def reset_pan(self):
        self.pan_offset = [0, 0]
        self.pan_start = None
        self.update_canvas_image_position()

    # ---------- Image display ----------

    def clear_canvas(self):
        self.canvas.delete("all")
        self.canvas_image_id = None

    def show_current_image(self):
        if not self.cases:
            self.clear_canvas()
            return

        idx = self.stage_var.get()
        idx = max(0, min(idx, len(self.stage_keys) - 1))
        key = self.stage_keys[idx]

        case = self.cases[self.case_index]
        if not self.ensure_valid_stage(case):
            self.clear_canvas()
            return
        key = self.stage_keys[self.stage_var.get()]
        img_path = case[key]
        if img_path is None:
            self.clear_canvas()
            return

        try:
            img = Image.open(img_path)
        except Exception as e:
            messagebox.showerror(
                "Image error", f"Failed to open image:\n{img_path}\n\n{e}"
            )
            return

        self.render_image(img)

    def render_image(self, img: Image.Image):
        # Fit to canvas while keeping aspect ratio, then apply zoom/pan
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            self.after(100, lambda: self.render_image(img))
            return

        img_ratio = img.width / img.height
        canvas_ratio = cw / ch

        if img_ratio > canvas_ratio:
            base_w = cw
            base_h = int(cw / img_ratio)
        else:
            base_h = ch
            base_w = int(ch * img_ratio)

        new_w = max(1, int(base_w * self.zoom_factor))
        new_h = max(1, int(base_h * self.zoom_factor))

        img_resized = img.resize((new_w, new_h), Image.LANCZOS)

        self.image_cache = ImageTk.PhotoImage(img_resized)
        self.clear_canvas()
        self.canvas_image_id = self.canvas.create_image(
            cw // 2 + self.pan_offset[0],
            ch // 2 + self.pan_offset[1],
            image=self.image_cache,
            anchor="center",
        )

    def update_canvas_image_position(self):
        if self.canvas_image_id is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.canvas.coords(
            self.canvas_image_id,
            cw // 2 + self.pan_offset[0],
            ch // 2 + self.pan_offset[1],
        )

if __name__ == "__main__":
    app = RetinalDemo()
    app.mainloop()
