#!/usr/bin/env python3
"""
tk_frontend_rebuilt.py

Rebuilt Local YOLO frontend (Image, Camera, History).
- Clean results area (label + confidence)
- "Show system process" toggle to reveal detailed JSON
- Click detection to highlight (fade background except that crop)
- Larger bbox text (tries to use 16pt truetype)
- Keeps camera and history functionality
"""

import os, time, datetime, threading, json, glob
from pathlib import Path
from collections import defaultdict

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageOps
import cv2
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODELS_DIR = Path("models")
UPLOAD_DIR = Path("uploads")
HISTORY_DIR = Path("history")

SUPPORTED_EXTS = ('.jpg','.jpeg','.png','.bmp','.tiff','.webp')

for d in (MODELS_DIR, UPLOAD_DIR, HISTORY_DIR):
    d.mkdir(exist_ok=True)

# default font size for bbox text
FONT_SIZE = 16

# Try to load a TTF font for better text sizing; fallback to default
try:
    FONT = ImageFont.truetype("arial.ttf", FONT_SIZE)
except Exception:
    try:
        FONT = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)
    except Exception:
        FONT = ImageFont.load_default()

# ---------------- Helpers ----------------

def list_models():
    return sorted([str(p) for p in MODELS_DIR.glob("*.pt")])

def load_model(path):
    try:
        m = YOLO(path)
        # store a stable display name and original path on the model instance
        display_name = Path(path).name
        # attach attributes to model object for our UI convenience
        setattr(m, "_model_path_str", str(path))
        setattr(m, "_display_name", display_name)
        return m
    except Exception as e:
        print("Failed to load model", path, e)
        return None

def save_local_copy(orig_path):
    dst = UPLOAD_DIR / f"{int(time.time())}_{Path(orig_path).name}"
    with open(orig_path,'rb') as fr, open(dst,'wb') as fw:
        fw.write(fr.read())
    return str(dst)

def infer(model, img_path):
    try:
        # try modern API
        try:
            res = model.predict(source=img_path, verbose=False)
            r = res[0]
        except Exception:
            # fallback older API
            res = model(img_path)
            if not res:
                return []
            r = res[0]

        # now work from r consistently
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return []

        # extract xyxy
        try:
            xy = boxes.xyxy.cpu().numpy()
        except:
            try:
                xy = boxes.xyxy.numpy()
            except:
                xy = boxes.xyxy.tolist()

        confs = getattr(boxes, "conf", None)
        clss  = getattr(boxes, "cls",  None)
        names = getattr(r, "names", {}) or {}

        out = []
        for i, vals in enumerate(xy):
            try:
                x1, y1, x2, y2 = map(int, vals[:4])
                conf = float(confs[i]) if confs is not None else float(vals[4])
                ci   = int(clss[i])  if clss  is not None else int(vals[5])
                label = names.get(ci, str(ci))
                out.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "label": label,
                    "confidence": conf
                })
            except:
                continue

        return out

    except Exception as e:
        return [{"error": str(e)}]

# Drawing with PIL Image and chosen FONT
def draw_boxes_on_image(pil_img, preds, font=FONT, box_color=(255,0,0), text_fill=(255,255,0), stroke_width=2):
    img = pil_img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    for p in preds:
        if not isinstance(p, dict) or "x1" not in p:
            continue
        x1,y1,x2,y2 = p["x1"], p["y1"], p["x2"], p["y2"]
        label = p.get("label","")
        conf = p.get("confidence", 0.0)
        text = f"{label} {conf:.2f}"
        # rectangle
        draw.rectangle([x1,y1,x2,y2], outline=box_color, width=stroke_width)
        # text background
        try:
            tw, th = draw.textsize(text, font=font)
        except Exception:
            tw, th = (len(text)*6, 12)
        draw.rectangle([x1, max(0,y1-th-6), x1+tw+6, y1], fill=box_color)
        draw.text((x1+3, max(0,y1-th-4)), text, font=font, fill=text_fill)
    return img.convert("RGB")

def dim_except_crop(pil_img, crop_box):
    """
    Return an image where everything is dimmed except the crop_box area (x1,y1,x2,y2).
    Works by creating alpha overlay.
    """
    x1,y1,x2,y2 = crop_box
    base = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0,0,0,150))  # semi-transparent black
    # paste the original crop back onto overlay area (i.e. make that area fully transparent)
    region = base.crop((x1,y1,x2,y2))
    overlay.paste(region, (x1,y1))
    # composite: show region (base) over overlay, but we want region visible and rest dark—swap
    # We'll create result by combining base with overlay using overlay alpha
    res = Image.alpha_composite(overlay, base.convert("RGBA"))
    return res.convert("RGB")

# ---------------- App ----------------

class App:
    def __init__(self, root):
        self.root = root
        root.title("Local YOLO — rebuilt")
        root.geometry("1200x800")

        self.models = []         # YOLO instances
        self.model_paths = []    # original model path strings
        self.model_names = []    # friendly names

        # state for current image preview
        self.current_image_path = None
        self.current_pil_image = None    # original PIL image (RGB)
        self.current_preds = {}          # dict: model_name -> [preds]
        self.highlight = None            # (model_name, pred_idx) or None

        # Notebook
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True)

        self.tab_image = ttk.Frame(self.nb)
        self.tab_cam = ttk.Frame(self.nb)
        self.tab_hist = ttk.Frame(self.nb)

        self.nb.add(self.tab_image, text="Image")
        self.nb.add(self.tab_cam, text="Camera")
        self.nb.add(self.tab_hist, text="History")

        # build UI
        self.build_image_tab()
        self.build_camera_tab()
        self.build_history_tab()

        # background model load
        threading.Thread(target=self.reload_models, daemon=True).start()

    # ---------------- Image Tab ----------------
    def build_image_tab(self):
        fr = self.tab_image
        left = ttk.Frame(fr, width=300)
        left.pack(side="left", fill="y", padx=6, pady=6)
        right = ttk.Frame(fr)
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        # Left controls
        ttk.Label(left, text="Models").pack(anchor="w")
        self.lb_models = tk.Listbox(left, selectmode="extended", height=8)
        self.lb_models.pack(fill="x", pady=4)

        ttk.Button(left, text="Reload models", command=lambda: threading.Thread(target=self.reload_models, daemon=True).start()).pack(fill="x", pady=6)
        ttk.Button(left, text="Select Image", command=self.select_image).pack(fill="x", pady=6)

        self.chk_gps = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Include GPS (IP) — disabled here", variable=self.chk_gps, state="disabled").pack(anchor="w", pady=(8,0))

        # Results area (clean)
        ttk.Label(left, text="Results").pack(anchor="w", pady=(12,0))
        self.results_tree = ttk.Treeview(left, columns=("conf",), show="tree headings", height=10)
        self.results_tree.heading("#0", text="Item")
        self.results_tree.heading("conf", text="Confidence")
        self.results_tree.column("conf", width=80, anchor="center")
        self.results_tree.pack(fill="both", expand=False, pady=4)
        self.results_tree.bind("<<TreeviewSelect>>", self.on_result_select)

        # Reset highlight button
        ttk.Button(left, text="Reset Highlight", command=self.reset_highlight).pack(fill="x", pady=6)

        # Show system process (collapsible)
        self.show_sys_var = tk.BooleanVar(value=False)
        chk = ttk.Checkbutton(left, text="Show system process", variable=self.show_sys_var, command=self.toggle_sys_area)
        chk.pack(anchor="w", pady=(8,2))
        self.sys_frame = ttk.Frame(left)
        self.sys_text = scrolledtext.ScrolledText(self.sys_frame, height=10)
        self.sys_text.pack(fill="both", expand=True)
        # initially hidden

        # Right: preview canvas + detailed JSON below
        self.preview_canvas = tk.Canvas(right, bg="#222")
        self.preview_canvas.pack(fill="both", expand=True)
        self.preview_canvas.bind("<Configure>", lambda e: self.redraw_preview())

        # store PhotoImage reference
        self._preview_tkimg = None

    def toggle_sys_area(self):
        if self.show_sys_var.get():
            self.sys_frame.pack(fill="both", pady=(6,0))
        else:
            self.sys_frame.pack_forget()

    def select_image(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.webp")])
        if p:
            threading.Thread(target=self.process_image, args=(p,), daemon=True).start()

    def process_image(self, path):
        if Path(path).suffix.lower() not in SUPPORTED_EXTS:
            self.root.after(0, lambda: messagebox.showerror("Unsupported", "Not a supported image type"))
            return

        saved = save_local_copy(path)
        self.current_image_path = saved

        # choose models selected or all
        sel = list(self.lb_models.curselection())
        use = [self.models[i] for i in sel] if sel else list(self.models)
        use_names = [self.model_names[i] for i in sel] if sel else list(self.model_names)

        if not use:
            self.root.after(0, lambda: messagebox.showwarning("No models", "No models loaded"))
            return

        # update UI
        self.root.after(0, lambda: self._append_sys(f"Processing {saved} with {len(use)} model(s)...\n"))

        per = {}
        # run models sequentially (safe)
        for m, name in zip(use, use_names):
            try:
                preds = infer(m, saved)
                # filter errors out of displayed preds
                good = [p for p in preds if isinstance(p, dict) and "error" not in p]
                per[name] = good
                self.root.after(0, lambda n=name, p=good: self._append_sys(f"[{n}] {len(p)} predictions\n"))
            except Exception as e:
                per[name] = [{"error": str(e)}]
                self.root.after(0, lambda n=name, e=e: self._append_sys(f"[{n}] ERROR: {e}\n"))

        # save history JSON
        js = saved + ".json"
        try:
            with open(js, "w") as f:
                json.dump({"file": saved, "results": per}, f, indent=2)
        except Exception as e:
            self.root.after(0, lambda: self._append_sys(f"Failed to save JSON: {e}\n"))

        # load original PIL image once for preview/drawing
        try:
            pil = Image.open(saved).convert("RGB")
            self.current_pil_image = pil
        except Exception as e:
            self.current_pil_image = None
            self.root.after(0, lambda: self._append_sys(f"Failed to open image: {e}\n"))

        # store predictions
        self.current_preds = per
        self.highlight = None

        # update UI (tree + preview)
        self.root.after(0, self._update_results_tree)
        self.root.after(0, self.redraw_preview)

    def _append_sys(self, text):
        if self.show_sys_var.get():
            self.sys_text.insert("end", text)
            self.sys_text.see("end")

    def _update_results_tree(self):
        self.results_tree.delete(*self.results_tree.get_children())

        flat = []
        for model_name, preds in self.current_preds.items():
            for i, p in enumerate(preds):
                if "error" not in p:
                    flat.append((model_name, i, p))

        if not flat:
            self.results_tree.insert("", "end", text="(no objects)", values=("",))
            return
    
        for (mname, idx, obj) in flat:
            label = obj.get("label", "object")
            conf = obj.get("confidence", 0.0)
    
            node = self.results_tree.insert(
                "",
                "end",
                text=label,
                values=(f"{conf:.2f}",)
            )

            self.results_tree.set(
                node,
                "_meta",
                json.dumps({"model": mname, "index": idx})
            )

    def on_result_select(self, event):
        sel = self.results_tree.selection()
        if not sel:
            return
        node = sel[0]
        # ignore top-level model nodes which have children
        parent = self.results_tree.parent(node)
        if parent == "":
            # top-level model node selected; do nothing
            return
        # locate model name (parent text) and index by mapping selection order
        model_name = self.results_tree.item(parent, "text")
        # find index by counting items under parent up to this node
        children = list(self.results_tree.get_children(parent))
        try:
            idx = children.index(node)
        except ValueError:
            idx = None
        if idx is None:
            return
        # set highlight and redraw preview
        self.highlight = (model_name, idx)
        self.redraw_preview()

    def reset_highlight(self):
        self.highlight = None
        self.redraw_preview()

    def redraw_preview(self):
        """
        Redraws the preview canvas based on current image, preds and highlight.
        """
        if not self.current_pil_image:
            self.preview_canvas.delete("all")
            return

        # draw boxes on a copy for normal preview
        all_preds = []
        for preds in self.current_preds.values():
            for p in preds:
                if isinstance(p, dict) and "error" not in p:
                    all_preds.append(p)

        # if highlight is set, dim everything except highlighted region
        if self.highlight:
            model_name, idx = self.highlight
            preds_list = self.current_preds.get(model_name, [])
            if idx < 0 or idx >= len(preds_list):
                self.highlight = None
                base_img = draw_boxes_on_image(self.current_pil_image, all_preds)
            else:
                target = preds_list[idx]
                if "x1" not in target:
                    base_img = draw_boxes_on_image(self.current_pil_image, all_preds)
                else:
                    # create dimmed image except target crop
                    # first draw boxes on original for clarity, then dim except crop
                    boxed = draw_boxes_on_image(self.current_pil_image, all_preds)
                    x1,y1,x2,y2 = target["x1"], target["y1"], target["x2"], target["y2"]
                    base_img = dim_except_crop(boxed, (x1,y1,x2,y2))
        else:
            base_img = draw_boxes_on_image(self.current_pil_image, all_preds)

        # resize to canvas keeping aspect ratio and center
        cw = max(10, self.preview_canvas.winfo_width())
        ch = max(10, self.preview_canvas.winfo_height())
        im_w, im_h = base_img.size
        # compute thumbnail size preserving aspect ratio
        ratio = min(cw / im_w, ch / im_h)
        new_w = max(1, int(im_w * ratio))
        new_h = max(1, int(im_h * ratio))
        display_img = base_img.copy().resize((new_w, new_h), Image.Resampling.LANCZOS)

        # convert to PhotoImage
        tkimg = ImageTk.PhotoImage(display_img)
        self._preview_tkimg = tkimg
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(cw//2, ch//2, image=tkimg, anchor="center")
        # store for GC protection
        self.preview_canvas.img = tkimg

    # ---------------- Camera Tab ----------------
    def build_camera_tab(self):
        fr = self.tab_cam
        left = ttk.Frame(fr, width=260)
        left.pack(side="left", fill="y", padx=6, pady=6)
        right = ttk.Frame(fr)
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        ttk.Button(left, text="Start Camera", command=self.start_camera).pack(fill="x", pady=(4,8))
        ttk.Button(left, text="Stop Camera", command=self.stop_camera).pack(fill="x")

        ttk.Label(left, text="Active Models").pack(anchor="w", pady=(12,0))
        self.lb_cam = tk.Listbox(left, selectmode="extended", height=8)
        self.lb_cam.pack(fill="x", pady=4)

        self.cam_label = ttk.Label(right)
        self.cam_label.pack(fill="both", expand=True)

        self.cam_running = False
        self.cap = None

    def start_camera(self):
        if not self.models:
            messagebox.showwarning("Models", "No models loaded")
            return
        if self.cam_running:
            return
        # open default camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        self.cam_running = True
        # populate listbox
        self.lb_cam.delete(0, "end")
        for n in self.model_names:
            self.lb_cam.insert("end", n)
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def stop_camera(self):
        self.cam_running = False
        try:
            if self.cap:
                self.cap.release()
        except:
            pass
        self.cam_label.configure(image="")

    def _camera_loop(self):
        while self.cam_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.02)
                    continue
                # run selected models or all
                sel = list(self.lb_cam.curselection())
                use = [self.models[i] for i in sel] if sel else list(self.models)

                allp = []
                for m in use:
                    try:
                        r = m.predict(source=frame, verbose=False)[0]
                        if r.boxes is not None:
                            names = getattr(r, "names", {})
                            # attempt to convert boxes to array
                            try:
                                xy = r.boxes.xyxy.cpu().numpy()
                            except Exception:
                                try:
                                    xy = r.boxes.xyxy.numpy()
                                except Exception:
                                    xy = []
                            confs = getattr(r.boxes, "conf", None)
                            clss = getattr(r.boxes, "cls", None)
                            for i, vals in enumerate(xy):
                                try:
                                    x1,y1,x2,y2 = map(int, (vals[0], vals[1], vals[2], vals[3]))
                                    conf = float(confs[i]) if confs is not None else 0.0
                                    ci = int(clss[i]) if clss is not None else 0
                                    lbl = names.get(ci, str(ci))
                                    allp.append((x1,y1,x2,y2,lbl,conf))
                                except Exception:
                                    continue
                    except Exception:
                        continue

                # draw on frame
                for (x1,y1,x2,y2,l,c) in allp:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(frame, f"{l} {c:.2f}", (x1, max(20, y1-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

                # convert to RGB PIL and display
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(rgb)
                # adapt to label size
                w = max(10, self.cam_label.winfo_width())
                h = max(10, self.cam_label.winfo_height())
                im.thumbnail((w,h))
                tkimg = ImageTk.PhotoImage(im)
                self.root.after(0, lambda img=tkimg: self._update_cam_label(img))
                time.sleep(0.02)
            except Exception as e:
                print("Camera loop error:", e)
                time.sleep(0.1)
        try:
            if self.cap:
                self.cap.release()
        except:
            pass

    def _update_cam_label(self, tkimg):
        self.cam_label.configure(image=tkimg)
        self.cam_label.img = tkimg

    # ---------------- History Tab ----------------
    def build_history_tab(self):
        fr = self.tab_hist
        left = ttk.Frame(fr, width=260)
        left.pack(side="left", fill="y", padx=6, pady=6)
        right = ttk.Frame(fr)
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        ttk.Button(left, text="Refresh History", command=self.refresh_history).pack(fill="x", pady=4)
        self.lb_history = tk.Listbox(left, height=20)
        self.lb_history.pack(fill="both", expand=True, pady=6)
        self.lb_history.bind("<<ListboxSelect>>", self.on_history_select)

        self.history_canvas = tk.Canvas(right, bg="#333")
        self.history_canvas.pack(fill="both", expand=True)

        self.refresh_history()

    def refresh_history(self):
        self.lb_history.delete(0, "end")
        files = sorted(glob.glob(str(UPLOAD_DIR / "*.jpg")))
        for f in files:
            self.lb_history.insert("end", f)

    def on_history_select(self, event):
        sel = self.lb_history.curselection()
        if not sel:
            return
        path = self.lb_history.get(sel[0])
        try:
            img = Image.open(path).convert("RGB")
            w = max(10, self.history_canvas.winfo_width())
            h = max(10, self.history_canvas.winfo_height())
            img.thumbnail((w,h))
            tkimg = ImageTk.PhotoImage(img)
            self.history_canvas.delete("all")
            self.history_canvas.create_image(w//2, h//2, image=tkimg, anchor="center")
            self.history_canvas.img = tkimg
        except Exception as e:
            messagebox.showerror("History", f"Failed to open image: {e}")

    # ---------------- Model loading ----------------
    def reload_models(self):
        self.models.clear()
        self.model_paths.clear()
        self.model_names.clear()
        self.lb_models.delete(0, "end")
        self.lb_cam.delete(0, "end")

        for p in list_models():
            m = load_model(p)
            if m:
                self.models.append(m)
                self.model_paths.append(str(p))
                name = Path(p).name
                self.model_names.append(name)
                self.lb_models.insert("end", name)
                self.lb_cam.insert("end", name)

# ---------------- Main ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
