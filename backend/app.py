import io
import os
import base64
import math
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from torchvision import transforms
from .models_s40 import build_s40_model, s40_classes, s40_image_transform

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "samples")
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=frontend_dir, html=True), name="static")
app.mount("/samples", StaticFiles(directory=samples_dir, html=False), name="samples")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

s40_model_path = os.path.join(models_dir, "s40_cnn_lstm.pth")
use_s40_default = os.path.exists(s40_model_path)
s40_model = None
s40_labels = list(s40_classes)
s40_error = None
def _strip_module_prefix(state):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
def _remap_keys(state):
    remapped = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("cnn."):
            nk = "features." + nk[len("cnn."):]
        if nk.startswith("lstm."):
            nk = "head.lstm." + nk[len("lstm."):]
        if nk.startswith("fc."):
            nk = "head.fc." + nk[len("fc."):]
        if nk.startswith("classifier."):
            nk = "head.fc." + nk[len("classifier."):]
        remapped[nk] = v
    return remapped
def _infer_num_classes(state):
    try:
        v = state.get("head.fc.weight")
        if v is not None and hasattr(v, "ndim") and v.ndim == 2:
            return int(v.shape[0])
    except Exception:
        pass
    for k, v in state.items():
        try:
            if k.endswith(".weight") and hasattr(v, "ndim") and v.ndim == 2:
                return int(v.shape[0])
        except Exception:
            continue
    return len(s40_classes)
def init_s40():
    global s40_model, s40_labels, s40_error
    s40_error = None
    s40_model = None
    if not use_s40_default:
        return
    try:
        loaded = torch.load(s40_model_path, map_location="cpu")
        labels = None
        state = loaded
        if isinstance(loaded, dict) and "state_dict" in loaded:
            state = loaded["state_dict"]
            for k in ("classes", "labels", "class_names", "idx_to_class", "names"):
                v = loaded.get(k)
                if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
                    labels = list(v)
                    break
        state = _strip_module_prefix(state)
        state = _remap_keys(state)
        num_classes = _infer_num_classes(state)
        s40_model = build_s40_model(num_classes)
        s40_model.load_state_dict(state, strict=False)
        s40_model.eval()
        file_labels_path = os.path.join(models_dir, "s40_labels.txt")
        file_labels = None
        if os.path.exists(file_labels_path):
            try:
                with open(file_labels_path, "r", encoding="utf-8") as f:
                    file_labels = [ln.strip() for ln in f.readlines() if ln.strip()]
            except Exception:
                file_labels = None
        if isinstance(labels, list) and len(labels) == num_classes:
            s40_labels = labels
        elif isinstance(file_labels, list) and len(file_labels) == num_classes:
            s40_labels = file_labels
        elif num_classes == len(s40_classes):
            s40_labels = list(s40_classes)
        else:
            s40_labels = [f"class_{i}" for i in range(num_classes)]
    except Exception as e:
        s40_error = str(e)
        s40_model = None

init_s40()

def read_image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")



def to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def annotate_image(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    pad = max(8, w // 100)
    font_size = max(16, w // 24)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    rect_w = tw + pad * 2
    rect_h = th + pad
    x = pad
    y = pad
    draw.rectangle([x - pad // 2, y - pad // 2, x + rect_w, y + rect_h], fill=(0, 0, 0, 180))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return img



def predict_action_image_s40(img: Image.Image) -> (str, float):
    if s40_model is None:
        frames = [img] * 16
        return predict_action_from_frames(frames)
    seq_len = 8
    frames = [s40_image_transform(img) for _ in range(seq_len)]
    x = torch.stack(frames, dim=0).unsqueeze(0)
    with torch.no_grad():
        y = s40_model(x)
        p = torch.softmax(y, dim=-1).squeeze(0)
        conf, idx = torch.max(p, dim=-1)
        return s40_labels[int(idx)], float(conf.item())

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/categories")
def get_categories():
    return {"categories": s40_labels}

@app.get("/api/samples")
def list_samples():
    files = []
    for name in os.listdir(samples_dir):
        p = os.path.join(samples_dir, name)
        if os.path.isfile(p) and name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            files.append(name)
    return {"files": files}

@app.get("/api/model")
def current_model():
    return {"s40_available": s40_model is not None, "default_s40": use_s40_default, "error": s40_error}

@app.api_route("/api/reload_s40", methods=["GET", "POST"])
def reload_s40_route():
    init_s40()
    return {"s40_available": s40_model is not None, "error": s40_error}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    ct = (file.content_type or "").lower()
    is_image = ct.startswith("image/") or file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    if not is_image:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    img = read_image_from_bytes(data)
    label, conf = predict_action_image_s40(img)
    annotated = annotate_image(img.copy(), f"{label} ({conf:.2f})")
    return JSONResponse({"label": label, "confidence": conf, "image_base64": to_base64(annotated)})

