import os
from PIL import Image
import numpy as np
from .app import predict_action_from_frames, annotate_image

def make_image(w=640, h=360):
    rng = np.random.default_rng(42)
    arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")

def run():
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "samples"), exist_ok=True)
    img = make_image()
    frames = [img] * 16
    label, conf = predict_action_from_frames(frames)
    annotated = annotate_image(img.copy(), f"{label} ({conf:.2f})")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "samples", "sample_random.jpg")
    annotated.save(out_path, format="JPEG")
    return out_path

if __name__ == "__main__":
    p = run()
    print(p)
