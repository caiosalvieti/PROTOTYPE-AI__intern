# core/vision/features.py
from __future__ import annotations

from typing import Dict, Tuple, List
import numpy as np

import cv2

QA_BLUR_MIN = 15.0
QA_SPECULAR_MAX_FRAC = 0.20
QA_WB_SHIFT_MAX = 50.0
QA_DARK_GRAY = 20.0
QA_BRIGHT_GRAY = 240.0

def imread_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def imwrite_rgb(path: str, img_rgb: np.ndarray) -> None:
    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

def gray_world(img_rgb: np.ndarray) -> np.ndarray:
    img = img_rgb.astype(np.float32)
    mean = np.mean(img, axis=(0, 1)) + 1e-6
    scale = np.mean(mean) / mean
    img *= scale
    return np.clip(img, 0, 255).astype(np.uint8)

def qa_checks(face_rgb: np.ndarray):
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    mean_gray = float(np.mean(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    specular = float(((v > 240) & (s < 30)).sum()) / v.size
    r, g, b = [np.mean(c) for c in cv2.split(face_rgb)]
    wb_shift = float(max(abs(r - b), abs(g - b)))

    issues: List[str] = []
    if mean_gray < QA_DARK_GRAY: issues.append("too_dark")
    if mean_gray > QA_BRIGHT_GRAY: issues.append("too_bright")
    if blur < QA_BLUR_MIN: issues.append("blurry")
    if specular > QA_SPECULAR_MAX_FRAC: issues.append("glare_wet")
    if wb_shift > QA_WB_SHIFT_MAX: issues.append("colored_light")

    fail = ("too_dark" in issues) or ("too_bright" in issues)
    qa = dict(mean_gray=mean_gray, blur=blur, specular=specular, wb_shift=wb_shift)
    return issues, fail, qa

def crop_zones(face_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    h, w, _ = face_rgb.shape
    z = {}
    fw, fh = int(0.6 * w), int(0.25 * h); fx, fy = (w - fw) // 2, 0
    z["forehead"] = face_rgb[fy:fy + fh, fx:fx + fw]
    y1, y2 = int(0.30 * h), int(0.65 * h); xL2, xR1 = int(0.40 * w), int(0.60 * w)
    z["cheek_left"]  = face_rgb[y1:y2, 0:xL2]
    z["cheek_right"] = face_rgb[y1:y2, xR1:w]
    cy1, cy2 = int(0.65 * h), int(0.95 * h)
    z["chin"] = face_rgb[cy1:cy2, fx:fx + fw]
    return z

def redness_index(rgb: np.ndarray) -> float:
    r, g, b = cv2.split(rgb.astype(np.float32))
    s = r + g + b + 1e-6
    return float(np.mean(r / s))

def texture_index(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def shine_index(rgb: np.ndarray) -> float:
    gray = cv2.medianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 3)
    return float(np.mean(gray)) / 255.0

def extract_features(face_rgb: np.ndarray):
    issues, fail, qa = qa_checks(face_rgb)
    zones = crop_zones(face_rgb)
    feats: Dict[str, float | int | str] = {}

    for name, z in zones.items():
        feats[f"{name}_red"] = redness_index(z)
        feats[f"{name}_txt"] = texture_index(z)
        feats[f"{name}_shn"] = shine_index(z)

    feats.update({
        "global_red": redness_index(face_rgb),
        "global_txt": texture_index(face_rgb),
        "global_shn": shine_index(face_rgb),
        **{f"qa_{k}": float(v) for k, v in qa.items()},
        "qa_fail": int(fail),
        "qa_issues": "|".join(issues),
    })

    return feats, zones

def save_debug_panel(rgb: np.ndarray, face_box_xywh: Tuple[int,int,int,int], save_path: str):
    x, y, w, h = face_box_xywh
    dbg = rgb.copy()
    cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
    imwrite_rgb(save_path, dbg)
