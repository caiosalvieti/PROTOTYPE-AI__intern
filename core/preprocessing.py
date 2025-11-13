# preprocessing.py
import cv2
import numpy as np

TARGET_SIZE = (640, 640)  # keep this fixed everywhere


def illumination_correction(img_bgr: np.ndarray) -> np.ndarray:
    """
    LAB + CLAHE on L channel: mild, deterministic illumination correction.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    corrected = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return corrected


def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Main deterministic preprocessing: illumination correction + fixed resize.
    """
    img = illumination_correction(img_bgr)
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return img
