# src/utils.py
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

def largest_circle_crop(rgb: np.ndarray) -> np.ndarray:
    try:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=100, param2=30, minRadius=30, maxRadius=0
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, r = max(circles[0, :], key=lambda c: c[2])
            m = int(r * 1.2)
            h, w = rgb.shape[:2]
            x1, y1 = max(x - m, 0), max(y - m, 0)
            x2, y2 = min(x + m, w), min(y + m, h)
            crop = rgb[y1:y2, x1:x2]
            if crop.size > 0:
                return crop
    except Exception:
        pass
    return rgb

def center_square(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return rgb[y0:y0 + side, x0:x0 + side]

def enhance_digits(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def pil_to_np(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))
