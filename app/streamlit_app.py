import streamlit as st
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# ---------------------------
# Page + Title
# ---------------------------
st.set_page_config(page_title="AI Traffic Sign Classifier", page_icon="🚦", layout="centered")
st.title("🚦 AI Traffic Sign Classifier")
st.write("Upload a traffic sign image and the model will predict its class (with accuracy booster for 30 vs 60).")

# ---------------------------
# Model discovery + loading
# ---------------------------
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

def find_model_file():
    preferred = MODELS_DIR / "tsc_mobilenetv2.keras"
    if preferred.exists():
        return preferred
    keras_files = sorted(MODELS_DIR.glob("*.keras"))
    if keras_files:
        return keras_files[0]
    h5_files = sorted(MODELS_DIR.glob("*.h5"))
    if h5_files:
        return h5_files[0]
    return None

def load_any_model(weights_path: Path):
    if weights_path.suffix == ".keras":
        return tf.keras.models.load_model(weights_path)
    return tf.keras.models.load_model(weights_path, compile=False)

@st.cache_resource
def load_model_and_classes():
    weights_path = find_model_file()
    if weights_path is None:
        return None, [], None
    model = load_any_model(weights_path)
    classes_path = weights_path.with_suffix(".classes.json")
    if classes_path.exists():
        class_names = json.loads(classes_path.read_text(encoding="utf-8"))
    else:
        class_names = [f"Class {i}" for i in range(model.output_shape[-1])]
    return model, class_names, weights_path

model, class_names, weights_path = load_model_and_classes()
if model is None:
    st.warning("No model found in the 'models' folder. Place a .keras/.h5 model there and refresh.")
    st.stop()

REQUIRED_SIZE = int(model.input_shape[1])

with st.sidebar:
    st.markdown("**Model info**")
    st.write(f"Path: `{weights_path.name}`")
    st.write(f"Input size: **{REQUIRED_SIZE}×{REQUIRED_SIZE}**")
    st.caption("The app automatically uses the model's required input size.")

# ---------------------------
# Image helpers
# ---------------------------
def largest_circle_crop(rgb: np.ndarray) -> np.ndarray:
    """Crop around the largest detected circle (traffic sign). Falls back silently."""
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
    """CLAHE on L channel to make digits pop."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def preprocess_for_model(img: Image.Image, img_size: int) -> np.ndarray:
    """Returns (1, img_size, img_size, 3) for MobileNetV2."""
    rgb = np.array(img.convert("RGB"))
    rgb = largest_circle_crop(rgb)
    rgb = center_square(rgb)
    rgb = enhance_digits(rgb)
    img_resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    arr = img_resized.astype(np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# ---------------------------
# Accuracy booster (no retrain)
# ---------------------------
def tta_predict(model, img: Image.Image, img_size: int, angles=(0, -6, 6, -10, 10)):
    """Test-Time Augmentation: average predictions over small rotations."""
    def rotate_pil(im, deg):
        return im.rotate(deg, resample=Image.BICUBIC, expand=False)
    probs_list = []
    for a in angles:
        x = preprocess_for_model(rotate_pil(img, a), img_size)
        p = model.predict(x, verbose=0)[0]
        probs_list.append(p)
    return np.mean(probs_list, axis=0)

def count_holes(binary_img: np.ndarray) -> int:
    """
    Count 'holes' (white enclosed regions) in a binary digit.
    Assumes digits are dark on light after inversion below.
    """
    # Clean noise
    binary_img = cv2.medianBlur(binary_img, 3)
    # Connected components on the inverted image (holes become components)
    inv = cv2.bitwise_not(binary_img)
    num_labels, labels = cv2.connectedComponents(inv)
    # subtract 1 for background
    return max(num_labels - 1, 0)

def left_digit_holes(rgb_sign_crop: np.ndarray) -> int:
    """
    Roughly splits the sign number area into left/right and counts holes on the left digit.
    Works well for 30 vs 60.
    """
    h, w = rgb_sign_crop.shape[:2]
    roi = rgb_sign_crop[int(h*0.25):int(h*0.75), int(w*0.20):int(w*0.80)]  # central band where digits live
    if roi.size == 0:
        return 0
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    # increase contrast a bit
    gray = cv2.equalizeHist(gray)
    # adaptive threshold to segment digits
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)
    # split left/right roughly equally
    h2, w2 = th.shape
    left = th[:, :w2//2]
    # count enclosed holes on left digit (3 -> 0 holes, 6 -> 1 hole)
    return count_holes(left)

def resolve_30_vs_60(raw_image: Image.Image, predicted_idx: int, probs: np.ndarray, class_names: list) -> int:
    """
    If the model is between 'Speed limit (30km/h)' and 'Speed limit (60km/h)',
    use a digit-hole heuristic to switch to the correct label.
    """
    # Find class indices
    try:
        idx30 = class_names.index("Speed limit (30km/h)")
        idx60 = class_names.index("Speed limit (60km/h)")
    except ValueError:
        return predicted_idx  # classes not found; skip

    # Only act if one of the top2 is 30/60 and their probs are close
    top2 = np.argsort(probs)[-2:][::-1]
    if not set(top2).issubset({idx30, idx60}):
        return predicted_idx
    if abs(probs[idx30] - probs[idx60]) < 0.30:  # only intervene when it's close
        # Make a clean crop for heuristic (no resize to keep detail)
        rgb = np.array(raw_image.convert("RGB"))
        rgb = largest_circle_crop(rgb)
        rgb = center_square(rgb)
        # Count holes on LEFT digit region
        holes_left = left_digit_holes(rgb)
        # Rule: 30 => left digit '3' => 0 holes, 60 => left digit '6' => 1 hole
        if holes_left >= 1:
            return idx60
        else:
            return idx30
    return predicted_idx

# ---------------------------
# UI: upload + predict
# ---------------------------
uploaded = st.file_uploader("Choose an image (.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_column_width=True)

    # 1) TTA average
    probs = tta_predict(model, image, REQUIRED_SIZE)

    # 2) Heuristic override for 30 vs 60
    pred_idx = int(np.argmax(probs))
    pred_idx = resolve_30_vs_60(image, pred_idx, probs, class_names)

    # 3) Show result
    st.subheader(f"Prediction: **{class_names[pred_idx]}**")
    st.write(f"Confidence: **{float(probs[pred_idx]):.4f}**")

    with st.expander("Show all class probabilities"):
        order = np.argsort(probs)[::-1]
        for i in order[:10]:
            st.write(f"{i:02d}. {class_names[i]}: {float(probs[i]):.4f}")
