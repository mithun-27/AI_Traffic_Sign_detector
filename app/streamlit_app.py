# app/streamlit_app.py
import streamlit as st
from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import torch

from src.utils import pil_to_np, largest_circle_crop, center_square, enhance_digits

st.set_page_config(page_title="Traffic Sign — Detect & Classify", layout="wide")
st.title("Traffic Sign Detection (YOLOv5) + Classifier")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_WEIGHTS = MODELS_DIR / "yolov5_best.pt"

# load YOLO (cached)
@st.cache_resource
def load_yolo(weights_path: Path):
    if not weights_path.exists():
        return None
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(weights_path), trust_repo=True)
    model.conf = 0.25
    model.iou = 0.45
    model.max_det = 300
    return model

# load classifier (optional)
@st.cache_resource
def load_classifier():
    candidates = [MODELS_DIR / "tsc_mobilenetv2.keras", MODELS_DIR / "tsc_mobilenetv2.h5"]
    for c in candidates:
        if c.exists():
            try:
                if c.suffix == ".keras":
                    model = tf.keras.models.load_model(c)
                else:
                    model = tf.keras.models.load_model(c, compile=False)
                classes_file = c.with_suffix(".classes.json")
                if classes_file.exists():
                    class_names = json.loads(classes_file.read_text(encoding="utf-8"))
                else:
                    class_names = [f"Class {i}" for i in range(model.output_shape[-1])]
                return model, class_names, c
            except Exception as e:
                st.warning(f"Failed to load classifier {c.name}: {e}")
    return None, [], None

yolo = load_yolo(YOLO_WEIGHTS)
clf_model, class_names, clf_path = load_classifier()
if clf_model is not None:
    clf_size = int(clf_model.input_shape[1])
else:
    clf_size = 96

# sidebar & mode
st.sidebar.title("Settings")
mode = st.sidebar.radio("Mode", ["Detector (YOLOv5)", "Classifier (Crop -> Classify)", "Detect+Classify (two-stage)"])
st.sidebar.write("YOLO weights:", YOLO_WEIGHTS.name if YOLO_WEIGHTS.exists() else "Not found")
st.sidebar.write("Classifier:", clf_path.name if clf_path else "None")

col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Upload image (jpg/png) or drag-drop", type=["jpg","jpeg","png"])
    camera = st.checkbox("Use webcam (Detector only)")
with col2:
    st.markdown("#### Info")
    st.write("Detector mode: draw boxes with YOLOv5.")
    st.write("Classifier mode: crops center circle and classifies with MobileNetV2.")
    st.write("Detect+Classify: runs detector, crops detections and re-classifies using classifier (best of both).")

def display_image(img_np, caption="Image"):
    st.image(Image.fromarray(img_np), caption=caption, use_column_width=True)

# helper to classify a crop using classifier with TTA
def classify_crop(img_pil: Image.Image, model, img_size: int, tta_angles=(0, -6, 6)):
    probs_list = []
    for a in tta_angles:
        im = img_pil.rotate(a, resample=Image.BICUBIC)
        arr = pil_to_np(im)
        arr = largest_circle_crop(arr)
        arr = center_square(arr)
        arr = enhance_digits(arr)
        arr = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_AREA).astype("float32")
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        p = model.predict(np.expand_dims(arr, 0))[0]
        probs_list.append(p)
    return np.mean(probs_list, axis=0)

# Process uploaded or webcam frame
if camera and mode == "Detector (YOLOv5)":
    st.info("To use webcam detection run the local detect script (see README). Streamlit webcam in Windows is less reliable.")
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    img_np = pil_to_np(image)
    display_image(img_np, "Uploaded image")

    if mode == "Detector (YOLOv5)":
        if yolo is None:
            st.error("YOLO weights not found. Put weights (.pt) into models/ and reload.")
        else:
            # run inference
            results = yolo(img_np)
            rendered = np.squeeze(results.render())  # BGR -> annotated image
            rendered = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
            display_image(rendered, "YOLO Detections")
            # show table
            try:
                df = results.pandas().xyxy[0]
                if not df.empty:
                    st.dataframe(df[["name","confidence","xmin","ymin","xmax","ymax"]])
            except Exception:
                pass

    elif mode == "Classifier (Crop -> Classify)":
        if clf_model is None:
            st.error("No classifier found. Place model .keras/.h5 + .classes.json into models/")
        else:
            # crop + preprocess + classify
            # try largest circle crop, then center square
            crop = largest_circle_crop(img_np)
            crop = center_square(crop)
            crop_pil = Image.fromarray(crop)
            probs = classify_crop(crop_pil, clf_model, clf_size)
            idx = int(np.argmax(probs))
            st.subheader(f"Predicted: **{class_names[idx]}**")
            st.write(f"Confidence: {float(probs[idx]):.4f}")
            display_image(crop, "Cropped sign for classification")
            with st.expander("Top probabilities"):
                order = np.argsort(probs)[::-1][:10]
                for i in order:
                    st.write(f"{i:02d}. {class_names[i]} — {float(probs[i]):.4f}")

    else:  # Detect + Classify two-stage
        if yolo is None:
            st.error("YOLO weights not found.")
        else:
            results = yolo(img_np)
            rendered = np.squeeze(results.render())
            rendered = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
            display_image(rendered, "YOLO Detections")

            # For each detection, crop and optionally reclassify with classifier
            try:
                df = results.pandas().xyxy[0]
                if df.empty:
                    st.info("No detections.")
                else:
                    table_rows = []
                    annotated = rendered.copy()
                    for _, row in df.iterrows():
                        name = row['name']
                        conf = float(row['confidence'])
                        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                        crop = img_np[y1:y2, x1:x2]
                        crop_display = crop.copy()
                        clf_pred = None
                        if clf_model is not None and crop.size > 0:
                            crop_pil = Image.fromarray(crop)
                            probs = classify_crop(crop_pil, clf_model, clf_size)
                            idx = int(np.argmax(probs))
                            clf_pred = class_names[idx]
                            # overlay classifier label
                            label = f"{name} / {clf_pred} ({conf:.2f})"
                        else:
                            label = f"{name} ({conf:.2f})"
                        # draw label on annotated image
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(annotated, label, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        table_rows.append({"box_label": label, "confidence": conf})
                    display_image(annotated, "Detections + Classifier overlay")
                    st.dataframe(table_rows)
            except Exception as e:
                st.error(f"Error processing detections: {e}")

else:
    st.info("Upload an image to run detection or classification. For webcam use: run src/detect.py with --source 0 (webcam).")
# app/streamlit_app.py