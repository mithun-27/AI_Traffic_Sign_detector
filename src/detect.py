# src/detect.py
import argparse
from pathlib import Path
import torch
import cv2
import numpy as np

def load_yolo(weights: Path):
    # loads Ultralytics YOLOv5 via torch.hub; will download the repo on first run
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(weights), trust_repo=True)
    model.conf = 0.25
    model.iou = 0.45
    model.max_det = 300
    return model

def run(weights: Path, source):
    model = load_yolo(weights)
    # `source` can be integer (webcam) or filename
    results = model(source)  # returns a Results object
    # save annotated image/video to runs/detect/exp
    save_dir = results.save()  # returns path
    print("Saved results to:", save_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to YOLOv5 .pt weights")
    ap.add_argument("--source", required=True, help="Image file path or webcam index (0)")
    args = ap.parse_args()
    source = int(args.source) if args.source.isnumeric() else args.source
    run(Path(args.weights), source)

if __name__ == "__main__":
    main()
