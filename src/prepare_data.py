import argparse
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from uuid import uuid4

# Saves HF dataset "tanganke/gtsrb" to folder-per-class image structure.
# Creates a stratified 90/10 split into data/train and data/valid.

def save_split(examples, labels, out_root: Path, split_name: str):
    for img, y in zip(examples, labels):
        cls_dir = out_root / split_name / f"{y:05d}"
        cls_dir.mkdir(parents=True, exist_ok=True)

        # Ensure we have a PIL.Image in RGB
        if isinstance(img, Image.Image):
            im = img.convert("RGB")
        else:
            im = Image.fromarray(np.array(img)).convert("RGB")

        # Use UUID for unique filenames (avoids int32 bounds issues)
        fname = cls_dir / f"img_{uuid4().hex}.png"

        # Save; skip any rare corrupt images
        try:
            im.save(fname)
        except Exception as e:
            print(f"[WARN] Skipping one image due to save error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--hf-dataset", type=str, default="tanganke/gtsrb")
    args = parser.parse_args()

    out_root = Path(args.out)
    (out_root / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "valid").mkdir(parents=True, exist_ok=True)

    print(f"Downloading HF dataset: {args.hf_dataset}")
    ds = load_dataset(args.hf_dataset)  # splits: 'train','test' (we'll use 'train')
    train_split = ds["train"]

    labels = list(train_split["label"])
    indices_by_class = defaultdict(list)
    for idx, y in enumerate(labels):
        indices_by_class[int(y)].append(idx)

    train_examples, train_labels = [], []
    valid_examples, valid_labels = [], []

    for cls, idxs in indices_by_class.items():
        if len(idxs) < 2:
            cls_train, cls_valid = idxs, []
        else:
            cls_train, cls_valid = train_test_split(
                idxs, test_size=args.valid_ratio, random_state=args.seed, shuffle=True
            )
        for i in cls_train:
            train_examples.append(train_split[i]["image"])
            train_labels.append(cls)
        for i in cls_valid:
            valid_examples.append(train_split[i]["image"])
            valid_labels.append(cls)

    print("Writing images to folder structure...")
    save_split(train_examples, train_labels, out_root, "train")
    save_split(valid_examples, valid_labels, out_root, "valid")

    print("Done.")
    print(f"Train: {out_root / 'train'}")
    print(f"Valid: {out_root / 'valid'}")
    print("Now run:  python .\\src\\train.py")

if __name__ == "__main__":
    main()
