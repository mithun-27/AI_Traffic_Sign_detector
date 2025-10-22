import argparse
from pathlib import Path
import json
from PIL import Image
import numpy as np
import tensorflow as tf

def load_any_model(path: Path):
    if path.suffix == ".keras":
        return tf.keras.models.load_model(path)
    return tf.keras.models.load_model(path, compile=False)

def load_image(path, img_size):
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to saved model (.keras preferred)")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=96)
    args = parser.parse_args()

    weights_path = Path(args.weights)
    model = load_any_model(weights_path)

    classes_path = weights_path.with_suffix(".classes.json")
    class_names = json.loads(Path(classes_path).read_text(encoding="utf-8"))

    x = load_image(args.image_path, args.img_size)
    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))
    print(f"Predicted: {class_names[idx]} (class {idx}) with confidence {probs[idx]:.4f}")

if __name__ == "__main__":
    main()
