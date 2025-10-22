import argparse
from pathlib import Path
import json
import tensorflow as tf

def load_any_model(path: Path):
    if path.suffix == ".keras":
        return tf.keras.models.load_model(path)
    return tf.keras.models.load_model(path, compile=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to saved .keras model")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--img-size", type=int, default=96)
    args = parser.parse_args()

    weights_path = Path(args.weights)
    model = load_any_model(weights_path)

    classes_path = weights_path.with_suffix(".classes.json")
    _ = json.loads(Path(classes_path).read_text(encoding="utf-8"))

    valid_dir = Path(args.data_dir) / "valid"
    if not valid_dir.exists() or not any(valid_dir.iterdir()):
        raise FileNotFoundError("No validation set found at data/valid. Either create it or re-run training with split.")

    val_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir, labels="inferred", image_size=(args.img_size, args.img_size), batch_size=64, shuffle=False
    )

    loss, acc = model.evaluate(val_ds, verbose=1)
    print(f"Validation accuracy: {acc:.4f}")
    print(f"Validation loss: {loss:.4f}")

if __name__ == "__main__":
    main()
