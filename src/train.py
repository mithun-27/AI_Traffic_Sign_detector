import argparse
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from class_map import CLASS_NAMES

def build_dataset_from_directory(data_dir, img_size, batch_size, val_split=None, seed=1337):
    data_dir = Path(data_dir)
    if (data_dir / "valid").exists() and any((data_dir / "valid").iterdir()):
        train_dir = data_dir / "train"
        valid_dir = data_dir / "valid"
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir, labels="inferred", label_mode="int",
            image_size=(img_size, img_size), batch_size=batch_size,
            shuffle=True, seed=seed
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            valid_dir, labels="inferred", label_mode="int",
            image_size=(img_size, img_size), batch_size=batch_size,
            shuffle=False
        )
    else:
        train_dir = data_dir / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"Expected {train_dir} to exist with class subfolders.")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir, labels="inferred", label_mode="int",
            validation_split=val_split if val_split else 0.1, subset="training",
            seed=seed, image_size=(img_size, img_size), batch_size=batch_size,
            shuffle=True
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir, labels="inferred", label_mode="int",
            validation_split=val_split if val_split else 0.1, subset="validation",
            seed=seed, image_size=(img_size, img_size), batch_size=batch_size,
            shuffle=False
        )
    AUTOTUNE = tf.data.AUTOTUNE
    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE)

def build_model(img_size, num_classes):
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])
    base = tf.keras.applications.MobileNetV2(
        include_top=False, input_shape=(img_size, img_size, 3), weights="imagenet"
    )
    base.trainable = False
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model, base

def fine_tune(model, base, unfreeze_at=100, lr=1e-4):
    base.trainable = True
    for layer in base.layers[:unfreeze_at]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--unfreeze-at", type=int, default=100)
    parser.add_argument("--model-path", type=str, default=str(Path(__file__).resolve().parents[1] / "models" / "tsc_mobilenetv2.keras"))
    args = parser.parse_args()

    num_classes = len(CLASS_NAMES)
    train_ds, val_ds = build_dataset_from_directory(args.data_dir, args.img_size, args.batch_size, args.val_split)
    model, base = build_model(args.img_size, num_classes)

    cb = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ModelCheckpoint(args.model_path, save_best_only=True, monitor="val_accuracy")
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cb)

    fine_tune(model, base, unfreeze_at=args.unfreeze_at, lr=1e-4)
    model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=cb)

    class_path = Path(args.model_path).with_suffix(".classes.json")
    with open(class_path, "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, ensure_ascii=False, indent=2)

    print(f"Saved model to: {args.model_path}")
    print(f"Saved classes to: {class_path}")

if __name__ == "__main__":
    main()
