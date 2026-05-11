import os
import json
import tensorflow as tf
import keras_tuner as kt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_split_dirs():
    pointer = os.path.join(BASE_DIR, "data", "SPLIT_POINTER.txt")
    if not os.path.exists(pointer):
        raise RuntimeError("SPLIT_POINTER.txt não encontrado. Rode: python split_dataset.py")

    train_dir = None
    val_dir = None
    with open(pointer, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("TRAIN="):
                train_dir = line.split("=", 1)[1]
            elif line.startswith("VAL="):
                val_dir = line.split("=", 1)[1]

    if not train_dir or not val_dir:
        raise RuntimeError("SPLIT_POINTER.txt inválido. Rode split_dataset.py novamente.")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise RuntimeError("Pastas train/val do pointer não existem. Rode split_dataset.py novamente.")

    return train_dir, val_dir

TRAIN_DIR, VAL_DIR = load_split_dirs()
print("Usando TRAIN_DIR:", TRAIN_DIR)
print("Usando VAL_DIR:  ", VAL_DIR)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
CLASSES = ["A", "E", "I", "O", "U"]
NUM_CLASSES = len(CLASSES)

def make_datasets(batch_size: int):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=False
    )

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    def prep(x, y):
        x = tf.cast(x, tf.float32)
        x = preprocess(x)
        return x, y

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train_ds, val_ds

def build_model(hp: kt.HyperParameters) -> tf.keras.Model:
    dense_units = hp.Choice("dense_units", [64, 128, 256, 512])
    dropout = hp.Float("dropout", min_value=0.1, max_value=0.6, step=0.1)
    lr = hp.Choice("lr", [1e-4, 3e-4, 1e-3, 3e-3])

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        raise RuntimeError("Pastas de treino/validação não existem (pointer). Rode: python split_dataset.py")

    batch_size = 32
    train_ds, val_ds = make_datasets(batch_size=batch_size)

    tuner = kt.BayesianOptimization(
        build_model,
        objective="val_accuracy",
        max_trials=12,
        directory=os.path.join(BASE_DIR, "tuner_runs"),
        project_name="vogais_automl"
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5),
    ]

    tuner.search(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks, verbose=1)

    best_model = tuner.get_best_models(num_models=1)[0]

    # Fine-tuning leve (opcional, mas costuma melhorar)
    base_model = None
    for layer in best_model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("mobilenetv2"):
            base_model = layer
            break

    if base_model is not None:
        base_model.trainable = True
        fine_tune_at = 120
        for l in base_model.layers[:fine_tune_at]:
            l.trainable = False

        best_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        best_model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=callbacks, verbose=1)

    model_path = os.path.join(MODEL_DIR, "best_model.keras")
    best_model.save(model_path)

    labels_path = os.path.join(MODEL_DIR, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"labels": CLASSES}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Modelo salvo em: {model_path}")
    print(f"✅ Labels salvas em: {labels_path}")

if __name__ == "__main__":
    main()