import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")
SPLIT_POINTER = os.path.join(DATA_DIR, "SPLIT_POINTER.txt")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
        labels = obj.get("labels")
        if labels and isinstance(labels, list):
            return labels
    return ["A", "E", "I", "O", "U"]

def read_split_pointer():
    """
    Espera algo tipo:
      TRAIN=C:\\...\\data\\train_YYYYMMDD_HHMMSS
      VAL=C:\\...\\data\\val_YYYYMMDD_HHMMSS
    """
    if not os.path.exists(SPLIT_POINTER):
        # fallback para data/train e data/val
        return (os.path.join(DATA_DIR, "train"), os.path.join(DATA_DIR, "val"))

    train_dir = None
    val_dir = None
    with open(SPLIT_POINTER, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().upper()
            v = v.strip()
            if k == "TRAIN":
                train_dir = v
            elif k == "VAL":
                val_dir = v

    if not train_dir or not val_dir:
        return (os.path.join(DATA_DIR, "train"), os.path.join(DATA_DIR, "val"))

    return train_dir, val_dir

def make_val_dataset(val_dir):
    ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    def prep(x, y):
        x = tf.cast(x, tf.float32)
        x = preprocess(x)
        return x, y

    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return ds

def predict_all(model, ds):
    y_true = []
    y_pred = []

    for batch_x, batch_y in ds:
        probs = model.predict(batch_x, verbose=0)
        preds = np.argmax(probs, axis=1)

        y_true.extend(batch_y.numpy().tolist())
        y_pred.extend(preds.tolist())

    return np.array(y_true), np.array(y_pred)

def save_confusion_matrix(cm, labels, out_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    # escrever números dentro
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")

    labels = load_labels()
    train_dir, val_dir = read_split_pointer()

    print(f"Usando VAL_DIR: {val_dir}")
    val_ds = make_val_dataset(val_dir)

    print(f"Carregando modelo: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Gerando previsões...")
    y_true, y_pred = predict_all(model, val_ds)

#aqui calcula a curacia#
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=labels,
        digits=4
    )

    # salvar outputs
    report_path = os.path.join(OUTPUTS_DIR, "report.txt")
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"VAL_DIR: {val_dir}\n")
        f.write(f"MODEL: {MODEL_PATH}\n\n")
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Confusion Matrix (rows=true, cols=pred):\n")
        f.write(np.array2string(cm) + "\n")

    save_confusion_matrix(cm, labels, cm_path)

    print("\n✅ Avaliação concluída!")
    print(f"Accuracy: {acc:.6f}")
    print(f"📄 Report salvo em: {report_path}")
    print(f"🖼️ Confusion matrix salva em: {cm_path}")

if __name__ == "__main__":
    main()