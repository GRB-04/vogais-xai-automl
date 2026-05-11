import os
import json
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.json")

print("MODEL_PATH:", MODEL_PATH, "exists:", os.path.exists(MODEL_PATH))
print("LABELS_PATH:", LABELS_PATH, "exists:", os.path.exists(LABELS_PATH))

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!")

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)["labels"]
print("✅ Labels:", labels)

print("Model output shape:", model.output_shape)