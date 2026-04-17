import os
import numpy as np
import tensorflow as tf
import requests

# =========================
# PATHS
# =========================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "classes.txt")

# =========================
# HUGGING FACE LINK
# =========================
MODEL_URL = "https://huggingface.co/taimaa47/seatbelt-model/resolve/main/seatbelt_classifier_final.keras?download=true"

# =========================
# DOWNLOAD MODEL (ROBUST)
# =========================
def download_if_needed(url, output_path):
    if not os.path.exists(output_path):
        print("Downloading model from HuggingFace...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print("Download complete!")

# =========================
# LOAD EVERYTHING
# =========================
def load_all():
    # تحميل الموديل
    download_if_needed(MODEL_URL, MODEL_PATH)

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded!")

    # تحميل أسماء الكلاسات
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        class_names = ["no_seatbelt", "seatbelt"]

    # threshold (مبدئي)
    threshold = 0.5

    return model, class_names, threshold

# =========================
# PREDICTION
# =========================
def predict(image_array, model, class_names, threshold=0.5):
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    prob = float(predictions[0][0])

    if prob >= threshold:
        return {
            "class": class_names[1],
            "confidence": prob
        }
    else:
        return {
            "class": class_names[0],
            "confidence": 1 - prob
        }
