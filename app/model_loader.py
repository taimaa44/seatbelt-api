import os
import requests
import numpy as np
import tensorflow as tf

MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "classes.txt")

# رابط الموديل من HuggingFace
MODEL_URL = "https://huggingface.co/taimaa47/seatbelt-model/resolve/main/seatbelt_classifier_final.keras?download=true"
def download_if_needed(url, output_path):
    if not os.path.exists(output_path):
        print("Downloading model...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        r = requests.get(url)
        with open(output_path, "wb") as f:
            f.write(r.content)

# تحميل الموديل إذا مش موجود
download_if_needed(MODEL_URL, MODEL_PATH)

# تحميل الموديل
model = tf.keras.models.load_model(MODEL_PATH)

# تحميل أسماء الكلاسات
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    class_names = ["no_seatbelt", "seatbelt"]  # fallback

def predict(image_array):
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    class_index = np.argmax(predictions)
    confidence = float(np.max(predictions))

    return {
        "class": class_names[class_index],
        "confidence": confidence
    }
