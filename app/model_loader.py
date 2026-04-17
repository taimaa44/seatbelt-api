import os
import gdown
import numpy as np
import tensorflow as tf

MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "classes.txt")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.npy")

# IDs تبعونك
MODEL_FILE_ID = "1yGWDyu5IrmAcr4xnWAJ22novIZLYBnV-"
CLASS_NAMES_FILE_ID = "1GF4ZVMGWthIsl9b8IWEVo6ZFimFHPgnD"
THRESHOLD_FILE_ID = "1roqiZPCk0OnJ1bWOjAeWJ8bc5fB_Vj1_"

model = None
class_names = None
threshold = None


def download_if_needed(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {output_path}...")
        gdown.download(id=file_id, output=output_path, quiet=False)


def load_all():
    global model, class_names, threshold

    if model is not None:
        return model, class_names, threshold

    os.makedirs(MODEL_DIR, exist_ok=True)

    download_if_needed(MODEL_FILE_ID, MODEL_PATH)
    download_if_needed(CLASS_NAMES_FILE_ID, CLASS_NAMES_PATH)
    download_if_needed(THRESHOLD_FILE_ID, THRESHOLD_PATH)

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f]

    threshold = float(np.load(THRESHOLD_PATH))

    print("Model loaded")
    print("Classes:", class_names)
    print("Threshold:", threshold)

    return model, class_names, threshold
