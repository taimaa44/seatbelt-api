from PIL import Image
import numpy as np
import io

def preprocess_image_bytes(image_bytes, img_size=(300, 300)):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(img_size)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return image, img_array