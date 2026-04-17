from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.model_loader import load_all
from app.utils import preprocess_image_bytes

app = FastAPI()

model, class_names, threshold = load_all()


@app.get("/")
def home():
    return {"message": "Seatbelt API running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    _, img_array = preprocess_image_bytes(image_bytes)

    prob = float(model.predict(img_array)[0][0])

    pred_idx = 1 if prob >= threshold else 0
    pred_class = class_names[pred_idx]
    confidence = prob if pred_idx == 1 else 1 - prob

    return JSONResponse({
        "class": pred_class,
        "confidence": confidence,
        "probability": prob
    })