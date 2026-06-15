import os
import io
import urllib.request
import numpy as np
from PIL import Image
import onnxruntime as ort
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "MobileNetV2_RiceLeaf.onnx")
MODEL_URL = os.getenv("MODEL_URL")
CLASS_NAMES = ["brown_spot", "healthy", "leaf_blast", "rice_hispa", "sheath_blight", "tungro"]

_session = None

def get_model():
    global _session
    if _session is None:
        if not os.path.exists(MODEL_PATH):
            if not MODEL_URL:
                raise RuntimeError("MODEL_PATH tidak ditemukan dan MODEL_URL tidak diset!")
            print(f"Model tidak ditemukan. Mengunduh dari {MODEL_URL}...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Download selesai!")

        _session = ort.InferenceSession(MODEL_PATH)
    return _session

def predict_image(image_bytes: bytes) -> dict:
    session = get_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    input_name = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: img_array})[0]
    idx = np.argmax(predictions[0])

    return {
        "predicted_class": CLASS_NAMES[idx],
        "confidence": round(float(np.max(predictions[0])) * 100, 2),
        "all_probabilities": {
            CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }
    }
