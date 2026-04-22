import os
import io
import urllib.request
import numpy as np
from PIL import Image
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "MobileNetV2_RiceLeaf.h5")
MODEL_URL = os.getenv("MODEL_URL") # Link download model (Hugging Face/Drive direct link)
CLASS_NAMES = ["brown_spot", "healthy", "leaf_blast", "rice_hispa", "sheath_blight", "tungro"]

_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            if not MODEL_URL:
                raise RuntimeError("MODEL_PATH tidak ditemukan dan MODEL_URL tidak diset di Environment Variables!")
            
            print(f"Model tidak ditemukan. Mengunduh dari {MODEL_URL}...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Download selesai!")

        # Load model setelah dipastikan filenya ada
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def predict_image(image_bytes: bytes) -> dict:
    model = get_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    predictions = model.predict(img_array, verbose=0)
    idx = np.argmax(predictions[0])
    
    return {
        "predicted_class": CLASS_NAMES[idx],
        "confidence": round(float(np.max(predictions[0])) * 100, 2),
        "all_probabilities": {
            CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2) 
            for i in range(len(CLASS_NAMES))
        }
    }