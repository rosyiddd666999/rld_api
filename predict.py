import os
import urllib.request
import numpy as np
from PIL import Image
import io

MODEL_PATH = os.getenv("MODEL_PATH", "MobileNetV2_RiceLeaf.h5")
MODEL_URL = os.getenv("MODEL_URL", "")

_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            if not MODEL_URL:
                raise RuntimeError("MODEL_URL tidak di-set.")
            print(f"Downloading model dari {MODEL_URL} ...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Model berhasil didownload!")

        import tensorflow as tf  # type: ignore # ← hanya tensorflow yang lazy
        print("Loading model...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("Model siap!")
    return _model


CLASS_NAMES = [
    "brown_spot",
    "healthy",
    "leaf_blast",
    "rice_hispa",
    "sheath_blight",
    "tungro"
]

IMAGE_SIZE = (224, 224)


def predict_image(image_bytes: bytes) -> dict:
    model = get_model()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    all_probabilities = {
        CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": round(confidence * 100, 2),
        "all_probabilities": all_probabilities
    }