import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os

# Load model sekali saja saat server start (lebih efisien)
MODEL_PATH = os.getenv("MODEL_PATH", "MobileNetV2_RiceLeaf.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Urutan class harus sama dengan saat training
CLASS_NAMES = [
    "brown_spot",
    "healthy",
    "leaf_blast",
    "rice_hispa",
    "sheath_blight",
    "tungro"
]

IMAGE_SIZE = (224, 224)  # Sesuai model kamu


def predict_image(image_bytes: bytes) -> dict:
    # Buka gambar dari bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize ke 224x224
    image = image.resize(IMAGE_SIZE)

    # Convert ke array dan normalize (0-1)
    img_array = np.array(image) / 255.0

    # Tambah dimensi batch → shape: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    predictions = model.predict(img_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    # Buat semua probabilitas per class
    all_probabilities = {
        CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": round(confidence * 100, 2),
        "all_probabilities": all_probabilities
    }