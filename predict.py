import os
import urllib.request
import numpy as np
from PIL import Image
import io
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Konfigurasi
MODEL_PATH = os.getenv("MODEL_PATH", "MobileNetV2_RiceLeaf.h5")
MODEL_URL = os.getenv("MODEL_URL")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            if not MODEL_URL:
                # Jika di Railway, pastikan variable ini ada di dashboard
                raise RuntimeError("MODEL_URL tidak ditemukan di Environment Variables.")
            print(f"Downloading model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

CLASS_NAMES = ["brown_spot", "healthy", "leaf_blast", "rice_hispa", "sheath_blight", "tungro"]

def predict_image(image_bytes: bytes) -> dict:
    model = get_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    idx = np.argmax(predictions[0])
    
    return {
        "predicted_class": CLASS_NAMES[idx],
        "confidence": round(float(np.max(predictions[0])) * 100, 2),
        "all_probabilities": {CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2) for i in range(len(CLASS_NAMES))}
    }

def get_rice_feedback(disease_name: str):
    if disease_name.lower() == "healthy":
        return "Tanaman padi Anda sehat. Tetap jaga sanitasi lingkungan dan pemupukan rutin."
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # PROMPT SPESIFIK & TEGAS
        prompt = (
            f"Berperanlah sebagai ahli patologi tanaman padi. "
            f"Berikan saran teknis untuk penyakit '{disease_name}' dalam format berikut: "
            f"1. **Penyebab & Kondisi Favorit**: [isi], "
            f"2. **Gejala yang Harus Diwaspadai**: [isi], "
            f"3. **Solusi (Pencegahan & Pengobatan)**: [isi]. "
            f"JAWAB LANGSUNG KE POINNYA. DILARANG menggunakan kalimat pembuka seperti 'Tentu', 'Berikut ini', atau kalimat sapaan lainnya."
        )
        
        response = model.generate_content(prompt)
        return response.text.strip() # .strip() untuk membersihkan spasi/newline di awal/akhir
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "Saran penanganan tidak tersedia saat ini."