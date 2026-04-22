import os
import json
import requests
import mysql.connector
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from predict import predict_image, get_rice_feedback, get_model
from models import PredictionResponse, HistoryItem

load_dotenv()


# Lifespan untuk preload model saat startup (Senior Practice)
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()  # Load model ke RAM saat server start
    yield


app = FastAPI(title="PadiCare AI BFF", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        connect_timeout=10,
    )
    try:
        yield conn
    finally:
        conn.close()


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    # 1. Validasi Ekstensi (Lakukan ini sebelum baca file agar cepat)
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Ekstensi .{file_ext} tidak didukung. Gunakan JPG, JPEG, atau PNG.",
        )

    # 2. Baca file HANYA SATU KALI
    content = await file.read()
    file_size = len(content)

    # 3. Validasi Ukuran & MIME Type
    if file_size > 5 * 1024 * 1024: # 5MB
        raise HTTPException(status_code=413, detail="File terlalu besar. Maksimal 5MB.")
    
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid.")

    # 4. AI Lokal: Prediksi
    # Gunakan 'content' yang sudah kita simpan di variabel
    result = predict_image(content)

    # 5. AI Gemini: Feedback
    feedback = get_rice_feedback(result["predicted_class"])

    # 6. Storage & DB Orchestration (Isolasi Error)
    image_name = file.filename
    try:
        # Kirim ke cPanel Storage
        files = {"file": (file.filename, content, file.content_type)}
        storage_res = requests.post(
            os.getenv("CPANEL_UPLOAD_URL"), files=files, timeout=10
        )
        if storage_res.status_code == 200:
            image_name = storage_res.json().get("file_name", file.filename)

        # Simpan ke MySQL cPanel
        db = next(get_db())
        cursor = db.cursor()
        query = """INSERT INTO history_prediksi 
                   (image_name, predicted_class, confidence, all_probabilities, feedback) 
                   VALUES (%s, %s, %s, %s, %s)"""
        cursor.execute(
            query,
            (
                image_name,
                result["predicted_class"],
                result["confidence"],
                json.dumps(result["all_probabilities"]),
                feedback,
            ),
        )
        db.commit()
        cursor.close()
    except Exception as e:
        # Kita log errornya saja, tapi jangan gagalkan respon ke user
        print(f"BFF Storage/DB Error: {e}")

    # 7. Hasil Akhir
    return {**result, "feedback": feedback}

@app.get("/history", response_model=List[HistoryItem])
async def history(api_key: str = Depends(verify_api_key)):
    try:
        db = next(get_db())
        cursor = db.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, image_name, predicted_class, confidence, feedback, created_at FROM history_prediksi ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()

        base_url = os.getenv("STORAGE_BASE_URL")
        for row in rows:
            row["image_url"] = f"{base_url}/{row['image_name']}"
            row["created_at"] = str(row["created_at"])

        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
