import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import asyncio
import requests
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from models import PredictionResponse, HistoryItem
from services.ai_engine import predict_image, get_model
from services.gemini_logic import get_rice_feedback
from services.database import get_or_create_user, save_prediction, fetch_history_by_user

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model() # Preload model
    yield

app = FastAPI(title="PadiCare AI BFF", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

@app.get("/")
async def root():
    return {"message": "PadiCare AI BFF is Running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    google_id: str = Form(...),
    email: str = Form(...),
    name: str = Form(...),
    alamat: str = Form(None),
    api_key: str = Depends(verify_key)
):
    # 1. Baca File & Prediksi Lokal (CPU)
    content = await file.read()
    result = predict_image(content)
    
    # 2. Eksekusi PARALEL untuk Gemini & Upload cPanel
    try:
        # Task 1: Tanya Gemini
        feedback_task = asyncio.to_thread(get_rice_feedback, result['predicted_class'])
        
        # Task 2: Upload ke cPanel
        upload_task = asyncio.to_thread(
            requests.post, 
            os.getenv("CPANEL_UPLOAD_URL"), 
            files={"file": (file.filename, content, file.content_type)},
            timeout=8
        )

        # Task 3: Find/Create User di DB
        user_task = asyncio.to_thread(get_or_create_user, google_id, email, name)

        # Jalankan semua secara bersamaan
        feedback, storage_res, user_id = await asyncio.gather(feedback_task, upload_task, user_task)

        # Ambil nama file asli dari storage
        image_name = storage_res.json().get("file_name", file.filename) if storage_res.status_code == 200 else file.filename

        # 3. Simpan ke Database History (Async)
        await asyncio.to_thread(save_prediction, user_id, image_name, result, feedback, alamat)

        return {
            **result,
            "feedback": feedback,
            "image_url": f"{os.getenv('STORAGE_BASE_URL')}/{image_name}"
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Gagal memproses data.")

@app.get("/history", response_model=List[HistoryItem])
async def get_history(user_id: Optional[int] = None, api_key: str = Depends(verify_key)):
    try:
        # Panggil fungsi fetch yang sudah menggunakan JOIN
        rows = await asyncio.to_thread(fetch_history_by_user, user_id)
        
        base_url = os.getenv("STORAGE_BASE_URL")
        history_list = []
        for row in rows:
            history_list.append({
                "id": row['id'],
                "user_id": row['user_id'],
                "user_name": row['user_name'],
                "image_url": f"{base_url}/{row['image_name']}",
                "predicted_class": row['predicted_class'],
                "confidence": row['confidence'],
                "feedback": row['feedback'],
                "alamat": row['alamat'],
                "created_at": str(row['created_at'])
            })
        return history_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))