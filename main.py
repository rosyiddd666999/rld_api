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
from models import PredictionResponse, HistoryItem, ProfileResponse
from services.ai_engine import predict_image, get_model
from services.gemini_logic import get_rice_feedback
from services.database import get_or_create_user, save_prediction, fetch_history_by_user, update_user_profile, get_user_profile

load_dotenv()


# ──────────────────────────────────────────────
# Lifespan: preload model saat server start
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield


app = FastAPI(title="PadiCare AI BFF", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────
def verify_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


# ──────────────────────────────────────────────
# Helper: Upload ke cPanel (non-fatal)
# ──────────────────────────────────────────────
def upload_to_cpanel(filename: str, content: bytes, content_type: str) -> str:
    """
    Upload file ke cPanel storage.
    Mengembalikan nama file hasil upload, atau nama file asli jika gagal.
    Fungsi ini TIDAK akan raise exception agar tidak memblokir response utama.
    """
    try:
        res = requests.post(
            os.getenv("CPANEL_UPLOAD_URL"),
            files={"file": (filename, content, content_type)},
            timeout=15,  # naikkan timeout dari 8 ke 15 detik
        )
        if res.status_code == 200:
            data = res.json()
            saved_name = data.get("file_name", filename)
            print(f"[Upload] Sukses: {saved_name}")
            return saved_name
        else:
            print(f"[Upload] Server cPanel error {res.status_code}: {res.text}")
            return filename
    except requests.exceptions.Timeout:
        print("[Upload] Timeout saat upload ke cPanel — menggunakan nama file asli.")
        return filename
    except requests.exceptions.ConnectionError as e:
        print(f"[Upload] Koneksi gagal ke cPanel: {e}")
        return filename
    except Exception as e:
        print(f"[Upload] Error tidak terduga: {e}")
        return filename


# ──────────────────────────────────────────────
# Root
# ──────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "PadiCare AI BFF is Running"}


# ──────────────────────────────────────────────
# POST /predict
# ──────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    google_id: str = Form(...),
    email: str = Form(...),
    name: str = Form(...),
    alamat: str = Form(None),
    latitude: float = Form(None),
    longitude: float = Form(None),
    api_key: str = Depends(verify_key),
):
    # 1. Baca file
    content = await file.read()

    # 2. Prediksi lokal (CPU)
    result = predict_image(content)

    # 3. Jalankan Gemini & get/create user secara PARALEL
    #    Upload dijalankan terpisah agar tidak memblokir jika timeout
    try:
        feedback_task = asyncio.to_thread(get_rice_feedback, result["predicted_class"])
        user_task = asyncio.to_thread(get_or_create_user, google_id, email, name)

        feedback, user_id = await asyncio.gather(feedback_task, user_task)
    except Exception as e:
        print(f"[Predict] Gagal saat Gemini/User task: {e}")
        raise HTTPException(status_code=500, detail="Gagal memproses prediksi atau data pengguna.")

    # 4. Upload ke cPanel — non-fatal, timeout-safe
    image_name = await asyncio.to_thread(
        upload_to_cpanel,
        file.filename,
        content,
        file.content_type,
    )

    # 5. Simpan ke database history
    try:
        await asyncio.to_thread(save_prediction, user_id, image_name, result, feedback, alamat, latitude, longitude)
    except Exception as e:
        # Log tapi jangan gagalkan response — prediksi sudah selesai
        print(f"[Predict] Gagal simpan history ke DB: {e}")

    return {
        **result,
        "feedback": feedback,
        "image_url": f"{os.getenv('STORAGE_BASE_URL')}/{image_name}",
    }


# ──────────────────────────────────────────────
# GET /history
# ──────────────────────────────────────────────
@app.get("/history", response_model=List[HistoryItem])
async def get_history(
    user_id: Optional[int] = None,
    api_key: str = Depends(verify_key),
):
    try:
        rows = await asyncio.to_thread(fetch_history_by_user, user_id)
    except Exception as e:
        print(f"[History] Gagal fetch dari DB: {e}")
        raise HTTPException(status_code=500, detail="Gagal mengambil riwayat.")

    base_url = os.getenv("STORAGE_BASE_URL")
    history_list = []

    for row in rows:
        history_list.append({
            "id": row["id"],
            "user_id": row["user_id"],
            "user_name": row["user_name"],
            "image_url": f"{base_url}/{row['image_name']}",
            "predicted_class": row["predicted_class"],
            "confidence": row["confidence"],
            "feedback": row["feedback"],
            "alamat": row["alamat"],
   	    "latitude": row["latitude"],
    	    "longitude": row["longitude"],
            "created_at": str(row["created_at"]),
        })

    return history_list

# ──────────────────────────────────────────────
# POST /profile (save address from register/edit)
# ──────────────────────────────────────────────
@app.post("/profile", response_model=ProfileResponse)
async def update_profile(
    google_id: str = Form(...),
    alamat: str = Form(None),
    latitude: float = Form(None),
    longitude: float = Form(None),
    api_key: str = Depends(verify_key),
):
    try:
        row = await asyncio.to_thread(update_user_profile, google_id, alamat, latitude, longitude)
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        return row
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Profile] Update error: {e}")
        raise HTTPException(status_code=500, detail="Gagal update profil")


# ──────────────────────────────────────────────
# GET /profile (auto-fill address before detect)
# ──────────────────────────────────────────────
@app.get("/profile", response_model=ProfileResponse)
async def get_profile(
    google_id: str,
    api_key: str = Depends(verify_key),
):
    try:
        row = await asyncio.to_thread(get_user_profile, google_id)
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        return row
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Profile] Fetch error: {e}")
        raise HTTPException(status_code=500, detail="Gagal mengambil profil")
