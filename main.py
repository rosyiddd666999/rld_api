import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from predict import predict_image
from models import PredictionResponse

app = FastAPI(
    title="Rice Leaf Disease API",
    description="API for predicting rice leaf disease using MobileNetV2",
    version="1.0.0"
)

# Allow requests from Laravel / Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain Laravel kamu di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple API Key auth (untuk keamanan komunikasi Laravel → FastAPI)
API_KEY = os.getenv("API_KEY")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


@app.get("/")
def root():
    return {"status": "ok", "message": "Rice Leaf Disease API is running"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    # Validasi format file
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="File must be JPG or PNG image")

    # Baca file
    image_bytes = await file.read()

    # Prediksi
    result = predict_image(image_bytes)

    return result