from pydantic import BaseModel
from typing import Dict


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float  # dalam persen, contoh: 97.43
    all_probabilities: Dict[str, float]  # semua class dengan persentasenya

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_class": "healthy",
                "confidence": 97.43,
                "all_probabilities": {
                    "brown_spot": 0.12,
                    "healthy": 97.43,
                    "leaf_blast": 0.89,
                    "rice_hispa": 0.34,
                    "sheath_blight": 0.78,
                    "tungro": 0.44
                }
            }
        }