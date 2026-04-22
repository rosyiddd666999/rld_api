from pydantic import BaseModel
from typing import Dict, List, Optional

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    feedback: Optional[str] = None
    image_url: Optional[str] = None

class HistoryItem(BaseModel):
    id: int
    user_id: int
    user_name: str
    image_url: str
    predicted_class: str
    confidence: float
    feedback: str
    alamat: Optional[str] = None
    created_at: str