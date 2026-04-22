from pydantic import BaseModel
from typing import Dict, List, Optional

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    feedback: Optional[str] = None

class HistoryItem(BaseModel):
    id: int
    image_url: str
    predicted_class: str
    confidence: float
    feedback: str
    created_at: str