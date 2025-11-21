# schemas.py
from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    score: float

class BatchPredictionRequest(BaseModel):
    texts: List[str]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class EvaluationRequest(BaseModel):
    dataset_path: str

class EvaluationResponse(BaseModel):
    accuracy: float
    f1_score: float

class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    model_loaded: bool
    model_name: str
    timestamp: float
