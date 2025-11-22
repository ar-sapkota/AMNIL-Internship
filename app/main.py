# app/main.py
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    EvaluationResponse,
    EvaluationRequest
)
from app.models import modifiedNepaliBert

# --- FastAPI App ---
app = FastAPI(
    title="NepaliBERT Sentiment Classification API",
    description="ONNX-based Nepali Sentiment Classification API",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize model ---
MODEL_PATH = "./models/model.onnx"
TOKENIZER_NAME = "arsapkota/nepali-sentiment-bert-regu"
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = modifiedNepaliBert(MODEL_PATH, TOKENIZER_NAME)


# --- Endpoints ---
@app.get("/")
async def root():
    return {"message": "NepaliBERT API is running", "model": MODEL_PATH}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model else "degraded",
        gpu_available=False,  # ONNXRuntime manages device automatically
        model_loaded=model is not None,
        model_name=MODEL_PATH,
        timestamp=time.time(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = model.predict(text)
    return PredictionResponse(label=result["label"], score=result["confidence"])


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    response = []
    for text in request.texts:
        result = model.predict(text)
        response.append(PredictionResponse(label=result["label"], score=result["confidence"]))
    
    return BatchPredictionResponse(predictions=response)


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    if not request.dataset_path:
        raise HTTPException(status_code=400, detail="Dataset path cannot be empty")
    try:
        metrics = model.evaluate_model(request.dataset_path)
        return EvaluationResponse(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system-status")
async def system_status():
    return {
        "model_loaded": model is not None,
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "model_name": MODEL_PATH,
        "inference_count": len(model.inference_times),
        "avg_inference_time": (
            sum(model.inference_times) / len(model.inference_times)
            if model.inference_times
            else 0.0
        ),
        "timestamp": time.time(),
    }

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    if not request.dataset_path:
        raise HTTPException(status_code=400, detail="Dataset path cannot be empty")
    try:
        metrics = model.evaluate_model(request.dataset_path)
        return EvaluationResponse(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Run app ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
