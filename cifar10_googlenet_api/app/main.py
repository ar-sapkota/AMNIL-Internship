# app/main.py
from fastapi import FastAPI, File, UploadFile
from app.utils import read_image
from app.predict import predict
from app.schemas import PredictionResponse
from app.models import GoogleNet

app = FastAPI(title="CIFAR-10 GoogLeNet API")

# Load model once on startup
model_instance = GoogleNet()
model = model_instance.model
class_names = model_instance.class_names

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Receives an image file and returns CIFAR-10 class prediction.
    """
    # 1. Convert uploaded file to PIL Image (async)
    image = await read_image(file)
    
    # 2. Run prediction
    result = predict(image, model, class_names)
    
    # 3. Return structured JSON
    return {
        "class_name": result["class"],
        "confidence": result["confidence"]
    }