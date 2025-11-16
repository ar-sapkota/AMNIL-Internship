# app/test_predict.py
from app.predict import predict
from PIL import Image
import os

# Build correct path relative to this file (app/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "data", "sample_airplane.png")  # or .jpg

# Optional: Create data folder and download image if missing
if not os.path.exists(image_path):
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    print(f"Downloading sample image to {image_path}...")
    import urllib.request
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane1.png"
    urllib.request.urlretrieve(url, image_path)
    print("Download complete.")

# Load and predict
image = Image.open(image_path).convert("RGB")
result = predict(image)
print("Prediction:", result)