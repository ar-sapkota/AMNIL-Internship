from torchvision import transforms
from PIL import Image
import torch
from app.models import GoogleNet  # import GoogleNet from models.py

# 1. Load the model (from models.py)
model_instance = GoogleNet()
model = model_instance.model          # PyTorch model
class_names = model_instance.class_names  # CIFAR-10 class names

# 2. Define preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((32, 32)),        # CIFAR-10 expects 32x32 images
    transforms.ToTensor(),               # Convert PIL image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1,1]
])

# 3. Prediction function
def predict(image: Image.Image):
    """
    Preprocesses the image and predicts its class using GoogLeNet.
    Args:
        image (PIL.Image): Input image
    Returns:
        dict: {'class': <class_name>, 'confidence': <float>}
    """
    # Preprocess
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)          # Model output logits
        probabilities = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
        confidence, predicted_class = torch.max(probabilities, dim=1)  # Highest probability

    # Return results
    return {
        "class": class_names[predicted_class.item()],
        "confidence": float(confidence.item())
    }

