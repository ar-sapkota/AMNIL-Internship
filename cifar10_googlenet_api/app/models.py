import torch
import torch.nn as nn
from torchvision import models
import os

class GoogleNet:
    def __init__(self, model_path: str = "../model/best_googlenet_cifar10.pth"):
        """Loads GoogLeNet model trained on CIFAR-10"""

        # Resolve model path relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))  # app/
        model_path = os.path.join(base_dir, model_path)
        model_path = os.path.normpath(model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Loading model from: {model_path}")

        # Load GoogLeNet architecture
        self.model = models.googlenet(weights=None, aux_logits=False)

        # Modify final layer for CIFAR-10 (10 classes)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

        # Load trained weights
        state = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state)

        # Set model to evaluation mode
        self.model.eval()

        # CIFAR-10 class names
        self.class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        print("GoogLeNet CIFAR-10 model loaded successfully.")

if __name__ == "__main__":
    model_instance = GoogleNet()
