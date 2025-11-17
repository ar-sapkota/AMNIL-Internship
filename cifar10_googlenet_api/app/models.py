# app/models.py
import torch
from torchvision import models

class GoogleNet:
    def __init__(self, device="cpu"):
        # Initialize the model (adapt for CIFAR-10)
        self.model = models.googlenet(weights=None, aux_logits=False)  # No default weights; we'll load yours
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)  # 10 classes for CIFAR-10
        
        # Load your trained weights (adjust filename if different)
# app/models.py
        weights_path = "model/best_googlenet_cifar10.pth"  # Correct path        
        try:
            state_dict = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded trained weights from {weights_path}")
        except FileNotFoundError:
            print(f"Error: Weights file not found at {weights_path}. Using random initialization.")
        except Exception as e:
            print(f"Error loading weights: {e}. Using random initialization.")
        
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode
        
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]