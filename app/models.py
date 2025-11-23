# app/models.py
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score
from huggingface_hub import hf_hub_download


# ---------- Ensure models folder exists ----------
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ONNX model path
MODEL_PATH = "./models/model2.o.onnx"
TOKENIZER_NAME = "arsapkota/nepali-sentiment-bert-regu2.o"

# Download ONNX from Hugging Face if not exists locally
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = hf_hub_download(
        repo_id="arsapkota/nepali-sentiment-bert-regu",
        filename="model.onnx",
        cache_dir=MODEL_DIR
    )

class modifiedNepaliBert:
    def __init__(self, onnx_model_path: str, tokenizer_name: str):
        self.onnx_model_path = onnx_model_path
        self.tokenizer_name = tokenizer_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # Load ONNX inference session
        self.session = ort.InferenceSession(self.onnx_model_path)
        self.input_names = [i.name for i in self.session.get_inputs()]

        # Label mapping
        self.id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
        self.label2id = {v: k for k, v in self.id2label.items()}

        # Optional: track inference times
        self.inference_times = []

    def _normalize_text(self, text: str) -> str:
        """
        CRITICAL: Must match training preprocessing exactly
        """
        text = text.strip()
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        return text

    def predict(self, text: str) -> Dict[str, Any]:

        text = self._normalize_text(text)
        # Tokenize
        inputs = self.tokenizer(
            text, return_tensors="np", padding=True, truncation=True, max_length=256
        )
        ort_inputs = {k: v for k, v in inputs.items() if k in self.input_names}

        # Run inference
        outputs = self.session.run(None, ort_inputs)
        logits = outputs[0]

        # Compute probabilities
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        sentiment_id = int(logits.argmax(axis=1)[0])
        probabilities = {self.id2label[i]: float(probs[0, i]) for i in range(len(probs[0]))}

        return {
            "label": self.id2label[sentiment_id],
            "confidence": float(probs[0, sentiment_id]),
            "probabilities": probabilities,
        }

    def evaluate_model(self, dataset_path: str) -> Dict[str, float]:
        """
        Evaluate ONNX model on a CSV or JSON dataset.
        Dataset must have columns: 'text' and 'labels' (or 'label').
        """
        # Load dataset
        if dataset_path.endswith(".csv"):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith(".json"):
            df = pd.read_json(dataset_path)
        else:
            raise ValueError("Only CSV or JSON files are supported")

        # Extract labels
        if "labels" in df.columns:
            labels = df["labels"].tolist()
        elif "label" in df.columns:
            labels = df["label"].tolist()
        else:
            raise ValueError("Dataset must have 'labels' or 'label' column")

        if "text" not in df.columns:
            raise ValueError("Dataset must have 'text' column")

        texts = df["text"].astype(str).tolist()

        # Predict in batch
        preds = [self.label2id[self.predict(t)["label"]] for t in texts]

        # Compute metrics
        accuracy = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro")
        f1_weighted = f1_score(labels, preds, average="weighted")

        return {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
        }
