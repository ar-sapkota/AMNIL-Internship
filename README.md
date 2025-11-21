# BERT Sentiment Classification
This repository contains a Nepali sentiment classification project using a BERT-based ONNX model, served via FastAPI and optionally visualized with Streamlit. The ONNX model is automatically downloaded from Hugging Face if it is not already present.

## Project Structure
.
├── app/
│ ├── main.py # FastAPI app
│ ├── models.py # ONNX model loading & prediction
│ ├── schemas.py # Pydantic schemas
│ ├── streamlit_app.py # Streamlit UI
│ └── models/ # ONNX model will be downloaded here automatically
├── requirements.txt
└── .gitignore



---

### Quick Start (No Docker)

```bash
git clone https://github.com/ar-sapkota/AMNIL-Internship.git
cd AMNIL-Internship

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run FastAPI backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# In another terminal → Run Streamlit UI
streamlit run app/streamlit_app.py

