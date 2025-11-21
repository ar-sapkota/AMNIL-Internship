# BERT Sentiment Classification

### Used Two datasets and merged into one.
**size of dataset: 42k approx**
**demoji used to handle emojis**
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
### Create virtual env


```bas

git clone https://github.com/ar-sapkota/AMNIL-Internship.git
cd AMNIL-Internship
git checkout bert-sentiment-classification

# Create a virtual environment (recommended)
python -m venv myenv

# Activate the virtual environment
# On Windows:
myenv\Scripts\activate
# On Linux/Mac:
source myenv/bin/activate

# Install dependencies recommendd to create virtual env
pip install --upgrade pip
pip install -r requirements.txt

#If you encounter conflicts with torch/torchvision, use the official CPU install
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


# Run FastAPI backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# In another terminal → Run Streamlit UI
streamlit run streamlit_app.py

