# BERT Sentiment Classification

### Used Two datasets and merged into one.
**size of dataset: 42k approx**
**demoji used to handle emojis**
This repository contains a Nepali sentiment classification project using a BERT-based ONNX model, served via FastAPI and optionally visualized with Streamlit. 

## Project Structure
📁 bert_sentiment_classification
├─ 📁 app
│  ├─ main.py
│  ├─ models.py
│  └─ ...
├─ 📁 models
│  └─ nepali-sentiment-bert.onnx
├─ streamlit_app.py
├─ Dockerfile
└─ README.md


**If the file is missing(.onnx), please download manually from**[https://huggingface.co/arsapkota/nepali-sentiment-bert/tree/main]
---

### Quick Start (No Docker)
### Create virtual env


```bas

git clone https://github.com/ar-sapkota/AMNIL-Internship.git
cd AMNIL-Internship
git checkout bert-sentiment-classification
git lfs pull

# OR (to fasten the speed)
this doesn't work for now<!-- git lfs fetch --all -->
# run for model.onnx
git lfs fetch --include="model.onnx"
git lfs checkout model.onnx


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
```

## Docker
**clone repo**
**goto the repo-folder**

``` bas

# Build and run containers
docker compose up --build

```

- **Streamlit UI:** [http://localhost:8501](http://localhost:8501)  
- **FastAPI docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

```
# To stop services
docker compose down

```
