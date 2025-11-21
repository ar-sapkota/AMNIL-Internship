# ---------- Base Image ----------
FROM python:3.12-slim

# ---------- Set working directory ----------
WORKDIR /app

# ---------- Install system dependencies ----------
RUN apt-get update && \
    apt-get install -y build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# ---------- Copy requirements and install ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy all project files ----------
COPY . .

# ---------- Expose ports ----------
EXPOSE 8000 8501

# ---------- Streamlit environment variables ----------
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false

# ---------- Run FastAPI + Streamlit with startup message ----------
CMD bash -c "echo ' FastAPI: http://localhost:8000'; echo '🖥 Streamlit: http://localhost:8501'; uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.port 8501"
