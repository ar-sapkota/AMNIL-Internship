# app.py
import streamlit as st
import requests

# --- Config ---
API_URL = "http://nepali-fastapi:8000"
  # your FastAPI URL

st.set_page_config(page_title="NepaliBERT Sentiment Classification", layout="centered")
st.title("NepaliBERT Sentiment Classification")
st.write("Classify Nepali text sentiment using ONNX-optimized NepaliBERT")

# --- Sidebar ---
st.sidebar.header("API Info")
st.sidebar.write(f"FastAPI URL: {API_URL}")

menu = ["Single Prediction", "Batch Prediction", "System Status", "Health Check"]
choice = st.sidebar.selectbox("Select Function", menu)

# --- Single Prediction ---
if choice == "Single Prediction":
    st.header("Single Text Prediction")
    user_text = st.text_area("Enter Nepali text here:")
    if st.button("Predict"):
        if not user_text.strip():
            st.warning("Please enter some text")
        else:
            try:
                response = requests.post(f"{API_URL}/predict", json={"text": user_text})
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Predicted Label: {result['label']}")
                    st.info(f"Confidence: {result['score']:.4f}")
                else:
                    st.error(f"Error: {response.json()['detail']}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")

# --- Batch Prediction ---
elif choice == "Batch Prediction":
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload a TXT or CSV file with texts", type=["txt", "csv"])
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "text/csv":
                import pandas as pd
                df = pd.read_csv(uploaded_file)
                texts = df['text'].tolist()
            else:
                # Assume each line is a text
                texts = [line.strip() for line in uploaded_file.read().decode("utf-8").splitlines() if line.strip()]

            if st.button("Predict Batch"):
                response = requests.post(f"{API_URL}/predict_batch", json={"texts": texts})
                if response.status_code == 200:
                    results = response.json()["predictions"]
                    for i, res in enumerate(results):
                        st.write(f"Text {i+1}: {res['label']} (Confidence: {res['score']:.4f})")
                else:
                    st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# --- System Status ---
elif choice == "System Status":
    st.header("System Status")
    try:
        response = requests.get(f"{API_URL}/system-status")
        if response.status_code == 200:
            status = response.json()
            st.json(status)
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Request failed: {str(e)}")

# --- Health Check ---
elif choice == "Health Check":
    st.header("Health Check")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            health = response.json()
            st.json(health)
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
