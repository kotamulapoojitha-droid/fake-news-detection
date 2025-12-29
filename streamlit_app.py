# app.py - Fake News Detection (TF-IDF, LSTM, BERT)

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except:
        pass

download_nltk_data()

# Load models
@st.cache_resource
def load_all_models():
    try:
        # TF-IDF model
        lr = joblib.load('lr_model.pkl')
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        
        # LSTM model
        lstm_model = load_model('lstm_model.h5')
        lstm_tokenizer = joblib.load('lstm_tokenizer.pkl')
        
        # BERT model
        bert_model = BertForSequenceClassification.from_pretrained('./bert_model')
        bert_tokenizer = BertTokenizer.from_pretrained('./bert_model')
        bert_model.eval()
        
        return lr, tfidf, lstm_model, lstm_tokenizer, bert_model, bert_tokenizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load models
lr, tfidf, lstm_model, lstm_tokenizer, bert_model, bert_tokenizer = load_all_models()

# UI
st.set_page_config(page_title="Fake News Detection", page_icon="🔍")
st.title("🔍 Fake News Detection")
st.write("Detect if news is **Real** or **Fake** using ML models")

# Text input box
user_input = st.text_area("📰 Enter News Article:", height=200)

# Dropdown: choose model
model_choice = st.selectbox(
    "Choose Model:",
    ["TF-IDF", "LSTM", "BERT"]
)

# Predict button
if st.button("🔎 Predict"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            cleaned_text = preprocess_text(user_input)
            
            # TF-IDF model
            if model_choice == "TF-IDF":
                text_tfidf = tfidf.transform([cleaned_text])
                prediction = lr.predict(text_tfidf)[0]
                confidence = lr.predict_proba(text_tfidf)[0]
            
            # LSTM model
            elif model_choice == "LSTM":
                sequence = lstm_tokenizer.texts_to_sequences([cleaned_text])
                padded = pad_sequences(sequence, maxlen=200)
                prediction_prob = lstm_model.predict(padded, verbose=0)[0][0]
                prediction = 1 if prediction_prob > 0.5 else 0
                confidence = [1-prediction_prob, prediction_prob]
            
            # BERT model
            elif model_choice == "BERT":
                inputs = bert_tokenizer(
                    cleaned_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probs).item()
                confidence = probs[0].detach().numpy()
            
            # Output: Fake / Real
            st.write("---")
            if prediction == 1:
                st.error("🚨 **FAKE NEWS**")
            else:
                st.success("✅ **REAL NEWS**")
            
            # Confidence score
            conf_score = confidence[prediction] * 100
            st.metric("Confidence Score", f"{conf_score:.2f}%")
            
            st.write(f"Model used: **{model_choice}**")
    else:
        st.warning("⚠️ Please enter text!")
