# app_streamlit.py
import joblib
import streamlit as st
import numpy as np

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
st.title("📰 Fake News Detector (Demo)")

@st.cache_resource
def load_model():
    return joblib.load("models/fake_news_pipeline.joblib")

pipe = load_model()

with st.sidebar:
    st.subheader("Settings")
    threshold = st.slider("Decision threshold (fake if P(fake) ≥ threshold)",
                          0.0, 1.0, 0.5, 0.01)

st.write("Paste an article/post below:")
text = st.text_area("", height=220, placeholder="Paste text here...")

if st.button("Predict"):
    if not text.strip():
        st.warning("Please paste some text.")
    else:
        proba = pipe.predict_proba([text])[0]
        p_real = float(proba[0])
        p_fake = float(proba[1])
        pred = "FAKE" if p_fake >= threshold else "REAL"

        st.metric("Prediction", pred)
        st.write({"prob_real": round(p_real, 3), "prob_fake": round(p_fake, 3)})

        st.subheader("Probability")
        st.bar_chart(np.array([[p_real], [p_fake]]))
