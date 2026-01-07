import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Quora Question Pair Similarity",
    page_icon="🧠",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
.stTextArea textarea {
    font-size: 16px;
}
.confidence-bar {
    height: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS (CACHED) ----------------
@st.cache_resource
def load_all_models():
    bert = SentenceTransformer("all-MiniLM-L6-v2")

    model = load_model("bert_ann.h5")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return bert, model, scaler

with st.spinner("🔄 Loading AI models..."):
    bert, model, scaler = load_all_models()

# ---------------- TITLE ----------------
st.markdown(
    "<h1 style='text-align:center;'>🔍 Quora Question Pair Similarity</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:gray;'>Powered by BERT + Neural Network</p>",
    unsafe_allow_html=True
)

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    q1 = st.text_area("📝 Question 1", height=120)

with col2:
    q2 = st.text_area("📝 Question 2", height=120)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🚀 Check Similarity", use_container_width=True):

    if q1.strip() == "" or q2.strip() == "":
        st.warning("⚠️ Please enter both questions.")
    else:
        with st.spinner("🤖 Understanding semantics..."):
            e1 = bert.encode([q1])
            e2 = bert.encode([q2])

            abs_diff = np.abs(e1 - e2)
            cos_sim = cosine_similarity(e1, e2).reshape(-1, 1)

            X = np.hstack((abs_diff, cos_sim))
            X = scaler.transform(X)

            prob = model.predict(X)[0][0]

        # ---------------- RESULT ----------------
        st.markdown("## 📊 Result")

        confidence = prob if prob > 0.5 else 1 - prob
        percent = int(confidence * 100)

        if prob > 0.5:
            st.success(f"✅ **SIMILAR QUESTIONS**")
            st.progress(percent)
            st.markdown(f"**Confidence:** {percent}%")
        else:
            st.error(f"❌ **NOT SIMILAR QUESTIONS**")
            st.progress(percent)
            st.markdown(f"**Confidence:** {percent}%")

        # ---------------- DETAILS ----------------
        with st.expander("🔎 Technical Details"):
            st.write("**Embedding Model:** BERT (all-mpnet-base-v2)")
            st.write("**Classifier:** Artificial Neural Network")
            st.write("**Decision Threshold:** 0.5")
            st.write(f"**Cosine Similarity:** {cos_sim[0][0]:.4f}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>Made with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)



