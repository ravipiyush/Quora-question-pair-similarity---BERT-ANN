import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Quora Question Similarity",
    page_icon="🧠",
    layout="centered"
)

# ---------------- LOAD MODEL (CACHED) ----------------
@st.cache_resource
def load_bert():
    return SentenceTransformer("all-MiniLM-L6-v2")

with st.spinner("🔄 Loading language model..."):
    bert = load_bert()

# ---------------- UI ----------------
st.title("🔍 Quora Question Pair Similarity")
st.caption("BERT-based Semantic Similarity (Production Safe)")

q1 = st.text_area("📝 Question 1", height=120)
q2 = st.text_area("📝 Question 2", height=120)

if st.button("🚀 Check Similarity", use_container_width=True):

    if not q1.strip() or not q2.strip():
        st.warning("Please enter both questions.")
    else:
        with st.spinner("🤖 Understanding meaning..."):
            e1 = bert.encode([q1])
            e2 = bert.encode([q2])

            sim = cosine_similarity(e1, e2)[0][0]

        st.subheader("📊 Result")

        if sim >= 0.75:
            st.success("✅ Questions are SEMANTICALLY SIMILAR")
        else:
            st.error("❌ Questions are NOT SIMILAR")

        st.progress(int(sim * 100))
        st.write(f"**Similarity Score:** `{sim:.3f}`")

        with st.expander("🔎 Details"):
            st.write("Model: Sentence-BERT (MiniLM)")
            st.write("Metric: Cosine Similarity")
            st.write("Threshold: 0.75")
