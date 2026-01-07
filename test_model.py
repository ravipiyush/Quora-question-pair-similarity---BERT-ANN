import pandas as pd
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
N_SAMPLES = 2000
THRESHOLD = 0.5

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/train.csv")
df = df[['question1', 'question2', 'is_duplicate']].dropna()
df = df.sample(N_SAMPLES, random_state=42)

X_text_1 = df['question1'].tolist()
X_text_2 = df['question2'].tolist()
y_true = df['is_duplicate'].values

# ---------------- LOAD SAVED MODELS ----------------
bert = SentenceTransformer("all-mpnet-base-v2")
model = load_model("model/bert_ann.h5")
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# ---------------- CREATE EMBEDDINGS ----------------
emb1 = bert.encode(X_text_1, batch_size=32, show_progress_bar=True)
emb2 = bert.encode(X_text_2, batch_size=32, show_progress_bar=True)

# ---------------- FEATURE ENGINEERING ----------------
abs_diff = np.abs(emb1 - emb2)

cos_sim = np.sum(emb1 * emb2, axis=1) / (
    np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1)
)
cos_sim = cos_sim.reshape(-1, 1)

X = np.hstack((abs_diff, cos_sim))
X = scaler.transform(X)

# ---------------- PREDICTIONS ----------------
y_prob = model.predict(X).ravel()
y_pred = (y_prob >= THRESHOLD).astype(int)

# ---------------- METRICS ----------------
print("\n📊 MODEL PERFORMANCE ON 2000 TEST SAMPLES\n")

print(f"Accuracy     : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision    : {precision_score(y_true, y_pred):.4f}")
print(f"Recall       : {recall_score(y_true, y_pred):.4f}")
print(f"F1-score     : {f1_score(y_true, y_pred):.4f}")
print(f"Log Loss     : {log_loss(y_true, y_prob):.4f}")
print(f"ROC-AUC      : {roc_auc_score(y_true, y_prob):.4f}")

print("\n🧮 Confusion Matrix")
print(confusion_matrix(y_true, y_pred))

print("\n📄 Classification Report")
print(classification_report(y_true, y_pred))
