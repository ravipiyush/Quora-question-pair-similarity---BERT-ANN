import pandas as pd
import numpy as np
import pickle
import os

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/train.csv")
df = df.sample(1000, random_state=42)
df = df[['question1', 'question2', 'is_duplicate']].dropna()

# ---------------- BERT ENCODER ----------------
bert = SentenceTransformer("all-mpnet-base-v2")

emb1 = bert.encode(df['question1'].tolist(), batch_size=32, show_progress_bar=True)
emb2 = bert.encode(df['question2'].tolist(), batch_size=32, show_progress_bar=True)

# ---------------- FEATURE ENGINEERING ----------------
abs_diff = np.abs(emb1 - emb2)

# ✅ Row-wise cosine similarity (FIXED)
cos_sim = np.sum(emb1 * emb2, axis=1) / (
    np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1)
)
cos_sim = cos_sim.reshape(-1, 1)

X = np.hstack((abs_diff, cos_sim))
y = df['is_duplicate'].values

# ---------------- TRAIN / TEST ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- ANN MODEL ----------------
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=128
)

# ---------------- SAVE MODELS ----------------
os.makedirs("model", exist_ok=True)

model.save("model/bert_ann.h5")

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ BERT + ANN model trained and saved successfully")
