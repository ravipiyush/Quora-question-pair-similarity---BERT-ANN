import re
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z ]', ' ', text)
    return " ".join([w for w in text.split() if w not in stop_words])


def tfidf_weighted_w2v(sentence, w2v_model, tfidf_dict, vector_size):
    vec = np.zeros(vector_size)
    weight_sum = 0

    for word in sentence.split():
        if word in w2v_model:
            weight = tfidf_dict.get(word, 1.0)  # fallback weight
            vec += w2v_model[word] * weight
            weight_sum += weight

    if weight_sum == 0:
        return vec
    return vec / weight_sum
