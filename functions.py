import tensorflow as tf
import tensorflow_hub as hub
import joblib
from sklearn.metrics import jaccard_score, classification_report
import pandas as pd
import numpy as np

def use_embeddings_batch(sentences, batch_size=100):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    sentence_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        batch_embeddings = embed(batch_sentences)
        sentence_embeddings.extend(batch_embeddings.numpy())

    return np.array(sentence_embeddings)


def evaluate_embedding_model(input_texts):
    model = joblib.load('use/model.joblib')
    mlb = joblib.load('mlb.joblib')

    for input_text in input_texts:
        predicted_tags = predict_tags_embedding(input_text, model, use_embeddings_batch, mlb)
        return predicted_tags


def predict_tags_embedding(text, model, embedding_function, mlb):
    # Transformer le texte en embeddings
    text_embeddings = embedding_function([text])
    # Prédire les probabilités des tags
    predicted_tags_bin_proba = model.predict_proba(text_embeddings)

    # Trouver les indices des 4 plus grandes probabilités
    top5_indices = np.argsort(predicted_tags_bin_proba[0])[-5:][::-1]

    # Récupérer les 4 plus grandes probabilités
    top5_probs = predicted_tags_bin_proba[0][top5_indices]

    # Récupérer les tags associés aux 4 plus grandes probabilités
    top5_tags = mlb.classes_[top5_indices]

    # Renvoyer un dictionnaire associant les tags aux 4 plus grandes probabilités
    return dict(zip(top5_tags, top5_probs))