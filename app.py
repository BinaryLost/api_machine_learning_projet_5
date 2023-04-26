import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def get_predicted_tags():
    data = request.get_json()
    vectorizer = joblib.load('vectorizer.joblib')
    mlb = joblib.load('mlb.joblib')
    text = data['text']
    model = joblib.load('LogisticRegression_model.joblib')

    text_vec = vectorizer.transform([text])
    predicted_tags_bin_proba = model.predict_proba(text_vec)

    # Trouver les indices des 5 plus grandes probabilités
    top5_indices = np.argsort(predicted_tags_bin_proba[0])[-5:][::-1]

    # Récupérer les 5 plus grandes probabilités et les tags associés
    top5_probs = predicted_tags_bin_proba[0][top5_indices]
    top5_tags = mlb.classes_[top5_indices]

    # Créer une liste de tuples associant les tags aux 5 plus grandes probabilités
    tags_prob_list = list(zip(top5_tags, top5_probs))

    # Trier la liste de tuples par probabilité en ordre décroissant
    sorted_tags_prob_list = sorted(tags_prob_list, key=lambda item: item[1], reverse=True)

    return jsonify({"tags": sorted_tags_prob_list})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

