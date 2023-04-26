import numpy as np
import joblib
import tensorflow_hub as hub
from flask import Flask, request, jsonify
from functions import evaluate_embedding_model
from sklearn.metrics import classification_report, jaccard_score

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def get_predicted_tags():
    data = request.get_json()
    input_texts = data['input_texts']
    predicted_tags = evaluate_embedding_model(input_texts)
    return jsonify({"tags": predicted_tags})

if __name__ == '__main__':
    app.run(debug=True)
