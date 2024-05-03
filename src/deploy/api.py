from flask import Flask, request, jsonify
import os
import sys
sys.path.append(os.getcwd())

from src.lightning_models.distilberrt import Distil_Bert
from src.lightning_models.naive_bayes import naive_bayes

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_type = data['model_type']
    raw_text = data['text']
    lightning_model = get_model(model_type)

    prediction = lightning_model.predict(raw_text)
    prediction_decoder = {0: 'Irrelevant', 1: 'Negative', 2: 'Neutral', 3: 'Positive'}
    prediction = prediction_decoder[prediction]

    return jsonify({
        'model_used': model_type,
        'input_text': raw_text,
        'predicted_sentiment': prediction
    })

def get_model(model_type: str):
    """
    Load pretrained model based on the model type

    Args:
    model_type (str): Model type to be loaded

    Returns:
    model: Pretrained model
    """
    assert model_type in ['distil_bert', 'naive_bayes']
    if model_type == 'distil_bert':
        lightning_model = Distil_Bert.load_from_checkpoint('saved_models/distil_bert.pkl', num_labels=4)
        lightning_model.eval()
    else:
        lightning_model = naive_bayes()

    return lightning_model

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
