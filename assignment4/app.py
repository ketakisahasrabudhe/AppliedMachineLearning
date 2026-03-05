import os
from flask import Flask, request, jsonify
import joblib
from score import score

app = Flask(__name__)

# load the saved model once at startup
model = joblib.load('best_model.pkl')
THRESHOLD = 0.5


@app.route('/score', methods=['POST'])
def score_endpoint():
    data = request.get_json()
    text = data['text']
    prediction, propensity = score(text, model, THRESHOLD)
    return jsonify({
        'prediction': bool(prediction),
        'propensity': float(propensity)
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
