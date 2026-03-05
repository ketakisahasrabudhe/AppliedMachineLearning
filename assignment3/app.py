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
    app.run(host='127.0.0.1', port=5000, debug=False)
