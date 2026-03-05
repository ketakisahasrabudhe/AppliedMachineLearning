import os
import time
import signal
import subprocess
import json
import joblib
import requests

from score import score


# load the saved model for unit tests
model = joblib.load('best_model.pkl')


def test_score():
    # smoke test: does it run without crashing
    prediction, propensity = score("hello world", model, 0.5)

    # format test: check output types
    assert isinstance(prediction, bool), "prediction should be a bool"
    assert isinstance(propensity, float), "propensity should be a float"

    # sanity check: prediction is 0 or 1 (True or False)
    assert prediction in [True, False]

    # sanity check: propensity is between 0 and 1
    assert 0 <= propensity <= 1, "propensity should be between 0 and 1"

    # edge case: threshold=0 means prediction should always be 1 (True)
    pred_zero, _ = score("hello", model, 0)
    assert pred_zero == True, "with threshold=0, prediction should be True"

    # edge case: threshold=1 means prediction should always be 0 (False)
    # because propensity is strictly less than 1 for any normal input
    pred_one, _ = score("hello", model, 1)
    assert pred_one == False, "with threshold=1, prediction should be False"

    # obvious spam input
    spam_text = "Congratulations! You won a free prize. Call now to claim your reward. Text WIN to 80808"
    spam_pred, spam_prop = score(spam_text, model, 0.5)
    assert spam_pred == True, "obvious spam should be predicted as spam"

    # obvious non-spam input
    ham_text = "Hey, are we still meeting for lunch tomorrow at noon?"
    ham_pred, ham_prop = score(ham_text, model, 0.5)
    assert ham_pred == False, "obvious ham should be predicted as not spam"


def test_flask():
    # start the flask app as a subprocess
    proc = subprocess.Popen(
        ['python3', 'app.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # give the server time to start
    time.sleep(3)

    try:
        # send a test request to the /score endpoint
        response = requests.post(
            'http://127.0.0.1:5000/score',
            json={'text': 'Free entry to win a brand new car! Text CAR to 90210'},
            timeout=5
        )

        assert response.status_code == 200, f"expected 200, got {response.status_code}"

        result = response.json()

        # check that response has the right keys
        assert 'prediction' in result, "response should have prediction"
        assert 'propensity' in result, "response should have propensity"

        # check types
        assert isinstance(result['prediction'], bool)
        assert isinstance(result['propensity'], float)

        # check value ranges
        assert result['propensity'] >= 0 and result['propensity'] <= 1

    finally:
        # shut down the flask app
        proc.terminate()
        proc.wait()
