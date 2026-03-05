import os
import time
import subprocess
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

    # sanity check: prediction is True or False
    assert prediction in [True, False]

    # sanity check: propensity is between 0 and 1
    assert 0 <= propensity <= 1, "propensity should be between 0 and 1"

    # edge case: threshold=0 means prediction should always be True
    pred_zero, _ = score("hello", model, 0)
    assert pred_zero == True, "with threshold=0, prediction should be True"

    # edge case: threshold=1 means prediction should always be False
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
    # start the flask app on port 5050 to avoid conflicts
    env = os.environ.copy()
    env['PORT'] = '5050'

    proc = subprocess.Popen(
        ['python3', 'app.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )

    time.sleep(3)

    try:
        response = requests.post(
            'http://127.0.0.1:5050/score',
            json={'text': 'Free entry to win a brand new car! Text CAR to 90210'},
            timeout=5
        )

        assert response.status_code == 200
        result = response.json()

        assert 'prediction' in result
        assert 'propensity' in result
        assert isinstance(result['prediction'], bool)
        assert isinstance(result['propensity'], float)
        assert 0 <= result['propensity'] <= 1

    finally:
        proc.terminate()
        proc.wait()


def test_docker():
    # clean up any leftover container from a previous run
    os.system("docker rm -f spam-test-container 2>/dev/null")

    # build the docker image
    os.system("docker build -t spam-app .")

    # run the container in detached mode, mapping host port 5051 to container port 5000
    os.system("docker run -d -p 5051:5000 --name spam-test-container spam-app")

    # wait for the container to start up
    time.sleep(5)

    try:
        response = requests.post(
            'http://127.0.0.1:5051/score',
            json={'text': 'You have won a free ticket! Call 0800 now to claim'},
            timeout=5
        )

        assert response.status_code == 200
        result = response.json()

        # check response format
        assert 'prediction' in result
        assert 'propensity' in result
        assert isinstance(result['prediction'], bool)
        assert isinstance(result['propensity'], float)
        assert 0 <= result['propensity'] <= 1

    finally:
        # stop and remove the container
        os.system("docker stop spam-test-container")
        os.system("docker rm spam-test-container")
