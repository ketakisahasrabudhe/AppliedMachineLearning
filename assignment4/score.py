import joblib


def score(text: str, model, threshold: float):
    """
    Score a single text using the trained model.
    Returns prediction (bool) and propensity (float).

    propensity is the model's predicted probability of spam (class 1).
    prediction is True if propensity >= threshold, else False.
    """
    # get probability of spam class
    propensity = model.predict_proba([text])[0][1]

    # apply threshold to get binary prediction
    prediction = bool(propensity >= threshold)

    return prediction, propensity
