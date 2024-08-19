import sys

sys.path.insert(0, "/home/kirillkruglikov/mlops-zoomcamp-project/app")

from model import prepare_features, model
from test_user import user, expercted_features


def test_prepare_features():
    features = prepare_features(user)
    assert features.to_dict("records") == expercted_features


def test_model_prediction():
    features = prepare_features(user)
    prediction = model.predict(features)
    assert prediction[0] == 0
