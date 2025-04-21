from artifacts.inference import Inference
from model import inference
import pytest


def test_can_import_inference():
    assert Inference is not None


def test_inference_is_created():
    assert inference is not None


@pytest.mark.long
def test_can_predict():
    model_path = "artifacts"
    inference = Inference(model_path=model_path)

    prediction = inference.predict("This is a test sentence.")

    assert inference is not None
    assert prediction in ["positive", "negative"]


@pytest.mark.skip(reason="Fatal Python error: Segmentation fault")
def test_can_load_with_cloudpickle():
    import cloudpickle

    with open("artifacts/inference_class.pkl", "rb") as file:
        Inference = cloudpickle.load(file)
    model_path = "artifacts"
    inference = Inference(model_path=model_path)

    assert inference is not None
