from artifacts.inference import Inference
import pytest


def test_can_import_inference():
    assert Inference is not None


@pytest.mark.long
def test_can_create_inference():
    model_path = "artifacts"
    inference = Inference(model_path=model_path)
    assert inference is not None
