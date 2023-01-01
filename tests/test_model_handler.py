import pytest

from handlers import ModelHandler


class TestModelHandler:
    def test_model_handler(self):
        model_handler = ModelHandler()
        assert model_handler.model_type == "ANN"
