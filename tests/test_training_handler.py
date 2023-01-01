import pytest

from handlers import TrainingHandler


class TestTrainingHandler:
    def test_training_handler(self):
        training_handler = TrainingHandler()
        assert training_handler.model_type == "ANN"
