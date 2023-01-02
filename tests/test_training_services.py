import pytest
from datetime import datetime


from services import TrainingServices


class TestTrainingServices:
    """
    This class holds all tests for the TrainingServices class.
    """

    def test_get_best_epoch(self):
        valid_accuracy = [0.5, 0.6, 0.7, 0.8, 0.9]
        best_epoch, best_valid_accuracy = TrainingServices.get_best_epoch(
            valid_accuracy
        )

        assert best_epoch == 5
        assert best_valid_accuracy == 0.9
