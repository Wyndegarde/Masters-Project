import pytest

from utils import Utilities


class TestUtilities:
    """
    This class holds all tests for the Utilities class.
    """

    def test_get_output_sizes(self):
        output_sizes = Utilities.get_output_sizes()

        # * placeholder for now.
        assert output_sizes[-1] > 0
        # assert output_sizes == [32, 16, 8, 4, 2, 1]
