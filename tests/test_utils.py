import pytest

from utils import Utilities


class TestUtilities:
    def test_get_output_sizes(self):
        output_sizes = Utilities.get_output_sizes()
        assert output_sizes == [32, 16, 8, 4, 2, 1]
