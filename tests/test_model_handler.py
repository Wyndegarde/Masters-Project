import pytest
import torch.nn as nn

from handlers import ModelHandler


class TestModelHandler:
    def test_spiking_model(self):
        model_config = ModelHandler()
        assert model_config.model_type == "SNN"
        assert model_config.spiking_model == True
        assert model_config.loss_fn == nn.MSELoss()
