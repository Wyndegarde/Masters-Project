import pytest
import torch.nn as nn

from handlers import ModelHandler


class TestModelHandler:
    def test_spiking_model(self):
        model_config = ModelHandler(model_type="SNN")
        assert model_config.model_type == "SNN"
        assert model_config.spiking_model == True
        assert isinstance(model_config.model, nn.Module)
        assert isinstance(model_config.loss_fn, nn.MSELoss)
