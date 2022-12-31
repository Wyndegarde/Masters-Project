from typing import Any

import torch.nn as nn

from config import TrainingParameters

from models import AnnNet, CnnNet, SnnNet, ScnnNet


class ModelHandler:
    def __init__(self, model_type: str = "ANN"):
        self.model_type = model_type

        self.model = self._get_model()
        self.spiking_model = self._get_spiking_model()
        self.loss_fn = self._get_loss_fn(self.spiking_model)

    def _get_model(self) -> nn.Module:
        match self.model_type:
            case "ANN":
                model: nn.Module = AnnNet()
            case "CNN":
                model = CnnNet()
            case "SNN":
                model = SnnNet()
            case "SCNN":
                model = ScnnNet()
            case _:
                raise ValueError("Model type not recognised")
        return model

    def _get_spiking_model(self) -> bool:
        if self.model_type in ["SNN", "SCNN"]:
            spiking_model = True
        else:
            spiking_model = False
        return spiking_model

    def _get_loss_fn(self, spiking_model: bool = False) -> Any:
        if spiking_model:
            loss_fn: Any = nn.MSELoss()
        else:
            loss_fn = TrainingParameters.LOSS_FUNCTION
        return loss_fn
