from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import PercentFormatter


from config import ModelParameters


class AnalysisHandler:
    """
    This class is responsible for analysing the output metrics from the training process.
    """

    def __init__(self, history) -> None:
        self.history = history

    def plot_history(
        self,
        res: int = ModelParameters.RESOLUTION,
        loss_upper: float = 1.05,
        acc_lower: float = -0.05,
        acc_higher: float = 105,
    ):
        """
        Plots the history of the training process
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        ax1.plot(self.history["avg_train_loss"], label="train loss", marker="o")
        ax1.plot(self.history["avg_valid_loss"], label="validation loss", marker="o")

        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_ylim([-0.05, loss_upper])
        ax1.legend()
        ax1.set_ylabel("Loss", fontsize=16)
        ax1.set_xlabel("Epoch", fontsize=16)

        ax2.plot(self.history["train_accuracy"], label="train accuracy", marker="o")
        ax2.plot(
            self.history["valid_accuracy"], label="validation accuracy", marker="o"
        )

        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_ylim([acc_lower, acc_higher])

        ax2.legend()

        ax2.set_ylabel("Accuracy", fontsize=16)
        ax2.yaxis.set_major_formatter(PercentFormatter(100))
        ax2.set_xlabel("Epoch", fontsize=16)
        fig.suptitle(f"Training history ({res}*{res})", fontsize=20)
        plt.show()
