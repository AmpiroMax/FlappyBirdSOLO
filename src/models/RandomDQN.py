import numpy as np
import torch


class RandomNet:
    def __init__(self) -> None:
        pass

    def getStatesQualities(self, state: np.ndarray):
        return np.random.choice([0, 1], size=2, replace=False)
