import numpy as np
from .base import BaseGenerator
from typing import Literal


class NormalGenerator(BaseGenerator):
    def __init__(
        self,
        shape: tuple[int] = None,
        ts: np.ndarray | None = None,
        combine_domain: Literal["time", "frequency"] = None,
        combine_mode: Literal["add", "mul"] | None = None,
        mean=0.0,
        std=0.1,
    ):
        """
        Initialize the NormalNoiseGenerator.

        Args:
            shape: (tuple[int]): the shape of the TS to generate
            ts: (np.ndarray): time series to "clone" as of shape and other stuff, not clone in the literal sense.
            combine_domain (str): Domain in which the combine operation happens when `apply` method is called ('time' or 'frequency').
            combine_mode (str): Mode in which the combine operation happens when `apply` method is called ('add' or 'mul').
            mean : float, default=0.0
                Mean of the Gaussian noise to be added.
            std : float, default=0.1
                Standard deviation of the Gaussian noise to be added.
        """
        super().__init__(shape, ts, combine_domain, combine_mode)
        self.mean = mean
        self.std = std

    def generate(self) -> np.ndarray:

        noise = np.random.normal(self.mean, self.std, self.shape)

        return noise
