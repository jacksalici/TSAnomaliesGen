import numpy as np
from .base import BaseGenerator
from typing import Literal


class GammaGenerator(BaseGenerator):
    def __init__(
        self,
        shape: tuple[int] = None,
        ts: np.ndarray | None = None,
        combine_domain: Literal["time", "frequency"] = None,
        combine_mode: Literal["add", "mul"] | None = None,
        shape_param=2.0,
        scale=1.0,
    ):
        """
        Initialize the GammaGenerator.

        Args:
            shape: (tuple[int]): the shape of the TS to generate
            ts: (np.ndarray): time series to "clone" as of shape and other stuff, not clone in the literal sense.
            combine_domain (str): Domain in which the combine operation happens when `apply` method is called ('time' or 'frequency').
            combine_mode (str): Mode in which the combine operation happens when `apply` method is called ('add' or 'mul').
            shape_param : float, default=2.0
                Shape parameter (k) of the gamma distribution.
            scale : float, default=1.0
                Scale parameter (theta) of the gamma distribution.
        """
        super().__init__(shape, ts, combine_domain, combine_mode)
        self.shape_param = shape_param
        self.scale = scale

    def generate(self) -> np.ndarray:

        noise = np.random.gamma(self.shape_param, self.scale, self.shape)

        return noise


if __name__ == "__main__":
    generator = GammaGenerator(
        shape=(500, 3),
        shape_param=2.0,
        scale=1.0
    )
    
    generator.test()
