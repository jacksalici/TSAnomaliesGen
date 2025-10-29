import numpy as np
from .base import BaseGenerator
from typing import Literal


class ExponentialGenerator(BaseGenerator):
    def __init__(
        self,
        shape: tuple[int] = None,
        ts: np.ndarray | None = None,
        combine_domain: Literal["time", "frequency"] = None,
        combine_mode: Literal["add", "mul"] | None = None,
        scale=1.0,
    ):
        """
        Initialize the ExponentialGenerator.

        Args:
            shape: (tuple[int]): the shape of the TS to generate
            ts: (np.ndarray): time series to "clone" as of shape and other stuff, not clone in the literal sense.
            combine_domain (str): Domain in which the combine operation happens when `apply` method is called ('time' or 'frequency').
            combine_mode (str): Mode in which the combine operation happens when `apply` method is called ('add' or 'mul').
            scale : float, default=1.0
                Scale parameter (1/lambda) of the exponential distribution.
        """
        super().__init__(shape, ts, combine_domain, combine_mode)
        self.scale = scale

    def generate(self) -> np.ndarray:

        noise = np.random.exponential(self.scale, self.shape)

        return noise


if __name__ == "__main__":
    generator = ExponentialGenerator(
        shape=(500, 3),
        scale=1.0
    )
    
    generator.test()
