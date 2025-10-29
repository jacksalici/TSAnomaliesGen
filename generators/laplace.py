import numpy as np
from .base import BaseGenerator
from typing import Literal


class LaplaceGenerator(BaseGenerator):
    def __init__(
        self,
        shape: tuple[int] = None,
        ts: np.ndarray | None = None,
        combine_domain: Literal["time", "frequency"] = None,
        combine_mode: Literal["add", "mul"] | None = None,
        loc=0.0,
        scale=1.0,
    ):
        """
        Initialize the LaplaceGenerator (Laplace/double exponential distribution).

        Args:
            shape: (tuple[int]): the shape of the TS to generate
            ts: (np.ndarray): time series to "clone" as of shape and other stuff, not clone in the literal sense.
            combine_domain (str): Domain in which the combine operation happens when `apply` method is called ('time' or 'frequency').
            combine_mode (str): Mode in which the combine operation happens when `apply` method is called ('add' or 'mul').
            loc : float, default=0.0
                Location parameter (mean) of the Laplace distribution.
            scale : float, default=1.0
                Scale parameter (diversity) of the Laplace distribution.
        """
        super().__init__(shape, ts, combine_domain, combine_mode)
        self.loc = loc
        self.scale = scale

    def generate(self) -> np.ndarray:

        noise = np.random.laplace(self.loc, self.scale, self.shape)

        return noise


if __name__ == "__main__":
    generator = LaplaceGenerator(
        shape=(500, 3),
        loc=0.0,
        scale=1.0
    )
    
    generator.test()
