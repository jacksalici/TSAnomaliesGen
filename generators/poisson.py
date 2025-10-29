import numpy as np
from .base import BaseGenerator
from typing import Literal


class PoissonGenerator(BaseGenerator):
    def __init__(
        self,
        shape: tuple[int] = None,
        ts: np.ndarray | None = None,
        combine_domain: Literal["time", "frequency"] = None,
        combine_mode: Literal["add", "mul"] | None = None,
        lam=1.0,
    ):
        """
        Initialize the PoissonGenerator.

        Args:
            shape: (tuple[int]): the shape of the TS to generate
            ts: (np.ndarray): time series to "clone" as of shape and other stuff, not clone in the literal sense.
            combine_domain (str): Domain in which the combine operation happens when `apply` method is called ('time' or 'frequency').
            combine_mode (str): Mode in which the combine operation happens when `apply` method is called ('add' or 'mul').
            lam : float, default=1.0
                Lambda parameter (expected number of events) of the Poisson distribution.
        """
        super().__init__(shape, ts, combine_domain, combine_mode)
        self.lam = lam

    def generate(self) -> np.ndarray:

        noise = np.random.poisson(self.lam, self.shape).astype(float)

        return noise


if __name__ == "__main__":
    generator = PoissonGenerator(
        shape=(500, 3),
        lam=3.0
    )
    
    generator.test()
