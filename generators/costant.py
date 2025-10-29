from .base import BaseGenerator
import numpy as np
from typing import Literal


class CostantGenerator(BaseGenerator):
    def __init__(
        self,
        shape: tuple[int] = None,
        ts: np.ndarray | None = None,
        combine_domain: Literal["time", "frequency"] = "time",
        combine_mode: Literal["add", "mul"] | None = "add",
        gen_fraction=0.01,
        gen_value: float | None = 1.0,
        gen_length=3,
        gen_length_variance=1,
        gen_points:bool = False,
    ):
        """
        Initialize the CostantGenerator.

        Args:
            shape: (tuple[int]): the shape of the TS to generate
            ts: (np.ndarray): time series to "clone" as of shape and other stuff, not clone in the literal sense.
            combine_domain (str): Domain in which the combine operation happens when `apply` method is called ('time' or 'frequency').
            combine_mode (str): Mode in which the combine operation happens when `apply` method is called ('add' or 'mul').
            gen_fraction : float, default=0.01
                Fraction of points in the time series to be replaced with anomalies.
            gen_value : float, default=1.0
                Constant value of the anomalies to be added. None to be random
            gen_length : int, default=3
                Average number of consecutive points to be replaced with anomalies.
            gen_length_variance : int, default=1
                Variance in the number of consecutive points for anomalies.
            gen_points: bool, default=False
                If true force lenght to be 1 and generate just random points

        """
        if gen_points and gen_length != 1:
            print(f"Warning, gen_points overwrite gen_lenght to 1 (currently: {gen_length})")
            gen_length = 1
        
        super().__init__(shape, ts, combine_domain, combine_mode)
        self.gen_fraction = gen_fraction
        self.gen_value = gen_value
        self.gen_length = gen_length
        self.gen_length_variance = gen_length_variance

    def generate(self) -> np.ndarray:

        ts = self.get_base_ts()

        num_anomalies = int(self.gen_fraction * self.seq_len * self.no_variates)
        gen_indices = np.random.choice(
            self.seq_len * self.no_variates, num_anomalies, replace=False
        )
        for idx in gen_indices:
            i = idx // self.no_variates
            j = idx % self.no_variates

            gen_len = max(
                1,
                int(
                    np.random.normal(self.gen_length, self.gen_length_variance)
                ),
            )
            
            cur_value = self.gen_value if self.gen_value else np.random.random(1)
            for k in range(gen_len):
                if i + k < self.seq_len:
                    ts[i + k, j] = cur_value
        return ts


if __name__ == "__main__":
    generator = CostantGenerator(
        shape=(500, 1),
        gen_fraction=0.05,
        gen_value=None,
        gen_length=10,
        gen_length_variance=2
    )
    
    generator.test()
