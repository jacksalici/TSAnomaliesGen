from .base import BaseGenerator
import numpy as np
from typing import Literal


class DriftGenerator(BaseGenerator):
    def __init__(
        self,
        shape: tuple[int] = None,
        ts: np.ndarray | None = None,
        combine_domain: Literal["time", "frequency"] = None,
        combine_mode: Literal["add", "mul"] | None = None,
        drift_type: Literal["linear", "exponential", "polynomial"] = "linear",
        drift_rate=0.01,
        polynomial_degree=2,
        random_drift=False,
        drift_rate_range=(0.005, 0.02),
    ):
        """
        Initialize the DriftGenerator.

        Args:
            shape: (tuple[int]): the shape of the TS to generate
            ts: (np.ndarray): time series to "clone" as of shape and other stuff, not clone in the literal sense.
            combine_domain (str): Domain in which the combine operation happens when `apply` method is called ('time' or 'frequency').
            combine_mode (str): Mode in which the combine operation happens when `apply` method is called ('add' or 'mul').
            drift_type : str, default="linear"
                Type of drift: 'linear', 'exponential', or 'polynomial'.
            drift_rate : float, default=0.01
                Rate of drift (slope for linear, base for exponential).
            polynomial_degree : int, default=2
                Degree of polynomial for polynomial drift.
            random_drift : bool, default=False
                Whether to randomize drift rate for each variate.
            drift_rate_range : tuple, default=(0.005, 0.02)
                Range for random drift rate if random_drift is True.
        """
        super().__init__(shape, ts, combine_domain, combine_mode)
        self.drift_type = drift_type
        self.drift_rate = drift_rate
        self.polynomial_degree = polynomial_degree
        self.random_drift = random_drift
        self.drift_rate_range = drift_rate_range

    def generate(self) -> np.ndarray:

        ts = self.get_base_ts()
        time_points = np.linspace(0, 1, self.seq_len)

        for variate in range(self.no_variates):
            # Get drift rate for this variate
            rate = self.drift_rate
            if self.random_drift:
                rate = np.random.uniform(*self.drift_rate_range)
            
            # Generate drift based on type
            if self.drift_type == "linear":
                drift = rate * time_points
                
            elif self.drift_type == "exponential":
                drift = np.exp(rate * time_points) - 1
                
            elif self.drift_type == "polynomial":
                drift = rate * (time_points ** self.polynomial_degree)
            
            else:
                raise ValueError(f"Unknown drift_type: {self.drift_type}")
            
            # Apply random direction
            if np.random.random() < 0.5:
                drift = -drift
            
            ts[:, variate] = drift
            
        return ts
