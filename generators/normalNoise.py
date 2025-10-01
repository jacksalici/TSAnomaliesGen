import numpy as np
from .base import BaseGenerator


class NormalNoiseGenerator(BaseGenerator):
    def __init__(self, mean=0.0, std=0.1):
        """
        Initialize the NormalNoiseGenerator.

        Parameters:
        ----------
        mean : float, default=0.0
            Mean of the Gaussian noise to be added.
        std : float, default=0.1
            Standard deviation of the Gaussian noise to be added.
        """
        self.mean = mean
        self.std = std

    def generate(self, ts: np.ndarray) -> np.ndarray:
        """
        Generate Gaussian noise and add it to the given time series.

        Parameters:
        ----------
        ts : np.ndarray
            Input time series data of shape (seq_len, no_variates).

        Returns:
        -------
        np.ndarray
            Time series data with added Gaussian noise.
        """
        noise = np.random.normal(self.mean, self.std, ts.shape)
        return ts + noise