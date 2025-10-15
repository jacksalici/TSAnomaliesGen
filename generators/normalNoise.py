import numpy as np
from .base import AdditiveGenerator


class NormalNoiseGenerator(AdditiveGenerator):
    def __init__(self, shape: tuple[int], mean=0.0, std=0.1, domain="time"):
        """
        Initialize the NormalNoiseGenerator.

        Args:
            shape: (seq_len, no_variates)
                Shape of the time series data.
            mean : float, default=0.0
                Mean of the Gaussian noise to be added.
            std : float, default=0.1
                Standard deviation of the Gaussian noise to be added.
            domain : str, default='time'
                Domain in which the generator operates ('time' or 'frequency').
        """
        self.mean = mean
        self.std = std
        self.domain = domain
        super().__init__(shape, domain)

    def generate(self, **params) -> np.ndarray:
        """
        Generate an additive Gaussian noise component.

        Returns:
            np.ndarray: Noise component to be added to the input.
        """
        self.base += np.random.normal(self.mean, self.std, self.base.shape)
        return self.base
