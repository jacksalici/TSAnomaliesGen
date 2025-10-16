from .base import AdditiveGenerator
import numpy as np


class SinusoidGenerator(AdditiveGenerator):
    def __init__(self, shape: tuple[int], frequency=1.0, amplitude=1.0, phase=0.0, domain="time"):
        """
        Initialize the SinusoidGenerator.

        Args:
            shape: (seq_len, no_variates)
                Shape of the time series data.
            frequency : float, default=1.0
                Frequency of the sinusoidal signal.
            amplitude : float, default=1.0
                Amplitude of the sinusoidal signal.
            phase : float, default=0.0
                Phase shift of the sinusoidal signal.
            domain : str, default='time'
                Domain in which the generator operates ('time' or 'frequency').
        """
        super().__init__(shape, domain)
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

    def generate(self, ts = None, params = None) -> np.ndarray:
        """
        Create a sinusoidal additive component.

        Returns:
            np.ndarray: Sinusoidal component to be added to the input.
        """
        seq_len, no_variates = self.base.shape
        t = np.linspace(0, 2 * np.pi, seq_len)  # Time vector

        component = np.zeros((seq_len, no_variates), dtype=float)

        for j in range(no_variates):
            component[:, j] = self.amplitude * np.sin(self.frequency * t + self.phase)

        return component
