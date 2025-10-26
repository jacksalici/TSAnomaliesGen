import numpy as np
from .base import BaseGenerator
from typing import Literal


class PinkNoiseGenerator(BaseGenerator):
    def __init__(
        self,
        shape: tuple[int] = None,
        ts: np.ndarray | None = None,
        combine_domain: Literal["time", "frequency"] = None,
        combine_mode: Literal["add", "mul"] | None = None,
        alpha=1.0,
        amplitude=1.0,
    ):
        """
        Initialize the PinkNoiseGenerator (1/f^alpha noise).

        Args:
            shape: (tuple[int]): the shape of the TS to generate
            ts: (np.ndarray): time series to "clone" as of shape and other stuff, not clone in the literal sense.
            combine_domain (str): Domain in which the combine operation happens when `apply` method is called ('time' or 'frequency').
            combine_mode (str): Mode in which the combine operation happens when `apply` method is called ('add' or 'mul').
            alpha : float, default=1.0
                Spectral decay exponent (1.0 for pink noise, 2.0 for brown noise).
            amplitude : float, default=1.0
                Overall amplitude scaling factor.
        """
        super().__init__(shape, ts, combine_domain, combine_mode)
        self.alpha = alpha
        self.amplitude = amplitude

    def generate(self) -> np.ndarray:
        """
        Generate pink noise using FFT method.
        """
        noise = np.zeros(self.shape)
        
        for variate in range(self.no_variates):
            # Generate white noise in frequency domain
            white_noise = np.random.randn(self.seq_len)
            
            # Compute FFT
            fft = np.fft.rfft(white_noise)
            
            # Create frequency array (avoiding division by zero)
            freqs = np.fft.rfftfreq(self.seq_len)
            freqs[0] = 1e-10  # Avoid division by zero
            
            # Apply 1/f^alpha scaling
            fft = fft / (freqs ** (self.alpha / 2.0))
            
            # Transform back to time domain
            pink = np.fft.irfft(fft, n=self.seq_len)
            
            # Normalize and scale
            pink = pink / np.std(pink) * self.amplitude
            
            noise[:, variate] = pink
        
        return noise
