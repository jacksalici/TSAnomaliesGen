from abc import ABC, abstractmethod
from typing import Dict, Literal
import numpy as np
from .base import BaseGenerator


class SigmoidMaskGenerator(BaseGenerator):
    def __init__(
        self,
        shape: tuple[int] = None,
        ts: np.ndarray | None = None,
        combine_domain: Literal["time", "frequency"] = None,
        combine_mode: Literal["add", "mul"] | None = None,
        num_peaks: int = 3,
        peak_length: float = 50.0,
        length_variance: float = 10.0,
        steepness: float = 0.1,
    ):
        """
        Initialize the SigmoidMaskGenerator with double sigmoid peaks.

        Args:
            shape: (tuple[int]): the shape of the TS to generate
            ts: (np.ndarray): time series to "clone" as of shape and other stuff, not clone in the literal sense.
            combine_domain (str): Domain in which the combine operation happens when `apply` method is called ('time' or 'frequency').
            combine_mode (str): Mode in which the combine operation happens when `apply` method is called ('add' or 'mul').
            num_peaks (int): Number of sigmoid peaks to generate.
            peak_length (float): Average length of each peak.
            length_variance (float): Variance in the length of peaks.
            steepness (float): Steepness parameter k for the sigmoid functions.
        """

        super().__init__(shape, ts, combine_domain, combine_mode)
        self.num_peaks = num_peaks
        self.peak_length = peak_length
        self.length_variance = length_variance
        self.steepness = steepness

    def _double_sigmoid(self, x: np.ndarray, a: float, b: float, k: float) -> np.ndarray:
        """
        Double sigmoid function: f(x) = 1/(1+e^(-k(x-a))) - 1/(1+e^(-k(x-b)))
        
        Args:
            x: Input array (positions)
            a: Start position of the peak
            b: End position of the peak
            k: Steepness parameter
            
        Returns:
            Array with double sigmoid values
        """
        sigmoid1 = 1 / (1 + np.exp(-k * (x - a)))
        sigmoid2 = 1 / (1 + np.exp(-k * (x - b)))
        return sigmoid1 - sigmoid2

    def generate(self) -> np.ndarray:
        """
        Generate a float mask with double sigmoid peaks.
        
        Returns:
            np.ndarray: Float array of shape self.shape with values between 0 and 1
        """
        mask = np.zeros(self.shape, dtype=np.float64)
        
        # Create x-axis for the time series
        x = np.arange(self.seq_len)
        
        for j in range(self.no_variates):
            variate_mask = np.zeros(self.seq_len, dtype=np.float64)
            
            for _ in range(self.num_peaks):
                current_length = np.random.normal(self.peak_length, self.length_variance)
                
                center = np.random.uniform(current_length / 2, self.seq_len - current_length / 2)
                
                a = center - current_length / 2
                b = center + current_length / 2
                
                peak = self._double_sigmoid(x, a, b, self.steepness)
                
                # Add to variate mask (sum overlapping peaks)
                variate_mask += peak
            
            # Clip values to [0, 1] range in case of overlapping peaks
            variate_mask = np.clip(variate_mask, 0.0, 1.0)
            
            # Assign to the mask
            mask[:, j] = variate_mask
            
        return mask


if __name__ == "__main__":
    generator = SigmoidMaskGenerator(
        shape=(500, 3),
        num_peaks=1,
        peak_length=100,
        length_variance=15,
        steepness=0.15
    )
    
    generator.test()