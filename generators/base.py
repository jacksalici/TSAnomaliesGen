from abc import ABC, abstractmethod
from typing import Any, Dict, Literal
import numpy as np


class BaseGenerator(ABC):
    def __init__(self, domain: Literal['time', 'frequency'] = 'time'):
        """
        Initialize the BaseGenerator.

        Args:
            domain (str): Domain in which the generator operates ('time' or 'frequency').
        """
        if domain not in ['time', 'frequency']:
            raise ValueError("Domain must be either 'time' or 'frequency'")
        self.domain = domain
    
    @abstractmethod
    def generate(self, ts: np.ndarray, **params) -> np.ndarray:
        pass
    
    def apply(self, ts: np.ndarray, params: Dict = {}) -> np.ndarray:
        """
        Apply the generator to the input time series in the specified domain.

        Args:
            ts (np.ndarray): Input time series data of shape (seq_len, no_variates).
            params (Dict): Additional parameters for the generate method.
        """
        
        if self.domain == 'frequency':
            ts = np.fft.fft(ts, axis=0)
        
        ts = self.generate(ts, params)
        
        if self.domain == 'frequency':
            ts = np.fft.ifft(ts, axis=0).real
        
        return ts


class AdditiveGenerator(BaseGenerator):
    def __init__(self, shape: tuple[int], domain: Literal['time', 'frequency'] = 'time'):
        """
        Initialize the AdditiveGenerator.
        
        Args:
            shape (tuple[int]): Shape of the base matrix to which components will be added.
            domain (str): Domain in which the generator operates ('time' or 'frequency').
        """
        super().__init__(domain)
        self.base = np.zeros(shape)
    
    @abstractmethod
    def generate(self, **params) -> np.ndarray:
        """
        Generate a component to be added to the base matrix.
        
        Returns:
            np.ndarray: Component to add to the base matrix.
        """
        pass
    