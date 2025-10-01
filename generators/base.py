from abc import ABC, abstractmethod
from typing import Any, Dict, Literal
import numpy as np


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, ts: np.ndarray) -> np.ndarray:
        pass
    
    def apply(self, ts: np.ndarray, domain: Literal['time', 'frequency'] = 'time', params: Dict = {}) -> np.ndarray:
        """
        Specify the domain in which the generator operates.

        Args:
            ts (np.ndarray): Input time series data of shape (seq_len, no_variates).
            domain (str): Domain in which to apply the generator ('time' or 'frequency').
            params (Dict): Additional parameters for the generate method.
        """
        if domain not in ['time', 'frequency']:
            raise ValueError("Domain must be either 'time' or 'frequency'")
        self.domain = domain   
        
        if domain == 'frequency':
            ts = np.fft.fft(ts, axis=0)
        
        ts = self.generate(ts, **params)
        
        if domain == 'frequency':
            ts = np.fft.ifft(ts, axis=0).real
        
        return ts
            
            
    