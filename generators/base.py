from abc import ABC, abstractmethod
from typing import Any, Dict, Literal
import numpy as np


class BaseGenerator(ABC):
    def __init__(self):
        self.domain = None
    
    
    @abstractmethod
    def generate(self, ts: np.ndarray) -> np.ndarray:
        pass
    
    def apply(self, ts: np.ndarray, params: Dict = {}) -> np.ndarray:
        """
        Specify the domain in which the generator operates.

        Args:
            ts (np.ndarray): Input time series data of shape (seq_len, no_variates).
            domain (str): Domain in which to apply the generator ('time' or 'frequency'). 
                         If None, uses the domain specified in the constructor.
            params (Dict): Additional parameters for the generate method.
        """
        
        if 'domain' not in self.__dict__ or self.domain is None:
            raise ValueError("Domain must be specified either in the constructor")
                   
        if self.domain not in ['time', 'frequency']:
            raise ValueError("Domain must be either 'time' or 'frequency'")
        
        if self.domain == 'frequency':
            ts = np.fft.fft(ts, axis=0)
        
        ts = self.generate(ts, **params)
        
        if self.domain == 'frequency':
            ts = np.fft.ifft(ts, axis=0).real
        
        return ts
            
            
    