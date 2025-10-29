from abc import ABC, abstractmethod
from typing import Any, Dict, Literal
import numpy as np


class BaseGenerator(ABC):
    def __init__(
        self,
        shape: tuple[int] = None,
        ts: np.ndarray | None = None,
        combine_domain: Literal["time", "frequency"] = None,
        combine_mode: Literal["add", "mul"] | None = None,
    ):
        """
        Initialize the BaseGenerator.

        Args:
            shape: (tuple[int]): the shape of the TS to generate
            ts: (np.ndarray): time series to "clone" as of shape and other stuff, not clone in the literal sense.
            combine_domain (str): Domain in which the combine operation happens when `apply` method is called ('time' or 'frequency').
            combine_mode (str): Mode in which the combine operation happens when `apply` method is called ('add' or 'mul').

        """

        assert shape != None or ts != None, "shape and ts can't be both None"

        if shape is None:
            shape = ts.shape

        self.shape = shape
        self.seq_len, self.no_variates = self.shape

        if combine_domain and combine_domain not in ["time", "frequency"]:
            raise ValueError("Domain must be either 'time' or 'frequency'")

        self.combine_domain = combine_domain

        if combine_mode and combine_mode not in ["add", "mul"]:
            raise ValueError("Combine Mode not allowed")

        self.combine_mode = combine_mode

    @abstractmethod
    def generate(self) -> np.ndarray:
        raise NotImplementedError("Can't generate with base generator.")

    def get_base_ts(self) -> np.ndarray:
        match self.combine_mode:
            case "add":
                return np.zeros(self.shape)
            case "mul":
                return np.ones(self.shape)
            case default:
                raise ValueError("Combine Mode not accepted")

    def combine(
        self, ts: np.ndarray, generated_ts: np.ndarray, mask_ts: np.ndarray = None
    ):
        assert ts.shape == generated_ts.shape
        assert self.combine_mode and self.combine_domain
        if mask_ts is not None:
            assert ts.shape == mask_ts.shape
           
        if self.combine_domain == "frequency":
            ts = np.fft.fft(ts, axis=0)

        match self.combine_mode:
            case "add":
                if mask_ts is not None:
                    ts = ts + mask_ts * generated_ts
                else:
                    ts = ts + generated_ts

            case "mul":
                if mask_ts is not None:
                    ts = ts * mask_ts * generated_ts
                else:
                    ts = ts * generated_ts

        if self.combine_mode == "frequency":
            ts = np.fft.ifft(ts, axis=0).real

        return ts

    def generate_and_combine(
        self, ts: np.ndarray, mask_ts: np.ndarray = None
    ) -> np.ndarray:
        """
        Generate the TS and combine it to the input time series in the specified domain.

        Args:
            ts (np.ndarray): Input time series data.
            mask_ts (np.ndarray): Optional boolean mask

        """

        generated_ts = self.generate()
        return self.combine(ts, generated_ts, mask_ts)
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.combine_domain}, {self.combine_mode})"
    
    
    def test(self):
        print(f"{self.__class__} test method called")
        
        import matplotlib.pyplot as plt
        ts = self.generate()
        for i in range(ts.shape[1]):
            plt.plot(ts[:, i], label=f'Variate {i}')
        plt.show()
        
        