from abc import ABC, abstractmethod
from typing import Dict, Literal
import numpy as np


class BaseGenerator(ABC):
    def __init__(self, domain: Literal["time", "frequency"] | None = None):
        """
        Base class for generators that produce a component signal which is then
        combined with the input time series.

        Args:
            domain: Domain in which this generator operates ('time' or 'frequency').
        """
        self.domain = domain

    @abstractmethod
    def generate(self, **params) -> np.ndarray:
        """
        Create the component to be combined with the input series.

        The component should be generated independently from the input values,
        based only on the desired output shape. For additive generators, this
        can be thought of as generating from a baseline of zeros; for
        multiplicative generators, from a baseline of ones.

        Args:
            shape: The (seq_len, no_variates) shape for the component.

        Returns:
            np.ndarray: Component array with the specified shape.
        """
        raise NotImplementedError

    @abstractmethod
    def combine(self, ts: np.ndarray, component: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Combine the input array with the generated mask/component (e.g., add or multiply).
        """
        raise NotImplementedError

    def apply(self, ts: np.ndarray, mask: np.ndarray | None = None, params: Dict = {}) -> np.ndarray:
        """
        Apply the generator by creating a component in the configured domain
        and combining it with the input time series.

        Args:
            ts (np.ndarray): Input time series data of shape (seq_len, no_variates).
            mask (np.ndarray) - Optional: Boolean mask of the same shape as ts indicating
            params (Dict): Additional parameters forwarded to component generation
                           if needed by specific generators.
        """

        if self.domain is None:
            raise ValueError("Domain must be specified in the constructor")

        if self.domain not in ["time", "frequency"]:
            raise ValueError("Domain must be either 'time' or 'frequency'")

        x = np.fft.fft(ts, axis=0) if self.domain == "frequency" else ts

        component = self.generate(**params)

        combined = self.combine(x, component, mask)

        if self.domain == "frequency":
            combined = np.fft.ifft(combined, axis=0).real

        return combined


class AdditiveGenerator(BaseGenerator):
    """Generator that adds its component to the input series."""
    def __init__(self, shape: list[int], domain = None):
        super().__init__(domain)
        self.base = np.zeros(shape)

    def combine(self, ts: np.ndarray, component: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        if ts.shape != component.shape:
            raise ValueError(
                f"Shape mismatch in AdditiveGenerator.combine: ts{ts.shape}, component{component.shape}, mask{mask.shape}"
            )
        if mask is not None and ts.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch in AdditiveGenerator.combine: ts{ts.shape}, component{component.shape}, mask{mask.shape}"
            )
        elif mask is None:
            return ts + component
        return ts + np.where(mask, component, 0)


class MultiplicativeGenerator(BaseGenerator):
    """Generator that multiplies the input series by its component."""
    def init__(self, shape: list[int], domain = None):
        super().__init__(domain)
        self.base = np.ones(shape)

    def combine(self, ts: np.ndarray, component: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        if ts.shape != component.shape:
            raise ValueError(
                f"Shape mismatch in MultiplicativeGenerator.combine: ts{ts.shape}, component{component.shape}, mask{mask.shape}"
            )
            
        if mask is not None and ts.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch in MultiplicativeGenerator.combine: ts{ts.shape}, component{component.shape}, mask{mask.shape}"
            )
        elif mask is None:
            return ts * component
        
        return ts * np.where(mask, component, 1)
