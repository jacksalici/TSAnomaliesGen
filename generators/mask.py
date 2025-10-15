from abc import ABC, abstractmethod
from typing import Dict, Literal
import numpy as np
from .base import BaseGenerator

class MaskGenerator(BaseGenerator):
    """Generator that creates a mask to be applied to the input series."""
    def __init__(self, shape: list[int]):
        """
        Initialize the MaskGenerator.

        Args:
            shape (list[int]): Shape of the mask to be generated.
            domain (Literal["time", "frequency"] | None): Domain in which this generator operates.
        """
        super().__init__(None)
        self.shape = shape

    def generate(self, inter_variates_probability = 0.5, intra_variates_probability = 0.5, **params) -> np.ndarray:
        """
        Generate a binary mask with the specified shape.

        Args:
            inter_variates_probability (float): Probability of masking across variates.
            intra_variates_probability (float): Probability of masking within variates.
            params (Dict): Additional parameters for mask generation.

        Returns:
            np.ndarray: Binary mask of the specified shape.
        """
        mask = np.random.rand(*self.shape)
        for i in range(self.shape[0]):  # Iterate over variates
            mask[i] = mask[i] < intra_variates_probability
        mask = mask < inter_variates_probability
        return mask.astype(np.bool)

    def combine(self, ts, component, mask):
        return ts