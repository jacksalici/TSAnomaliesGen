from abc import ABC, abstractmethod
from typing import Dict, Literal
import numpy as np
from .base import BaseGenerator


class MaskGenerator(BaseGenerator):
    def __init__(
        self,
        shape: tuple[int] = None,
        ts: np.ndarray | None = None,
        combine_domain: Literal["time", "frequency"] = None,
        combine_mode: Literal["add", "mul"] | None = None,
        inter_variates_probability=0.5,
        intra_variates_probability=0.5,
    ):
        """
        Initialize the MaskGenerator.

        Args:
            shape: (tuple[int]): the shape of the TS to generate
            ts: (np.ndarray): time series to "clone" as of shape and other stuff, not clone in the literal sense.
            combine_domain (str): Domain in which the combine operation happens when `apply` method is called ('time' or 'frequency').
            combine_mode (str): Mode in which the combine operation happens when `apply` method is called ('add' or 'mul').
            inter_variates_probability (float): Probability of masking across variates.
            intra_variates_probability (float): Probability of masking within variates.
        """

        super().__init__(shape, ts, combine_domain, combine_mode)
        self.inter_variates_probability = inter_variates_probability
        self.intra_variates_probability = intra_variates_probability

    def generate(self) -> np.ndarray:

        l_mask = np.random.rand(*self.shape)
        v_mask = np.random.rand(self.no_variates)
        mask = np.ones(self.shape, dtype=np.bool)
        
        for i in range(self.seq_len):
            mask[i] = l_mask[i] < self.intra_variates_probability  
            for j in range(self.no_variates):
                if mask[i][j]:
                     mask[i][j] = v_mask[j] < self.inter_variates_probability
            
        return mask
