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
        cluster_size=50,
        cluster_variance=10,
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
            cluster_size (int): Average size of mask clusters (consecutive True values). 1 means no clustering.
            cluster_variance (int): Variance in cluster size. 0 means fixed cluster size.
        """

        super().__init__(shape, ts, combine_domain, combine_mode)
        self.inter_variates_probability = inter_variates_probability
        self.intra_variates_probability = intra_variates_probability
        self.cluster_size = cluster_size
        self.cluster_variance = cluster_variance

    def generate(self) -> np.ndarray:

        mask = np.zeros(self.shape, dtype=np.bool_)
        v_mask = np.random.rand(self.no_variates)
        
        # Determine which variates are active based on inter_variates_probability
        active_variates = v_mask < self.inter_variates_probability
        
        for j in range(self.no_variates):
            if not active_variates[j]:
                continue
                
            # Calculate number of True values based on intra_variates_probability
            num_true = int(self.seq_len * self.intra_variates_probability)
            
            if self.cluster_size <= 1:
                # No clustering - original random behavior
                true_indices = np.random.choice(self.seq_len, num_true, replace=False)
                mask[true_indices, j] = True
            else:
                # Clustering mode - create clusters of True values
                i = 0
                remaining_true = num_true
                
                while remaining_true > 0 and i < self.seq_len:
                    # Randomly decide if this position starts a cluster
                    if np.random.rand() < (remaining_true / (self.seq_len - i)):
                        # Determine cluster size
                        current_cluster_size = max(
                            1,
                            int(np.random.normal(self.cluster_size, self.cluster_variance))
                        )
                        current_cluster_size = min(current_cluster_size, remaining_true, self.seq_len - i)
                        
                        # Apply cluster
                        mask[i:i + current_cluster_size, j] = True
                        remaining_true -= current_cluster_size
                        i += current_cluster_size
                    else:
                        i += 1
            
        return mask
