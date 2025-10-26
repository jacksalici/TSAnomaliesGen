from dataclasses import dataclass
from .base import BaseGenerator
import numpy as np
import random

@dataclass
class Maybe():
    generator: BaseGenerator
    mask: BaseGenerator = None
    probability: float = 0.5
    

class Some():
    """Apply only SOME of generators"""
    def __init__(self, generators: list[Maybe], shuffle = False, max_generators = None):
        """
        Initialize the Some generator.
        
        Args:
            generators (list[Maybe]): List of Maybe generator objects.
            shuffle (bool): Whether to shuffle the order of generators before applying.
            max_generators (int | None): Maximum number of generators to apply. If None, apply all.
        """
        
        
        self.generators = generators
        self._r = np.random.rand(len(generators))
        
        if shuffle:
            random.shuffle(self.generators)
            
        if max_generators is not None:
            self.generators = self.generators[:max_generators]
    
    def generate_and_combine(self, first_ts: np.ndarray):
        ts = first_ts.copy()
        for index, elem in enumerate(self.generators):
            if elem.probability < self._r[index]:
                print(f"Skipped {elem} (index: {index}) due to `some` probability.")
                continue
            ts = elem.generator.generate_and_combine(ts, elem.mask.generate())
            print(f"Applied {elem} (index: {index}).")

        return ts
        
    