from dataclasses import dataclass
from .base import BaseGenerator
import numpy as np

@dataclass
class Maybe():
    generator: BaseGenerator
    mask: BaseGenerator = None
    probability: float = 1.0
    

class Some():
    """Apply only SOME of generators"""
    def __init__(self, generators: list[Maybe]):
        self.generators = generators
    
    def generate_and_combine(self, first_ts: np.ndarray):
        ts = first_ts.copy()
        for elem in self.generators:
            ts = elem.generator.generate_and_combine(ts, elem.mask.generate())
        
        return ts
        
    