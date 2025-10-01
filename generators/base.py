from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, ts: np.ndarray) -> np.ndarray:
        pass