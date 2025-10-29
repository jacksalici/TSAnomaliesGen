from .base import BaseGenerator
import numpy as np
from typing import Literal


class SinusoidGenerator(BaseGenerator):
    def __init__(
        self,
        shape: tuple[int] = None,
        ts: np.ndarray | None = None,
        combine_domain: Literal["time", "frequency"] = None,
        combine_mode: Literal["add", "mul"] | None = None,
        frequency: float | None =None,
        random_frequency:bool = True,
        max_frequency: bool = 5,
        amplitude = 1.0,
        phase = 0.0,
        random_phase = True
    ):
        """
        Initialize the SinusoidGenerator.

        Args:
            shape: (tuple[int]): the shape of the TS to generate
            ts: (np.ndarray): time series to "clone" as of shape and other stuff, not clone in the literal sense.
            combine_domain (str): Domain in which the combine operation happens when `apply` method is called ('time' or 'frequency').
            combine_mode (str): Mode in which the combine operation happens when `apply` method is called ('add' or 'mul').
            shape: (seq_len, no_variates)
                Shape of the time series data.
            frequency : float or None, default=1.0
                Frequency of the sinusoidal signal.
            random_frequency: randomize frequencies overriding the frequency parameter.
            max_frequency: int, default=10
                Maximum random frequency multiplier for the sine wave.
            amplitude : float, default=1.0
                Amplitude of the sinusoidal signal.
            phase : float, default=0.0
                Phase shift of the sinusoidal signal.
            random_phase : bool, default=True
                Whether to add a small random phase shift to each variate.

        """
        super().__init__(shape, ts, combine_domain, combine_mode)
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.random_frequency = random_frequency
        self.max_frequency = max_frequency
        self.random_phase = random_phase

    def generate(self) -> np.ndarray:

        seq_len, no_variates = self.shape
        t = np.linspace(0, 2 * np.pi, seq_len)  # Time vector

        data = []
        for i in range(no_variates):
            # Frequency for this variate
            freq = np.random.randint(1, self.max_frequency) if self.random_frequency else self.frequency
            # Phase shift (constant + small random if enabled)
            if self.random_phase:
                self.phase += np.random.uniform(-1, 1)

            # Base sine wave
            signal = self.amplitude * np.sin(freq * t + self.phase)
            

            data.append(signal)

        return np.array(data).T


if __name__ == "__main__":
    generator = SinusoidGenerator(
        shape=(500, 3),
        frequency=2.0,
        random_frequency=True,
        max_frequency=5,
        amplitude=1.0,
        random_phase=True
    )
    
    generator.test()
