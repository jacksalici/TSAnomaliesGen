from .base import AdditiveGenerator
import numpy as np


class PointAnomaliesGenerator(AdditiveGenerator):
    def __init__(self, shape: tuple[int], anomaly_fraction=0.01, anomaly_magnitude=0.5, domain="time"):
        """
        Initialize the PointAnomaliesGenerator.

        Args:
            shape: (seq_len, no_variates)
                Shape of the time series data.
            anomaly_fraction : float, default=0.01
                Fraction of points in the time series to be replaced with anomalies.
            anomaly_magnitude : float, default=5.0
                Magnitude of the anomalies to be added.
            domain : str, default='time'
                Domain in which the generator operates ('time' or 'frequency').
        """
        self.anomaly_fraction = anomaly_fraction
        self.anomaly_magnitude = anomaly_magnitude
        self.domain = domain
        super().__init__(shape, domain)

    def generate(self, **params) -> np.ndarray:
        """
        Create a sparse additive component with point anomalies.

        Returns:
            np.ndarray: Anomalies component to be added to the input.
        """
        seq_len, no_variates = self.base.shape
        num_anomalies = int(self.anomaly_fraction * seq_len * no_variates)

        component = np.zeros((seq_len, no_variates), dtype=float)

        # Randomly select indices for anomalies
        anomaly_indices = np.random.choice(
            seq_len * no_variates, num_anomalies, replace=False
        )

        for idx in anomaly_indices:
            i = idx // no_variates  # Time index
            j = idx % no_variates  # Variate index
            component[i, j] += self.anomaly_magnitude * np.random.choice([-1, 1])

        return component
