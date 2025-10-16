from .base import BaseGenerator
import numpy as np


class PointAnomaliesGenerator(BaseGenerator):
    def __init__(self,  shape: tuple[int], anomaly_fraction=0.01, anomaly_magnitude=0.5, domain: str = 'time'):
        """
        Initialize the PointAnomaliesGenerator.

        Parameters:
        ----------
        anomaly_fraction : float, default=0.01
            Fraction of points in the time series to be replaced with anomalies.
        anomaly_magnitude : float, default=0.5
            Magnitude of the anomalies to be added.
        domain : str, default='time'
            Domain in which the generator operates ('time' or 'frequency').
        """
        super().__init__(domain)
        self.anomaly_fraction = anomaly_fraction
        self.anomaly_magnitude = anomaly_magnitude

    def generate(self, ts: np.ndarray, params = None) -> np.ndarray:
        """
        Generate point anomalies in the given time series.

        Parameters:
        ----------
        ts : np.ndarray
            Input time series data of shape (seq_len, no_variates).

        Returns:
        -------
        np.ndarray
            Time series data with point anomalies added.
        """
        ts_with_anomalies = ts.copy()
        seq_len, no_variates = ts.shape
        num_anomalies = int(self.anomaly_fraction * seq_len * no_variates)

        # Randomly select indices for anomalies
        anomaly_indices = np.random.choice(seq_len * no_variates, num_anomalies, replace=False)

        for idx in anomaly_indices:
            i = idx // no_variates  # Time index
            j = idx % no_variates   # Variate index
            # Add an anomaly by adding a large random value
            ts_with_anomalies[i, j] += self.anomaly_magnitude * np.random.choice([-1, 1])

        return ts_with_anomalies