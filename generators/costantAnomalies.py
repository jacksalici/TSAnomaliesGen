from .base import BaseGenerator
import numpy as np


class CostantAnomaliesGenerator(BaseGenerator):
    def __init__(self, anomaly_fraction=0.01, anomaly_value=1.0, anomaly_length=3, anomaly_length_variance=1):
        """
        Initialize the CostantAnomaliesGenerator.
        Parameters:
        ----------
        anomaly_fraction : float, default=0.01
            Fraction of points in the time series to be replaced with anomalies.
        anomaly_value : float, default=1.0
            Constant value of the anomalies to be added.
        anomaly_length : int, default=3
            Average number of consecutive points to be replaced with anomalies.
        anomaly_length_variance : int, default=1
            Variance in the number of consecutive points for anomalies.
        """
        self.anomaly_fraction = anomaly_fraction
        self.anomaly_value = anomaly_value
        self.anomaly_length = anomaly_length
        self.anomaly_length_variance = anomaly_length_variance
    def generate(self, ts: np.ndarray) -> np.ndarray:
        """
        Generate constant value anomalies in the given time series.
        Parameters:
        ----------
        ts : np.ndarray
            Input time series data of shape (seq_len, no_variates).
        Returns:
        -------
        np.ndarray
            Time series data with constant value anomalies added.
        """
        ts_with_anomalies = ts.copy()
        seq_len, no_variates = ts.shape
        num_anomalies = int(self.anomaly_fraction * seq_len * no_variates)
        # Randomly select starting indices for anomalies
        anomaly_indices = np.random.choice(seq_len * no_variates, num_anomalies, replace=False)
        for idx in anomaly_indices:
            i = idx // no_variates  # Time index
            j = idx % no_variates   # Variate index
            # Determine the length of the anomaly
            anomaly_len = max(1, int(np.random.normal(self.anomaly_length, self.anomaly_length_variance)))
            # Add anomalies to consecutive points
            for k in range(anomaly_len):
                if i + k < seq_len:  # Ensure we don't go out of bounds
                    ts_with_anomalies[i + k, j] = self.anomaly_value
        return ts_with_anomalies