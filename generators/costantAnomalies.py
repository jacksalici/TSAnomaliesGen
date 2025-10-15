from .base import AdditiveGenerator
import numpy as np


class CostantAnomaliesGenerator(AdditiveGenerator):
    def __init__(
        self,
        shape: tuple[int],
        anomaly_fraction=0.01,
        anomaly_value=1.0,
        anomaly_length=3,
        anomaly_length_variance=1,
        domain="time",
    ):
        """
        Initialize the CostantAnomaliesGenerator.
        Args:
            shape: (seq_len, no_variates)
                Shape of the time series data.
            anomaly_fraction : float, default=0.01
                Fraction of points in the time series to be replaced with anomalies.
            anomaly_value : float, default=1.0
                Constant value of the anomalies to be added.
            anomaly_length : int, default=3
                Average number of consecutive points to be replaced with anomalies.
            anomaly_length_variance : int, default=1
                Variance in the number of consecutive points for anomalies.
            domain : str, default='time'
                Domain in which the generator operates ('time' or 'frequency').
        """
        self.anomaly_fraction = anomaly_fraction
        self.anomaly_value = anomaly_value
        self.anomaly_length = anomaly_length
        self.anomaly_length_variance = anomaly_length_variance
        self.domain = domain
        super().__init__(shape, domain)

    def generate(self, **params) -> np.ndarray:
        """
        Generate an additive component with constant value anomaly segments.

        Returns:
            np.ndarray: Component to be added to the input containing constant blocks.
        """
        seq_len, no_variates = self.base.shape
        num_anomalies = int(self.anomaly_fraction * seq_len * no_variates)
        # Randomly select starting indices for anomalies
        anomaly_indices = np.random.choice(seq_len * no_variates, num_anomalies, replace=False)
        for idx in anomaly_indices:
            i = idx // no_variates  # Time index
            j = idx % no_variates  # Variate index
            # Determine the length of the anomaly
            anomaly_len = max(1, int(np.random.normal(self.anomaly_length, self.anomaly_length_variance)))
            # Add anomalies to consecutive points
            for k in range(anomaly_len):
                if i + k < seq_len:  # Ensure we don't go out of bounds
                    self.base[i + k, j] = self.anomaly_value
        return self.base
