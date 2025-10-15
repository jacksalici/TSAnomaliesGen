import numpy as np
from utils import displayTS, setSeed


def createDummyTS(
    seq_len=1000,
    no_variates=2,
    max_freq=10,
    noise_std=0.05,
    phase_shift=np.pi,
    random_phase=True,
):
    """
    Create a dummy time series using sine waves with added noise.

    Parameters:
    ----------
    seq_len : int, default=1000
        Length of the time series sequence.
    no_variates : int, default=2
        Number of variates (features) in the time series.
    max_freq : int, default=10
        Maximum frequency multiplier for the sine wave.
    noise_std : float, default=0.05
        Standard deviation of Gaussian noise added to the signal.
    phase_shift : float, default=np.pi
        Constant phase shift between successive variates.
    random_phase : bool, default=True
        Whether to add a small random phase shift to each variate.

    Returns:
    -------
    np.ndarray of shape (seq_len, no_variates)
        Generated time series data.
    """

    # Time axis
    t = np.linspace(0, 10 * np.pi, seq_len)

    data = []
    for i in range(no_variates):
        # Frequency for this variate
        freq = np.random.randint(1, max_freq)

        # Phase shift (constant + small random if enabled)
        phase = i * phase_shift
        if random_phase:
            phase += np.random.uniform(-1, 1)

        # Base sine wave
        signal = np.sin(freq * t + phase)

        # Add Gaussian noise
        noisy_signal = signal + np.random.normal(0, noise_std, seq_len)

        data.append(noisy_signal)

    return np.array(data).T


from generators.pointAnomalies import PointAnomaliesGenerator
from generators.normalNoise import NormalNoiseGenerator
from generators.costantAnomalies import CostantAnomaliesGenerator
from generators.sinusoid import SinusoidGenerator

if __name__ == "__main__":
    setSeed(42)
    
    shape = (1000, 2)
    
    anomalies = [
        SinusoidGenerator(shape, frequency=5, amplitude=0.5, phase=0, domain="time"),
        PointAnomaliesGenerator(shape, anomaly_fraction=0.01, anomaly_magnitude=1, domain="time"),
        NormalNoiseGenerator(shape, mean=0, std=0.1, domain="time"),
        CostantAnomaliesGenerator(shape, anomaly_fraction=0.01, anomaly_value=1.0, anomaly_length=5, anomaly_length_variance=2, domain="time"),
        NormalNoiseGenerator(shape, mean=0, std=0.1, domain="frequency"),
        PointAnomaliesGenerator(shape, anomaly_fraction=0.1, anomaly_magnitude=20, domain="frequency"),
    ]

    raw_ts = createDummyTS(shape[0], shape[1])

    ts = raw_ts.copy()
    for anomaly in anomalies:
        ts = anomaly.apply(ts)

    displayTS(ts, raw_ts, save_path="dummy_time_series.png")
