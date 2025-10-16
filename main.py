import numpy as np
from utils import displayTS, setSeed


from generators.normalNoise import NormalNoiseGenerator
from generators.costant import CostantGenerator
from generators.sinusoid import SinusoidGenerator
from generators.mask import MaskGenerator

if __name__ == "__main__":
    setSeed(42)
    
    shape = (1000, 2)
    
    
    
    anomalies = [
        NormalNoiseGenerator(mean=0, std=0.1, domain="time"),
        CostantGenerator(anomaly_fraction=0.01, anomaly_value=1.0, anomaly_length=5, anomaly_length_variance=2, domain="time"),
        NormalNoiseGenerator(mean=0, std=0.1, domain="frequency"),
    ]
    
    masks = MaskGenerator(shape) 
    raw_ts = SinusoidGenerator(shape, frequency=0.5, amplitude=0.5, phase=0)

    ts = raw_ts.copy()
    for anomaly in anomalies:
        ts = anomaly.apply(ts, masks.generate())

    displayTS(ts, raw_ts, save_path="dummy_time_series.png")
