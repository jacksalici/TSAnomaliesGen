# Time Series Generator ⚗️

🧰 Library for generating synthetic time series data with anomalies for testing anomaly detection algorithms. 


## Usage Examples

```python
shape = (1000, 3)  

# Gaussian noise
normal_gen = NormalGenerator(
    shape=shape, 
    combine_mode="add", 
    combine_domain="time",
    mean=0.0, 
    std=0.5
)

# Drift
drift_gen = DriftGenerator(
    shape=shape,
    combine_mode="add",
    combine_domain="time",
    drift_type="linear",
    drift_rate=0.01
)

# Generate sinusoidal base time series
raw_ts = SinusoidGenerator(
        shape, amplitude=0.5, phase=np.pi, max_frequency=5
    ).generate()
    

# Combine anomalies with specified probabilities
Some(
    Maybe(normal_gen, probability=0.5),
    Maybe(drift_gen, probability=0.3)
).generate_and_combine(raw_ts)

```


## Available Generators

| Generator | Category | Use Case |
|-----------|----------|----------|
| **NormalGenerator** | Noise | Gaussian noise, measurement errors, random fluctuations |
| **UniformGenerator** | Noise | Bounded random noise, uniform perturbations |
| **LaplaceGenerator** | Noise | Heavy-tailed noise, outlier-prone data |
| **ExponentialGenerator** | Noise | Positive noise, waiting times, inter-arrival times |
| **GammaGenerator** | Noise | Positive skewed distributions, duration modeling |
| **PoissonGenerator** | Noise | Count anomalies, discrete value noise |
| **PinkNoiseGenerator** | Noise | Colored noise (1/f^α), long-range correlations |
| **CostantGenerator** | Pattern | Sensor freezing, stuck values, plateau anomalies |
| **DriftGenerator** | Pattern | Sensor degradation, gradual trends |
| **SinusoidGenerator** | Pattern | Periodic anomalies, oscillatory patterns |
| **MaskGenerator** | Utility | Boolean masks for selective anomaly application |
| **SigmoidMaskGenerator** | Utility | Float masks with double sigmoid peaks |

---

© 2025 G.S.