# Time Series Anomalies Generator

🧰 Library for generating synthetic time series data with anomalies for testing anomaly detection algorithms. 

🚧 Work in progress _(Anomalies should be as good - bad? - as possible)_  


> Entry point: `main.py`

## Table of Contents

- [Usage Examples](#usage-examples)
    - [Combination Strategies](#combination-strategies)
- [Base Architecture](#base-architecture)
    - [BaseGenerator](#basegenerator)
- [Noise-Based Generators](#noise-based-generators)
    - [NormalGenerator](#normalgenerator)
    - [UniformGenerator](#uniformgenerator)
    - [LaplaceGenerator](#laplacegenerator)
    - [ExponentialGenerator](#exponentialgenerator)
    - [GammaGenerator](#gammagenerator)
    - [PoissonGenerator](#poissongenerator)
    - [PinkNoiseGenerator](#pinknoisegenerator)
- [Pattern-Based Generators](#pattern-based-generators)
    - [CostantGenerator](#costantgenerator)
    - [DriftGenerator](#driftgenerator)
    - [SinusoidGenerator](#sinusoidgenerator)
- [Utility Generators](#utility-generators)
    - [MaskGenerator](#maskgenerator)

## Usage Examples

```python
import numpy as np
from generators import NormalGenerator, DriftGenerator

# Generate time series shape
shape = (1000, 3)  # 1000 time steps, 3 variates

# Example 1: Add Gaussian noise
normal_gen = NormalGenerator(
    shape=shape, 
    combine_mode="add", 
    combine_domain="time",
    mean=0.0, 
    std=0.5
)
base_ts = np.zeros(shape)
noisy_ts = normal_gen.generate_and_combine(base_ts)


# Example 2: Add drift
drift_gen = DriftGenerator(
    shape=shape,
    combine_mode="add",
    combine_domain="time",
    drift_type="linear",
    drift_rate=0.01
)
final_ts = drift_gen.generate_and_combine(spiked_ts)
```

### Combination Strategies

- **Time domain addition**: Direct addition of anomaly to signal
- **Time domain multiplication**: Scaling of signal by anomaly factor
- **Frequency domain addition**: Add anomaly in frequency space (good for spectral anomalies)
- **Frequency domain multiplication**: Multiply in frequency space (good for filtering effects)


## Base Architecture

### BaseGenerator
The abstract base class that all generators inherit from. It provides:
- **Shape management**: Define time series dimensions (sequence length, number of variates)
- **Combine operations**: Merge generated data with existing time series
- **Domain control**: Apply operations in time or frequency domain
- **Mode control**: Add or multiply generated data

Common parameters for all generators:
- `shape`: Tuple defining (sequence_length, number_of_variates)
- `ts`: Existing time series to clone shape from
- `combine_domain`: "time" or "frequency" domain for combination
- `combine_mode`: "add" or "mul" for combination operation

---

## Noise-Based Generators

### NormalGenerator
Generates Gaussian (normal) distributed noise.

**Use case**: White noise, natural measurement errors, random fluctuations

### UniformGenerator
Generates uniformly distributed noise where all values within a range are equally likely.

**Use case**: Random noise with bounded range, testing robustness to uniform perturbations

### LaplaceGenerator
Generates Laplace (double exponential) distributed noise with heavier tails than normal distribution.

**Use case**: Noise with occasional large deviations, outlier-prone data

### ExponentialGenerator
Generates exponentially distributed noise (always positive values).

**Use case**: Waiting times, inter-arrival times, lifetime data anomalies

### GammaGenerator
Generates gamma distributed noise with flexible shape (always positive values).

**Use case**: Positive-valued anomalies, skewed distributions, duration modeling

### PoissonGenerator
Generates Poisson distributed noise, suitable for count data.

**Use case**: Count anomalies, event frequency changes, discrete value noise

### PinkNoiseGenerator
Generates 1/f^α noise (colored noise) with power spectral density inversely proportional to frequency.

**Use case**: Natural phenomena noise, long-range correlations, realistic background noise

**Note**: Uses FFT method to generate frequency-domain colored noise

## Pattern-Based Generators

### CostantGenerator
Generates constant-value anomalies at random locations with configurable length.

**Use case**: Sensor freezing, stuck values, plateau anomalies

### DriftGenerator
Generates gradual drift/trend patterns over time.

**Use case**: Sensor degradation, gradual system changes, trend anomalies

### SinusoidGenerator
Generates sinusoidal patterns with configurable frequency, amplitude, and phase.

**Use case**: Periodic anomalies, oscillatory patterns, seasonal effects

## Utility Generators

### MaskGenerator
Generates boolean masks for selective anomaly application.

**Use case**: Control which time points and variates receive anomalies

**Note**: Returns boolean mask instead of numeric values


---

© 2025 G.S.