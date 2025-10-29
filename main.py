import numpy as np
from utils import displayTS, setSeed
from generators.utils import Maybe, Some

# Import all generators
from generators.normal import NormalGenerator
from generators.costant import CostantGenerator
from generators.sinusoid import SinusoidGenerator
from generators.mask_sigmoids import SigmoidMaskGenerator
from generators.drift import DriftGenerator
from generators.exponential import ExponentialGenerator
from generators.gamma import GammaGenerator
from generators.laplace import LaplaceGenerator
from generators.poisson import PoissonGenerator
from generators.pink_noise import PinkNoiseGenerator

if __name__ == "__main__":
    #setSeed(21)

    shape = (1000, 4)

    raw_ts = SinusoidGenerator(
        shape, amplitude=0.5, phase=np.pi, max_frequency=5
    ).generate()
    
    mask = SigmoidMaskGenerator(
        shape,
        num_peaks=4,
        peak_length=100,
        length_variance=50,
        steepness=0.15
    ).generate()
    
    some = Some(
        [
            Maybe(
                NormalGenerator(
                    shape, mean=0, std=0.01, combine_domain="time", combine_mode="add"
                ),
                mask,
                probability=1,
            ),
            Maybe(
                LaplaceGenerator(
                    shape, loc=0, scale=0.05, combine_domain="time", combine_mode="add"
                ),
                mask,
                probability=0.7,
            ),
            Maybe(
                ExponentialGenerator(
                    shape, scale=0.02, combine_domain="time", combine_mode="add"
                ),
                mask,
                probability=0.5,
            ),
            Maybe(
                GammaGenerator(
                    shape, shape_param=2.0, scale=0.01, combine_domain="time", combine_mode="add"
                ),
                mask,
                probability=0.4,
            ),
            Maybe(
                PoissonGenerator(
                    shape, lam=0.5, combine_domain="time", combine_mode="add"
                ),
                mask,
                probability=0.3,
            ),
            Maybe(
                PinkNoiseGenerator(
                    shape, alpha=1.0, amplitude=0.1, combine_domain="time", combine_mode="add"
                ),
                mask,
                probability=0.6,
            ),
            Maybe(
                SinusoidGenerator(
                    shape,
                    amplitude=0.1,
                    phase=2 * np.pi,
                    max_frequency=7,
                    combine_domain="time",
                    combine_mode="add",
                ),
                mask,
                probability=1,
            ),
            Maybe(
                CostantGenerator(
                    shape,
                    gen_fraction=0.02,
                    gen_value=0.01,
                    gen_length=5,
                    gen_length_variance=2,
                    combine_domain="time",
                    combine_mode="add",
                ),
                mask,
                probability=0.3,
            ),
            Maybe(
                DriftGenerator(
                    shape,
                    drift_type="exponential",
                    drift_rate=0.01,
                    random_drift=True,
                    drift_rate_range=(0.005, 0.02),
                    combine_domain="time",
                    combine_mode="add",
                ),
                mask,
                probability=0.5,
            ),
        ],
        shuffle=True,
        max_generators = 5,
        
    )

    ts = some.generate_and_combine(raw_ts)

    displayTS(ts, raw_ts, mask, save_path="dummy_time_series.png")

