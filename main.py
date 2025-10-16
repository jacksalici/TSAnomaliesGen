import numpy as np
from utils import displayTS, setSeed


from generators.normal import NormalGenerator
from generators.costant import CostantGenerator
from generators.sinusoid import SinusoidGenerator
from generators.mask import MaskGenerator

if __name__ == "__main__":
    setSeed(21)
    
    shape = (1000, 2)
    
    
    
    anomalies = [
        NormalGenerator(shape = shape, mean=0, std=0.1, combine_domain="time", combine_mode="add"),
        CostantGenerator(shape = shape, gen_fraction=0.01, gen_value=1.2, gen_length=5, gen_length_variance=2, combine_domain="time", combine_mode="mul"),
        NormalGenerator(shape = shape, mean=0, std=0.1, combine_domain="time", combine_mode="add"),
        CostantGenerator(shape = shape, gen_fraction=0.01, gen_value=1.2, gen_length=5, gen_length_variance=2, combine_domain="time", combine_mode="mul"),
        #NormalGenerator(shape = shape, mean=1, std=0.1, combine_domain="time", combine_mode="mul"),
    ]
    
    
    raw_ts = SinusoidGenerator(shape, amplitude=0.5, phase=np.pi, max_frequency=6).generate()

    ts = raw_ts.copy()
    for anomaly in anomalies:
        mask = MaskGenerator(shape).generate()
        ts = anomaly.generate_and_combine(ts, mask)

    displayTS(ts, raw_ts, save_path="dummy_time_series.png")
