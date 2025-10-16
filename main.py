import numpy as np
from utils import displayTS, setSeed


from generators.normal import NormalGenerator
from generators.costant import CostantGenerator
from generators.sinusoid import SinusoidGenerator
from generators.mask import MaskGenerator

if __name__ == "__main__":
    #setSeed(21)
    
    shape = (1000, 4)
    
    generators = [
        NormalGenerator(shape, mean=0, std=0.01, combine_domain="frequency", combine_mode="add"),
        SinusoidGenerator(shape, amplitude=0.5, phase=np.pi, max_frequency=5, combine_domain = "time", combine_mode="add"),
        #CostantGenerator(shape, gen_fraction=0.1, gen_value=2, gen_length=1, gen_length_variance=10, combine_domain="time", combine_mode="mul"),
        #NormalGenerator(shape, mean=1, std=0.1, combine_domain="time", combine_mode="mul"),
    ]
    
    raw_ts = np.zeros(shape)
    ts = raw_ts.copy()
    for g in generators:
        mask = MaskGenerator(shape, intra_variates_probability=0.2, inter_variates_probability=0.35).generate()
        ts = g.generate_and_combine(ts, mask)

    displayTS(ts, raw_ts, save_path="dummy_time_series.png")
