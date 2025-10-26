import numpy as np
from utils import displayTS, setSeed
from generators.utils import Maybe, Some

from generators.normal import NormalGenerator
from generators.costant import CostantGenerator
from generators.sinusoid import SinusoidGenerator
from generators.mask import MaskGenerator

from generators.drift import DriftGenerator

if __name__ == "__main__":
    #setSeed(21)
    
    shape = (1000, 4)
        
    
    raw_ts = np.zeros(shape)
   
    
    some = Some(
        [
            Maybe(
                NormalGenerator(shape, mean=0, std=0.01, combine_domain="frequency", combine_mode="add"),
                MaskGenerator(shape, intra_variates_probability=1, inter_variates_probability=1),
                probability=1
            ),
            Maybe(
                SinusoidGenerator(shape, amplitude=0.5, phase=np.pi, max_frequency=5, combine_domain = "time", combine_mode="add"),
                MaskGenerator(shape, intra_variates_probability=0.2, inter_variates_probability=0.35),
                probability=0.5
            ),
            Maybe(
                DriftGenerator(shape, drift_type="exponential", combine_domain = "time", combine_mode="add"),
                MaskGenerator(shape, intra_variates_probability=1, inter_variates_probability=1),
                probability=1
            ),
            
        ]
    )
    
    ts = some.generate_and_combine(raw_ts) 

    displayTS(ts, raw_ts, save_path="dummy_time_series.png")
