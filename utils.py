import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import seaborn as sns
import random

def setSeed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

def displayTS(ts: np.ndarray, save_path: str = None):
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    rcParams['font.family'] = "DejaVu Sans"
    rcParams['axes.titlesize'] = 12
    rcParams['axes.labelsize'] = 10
    rcParams['legend.fontsize'] = 8
    rcParams['lines.linewidth'] = 3.0

    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)
    num_variates = ts.shape[1]
    
    fig, axes = plt.subplots(
        num_variates, 1, figsize=(12, 2.5 * num_variates),
        sharex=True, constrained_layout=True
    )
    
    if num_variates == 1:
        axes = [axes]  
    
    palette = sns.color_palette("husl", num_variates)
    
    for i, ax in enumerate(axes):
        ax.plot(ts[:, i], color=palette[i], label=f'Variate {i+1}', alpha=0.9)
        ax.set_title(f'Variate {i+1}', fontsize=14)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(frameon=False, fancybox=False, shadow=False, loc="best")
    
    axes[-1].set_xlabel('Time Steps', fontsize=12)
        
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
