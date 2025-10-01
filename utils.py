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


def displayTS(ts: np.ndarray, raw_ts: np.ndarray = None, save_path: str = None):
    """
    Display time series (ts) with optional raw time series (raw_ts).
    
    Args:
        ts (np.ndarray): Processed time series (T x N).
        raw_ts (np.ndarray, optional): Raw time series of same shape as ts.
        save_path (str, optional): Path to save the figure.
    """
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    rcParams['font.family'] = "DejaVu Sans"
    rcParams['axes.titlesize'] = 12
    rcParams['axes.labelsize'] = 10
    rcParams['legend.fontsize'] = 9

    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)
    num_variates = ts.shape[1]

    # If raw_ts is provided, validate shape
    if raw_ts is not None:
        if raw_ts.ndim == 1:
            raw_ts = raw_ts.reshape(-1, 1)
        assert raw_ts.shape == ts.shape, \
            f"raw_ts must have the same shape as ts, got {raw_ts.shape} vs {ts.shape}"

    fig, axes = plt.subplots(
        num_variates, 1, figsize=(12, 2.5 * num_variates),
        sharex=True, constrained_layout=True
    )

    if num_variates == 1:
        axes = [axes]  

    for i, ax in enumerate(axes):
        if raw_ts is not None:
            ax.plot(raw_ts[:, i], label=f'Raw TS', color="#1f77b4", linewidth=3)
        ax.plot(ts[:, i], label=f'TS', color="#ff357c", linewidth=1)
        ax.set_title(f'Variate {i+1}', fontsize=14)
        ax.set_ylabel('Value', fontsize=12)

    axes[-1].set_xlabel('Time Steps', fontsize=12)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', ncol=2, frameon=False, fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

