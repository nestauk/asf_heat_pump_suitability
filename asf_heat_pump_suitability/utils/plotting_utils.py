import math

import matplotlib.pyplot as plt
import polars as pl
from typing import List
from pathlib import Path
import os


def plot_feature_distribution_binary_classes(
    df: pl.DataFrame,
    features: List[str],
    target: str,
    save_as: str = None,
    density: bool = False,
) -> plt.Figure:
    """
    Plot histograms of distribution of features across binary classes.

    Args:
        df (pl.DataFrame): dataframe containing feature data and class information
        features (List[str]): features to plot
        target (str): binary class
        save_as (str): file name to save as. Saves a local png file copy to /outputs/figures/. Optional.
        density (bool): set to `True` to normalize histogram across classes. Default False.

    Returns:
        plt.Figure
    """
    nrows = 3
    ncols = math.ceil(len(features) / nrows)

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 8))

    if density:
        y_label = "Density"
    else:
        y_label = "Count"

    for ax, feature in zip(axs.ravel(), features):
        ax.hist(df.filter(pl.col(target))[feature], bins=40, alpha=0.5, density=density)
        ax.hist(
            df.filter(~pl.col(target))[feature], bins=40, alpha=0.5, density=density
        )
        ax.set_title(feature)
        ax.set_ylabel(y_label)

    target_label = target.replace("_", " ").title()

    fig.legend([target_label, f"Not {target_label}"], loc="upper right")
    fig.suptitle("Distribution of features across classes")
    fig.tight_layout()

    if save_as:
        PROJECT_DIR = Path(__file__).resolve().parents[2]
        file_path = os.path.join(PROJECT_DIR, "outputs", "figures", f"{save_as}.png")
        fig.savefig(file_path)
        plt.close(fig)
    else:
        return fig
