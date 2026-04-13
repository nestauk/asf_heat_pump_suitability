import math
import matplotlib.pyplot as plt
import polars as pl
from typing import List
from pathlib import Path
import os


def plot_feature_distribution_binary_classes(
    df: pl.DataFrame, features: List[str], target: str, save_as: str
):
    nrows = 3
    ncols = math.ceil(len(features) / nrows)

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 8))

    for ax, feature in zip(axs.ravel(), features):
        ax.hist(df.filter(pl.col(target))[feature], bins=40, alpha=0.5)
        ax.hist(df.filter(~pl.col(target))[feature], bins=40, alpha=0.5)
        ax.set_title(feature)

    target_label = target.replace("_", " ").title()

    fig.legend([target_label, f"Not {target_label}"], loc="upper right")
    fig.suptitle("Distribution of features across classes")
    fig.tight_layout()

    if save_as:
        PROJECT_DIR = Path(__file__).resolve().parents[2]
        file_path = os.path.join(PROJECT_DIR, "outputs", "figures", f"{save_as}.png")
        fig.savefig(file_path)
