import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List
import os
from src.data.baseline_ppl import BASELINE_PPL
from src.config import MainConfiguration


### DF utilities

def parse_modif_type(modif_type: str) -> str:
    if "ckpt" in modif_type:
        modif_type = f"finetuning-{modif_type.split('-')[-1]}"
    return modif_type


def get_unique_checkpoints(df: pd.DataFrame) -> List[str]:
    modif_types = df["modif_type"].unique()
    checkpoints = []
    for modif_type in modif_types:
        if "finetuning-" in modif_type:
            checkpoint = int(modif_type.split("-")[-1])
            checkpoints.append(checkpoint)
        if modif_type == "original":
            checkpoints.append(0)
    
    return checkpoints

### Loading utilities

### PLotting utilities

def get_tpr_at5(df, modif_type, eval_type):
    group = df[(df["modif_type"] == modif_type) & (df["eval_type"] == eval_type)]
    if group.empty:
        return 0
    return (group["pvalue"] < 0.05).mean()


def compute_roc_from_pvalues(df, modif_type, eval_type, alpha_vals):
    group = df[(df["modif_type"] == modif_type) & (df["eval_type"] == eval_type)]
    if group.empty:
        return None
    return pd.DataFrame({
        "alpha": alpha_vals,
        "tpr": [ (group["pvalue"] < a).mean() for a in alpha_vals ]
    })

def create_roc_plot_df(dfs, names, checkpoints, eval_types, alpha_vals, is_log):
    rows = []
    for name, df in zip(names, dfs):
        
        for eval_type, log in zip(eval_types, is_log):
            
            alphas = np.geomspace(1e-4,1, alpha_vals) if log else np.linspace(0, 1, alpha_vals)
            
            for checkpoint in checkpoints:
                modif_type = f"finetuning-{checkpoint}" if checkpoint != 0 else "original"
                roc_df = compute_roc_from_pvalues(df, modif_type, eval_type, alphas)
                if roc_df is None:
                    continue
                roc_df["name"] = name
                roc_df["eval_type"] = eval_type
                roc_df["checkpoint"] = checkpoint
                rows.append(roc_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def plot_roc_curves(plot_df, names, eval_types, is_log):
    n_rows = len(names)
    n_cols = len(eval_types)
    
    sns.set_palette("flare", n_colors=len(plot_df["checkpoint"].unique()))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 6 * n_rows), sharex=False, sharey=True)
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])
    
    for i, name in enumerate(names):
        for j, eval_type in enumerate(eval_types):
            ax = axes[i, j]
            sub_df = plot_df[(plot_df["name"] == name) & (plot_df["eval_type"] == eval_type)]
            for checkpoint, group in sub_df.groupby("checkpoint"):
                ax.plot(group["alpha"], group["tpr"], label=str(checkpoint))
            if is_log[j]:
                ax.set_xscale("log")
                #ax.set_xlim(1e-2, 1)
        
            ax.plot(group["alpha"],group["alpha"], color="black" )
        
            ax.set_title(f"{name} - {eval_type}")
            ax.set_xlabel("FPR")
            if j == 0:
                ax.set_ylabel("TPR")
            else:
                ax.set_ylabel("")
            ax.grid(True)
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title="Checkpoint")
    for ax in axes.flatten():
        sns.despine(ax=ax)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    return fig, axes


def create_ppl_plot_df(dfs, names, checkpoints, eval_types):
    rows = []
    for name, df in zip(names, dfs):
        for eval_type in eval_types:
            for checkpoint in checkpoints:
                modif_type = f"finetuning-{checkpoint}" if checkpoint != 0 else "original"
                group = df[(df["modif_type"] == modif_type) & (df["eval_type"] == eval_type)]
                if group.empty:
                    continue
                rows.append({
                    "name": name,
                    "eval_type": eval_type,
                    "checkpoint": checkpoint,
                    "median_ppl": group["ppl"].median(),
                })
    return pd.DataFrame(rows)

def plot_ppl_scatter(plot_df, names, eval_types, offset_width=20):
    n_plots = len(names)
        
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5), sharey=True)
    if n_plots == 1:
        axes = [axes]
        
    checkpoints_sorted = sorted(plot_df["checkpoint"].unique())
    cp_palette = sns.color_palette("flare", len(checkpoints_sorted))
    cp_colors = {cp: color for cp, color in zip(checkpoints_sorted, cp_palette)}
    eval_types_sorted = sorted(eval_types)
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    eval_markers = {et: markers[i % len(markers)] for i, et in enumerate(eval_types_sorted)}
    for ax, name in zip(axes, names):
        sub_df = plot_df[plot_df["name"] == name]
        for _, row in sub_df.iterrows():
            cp = row["checkpoint"]
            et = row["eval_type"]
            x = cp
            y = row["median_ppl"]
            ax.scatter(x, y, marker=eval_markers[et], color=cp_colors[cp], s=100)
        ax.set_title(name)
        ax.set_xlabel("Checkpoint")
        ax.grid(True)
        sns.despine(ax=ax)
    axes[0].set_ylabel("Median PPL")
    
    cp_handles = [Line2D([0], [0], marker='o', color=cp_colors[cp], linestyle='',
                           markersize=8, label=f"CP {cp}") for cp in checkpoints_sorted]
    et_handles = [Line2D([0], [0], marker=eval_markers[et], color='k', linestyle='',
                           markersize=8, label=et) for et in eval_types_sorted]
    handles = cp_handles + et_handles
    fig.legend(handles, [h.get_label() for h in handles], loc=(0.7,0.2), frameon=False,
               title="")
    
    plt.tight_layout(rect=[0, 0, 0.7, 1])
    return fig, axes

def add_baseline_ppl(base_model: str, delta: float, axes, eval_types):
    
    for ax in axes:
        for et in eval_types:
            try:
                ppl = BASELINE_PPL[base_model][str(delta)][et]
                ax.axhline(ppl, color="grey", linestyle="--", alpha=0.5)
                # Add et as text just above the line
                ax.text(0, ppl, et, color="grey", ha="right", va="bottom")
            except KeyError:
                continue
    
    return axes
    

def process_results(df: pd.DataFrame, config: MainConfiguration, eval_types=None):

    if eval_types is not None:
        eval_types = eval_types
    else:
        eval_types = df["eval_type"].unique()
    
    # Roc curves
    checkpoints = get_unique_checkpoints(df)
    print(checkpoints)
    alpha_vals = 1000
    if len(checkpoints) > 1:
        max_cp = max([int(cp) for cp in checkpoints])
    else:
        max_cp = 0
    
    if max_cp > 0:
        is_log = [get_tpr_at5(df, f"finetuning-{max_cp}", eval_type) > 0.3 for eval_type in eval_types]
    else:
        is_log = [get_tpr_at5(df, "original", eval_type) > 0.3 for eval_type in eval_types]
    
    os.makedirs("figures", exist_ok=True)
    
    figs = []
    fig_names = []
    
    roc_df = create_roc_plot_df([df], ["QuickEval"], checkpoints, eval_types, alpha_vals, is_log)
    fig_roc, _ = plot_roc_curves(roc_df, ["QuickEval"], eval_types, is_log)
    fig_names.append("roc_curves")
    figs.append(fig_roc)
    
    # PPL scatter
    ppl_df = create_ppl_plot_df([df], ["QuickEval"], checkpoints, eval_types)
    fig_ppl, axes_ppl = plot_ppl_scatter(ppl_df, ["QuickEval"], eval_types)
    add_baseline_ppl(config.base_model, config.watermark_config.watermark_config.delta, axes_ppl, eval_types)
    fig_names.append("ppl_scatter")
    figs.append(fig_ppl)
    
    return figs, fig_names
    

    
    
